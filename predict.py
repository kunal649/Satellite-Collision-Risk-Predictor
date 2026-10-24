from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_PATH = Path(__file__).resolve().with_name("collision_model_pipeline.joblib")


def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _patch_xgboost_compatibility(model: Any) -> Any:
    if model is None:
        return model

    try:
        defaults = xgb.XGBClassifier().get_params()
        for key, value in defaults.items():
            if not hasattr(model, key):
                setattr(model, key, value)
        if not hasattr(model, "use_label_encoder"):
            model.use_label_encoder = False
    except Exception:
        pass

    return model


def load_pipeline(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    warnings.filterwarnings("ignore")
    try:
        xgb.set_config(verbosity=0)
    except Exception:
        pass

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        model = artifact.get("model")
        feature_cols = artifact.get("feature_cols")
        imputer = artifact.get("imputer")
        threshold = artifact.get("threshold")
    else:
        model = artifact
        feature_cols = None
        imputer = None
        threshold = None

    if model is None:
        raise ValueError("The saved joblib object does not contain a valid model.")

    model = _patch_xgboost_compatibility(model)

    if feature_cols is None:
        feature_cols = getattr(model, "feature_names_in_", None)
    if feature_cols is None:
        n_features = getattr(model, "n_features_in_", None)
        feature_cols = [f"feature_{idx}" for idx in range(int(n_features))] if n_features is not None else []

    if threshold is None:
        threshold = getattr(model, "threshold", None)
    if threshold is None:
        threshold = 0.5
        print("Warning: saved threshold not found. Using default threshold 0.5.")

    return {
        "model": model,
        "feature_cols": list(feature_cols) if feature_cols else [],
        "imputer": imputer,
        "threshold": float(threshold),
    }


def _normalize_events(events: Any) -> list[dict[str, Any]]:
    if events is None:
        return []
    if isinstance(events, dict):
        events = [events]
    if isinstance(events, pd.DataFrame):
        events = events.to_dict(orient="records")

    normalized: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise TypeError(f"Each event must be a dictionary, received: {type(event).__name__}")

        row = dict(event)
        if "features" in event and isinstance(event["features"], dict):
            row = dict(event["features"])
            for key, value in event.items():
                if key != "features":
                    row.setdefault(key, value)

        normalized.append(row)
    return normalized


def _apply_imputer(df: pd.DataFrame, imputer: Any) -> pd.DataFrame:
    if imputer is None:
        return df

    if hasattr(imputer, "transform") and hasattr(imputer, "_fill_dtype"):
        try:
            transformed = imputer.transform(df)
            return pd.DataFrame(transformed, columns=list(df.columns), index=df.index)
        except Exception:
            pass

    if not hasattr(imputer, "statistics_"):
        return df

    statistics = np.asarray(imputer.statistics_, dtype=float)
    result = df.copy()
    for idx, column in enumerate(result.columns):
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        fill_value = float(statistics[idx]) if idx < len(statistics) else np.nan
        values = np.where(np.isnan(values), fill_value, values)
        result[column] = values
    return result


def prepare_input(
    events: Any,
    feature_cols: Sequence[str] | None = None,
    imputer: Any = None,
) -> pd.DataFrame:
    rows = _normalize_events(events)
    if not rows:
        return pd.DataFrame(columns=list(feature_cols) if feature_cols else [])

    df = pd.DataFrame(rows)
    if feature_cols is None:
        feature_cols = list(df.columns)

    for feature in feature_cols:
        if feature not in df.columns:
            df[feature] = np.nan

    df = df[list(feature_cols)]
    if imputer is not None:
        return _apply_imputer(df, imputer)
    return df


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return np.asarray(probabilities).ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if np.ndim(scores) == 2:
            scores = scores[:, 1]
        return 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))

    predictions = model.predict(X)
    return np.asarray(predictions, dtype=float)


def predict_event(event: dict[str, Any], pipeline: dict[str, Any]) -> dict[str, Any]:
    feature_cols = pipeline.get("feature_cols", [])
    imputer = pipeline.get("imputer")
    model = pipeline["model"]
    threshold = float(pipeline.get("threshold", 0.5))

    prepared = prepare_input([event], feature_cols=feature_cols, imputer=imputer)
    if prepared.empty:
        probability = 0.0
    else:
        probability = float(_predict_proba(model, prepared)[0])

    prediction = "HIGH RISK" if probability >= threshold else "LOW RISK"
    event_id = event.get("event_id", "unknown")
    return {
        "event_id": str(event_id),
        "probability": probability,
        "threshold": threshold,
        "prediction": prediction,
    }


def predict_events(events: Any, pipeline: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if pipeline is None:
        pipeline = load_pipeline()

    feature_cols = pipeline.get("feature_cols", [])
    imputer = pipeline.get("imputer")
    model = pipeline["model"]
    threshold = float(pipeline.get("threshold", 0.5))

    normalized = _normalize_events(events)
    if not normalized:
        return []

    prepared = prepare_input(normalized, feature_cols=feature_cols, imputer=imputer)
    probabilities = _predict_proba(model, prepared)

    results: list[dict[str, Any]] = []
    for index, probability in enumerate(probabilities):
        event = normalized[index]
        prediction = "HIGH RISK" if float(probability) >= threshold else "LOW RISK"
        results.append({
            "event_id": str(event.get("event_id", f"event_{index + 1}")),
            "probability": float(probability),
            "threshold": threshold,
            "prediction": prediction,
        })

    return results


def load_sample_events(json_path: str | Path = "sample_test_events.json") -> list[dict[str, Any]]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Sample events file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        items = data.get("events", data.get("data", [data]))
    else:
        items = data

    if not isinstance(items, list):
        raise ValueError("Sample event payload must be a list of event dictionaries.")

    return items
