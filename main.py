from __future__ import annotations

import json
from pathlib import Path

from evaluate import compute_metrics
from predict import load_pipeline, load_sample_events, predict_events
from visualize import save_confusion_matrix, save_feature_importance, save_prediction_chart


def main() -> None:
    print("====================================================")
    print("SATELLITE COLLISION RISK PREDICTOR")
    print("====================================================")
    print()

    try:
        print("Loading model...")
        pipeline = load_pipeline()
        print("Model loaded successfully")
    except Exception as exc:  # pragma: no cover - user-facing failure path
        print(f"Error loading model: {exc}")
        raise

    model = pipeline["model"]
    feature_cols = pipeline.get("feature_cols", [])
    threshold = float(pipeline.get("threshold", 0.5))

    print()
    print("Model:")
    print(type(model).__name__)
    print()
    print("Features:")
    print(len(feature_cols))
    print()
    print("Decision threshold:")
    print(f"{threshold:.4f}")

    try:
        print()
        print("Loading sample events...")
        events = load_sample_events("sample_test_events.json")
        print(f"{len(events)} events loaded")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise

    predictions = predict_events(events, pipeline)

    print()
    print("PREDICTIONS")
    print("----------------------------------------------------")
    for index, item in enumerate(predictions, start=1):
        print(f"\nEvent {index}")
        print(f"Probability : {item['probability']:.4f}")
        print(f"Threshold   : {item['threshold']:.4f}")
        print(f"Prediction  : {item['prediction']}")

    output_path = Path("prediction_results.json")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)

    print()
    print("Generating outputs...")
    print("prediction_results.json")

    save_prediction_chart(predictions, threshold, "prediction_results.png")
    print("prediction_results.png")

    feature_saved = save_feature_importance(model, feature_cols, "feature_importance.png")
    if feature_saved:
        print("feature_importance.png")
    else:
        print("feature_importance.png not available for this model")

    metrics = compute_metrics(events, predictions)
    if metrics is not None:
        confusion_saved = save_confusion_matrix(events, predictions, "confusion_matrix_comparison.png")
        if confusion_saved:
            print("confusion_matrix_comparison.png")
    else:
        print("confusion_matrix_comparison.png left unchanged")

    print()
    print("====================================================")
    print("PROJECT RUN COMPLETED SUCCESSFULLY")
    print("====================================================")


if __name__ == "__main__":
    main()
