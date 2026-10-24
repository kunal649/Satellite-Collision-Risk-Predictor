# Satellite-Collision-Risk-Predictor

## Run the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

Generated outputs:

- `prediction_results.json`
- `prediction_results.png`
- `feature_importance.png`
- `confusion_matrix_comparison.png` when labels are available

Input:

```text
sample_test_events.json
```

Model:

```text
collision_model_pipeline.joblib
```

Output:

```text
HIGH RISK / LOW RISK classification
```
