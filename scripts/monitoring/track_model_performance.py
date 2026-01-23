"""
Track model performance across versions (MLflow Registry)

Purpose:
- Print a readable table (CI/CD logs)
"""

import os
import mlflow
import pandas as pd
from datetime import datetime

DEFAULT_MLFLOW_TRACKING_URI = "http://ec2-52-73-142-244.compute-1.amazonaws.com:5000/"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "creatorinsight_sentiment_pipeline"


def get_all_models_performance():
    client = mlflow.MlflowClient()
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    rows = []
    for v in all_versions:
        run = client.get_run(v.run_id)
        m = run.data.metrics

        rows.append({
            "version": int(v.version),
            "created_at": datetime.fromtimestamp(v.creation_timestamp / 1000),
            "stage": v.current_stage,
            "f1_score": float(m.get("test_weighted avg_f1-score", 0.0)),
            "accuracy": float(m.get("test_accuracy", 0.0)),
            "negative_recall": float(m.get("test_-1_recall", 0.0)),
            "run_id": v.run_id
        })

    df = pd.DataFrame(rows).sort_values("version")

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE HISTORY (MLFLOW)")
    print("=" * 70)
    if df.empty:
        print("No model versions found in MLflow Registry.")
    else:
        # Keep it readable
        show = df[["version", "stage", "f1_score", "accuracy", "negative_recall", "created_at"]].copy()
        show["f1_score"] = show["f1_score"].round(3)
        show["accuracy"] = show["accuracy"].round(3)
        show["negative_recall"] = show["negative_recall"].round(3)
        print(show.to_string(index=False))

        df.to_csv("model_performance_history.csv", index=False)
        print("\nSaved to: model_performance_history.csv\n")

    return df


if __name__ == "__main__":
    get_all_models_performance()
