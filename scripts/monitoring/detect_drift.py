"""
Drift Detection Module
---------------------
Detects:
1. Data drift using comment length distribution
2. Prediction drift using sentiment distribution

Outputs:
- console report 
- Machine-readable JSON
- Exit code for CI/CD
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import boto3


# ============================================================
# Utility
# ============================================================
def r(value, decimals=2):
    """Round values safely for reporting and JSON serialization"""
    return round(float(value), decimals)


# ============================================================
# Load training baseline
# ============================================================
def load_training_baseline():
    df = pd.read_csv("data/processed/train_processed.csv")

    baseline = {
        "avg_length": df["clean_comment"].str.split().str.len().mean(),
        "std_length": df["clean_comment"].str.split().str.len().std(),
        "class_distribution": {
            "-1": (df["category"] == -1).sum() / len(df),
            "0": (df["category"] == 0).sum() / len(df),
            "1": (df["category"] == 1).sum() / len(df),
        },
    }

    return baseline


# ============================================================
# Load production predictions
# ============================================================
def load_production_predictions():
    s3 = boto3.client("s3")
    bucket = "creator-insight-dvc-bucket"
    prefix = "retraining_data/archive/"

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" not in response:
        print("No production predictions found")
        return None

    files = [f for f in response["Contents"] if f["Key"].endswith(".json")]

    if not files:
        print("No JSON prediction files found")
        return None

    latest = sorted(files, key=lambda x: x["LastModified"], reverse=True)[0]
    obj = s3.get_object(Bucket=bucket, Key=latest["Key"])
    data = json.loads(obj["Body"].read())

    return data["data"]


# ============================================================
# Drift detection
# ============================================================
def detect_drift(baseline, production_data):

    # -----------------------------
    # Comment length drift
    # -----------------------------
    prod_lengths = [
        len(str(p.get("comment", "")).split())
        for p in production_data
    ]

    prod_avg_length = np.mean(prod_lengths)
    prod_std_length = np.std(prod_lengths)

    length_change_pct = (
        (prod_avg_length - baseline["avg_length"])
        / baseline["avg_length"]
        * 100
    )

    # -----------------------------
    # Sentiment distribution drift
    # -----------------------------
    prod_sentiments = [
        str(p.get("sentiment"))
        for p in production_data
        if p.get("sentiment") is not None
    ]

    prod_dist = {
        "-1": prod_sentiments.count("-1") / len(prod_sentiments),
        "0": prod_sentiments.count("0") / len(prod_sentiments),
        "1": prod_sentiments.count("1") / len(prod_sentiments),
    }

    dist_change = {
        k: prod_dist[k] - baseline["class_distribution"][k]
        for k in baseline["class_distribution"]
    }

    # -----------------------------
    # Drift decision rules
    # -----------------------------
    length_drift = abs(length_change_pct) > 20
    sentiment_drift = any(abs(v) > 0.15 for v in dist_change.values())
    drift_detected = length_drift or sentiment_drift

    # ============================================================
    # Console Report (Professional)
    # ============================================================
    print("\n" + "=" * 70)
    print("DRIFT DETECTION REPORT")
    print("=" * 70)

    print("\nTraining Baseline:")
    print(f"Average comment length : {r(baseline['avg_length'])}")
    print(f"Standard deviation     : {r(baseline['std_length'])}")
    print(
        "Class distribution     : "
        f"{{'-1': {r(baseline['class_distribution']['-1'], 3)}, "
        f"'0': {r(baseline['class_distribution']['0'], 3)}, "
        f"'1': {r(baseline['class_distribution']['1'], 3)}}}"
    )

    print("\nProduction Data:")
    print(f"Average comment length : {r(prod_avg_length)}")
    print(f"Standard deviation     : {r(prod_std_length)}")
    print(
        "Class distribution     : "
        f"{{'-1': {r(prod_dist['-1'], 3)}, "
        f"'0': {r(prod_dist['0'], 3)}, "
        f"'1': {r(prod_dist['1'], 3)}}}"
    )

    print("\nDrift Metrics:")
    print(f"Comment length change (%) : {r(length_change_pct)}")

    print("\nSentiment distribution change (%):")
    for k, v in dist_change.items():
        print(f"Class {k}: {r(v * 100)}")

    print("\nDrift Status:")
    if drift_detected:
        print("Drift detected. Retraining recommended.")
    else:
        print("No significant drift detected.")

    print("=" * 70 + "\n")

    # ============================================================
    # JSON Output
    # ============================================================
    return {
        "drift_detected": bool(drift_detected),
        "length_drift": bool(length_drift),
        "sentiment_drift": bool(sentiment_drift),
        "baseline_avg_length": r(baseline["avg_length"]),
        "production_avg_length": r(prod_avg_length),
        "length_change_percentage": r(length_change_pct),
        "sentiment_distribution_change": {
            k: r(v * 100) for k, v in dist_change.items()
        },
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# Main
# ============================================================
def main():
    print("Loading training baseline...")
    baseline = load_training_baseline()

    print("Loading production predictions...")
    production_data = load_production_predictions()

    if not production_data:
        print("No production data available")
        return 1

    result = detect_drift(baseline, production_data)

    with open("drift_detection_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Drift detection results saved to drift_detection_result.json")

    return 1 if result["drift_detected"] else 0


if __name__ == "__main__":
    exit(main())
