
import pandas as pd
import numpy as np
import joblib
import os
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import GroupShuffleSplit


# 1. Load dataset
df = pd.read_csv("data.csv")

targets = [
    "fall_detected",
    "abnormal_heart_rhythm",
    "hypoxemia_event",
    "fever_event"
]

X = df.drop(columns=[
    "patient_id",
    "room_number"
] + targets)

y = df[targets]

groups = df["patient_id"]


# 2. Recreate the same patient-wise test split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]


# 3. Load saved models and scalers
edge_models = joblib.load("models/edge_models.pkl")
edge_scaler = joblib.load("models/scaler.pkl")

cloud_models = joblib.load("models/cloud_models.pkl")
cloud_scaler = joblib.load("models/cloud_scaler.pkl")


# 4. Scale test data separately
X_edge_test = edge_scaler.transform(X_test)
X_cloud_test = cloud_scaler.transform(X_test)


# 5. Compare models
results = []

for target in targets:

    print("\nBenchmarking:", target)

    actual = y_test[target]

    # Edge prediction
    edge_pred = edge_models[target].predict(X_edge_test)

    # Cloud prediction
    cloud_pred = cloud_models[target].predict(X_cloud_test)

    # Edge metrics
    edge_accuracy = accuracy_score(actual, edge_pred)
    edge_precision = precision_score(actual, edge_pred, zero_division=0)
    edge_recall = recall_score(actual, edge_pred, zero_division=0)
    edge_f1 = f1_score(actual, edge_pred, zero_division=0)

    # Cloud metrics
    cloud_accuracy = accuracy_score(actual, cloud_pred)
    cloud_precision = precision_score(actual, cloud_pred, zero_division=0)
    cloud_recall = recall_score(actual, cloud_pred, zero_division=0)
    cloud_f1 = f1_score(actual, cloud_pred, zero_division=0)

    # Model sizes
    edge_size = len(pickle.dumps(edge_models[target]))
    cloud_size = len(pickle.dumps(cloud_models[target]))

    results.append({
        "Target": target,
        "Edge Accuracy": edge_accuracy,
        "Cloud Accuracy": cloud_accuracy,
        "Edge Precision": edge_precision,
        "Cloud Precision": cloud_precision,
        "Edge Recall": edge_recall,
        "Cloud Recall": cloud_recall,
        "Edge F1": edge_f1,
        "Cloud F1": cloud_f1,
        "Edge Size (KB)": edge_size / 1024,
        "Cloud Size (KB)": cloud_size / 1024
    })


# 6. Create results table
results_df = pd.DataFrame(results)

print("\n========== EDGE vs CLOUD BENCHMARK ==========\n")

print(results_df.round(4).to_string(index=False))


# 7. Save benchmark results
os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/model_benchmark.csv",
    index=False
)

print("\nBenchmark results saved to results/model_benchmark.csv")
