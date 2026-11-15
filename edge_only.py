
import pandas as pd
import numpy as np
import joblib
import time
import os

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score


# 1. Define targets
targets = [
    "fall_detected",
    "abnormal_heart_rhythm",
    "hypoxemia_event",
    "fever_event"
]


# 2. Load dataset
df = pd.read_csv("data.csv")

X = df.drop(columns=[
    "patient_id",
    "room_number"
] + targets)

y = df[targets]

groups = df["patient_id"]


# 3. Recreate patient-wise test split
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


# 4. Load Edge models and scaler
edge_models = joblib.load("models/edge_models.pkl")
scaler = joblib.load("models/scaler.pkl")

X_test_scaled = scaler.transform(X_test)


# 5. Edge-only simulation
results = []

print("\n========== EDGE-ONLY STRATEGY ==========")

for target in targets:

    model = edge_models[target]

    start_time = time.perf_counter()

    predictions = model.predict(X_test_scaled)

    total_time = time.perf_counter() - start_time

    avg_latency_ms = (
        total_time / len(X_test_scaled)
    ) * 1000

    accuracy = accuracy_score(
        y_test[target],
        predictions
    )

    f1 = f1_score(
        y_test[target],
        predictions,
        zero_division=0
    )

    results.append({
        "Target": target,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Total Prediction Time (s)": total_time,
        "Average Latency (ms)": avg_latency_ms,
        "Cloud Requests": 0
    })

    print("\nTarget:", target)
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))
    print("Total Prediction Time:", round(total_time, 6), "seconds")
    print("Average Latency:", round(avg_latency_ms, 6), "ms")
    print("Cloud Requests: 0")


# 6. Save results
results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/edge_only_results.csv",
    index=False
)

print("\n========== FINAL EDGE-ONLY RESULTS ==========")
print(results_df.round(4).to_string(index=False))

print("\nResults saved to results/edge_only_results.csv")
