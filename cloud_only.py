
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


# 4. Load Cloud models and scaler
cloud_models = joblib.load("models/cloud_models.pkl")
cloud_scaler = joblib.load("models/cloud_scaler.pkl")

X_test_scaled = cloud_scaler.transform(X_test)


# 5. Simulate Cloud-only prediction
results = []

# Simulated network conditions
bandwidth_mbps = 10
network_latency_ms = 20

# Size of one sensor request in bytes
request_size_bytes = X_test.shape[1] * 8

transmission_time_ms = (
    request_size_bytes * 8
    / (bandwidth_mbps * 1_000_000)
) * 1000

print("\n========== CLOUD-ONLY STRATEGY ==========")

for target in targets:

    model = cloud_models[target]

    # Measure cloud model prediction time
    start_time = time.perf_counter()

    predictions = model.predict(X_test_scaled)

    processing_time = time.perf_counter() - start_time

    # Simulated network round-trip delay
    network_time_ms = (
        2 * network_latency_ms
        + 2 * transmission_time_ms
    )

    processing_time_ms = processing_time * 1000

    total_latency_ms = (
        network_time_ms + processing_time_ms
    )

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
        "Cloud Processing (ms)": processing_time_ms,
        "Network Delay (ms)": network_time_ms,
        "Total Latency (ms)": total_latency_ms,
        "Cloud Requests": len(X_test)
    })

    print("\nTarget:", target)
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))
    print("Cloud Processing:", round(processing_time_ms, 4), "ms")
    print("Network Delay:", round(network_time_ms, 4), "ms")
    print("Total Latency:", round(total_latency_ms, 4), "ms")
    print("Cloud Requests:", len(X_test))


# 6. Save results
results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/cloud_only_results.csv",
    index=False
)

print("\n========== FINAL CLOUD-ONLY RESULTS ==========")
print(results_df.round(4).to_string(index=False))

print("\nResults saved to results/cloud_only_results.csv")
