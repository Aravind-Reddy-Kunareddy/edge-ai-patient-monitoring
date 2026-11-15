
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


# 4. Load Edge and Cloud models
edge_models = joblib.load("models/edge_models.pkl")
edge_scaler = joblib.load("models/scaler.pkl")

cloud_models = joblib.load("models/cloud_models.pkl")
cloud_scaler = joblib.load("models/cloud_scaler.pkl")


# 5. Scale test data
X_edge_test = edge_scaler.transform(X_test)


# 6. Network configuration
CONFIDENCE_THRESHOLD = 0.80

BANDWIDTH_MBPS = 10
NETWORK_LATENCY_MS = 20

request_size_bytes = X_test.shape[1] * 8

transmission_time_ms = (
    request_size_bytes * 8
    / (BANDWIDTH_MBPS * 1_000_000)
) * 1000

network_delay_ms = (
    2 * NETWORK_LATENCY_MS
    + 2 * transmission_time_ms
)


# 7. Hybrid offloading simulation
results = []

print("\n========== HYBRID OFFLOADING STRATEGY ==========")

print("Confidence Threshold:", CONFIDENCE_THRESHOLD)
print("Network Bandwidth:", BANDWIDTH_MBPS, "Mbps")
print("Network Round-trip Delay:", round(network_delay_ms, 4), "ms")


for target in targets:

    print("\nProcessing:", target)

    edge_model = edge_models[target]
    cloud_model = cloud_models[target]

    # Edge prediction and confidence
    start_time = time.perf_counter()

    edge_probabilities = edge_model.predict_proba(X_edge_test)

    edge_predictions = edge_model.classes_[
        np.argmax(edge_probabilities, axis=1)
    ]

    edge_confidences = np.max(edge_probabilities, axis=1)

    edge_processing_time = time.perf_counter() - start_time

    # Identify low-confidence requests
    offload_mask = edge_confidences < CONFIDENCE_THRESHOLD

    offload_indices = np.where(offload_mask)[0]

    # Start with edge predictions
    final_predictions = edge_predictions.copy()

    cloud_processing_time = 0

    # Send uncertain requests to cloud
    if len(offload_indices) > 0:

        X_uncertain = X_test.iloc[offload_indices]

        X_cloud_scaled = cloud_scaler.transform(X_uncertain)

        start_time = time.perf_counter()

        cloud_predictions = cloud_model.predict(X_cloud_scaled)

        cloud_processing_time = time.perf_counter() - start_time

        # Replace uncertain edge predictions with cloud predictions
        final_predictions[offload_indices] = cloud_predictions

    # Evaluation
    accuracy = accuracy_score(
        y_test[target],
        final_predictions
    )

    f1 = f1_score(
        y_test[target],
        final_predictions,
        zero_division=0
    )

    total_requests = len(X_test)
    cloud_requests = len(offload_indices)

    offload_percentage = (
        cloud_requests / total_requests
    ) * 100

    # Calculate simulated total latency
    total_latency_ms = (
        edge_processing_time * 1000
        + cloud_processing_time * 1000
        + cloud_requests * network_delay_ms
    )

    average_latency_ms = total_latency_ms / total_requests

    results.append({
        "Target": target,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Edge Requests": total_requests - cloud_requests,
        "Cloud Requests": cloud_requests,
        "Offload Percentage": offload_percentage,
        "Average Latency (ms)": average_latency_ms
    })

    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))
    print("Edge Requests:", total_requests - cloud_requests)
    print("Cloud Requests:", cloud_requests)
    print("Offload Percentage:", round(offload_percentage, 2), "%")
    print("Average Latency:", round(average_latency_ms, 4), "ms")


# 8. Save results
results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/hybrid_results.csv",
    index=False
)

print("\n========== FINAL HYBRID RESULTS ==========")
print(results_df.round(4).to_string(index=False))

print("\nResults saved to results/hybrid_results.csv")
