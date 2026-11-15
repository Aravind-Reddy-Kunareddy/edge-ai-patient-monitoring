
import pandas as pd
import numpy as np
import joblib
import time
import os

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from system_monitor import get_all_resources
from network_monitor import measure_network


# ============================================================
# 1. DEFINE TARGETS
# ============================================================

targets = [
    "fall_detected",
    "abnormal_heart_rhythm",
    "hypoxemia_event",
    "fever_event"
]


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("data.csv")

X = df.drop(columns=[
    "patient_id",
    "room_number"
] + targets)

y = df[targets]

groups = df["patient_id"]


# ============================================================
# 3. RECREATE PATIENT-WISE TEST SPLIT
# ============================================================

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


# ============================================================
# 4. LOAD EDGE AND CLOUD MODELS
# ============================================================

edge_models = joblib.load("models/edge_models.pkl")
edge_scaler = joblib.load("models/scaler.pkl")

cloud_models = joblib.load("models/cloud_models.pkl")
cloud_scaler = joblib.load("models/cloud_scaler.pkl")


# ============================================================
# 5. LOAD PREVIOUS CLOUD BENCHMARK
# ============================================================

cloud_results = pd.read_csv(
    "results/cloud_only_results.csv"
)

cloud_processing_times = dict(zip(
    cloud_results["Target"],
    cloud_results["Cloud Processing (ms)"]
))


# ============================================================
# 6. SCALE TEST DATA FOR EDGE MODEL
# ============================================================

X_edge_test = edge_scaler.transform(X_test)


# ============================================================
# 7. ADAPTIVE SYSTEM CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.80

DEADLINE_MS = 5

MAX_CPU_USAGE_PERCENT = 90

MIN_AVAILABLE_RAM_MB = 750

request_size_bytes = X_test.shape[1] * 8
response_size_bytes = 8


# ============================================================
# 8. MEASURE INTERNET CONNECTION
# ============================================================

print("\n========== INITIAL NETWORK MEASUREMENT ==========")

network = measure_network()

upload_mbps = network["upload_mbps"]
download_mbps = network["download_mbps"]
network_latency_ms = network["latency_ms"]

print("\nMeasured Upload:", upload_mbps, "Mbps")
print("Measured Download:", download_mbps, "Mbps")
print("Measured Latency:", network_latency_ms, "ms")


if upload_mbps <= 0 or download_mbps <= 0:
    raise ValueError("Network throughput measurement is invalid.")


# Estimate data transmission time

upload_time_ms = (
    request_size_bytes * 8
    / (upload_mbps * 1_000_000)
) * 1000

download_time_ms = (
    response_size_bytes * 8
    / (download_mbps * 1_000_000)
) * 1000

network_delay_ms = (
    network_latency_ms
    + upload_time_ms
    + download_time_ms
)


# ============================================================
# 9. ADAPTIVE OFFLOADING
# ============================================================

results = []

print("\n========== ACCURACY-FIRST ADAPTIVE OFFLOADING ==========")

print("Confidence Threshold:", CONFIDENCE_THRESHOLD)
print("Deadline:", DEADLINE_MS, "ms")


for target in targets:

    print("\n====================================")
    print("TARGET:", target)
    print("====================================")

    edge_model = edge_models[target]
    cloud_model = cloud_models[target]

    total_requests = len(X_test)

    # --------------------------------------------------------
    # 9A. RUN EDGE MODEL FIRST
    # --------------------------------------------------------

    print("\nRunning Edge Model...")

    start_time = time.perf_counter()

    edge_probabilities = edge_model.predict_proba(
        X_edge_test
    )

    edge_predictions = edge_model.classes_[
        np.argmax(edge_probabilities, axis=1)
    ]

    edge_confidences = np.max(
        edge_probabilities,
        axis=1
    )

    edge_processing_time = (
        time.perf_counter() - start_time
    )

    # Initially, all predictions come from Edge
    final_predictions = edge_predictions.copy()

    # --------------------------------------------------------
    # 9B. CHECK EDGE PREDICTION CONFIDENCE
    # --------------------------------------------------------

    uncertain_mask = (
        edge_confidences < CONFIDENCE_THRESHOLD
    )

    uncertain_indices = np.where(
        uncertain_mask
    )[0]

    print("\nTotal Requests:", total_requests)

    print("Uncertain Requests:", len(uncertain_indices))

    print(
        "Confident Requests:",
        total_requests - len(uncertain_indices)
    )

    # --------------------------------------------------------
    # 9C. INITIALIZE RESOURCE INFORMATION
    # --------------------------------------------------------

    cpu_usage = np.nan
    available_ram = np.nan

    edge_resource_pressure = False

    cloud_requests = 0
    cloud_processing_time = 0

    offload_indices = np.array([], dtype=int)

    # --------------------------------------------------------
    # 9D. CHECK RESOURCES ONLY IF UNCERTAIN PREDICTIONS EXIST
    # --------------------------------------------------------

    if len(uncertain_indices) > 0:

        print("\nEdge confidence is below threshold.")

        print("Checking current system resources...")

        resources = get_all_resources(
            test_internet=False
        )

        cpu_usage = resources["cpu_usage_percent"]

        available_ram = resources["available_ram_mb"]

        edge_resource_pressure = (
            cpu_usage >= MAX_CPU_USAGE_PERCENT
            or available_ram <= MIN_AVAILABLE_RAM_MB
        )

        print("\nCPU Usage:", cpu_usage, "%")

        print("Available RAM:", available_ram, "MB")

        print("Resource Pressure:", edge_resource_pressure)

    else:

        print("\nAll predictions have sufficient confidence.")

        print("No cloud offloading required.")

    # --------------------------------------------------------
    # 9E. ESTIMATE CLOUD RESPONSE TIME
    # --------------------------------------------------------

    # Approximate cloud processing time per request
    # using the previous Cloud Only benchmark.

    estimated_cloud_processing = (
        cloud_processing_times[target]
        / total_requests
    )

    estimated_cloud_response = (
        network_delay_ms
        + estimated_cloud_processing
    )

    cloud_feasible = (
        estimated_cloud_response <= DEADLINE_MS
    )

    print(
        "\nEstimated Network Delay:",
        round(network_delay_ms, 4),
        "ms"
    )

    print(
        "Estimated Cloud Response:",
        round(estimated_cloud_response, 4),
        "ms"
    )

    print("Cloud Feasible:", cloud_feasible)

    # --------------------------------------------------------
    # 9F. ADAPTIVE OFFLOADING DECISION
    # --------------------------------------------------------

    if len(uncertain_indices) > 0:

        if not edge_resource_pressure and cloud_feasible:

            print("\nResources are available.")

            print("Cloud deadline can be satisfied.")

            print("Offloading uncertain requests to Cloud.")

            offload_indices = uncertain_indices

        else:

            print("\nCloud offloading conditions not satisfied.")

            if edge_resource_pressure:
                print("Reason: Edge resource pressure.")

            if not cloud_feasible:
                print("Reason: Cloud deadline exceeded.")

            print("Retaining Edge predictions.")

    # --------------------------------------------------------
    # 9G. EXECUTE CLOUD MODEL
    # --------------------------------------------------------

    if len(offload_indices) > 0:

        # IMPORTANT: Use iloc to select rows by position.
        X_cloud = X_test.iloc[offload_indices]

        X_cloud_scaled = cloud_scaler.transform(
            X_cloud
        )

        print("\nExecuting Cloud Model...")

        start_time = time.perf_counter()

        cloud_predictions = cloud_model.predict(
            X_cloud_scaled
        )

        cloud_processing_time = (
            time.perf_counter() - start_time
        )

        # Replace Edge predictions with Cloud predictions
        final_predictions[offload_indices] = (
            cloud_predictions
        )

        cloud_requests = len(offload_indices)

    # --------------------------------------------------------
    # 9H. REQUEST COUNTS
    # --------------------------------------------------------

    edge_requests = total_requests - cloud_requests

    offload_percentage = (
        cloud_requests / total_requests
    ) * 100

    # --------------------------------------------------------
    # 9I. EVALUATION
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test[target],
        final_predictions
    )

    f1 = f1_score(
        y_test[target],
        final_predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # 9J. LATENCY CALCULATION
    # --------------------------------------------------------

    # Average Edge processing time per request
    edge_latency_per_request_ms = (
        edge_processing_time * 1000 / total_requests
    )

    # Average Cloud processing time per offloaded request
    # Reuse the Cloud execution time measured in section 9G.
    if cloud_requests > 0:

        cloud_latency_per_request_ms = (
            cloud_processing_time * 1000 / cloud_requests
        )

    else:

        cloud_latency_per_request_ms = 0

    # Latency experienced by an Edge-only request
    edge_request_latency_ms = (
        edge_latency_per_request_ms
    )

    # Latency experienced by an offloaded request
    cloud_request_latency_ms = (
        edge_latency_per_request_ms
        + network_delay_ms
        + cloud_latency_per_request_ms
    )

    # Average latency across all requests
    average_latency_ms = (
        edge_requests * edge_request_latency_ms
        + cloud_requests * cloud_request_latency_ms
    ) / total_requests

    # Average latency for cloud requests
    average_cloud_latency_ms = (
        cloud_request_latency_ms
        if cloud_requests > 0
        else 0
    )

    print("\n========== LATENCY BREAKDOWN ==========")

    print(
        f"Edge processing per request: "
        f"{edge_latency_per_request_ms:.6f} ms"
    )

    print(
        f"Cloud processing per request: "
        f"{cloud_latency_per_request_ms:.6f} ms"
    )

    print(
        f"Network delay: "
        f"{network_delay_ms:.4f} ms"
    )

    print(
        f"Edge request latency: "
        f"{edge_request_latency_ms:.6f} ms"
    )

    print(
        f"Cloud request latency: "
        f"{cloud_request_latency_ms:.4f} ms"
    )

    print(
        f"Overall average latency: "
        f"{average_latency_ms:.6f} ms"
    )

    # --------------------------------------------------------
    # 9K. SAVE RESULTS
    # --------------------------------------------------------

    results.append({

        "Target": target,

        "Accuracy": accuracy,

        "F1 Score": f1,

        "CPU Usage (%)": cpu_usage,

        "Available RAM (MB)": available_ram,

        "Upload Speed (Mbps)": upload_mbps,

        "Download Speed (Mbps)": download_mbps,

        "Network Latency (ms)": network_latency_ms,

        "Edge Resource Pressure": edge_resource_pressure,

        "Edge Requests": edge_requests,

        "Cloud Requests": cloud_requests,

        "Offload Percentage": offload_percentage,

        "Cloud Feasible": cloud_feasible,

        "Estimated Cloud Response (ms)": estimated_cloud_response,

        "Average Latency (ms)": average_latency_ms,

        "Average Cloud Request Latency (ms)": average_cloud_latency_ms
    })

    print("\n========== FINAL DECISION ==========")

    print("Accuracy:", round(accuracy, 4))

    print("F1 Score:", round(f1, 4))

    print("Edge Requests:", edge_requests)

    print("Cloud Requests:", cloud_requests)

    print(
        "Offload Percentage:",
        round(offload_percentage, 2),
        "%"
    )

    print(
        "Average Latency:",
        round(average_latency_ms, 4),
        "ms"
    )

    print(
        "Average Cloud Request Latency:",
        round(average_cloud_latency_ms, 4),
        "ms"
    )


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/final_adaptive_results.csv",
    index=False
)

print("\n========== FINAL RESULTS ==========")

print(results_df.round(4).to_string(index=False))

print(
    "\nResults saved to "
    "results/final_adaptive_results.csv"
)
