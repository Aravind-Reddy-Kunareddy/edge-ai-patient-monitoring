import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score

# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from system_monitor import get_all_resources
from network_monitor import measure_network

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.csv")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

CLOUD_BENCHMARK_PATH = os.path.join(
    PROJECT_ROOT, "results", "cloud_only_results.csv"
)

SPLIT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "shared_split"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "custom"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR, "custom_adaptive_summary.csv"
)

PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR, "custom_adaptive_predictions.csv"
)

# Prevent overwriting previous comparison results
if os.path.exists(SUMMARY_PATH) or os.path.exists(PREDICTIONS_PATH):
    raise FileExistsError(
        "Comparison results already exist. "
        "Rename the output files or use a new experiment folder."
    )


# =========================================================
# 2. SHARED CONFIGURATION
# =========================================================

CONFIDENCE_THRESHOLD = 0.90

DEADLINE_MS = 100

MAX_CPU_USAGE_PERCENT = 90

MIN_AVAILABLE_RAM_MB = 750

TARGETS = [
    "fall_detected",
    "abnormal_heart_rhythm",
    "hypoxemia_event",
    "fever_event"
]


# =========================================================
# 3. LOAD THE ENTIRE DATASET
# =========================================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["patient_id", "room_number"] + TARGETS)

y = df[TARGETS]

X_test = X.copy()

y_test = y.copy()

print("=" * 60)
print("CUSTOM SIMULATION - FULL DATASET EXPERIMENT")
print("=" * 60)

print("Total samples:", len(X_test))
print("Total patients:", df["patient_id"].nunique())
print("Total tasks:", len(X_test) * len(TARGETS))

# =========================================================
# 4. LOAD MODELS
# =========================================================

edge_models = joblib.load(
    os.path.join(MODEL_DIR, "edge_models.pkl")
)

edge_scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

cloud_models = joblib.load(
    os.path.join(MODEL_DIR, "cloud_models.pkl")
)

cloud_scaler = joblib.load(
    os.path.join(MODEL_DIR, "cloud_scaler.pkl")
)


# =========================================================
# 5. LOAD CLOUD BENCHMARK
# =========================================================

cloud_benchmark = pd.read_csv(CLOUD_BENCHMARK_PATH)

cloud_processing_times = dict(
    zip(
        cloud_benchmark["Target"],
        cloud_benchmark["Cloud Processing (ms)"]
    )
)


# =========================================================
# 6. SCALE TEST DATA
# =========================================================

X_edge_test = edge_scaler.transform(X_test)


# =========================================================
# 7. NETWORK MEASUREMENT
# =========================================================

print("\n========== NETWORK MEASUREMENT ==========")

network = measure_network()

upload_mbps = network["upload_mbps"]
download_mbps = network["download_mbps"]
network_latency_ms = network["latency_ms"]

print("\nUpload:", upload_mbps, "Mbps")
print("Download:", download_mbps, "Mbps")
print("Latency:", network_latency_ms, "ms")

if upload_mbps <= 0 or download_mbps <= 0:
    raise ValueError("Invalid network bandwidth measurement.")

# Same request size estimation as original Custom Simulation
request_size_bytes = X_test.shape[1] * 8

response_size_bytes = 8

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

print("Estimated network delay:", network_delay_ms, "ms")


# =========================================================
# 8. ADAPTIVE SIMULATION
# =========================================================

summary_results = []

prediction_results = []

total_runtime_start = time.perf_counter()

for target in TARGETS:

    print("\n" + "=" * 60)
    print("TARGET:", target)
    print("=" * 60)

    edge_model = edge_models[target]
    cloud_model = cloud_models[target]

    actual = y_test[target].values

    total_requests = len(X_test)

    # -----------------------------------------------------
    # EDGE INFERENCE
    # -----------------------------------------------------

    edge_start = time.perf_counter()

    edge_probabilities = edge_model.predict_proba(
        X_edge_test
    )

    edge_predictions = edge_model.classes_[
        np.argmax(edge_probabilities, axis=1)
    ]

    edge_confidences = np.max(
        edge_probabilities, axis=1
    )

    edge_total_time = time.perf_counter() - edge_start

    edge_time_per_request_ms = (
        edge_total_time * 1000 / total_requests
    )

    final_predictions = edge_predictions.copy()

    # -----------------------------------------------------
    # CONFIDENCE CHECK
    # -----------------------------------------------------

    uncertain_mask = (
        edge_confidences < CONFIDENCE_THRESHOLD
    )

    uncertain_indices = np.where(uncertain_mask)[0]

    confident_count = total_requests - len(uncertain_indices)

    print("Total requests:", total_requests)
    print("Confident requests:", confident_count)
    print("Uncertain requests:", len(uncertain_indices))

    # -----------------------------------------------------
    # RESOURCE CHECK
    # -----------------------------------------------------

    cpu_usage = np.nan
    available_ram = np.nan

    resource_pressure = False

    resource_check_time_ms = 0

    if len(uncertain_indices) > 0:

        resource_start = time.perf_counter()

        resources = get_all_resources(
            test_internet=False
        )

        resource_check_time_ms = (
            time.perf_counter() - resource_start
        ) * 1000

        cpu_usage = resources["cpu_usage_percent"]

        available_ram = resources["available_ram_mb"]

        resource_pressure = (
            cpu_usage >= MAX_CPU_USAGE_PERCENT
            or available_ram <= MIN_AVAILABLE_RAM_MB
        )

    print("CPU usage:", cpu_usage)
    print("Available RAM:", available_ram)
    print("Resource pressure:", resource_pressure)

    # -----------------------------------------------------
    # CLOUD FEASIBILITY
    # -----------------------------------------------------

    estimated_cloud_processing_ms = (
        cloud_processing_times[target] / total_requests
    )

    estimated_cloud_response_ms = (
        network_delay_ms + estimated_cloud_processing_ms
    )

    cloud_feasible = (
        estimated_cloud_response_ms <= DEADLINE_MS
    )

    print(
        "Estimated cloud processing per request:",
        estimated_cloud_processing_ms, "ms"
    )

    print(
        "Estimated cloud response:",
        estimated_cloud_response_ms, "ms"
    )

    print("Cloud feasible:", cloud_feasible)

    # -----------------------------------------------------
    # OFFLOADING DECISION
    # -----------------------------------------------------

    offload_indices = np.array([], dtype=int)

    if (
        len(uncertain_indices) > 0
        and not resource_pressure
        and cloud_feasible
    ):

        offload_indices = uncertain_indices.copy()

        print("Decision: OFFLOAD uncertain requests")

    else:

        print("Decision: RETAIN requests on Edge")

    # -----------------------------------------------------
    # CLOUD INFERENCE
    # -----------------------------------------------------

    cloud_total_time = 0

    cloud_requests = len(offload_indices)

    if cloud_requests > 0:

        X_cloud = X_test.iloc[offload_indices]

        X_cloud_scaled = cloud_scaler.transform(X_cloud)

        cloud_start = time.perf_counter()

        cloud_predictions = cloud_model.predict(
            X_cloud_scaled
        )

        cloud_total_time = (
            time.perf_counter() - cloud_start
        )

        final_predictions[offload_indices] = (
            cloud_predictions
        )

    cloud_time_per_request_ms = (
        cloud_total_time * 1000 / cloud_requests
        if cloud_requests > 0
        else 0
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        actual, final_predictions
    )

    f1 = f1_score(
        actual,
        final_predictions,
        zero_division=0
    )

    edge_requests = total_requests - cloud_requests

    offload_percentage = (
        cloud_requests / total_requests
    ) * 100

    # -----------------------------------------------------
    # LATENCY
    # -----------------------------------------------------

    edge_request_latency_ms = edge_time_per_request_ms

    cloud_request_latency_ms = (
        edge_time_per_request_ms
        + network_delay_ms
        + cloud_time_per_request_ms
    )

    average_latency_ms = (
        edge_requests * edge_request_latency_ms
        + cloud_requests * cloud_request_latency_ms
    ) / total_requests

    # -----------------------------------------------------
    # SAVE SUMMARY
    # -----------------------------------------------------

    summary_results.append({

        "Target": target,

        "Accuracy": accuracy,

        "F1 Score": f1,

        "Total Requests": total_requests,

        "Confident Requests": confident_count,

        "Uncertain Requests": len(uncertain_indices),

        "Edge Requests": edge_requests,

        "Cloud Requests": cloud_requests,

        "Offload Percentage": offload_percentage,

        "CPU Usage (%)": cpu_usage,

        "Available RAM (MB)": available_ram,

        "Resource Pressure": resource_pressure,

        "Upload (Mbps)": upload_mbps,

        "Download (Mbps)": download_mbps,

        "Network Latency (ms)": network_latency_ms,

        "Network Delay (ms)": network_delay_ms,

        "Cloud Feasible": cloud_feasible,

        "Estimated Cloud Response (ms)": estimated_cloud_response_ms,

        "Edge Inference (ms)": edge_total_time * 1000,

        "Edge Per Request (ms)": edge_time_per_request_ms,

        "Cloud Inference Total (ms)": cloud_total_time * 1000,

        "Cloud Per Request (ms)": cloud_time_per_request_ms,

        "Resource Check Time (ms)": resource_check_time_ms,

        "Average Latency (ms)": average_latency_ms,

        "Average Cloud Request Latency (ms)": (
            cloud_request_latency_ms if cloud_requests > 0 else 0
        )
    })

    # -----------------------------------------------------
    # SAVE SAMPLE-LEVEL PREDICTIONS
    # -----------------------------------------------------

    for i in range(total_requests):

        prediction_results.append({

            "Sample Index": int(i),

            "Target": target,

            "Actual": actual[i],

            "Edge Prediction": edge_predictions[i],

            "Final Prediction": final_predictions[i],

            "Confidence": edge_confidences[i],

            "Uncertain": bool(uncertain_mask[i]),

            "Offloaded": bool(i in offload_indices),

        })

    print("\nAccuracy:", round(accuracy, 5))
    print("F1 Score:", round(f1, 5))
    print("Edge requests:", edge_requests)
    print("Cloud requests:", cloud_requests)
    print("Offload:", round(offload_percentage, 2), "%")
    print("Average latency:", round(average_latency_ms, 6), "ms")


# =========================================================
# 9. SAVE FINAL RESULTS
# =========================================================

total_runtime = (
    time.perf_counter() - total_runtime_start
)

summary_df = pd.DataFrame(summary_results)

predictions_df = pd.DataFrame(prediction_results)

summary_df.to_csv(SUMMARY_PATH, index=False)

predictions_df.to_csv(PREDICTIONS_PATH, index=False)


# =========================================================
# 10. FINAL REPORT
# =========================================================

print("\n" + "=" * 80)
print("CUSTOM SIMULATION - FINAL COMPARISON RESULTS")
print("=" * 80)

print(summary_df.round(5).to_string(index=False))

print("\nTotal runtime:", round(total_runtime, 3), "seconds")

print("\nSaved summary:")
print(SUMMARY_PATH)

print("\nSaved sample-level predictions:")
print(PREDICTIONS_PATH)

print("\nExperiment completed successfully.")