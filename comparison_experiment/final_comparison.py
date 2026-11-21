import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CUSTOM_PATH = os.path.join(
    RESULTS_DIR, "custom", "custom_adaptive_predictions.csv"
)

EDGESIMPY_PATH = os.path.join(
    RESULTS_DIR, "edgesimpy", "edgesimpy_predictions.csv"
)

CUSTOM_SUMMARY_PATH = os.path.join(
    RESULTS_DIR, "custom", "custom_adaptive_summary.csv"
)

EDGESIMPY_SUMMARY_PATH = os.path.join(
    RESULTS_DIR, "edgesimpy", "edgesimpy_summary.csv"
)

OUTPUT_DIR = os.path.join(RESULTS_DIR, "final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

custom = pd.read_csv(CUSTOM_PATH)
edgesimpy = pd.read_csv(EDGESIMPY_PATH)

custom_summary = pd.read_csv(CUSTOM_SUMMARY_PATH)
edgesimpy_summary = pd.read_csv(EDGESIMPY_SUMMARY_PATH)

# --------------------------------------------------
# STANDARDIZE COLUMN NAMES
# --------------------------------------------------

custom = custom.rename(columns={
    "Sample Index": "sample_index",
    "Target": "target",
    "Actual": "actual",
    "Final Prediction": "final_prediction",
    "Offloaded": "offloaded"
})

custom["system"] = "Custom Adaptive"

edgesimpy["system"] = "EdgeSimPy"

# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

def calculate_metrics(df, system_name):

    results = []

    for target in df["target"].unique():

        target_df = df[df["target"] == target]

        y_true = target_df["actual"]
        y_pred = target_df["final_prediction"]

        results.append({
            "System": system_name,
            "Target": target,
            "Accuracy (%)": accuracy_score(y_true, y_pred) * 100,
            "Precision (%)": precision_score(
                y_true, y_pred, zero_division=0
            ) * 100,
            "Recall (%)": recall_score(
                y_true, y_pred, zero_division=0
            ) * 100,
            "F1 Score (%)": f1_score(
                y_true, y_pred, zero_division=0
            ) * 100,
            "Total Tasks": len(target_df)
        })

    return pd.DataFrame(results)


custom_metrics = calculate_metrics(custom, "Custom Adaptive")
edgesimpy_metrics = calculate_metrics(edgesimpy, "EdgeSimPy")

target_metrics = pd.concat(
    [custom_metrics, edgesimpy_metrics],
    ignore_index=True
)

# Save target-wise metrics
target_metrics.to_csv(
    os.path.join(OUTPUT_DIR, "target_wise_comparison.csv"),
    index=False
)

# --------------------------------------------------
# OVERALL METRICS
# --------------------------------------------------

custom_f1 = custom_metrics["F1 Score (%)"].mean()
edgesimpy_f1 = edgesimpy_metrics["F1 Score (%)"].mean()

custom_accuracy = custom_metrics["Accuracy (%)"].mean()
edgesimpy_accuracy = edgesimpy_metrics["Accuracy (%)"].mean()

# --------------------------------------------------
# CUSTOM ADAPTIVE STATISTICS
# --------------------------------------------------

custom_tasks = len(custom)

custom_cloud = custom["offloaded"].astype(bool).sum()
custom_edge = custom_tasks - custom_cloud

custom_offload_percent = (
    custom_cloud / custom_tasks * 100
)

custom_avg_latency = custom_summary[
    "Average Latency (ms)"
].mean()

custom_resource_check = custom_summary[
    "Resource Check Time (ms)"
].mean()

custom_total_resource_check = custom_summary[
    "Resource Check Time (ms)"
].sum()

custom_simulation_time = None

# --------------------------------------------------
# EDGESIMPY STATISTICS
# --------------------------------------------------

edgesimpy_tasks = len(edgesimpy)

cloud_tasks = (
    edgesimpy["execution"] == "cloud"
).sum()

fallback_tasks = (
    edgesimpy["execution"] == "edge_fallback"
).sum()

direct_edge_tasks = (
    edgesimpy["execution"] == "edge"
).sum()

total_edge_tasks = direct_edge_tasks + fallback_tasks

edgesimpy_offload_percent = (
    cloud_tasks / edgesimpy_tasks * 100
)

edgesimpy_avg_latency = edgesimpy[
    "total_latency_ms"
].mean()

edgesimpy_avg_resource_check = edgesimpy[
    "resource_check_ms"
].mean()

edgesimpy_total_resource_check = edgesimpy[
    "resource_check_ms"
].sum()

# Latency excluding resource-monitoring overhead
edgesimpy_latency_without_resource_check = (
    edgesimpy["total_latency_ms"]
    - edgesimpy["resource_check_ms"]
).mean()

edgesimpy_avg_network_delay = edgesimpy[
    "network_delay_ms"
].mean()

edgesimpy_simulation_seconds = edgesimpy_summary[
    "total_simulation_time_seconds"
].iloc[0]

# --------------------------------------------------
# CUSTOM LATENCY / RESOURCE INFORMATION
# --------------------------------------------------

custom_avg_network_delay = custom_summary[
    "Network Delay (ms)"
].mean()

custom_cloud_latency = custom_summary[
    "Average Cloud Request Latency (ms)"
].mean()

# --------------------------------------------------
# FINAL COMPARISON TABLE
# --------------------------------------------------

comparison = pd.DataFrame({

    "Metric": [
        "Total Samples",
        "Total Tasks",
        "Accuracy (%)",
        "Macro F1 Score (%)",
        "Edge Direct Tasks",
        "Edge Fallback Tasks",
        "Total Edge Executions",
        "Cloud Tasks",
        "Cloud Offloading (%)",
        "Average Latency (ms)",
        "Latency Excluding Resource Checks (ms)",
        "Average Resource Check (ms/task)",
        "Total Resource Check Time (ms)",
        "Average Network Delay (ms)",
        "Average Cloud Request Latency (ms)",
        "Simulation Runtime (minutes)"
    ],

    "Custom Adaptive": [
        10000,
        custom_tasks,
        custom_accuracy,
        custom_f1,
        custom_edge,
        0,
        custom_edge,
        custom_cloud,
        custom_offload_percent,
        custom_avg_latency,
        custom_avg_latency,
        custom_resource_check,
        custom_total_resource_check,
        custom_avg_network_delay,
        custom_cloud_latency,
        None
    ],

    "EdgeSimPy": [
        10000,
        edgesimpy_tasks,
        edgesimpy_accuracy,
        edgesimpy_f1,
        direct_edge_tasks,
        fallback_tasks,
        total_edge_tasks,
        cloud_tasks,
        edgesimpy_offload_percent,
        edgesimpy_avg_latency,
        edgesimpy_latency_without_resource_check,
        edgesimpy_avg_resource_check,
        edgesimpy_total_resource_check,
        edgesimpy_avg_network_delay,
        None,
        edgesimpy_simulation_seconds / 60
    ]
})

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

comparison.to_csv(
    os.path.join(OUTPUT_DIR, "system_comparison.csv"),
    index=False
)

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\n" + "=" * 80)
print("FINAL SYSTEM COMPARISON")
print("=" * 80)

print(comparison.to_string(index=False))

print("\n" + "=" * 80)
print("TARGET-WISE PERFORMANCE")
print("=" * 80)

print(target_metrics.to_string(index=False))

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print("System comparison:")
print(os.path.join(OUTPUT_DIR, "system_comparison.csv"))

print("\nTarget-wise comparison:")
print(os.path.join(OUTPUT_DIR, "target_wise_comparison.csv"))

print("\nComparison completed successfully.")