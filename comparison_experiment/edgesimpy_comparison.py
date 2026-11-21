
import time
import json
import pandas as pd

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATIENT_MONITORING_DIR = os.path.join(
    PROJECT_ROOT, "simulations", "patient_monitoring"
)

sys.path.insert(0, PATIENT_MONITORING_DIR)
sys.path.insert(0, PROJECT_ROOT)

from load_project import df, FEATURES, TARGETS
from model_executor import execute_model

from resource_aware_policy import (
    CONFIDENCE_THRESHOLD,
    make_offloading_decision
)


# ==================================================
# CONFIGURATION
# ==================================================

# None means run the entire dataset
NUM_SAMPLES = None

RESULT_FILE = "confidence_offloading_results.csv"
SUMMARY_FILE = "confidence_offloading_summary.csv"

PROGRESS_INTERVAL = 1000


# ==================================================
# TRANSFER LATENCY
# ==================================================

def transfer_time_ms(size_bytes, bandwidth_mbps):

    if bandwidth_mbps is None or bandwidth_mbps <= 0:
        return float("inf")

    return (size_bytes * 8) / (bandwidth_mbps * 1000)


# ==================================================
# SIMULATION
# ==================================================

results = []

if NUM_SAMPLES is None:
    num_samples = len(df)
else:
    num_samples = min(NUM_SAMPLES, len(df))


total_expected_tasks = num_samples * len(TARGETS)

print("=" * 70)
print("CONFIDENCE-BASED RESOURCE-AWARE OFFLOADING")
print("=" * 70)

print(f"Dataset samples       : {num_samples}")
print(f"Targets per sample    : {len(TARGETS)}")
print(f"Total expected tasks  : {total_expected_tasks}")
print(f"Confidence threshold  : {CONFIDENCE_THRESHOLD * 100:.0f}%")
print("=" * 70)


simulation_start = time.perf_counter()


for sample_index in range(num_samples):

    for target in TARGETS:

        # --------------------------------------------------
        # 1. EXECUTE EDGE MODEL FIRST
        # --------------------------------------------------

        edge_result = execute_model(
            sample_index,
            target,
            "edge"
        )

        edge_prediction = edge_result["prediction"]
        edge_confidence = edge_result["confidence"]
        actual = edge_result["actual"]

        edge_inference_ms = edge_result["inference_time_ms"]

        # --------------------------------------------------
        # 2. HIGH CONFIDENCE -> EDGE
        # --------------------------------------------------

        if edge_confidence >= CONFIDENCE_THRESHOLD:

            execution = "edge"

            final_prediction = edge_prediction
            final_confidence = edge_confidence

            reason = "High Edge confidence"

            resources = None

            resource_check_ms = 0.0
            cloud_inference_ms = 0.0
            network_delay_ms = 0.0

            total_latency_ms = edge_inference_ms

        # --------------------------------------------------
        # 3. LOW CONFIDENCE -> CHECK RESOURCES
        # --------------------------------------------------

        else:

            resource_start = time.perf_counter()

            decision = make_offloading_decision(
                edge_confidence
            )

            resource_end = time.perf_counter()

            resource_check_ms = (
                resource_end - resource_start
            ) * 1000

            resources = decision["resources"]

            reason = decision["reason"]

            # --------------------------------------------------
            # 4. CLOUD AVAILABLE -> EXECUTE CLOUD MODEL
            # --------------------------------------------------

            if decision["execution"] == "cloud":

                cloud_result = execute_model(
                    sample_index,
                    target,
                    "cloud"
                )

                execution = "cloud"

                final_prediction = cloud_result["prediction"]

                final_confidence = cloud_result["confidence"]

                cloud_inference_ms = (
                    cloud_result["inference_time_ms"]
                )

                # Sensor input payload
                sample = df.loc[sample_index, FEATURES]

                input_bytes = len(
                    sample.to_json().encode("utf-8")
                )

                # Cloud output payload
                output_bytes = len(
                    json.dumps({
                        "prediction": int(final_prediction),
                        "confidence": float(final_confidence)
                    }).encode("utf-8")
                )

                # Dynamic bandwidth measurements
                upload_delay_ms = transfer_time_ms(
                    input_bytes,
                    resources["upload_mbps"]
                )

                download_delay_ms = transfer_time_ms(
                    output_bytes,
                    resources["download_mbps"]
                )

                # Network RTT + transfer delays
                network_delay_ms = (
                    resources["network_latency_ms"]
                    + upload_delay_ms
                    + download_delay_ms
                )

                # Edge inference is already completed
                total_latency_ms = (
                    edge_inference_ms
                    + resource_check_ms
                    + cloud_inference_ms
                    + network_delay_ms
                )

            # --------------------------------------------------
            # 5. CLOUD UNAVAILABLE -> EDGE FALLBACK
            # --------------------------------------------------

            else:

                execution = "edge_fallback"

                final_prediction = edge_prediction
                final_confidence = edge_confidence

                cloud_inference_ms = 0.0
                network_delay_ms = 0.0

                total_latency_ms = (
                    edge_inference_ms
                    + resource_check_ms
                )

        # --------------------------------------------------
        # 6. RECORD TASK RESULTS
        # --------------------------------------------------

        results.append({

            "sample_index": sample_index,
            "target": target,

            "edge_prediction": edge_prediction,
            "edge_confidence": edge_confidence,

            "final_prediction": final_prediction,
            "final_confidence": final_confidence,

            "actual": actual,

            "execution": execution,
            "reason": reason,

            "correct": int(final_prediction == actual),

            "edge_inference_ms": edge_inference_ms,
            "cloud_inference_ms": cloud_inference_ms,

            "resource_check_ms": resource_check_ms,
            "network_delay_ms": network_delay_ms,

            "total_latency_ms": total_latency_ms,

            "cpu_percent": (
                resources["cpu_percent"]
                if resources else None
            ),

            "available_ram_mb": (
                resources["available_ram_mb"]
                if resources else None
            ),

            "upload_mbps": (
                resources["upload_mbps"]
                if resources else None
            ),

            "download_mbps": (
                resources["download_mbps"]
                if resources else None
            ),

            "network_latency_ms": (
                resources["network_latency_ms"]
                if resources else None
            ),

            "cloud_available": (
                resources["cloud_available"]
                if resources else None
            )
        })

    # --------------------------------------------------
    # 7. PROGRESS + CHECKPOINT SAVING
    # --------------------------------------------------

    completed_samples = sample_index + 1

    completed_tasks = completed_samples * len(TARGETS)

    if (
        completed_samples % PROGRESS_INTERVAL == 0
        or completed_samples == num_samples
    ):

        elapsed_seconds = (
            time.perf_counter() - simulation_start
        )

        progress_percent = (
            completed_samples / num_samples * 100
        )

        print(
            f"Progress: {completed_samples}/{num_samples} samples "
            f"({progress_percent:.1f}%) | "
            f"Tasks: {completed_tasks}/{total_expected_tasks} | "
            f"Elapsed: {elapsed_seconds / 60:.2f} min"
        )

        # Save checkpoint to prevent losing completed results
        checkpoint_df = pd.DataFrame(results)

        checkpoint_df.to_csv(
            RESULT_FILE,
            index=False
        )


# ==================================================
# FINAL RESULTS
# ==================================================

results_df = pd.DataFrame(results)

total_tasks = len(results_df)

edge_tasks = (
    results_df["execution"] == "edge"
).sum()

fallback_tasks = (
    results_df["execution"] == "edge_fallback"
).sum()

cloud_tasks = (
    results_df["execution"] == "cloud"
).sum()

low_confidence_tasks = (
    results_df["resource_check_ms"] > 0
).sum()

accuracy = results_df["correct"].mean() * 100

average_latency = results_df["total_latency_ms"].mean()

average_edge_inference = (
    results_df["edge_inference_ms"].mean()
)

average_cloud_inference = (
    results_df["cloud_inference_ms"].mean()
)

# Average resource checking time across ALL tasks
average_resource_check_all_tasks = (
    results_df["resource_check_ms"].mean()
)

# Average resource checking time only for low-confidence tasks
if low_confidence_tasks > 0:

    average_resource_check_low_confidence = (
        results_df["resource_check_ms"].sum()
        / low_confidence_tasks
    )

else:

    average_resource_check_low_confidence = 0.0


total_resource_check_time = (
    results_df["resource_check_ms"].sum()
)

average_network_delay = (
    results_df["network_delay_ms"].mean()
)

total_simulation_time = (
    time.perf_counter() - simulation_start
)


# ==================================================
# SUMMARY
# ==================================================

summary = pd.DataFrame([{

    "total_samples": num_samples,

    "total_tasks": total_tasks,

    "accuracy_percent": accuracy,

    "confidence_threshold": CONFIDENCE_THRESHOLD,

    "low_confidence_tasks": low_confidence_tasks,

    "edge_tasks": edge_tasks,

    "edge_fallback_tasks": fallback_tasks,

    "cloud_tasks": cloud_tasks,

    "cloud_offloading_percent": (
        cloud_tasks / total_tasks * 100
        if total_tasks > 0 else 0
    ),

    "average_latency_ms": average_latency,

    "average_edge_inference_ms": average_edge_inference,

    "average_cloud_inference_ms": average_cloud_inference,

    "average_resource_check_ms_all_tasks":
        average_resource_check_all_tasks,

    "average_resource_check_ms_low_confidence":
        average_resource_check_low_confidence,

    "total_resource_check_time_ms":
        total_resource_check_time,

    "average_network_delay_ms": average_network_delay,

    "total_simulation_time_seconds": total_simulation_time,

    "total_simulation_time_minutes":
        total_simulation_time / 60

}])


# ==================================================
# DISPLAY
# ==================================================

print("\n" + "=" * 70)
print("FINAL PERFORMANCE SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))


print("\nEXECUTION DISTRIBUTION")
print("-" * 50)

print(results_df["execution"].value_counts())


print("\nDECISION REASONS")
print("-" * 50)

print(results_df["reason"].value_counts())


print("\nTARGET-WISE ACCURACY")
print("-" * 50)

target_accuracy = (
    results_df.groupby("target")["correct"]
    .mean() * 100
)

print(target_accuracy)


print("\nTARGET-WISE EXECUTION DISTRIBUTION")
print("-" * 50)

target_execution = pd.crosstab(
    results_df["target"],
    results_df["execution"]
)

print(target_execution)


# ==================================================
# SAVE FINAL FILES
# ==================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


print("\n" + "=" * 70)
print("SAVED RESULTS")
print("=" * 70)

print(f"Detailed results : {RESULT_FILE}")
print(f"Summary results  : {SUMMARY_FILE}")

print(f"Total tasks     : {total_tasks}")
print(f"Total time      : {total_simulation_time / 60:.2f} minutes")

print("\nSimulation completed successfully.")
