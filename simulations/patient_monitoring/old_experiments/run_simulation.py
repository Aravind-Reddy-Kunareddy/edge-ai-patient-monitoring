
import pandas as pd
import psutil
import subprocess
import re
import time
import json

from resource_aware_policy import (
    CONFIDENCE_THRESHOLD,
    make_offloading_decision
)

from load_project import df, FEATURES, TARGETS
from model_executor import execute_model


# =========================================================
# CONFIGURATION: REAL MEASURED NETWORK BASELINE
# =========================================================

NUM_SAMPLES = 100

# Your actual speed-test measurements
UPLOAD_MBPS = 31.52
DOWNLOAD_MBPS = 10.89

# Measured ping to Cloudflare
CLOUD_LATENCY_MS = 20.80

# Previous measured local-network delay
EDGE_LATENCY_FALLBACK_MS = 5.0

# Refresh ping every 20 prediction tasks
NETWORK_REFRESH_INTERVAL = 20

# Fixed adaptive policy
EDGE_CAPACITY = 12
TASK_WINDOW_SIZE = 20

# Memory pressure threshold
LOW_RAM_THRESHOLD_MB = 1024

# Cloud inference model is currently executed locally
# This is a simulation limitation, not a remote cloud measurement.


# =========================================================
# NETWORK FUNCTIONS
# =========================================================

def get_gateway():

    try:

        output = subprocess.run(
            ["ipconfig"],
            capture_output=True,
            text=True,
            timeout=5
        ).stdout

        matches = re.findall(
            r"Default Gateway[^:]*:\s*(\d+\.\d+\.\d+\.\d+)",
            output
        )

        for gateway in matches:

            if gateway != "0.0.0.0":
                return gateway

    except Exception:
        pass

    return None


def ping_ms(host):

    if not host:
        return None

    try:

        output = subprocess.run(
            ["ping", "-n", "3", "-w", "1000", host],
            capture_output=True,
            text=True,
            timeout=5
        )

        matches = re.findall(
            r"time[=<]\s*([\d.]+)\s*ms",
            output.stdout,
            re.IGNORECASE
        )

        values = [float(x) for x in matches]

        if values:
            return sum(values) / len(values)

    except Exception:
        pass

    return None


# =========================================================
# HARDWARE MONITORING
# =========================================================

def get_resources():

    cpu = psutil.cpu_percent(interval=0.1)

    memory = psutil.virtual_memory()

    return {
        "cpu_percent": cpu,
        "memory_percent": memory.percent,
        "available_ram_mb": memory.available / (1024 * 1024),
        "used_ram_mb": memory.used / (1024 * 1024)
    }


# =========================================================
# DATA TRANSFER SIZE
# =========================================================

def calculate_transfer_sizes(sample_index):

    # Convert sensor sample to JSON representation
    sample = df.loc[sample_index, FEATURES]

    input_json = sample.to_json()

    input_bytes = len(input_json.encode("utf-8"))

    # Small prediction result sent back to edge
    output_bytes = len(
        json.dumps({
            "prediction": 1,
            "target": "sensor_prediction"
        }).encode("utf-8")
    )

    return input_bytes, output_bytes


# =========================================================
# BANDWIDTH-BASED TRANSFER LATENCY
# =========================================================

def transfer_time_ms(size_bytes, bandwidth_mbps):

    if bandwidth_mbps <= 0:
        return float("inf")

    return (
        size_bytes * 8
        / (bandwidth_mbps * 1000)
    )


# =========================================================
# RESOURCE-AWARE POLICY
# =========================================================

def resource_aware_policy(

    edge_inference_ms,
    cloud_inference_ms,

    cpu_percent,
    available_ram_mb,

    upload_mbps,
    download_mbps,

    cloud_rtt_ms,

    input_bytes,
    output_bytes
):

    # -----------------------------------------
    # EDGE ESTIMATED LATENCY
    # -----------------------------------------

    # CPU pressure penalty
    cpu_penalty = (
        edge_inference_ms
        * cpu_percent / 100
    )

    # Memory pressure penalty
    if available_ram_mb < LOW_RAM_THRESHOLD_MB:

        memory_penalty = (
            edge_inference_ms
            * (LOW_RAM_THRESHOLD_MB - available_ram_mb)
            / LOW_RAM_THRESHOLD_MB
        )

    else:

        memory_penalty = 0


    edge_estimated_latency = (
        edge_inference_ms
        + cpu_penalty
        + memory_penalty
    )


    # -----------------------------------------
    # CLOUD ESTIMATED LATENCY
    # -----------------------------------------

    # Sensor upload delay
    upload_delay = transfer_time_ms(
        input_bytes,
        upload_mbps
    )

    # Prediction result download delay
    download_delay = transfer_time_ms(
        output_bytes,
        download_mbps
    )

    cloud_estimated_latency = (
        cloud_inference_ms
        + cloud_rtt_ms
        + upload_delay
        + download_delay
    )


    # -----------------------------------------
    # DECISION
    # -----------------------------------------

    if edge_estimated_latency <= cloud_estimated_latency:

        selected = "edge"

    else:

        selected = "cloud"


    return {

        "execution": selected,

        "estimated_edge_latency_ms":
            edge_estimated_latency,

        "estimated_cloud_latency_ms":
            cloud_estimated_latency,

        "cpu_penalty_ms":
            cpu_penalty,

        "memory_penalty_ms":
            memory_penalty,

        "upload_transfer_ms":
            upload_delay,

        "download_transfer_ms":
            download_delay
    }


# =========================================================
# INITIALIZATION
# =========================================================

gateway = get_gateway()

edge_network_ms = EDGE_LATENCY_FALLBACK_MS
cloud_network_ms = CLOUD_LATENCY_MS

results = []

fixed_edge_load = 0
fixed_task_position = 0

resource_edge_count = 0
resource_cloud_count = 0

total_tasks = NUM_SAMPLES * len(TARGETS)

print("=" * 70)
print("FIVE-RESOURCE ADAPTIVE EDGE-CLOUD SIMULATION")
print("=" * 70)

print(f"CPU: Dynamically measured")
print(f"RAM: Dynamically measured")
print(f"Upload: {UPLOAD_MBPS} Mbps (measured)")
print(f"Download: {DOWNLOAD_MBPS} Mbps (measured)")
print(f"Cloud RTT baseline: {CLOUD_LATENCY_MS} ms (measured)")
print(f"Gateway: {gateway}")


# =========================================================
# MAIN SIMULATION
# =========================================================

for sample_index in range(NUM_SAMPLES):

    for target in TARGETS:

        task_position = (
            sample_index * len(TARGETS)
            + TARGETS.index(target)
        )


        # -----------------------------------------
        # REFRESH NETWORK LATENCY
        # -----------------------------------------

        if task_position % NETWORK_REFRESH_INTERVAL == 0:

            measured_edge = ping_ms(gateway)

            measured_cloud = ping_ms("1.1.1.1")

            if measured_edge is not None:
                edge_network_ms = measured_edge

            else:
                edge_network_ms = EDGE_LATENCY_FALLBACK_MS

            if measured_cloud is not None:
                cloud_network_ms = measured_cloud

            else:
                cloud_network_ms = CLOUD_LATENCY_MS


        # -----------------------------------------
        # MEASURE ACTUAL RESOURCES
        # -----------------------------------------

        resources = get_resources()

        cpu = resources["cpu_percent"]

        available_ram = resources["available_ram_mb"]

        memory_percent = resources["memory_percent"]


        # -----------------------------------------
        # EXECUTE BOTH MODELS
        # -----------------------------------------

        edge_result = execute_model(
            sample_index,
            target,
            "edge"
        )

        cloud_result = execute_model(
            sample_index,
            target,
            "cloud"
        )


        edge_inference = edge_result["inference_time_ms"]

        cloud_inference = cloud_result["inference_time_ms"]


        # -----------------------------------------
        # INPUT / OUTPUT SIZE
        # -----------------------------------------

        input_bytes, output_bytes = calculate_transfer_sizes(
            sample_index
        )


        # -----------------------------------------
        # EDGE ONLY
        # -----------------------------------------

        edge_row = dict(edge_result)

        edge_row["strategy"] = "Edge Only"

        edge_row["execution"] = "edge"

        edge_row["network_delay_ms"] = edge_network_ms

        edge_row["total_latency_ms"] = (
            edge_inference
            + edge_network_ms
        )

        edge_row.update(resources)

        edge_row["upload_mbps"] = UPLOAD_MBPS

        edge_row["download_mbps"] = DOWNLOAD_MBPS

        edge_row["cloud_rtt_ms"] = cloud_network_ms

        edge_row["input_bytes"] = input_bytes

        edge_row["output_bytes"] = output_bytes

        results.append(edge_row)


        # -----------------------------------------
        # CLOUD ONLY
        # -----------------------------------------

        cloud_upload_delay = transfer_time_ms(
            input_bytes,
            UPLOAD_MBPS
        )

        cloud_download_delay = transfer_time_ms(
            output_bytes,
            DOWNLOAD_MBPS
        )

        cloud_total_network_delay = (
            cloud_network_ms
            + cloud_upload_delay
            + cloud_download_delay
        )

        cloud_row = dict(cloud_result)

        cloud_row["strategy"] = "Cloud Only"

        cloud_row["execution"] = "cloud"

        cloud_row["network_delay_ms"] = (
            cloud_total_network_delay
        )

        cloud_row["total_latency_ms"] = (
            cloud_inference
            + cloud_total_network_delay
        )

        cloud_row.update(resources)

        cloud_row["upload_mbps"] = UPLOAD_MBPS

        cloud_row["download_mbps"] = DOWNLOAD_MBPS

        cloud_row["cloud_rtt_ms"] = cloud_network_ms

        cloud_row["input_bytes"] = input_bytes

        cloud_row["output_bytes"] = output_bytes

        results.append(cloud_row)


        # -----------------------------------------
        # FIXED ADAPTIVE
        # -----------------------------------------

        if fixed_task_position % TASK_WINDOW_SIZE == 0:

            fixed_edge_load = 0


        if fixed_edge_load < EDGE_CAPACITY:

            fixed_execution = "edge"

            fixed_result = edge_result

            fixed_network = edge_network_ms

            fixed_edge_load += 1

        else:

            fixed_execution = "cloud"

            fixed_result = cloud_result

            fixed_network = cloud_total_network_delay


        fixed_row = dict(fixed_result)

        fixed_row["strategy"] = "Fixed Adaptive"

        fixed_row["execution"] = fixed_execution

        fixed_row["network_delay_ms"] = fixed_network

        fixed_row["total_latency_ms"] = (
            fixed_result["inference_time_ms"]
            + fixed_network
        )

        fixed_row.update(resources)

        fixed_row["upload_mbps"] = UPLOAD_MBPS

        fixed_row["download_mbps"] = DOWNLOAD_MBPS

        fixed_row["cloud_rtt_ms"] = cloud_network_ms

        fixed_row["input_bytes"] = input_bytes

        fixed_row["output_bytes"] = output_bytes

        results.append(fixed_row)

        fixed_task_position += 1


        # -----------------------------------------
        # RESOURCE-AWARE ADAPTIVE
        # -----------------------------------------

        decision = resource_aware_policy(

            edge_inference_ms=edge_inference,

            cloud_inference_ms=cloud_inference,

            cpu_percent=cpu,

            available_ram_mb=available_ram,

            upload_mbps=UPLOAD_MBPS,

            download_mbps=DOWNLOAD_MBPS,

            cloud_rtt_ms=cloud_network_ms,

            input_bytes=input_bytes,

            output_bytes=output_bytes
        )


        selected = decision["execution"]


        if selected == "edge":

            selected_result = edge_result

            selected_network = edge_network_ms

            resource_edge_count += 1

        else:

            selected_result = cloud_result

            selected_network = cloud_total_network_delay

            resource_cloud_count += 1


        resource_row = dict(selected_result)

        resource_row["strategy"] = "Resource-Aware"

        resource_row["execution"] = selected

        resource_row["network_delay_ms"] = selected_network

        resource_row["total_latency_ms"] = (
            selected_result["inference_time_ms"]
            + selected_network
        )

        resource_row.update(resources)

        resource_row["upload_mbps"] = UPLOAD_MBPS

        resource_row["download_mbps"] = DOWNLOAD_MBPS

        resource_row["cloud_rtt_ms"] = cloud_network_ms

        resource_row["input_bytes"] = input_bytes

        resource_row["output_bytes"] = output_bytes

        resource_row["estimated_edge_latency_ms"] = (
            decision["estimated_edge_latency_ms"]
        )

        resource_row["estimated_cloud_latency_ms"] = (
            decision["estimated_cloud_latency_ms"]
        )

        resource_row["cpu_penalty_ms"] = (
            decision["cpu_penalty_ms"]
        )

        resource_row["memory_penalty_ms"] = (
            decision["memory_penalty_ms"]
        )

        resource_row["upload_transfer_ms"] = (
            decision["upload_transfer_ms"]
        )

        resource_row["download_transfer_ms"] = (
            decision["download_transfer_ms"]
        )

        results.append(resource_row)


# =========================================================
# SAVE DETAILED RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "simulation_results_all_resources.csv",
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

summary = results_df.groupby("strategy").agg(

    total_tasks=("prediction", "count"),

    average_latency_ms=("total_latency_ms", "mean"),

    average_inference_ms=("inference_time_ms", "mean"),

    average_network_delay_ms=("network_delay_ms", "mean"),

    average_cpu_percent=("cpu_percent", "mean"),

    average_memory_percent=("memory_percent", "mean"),

    average_available_ram_mb=("available_ram_mb", "mean"),

    average_upload_mbps=("upload_mbps", "mean"),

    average_download_mbps=("download_mbps", "mean")
)


summary["accuracy"] = 0.0


for strategy in summary.index:

    subset = results_df[
        results_df["strategy"] == strategy
    ]

    summary.loc[strategy, "accuracy"] = (
        subset["prediction"] == subset["actual"]
    ).mean()


summary = summary[
    [
        "total_tasks",
        "accuracy",
        "average_latency_ms",
        "average_inference_ms",
        "average_network_delay_ms",
        "average_cpu_percent",
        "average_memory_percent",
        "average_available_ram_mb",
        "average_upload_mbps",
        "average_download_mbps"
    ]
]


summary.to_csv(
    "simulation_summary_all_resources.csv"
)


# =========================================================
# DISPLAY
# =========================================================

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

print(summary.round(4).to_string())


print("\nFIXED ADAPTIVE DISTRIBUTION")

print(
    results_df[
        results_df["strategy"] == "Fixed Adaptive"
    ]["execution"].value_counts()
)


print("\nRESOURCE-AWARE DISTRIBUTION")

print(
    results_df[
        results_df["strategy"] == "Resource-Aware"
    ]["execution"].value_counts()
)


print("\nResource-Aware Decisions")

print("Edge:", resource_edge_count)

print("Cloud:", resource_cloud_count)


print("\nTotal Samples:", NUM_SAMPLES)

print("Total Tasks:", len(results_df))


print("\nSaved:")

print("simulation_results_all_resources.csv")

print("simulation_summary_all_resources.csv")


print("\nSimulation completed.")
