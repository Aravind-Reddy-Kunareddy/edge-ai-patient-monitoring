import psutil
import subprocess
import re
import time
import urllib.request

# ==================================================
# CONFIGURATION
# ==================================================

CONFIDENCE_THRESHOLD = 0.90

MIN_UPLOAD_MBPS = 3.0
MIN_DOWNLOAD_MBPS = 2.0

MAX_NETWORK_LATENCY_MS = 100.0

MAX_CPU_PERCENT = 85.0
MIN_AVAILABLE_RAM_MB = 500.0

# Lightweight measurement payloads
DOWNLOAD_TEST_BYTES = 1_000_000   # 1 MB
UPLOAD_TEST_BYTES = 512_000      # 512 KB


# ==================================================
# MEASURE DOWNLOAD BANDWIDTH
# ==================================================

def measure_download_bandwidth():

    url = (
        "https://speed.cloudflare.com/__down"
        f"?bytes={DOWNLOAD_TEST_BYTES}"
    )

    try:

        start = time.perf_counter()

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "EdgeAI-ResourceMonitor"}
        )

        with urllib.request.urlopen(
            request,
            timeout=10
        ) as response:

            data = response.read()

        end = time.perf_counter()

        elapsed = end - start

        if elapsed <= 0 or len(data) == 0:
            return None

        download_mbps = (
            len(data) * 8 / elapsed / 1_000_000
        )

        return download_mbps

    except Exception as e:

        print(f"Download measurement failed: {e}")

        return None


# ==================================================
# MEASURE UPLOAD BANDWIDTH
# ==================================================

def measure_upload_bandwidth():

    url = "https://speed.cloudflare.com/__up"

    payload = b"0" * UPLOAD_TEST_BYTES

    try:

        start = time.perf_counter()

        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/octet-stream",
                "User-Agent": "EdgeAI-ResourceMonitor"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=10
        ) as response:

            response.read()

        end = time.perf_counter()

        elapsed = end - start

        if elapsed <= 0:
            return None

        upload_mbps = (
            len(payload) * 8 / elapsed / 1_000_000
        )

        return upload_mbps

    except Exception as e:

        print(f"Upload measurement failed: {e}")

        return None


# ==================================================
# MEASURE NETWORK LATENCY
# ==================================================

def measure_network_latency():

    try:

        result = subprocess.run(
            ["ping", "-n", "1", "-w", "2000", "1.1.1.1"],
            capture_output=True,
            text=True,
            timeout=4
        )

        match = re.search(
            r"time[=<]\s*(\d+)\s*ms",
            result.stdout
        )

        if match:
            return float(match.group(1))

        return None

    except Exception:

        return None


# ==================================================
# MEASURE ALL RESOURCES
# ==================================================

def check_cloud_resources():

    print("\nMeasuring current resources...")

    measurement_start = time.perf_counter()

    # 1. CPU utilization
    cpu_percent = psutil.cpu_percent(interval=0.2)

    # 2. Available RAM
    memory = psutil.virtual_memory()

    available_ram_mb = memory.available / (1024 * 1024)

    # 3. Dynamic upload bandwidth
    upload_mbps = measure_upload_bandwidth()

    # 4. Dynamic download bandwidth
    download_mbps = measure_download_bandwidth()

    # 5. Dynamic network latency
    latency_ms = measure_network_latency()

    measurement_end = time.perf_counter()

    measurement_time_ms = (
        measurement_end - measurement_start
    ) * 1000

    # --------------------------------------------------
    # RESOURCE AVAILABILITY
    # --------------------------------------------------

    cpu_available = cpu_percent <= MAX_CPU_PERCENT

    ram_available = available_ram_mb >= MIN_AVAILABLE_RAM_MB

    upload_available = (
        upload_mbps is not None
        and upload_mbps >= MIN_UPLOAD_MBPS
    )

    download_available = (
        download_mbps is not None
        and download_mbps >= MIN_DOWNLOAD_MBPS
    )

    latency_available = (
        latency_ms is not None
        and latency_ms <= MAX_NETWORK_LATENCY_MS
    )

    # --------------------------------------------------
    # CLOUD AVAILABILITY
    # --------------------------------------------------

    cloud_available = all([
        cpu_available,
        ram_available,
        upload_available,
        download_available,
        latency_available
    ])

    # --------------------------------------------------
    # RETURN MEASUREMENTS
    # --------------------------------------------------

    resources = {

        "cpu_percent": cpu_percent,

        "available_ram_mb": available_ram_mb,

        "upload_mbps": upload_mbps,

        "download_mbps": download_mbps,

        "network_latency_ms": latency_ms,

        "measurement_time_ms": measurement_time_ms,

        "cpu_available": cpu_available,

        "ram_available": ram_available,

        "upload_available": upload_available,

        "download_available": download_available,

        "latency_available": latency_available,

        "cloud_available": cloud_available
    }

    return resources


# ==================================================
# CONFIDENCE-BASED OFFLOADING
# ==================================================

def make_offloading_decision(edge_confidence):

    # HIGH CONFIDENCE -> EDGE
    if edge_confidence >= CONFIDENCE_THRESHOLD:

        return {
            "execution": "edge",
            "reason": "High Edge confidence",
            "resources_checked": False,
            "resources": None
        }

    # LOW CONFIDENCE -> MEASURE CURRENT RESOURCES
    resources = check_cloud_resources()

    # CLOUD AVAILABLE
    if resources["cloud_available"]:

        return {
            "execution": "cloud",
            "reason": "Low Edge confidence and Cloud resources available",
            "resources_checked": True,
            "resources": resources
        }

    # CLOUD UNAVAILABLE -> EDGE FALLBACK
    return {
        "execution": "edge",
        "reason": "Low Edge confidence but Cloud resources unavailable",
        "resources_checked": True,
        "resources": resources
    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    test_confidence = 0.45

    decision = make_offloading_decision(test_confidence)

    print("\nOFFLOADING POLICY TEST")
    print("=" * 50)

    print("Edge Confidence:", test_confidence)
    print("Execution:", decision["execution"])
    print("Reason:", decision["reason"])

    if decision["resources_checked"]:

        print("\nRESOURCE MEASUREMENTS")
        print("-" * 50)

        for key, value in decision["resources"].items():

            print(f"{key}: {value}")