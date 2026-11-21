import speedtest
import subprocess
import re


def measure_network():

    print("\nMEASURING NETWORK RESOURCES")
    print("-" * 40)

    st = speedtest.Speedtest(timeout=20)

    # Find the best available test server
    st.get_best_server()

    # Measure download speed
    download_mbps = st.download() / 1_000_000

    # Measure upload speed
    upload_mbps = st.upload() / 1_000_000

    # Measure latency using ping
    ping_result = subprocess.run(
        ["ping", "-n", "4", "1.1.1.1"],
        capture_output=True,
        text=True,
        timeout=10
    )

    times = re.findall(
        r"time[=<]\s*([\d.]+)\s*ms",
        ping_result.stdout,
        re.IGNORECASE
    )

    if times:
        latency_ms = sum(map(float, times)) / len(times)
    else:
        latency_ms = None

    results = {
        "download_mbps": round(download_mbps, 2),
        "upload_mbps": round(upload_mbps, 2),
        "latency_ms": (
            round(latency_ms, 2)
            if latency_ms is not None
            else None
        )
    }

    print("\nNETWORK RESULTS")

    for name, value in results.items():
        print(f"{name}: {value}")

    return results


if __name__ == "__main__":
    measure_network()