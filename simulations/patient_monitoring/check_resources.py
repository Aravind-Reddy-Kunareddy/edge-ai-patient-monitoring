
import psutil
import subprocess
import re
import statistics
import time

try:
    import speedtest
except ImportError:
    speedtest = None


# ==========================================
# CPU
# ==========================================

print("=" * 55)
print("REAL SYSTEM RESOURCE MEASUREMENTS")
print("=" * 55)

cpu = psutil.cpu_percent(interval=2)

print(f"\nCPU Utilization: {cpu:.2f}%")


# ==========================================
# RAM
# ==========================================

ram = psutil.virtual_memory()

print(f"RAM Total: {ram.total / (1024**3):.2f} GB")
print(f"RAM Used: {ram.used / (1024**3):.2f} GB")
print(f"RAM Available: {ram.available / (1024**3):.2f} GB")
print(f"RAM Utilization: {ram.percent:.2f}%")


# ==========================================
# NETWORK LATENCY
# ==========================================

def measure_ping(host, count=5):

    try:

        output = subprocess.run(
            ["ping", "-n", str(count), host],
            capture_output=True,
            text=True,
            timeout=15
        )

        times = re.findall(
            r"time[=<]\s*([\d.]+)\s*ms",
            output.stdout,
            re.IGNORECASE
        )

        times = [float(x) for x in times]

        if times:

            return {
                "average": statistics.mean(times),
                "minimum": min(times),
                "maximum": max(times),
                "samples": len(times)
            }

    except Exception as e:

        print("Ping error:", e)

    return None


print("\n--- Network Latency ---")

for host, label in [
    ("1.1.1.1", "Cloudflare"),
    ("8.8.8.8", "Google DNS")
]:

    result = measure_ping(host)

    if result:

        print(f"\n{label} ({host})")
        print(f"Average: {result['average']:.2f} ms")
        print(f"Minimum: {result['minimum']:.2f} ms")
        print(f"Maximum: {result['maximum']:.2f} ms")
        print(f"Successful replies: {result['samples']}/5")

    else:

        print(f"\n{label}: Ping unavailable")


# ==========================================
# UPLOAD / DOWNLOAD BANDWIDTH
# ==========================================

print("\n--- Internet Bandwidth ---")

if speedtest is None:

    print("speedtest module not installed.")
    print("Install using: pip install speedtest-cli")

else:

    try:

        st = speedtest.Speedtest()

        print("Finding best speed test server...")
        st.get_best_server()

        print("Measuring download speed...")
        download_bps = st.download()

        print("Measuring upload speed...")
        upload_bps = st.upload()

        download_mbps = download_bps / 1_000_000
        upload_mbps = upload_bps / 1_000_000

        print(f"\nDownload: {download_mbps:.2f} Mbps")
        print(f"Upload: {upload_mbps:.2f} Mbps")

        print(
            "Ping to speed-test server:",
            f"{st.results.ping:.2f} ms"
        )

    except Exception as e:

        print("\nSpeed test failed:", e)


# ==========================================
# SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("MEASUREMENT COMPLETE")
print("=" * 55)
