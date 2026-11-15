
import speedtest
import time


def measure_network():

    print("\nMEASURING NETWORK RESOURCES")
    print("-" * 40)

    st = speedtest.Speedtest()

    # Find the best available test server
    st.get_best_server()

    # Measure download speed
    download_mbps = st.download() / 1_000_000

    # Measure upload speed
    upload_mbps = st.upload() / 1_000_000

    # Measure latency to the selected test server
    latency_ms = st.results.ping

    results = {
        "download_mbps": round(download_mbps, 2),
        "upload_mbps": round(upload_mbps, 2),
        "latency_ms": round(latency_ms, 2)
    }

    print("\nNETWORK RESULTS")

    for name, value in results.items():
        print(f"{name}: {value}")

    return results


if __name__ == "__main__":
    measure_network()
