
import psutil
import time


def get_resource_snapshot(sample_interval=1):

    # 1. CPU utilization
    cpu_usage = psutil.cpu_percent(interval=sample_interval)

    # 2. RAM information
    memory = psutil.virtual_memory()

    total_ram_mb = memory.total / (1024 ** 2)
    available_ram_mb = memory.available / (1024 ** 2)
    ram_usage_percent = memory.percent

    # 3. Network traffic
    # These counters track total bytes sent and received.
    network_before = psutil.net_io_counters()

    time.sleep(sample_interval)

    network_after = psutil.net_io_counters()

    bytes_sent = network_after.bytes_sent - network_before.bytes_sent
    bytes_received = network_after.bytes_recv - network_before.bytes_recv

    # Convert bytes per second into Mbps
    upload_mbps = (bytes_sent * 8) / (sample_interval * 1_000_000)
    download_mbps = (bytes_received * 8) / (sample_interval * 1_000_000)

    return {
        "cpu_usage_percent": cpu_usage,
        "total_ram_mb": round(total_ram_mb, 2),
        "available_ram_mb": round(available_ram_mb, 2),
        "ram_usage_percent": ram_usage_percent,
        "upload_mbps": round(upload_mbps, 4),
        "download_mbps": round(download_mbps, 4)
    }


if __name__ == "__main__":

    print("\nREAL-TIME RESOURCE MONITOR")
    print("-" * 40)

    resources = get_resource_snapshot()

    for resource, value in resources.items():
        print(f"{resource}: {value}")

    print("-" * 40)
