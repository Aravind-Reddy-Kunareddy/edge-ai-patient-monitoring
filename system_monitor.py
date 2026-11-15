
from resource_monitor import get_resource_snapshot
from network_monitor import measure_network


def get_all_resources(test_internet=True):

    print("\nCOLLECTING SYSTEM RESOURCES")
    print("=" * 45)

    # Actual CPU, RAM and network traffic
    system = get_resource_snapshot()

    resources = dict(system)

    # Run active internet test only when requested
    if test_internet:
        network = measure_network()
        resources.update(network)

    print("\nRESOURCE REPORT")
    print("=" * 45)

    for name, value in resources.items():
        print(f"{name}: {value}")

    return resources


if __name__ == "__main__":
    get_all_resources()
