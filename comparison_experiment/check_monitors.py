import os
import sys

# Add the main EDGE_AI project directory to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from system_monitor import get_all_resources
from network_monitor import measure_network

print("\n--- SYSTEM RESOURCE OUTPUT ---")

resources = get_all_resources(test_internet=False)
print("\nReturned dictionary:")
print(resources)

print("\n--- NETWORK OUTPUT ---")

network = measure_network()
print("\nReturned dictionary:")
print(network)