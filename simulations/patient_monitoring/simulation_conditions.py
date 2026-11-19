
import random


# Reproducible simulation
random.seed(42)


def generate_conditions(target):

    # Simulated Edge resource utilization
    edge_cpu_percent = random.randint(20, 95)

    edge_memory_percent = random.randint(30, 90)

    # Simulated network conditions
    edge_network_delay_ms = random.uniform(2, 25)

    cloud_network_delay_ms = random.uniform(10, 60)

    # Assign urgency based on task type
    if target in ["fall_detected", "hypoxemia_event"]:
        urgency = "high"
    else:
        urgency = "normal"

    return {
        "edge_cpu_percent": edge_cpu_percent,
        "edge_memory_percent": edge_memory_percent,
        "edge_network_delay_ms": round(edge_network_delay_ms, 2),
        "cloud_network_delay_ms": round(cloud_network_delay_ms, 2),
        "urgency": urgency
    }


# Test
if __name__ == "__main__":

    for target in [
        "fall_detected",
        "abnormal_heart_rhythm",
        "hypoxemia_event",
        "fever_event"
    ]:

        conditions = generate_conditions(target)

        print(f"\nTask: {target}")
        print(conditions)
