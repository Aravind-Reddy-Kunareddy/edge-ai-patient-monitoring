
import pandas as pd

from load_project import df, FEATURES, TARGETS

# Select first 1000 patient sensor records
workload = df.head(1000).copy()

# Create simulation task IDs
workload.insert(
    0,
    "task_id",
    range(1, len(workload) + 1)
)

# Assign patient monitoring task type
workload["task_type"] = "patient_monitoring"

# Add the input and target information
columns = (
    ["task_id", "patient_id", "room_number", "task_type"]
    + FEATURES
    + TARGETS
)

workload = workload[columns]

# Save workload
workload.to_csv("patient_workload.csv", index=False)

print("Patient workload created successfully!")
print("Total tasks:", len(workload))
print("Sensor features:", len(FEATURES))
print("Prediction targets:", len(TARGETS))
print("\nFirst five tasks:")
print(workload.head().to_string(index=False))
