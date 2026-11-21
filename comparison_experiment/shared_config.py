# shared_config.py

import os
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data.csv"),
    os.path.join(PROJECT_ROOT, "data", "data.csv"),
    os.path.join(PROJECT_ROOT, "dataset", "data.csv"),
]

DATA_PATH = next(
    (path for path in DATA_CANDIDATES if os.path.exists(path)),
    None
)

if DATA_PATH is None:
    raise FileNotFoundError(
        "data.csv not found. Check the dataset location."
    )

COMPARISON_DIR = os.path.dirname(os.path.abspath(__file__))

SPLIT_DIR = os.path.join(COMPARISON_DIR, "shared_split")


# ==================================================
# COMMON TARGETS
# ==================================================

TARGETS = [
    "fall_detected",
    "abnormal_heart_rhythm",
    "hypoxemia_event",
    "fever_event"
]


# ==================================================
# COMMON OFFLOADING THRESHOLDS
# ==================================================

CONFIDENCE_THRESHOLD = 0.90

DEADLINE_MS = 5

MAX_CPU_USAGE_PERCENT = 90

MIN_AVAILABLE_RAM_MB = 750


# ==================================================
# COMMON DATA SPLIT
# ==================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ==================================================
# CREATE SHARED TEST SPLIT
# ==================================================

def create_shared_split():

    df = pd.read_csv(DATA_PATH)

    groups = df["patient_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=groups)
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    os.makedirs(SPLIT_DIR, exist_ok=True)

    train_file = os.path.join(
        SPLIT_DIR, "train_indices.csv"
    )

    test_file = os.path.join(
        SPLIT_DIR, "test_indices.csv"
    )

    metadata_file = os.path.join(
        SPLIT_DIR, "split_metadata.json"
    )

    # Protect previously created split files
    if any(os.path.exists(path) for path in [
        train_file, test_file, metadata_file
    ]):
        raise FileExistsError(
            "Shared split already exists. "
            "Existing files were not overwritten."
        )

    pd.DataFrame({
        "sample_index": train_idx
    }).to_csv(train_file, index=False)

    pd.DataFrame({
        "sample_index": test_idx
    }).to_csv(test_file, index=False)

    metadata = {
        "dataset_path": DATA_PATH,
        "total_samples": len(df),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "patient_wise_split": True,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "deadline_ms": DEADLINE_MS,
        "max_cpu_usage_percent": MAX_CPU_USAGE_PERCENT,
        "min_available_ram_mb": MIN_AVAILABLE_RAM_MB,
        "targets": TARGETS
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print("=" * 50)
    print("SHARED DATA SPLIT CREATED")
    print("=" * 50)

    print(f"Dataset       : {DATA_PATH}")
    print(f"Total samples : {len(df)}")
    print(f"Train samples : {len(train_idx)}")
    print(f"Test samples  : {len(test_idx)}")
    print(f"Train patients: {train_df['patient_id'].nunique()}")
    print(f"Test patients : {test_df['patient_id'].nunique()}")

    print("\nThresholds:")
    print(f"Confidence    : {CONFIDENCE_THRESHOLD}")
    print(f"Deadline      : {DEADLINE_MS} ms")
    print(f"Maximum CPU   : {MAX_CPU_USAGE_PERCENT}%")
    print(f"Minimum RAM   : {MIN_AVAILABLE_RAM_MB} MB")

    print("\nSaved files:")
    print(train_file)
    print(test_file)
    print(metadata_file)


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    create_shared_split()