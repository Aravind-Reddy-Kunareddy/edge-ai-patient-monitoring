
from pathlib import Path
import pandas as pd
import joblib

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load dataset
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
df = pd.read_csv(DATA_PATH)

# Load trained models
edge_models = joblib.load(
    PROJECT_ROOT / "models" / "edge_models.pkl"
)

cloud_models = joblib.load(
    PROJECT_ROOT / "models" / "cloud_models.pkl"
)

# Load scalers
edge_scaler = joblib.load(
    PROJECT_ROOT / "models" / "scaler.pkl"
)

cloud_scaler = joblib.load(
    PROJECT_ROOT / "models" / "cloud_scaler.pkl"
)

# Sensor features
FEATURES = [
    "heart_rate_bpm",
    "spo2_percent",
    "ppg_red",
    "ppg_ir",
    "pulse_interval_ms",
    "ecg_signal_mv",
    "rr_interval_ms",
    "hrv_sdnn_ms",
    "hrv_rmssd_ms",
    "qrs_duration_ms",
    "acc_x_g",
    "acc_y_g",
    "acc_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "acc_magnitude_g",
    "peak_acceleration_g",
    "body_temp_c",
    "temp_variation_c",
    "resp_rate_bpm",
    "resp_signal",
    "breathing_interval_s"
]

TARGETS = [
    "fall_detected",
    "abnormal_heart_rhythm",
    "hypoxemia_event",
    "fever_event"
]

print("Dataset loaded:", df.shape)
print("Sensor features:", len(FEATURES))
print("Prediction targets:", TARGETS)
print("Edge models:", list(edge_models.keys()))
print("Cloud models:", list(cloud_models.keys()))
print("All project components loaded successfully!")
