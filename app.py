import os
import time
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from system_monitor import get_all_resources
from network_monitor import measure_network


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Edge AI Monitoring",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# CONSTANTS
# ============================================================

TARGETS = [
    "fall_detected",
    "abnormal_heart_rhythm",
    "hypoxemia_event",
    "fever_event"
]

FEATURE_COLUMNS = [
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

CONFIDENCE_THRESHOLD = 0.95
HYBRID_BANDWIDTH_MBPS = 5
HYBRID_NETWORK_LATENCY_MS = 10
DEADLINE_MS = 5
MAX_CPU_USAGE_PERCENT = 85
MIN_AVAILABLE_RAM_MB = 1000

RESULTS_DIR_CANDIDATES = ["results", "Results"]


# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def find_results_file(base_name):
    """Find a result file regardless of results-folder case or extension."""
    extensions = [".csv", ".xls", ".xlsx"]

    for folder in RESULTS_DIR_CANDIDATES:
        for extension in extensions:
            path = os.path.join(folder, base_name + extension)
            if os.path.exists(path):
                return path

    return None


def read_result_file(base_name):
    """
    The supplied result files have CSV text inside files named .xls.
    Try CSV first, then Excel as a fallback.
    """
    path = find_results_file(base_name)

    if path is None:
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_excel(path)
        except Exception as error:
            st.warning(f"Could not read {os.path.basename(path)}: {error}")
            return None


def get_value(dictionary, possible_keys, default=None):
    """Read a value from a dictionary using possible key names."""
    if not isinstance(dictionary, dict):
        return default

    for key in possible_keys:
        if key in dictionary:
            return dictionary[key]

    return default


def calculate_network_delay(
    request_size_bytes,
    response_size_bytes,
    upload_mbps,
    download_mbps,
    network_latency_ms
):
    if upload_mbps is None or download_mbps is None:
        return float("inf")

    if upload_mbps <= 0 or download_mbps <= 0:
        return float("inf")

    upload_time = (
        request_size_bytes * 8
        / (upload_mbps * 1_000_000)
        * 1000
    )

    download_time = (
        response_size_bytes * 8
        / (download_mbps * 1_000_000)
        * 1000
    )

    return network_latency_ms + upload_time + download_time


def get_model_prediction(model, scaler, input_data):
    scaled_data = scaler.transform(input_data)

    start_time = time.perf_counter()
    prediction = model.predict(scaled_data)
    inference_latency = (time.perf_counter() - start_time) * 1000

    probabilities = model.predict_proba(scaled_data)
    confidence = float(np.max(probabilities[0]))

    return int(prediction[0]), confidence, inference_latency


def get_resource_values():
    """Read current resource values without assuming one fixed key spelling."""
    resources = get_all_resources(test_internet=False)

    cpu = get_value(
        resources,
        ["cpu_percent", "cpu_usage", "CPU Usage (%)", "cpu"],
    )

    ram = get_value(
        resources,
        [
            "available_ram_mb",
            "available_memory_mb",
            "ram_available_mb",
            "available_ram",
            "Available RAM (MB)"
        ],
    )

    return resources, cpu, ram


def get_network_values():
    network = measure_network()

    upload = get_value(
        network,
        ["upload_mbps", "upload_speed_mbps", "Upload Speed (Mbps)"]
    )

    download = get_value(
        network,
        ["download_mbps", "download_speed_mbps", "Download Speed (Mbps)"]
    )

    latency = get_value(
        network,
        ["latency_ms", "network_latency_ms", "Network Latency (ms)"]
    )

    return network, upload, download, latency


def run_one_target(target, strategy, input_data, models, scalers):
    edge_model = models["edge"][target]
    cloud_model = models["cloud"][target]

    edge_prediction, edge_confidence, edge_latency = get_model_prediction(
        edge_model,
        scalers["edge"],
        input_data
    )

    result = {
        "Target": target,
        "Strategy": strategy,
        "Prediction": None,
        "Confidence (%)": None,
        "Selected Model": None,
        "Edge Prediction": edge_prediction,
        "Edge Confidence (%)": edge_confidence * 100,
        "Edge Latency (ms)": edge_latency,
        "Cloud Prediction": None,
        "Cloud Confidence (%)": None,
        "Cloud Latency (ms)": None,
        "Network Delay (ms)": None,
        "Estimated Cloud Time (ms)": None,
        "Resource Pressure": None,
        "Cloud Feasible": None
    }

    if strategy == "Edge Only":
        result["Prediction"] = edge_prediction
        result["Confidence (%)"] = edge_confidence * 100
        result["Selected Model"] = "Edge"

    elif strategy == "Cloud Only":
        cloud_prediction, cloud_confidence, cloud_latency = get_model_prediction(
            cloud_model,
            scalers["cloud"],
            input_data
        )

        result["Prediction"] = cloud_prediction
        result["Confidence (%)"] = cloud_confidence * 100
        result["Selected Model"] = "Cloud"
        result["Cloud Prediction"] = cloud_prediction
        result["Cloud Confidence (%)"] = cloud_confidence * 100
        result["Cloud Latency (ms)"] = cloud_latency

    elif strategy == "Hybrid Offloading":
        if edge_confidence >= CONFIDENCE_THRESHOLD:
            result["Prediction"] = edge_prediction
            result["Confidence (%)"] = edge_confidence * 100
            result["Selected Model"] = "Edge"
        else:
            request_size = input_data.memory_usage(deep=True).sum()
            network_delay = calculate_network_delay(
                request_size,
                8,
                HYBRID_BANDWIDTH_MBPS,
                HYBRID_BANDWIDTH_MBPS,
                HYBRID_NETWORK_LATENCY_MS
            )

            cloud_prediction, cloud_confidence, cloud_latency = get_model_prediction(
                cloud_model,
                scalers["cloud"],
                input_data
            )

            result["Prediction"] = cloud_prediction
            result["Confidence (%)"] = cloud_confidence * 100
            result["Selected Model"] = "Cloud"
            result["Cloud Prediction"] = cloud_prediction
            result["Cloud Confidence (%)"] = cloud_confidence * 100
            result["Cloud Latency (ms)"] = cloud_latency
            result["Network Delay (ms)"] = network_delay

    elif strategy == "Adaptive Offloading":
        # Adaptive flow: run Edge first. Only inspect resources and consider
        # Cloud when Edge confidence is below the configured threshold.
        result["CPU Usage (%)"] = None
        result["Available RAM (MB)"] = None
        result["CPU Threshold (%)"] = MAX_CPU_USAGE_PERCENT
        result["RAM Threshold (MB)"] = MIN_AVAILABLE_RAM_MB
        result["Deadline Threshold (ms)"] = DEADLINE_MS
        result["Decision Reason"] = ""

        if edge_confidence >= CONFIDENCE_THRESHOLD:
            result["Prediction"] = edge_prediction
            result["Confidence (%)"] = edge_confidence * 100
            result["Selected Model"] = "Edge"
            result["Resource Pressure"] = False
            result["Cloud Feasible"] = None
            result["Decision Reason"] = (
                f"Edge confidence ({edge_confidence * 100:.2f}%) is at or above "
                f"the {CONFIDENCE_THRESHOLD * 100:.0f}% threshold. Cloud offloading "
                "was not needed, so resource checks were skipped."
            )
        else:
            # Edge confidence is low: check system resources before Cloud.
            resources, cpu, ram = get_resource_values()

            if cpu is None or ram is None:
                raise ValueError(
                    "Could not identify CPU/RAM values returned by system_monitor.py. "
                    f"Available resource keys: {list(resources.keys())}"
                )

            cpu = float(cpu)
            ram = float(ram)
            result["CPU Usage (%)"] = cpu
            result["Available RAM (MB)"] = ram

            resource_pressure = (
                cpu >= MAX_CPU_USAGE_PERCENT
                or ram <= MIN_AVAILABLE_RAM_MB
            )
            result["Resource Pressure"] = resource_pressure

            if resource_pressure:
                reasons = []
                if cpu >= MAX_CPU_USAGE_PERCENT:
                    reasons.append(
                        f"CPU usage is {cpu:.2f}%, which reaches/exceeds the "
                        f"{MAX_CPU_USAGE_PERCENT}% limit"
                    )
                if ram <= MIN_AVAILABLE_RAM_MB:
                    reasons.append(
                        f"available RAM is {ram:.2f} MB, which is at/below the "
                        f"{MIN_AVAILABLE_RAM_MB} MB minimum"
                    )

                result["Prediction"] = edge_prediction
                result["Confidence (%)"] = edge_confidence * 100
                result["Selected Model"] = "Edge"
                result["Cloud Feasible"] = False
                result["Decision Reason"] = (
                    "Cloud model was NOT executed because resource pressure was "
                    "detected: " + "; ".join(reasons) + ". The Edge prediction "
                    "is shown instead."
                )
            else:
                # Resources are sufficient. Now verify whether the network and
                # cloud processing estimate can satisfy the 5 ms deadline.
                network, upload, download, network_latency = get_network_values()

                if upload is None or download is None or network_latency is None:
                    raise ValueError(
                        "Could not identify network values returned by "
                        f"network_monitor.py. Available keys: {list(network.keys())}"
                    )

                upload = float(upload)
                download = float(download)
                network_latency = float(network_latency)

                cloud_benchmark = read_result_file("cloud_only_results")
                cloud_processing_time = 0.0

                if cloud_benchmark is not None and "Target" in cloud_benchmark.columns:
                    cloud_row = cloud_benchmark[
                        cloud_benchmark["Target"].astype(str) == target
                    ]
                    if not cloud_row.empty:
                        processing_column = next(
                            (column for column in ["Cloud Processing (ms)", "Cloud Latency (ms)", "Average Latency (ms)"]
                             if column in cloud_row.columns),
                            None
                        )
                        if processing_column:
                            cloud_processing_time = float(
                                cloud_row[processing_column].iloc[0]
                            )

                request_size = int(input_data.memory_usage(deep=True).sum())
                network_delay = calculate_network_delay(
                    request_size,
                    8,
                    upload,
                    download,
                    network_latency
                )
                estimated_cloud_time = network_delay + cloud_processing_time
                cloud_feasible = estimated_cloud_time <= DEADLINE_MS

                result["Network Delay (ms)"] = network_delay
                result["Estimated Cloud Time (ms)"] = estimated_cloud_time
                result["Cloud Feasible"] = cloud_feasible

                if cloud_feasible:
                    cloud_prediction, cloud_confidence, cloud_latency = get_model_prediction(
                        cloud_model,
                        scalers["cloud"],
                        input_data
                    )

                    result["Prediction"] = cloud_prediction
                    result["Confidence (%)"] = cloud_confidence * 100
                    result["Selected Model"] = "Cloud"
                    result["Cloud Prediction"] = cloud_prediction
                    result["Cloud Confidence (%)"] = cloud_confidence * 100
                    result["Cloud Latency (ms)"] = cloud_latency
                    result["Decision Reason"] = (
                        f"Edge confidence ({edge_confidence * 100:.2f}%) is below "
                        f"the {CONFIDENCE_THRESHOLD * 100:.0f}% threshold. CPU and RAM "
                        "are within limits, and estimated Cloud time "
                        f"({estimated_cloud_time:.3f} ms) meets the {DEADLINE_MS} ms "
                        "deadline. Cloud model was executed."
                    )
                else:
                    result["Prediction"] = edge_prediction
                    result["Confidence (%)"] = edge_confidence * 100
                    result["Selected Model"] = "Edge"
                    result["Decision Reason"] = (
                        f"Edge confidence ({edge_confidence * 100:.2f}%) is below "
                        f"the {CONFIDENCE_THRESHOLD * 100:.0f}% threshold, but Cloud "
                        f"was not executed because estimated Cloud time "
                        f"({estimated_cloud_time:.3f} ms) exceeds the "
                        f"{DEADLINE_MS} ms deadline. The Edge prediction is shown."
                    )

    return result


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    edge_models = joblib.load("models/edge_models.pkl")
    edge_scaler = joblib.load("models/scaler.pkl")
    cloud_models = joblib.load("models/cloud_models.pkl")
    cloud_scaler = joblib.load("models/cloud_scaler.pkl")

    return {
        "edge": edge_models,
        "cloud": cloud_models
    }, {
        "edge": edge_scaler,
        "cloud": cloud_scaler
    }


try:
    models, scalers = load_models()
    models_loaded = True
except Exception as error:
    models_loaded = False
    st.error(f"Model loading failed: {error}")
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Project Overview"

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Edge AI Patient Monitoring")
st.sidebar.divider()

pages = [
    "Project Overview",
    "Run Predictions",
    "Strategy Comparison",
    "Resource Monitoring",
    "Experiment Results"
]

for page_name in pages:
    if st.sidebar.button(page_name, use_container_width=True):
        st.session_state.page = page_name

st.sidebar.divider()
st.sidebar.subheader("Patient Sensor Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Patient CSV",
    type=["csv"]
)

patient_data = None

if uploaded_file is not None:
    try:
        patient_data = pd.read_csv(BytesIO(uploaded_file.getvalue()))

        if patient_data.empty:
            st.sidebar.error("The uploaded CSV has no records.")
            patient_data = None
        else:
            st.sidebar.success("CSV uploaded successfully!")
            st.sidebar.caption(f"{len(patient_data):,} patient records")

    except Exception as error:
        st.sidebar.error(f"CSV reading error: {error}")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0;">
        Edge AI Based Patient Monitoring System
    </h1>
    <p style="text-align:center;">
        Resource-Aware Intelligent Offloading
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)



# ============================================================
# PROJECT OVERVIEW
# ============================================================

if st.session_state.page == "Project Overview":

    st.title("Project Overview")

    st.subheader(
        "Resource-Aware Intelligent Offloading for Edge AI"
    )

    st.write(
        """
        This project develops an intelligent patient monitoring
        system using Edge AI and Cloud AI.

        It aims to process patient sensor data efficiently by
        deciding whether predictions should be generated locally
        using an Edge model or remotely using a Cloud model.
        """
    )

    st.divider()

    st.subheader("1. Problem Statement")

    st.write(
        """
        Patient monitoring sensors continuously generate data
        such as heart rate, SpO2, ECG, body temperature and
        movement information.

        Sending every record to the Cloud can introduce network
        delays. Processing everything on Edge devices can also
        be difficult because of limited CPU and memory resources.

        Our project addresses these challenges through
        resource-aware adaptive offloading.
        """
    )

    st.subheader("2. Project Objectives")

    st.markdown("""
    - Perform fast predictions using Edge AI.
    - Use Cloud AI when Edge confidence is insufficient.
    - Monitor CPU and available RAM before Cloud execution.
    - Consider network delay and Cloud processing time.
    - Meet the configured response deadline.
    - Reduce unnecessary Cloud communication.
    """)

    st.subheader("3. How Adaptive Offloading Works")

    st.markdown("""
    1. Patient sensor data is provided as input.
    2. The Edge model generates predictions.
    3. Edge prediction confidence is checked.
    4. If confidence is sufficient, the Edge result is used.
    5. Otherwise, CPU and available RAM are checked.
    6. If resources are sufficient, Cloud response time is estimated.
    7. If the deadline can be met, the Cloud model is executed.
    8. Otherwise, the Edge prediction is retained.
    """)

    st.info(
        f"""
        Current configuration:

        - Confidence threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%
        - Maximum CPU usage: {MAX_CPU_USAGE_PERCENT}%
        - Minimum available RAM: {MIN_AVAILABLE_RAM_MB} MB
        - Cloud deadline: {DEADLINE_MS} ms
        """
    )

    st.subheader("4. Prediction Targets")

    target_table = pd.DataFrame({
        "Target": TARGETS,
        "Description": [
            "Fall detection",
            "Abnormal heart rhythm detection",
            "Hypoxemia event detection",
            "Fever event detection"
        ]
    })

    st.dataframe(
        target_table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("5. Models Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Edge Model")
        st.write(
            "Logistic Regression is used for lightweight "
            "local prediction."
        )

    with col2:
        st.markdown("### Cloud Model")
        st.write(
            "MLP Neural Network is used for Cloud-side "
            "prediction."
        )

    st.subheader("6. Technologies Used")

    st.markdown("""
    - Python
    - Scikit-learn
    - Pandas and NumPy
    - Streamlit
    - Plotly
    - Joblib
    """)

    st.subheader("7. Explore the Project")

    st.markdown("""
    - **Run Predictions:** Test Adaptive Offloading using
      patient sensor records.
    - **Strategy Comparison:** Compare Edge, Cloud, Hybrid
      and Adaptive strategies.
    - **Resource Monitoring:** Monitor CPU, RAM and network.
    - **Experiment Results:** Explore saved experimental
      results and visualizations.
    """)

    st.warning(
        "This is a research demonstration using a dataset "
        "that is not clinically validated. Predictions are "
        "not medical diagnoses."
    )

    st.caption(
        "Cloud inference is simulated using a separately "
        "saved Cloud model on the machine running this app. "
        "It is not a remote Cloud server."
    )



# ============================================================
# RUN PREDICTIONS
# ============================================================

elif st.session_state.page == "Run Predictions":

    st.title("Run Predictions")

    st.write(
        "Select one patient record and run the selected processing strategy "
        "for all four patient-monitoring targets."
    )

    if patient_data is None:
        st.warning("Upload a patient CSV file using the sidebar first.")

    else:
        missing_columns = [
            column for column in FEATURE_COLUMNS
            if column not in patient_data.columns
        ]

        if missing_columns:
            st.error(f"Missing required sensor features: {missing_columns}")

        else:
            col1, col2 = st.columns(2)

            with col1:
                selected_row = st.selectbox(
                    "Select CSV record",
                    options=list(range(len(patient_data))),
                    format_func=lambda index: f"Record {index + 1}"
                )

            with col2:
                st.text_input(
                    "Processing strategy",
                    value="Adaptive Offloading",
                    disabled=True
                )

            selected_strategy = "Adaptive Offloading"

            with st.expander("Preview selected patient record"):
                st.dataframe(
                    patient_data.iloc[[selected_row]],
                    use_container_width=True
                )

            if st.button("Run Predictions for All 4 Targets", type="primary"):
                input_data = patient_data[FEATURE_COLUMNS].iloc[[selected_row]]

                try:
                    results = []

                    progress = st.progress(0)

                    for index, target_name in enumerate(TARGETS):
                        result = run_one_target(
                            target_name,
                            selected_strategy,
                            input_data,
                            models,
                            scalers
                        )
                        results.append(result)
                        progress.progress((index + 1) / len(TARGETS))

                    st.session_state.prediction_results = pd.DataFrame(results)
                    st.session_state.prediction_strategy = selected_strategy

                    st.success("All four predictions completed!")

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

            if st.session_state.prediction_results is not None:
                results_df = st.session_state.prediction_results

                st.divider()
                st.subheader(
                    f"Prediction Results — {st.session_state.prediction_strategy}"
                )

                display_columns = [
                    "Target",
                    "Prediction",
                    "Confidence (%)",
                    "Selected Model",
                    "Edge Confidence (%)",
                    "Edge Latency (ms)",
                    "Cloud Latency (ms)",
                    "CPU Usage (%)",
                    "Available RAM (MB)"
                ]

                st.dataframe(
                    results_df[display_columns],
                    use_container_width=True,
                    hide_index=True
                )

                for _, result in results_df.iterrows():
                    with st.expander(str(result["Target"])):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Predicted Class",
                                int(result["Prediction"])
                            )

                        with col2:
                            st.metric(
                                "Confidence",
                                f'{result["Confidence (%)"]:.2f}%'
                            )

                        with col3:
                            st.metric(
                                "Selected Model",
                                result["Selected Model"]
                            )

                        st.write(
                            f'Edge confidence: {result["Edge Confidence (%)"]:.2f}%'
                        )
                        st.write(
                            f'Edge inference latency: {result["Edge Latency (ms)"]:.4f} ms'
                        )

                        st.info(result["Decision Reason"])

                        if pd.notna(result.get("CPU Usage (%)")):
                            st.subheader("Resource Check")
                            resource_col1, resource_col2 = st.columns(2)
                            with resource_col1:
                                st.metric(
                                    "Current CPU Usage",
                                    f'{result["CPU Usage (%)"]:.2f}%',
                                    delta=f'Threshold: {MAX_CPU_USAGE_PERCENT}%',
                                    delta_color="inverse"
                                )
                            with resource_col2:
                                st.metric(
                                    "Available RAM",
                                    f'{result["Available RAM (MB)"]:.2f} MB',
                                    delta=f'Minimum: {MIN_AVAILABLE_RAM_MB} MB',
                                    delta_color="normal"
                                )
                            st.caption(
                                "Cloud resource rule: CPU must be below "
                                f"{MAX_CPU_USAGE_PERCENT}% and available RAM must be "
                                f"above {MIN_AVAILABLE_RAM_MB} MB."
                            )

                        if pd.notna(result["Cloud Latency (ms)"]):
                            st.write(
                                f'Cloud inference latency: {result["Cloud Latency (ms)"]:.4f} ms'
                            )

                        if pd.notna(result["Network Delay (ms)"]):
                            st.write(
                                f'Network delay: {result["Network Delay (ms)"]:.4f} ms'
                            )

                        if pd.notna(result["Resource Pressure"]):
                            st.write(
                                f'Resource pressure: {result["Resource Pressure"]}'
                            )
                            st.write(
                                f'Cloud feasible under {DEADLINE_MS} ms deadline: '
                                f'{result["Cloud Feasible"]}'
                            )

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Prediction Results",
                    data=csv_bytes,
                    file_name="patient_prediction_results.csv",
                    mime="text/csv"
                )

                st.caption(
                    "These are model predictions for the selected CSV record. "
                    "They are not clinical diagnoses."
                )


# ============================================================
# STRATEGY COMPARISON
# ============================================================

elif st.session_state.page == "Strategy Comparison":

    st.title("Strategy Comparison")

    summary = read_result_file("overall_strategy_summary")

    if summary is None:
        st.warning(
            "overall_strategy_summary result file was not found in the "
            "results or Results folder."
        )

    else:
        st.subheader("Overall Strategy Summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if "Strategy" in summary.columns and "Accuracy" in summary.columns:
            accuracy_chart = px.bar(
                summary,
                x="Strategy",
                y="Accuracy",
                title="Accuracy by Strategy",
                text_auto=".4f"
            )
            st.plotly_chart(accuracy_chart, use_container_width=True)

        if "Average Latency (ms)" in summary.columns:
            latency_chart = px.bar(
                summary,
                x="Strategy",
                y="Average Latency (ms)",
                title="Average Latency by Strategy",
                text_auto=".4f"
            )
            st.plotly_chart(latency_chart, use_container_width=True)

        if "F1 Score" in summary.columns:
            f1_chart = px.bar(
                summary,
                x="Strategy",
                y="F1 Score",
                title="F1 Score by Strategy",
                text_auto=".4f"
            )
            st.plotly_chart(f1_chart, use_container_width=True)


# ============================================================
# RESOURCE MONITORING
# ============================================================

elif st.session_state.page == "Resource Monitoring":

    st.title("Resource Monitoring")

    st.write(
        "Measure the current laptop resources and network conditions "
        "used by the adaptive offloading demonstration."
    )

    if st.button("Measure Current Resources", type="primary"):

        try:
            resources, cpu, ram = get_resource_values()
            network, upload, download, network_latency = get_network_values()

            if cpu is None:
                st.warning(
                    "CPU value was not identified. Resource keys returned: "
                    f"{list(resources.keys())}"
                )
            else:
                st.metric("CPU Usage", f"{float(cpu):.2f}%")

            if ram is None:
                st.warning(
                    "Available RAM value was not identified. Resource keys returned: "
                    f"{list(resources.keys())}"
                )
            else:
                st.metric("Available RAM", f"{float(ram):.2f} MB")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Upload Speed",
                    "Unavailable" if upload is None else f"{float(upload):.2f} Mbps"
                )

            with col2:
                st.metric(
                    "Download Speed",
                    "Unavailable" if download is None else f"{float(download):.2f} Mbps"
                )

            with col3:
                st.metric(
                    "Network Latency",
                    "Unavailable" if network_latency is None else f"{float(network_latency):.2f} ms"
                )

            if cpu is not None and ram is not None:
                pressure = (
                    float(cpu) >= MAX_CPU_USAGE_PERCENT
                    or float(ram) <= MIN_AVAILABLE_RAM_MB
                )

                if pressure:
                    st.warning("Resource Pressure: TRUE")
                else:
                    st.success("Resource Pressure: FALSE")

            st.subheader("Raw System Resource Values")
            st.json(resources)

            st.subheader("Raw Network Values")
            st.json(network)

        except Exception as error:
            st.error(f"Resource measurement failed: {error}")

    st.divider()
    st.subheader("Adaptive Offloading Configuration")

    config_df = pd.DataFrame({
        "Parameter": [
            "Confidence Threshold",
            "Maximum CPU Usage",
            "Minimum Available RAM",
            "Cloud Deadline"
        ],
        "Value": [
            f"{CONFIDENCE_THRESHOLD:.2f}",
            f"{MAX_CPU_USAGE_PERCENT}%",
            f"{MIN_AVAILABLE_RAM_MB} MB",
            f"{DEADLINE_MS} ms"
        ]
    })

    st.dataframe(config_df, use_container_width=True, hide_index=True)


# ============================================================
# EXPERIMENT RESULTS
# ============================================================

elif st.session_state.page == "Experiment Results":

    st.title("Experiment Results")

    st.write(
        "Explore the saved training, testing, benchmarking, offloading, "
        "deadline, and adaptive experiment results."
    )

    experiment_tabs = st.tabs([
        "Model Benchmark",
        "Edge vs Cloud",
        "Hybrid",
        "Adaptive",
        "Deadline",
        "Resource Pressure",
        "Scenario",
        "Overall Performance"
    ])

    # --------------------------------------------------------
    # MODEL BENCHMARK
    # --------------------------------------------------------

    with experiment_tabs[0]:

        st.subheader("Model Benchmark")

        benchmark = read_result_file("model_benchmark")

        if benchmark is None:
            st.info("model_benchmark result file is not available.")
        else:
            st.dataframe(benchmark, use_container_width=True, hide_index=True)

            if {"Target", "Edge Accuracy", "Cloud Accuracy"}.issubset(benchmark.columns):
                plot_data = benchmark.melt(
                    id_vars="Target",
                    value_vars=["Edge Accuracy", "Cloud Accuracy"],
                    var_name="Model",
                    value_name="Accuracy"
                )

                fig = px.bar(
                    plot_data,
                    x="Target",
                    y="Accuracy",
                    color="Model",
                    barmode="group",
                    title="Edge vs Cloud Accuracy"
                )
                st.plotly_chart(fig, use_container_width=True)

            if {"Target", "Edge F1", "Cloud F1"}.issubset(benchmark.columns):
                plot_data = benchmark.melt(
                    id_vars="Target",
                    value_vars=["Edge F1", "Cloud F1"],
                    var_name="Model",
                    value_name="F1 Score"
                )

                fig = px.bar(
                    plot_data,
                    x="Target",
                    y="F1 Score",
                    color="Model",
                    barmode="group",
                    title="Edge vs Cloud F1 Score"
                )
                st.plotly_chart(fig, use_container_width=True)

            if {"Target", "Edge Size (KB)", "Cloud Size (KB)"}.issubset(benchmark.columns):
                plot_data = benchmark.melt(
                    id_vars="Target",
                    value_vars=["Edge Size (KB)", "Cloud Size (KB)"],
                    var_name="Model",
                    value_name="Model Size (KB)"
                )

                fig = px.bar(
                    plot_data,
                    x="Target",
                    y="Model Size (KB)",
                    color="Model",
                    barmode="group",
                    title="Model Size Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # EDGE VS CLOUD
    # --------------------------------------------------------

    with experiment_tabs[1]:

        st.subheader("Edge Only Results")

        edge_results = read_result_file("edge_only_results")

        if edge_results is None:
            st.info("edge_only_results file is not available.")
        else:
            st.dataframe(edge_results, use_container_width=True, hide_index=True)

        st.subheader("Cloud Only Results")

        cloud_results = read_result_file("cloud_only_results")

        if cloud_results is None:
            st.info("cloud_only_results file is not available.")
        else:
            st.dataframe(cloud_results, use_container_width=True, hide_index=True)

            if {"Target", "Total Latency (ms)"}.issubset(cloud_results.columns):
                fig = px.bar(
                    cloud_results,
                    x="Target",
                    y="Total Latency (ms)",
                    title="Cloud Total Latency by Target"
                )
                st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    with experiment_tabs[2]:

        st.subheader("Hybrid Offloading Results")

        hybrid_results = read_result_file("hybrid_results")

        if hybrid_results is None:
            st.info("hybrid_results file is not available.")
        else:
            st.dataframe(hybrid_results, use_container_width=True, hide_index=True)

            if {"Target", "Offload Percentage"}.issubset(hybrid_results.columns):
                fig = px.bar(
                    hybrid_results,
                    x="Target",
                    y="Offload Percentage",
                    title="Hybrid Cloud Offload Percentage"
                )
                st.plotly_chart(fig, use_container_width=True)

            if {"Target", "Edge Requests", "Cloud Requests"}.issubset(hybrid_results.columns):
                plot_data = hybrid_results.melt(
                    id_vars="Target",
                    value_vars=["Edge Requests", "Cloud Requests"],
                    var_name="Processing Location",
                    value_name="Requests"
                )

                fig = px.bar(
                    plot_data,
                    x="Target",
                    y="Requests",
                    color="Processing Location",
                    barmode="group",
                    title="Hybrid Edge and Cloud Requests"
                )
                st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # ADAPTIVE
    # --------------------------------------------------------

    with experiment_tabs[3]:

        st.subheader("Adaptive Offloading Results")

        adaptive_files = [
            ("adaptive_results", "Adaptive Network Scenarios"),
            ("final_adaptive_results", "Final Adaptive Results"),
            ("accuracy_first_adaptive_results", "Accuracy-First Adaptive Results")
        ]

        for file_key, heading in adaptive_files:
            st.markdown(f"#### {heading}")

            adaptive_df = read_result_file(file_key)

            if adaptive_df is None:
                st.info(f"{file_key} file is not available.")
            else:
                st.dataframe(
                    adaptive_df,
                    use_container_width=True,
                    hide_index=True
                )

                if {"Network", "Offload Percentage"}.issubset(adaptive_df.columns):
                    fig = px.bar(
                        adaptive_df,
                        x="Network",
                        y="Offload Percentage",
                        color="Target" if "Target" in adaptive_df.columns else None,
                        barmode="group",
                        title=f"{heading}: Offload Percentage"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if {"Network", "Estimated Cloud Response (ms)"}.issubset(adaptive_df.columns):
                    fig = px.bar(
                        adaptive_df,
                        x="Network",
                        y="Estimated Cloud Response (ms)",
                        color="Target" if "Target" in adaptive_df.columns else None,
                        barmode="group",
                        title=f"{heading}: Estimated Cloud Response"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # DEADLINE EXPERIMENT
    # --------------------------------------------------------

    with experiment_tabs[4]:

        st.subheader("Deadline Experiment Results")

        deadline_results = read_result_file("deadline_test_results")

        if deadline_results is None:
            st.info("deadline_test_results file is not available.")
        else:
            st.dataframe(
                deadline_results,
                use_container_width=True,
                hide_index=True
            )

            if {"Target", "Estimated Cloud Response (ms)"}.issubset(deadline_results.columns):
                fig = px.bar(
                    deadline_results,
                    x="Target",
                    y="Estimated Cloud Response (ms)",
                    title="Estimated Cloud Response Time"
                )
                fig.add_hline(
                    y=DEADLINE_MS,
                    line_dash="dash",
                    annotation_text=f"Current app deadline: {DEADLINE_MS} ms"
                )
                st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # RESOURCE PRESSURE
    # --------------------------------------------------------

    with experiment_tabs[5]:

        st.subheader("Resource Pressure Experiment Results")

        pressure_results = read_result_file("resource_pressure_test_results")

        if pressure_results is None:
            st.info(
                "resource_pressure_test_results was not included in the uploaded "
                "ZIP. Add that file to the results folder to display this experiment."
            )
        else:
            st.dataframe(
                pressure_results,
                use_container_width=True,
                hide_index=True
            )

            if {"Available RAM (MB)", "Offload Percentage"}.issubset(pressure_results.columns):
                fig = px.scatter(
                    pressure_results,
                    x="Available RAM (MB)",
                    y="Offload Percentage",
                    color="Target" if "Target" in pressure_results.columns else None,
                    title="Available RAM vs Offload Percentage"
                )
                st.plotly_chart(fig, use_container_width=True)

            if {"CPU Usage (%)", "Offload Percentage"}.issubset(pressure_results.columns):
                fig = px.scatter(
                    pressure_results,
                    x="CPU Usage (%)",
                    y="Offload Percentage",
                    color="Target" if "Target" in pressure_results.columns else None,
                    title="CPU Usage vs Offload Percentage"
                )
                st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # SCENARIO COMPARISON
    # --------------------------------------------------------

    with experiment_tabs[6]:

        st.subheader("Scenario Comparison")

        scenario_results = read_result_file("scenario_comparison")

        if scenario_results is None:
            st.info(
                "scenario_comparison.csv was visible in your screenshot but "
                "was not included in the uploaded ZIP."
            )
        else:
            st.dataframe(
                scenario_results,
                use_container_width=True,
                hide_index=True
            )

            numeric_columns = scenario_results.select_dtypes(
                include=np.number
            ).columns.tolist()

            if len(numeric_columns) >= 1:
                selected_metric = st.selectbox(
                    "Select scenario metric",
                    numeric_columns,
                    key="scenario_metric"
                )

                category_columns = [
                    column for column in scenario_results.columns
                    if column not in numeric_columns
                ]

                if category_columns:
                    selected_category = category_columns[0]

                    fig = px.bar(
                        scenario_results,
                        x=selected_category,
                        y=selected_metric,
                        title=f"{selected_metric} by {selected_category}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # OVERALL PERFORMANCE
    # --------------------------------------------------------

    with experiment_tabs[7]:

        st.subheader("Overall Strategy Performance")

        overall_results = read_result_file("overall_strategy_summary")

        if overall_results is None:
            st.info("overall_strategy_summary file is not available.")
        else:
            st.dataframe(
                overall_results,
                use_container_width=True,
                hide_index=True
            )

            if {"Strategy", "Accuracy"}.issubset(overall_results.columns):
                fig = px.bar(
                    overall_results,
                    x="Strategy",
                    y="Accuracy",
                    title="Overall Accuracy by Strategy",
                    text_auto=".4f"
                )
                st.plotly_chart(fig, use_container_width=True)

            if {"Strategy", "F1 Score"}.issubset(overall_results.columns):
                fig = px.bar(
                    overall_results,
                    x="Strategy",
                    y="F1 Score",
                    title="Overall F1 Score by Strategy",
                    text_auto=".4f"
                )
                st.plotly_chart(fig, use_container_width=True)

            if {"Strategy", "Average Latency (ms)"}.issubset(overall_results.columns):
                fig = px.bar(
                    overall_results,
                    x="Strategy",
                    y="Average Latency (ms)",
                    title="Overall Average Latency by Strategy",
                    text_auto=".4f"
                )
                st.plotly_chart(fig, use_container_width=True)

        performance_image = None

        for folder in RESULTS_DIR_CANDIDATES:
            candidate = os.path.join(folder, "overall_performance.png")
            if os.path.exists(candidate):
                performance_image = candidate
                break

        if performance_image is not None:
            st.subheader("Saved Overall Performance Plot")
            st.image(performance_image, use_container_width=True)
        else:
            st.info("overall_performance.png was not found in the results folder.")

        st.caption(
            "Experiment tables and plots are loaded from saved result files. "
            "Missing files are not replaced with estimated or fabricated results."
        )
