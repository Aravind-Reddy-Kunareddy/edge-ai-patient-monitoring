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
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

COMPARISON_DIR = os.path.join(
    PROJECT_DIR,
    "comparison_experiment",
    "results",
    "final"
)

SYSTEM_COMPARISON_PATH = os.path.join(
    COMPARISON_DIR,
    "system_comparison.csv"
)

TARGET_COMPARISON_PATH = os.path.join(
    COMPARISON_DIR,
    "target_wise_comparison.csv"
)

EDGESIMPY_RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "comparison_experiment",
    "results",
    "edgesimpy"
)

CUSTOM_RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "comparison_experiment",
    "results",
    "custom"
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

CONFIDENCE_THRESHOLD = 0.90
HYBRID_BANDWIDTH_MBPS = 5
HYBRID_NETWORK_LATENCY_MS = 10
DEADLINE_MS = 100
MAX_CPU_USAGE_PERCENT = 90
MIN_AVAILABLE_RAM_MB = 750

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
    The supplied result files may contain CSV text inside files
    named .xls. Try CSV first, then Excel as fallback.
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
            st.warning(
                f"Could not read {os.path.basename(path)}: {error}"
            )
            return None


def get_value(dictionary, possible_keys, default=None):

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

    inference_latency = (
        time.perf_counter() - start_time
    ) * 1000

    probabilities = model.predict_proba(scaled_data)

    confidence = float(np.max(probabilities[0]))

    return (
        int(prediction[0]),
        confidence,
        inference_latency
    )


def get_resource_values():

    resources = get_all_resources(test_internet=False)

    cpu = get_value(
        resources,
        [
            "cpu_percent",
            "cpu_usage",
            "CPU Usage (%)",
            "cpu"
        ]
    )

    ram = get_value(
        resources,
        [
            "available_ram_mb",
            "available_memory_mb",
            "ram_available_mb",
            "available_ram",
            "Available RAM (MB)"
        ]
    )

    return resources, cpu, ram


def get_network_values():

    network = measure_network()

    upload = get_value(
        network,
        [
            "upload_mbps",
            "upload_speed_mbps",
            "Upload Speed (Mbps)"
        ]
    )

    download = get_value(
        network,
        [
            "download_mbps",
            "download_speed_mbps",
            "Download Speed (Mbps)"
        ]
    )

    latency = get_value(
        network,
        [
            "latency_ms",
            "network_latency_ms",
            "Network Latency (ms)"
        ]
    )

    return network, upload, download, latency


# ============================================================
# COMPARISON DATA FUNCTIONS
# ============================================================

def load_comparison_data():

    system_df = None
    target_df = None

    if os.path.exists(SYSTEM_COMPARISON_PATH):

        system_df = pd.read_csv(
            SYSTEM_COMPARISON_PATH
        )

    if os.path.exists(TARGET_COMPARISON_PATH):

        target_df = pd.read_csv(
            TARGET_COMPARISON_PATH
        )

    return system_df, target_df


def get_comparison_metric(
    system_df,
    metric_name,
    system_name,
    default=np.nan
):

    if system_df is None:
        return default

    row = system_df[
        system_df["Metric"] == metric_name
    ]

    if row.empty:
        return default

    if system_name not in row.columns:
        return default

    return row[system_name].iloc[0]


def make_comparison_chart(
    dataframe,
    x,
    y,
    title,
    color=None,
    barmode=None,
    text=None
):

    fig = px.bar(
        dataframe,
        x=x,
        y=y,
        color=color,
        barmode=barmode,
        text=text,
        title=title
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# ============================================================
# RUN PREDICTION FUNCTION
# ============================================================

def run_one_target(
    target,
    strategy,
    input_data,
    models,
    scalers
):

    edge_model = models["edge"][target]
    cloud_model = models["cloud"][target]

    edge_prediction, edge_confidence, edge_latency = (
        get_model_prediction(
            edge_model,
            scalers["edge"],
            input_data
        )
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

    # --------------------------------------------------------
    # EDGE ONLY
    # --------------------------------------------------------

    if strategy == "Edge Only":

        result["Prediction"] = edge_prediction
        result["Confidence (%)"] = edge_confidence * 100
        result["Selected Model"] = "Edge"

    # --------------------------------------------------------
    # CLOUD ONLY
    # --------------------------------------------------------

    elif strategy == "Cloud Only":

        cloud_prediction, cloud_confidence, cloud_latency = (
            get_model_prediction(
                cloud_model,
                scalers["cloud"],
                input_data
            )
        )

        result["Prediction"] = cloud_prediction
        result["Confidence (%)"] = cloud_confidence * 100
        result["Selected Model"] = "Cloud"

        result["Cloud Prediction"] = cloud_prediction
        result["Cloud Confidence (%)"] = cloud_confidence * 100
        result["Cloud Latency (ms)"] = cloud_latency

    # --------------------------------------------------------
    # HYBRID OFFLOADING
    # --------------------------------------------------------

    elif strategy == "Hybrid Offloading":

        if edge_confidence >= CONFIDENCE_THRESHOLD:

            result["Prediction"] = edge_prediction
            result["Confidence (%)"] = edge_confidence * 100
            result["Selected Model"] = "Edge"

        else:

            request_size = input_data.memory_usage(
                deep=True
            ).sum()

            network_delay = calculate_network_delay(
                request_size,
                8,
                HYBRID_BANDWIDTH_MBPS,
                HYBRID_BANDWIDTH_MBPS,
                HYBRID_NETWORK_LATENCY_MS
            )

            cloud_prediction, cloud_confidence, cloud_latency = (
                get_model_prediction(
                    cloud_model,
                    scalers["cloud"],
                    input_data
                )
            )

            result["Prediction"] = cloud_prediction
            result["Confidence (%)"] = cloud_confidence * 100
            result["Selected Model"] = "Cloud"

            result["Cloud Prediction"] = cloud_prediction
            result["Cloud Confidence (%)"] = cloud_confidence * 100
            result["Cloud Latency (ms)"] = cloud_latency

            result["Network Delay (ms)"] = network_delay

    # --------------------------------------------------------
    # ADAPTIVE OFFLOADING
    # --------------------------------------------------------

    elif strategy == "Adaptive Offloading":

        result["CPU Usage (%)"] = None
        result["Available RAM (MB)"] = None

        result["CPU Threshold (%)"] = MAX_CPU_USAGE_PERCENT
        result["RAM Threshold (MB)"] = MIN_AVAILABLE_RAM_MB
        result["Deadline Threshold (ms)"] = DEADLINE_MS

        result["Decision Reason"] = ""

        # High confidence: Edge
        if edge_confidence >= CONFIDENCE_THRESHOLD:

            result["Prediction"] = edge_prediction
            result["Confidence (%)"] = edge_confidence * 100
            result["Selected Model"] = "Edge"

            result["Resource Pressure"] = False
            result["Cloud Feasible"] = None

            result["Decision Reason"] = (
                f"Edge confidence ({edge_confidence * 100:.2f}%) "
                f"is at or above the {CONFIDENCE_THRESHOLD * 100:.0f}% "
                f"threshold. Cloud offloading was not needed."
            )

        else:

            # Low confidence: check resources
            resources, cpu, ram = get_resource_values()

            if cpu is None or ram is None:

                raise ValueError(
                    "Could not identify CPU/RAM values returned "
                    f"by system_monitor.py. Keys: {list(resources.keys())}"
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

            # Resource pressure: fallback to Edge
            if resource_pressure:

                reasons = []

                if cpu >= MAX_CPU_USAGE_PERCENT:

                    reasons.append(
                        f"CPU usage is {cpu:.2f}%, "
                        f"reaching the {MAX_CPU_USAGE_PERCENT}% limit"
                    )

                if ram <= MIN_AVAILABLE_RAM_MB:

                    reasons.append(
                        f"available RAM is {ram:.2f} MB, "
                        f"below the {MIN_AVAILABLE_RAM_MB} MB minimum"
                    )

                result["Prediction"] = edge_prediction
                result["Confidence (%)"] = edge_confidence * 100
                result["Selected Model"] = "Edge"

                result["Cloud Feasible"] = False

                result["Decision Reason"] = (
                    "Cloud was not executed because resource "
                    "pressure was detected: "
                    + "; ".join(reasons)
                )

            else:

                # Resource conditions are satisfied
                network, upload, download, network_latency = (
                    get_network_values()
                )

                if (
                    upload is None
                    or download is None
                    or network_latency is None
                ):

                    raise ValueError(
                        "Could not identify network values returned "
                        f"by network_monitor.py. Keys: {list(network.keys())}"
                    )

                upload = float(upload)
                download = float(download)
                network_latency = float(network_latency)

                cloud_benchmark = read_result_file(
                    "cloud_only_results"
                )

                cloud_processing_time = 0.0

                if (
                    cloud_benchmark is not None
                    and "Target" in cloud_benchmark.columns
                ):

                    cloud_row = cloud_benchmark[
                        cloud_benchmark["Target"].astype(str)
                        == target
                    ]

                    if not cloud_row.empty:

                        processing_column = next(
                            (
                                column
                                for column in [
                                    "Cloud Processing (ms)",
                                    "Cloud Latency (ms)",
                                    "Average Latency (ms)"
                                ]
                                if column in cloud_row.columns
                            ),
                            None
                        )

                        if processing_column:

                            cloud_processing_time = float(
                                cloud_row[processing_column].iloc[0]
                            )

                request_size = int(
                    input_data.memory_usage(deep=True).sum()
                )

                network_delay = calculate_network_delay(
                    request_size,
                    8,
                    upload,
                    download,
                    network_latency
                )

                estimated_cloud_time = (
                    network_delay + cloud_processing_time
                )

                cloud_feasible = (
                    estimated_cloud_time <= DEADLINE_MS
                )

                result["Network Delay (ms)"] = network_delay
                result["Estimated Cloud Time (ms)"] = estimated_cloud_time
                result["Cloud Feasible"] = cloud_feasible

                # Cloud feasible
                if cloud_feasible:

                    cloud_prediction, cloud_confidence, cloud_latency = (
                        get_model_prediction(
                            cloud_model,
                            scalers["cloud"],
                            input_data
                        )
                    )

                    result["Prediction"] = cloud_prediction
                    result["Confidence (%)"] = cloud_confidence * 100
                    result["Selected Model"] = "Cloud"

                    result["Cloud Prediction"] = cloud_prediction
                    result["Cloud Confidence (%)"] = cloud_confidence * 100
                    result["Cloud Latency (ms)"] = cloud_latency

                    result["Decision Reason"] = (
                        f"Edge confidence ({edge_confidence * 100:.2f}%) "
                        f"is below threshold. Resources are sufficient "
                        f"and estimated Cloud time "
                        f"({estimated_cloud_time:.3f} ms) meets the "
                        f"{DEADLINE_MS} ms deadline."
                    )

                # Cloud deadline cannot be met
                else:

                    result["Prediction"] = edge_prediction
                    result["Confidence (%)"] = edge_confidence * 100
                    result["Selected Model"] = "Edge"

                    result["Decision Reason"] = (
                        f"Cloud was not selected because estimated "
                        f"Cloud time ({estimated_cloud_time:.3f} ms) "
                        f"exceeds the {DEADLINE_MS} ms deadline."
                    )

    return result


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    edge_models = joblib.load(
        os.path.join(PROJECT_DIR, "models", "edge_models.pkl")
    )

    edge_scaler = joblib.load(
        os.path.join(PROJECT_DIR, "models", "scaler.pkl")
    )

    cloud_models = joblib.load(
        os.path.join(PROJECT_DIR, "models", "cloud_models.pkl")
    )

    cloud_scaler = joblib.load(
        os.path.join(PROJECT_DIR, "models", "cloud_scaler.pkl")
    )

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
    "Experiment Results",
    "EdgeSimPy Simulation",
    "Comparison Results"
]

for page_name in pages:

    if st.sidebar.button(
        page_name,
        use_container_width=True,
        key=f"navigation_{page_name}"
    ):

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

        patient_data = pd.read_csv(
            BytesIO(uploaded_file.getvalue())
        )

        if patient_data.empty:

            st.sidebar.error("The uploaded CSV has no records.")
            patient_data = None

        else:

            st.sidebar.success("CSV uploaded successfully!")

            st.sidebar.caption(
                f"{len(patient_data):,} patient records"
            )

    except Exception as error:

        st.sidebar.error(
            f"CSV reading error: {error}"
        )


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
        This project develops a resource-aware patient monitoring
        system that combines Edge AI and Cloud AI. Patient sensor
        data is processed locally first, and Cloud processing is
        considered only when Edge prediction confidence is low
        and the available system/network resources allow it.
        """
    )

    st.subheader("Simulation Experiments")

    st.markdown(
        """
        **1. Edge–Cloud infrastructure simulation (EdgeSimPy)**

        EdgeSimPy is used to study the behavior of an edge/cloud
        infrastructure and resource-allocation scenarios.

        **2. Confidence-based patient-monitoring offloading experiment**

        A full experiment was run using **10,000 patient samples**
        across **4 prediction targets**, producing **40,000 tasks**.
        The Edge model is selected when its confidence is at least
        90%. For low-confidence tasks, resource conditions are
        checked and Cloud is selected when available; otherwise,
        the task falls back to Edge.
        """
    )

    system_comparison, target_comparison = load_comparison_data()

    if system_comparison is not None:

        accuracy = get_comparison_metric(
            system_comparison,
            "Accuracy (%)",
            "Custom Adaptive"
        )

        offload = get_comparison_metric(
            system_comparison,
            "Cloud Offloading (%)",
            "Custom Adaptive"
        )

        total_tasks = get_comparison_metric(
            system_comparison,
            "Total Tasks",
            "Custom Adaptive"
        )

        overview_metrics = st.columns(4)

        overview_metrics[0].metric(
            "Samples",
            "10,000"
        )

        overview_metrics[1].metric(
            "Prediction Tasks",
            f"{int(total_tasks):,}"
        )

        overview_metrics[2].metric(
            "Custom Accuracy",
            f"{accuracy:.4f}%"
        )

        overview_metrics[3].metric(
            "Custom Cloud Offloading",
            f"{offload:.4f}%"
        )

    else:

        overview_metrics = st.columns(4)

        overview_metrics[0].metric("Samples", "10,000")
        overview_metrics[1].metric("Prediction Tasks", "40,000")
        overview_metrics[2].metric("Accuracy", "—")
        overview_metrics[3].metric("Cloud Offloading", "—")

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
        Current application configuration:

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
    - **EdgeSimPy Simulation:** View simulation metrics.
    - **Comparison Results:** Compare Custom Adaptive and
      EdgeSimPy using interactive Plotly charts.
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
        "Select one patient record and run the selected processing "
        "strategy for all four patient-monitoring targets."
    )

    if patient_data is None:

        st.warning(
            "Upload a patient CSV file using the sidebar first."
        )

    else:

        missing_columns = [
            column
            for column in FEATURE_COLUMNS
            if column not in patient_data.columns
        ]

        if missing_columns:

            st.error(
                f"Missing required sensor features: {missing_columns}"
            )

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

            if st.button(
                "Run Predictions for All 4 Targets",
                type="primary"
            ):

                input_data = patient_data[
                    FEATURE_COLUMNS
                ].iloc[[selected_row]]

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

                        progress.progress(
                            (index + 1) / len(TARGETS)
                        )

                    st.session_state.prediction_results = (
                        pd.DataFrame(results)
                    )

                    st.session_state.prediction_strategy = (
                        selected_strategy
                    )

                    st.success(
                        "All four predictions completed!"
                    )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )

            if st.session_state.prediction_results is not None:

                results_df = st.session_state.prediction_results

                st.divider()

                st.subheader(
                    f"Prediction Results — "
                    f"{st.session_state.prediction_strategy}"
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

                available_columns = [
                    column
                    for column in display_columns
                    if column in results_df.columns
                ]

                st.dataframe(
                    results_df[available_columns],
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
                            f'Edge confidence: '
                            f'{result["Edge Confidence (%)"]:.2f}%'
                        )

                        st.write(
                            f'Edge inference latency: '
                            f'{result["Edge Latency (ms)"]:.4f} ms'
                        )

                        st.info(
                            result.get(
                                "Decision Reason",
                                "No decision reason available."
                            )
                        )

                        if pd.notna(
                            result.get("CPU Usage (%)")
                        ):

                            st.subheader("Resource Check")

                            resource_col1, resource_col2 = st.columns(2)

                            with resource_col1:

                                st.metric(
                                    "Current CPU Usage",
                                    f'{result["CPU Usage (%)"]:.2f}%'
                                )

                            with resource_col2:

                                st.metric(
                                    "Available RAM",
                                    f'{result["Available RAM (MB)"]:.2f} MB'
                                )

                        if pd.notna(
                            result.get("Cloud Latency (ms)")
                        ):

                            st.write(
                                f'Cloud inference latency: '
                                f'{result["Cloud Latency (ms)"]:.4f} ms'
                            )

                        if pd.notna(
                            result.get("Network Delay (ms)")
                        ):

                            st.write(
                                f'Network delay: '
                                f'{result["Network Delay (ms)"]:.4f} ms'
                            )

                csv_bytes = results_df.to_csv(
                    index=False
                ).encode("utf-8")

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

    summary = read_result_file(
        "overall_strategy_summary"
    )

    if summary is None:

        st.warning(
            "overall_strategy_summary result file was not found."
        )

    else:

        st.subheader("Overall Strategy Summary")

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        if {
            "Strategy",
            "Accuracy"
        }.issubset(summary.columns):

            accuracy_chart = px.bar(
                summary,
                x="Strategy",
                y="Accuracy",
                title="Accuracy by Strategy",
                text_auto=".4f"
            )

            st.plotly_chart(
                accuracy_chart,
                use_container_width=True
            )

        if {
            "Strategy",
            "Average Latency (ms)"
        }.issubset(summary.columns):

            latency_chart = px.bar(
                summary,
                x="Strategy",
                y="Average Latency (ms)",
                title="Average Latency by Strategy",
                text_auto=".4f"
            )

            st.plotly_chart(
                latency_chart,
                use_container_width=True
            )

        if {
            "Strategy",
            "F1 Score"
        }.issubset(summary.columns):

            f1_chart = px.bar(
                summary,
                x="Strategy",
                y="F1 Score",
                title="F1 Score by Strategy",
                text_auto=".4f"
            )

            st.plotly_chart(
                f1_chart,
                use_container_width=True
            )


# ============================================================
# RESOURCE MONITORING
# ============================================================

elif st.session_state.page == "Resource Monitoring":

    st.title("Resource Monitoring")

    st.write(
        "Measure the current laptop resources and network conditions."
    )

    if st.button(
        "Measure Current Resources",
        type="primary"
    ):

        try:

            resources, cpu, ram = get_resource_values()

            network, upload, download, network_latency = (
                get_network_values()
            )

            if cpu is not None:

                st.metric(
                    "CPU Usage",
                    f"{float(cpu):.2f}%"
                )

            else:

                st.warning(
                    "CPU value was not identified."
                )

            if ram is not None:

                st.metric(
                    "Available RAM",
                    f"{float(ram):.2f} MB"
                )

            else:

                st.warning(
                    "Available RAM value was not identified."
                )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Upload Speed",
                    "Unavailable"
                    if upload is None
                    else f"{float(upload):.2f} Mbps"
                )

            with col2:

                st.metric(
                    "Download Speed",
                    "Unavailable"
                    if download is None
                    else f"{float(download):.2f} Mbps"
                )

            with col3:

                st.metric(
                    "Network Latency",
                    "Unavailable"
                    if network_latency is None
                    else f"{float(network_latency):.2f} ms"
                )

            if cpu is not None and ram is not None:

                pressure = (
                    float(cpu) >= MAX_CPU_USAGE_PERCENT
                    or float(ram) <= MIN_AVAILABLE_RAM_MB
                )

                if pressure:

                    st.warning(
                        "Resource Pressure: TRUE"
                    )

                else:

                    st.success(
                        "Resource Pressure: FALSE"
                    )

            st.subheader("Raw System Resource Values")

            st.json(resources)

            st.subheader("Raw Network Values")

            st.json(network)

        except Exception as error:

            st.error(
                f"Resource measurement failed: {error}"
            )

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

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EXPERIMENT RESULTS
# ============================================================

elif st.session_state.page == "Experiment Results":

    st.title("Experiment Results")

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

        benchmark = read_result_file(
            "model_benchmark"
        )

        if benchmark is None:

            st.info(
                "model_benchmark result file is not available."
            )

        else:

            st.dataframe(
                benchmark,
                use_container_width=True,
                hide_index=True
            )

            if {
                "Target",
                "Edge Accuracy",
                "Cloud Accuracy"
            }.issubset(benchmark.columns):

                plot_data = benchmark.melt(
                    id_vars="Target",
                    value_vars=[
                        "Edge Accuracy",
                        "Cloud Accuracy"
                    ],
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

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            if {
                "Target",
                "Edge F1",
                "Cloud F1"
            }.issubset(benchmark.columns):

                plot_data = benchmark.melt(
                    id_vars="Target",
                    value_vars=[
                        "Edge F1",
                        "Cloud F1"
                    ],
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

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    # --------------------------------------------------------
    # EDGE VS CLOUD
    # --------------------------------------------------------

    with experiment_tabs[1]:

        st.subheader("Edge Only Results")

        edge_results = read_result_file(
            "edge_only_results"
        )

        if edge_results is None:

            st.info(
                "edge_only_results file is not available."
            )

        else:

            st.dataframe(
                edge_results,
                use_container_width=True,
                hide_index=True
            )

        st.subheader("Cloud Only Results")

        cloud_results = read_result_file(
            "cloud_only_results"
        )

        if cloud_results is None:

            st.info(
                "cloud_only_results file is not available."
            )

        else:

            st.dataframe(
                cloud_results,
                use_container_width=True,
                hide_index=True
            )

            if {
                "Target",
                "Total Latency (ms)"
            }.issubset(cloud_results.columns):

                fig = px.bar(
                    cloud_results,
                    x="Target",
                    y="Total Latency (ms)",
                    title="Cloud Total Latency by Target"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    with experiment_tabs[2]:

        st.subheader("Hybrid Offloading Results")

        hybrid_results = read_result_file(
            "hybrid_results"
        )

        if hybrid_results is None:

            st.info(
                "hybrid_results file is not available."
            )

        else:

            st.dataframe(
                hybrid_results,
                use_container_width=True,
                hide_index=True
            )

            if {
                "Target",
                "Offload Percentage"
            }.issubset(hybrid_results.columns):

                fig = px.bar(
                    hybrid_results,
                    x="Target",
                    y="Offload Percentage",
                    title="Hybrid Cloud Offload Percentage"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    # --------------------------------------------------------
    # ADAPTIVE
    # --------------------------------------------------------

    with experiment_tabs[3]:

        st.subheader("Adaptive Offloading Results")

        adaptive_files = [
            (
                "adaptive_results",
                "Adaptive Network Scenarios"
            ),
            (
                "final_adaptive_results",
                "Final Adaptive Results"
            ),
            (
                "accuracy_first_adaptive_results",
                "Accuracy-First Adaptive Results"
            )
        ]

        for file_key, heading in adaptive_files:

            st.markdown(
                f"#### {heading}"
            )

            adaptive_df = read_result_file(
                file_key
            )

            if adaptive_df is None:

                st.info(
                    f"{file_key} file is not available."
                )

            else:

                st.dataframe(
                    adaptive_df,
                    use_container_width=True,
                    hide_index=True
                )

                if {
                    "Network",
                    "Offload Percentage"
                }.issubset(adaptive_df.columns):

                    fig = px.bar(
                        adaptive_df,
                        x="Network",
                        y="Offload Percentage",
                        color=(
                            "Target"
                            if "Target" in adaptive_df.columns
                            else None
                        ),
                        barmode="group",
                        title=f"{heading}: Offload Percentage"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

    # --------------------------------------------------------
    # DEADLINE
    # --------------------------------------------------------

    with experiment_tabs[4]:

        st.subheader("Deadline Experiment Results")

        deadline_results = read_result_file(
            "deadline_test_results"
        )

        if deadline_results is None:

            st.info(
                "deadline_test_results file is not available."
            )

        else:

            st.dataframe(
                deadline_results,
                use_container_width=True,
                hide_index=True
            )

            if {
                "Target",
                "Estimated Cloud Response (ms)"
            }.issubset(deadline_results.columns):

                fig = px.bar(
                    deadline_results,
                    x="Target",
                    y="Estimated Cloud Response (ms)",
                    title="Estimated Cloud Response Time"
                )

                fig.add_hline(
                    y=DEADLINE_MS,
                    line_dash="dash",
                    annotation_text=(
                        f"App deadline: {DEADLINE_MS} ms"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    # --------------------------------------------------------
    # RESOURCE PRESSURE
    # --------------------------------------------------------

    with experiment_tabs[5]:

        st.subheader("Resource Pressure Results")

        pressure_results = read_result_file(
            "resource_pressure_test_results"
        )

        if pressure_results is None:

            st.info(
                "resource_pressure_test_results was not found."
            )

        else:

            st.dataframe(
                pressure_results,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # SCENARIO
    # --------------------------------------------------------

    with experiment_tabs[6]:

        st.subheader("Scenario Comparison")

        scenario_results = read_result_file(
            "scenario_comparison"
        )

        if scenario_results is None:

            st.info(
                "scenario_comparison file is not available."
            )

        else:

            st.dataframe(
                scenario_results,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    with experiment_tabs[7]:

        st.subheader("Overall Strategy Performance")

        overall_results = read_result_file(
            "overall_strategy_summary"
        )

        if overall_results is None:

            st.info(
                "overall_strategy_summary file is not available."
            )

        else:

            st.dataframe(
                overall_results,
                use_container_width=True,
                hide_index=True
            )

            for metric in [
                "Accuracy",
                "F1 Score",
                "Average Latency (ms)"
            ]:

                if {
                    "Strategy",
                    metric
                }.issubset(overall_results.columns):

                    fig = px.bar(
                        overall_results,
                        x="Strategy",
                        y=metric,
                        title=f"{metric} by Strategy"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


# ============================================================
# EDGESIMPY SIMULATION
# ============================================================

elif st.session_state.page == "EdgeSimPy Simulation":

    st.title("EdgeSimPy Simulation Results")

    st.subheader(
        "Confidence-Based Edge–Cloud Patient Monitoring"
    )

    summary_path = os.path.join(
        EDGESIMPY_RESULTS_DIR,
        "edgesimpy_summary.csv"
    )

    predictions_path = os.path.join(
        EDGESIMPY_RESULTS_DIR,
        "edgesimpy_predictions.csv"
    )

    if not os.path.exists(summary_path):

        st.error(
            "EdgeSimPy summary CSV was not found."
        )

    else:

        summary = pd.read_csv(summary_path)

        predictions = None

        if os.path.exists(predictions_path):

            predictions = pd.read_csv(
                predictions_path
            )

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        accuracy = summary["accuracy_percent"].iloc[0]

        total_tasks = summary["total_tasks"].iloc[0]

        cloud_tasks = summary["cloud_tasks"].iloc[0]

        edge_tasks = summary["edge_tasks"].iloc[0]

        offload = summary[
            "cloud_offloading_percent"
        ].iloc[0]

        latency = summary[
            "average_latency_ms"
        ].iloc[0]

        runtime = summary[
            "total_simulation_time_minutes"
        ].iloc[0]

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        kpi1.metric(
            "Accuracy",
            f"{accuracy:.4f}%"
        )

        kpi2.metric(
            "Total Tasks",
            f"{total_tasks:,}"
        )

        kpi3.metric(
            "Cloud Offloading",
            f"{offload:.4f}%"
        )

        kpi4.metric(
            "Runtime",
            f"{runtime:.2f} min"
        )

        kpi5, kpi6, kpi7, kpi8 = st.columns(4)

        kpi5.metric(
            "Edge Tasks",
            f"{edge_tasks:,}"
        )

        kpi6.metric(
            "Cloud Tasks",
            f"{cloud_tasks:,}"
        )

        kpi7.metric(
            "Edge Fallback",
            f"{summary['edge_fallback_tasks'].iloc[0]:,}"
        )

        kpi8.metric(
            "Confidence Threshold",
            f"{summary['confidence_threshold'].iloc[0]:.2f}"
        )

        st.divider()

        # ----------------------------------------------------
        # EXECUTION DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("Execution Distribution")

        execution_df = pd.DataFrame({
            "Execution": [
                "Edge",
                "Edge Fallback",
                "Cloud"
            ],
            "Tasks": [
                summary["edge_tasks"].iloc[0],
                summary["edge_fallback_tasks"].iloc[0],
                summary["cloud_tasks"].iloc[0]
            ]
        })

        col1, col2 = st.columns(2)

        with col1:

            fig = px.pie(
                execution_df,
                names="Execution",
                values="Tasks",
                title="Task Distribution",
                hole=0.4
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.bar(
                execution_df,
                x="Execution",
                y="Tasks",
                text="Tasks",
                title="Task Count"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # ----------------------------------------------------
        # LATENCY
        # ----------------------------------------------------

        st.subheader("Latency Metrics")

        latency_df = pd.DataFrame({
            "Metric": [
                "Average Latency",
                "Average Edge Inference",
                "Average Cloud Inference",
                "Average Resource Check",
                "Average Network Delay"
            ],
            "Value (ms)": [
                summary["average_latency_ms"].iloc[0],
                summary["average_edge_inference_ms"].iloc[0],
                summary["average_cloud_inference_ms"].iloc[0],
                summary[
                    "average_resource_check_ms_all_tasks"
                ].iloc[0],
                summary["average_network_delay_ms"].iloc[0]
            ]
        })

        fig = px.bar(
            latency_df,
            x="Metric",
            y="Value (ms)",
            title="Latency Components",
            text="Value (ms)"
        )

        fig.update_layout(
            yaxis_type="log"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            latency_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------------------------
        # TARGET-WISE ACCURACY
        # ----------------------------------------------------

        st.subheader("Target-Wise Accuracy")

        if predictions is not None:

            target_accuracy = (
                predictions.groupby("target")["correct"]
                .mean()
                .reset_index()
            )

            target_accuracy["Accuracy (%)"] = (
                target_accuracy["correct"] * 100
            )

            fig = px.bar(
                target_accuracy,
                x="target",
                y="Accuracy (%)",
                title="Accuracy by Target"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                target_accuracy[
                    ["target", "Accuracy (%)"]
                ],
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader("Full EdgeSimPy Summary")

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download EdgeSimPy Summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="edgesimpy_summary.csv",
            mime="text/csv"
        )


# ============================================================
# COMPARISON RESULTS  ⭐ NEW PAGE
# ============================================================

elif st.session_state.page == "Comparison Results":

    st.title("System Comparison Results")

    st.subheader(
        "Custom Adaptive Offloading vs EdgeSimPy"
    )

    st.write(
        """
        This page compares the results of both completed experiments
        using the saved CSV files. No models are retrained and
        no simulations are rerun.
        """
    )

    system_df, target_df = load_comparison_data()

    if system_df is None:

        st.error(
            "system_comparison.csv was not found."
        )

        st.code(
            SYSTEM_COMPARISON_PATH
        )

        st.stop()

    if target_df is None:

        st.warning(
            "target_wise_comparison.csv was not found. "
            "System-level comparison will still be available."
        )

    # --------------------------------------------------------
    # SYSTEM VALUES
    # --------------------------------------------------------

    custom_accuracy = get_comparison_metric(
        system_df,
        "Accuracy (%)",
        "Custom Adaptive"
    )

    edgesimpy_accuracy = get_comparison_metric(
        system_df,
        "Accuracy (%)",
        "EdgeSimPy"
    )

    custom_f1 = get_comparison_metric(
        system_df,
        "Macro F1 Score (%)",
        "Custom Adaptive"
    )

    edgesimpy_f1 = get_comparison_metric(
        system_df,
        "Macro F1 Score (%)",
        "EdgeSimPy"
    )

    custom_latency = get_comparison_metric(
        system_df,
        "Average Latency (ms)",
        "Custom Adaptive"
    )

    edgesimpy_latency = get_comparison_metric(
        system_df,
        "Average Latency (ms)",
        "EdgeSimPy"
    )

    custom_offload = get_comparison_metric(
        system_df,
        "Cloud Offloading (%)",
        "Custom Adaptive"
    )

    edgesimpy_offload = get_comparison_metric(
        system_df,
        "Cloud Offloading (%)",
        "EdgeSimPy"
    )

    custom_edge = get_comparison_metric(
        system_df,
        "Total Edge Executions",
        "Custom Adaptive"
    )

    edgesimpy_edge = get_comparison_metric(
        system_df,
        "Total Edge Executions",
        "EdgeSimPy"
    )

    custom_cloud = get_comparison_metric(
        system_df,
        "Cloud Tasks",
        "Custom Adaptive"
    )

    edgesimpy_cloud = get_comparison_metric(
        system_df,
        "Cloud Tasks",
        "EdgeSimPy"
    )

    custom_adjusted_latency = get_comparison_metric(
        system_df,
        "Latency Excluding Resource Checks (ms)",
        "Custom Adaptive"
    )

    edgesimpy_adjusted_latency = get_comparison_metric(
        system_df,
        "Latency Excluding Resource Checks (ms)",
        "EdgeSimPy"
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.header("Overall Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Custom Accuracy",
        f"{custom_accuracy:.4f}%"
    )

    col2.metric(
        "EdgeSimPy Accuracy",
        f"{edgesimpy_accuracy:.4f}%"
    )

    col3.metric(
        "Custom Macro F1",
        f"{custom_f1:.4f}%"
    )

    col4.metric(
        "EdgeSimPy Macro F1",
        f"{edgesimpy_f1:.4f}%"
    )

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "Custom Latency",
        f"{custom_latency:.4f} ms"
    )

    col6.metric(
        "EdgeSimPy Latency",
        f"{edgesimpy_latency:.4f} ms"
    )

    col7.metric(
        "Custom Offloading",
        f"{custom_offload:.4f}%"
    )

    col8.metric(
        "EdgeSimPy Offloading",
        f"{edgesimpy_offload:.4f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # ACCURACY & F1 COMPARISON
    # --------------------------------------------------------

    st.header("1. Accuracy and F1-Score Comparison")

    performance_df = pd.DataFrame({
        "System": [
            "Custom Adaptive",
            "Custom Adaptive",
            "EdgeSimPy",
            "EdgeSimPy"
        ],
        "Metric": [
            "Accuracy",
            "Macro F1 Score",
            "Accuracy",
            "Macro F1 Score"
        ],
        "Value (%)": [
            custom_accuracy,
            custom_f1,
            edgesimpy_accuracy,
            edgesimpy_f1
        ]
    })

    fig = px.bar(
        performance_df,
        x="Metric",
        y="Value (%)",
        color="System",
        barmode="group",
        text="Value (%)",
        title="Accuracy and Macro F1-Score"
    )

    fig.update_traces(
        texttemplate="%{y:.4f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Score (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # LATENCY COMPARISON
    # --------------------------------------------------------

    st.header("2. Latency Comparison")

    latency_df = pd.DataFrame({
        "System": [
            "Custom Adaptive",
            "EdgeSimPy"
        ],
        "Reported Average Latency (ms)": [
            custom_latency,
            edgesimpy_latency
        ],
        "Latency Excluding Resource Checks (ms)": [
            custom_adjusted_latency,
            edgesimpy_adjusted_latency
        ]
    })

    latency_plot_df = latency_df.melt(
        id_vars="System",
        var_name="Latency Type",
        value_name="Latency (ms)"
    )

    fig = px.bar(
        latency_plot_df,
        x="System",
        y="Latency (ms)",
        color="Latency Type",
        barmode="group",
        title="Reported Latency vs Latency Excluding Resource Checks"
    )

    fig.update_layout(
        yaxis_title="Latency (ms)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        latency_df,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        """
        The reported average latencies are not directly equivalent
        unless resource-monitoring overhead is measured and included
        using the same method in both experiments.

        The adjusted latency is displayed separately to show the
        impact of resource-check overhead.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # EDGE VS CLOUD TASK DISTRIBUTION
    # --------------------------------------------------------

    st.header("3. Edge vs Cloud Task Distribution")

    custom_fallback = get_comparison_metric(
        system_df,
        "Edge Fallback Tasks",
        "Custom Adaptive",
        0
    )

    edgesimpy_fallback = get_comparison_metric(
        system_df,
        "Edge Fallback Tasks",
        "EdgeSimPy",
        0
    )

    execution_df = pd.DataFrame({
        "System": [
            "Custom Adaptive",
            "Custom Adaptive",
            "Custom Adaptive",
            "EdgeSimPy",
            "EdgeSimPy",
            "EdgeSimPy"
        ],
        "Execution Type": [
            "Edge Direct",
            "Edge Fallback",
            "Cloud",
            "Edge Direct",
            "Edge Fallback",
            "Cloud"
        ],
        "Tasks": [
            custom_edge - custom_fallback,
            custom_fallback,
            custom_cloud,
            edgesimpy_edge - edgesimpy_fallback,
            edgesimpy_fallback,
            edgesimpy_cloud
        ]
    })

    fig = px.bar(
        execution_df,
        x="System",
        y="Tasks",
        color="Execution Type",
        barmode="stack",
        title="Task Execution Distribution",
        text="Tasks"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        execution_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CLOUD OFFLOADING COMPARISON
    # --------------------------------------------------------

    st.header("4. Cloud Offloading Comparison")

    offloading_df = pd.DataFrame({
        "System": [
            "Custom Adaptive",
            "EdgeSimPy"
        ],
        "Cloud Offloading (%)": [
            custom_offload,
            edgesimpy_offload
        ]
    })

    fig = px.bar(
        offloading_df,
        x="System",
        y="Cloud Offloading (%)",
        text="Cloud Offloading (%)",
        title="Cloud Offloading Percentage"
    )

    fig.update_traces(
        texttemplate="%{y:.4f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        offloading_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # TARGET-WISE COMPARISON
    # --------------------------------------------------------

    st.header("5. Target-Wise Performance")

    if target_df is not None:

        if "System" in target_df.columns:

            target_df["System"] = (
                target_df["System"].astype(str)
            )

        # Accuracy chart
        if "Accuracy (%)" in target_df.columns:

            fig = px.bar(
                target_df,
                x="Target",
                y="Accuracy (%)",
                color="System",
                barmode="group",
                title="Target-Wise Accuracy Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # F1 chart
        if "F1 Score (%)" in target_df.columns:

            fig = px.bar(
                target_df,
                x="Target",
                y="F1 Score (%)",
                color="System",
                barmode="group",
                title="Target-Wise F1-Score Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Precision
        if "Precision (%)" in target_df.columns:

            fig = px.bar(
                target_df,
                x="Target",
                y="Precision (%)",
                color="System",
                barmode="group",
                title="Target-Wise Precision Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Recall
        if "Recall (%)" in target_df.columns:

            fig = px.bar(
                target_df,
                x="Target",
                y="Recall (%)",
                color="System",
                barmode="group",
                title="Target-Wise Recall Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Target-Wise Results Table")

        st.dataframe(
            target_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Target-wise comparison CSV is unavailable."
        )

    st.divider()

    # --------------------------------------------------------
    # RESOURCE CHECK OVERHEAD
    # --------------------------------------------------------

    st.header("6. Resource Monitoring Overhead")

    custom_resource_check = get_comparison_metric(
        system_df,
        "Average Resource Check (ms/task)",
        "Custom Adaptive"
    )

    edgesimpy_resource_check = get_comparison_metric(
        system_df,
        "Average Resource Check (ms/task)",
        "EdgeSimPy"
    )

    resource_df = pd.DataFrame({
        "System": [
            "Custom Adaptive",
            "EdgeSimPy"
        ],
        "Average Resource Check (ms)": [
            custom_resource_check,
            edgesimpy_resource_check
        ]
    })

    fig = px.bar(
        resource_df,
        x="System",
        y="Average Resource Check (ms)",
        title="Resource Monitoring Overhead"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        resource_df,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        """
        Resource-monitoring time depends on how each experiment
        measured and accumulated resource-check operations.
        Confirm that both systems use equivalent measurement
        definitions before treating these values as a direct
        benchmark.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # FINAL COMPARISON TABLE
    # --------------------------------------------------------

    st.header("7. Complete Comparison Table")

    st.dataframe(
        system_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download System Comparison CSV",
        data=system_df.to_csv(index=False).encode("utf-8"),
        file_name="system_comparison.csv",
        mime="text/csv"
    )

    if target_df is not None:

        st.download_button(
            "Download Target-Wise Comparison CSV",
            data=target_df.to_csv(index=False).encode("utf-8"),
            file_name="target_wise_comparison.csv",
            mime="text/csv"
        )

    st.success(
        "Comparison results loaded successfully from the saved CSV files."
    )

    st.caption(
        "All charts use saved experimental results. No simulations "
        "were rerun while displaying this page."
<<<<<<< HEAD
    )
=======
>>>>>>> 7692ee5 (Add confidence offloading and comparison experiments)
