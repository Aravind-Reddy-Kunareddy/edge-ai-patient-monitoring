# Resource-Aware Intelligent Offloading for Edge AI

## Project Overview

This project implements a resource-aware intelligent Edge–Cloud offloading system for patient monitoring using machine learning.

The system analyzes patient sensor data and decides whether a prediction should be processed locally on an edge device or offloaded to the cloud. The decision is based on prediction confidence, CPU utilization, available RAM, network bandwidth, and network latency.

The main objective is to reduce unnecessary cloud communication, maintain prediction accuracy, and efficiently utilize computational and network resources.

The project includes machine learning models, multiple offloading strategies, resource monitoring, simulation experiments, and an interactive Streamlit dashboard for performance analysis.

## Objectives

- Develop a lightweight machine learning model suitable for edge devices.
- Develop a more complex cloud-side prediction model.
- Implement Edge-Only, Cloud-Only, Hybrid, and Adaptive Offloading strategies.
- Monitor system and network resources dynamically.
- Make intelligent offloading decisions using prediction confidence and resource availability.
- Evaluate the performance of different offloading strategies.
- Simulate edge-cloud infrastructure using EdgeSimPy.
- Analyze system performance through an interactive dashboard.

## System Architecture

1. Patient sensor data is provided as input.
2. Data preprocessing prepares the sensor features.
3. The edge model generates a prediction and confidence score.
4. If the prediction confidence is sufficiently high, the edge prediction is returned.
5. If confidence is low, system and network resources are monitored.
6. CPU utilization, available RAM, network bandwidth, and latency are evaluated.
7. If cloud resources and network conditions satisfy the configured requirements, the task is offloaded to the cloud model.
8. If cloud resources are unavailable, the edge prediction is retained as a fallback.
9. Predictions, execution decisions, and performance metrics are recorded for analysis.

## Dataset

The project uses a patient-monitoring dataset containing **10,000 records** and 23 sensor features.

### Input Features

- Heart rate
- SpO2
- PPG red and infrared signals
- Pulse interval
- ECG signal
- RR interval
- Heart rate variability
- QRS duration
- Accelerometer readings
- Gyroscope readings
- Acceleration magnitude
- Peak acceleration
- Body temperature
- Temperature variation
- Respiratory rate
- Respiratory signal
- Breathing interval

### Prediction Targets

The system predicts four health-related events:

| Target | Description |
|---|---|
| Fall Detection | Detects potential fall events |
| Abnormal Heart Rhythm | Identifies abnormal rhythm patterns |
| Hypoxemia Event | Detects low oxygen-related events |
| Fever Event | Detects fever-related events |

The system is designed for experimental evaluation and is not intended for clinical diagnosis.

## Machine Learning Models

| Component | Model |
|---|---|
| Edge Model | Logistic Regression |
| Cloud Model | Multi-Layer Perceptron (MLP) |

The Edge model provides lightweight local inference, while the Cloud model represents a more complex model for processing uncertain predictions.

## Offloading Strategies

### 1. Edge-Only

All prediction tasks are processed using the Edge model.

### 2. Cloud-Only

All prediction tasks are processed using the Cloud model.

### 3. Hybrid Offloading

Tasks are distributed between Edge and Cloud based on the configured hybrid strategy.

### 4. Adaptive Offloading

The adaptive strategy evaluates Edge prediction confidence and resource availability.

- High-confidence predictions are processed directly at the Edge.
- Low-confidence predictions trigger resource monitoring.
- Cloud offloading is performed when the required resource and network conditions are satisfied.
- Edge fallback is used when Cloud offloading is unavailable.

## Resource Monitoring

The system monitors the following resources:

| Resource | Purpose |
|---|---|
| CPU Utilization | Evaluates computational load |
| Available RAM | Checks memory availability |
| Upload Bandwidth | Determines whether data can be transmitted efficiently |
| Download Bandwidth | Evaluates response transfer capability |
| Network Latency | Estimates communication delay |

### Adaptive Offloading Conditions

The experimental configuration uses the following thresholds:

| Parameter | Threshold |
|---|---:|
| Edge confidence threshold | 90% |
| Minimum upload bandwidth | 3 Mbps |
| Minimum download bandwidth | 2 Mbps |
| Maximum network latency | 100 ms |
| Maximum CPU utilization | 90% |
| Minimum available RAM | 750 MB |

These thresholds can be configured to evaluate different operating conditions.

## Simulation Experiments

The project contains two simulation components.

### 1. EdgeSimPy Infrastructure Simulation

EdgeSimPy is used to explore edge computing infrastructure and simulation workflows.

The simulation environment includes experimentation with edge-cloud infrastructure, resource allocation, and simulation monitoring.

The EdgeSimPy experiment files and notebooks are maintained under:

`simulations/edgesimpy/`

### 2. Patient Monitoring Offloading Simulation

A custom Python simulation evaluates the confidence-based Edge–Cloud offloading strategy using the patient-monitoring dataset.

The experiment processes:

- 10,000 samples
- 4 prediction targets
- 40,000 total prediction tasks

For each task, the system evaluates prediction confidence and determines the execution location.

The simulation records prediction accuracy, execution distribution, inference time, resource-check overhead, network delay, and total simulation time.

## Final Simulation Results

The completed confidence-based offloading experiment achieved the following results:

| Metric | Result |
|---|---:|
| Total Samples | 10,000 |
| Total Prediction Tasks | 40,000 |
| Overall Accuracy | 99.7775% |
| Confidence Threshold | 90% |
| Edge Executions | 39,135 |
| Cloud Executions | 696 |
| Edge Fallback Executions | 169 |
| Cloud Offloading Rate | 1.74% |
| Average Latency | 59.1844 ms |
| Average Edge Inference Time | 0.5461 ms |
| Average Cloud Inference Time | 0.0115 ms |
| Average Resource-Check Overhead | 58.2577 ms |
| Average Network Delay | 0.3691 ms |
| Total Simulation Time | 42.71 minutes |

### Execution Distribution

| Execution Mode | Tasks | Percentage |
|---|---:|---:|
| Edge | 39,135 | 97.8375% |
| Cloud | 696 | 1.7400% |
| Edge Fallback | 169 | 0.4225% |

The system processed most tasks locally at the Edge. Only 1.74% of tasks were offloaded to Cloud, while tasks that could not be offloaded were handled using Edge fallback.

### Target-Wise Accuracy

| Prediction Target | Accuracy |
|---|---:|
| Abnormal Heart Rhythm | 99.46% |
| Fall Detection | 99.97% |
| Fever Event | 99.81% |
| Hypoxemia Event | 99.87% |

## Performance Evaluation

The project evaluates offloading strategies using:

- Accuracy
- Precision
- Recall
- F1-score
- Average inference time
- Average latency
- Cloud offloading percentage
- Edge and Cloud execution distribution
- Resource-check overhead
- Network delay
- Total simulation runtime

## Interactive Dashboard

The project includes a Streamlit dashboard for monitoring and analyzing the Edge AI system.

The dashboard provides:

- Project Overview
- Prediction Interface
- Strategy Comparison
- Resource Monitoring
- Experiment Results
- Simulation Result Visualizations

Simulation visualizations include execution distribution, latency analysis, resource-check overhead, prediction accuracy, and offloading statistics.

## Project Structure

    EDGE_AI/
    │
    ├── app.py
    ├── adaptive_offloading.py
    ├── edge_model.py
    ├── cloud_model.py
    ├── network_monitor.py
    ├── resource_monitor.py
    ├── system_monitor.py
    │
    ├── edge_only.py
    ├── cloud_only.py
    ├── hybrid_offloading.py
    ├── compare_strategies.py
    ├── strategy_comparison.py
    ├── strategy_summary.py
    ├── overall_performance.py
    ├── scenario_comparison.py
    ├── model_benchmark.py
    ├── inspect_models.py
    │
    ├── data/
    ├── models/
    ├── results/
    │
    ├── simulations/
    │   ├── edgesimpy/
    │   └── patient_monitoring/
    │       ├── load_project.py
    │       ├── model_executor.py
    │       ├── resource_aware_policy.py
    │       ├── run_confidence_simulation.py
    │       ├── check_resources.py
    │       ├── check_components.py
    │       ├── create_network.py
    │       ├── create_workload.py
    │       ├── simulation_conditions.py
    │       ├── patient_workload.csv
    │       └── final_results/
    │
    ├── requirements.txt
    └── README.md

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit
- Psutil
- Joblib
- EdgeSimPy

## Project Limitations

- Cloud model execution is simulated locally and does not currently use a physically deployed remote cloud server.
- Network bandwidth is measured using external network testing services.
- Cloud communication and processing delays are simulated or estimated rather than measured from a real cloud deployment.
- The dataset is intended for experimental evaluation and is not intended for real-world medical diagnosis.
- Simulation results may vary depending on the available system resources and network conditions.

## Future Improvements

- Deploy the Cloud model on a remote server.
- Integrate real-time patient sensor data.
- Implement real Edge–Cloud communication.
- Improve network delay and resource prediction.
- Test on physical edge hardware.
- Evaluate under additional network and resource conditions.
- Extend the system to additional patient-monitoring tasks.

## Author

Developed as an academic Edge AI and Computer Networks project.
