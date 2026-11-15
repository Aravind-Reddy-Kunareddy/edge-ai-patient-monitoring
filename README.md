# Resource-Aware Intelligent Offloading for Edge AI

## Project Overview

This project implements an intelligent edge-cloud offloading system for patient monitoring using machine learning.

The system analyzes patient sensor data and decides whether a prediction should be processed locally at the edge or offloaded to the cloud. The decision considers prediction confidence, CPU usage, available RAM, network conditions, and response deadlines.

The main objective is to reduce unnecessary cloud communication while maintaining prediction performance and meeting latency requirements.

## Objectives

- Develop a lightweight machine learning model for edge devices.
- Develop a more powerful cloud-side prediction model.
- Implement Edge-Only, Cloud-Only, Hybrid, and Adaptive Offloading strategies.
- Monitor available system and network resources.
- Make offloading decisions based on prediction confidence and resource availability.
- Compare the performance of different offloading strategies.

## System Architecture

1. Patient sensor data is provided as input.
2. Data preprocessing prepares the sensor features.
3. The edge model generates a prediction and confidence score.
4. If confidence is high, the edge prediction is returned.
5. If confidence is low, the system checks CPU usage and available RAM.
6. The system estimates network and cloud processing delay.
7. If resources are available and the deadline can be met, the request is offloaded to the cloud.
8. Otherwise, the edge prediction is retained.
9. The final prediction and performance metrics are recorded.

## Dataset

The project uses patient-monitoring sensor data containing 10,000 records.

The dataset includes 23 sensor features, such as:

- Heart rate
- SpO2
- ECG signal
- PPG signals
- Respiratory rate
- Body temperature
- Accelerometer readings
- Gyroscope readings
- Heart rate variability

### Prediction Targets

The system predicts four events:

- Fall Detection
- Abnormal Heart Rhythm
- Hypoxemia Event
- Fever Event

## Machine Learning Models

| Component | Model |
|---|---|
| Edge Model | Logistic Regression |
| Cloud Model | Multi-Layer Perceptron (MLP) |

The edge model is designed to provide lightweight local predictions, while the cloud model represents a more complex prediction model.

## Offloading Strategies

### 1. Edge Only
All predictions are processed using the edge model.

### 2. Cloud Only
All predictions are processed using the cloud model.

### 3. Hybrid Offloading
Requests are routed between edge and cloud based on the configured hybrid strategy.

### 4. Adaptive Offloading
The system first checks edge prediction confidence. Uncertain predictions may be offloaded when system resources and estimated network delay allow it.

If resources are insufficient or the response deadline cannot be met, the edge prediction is retained.

## Resource Monitoring

The system monitors:

- CPU utilization
- Available RAM
- Network upload bandwidth
- Network download bandwidth
- Network latency

These measurements help the adaptive strategy make resource-aware decisions.

## Experimental Scenarios

The project evaluates different operating conditions:

- Normal network and resource conditions
- Resource pressure
- Network deadline exceeded
- Comparison of different offloading strategies

Experiment results and visualizations are stored in the `results/` directory.

## Performance Evaluation

The strategies are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Average latency
- Cloud offloading percentage

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Psutil
- Joblib

## Project Limitations

- The cloud execution is simulated locally; the project does not use a physically deployed remote cloud server.
- Network conditions are measured using an internet speed-test service, and cloud response latency is estimated.
- The dataset is used for experimental evaluation and is not intended for real-world medical diagnosis.

## Future Improvements

- Deploy the cloud model on a remote server.
- Integrate real-time patient sensor data.
- Improve network delay estimation.
- Test the system on actual edge hardware.
- Evaluate performance under additional network conditions.

## Author

Developed as an academic Edge AI and Computer Networks project.