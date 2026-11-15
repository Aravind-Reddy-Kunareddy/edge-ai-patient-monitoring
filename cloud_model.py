
import pandas as pd
import numpy as np
import time
import os
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import GroupShuffleSplit


# 1. Define target labels
targets = [
    "fall_detected",
    "abnormal_heart_rhythm",
    "hypoxemia_event",
    "fever_event"
]


# 2. Load dataset
df = pd.read_csv("data.csv")

# Separate features and targets
X = df.drop(columns=[
    "patient_id",
    "room_number"
] + targets)

y = df[targets]

# Patient IDs are used only for grouping
groups = df["patient_id"]


# 3. Patient-wise train-test split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# 4. Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed!")
print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)


# 5. Create dictionaries
cloud_models = {}

cloud_predictions = {}
cloud_probabilities = {}
cloud_confidences = {}

cloud_training_times = {}


# 6. Train one Neural Network per target
for target in targets:

    print("\nTraining Cloud Model:", target)

    X_target = X_train_scaled
    y_target = y_train[target].to_numpy()

    # Separate positive and negative samples
    positive_idx = np.where(y_target == 1)[0]
    negative_idx = np.where(y_target == 0)[0]

    # Oversample minority class
    positive_oversampled = resample(
        positive_idx,
        replace=True,
        n_samples=len(negative_idx),
        random_state=42
    )

    # Combine balanced indices
    balanced_idx = np.concatenate([
        negative_idx,
        positive_oversampled
    ])

    np.random.RandomState(42).shuffle(balanced_idx)

    X_balanced = X_target[balanced_idx]
    y_balanced = y_target[balanced_idx]

    # Create Neural Network
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42
    )

    # Train model
    start_time = time.perf_counter()

    model.fit(X_balanced, y_balanced)

    training_time = time.perf_counter() - start_time

    cloud_training_times[target] = training_time

    # Store trained model
    cloud_models[target] = model

    # Predictions on original test data
    predictions = model.predict(X_test_scaled)

    probabilities = model.predict_proba(X_test_scaled)

    positive_index = list(model.classes_).index(1)

    positive_prob = probabilities[:, positive_index]

    confidence = np.max(probabilities, axis=1)

    cloud_predictions[target] = predictions
    cloud_probabilities[target] = positive_prob
    cloud_confidences[target] = confidence

    # Evaluation
    print("Training Time:",
          round(training_time, 4), "seconds")

    print("Iterations:", model.n_iter_)

    print("Accuracy:", round(
        accuracy_score(y_test[target], predictions), 4
    ))

    print("Precision:", round(
        precision_score(y_test[target], predictions, zero_division=0), 4
    ))

    print("Recall:", round(
        recall_score(y_test[target], predictions, zero_division=0), 4
    ))

    print("F1 Score:", round(
        f1_score(y_test[target], predictions, zero_division=0), 4
    ))

    print("\nClassification Report:")
    print(classification_report(
        y_test[target],
        predictions,
        zero_division=0
    ))


# 7. Save trained Cloud Neural Networks
os.makedirs("models", exist_ok=True)

joblib.dump(cloud_models, "models/cloud_models.pkl")

# Save the scaler used by the Cloud models
joblib.dump(scaler, "models/cloud_scaler.pkl")

print("\nAll four Cloud Neural Networks trained successfully!")
print("Cloud models and scaler saved successfully!")
