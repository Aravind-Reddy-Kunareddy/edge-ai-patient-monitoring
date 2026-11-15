
import pandas as pd
import numpy as np
import time
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
edge_models = {}
edge_predictions = {}
edge_probabilities = {}
edge_confidences = {}


# 6. Train one Logistic Regression model per target
for target in targets:

    print("\nTraining:", target)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    start_time = time.perf_counter()

    model.fit(X_train_scaled, y_train[target])

    training_time = time.perf_counter() - start_time

    # Store trained model
    edge_models[target] = model

    # Predictions
    predictions = model.predict(X_test_scaled)

    # Prediction probabilities
    probabilities = model.predict_proba(X_test_scaled)

    positive_index = list(model.classes_).index(1)
    positive_probabilities = probabilities[:, positive_index]

    # Confidence in predicted class
    confidence = np.max(probabilities, axis=1)

    edge_predictions[target] = predictions
    edge_probabilities[target] = positive_probabilities
    edge_confidences[target] = confidence

    # Evaluation
    print("Training Time:", round(training_time, 4), "seconds")

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


# 7. Save trained models and scaler
os.makedirs("models", exist_ok=True)

joblib.dump(edge_models, "models/edge_models.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nAll four Edge Models trained successfully!")
print("Edge models and scaler saved successfully!")
