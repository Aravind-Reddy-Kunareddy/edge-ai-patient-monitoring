
import time

from load_project import (
    df,
    FEATURES,
    TARGETS,
    edge_models,
    cloud_models,
    edge_scaler,
    cloud_scaler
)


def execute_model(sample_index, target, execution_type):

    # Select model and scaler
    if execution_type == "edge":
        model = edge_models[target]
        scaler = edge_scaler

    elif execution_type == "cloud":
        model = cloud_models[target]
        scaler = cloud_scaler

    else:
        raise ValueError("execution_type must be 'edge' or 'cloud'")

    # Get sensor input
    sample = df.loc[[sample_index], FEATURES]

    # Scale input
    scaled_sample = scaler.transform(sample)

    # Measure inference time
    start_time = time.perf_counter()

    prediction = model.predict(scaled_sample)[0]

    # Calculate prediction confidence
    probabilities = model.predict_proba(scaled_sample)[0]
    confidence = max(probabilities)

    end_time = time.perf_counter()

    # Actual label
    actual = df.loc[sample_index, target]

    # Return results
    return {
        "sample_index": sample_index,
        "target": target,
        "execution": execution_type,
        "prediction": int(prediction),
        "confidence": float(confidence),
        "actual": int(actual),
        "inference_time_ms": (end_time - start_time) * 1000
    }


if __name__ == "__main__":

    # Test Edge model
    result = execute_model(
        sample_index=0,
        target=TARGETS[0],
        execution_type="edge"
    )

    print("\nEDGE MODEL TEST")
    print("=" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")
