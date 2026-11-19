
import joblib

files = [
    "models/edge_models.pkl",
    "models/cloud_models.pkl",
    "models/scaler.pkl",
    "models/cloud_scaler.pkl"
]

for file in files:
    obj = joblib.load(file)

    print("\nFile:", file)
    print("Type:", type(obj))

    if isinstance(obj, dict):
        print("Keys:", list(obj.keys()))

        for key, model in obj.items():
            print(
                "Target:", key,
                "| Model:", type(model).__name__,
                "| Features:", getattr(model, "n_features_in_", None)
            )
    else:
        print("Features:", getattr(obj, "n_features_in_", None))
