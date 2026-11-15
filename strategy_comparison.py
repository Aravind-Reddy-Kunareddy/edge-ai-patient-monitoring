
import pandas as pd
import plotly.express as px

# Load strategy comparison results
df = pd.read_csv("results/strategy_comparison.csv")

# Create separate graphs for each metric
metrics = [
    ("Accuracy", "Accuracy Comparison"),
    ("F1 Score", "F1 Score Comparison"),
    ("Average Latency (ms)", "Average Latency Comparison"),
    ("Offload Percentage", "Offload Percentage Comparison")
]

for column, title in metrics:
    fig = px.bar(
        df,
        x="Target",
        y=column,
        color="Strategy",
        barmode="group",
        title=title,
        text_auto=".3f"
    )

    fig.update_layout(
        xaxis_title="Prediction Target",
        yaxis_title=column,
        hovermode="x unified"
    )

    fig.show()
