
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load comparison results
df = pd.read_csv("results/strategy_comparison.csv")

strategies = [
    "Edge Only",
    "Cloud Only",
    "Hybrid",
    "Adaptive"
]

targets = [
    "fall_detected",
    "abnormal_heart_rhythm",
    "hypoxemia_event",
    "fever_event"
]

# Create 2x2 dashboard
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Accuracy Comparison",
        "F1 Score Comparison",
        "Average Latency Comparison",
        "Cloud Offloading Percentage"
    ),
    vertical_spacing=0.15,
    horizontal_spacing=0.12
)

metrics = [
    ("Accuracy", 1, 1),
    ("F1 Score", 1, 2),
    ("Average Latency (ms)", 2, 1),
    ("Offload Percentage", 2, 2)
]

# Add grouped bars for each strategy
for metric, row, col in metrics:

    for strategy in strategies:

        strategy_data = df[df["Strategy"] == strategy]

        strategy_data = (
            strategy_data.set_index("Target")
            .reindex(targets)
            .reset_index()
        )

        fig.add_trace(
            go.Bar(
                x=strategy_data["Target"],
                y=strategy_data[metric],
                name=strategy,
                legendgroup=strategy,
                showlegend=(row == 1 and col == 1),
                hovertemplate=(
                    "Target: %{x}<br>"
                    + metric + ": %{y:.4f}"
                    + "<extra>" + strategy + "</extra>"
                )
            ),
            row=row,
            col=col
        )

# Group bars side by side
fig.update_layout(
    title={
        "text": "Edge, Cloud, Hybrid and Adaptive Strategy Comparison",
        "x": 0.5
    },
    barmode="group",
    height=850,
    width=1400,
    template="plotly_white",
    legend_title="Strategy",
    hovermode="x unified"
)

# Focus the metric axes for better visibility
fig.update_yaxes(
    title_text="Accuracy",
    range=[0.975, 1.005],
    row=1,
    col=1
)

fig.update_yaxes(
    title_text="F1 Score",
    range=[0.88, 1.005],
    row=1,
    col=2
)

# Log scale because cloud-only latency is much higher
fig.update_yaxes(
    title_text="Latency (ms, logarithmic scale)",
    type="log",
    row=2,
    col=1
)

fig.update_yaxes(
    title_text="Offloading (%)",
    range=[0, 110],
    row=2,
    col=2
)

fig.update_xaxes(title_text="Prediction Target")

# Display dashboard
fig.show()

# Save interactive dashboard
fig.write_html("results/strategy_comparison_dashboard.html")

print("\nDashboard saved successfully!")
