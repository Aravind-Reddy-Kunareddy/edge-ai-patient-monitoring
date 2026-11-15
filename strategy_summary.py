
import pandas as pd
import plotly.express as px

# Load comparison results
df = pd.read_csv("results/strategy_comparison.csv")

# Calculate average metrics for each strategy
summary = (
    df.groupby("Strategy")
    .agg({
        "Accuracy": "mean",
        "F1 Score": "mean",
        "Average Latency (ms)": "mean",
        "Offload Percentage": "mean"
    })
    .reset_index()
)

# Display summary
print("\n========== OVERALL STRATEGY SUMMARY ==========")
print(summary.round(4).to_string(index=False))

# Save summary
summary.to_csv("results/overall_strategy_summary.csv", index=False)

# Plot average F1 score
fig = px.bar(
    summary,
    x="Strategy",
    y="F1 Score",
    color="Strategy",
    text_auto=".4f",
    title="Average F1 Score Across All Prediction Targets",
    labels={"F1 Score": "Mean F1 Score"}
)

fig.update_layout(
    template="plotly_white",
    showlegend=False,
    yaxis_range=[0.85, 1.0]
)

fig.show()

# Plot average offloading percentage
fig2 = px.bar(
    summary,
    x="Strategy",
    y="Offload Percentage",
    color="Strategy",
    text_auto=".2f",
    title="Average Cloud Offloading Percentage",
    labels={"Offload Percentage": "Mean Offloading (%)"}
)

fig2.update_layout(
    template="plotly_white",
    showlegend=False
)

fig2.show()

print("\nSummary saved to results/overall_strategy_summary.csv")
