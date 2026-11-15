
import pandas as pd
import plotly.express as px

# Load results from previous experiments
hybrid = pd.read_csv("results/hybrid_results.csv")
normal = pd.read_csv("results/final_adaptive_results.csv")
pressure = pd.read_csv("results/resource_pressure_test_results.csv")
deadline = pd.read_csv("results/deadline_test_results.csv")

# Prepare scenario comparison
hybrid_data = hybrid[["Target", "Offload Percentage"]].copy()
hybrid_data["Scenario"] = "Hybrid Baseline"

normal_data = normal[["Target", "Offload Percentage"]].copy()
normal_data["Scenario"] = "Adaptive - Normal"

pressure_data = pressure[["Target", "Offload Percentage"]].copy()
pressure_data["Scenario"] = "Adaptive - Resource Pressure"

deadline_data = deadline[["Target", "Offload Percentage"]].copy()
deadline_data["Scenario"] = "Adaptive - Deadline Exceeded"

# Combine results
comparison = pd.concat(
    [hybrid_data, normal_data, pressure_data, deadline_data],
    ignore_index=True
)

# Display results
print("\n========== SCENARIO COMPARISON ==========")
print(
    comparison.pivot(
        index="Target",
        columns="Scenario",
        values="Offload Percentage"
    ).round(3).to_string()
)

# Plot interactive chart
fig = px.bar(
    comparison,
    x="Target",
    y="Offload Percentage",
    color="Scenario",
    barmode="group",
    text_auto=".2f",
    title="Cloud Offloading Under Different Operating Conditions",
    labels={
        "Offload Percentage": "Cloud Offloading (%)",
        "Target": "Prediction Target"
    }
)

fig.update_layout(
    template="plotly_white",
    height=650,
    hovermode="x unified"
)

fig.show()

fig.write_html(
    "results/scenario_comparison.html"
)

comparison.to_csv(
    "results/scenario_comparison.csv",
    index=False
)

print("\nScenario comparison saved successfully!")
