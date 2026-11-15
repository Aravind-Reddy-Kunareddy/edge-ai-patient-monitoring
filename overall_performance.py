import pandas as pd
import matplotlib.pyplot as plt

# Load overall strategy summary
df = pd.read_csv("results/overall_strategy_summary.csv")

metrics = [
    ("Accuracy", "Accuracy Comparison"),
    ("F1 Score", "F1-Score Comparison"),
    ("Average Latency (ms)", "Average Latency Comparison")
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (column, title) in zip(axes, metrics):
    ax.bar(df["Strategy"], df[column])
    ax.set_title(title)
    ax.set_ylabel(column)
    ax.tick_params(axis="x", rotation=25)

    for i, value in enumerate(df[column]):
        ax.text(i, value, f"{value:.4f}",
                ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("results/overall_performance.png", dpi=300)
plt.show()

print("Overall performance graph saved successfully!")