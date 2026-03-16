import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Dataset class distribution
# -----------------------------

df = pd.DataFrame({
    "Fish":        {"Train": 2778, "Val": 584,  "Test": 621},
    "Red Algae":   {"Train": 305,  "Val": 38,   "Test": 48},
    "Algae":       {"Train": 5254, "Val": 1120, "Test": 1005},
    "Hydroids":    {"Train": 1932, "Val": 337,  "Test": 436},
    "Cyanea Capillata":   {"Train": 27,   "Val": 23,   "Test": 10},
}).T

#df = pd.DataFrame({
   #"Fish":        {"Train": 0, "Val": 0,  "Test": 119},
   #"Red Algae":   {"Train": 0,  "Val": 0,   "Test": 69},
   #"Algae":       {"Train": 0, "Val": 0, "Test": 308},
   #"Hydroids":    {"Train": 0, "Val": 0,  "Test": 99},
   #"Jellyfish":   {"Train": 0,   "Val": 0,   "Test": 59},
#}).T

splits = ["Train", "Val", "Test"]
df = df[splits]  # reorder columns

# Colors for each split
colors = {
    "Train": "#0072B2",  # blue
    "Val":   "#D55E00",  # orange
    "Test":  "#009E73",  # green
}

# -----------------------------
# Plotting function
# -----------------------------
def stacked(ax, d, title, ymax=None):
    x = np.arange(len(d.index))
    bottom = np.zeros(len(d.index))

    # Stacked bars
    for split in splits:
        ax.bar(
            x,
            d[split].values,
            bottom=bottom,
            label=split,
            color=colors[split],
            edgecolor="black",
            linewidth=0.4
        )
        bottom += d[split].values

    # Total labels above bars
    totals = d.sum(axis=1).values
    for i, t in enumerate(totals):
        ax.text(i, t + max(totals) * 0.01, f"{int(t)}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(d.index, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Annotated Instances", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if ymax is not None:
        ax.set_ylim(0, ymax)

# -----------------------------
# Create figure
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

stacked(ax, df, "complete_dataset_1 Composition by Class and Split")

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper right", frameon=True)

plt.tight_layout()
plt.show()