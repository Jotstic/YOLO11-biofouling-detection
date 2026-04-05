#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("map50", "mAP50"),
    ("map50_95", "mAP50-95"),
]


def clean_class_name(s: str) -> str:
    """Keep class labels clean and consistent."""
    if not isinstance(s, str):
        return str(s)
    t = s.strip()

    
    replacements = {
        "Cyanea": "Cyanea",   
        "hydroid": "Hydroids",
        "hydroids": "Hydroids",
        "fish": "Fish",
        "algae": "Algae",
       "red algae": "Red Algae",
        "green algae": "Green Algae",
       "brown algae": "Brown Algae",
        "jellyfish": "Cyanea",
    }
    return replacements.get(t.lower(), t)


def plot_from_csv(
    csv_path: Path,
    out_path: Path,
    ylim: tuple[float, float] = (0.0, 1.0),
    title: str = "Image size variation in detection performance of YOLO11s on test split in complete_dataset_1 (95% CI)",
):
    df = pd.read_csv(csv_path)

    # Figure out class column name
    if "imgsz" in df.columns:
        class_col = "imgsz"
        df["imgsz"] = df["imgsz"].astype(str)
    elif "class_name" in df.columns:
        class_col = "class_name"
    elif "class" in df.columns:
        class_col = "class"
    elif "model" in df.columns:
        # fallback if your CSV accidentally used 'model' for class names
        class_col = "model"
    else:
        raise ValueError(
            f"Could not find class label column. Expected one of: imgsz, class_name, class, model.\n"
            f"Found: {list(df.columns)}"
        )

    # Required metric columns
    required = [class_col]
    for m, _ in METRICS:
        required += [m, f"{m}_ci_low", f"{m}_ci_high"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}\nFound: {list(df.columns)}")

    # Sort classes in a sensible order (customize here)
    def class_order_key(name: str):
        n = str(name).strip().lower()
        order = {
            "fish": 0,
            "red algae": 1,
            "green algae": 2,
            "brown algae": 3,
            "algae": 1,       # for dataset_algae (single algae class)
            "Cyanea": 4,
            "cnidaria": 4,
            "sea anemones": 4,
            "hydroids": 5,
            "bivalves": 6,
        }
        return order.get(n, 999), n

    df["class_label"] = df[class_col].apply(clean_class_name)
    if class_col == "imgsz":
        df["class_label"] = df["imgsz"].astype(int).astype(str)
        df = df.sort_values(by="imgsz").reset_index(drop=True)
    else:
        MODEL_ORDER = ["YOLO11n", "YOLO11s", "YOLO11m", "YOLO11l", "YOLO11x"]
        df["class_label"] = pd.Categorical(df["class_label"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("class_label").reset_index(drop=True)



    classes = df["class_label"].tolist()
    n_classes = len(classes)
    n_metrics = len(METRICS)

    means = np.zeros((n_classes, n_metrics), dtype=float)
    lows = np.zeros((n_classes, n_metrics), dtype=float)
    highs = np.zeros((n_classes, n_metrics), dtype=float)

    for j, (m, _) in enumerate(METRICS):
        means[:, j] = df[m].astype(float).to_numpy()
        lows[:, j] = df[f"{m}_ci_low"].astype(float).to_numpy()
        highs[:, j] = df[f"{m}_ci_high"].astype(float).to_numpy()

    
    yerr = np.stack([means - lows, highs - means], axis=0)

    x = np.arange(n_classes)
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(9, 1.8 * n_classes), 5.4))

    for j, (_, label) in enumerate(METRICS):
        xpos = x + (j - (n_metrics - 1) / 2) * width

        ax.bar(
            xpos,
            means[:, j],
            width,
            yerr=yerr[:, :, j],
            capsize=4,
            label=label,
        )

        # Optional: vertical score labels inside bars
        for i in range(n_classes):
            mean_val = float(means[i, j])
            hi = float(highs[i, j])

            # place in lower/mid part of bar to avoid CI overlap
            y_text = max(0.02, mean_val * 0.20)

            # If bar is extremely small, place just above the bar (but under error cap if possible)
            if mean_val < 0.10:
                y_text = min(mean_val + 0.01, hi - 0.01 if hi > mean_val else mean_val + 0.01)
                y_text = max(y_text, 0.02)
                


    SHOW_BAR_LABELS = True   # ← set to False to hide labels
    if SHOW_BAR_LABELS:
        for j, (_, _) in enumerate(METRICS):
            xpos = x + (j - (n_metrics - 1) / 2) * width
            for i in range(n_classes):
                mean_val = float(means[i, j])
                ax.text(
                xpos[i],
                mean_val / 2,            # vertically centred in the bar
                f"{mean_val:.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
                fontweight="bold",
                rotation=90,
            )
# END BAR LABELS SNIPPET

            

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Mean Detection Score")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_all_imgsz_ci.csv", help="Path to per-class CI CSV")
    ap.add_argument("--out", default="metrics_with_ci.png", help="Output image path (.png or .pdf)")
    ap.add_argument("--title", default="Mean detection performance of YOLO11s on test split in complete_dataset_1 with different image sizes (95% CI)")
    ap.add_argument("--ymax", type=float, default=1.0)
    args = ap.parse_args()

    plot_from_csv(
        csv_path=Path(args.csv).expanduser().resolve(),
        out_path=Path(args.out).expanduser().resolve(),
        ylim=(0.0, float(args.ymax)),
        title=args.title,
    )
    print(f"Saved plot: {Path(args.out).expanduser().resolve()}")


if __name__ == "__main__":
    main()
