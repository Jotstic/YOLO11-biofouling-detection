#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("map50", "AP50"),
    ("map50_95", "AP50-95"),
]


def norm(s: str) -> str:
    return str(s).strip().lower()


def detect_dataset_tag(csv_path: Path) -> str:
    """
    Guess dataset type from filename.
    You can also override by renaming files consistently.
    """
    name = csv_path.name.lower()
    if "algae" in name and "complete_dataset_1" not in name and "complete_dataset" not in name:
        return "dataset_algae"          # all algae merged
    if "complete_dataset_1" in name:
        return "complete_dataset_1"     # brown+green merged, red separate
    if "complete_dataset" in name:
        return "complete_dataset"       # red, brown, green separate
    return csv_path.stem.lower()


def pick_algae_rows(df: pd.DataFrame, dataset_tag: str) -> list[dict]:
    """
    Returns rows for algae-related classes with a clear x-axis label.
    Expects CSV columns:
      class_name, precision, recall, map50, map50_95, *_ci_low, *_ci_high
    """
    if "class_name" not in df.columns:
        raise ValueError(f"CSV missing 'class_name' column. Found: {list(df.columns)}")

    rows_out = []
    df = df.copy()
    df["_class_norm"] = df["class_name"].astype(str).map(norm)

    # dataset_algae: one merged algae class ("Algae")
    if dataset_tag == "dataset_algae":
        mask = df["_class_norm"].isin(["algae"])
        for _, r in df[mask].iterrows():
            rows_out.append(make_row(r, "Algae (all merged)"))

    # complete_dataset_1: likely has "Red Algae" + merged "Algae" (brown+green)
    elif dataset_tag == "complete_dataset_1":
        # merged brown+green class often named "Algae"
        mask_merged = df["_class_norm"].isin(["algae"])
        for _, r in df[mask_merged].iterrows():
            rows_out.append(make_row(r, "Algae (brown+green merged)"))

        # red algae separate
        mask_red = df["_class_norm"].isin(["red algae", "red_algae"])
        for _, r in df[mask_red].iterrows():
            rows_out.append(make_row(r, "Red algae"))

    # complete_dataset: red, brown, green separate
    elif dataset_tag == "complete_dataset":
        wanted = {
            "green algae": "Green algae",
            "brown algae": "Brown algae",
        }
        for key, label in wanted.items():
            mask = df["_class_norm"] == key
            for _, r in df[mask].iterrows():
                rows_out.append(make_row(r, label))

    else:
        # fallback: pick any algae-named classes
        algae_like = ["algae", "red algae", "green algae", "brown algae"]
        for _, r in df[df["_class_norm"].isin(algae_like)].iterrows():
            label = str(r["class_name"])
            rows_out.append(make_row(r, label))

    return rows_out


def make_row(r: pd.Series, label: str) -> dict:
    out = {"label": label}
    for m, _ in METRICS:
        out[m] = float(r[m])
        out[f"{m}_ci_low"] = float(r[f"{m}_ci_low"])
        out[f"{m}_ci_high"] = float(r[f"{m}_ci_high"])
    return out


def plot_rows(rows: list[dict], out_path: Path, title: str, ymax: float):
    if not rows:
        raise ValueError("No algae rows found to plot.")

    # Fixed order so figure is consistent
    desired_order = {
        "Algae (all merged)": 0,
        "Algae (brown+green merged)": 1,
        "Red algae": 2,
        "Brown algae": 3,
        "Green algae": 4,
    }
    rows = sorted(rows, key=lambda r: desired_order.get(r["label"], 999))

    labels = [r["label"] for r in rows]
    n_groups = len(rows)
    n_metrics = len(METRICS)

    means = np.zeros((n_groups, n_metrics), dtype=float)
    lows = np.zeros((n_groups, n_metrics), dtype=float)
    highs = np.zeros((n_groups, n_metrics), dtype=float)

    for j, (m, _) in enumerate(METRICS):
        means[:, j] = [r[m] for r in rows]
        lows[:, j] = [r[f"{m}_ci_low"] for r in rows]
        highs[:, j] = [r[f"{m}_ci_high"] for r in rows]

    yerr = np.stack([means - lows, highs - means], axis=0)

    x = np.arange(n_groups)
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(10, 1.9 * n_groups), 5.4))

    for j, (_, metric_label) in enumerate(METRICS):
        xpos = x + (j - (n_metrics - 1) / 2) * width
        ax.bar(
            xpos,
            means[:, j],
            width,
            yerr=yerr[:, :, j],
            capsize=4,
            label=metric_label
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Detection score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs=3, required=True,
                    help="Three per-class CI CSVs: algae_per_class_ci.csv, complete_dataset_1_per_class_ci.csv, complete_dataset_per_class_ci.csv")
    ap.add_argument("--out", default="algae_comparison_ci.png")
    ap.add_argument("--selected_csv", default="algae_comparison_selected_rows.csv",
                    help="Optional: save the exact selected rows used in the plot")
    ap.add_argument("--title", default="Algae class comparison across datasets (YOLO11s, 95% CI)")
    ap.add_argument("--ymax", type=float, default=1.0)
    args = ap.parse_args()

    all_rows = []

    for csv_str in args.csvs:
        csv_path = Path(csv_str).expanduser().resolve()
        df = pd.read_csv(csv_path)
        tag = detect_dataset_tag(csv_path)
        picked = pick_algae_rows(df, tag)
        all_rows.extend(picked)

    # Save selected rows for transparency/debugging
    pd.DataFrame(all_rows).to_csv(Path(args.selected_csv).expanduser().resolve(), index=False)

    plot_rows(
        rows=all_rows,
        out_path=Path(args.out).expanduser().resolve(),
        title=args.title,
        ymax=float(args.ymax),
    )

    print(f"Saved plot: {Path(args.out).expanduser().resolve()}")
    print(f"Saved selected rows: {Path(args.selected_csv).expanduser().resolve()}")


if __name__ == "__main__":
    main()
