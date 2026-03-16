#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import yaml
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METRIC_KEYS = ["precision", "recall", "map50", "map50_95"]


# -----------------------------
# Paths / YAML
# -----------------------------

def resolve_weights_path(p: Path) -> Path:
    """
    Accepts:
      - direct .pt file
      - directory containing weights/best.pt or weights/last.pt
    """
    p = p.expanduser().resolve()
    if p.is_file() and p.suffix.lower() == ".pt":
        return p
    for name in ("best.pt", "last.pt"):
        cand = p / "weights" / name
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Weights not found at '{p}' (expected .pt or <dir>/weights/best.pt|last.pt)"
    )


def make_abs_path(base: Path, val):
    if val is None:
        return None
    valp = Path(val)
    return str(valp if valp.is_absolute() else (base / valp).resolve())


def resolve_dataset_yaml_abs(data_yaml: Path, outdir: Path) -> Path:
    """
    Writes resolved dataset yaml with absolute paths for train/val/test and removes 'path'.
    Includes a fallback if YAML 'path' is wrong.
    """
    data_yaml = data_yaml.expanduser().resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)

    base = Path(ds.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    # Fallback: if YAML path is invalid, use folder containing dataset.yaml
    if not base.exists():
        print(f"[warn] YAML path does not exist: {base}")
        print(f"[warn] Falling back to dataset.yaml folder: {data_yaml.parent}")
        base = data_yaml.parent.resolve()

    for key in ("train", "val", "test"):
        if key in ds and ds[key] is not None:
            ds[key] = make_abs_path(base, ds[key])

    ds.pop("path", None)

    # Ensure nc exists and normalize names
    if "nc" not in ds and "names" in ds:
        names = ds["names"]
        if isinstance(names, dict):
            ds["nc"] = len(names)
            ds["names"] = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        elif isinstance(names, (list, tuple)):
            ds["nc"] = len(names)

    outdir.mkdir(parents=True, exist_ok=True)
    dataset_id = data_yaml.parent.name
    resolved = outdir / f"{dataset_id}_resolved.yaml"
    with resolved.open("w", encoding="utf-8") as f:
        yaml.safe_dump(ds, f, sort_keys=False)

    return resolved


def read_class_names(resolved_yaml: Path) -> list[str]:
    with resolved_yaml.open("r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)

    names = ds.get("names", None)
    if names is None:
        raise ValueError(f"No 'names' found in YAML: {resolved_yaml}")

    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, (list, tuple)):
        return list(names)

    raise ValueError(f"Unsupported names format in {resolved_yaml}: {type(names)}")


def collect_test_images(resolved_yaml: Path) -> list[Path]:
    with resolved_yaml.open("r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)

    test_entry = ds.get("test", None)
    if not test_entry:
        raise ValueError("dataset yaml has no 'test' entry.")

    test_path = Path(str(test_entry)).expanduser().resolve()

    # test can be txt file listing image paths
    if test_path.is_file() and test_path.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in test_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        imgs = []
        for ln in lines:
            p = Path(ln).expanduser()
            p = p if p.is_absolute() else (test_path.parent / p)
            p = p.resolve()
            if p.exists() and p.suffix.lower() in IMG_EXTS:
                imgs.append(p)
        imgs.sort()
        return imgs

    # test can be folder
    if test_path.is_dir():
        imgs = [p for p in test_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
        imgs.sort()
        return imgs

    # test can be single image
    if test_path.is_file() and test_path.suffix.lower() in IMG_EXTS:
        return [test_path]

    raise FileNotFoundError(f"Could not interpret test path: {test_path}")


def write_boot_yaml(base_yaml: Path, out_yaml: Path, test_list_txt: Path):
    with base_yaml.open("r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)
    ds["test"] = str(test_list_txt.resolve())
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(ds, f, sort_keys=False)


# -----------------------------
# Metrics extraction (per-class)
# -----------------------------

def _safe_to_1d(arr_like):
    """Convert to flat float numpy array, or empty array if missing."""
    if arr_like is None:
        return np.array([], dtype=float)
    arr = np.asarray(arr_like, dtype=float)
    return arr.reshape(-1)


def _fit_len(arr: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Make array length exactly n_classes (pad with NaN or trim).
    This avoids IndexError if Ultralytics returns fewer values than class count.
    """
    out = np.full(n_classes, np.nan, dtype=float)
    n = min(n_classes, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


def extract_per_class_metrics_ultralytics(metrics_obj, n_classes: int) -> dict:
    """
    Extract per-class metrics:
      precision, recall, mAP50, mAP50-95
    from Ultralytics metrics object in a robust way.
    """
    box = getattr(metrics_obj, "box", None)
    if box is None:
        raise RuntimeError("Could not find metrics_obj.box. Ultralytics version may differ.")

    # Common ultralytics attributes
    p = _safe_to_1d(getattr(box, "p", None))       # precision per class
    r = _safe_to_1d(getattr(box, "r", None))       # recall per class
    ap50 = _safe_to_1d(getattr(box, "ap50", None)) # AP50 per class
    ap = _safe_to_1d(getattr(box, "ap", None))     # AP50-95 per class

    # Some versions may store AP arrays on mp/mr/map fields differently; we keep this strict and safe.
    p = _fit_len(p, n_classes)
    r = _fit_len(r, n_classes)
    ap50 = _fit_len(ap50, n_classes)
    ap = _fit_len(ap, n_classes)

    return {
        "precision": p.tolist(),
        "recall": r.tolist(),
        "map50": ap50.tolist(),
        "map50_95": ap.tolist(),
        "raw_results_dict": getattr(metrics_obj, "results_dict", {}) or {},
    }


def percentile_ci_nan(vals: list[float], alpha: float = 0.05) -> tuple[float, float]:
    arr = np.array(vals, dtype=float)
    # ignore NaNs (some classes may be absent in a bootstrap sample)
    lo = float(np.nanpercentile(arr, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(arr, 100 * (1 - alpha / 2)))
    return lo, hi


# -----------------------------
# Bootstrap CI (per class)
# -----------------------------

def bootstrap_ci_per_class(
    model: YOLO,
    resolved_yaml: Path,
    test_images: list[Path],
    n_classes: int,
    outdir: Path,
    n_boot: int,
    seed: int,
    imgsz: int,
    conf: float,
    iou: float,
    batch: int,
    device: str,
    workers: int,
    project: str,
    run_name: str,
) -> dict:
    """
    Returns:
      {
        metric_name: {
          class_idx: {"low": ..., "high": ...}
        }
      }
    """
    rng = random.Random(seed)
    tmp = outdir / "tmp_boot"
    tmp.mkdir(parents=True, exist_ok=True)

    # store bootstrap reps per metric per class
    reps = {
        k: [[] for _ in range(n_classes)]
        for k in METRIC_KEYS
    }

    for b in range(n_boot):
        sampled = [rng.choice(test_images) for _ in range(len(test_images))]

        test_list_txt = tmp / f"{run_name}_test_boot_{b:04d}.txt"
        test_list_txt.write_text("\n".join(str(p) for p in sampled) + "\n", encoding="utf-8")

        boot_yaml = tmp / f"{run_name}_data_boot_{b:04d}.yaml"
        write_boot_yaml(resolved_yaml, boot_yaml, test_list_txt)

        m = model.val(
            data=str(boot_yaml),
            split="test",
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            batch=batch,
            device=device,
            workers=workers,
            verbose=False,
            cache=False,  # more stable for repeated evals
            project=project,
            name=f"{run_name}_boot",
            exist_ok=True,
        )

        per = extract_per_class_metrics_ultralytics(m, n_classes=n_classes)

        for k in METRIC_KEYS:
            vals = per[k]
            for ci in range(n_classes):
                v = vals[ci] if ci < len(vals) else np.nan
                reps[k][ci].append(float(v))

    ci_out = {}
    for k in METRIC_KEYS:
        ci_out[k] = {}
        for ci in range(n_classes):
            lo, hi = percentile_ci_nan(reps[k][ci], alpha=0.05)
            ci_out[k][ci] = {"low": lo, "high": hi}

    return ci_out


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dataset.yaml (ONE dataset)")
    ap.add_argument("--weights", required=True, help="YOLO11s weights (.pt or run dir)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=100, help="Bootstrap replicates (0 disables CI)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--csv_name", default="results_per_class_ci.csv")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    weights = resolve_weights_path(Path(args.weights))
    model = YOLO(str(weights))

    resolved_yaml = resolve_dataset_yaml_abs(Path(args.data), outdir)
    class_names = read_class_names(resolved_yaml)
    n_classes = len(class_names)

    test_images = collect_test_images(resolved_yaml)
    if not test_images:
        print(f"[error] No test images found from yaml: {resolved_yaml}")
        sys.exit(2)

    dataset_id = Path(args.data).expanduser().resolve().parent.name
    run_name = f"{dataset_id}_YOLO11s"

    print(f"Resolved yaml: {resolved_yaml}")
    print(f"Dataset: {dataset_id}")
    print(f"Test images: {len(test_images)}")
    print(f"Classes: {n_classes}")
    print(f"Weights: {weights}")

    # Point estimate (one run)
    m_full = model.val(
        data=str(resolved_yaml),
        split="test",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        verbose=False,
        cache=False,
        project=str(outdir),
        name=f"{run_name}_point",
        exist_ok=True,
    )
    per_full = extract_per_class_metrics_ultralytics(m_full, n_classes=n_classes)

    # Save raw dict for debugging
    raw_path = outdir / f"raw_results_dict_{run_name}.json"
    raw_path.write_text(json.dumps(per_full["raw_results_dict"], indent=2), encoding="utf-8")

    # Bootstrap CI (optional)
    if args.n_boot > 0:
        ci = bootstrap_ci_per_class(
            model=model,
            resolved_yaml=resolved_yaml,
            test_images=test_images,
            n_classes=n_classes,
            outdir=outdir,
            n_boot=args.n_boot,
            seed=args.seed,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(outdir),
            run_name=run_name,
        )
    else:
        ci = {
            k: {i: {"low": float("nan"), "high": float("nan")} for i in range(n_classes)}
            for k in METRIC_KEYS
        }

    # Build rows (one row per class)
    rows = []
    for i, cname in enumerate(class_names):
        row = {
            "dataset": dataset_id,
            "class_id": i,
            "class_name": cname,
            "model": "YOLO11s",
            "weights": str(weights),
        }
        for k in METRIC_KEYS:
            row[k] = float(per_full[k][i])
            row[f"{k}_ci_low"] = float(ci[k][i]["low"])
            row[f"{k}_ci_high"] = float(ci[k][i]["high"])
        rows.append(row)

    # Write CSV
    csv_path = outdir / args.csv_name
    fieldnames = [
        "dataset", "class_id", "class_name", "model", "weights",
        "precision", "precision_ci_low", "precision_ci_high",
        "recall", "recall_ci_low", "recall_ci_high",
        "map50", "map50_ci_low", "map50_ci_high",
        "map50_95", "map50_95_ci_low", "map50_95_ci_high",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\nSaved CSV:", csv_path)
    print("Rows written:", len(rows))


if __name__ == "__main__":
    main()
