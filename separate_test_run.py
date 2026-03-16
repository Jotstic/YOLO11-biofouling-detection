import os
import glob
import shutil
import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "/Users/jot/Documents/MASTEROPPGAVE/BEST WEIGHTS/yolo11s_Complete_dataset_1_best.pt"

DATASET_ROOT = "/Users/jot/Documents/MASTEROPPGAVE/datasets/test_dataset"
DATASET_YAML = f"{DATASET_ROOT}/dataset.yaml"

IMG_DIR = f"{DATASET_ROOT}/images/test"
GT_DIR  = f"{DATASET_ROOT}/labels/test"

# ✅ Put EVERYTHING under this root
RUNS_ROOT = "/Users/jot/Documents/MASTEROPPGAVE/Separate test runs"
RUN_NAME  = ""  # folder inside RUNS_ROOT

OUTPUT_ROOT = os.path.join(RUNS_ROOT, RUN_NAME)

PRED_DIR = os.path.join(OUTPUT_ROOT, "labels_pred")
COMP_DIR = os.path.join(OUTPUT_ROOT, "comparisons")

METRIC_CSV = os.path.join(OUTPUT_ROOT, "metrics.csv")
METRIC_TXT = os.path.join(OUTPUT_ROOT, "metrics.txt")
CLASS_COUNT_PATH = os.path.join(OUTPUT_ROOT, "class_counts.csv")
CLASS_METRIC_CSV = os.path.join(OUTPUT_ROOT, "metrics_per_class.csv")

IMGSZ = (640)
CONF_PRED_SAVE = 0.25  # set to e.g. 0.25 to reduce low-conf predictions in saved labels

# ============================================================
# UTILS
# ============================================================

def ensure_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"x Missing {name}: {path}")
    print(f"Found {name}: {path}")

def read_yaml_names_and_nc(yaml_path):
    with open(yaml_path, "r") as f:
        d = yaml.safe_load(f)

    names = d.get("names", {})
    nc = d.get("nc", None)

    # names can be list or dict
    if nc is None:
        if isinstance(names, dict) and len(names):
            nc = max(map(int, names.keys())) + 1
        elif isinstance(names, list):
            nc = len(names)
        else:
            nc = 0
    nc = int(nc)

    def name_of(i):
        if isinstance(names, dict):
            return str(names.get(i, f"class_{i}"))
        if isinstance(names, list):
            return str(names[i]) if i < len(names) else f"class_{i}"
        return f"class_{i}"

    return names, nc, name_of

def list_images(img_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(img_dir, e)))
    return sorted(files)

def load_counts(path):
    """return list of class ids from YOLO label file (first column)"""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if not p:
                continue
            out.append(int(float(p[0])))
    return out

def load_boxes(path, pred=False):
    """Return list of (cls, x, y, w, h, conf) in normalized YOLO format."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if not p:
                continue
            cls = int(float(p[0]))
            x, y, w, h = map(float, p[1:5])
            confv = None
            if pred and len(p) >= 6:
                try:
                    confv = float(p[5])
                except Exception:
                    confv = None
            out.append((cls, x, y, w, h, confv))
    return out

def draw_boxes(img, boxes, color, show_conf=False):
    H, W = img.shape[:2]
    for cls, x, y, w, h, conf in boxes:
        x1 = int((x - w/2) * W)
        y1 = int((y - h/2) * H)
        x2 = int((x + w/2) * W)
        y2 = int((y + h/2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if show_conf and conf is not None:
            cv2.putText(img, f"{conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# ============================================================
# MAIN
# ============================================================

def main():
    # ---- folders ----
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(COMP_DIR, exist_ok=True)

    # ---- structure check ----
    ensure_exists(DATASET_YAML, "dataset.yaml")
    ensure_exists(IMG_DIR, "test images folder")
    ensure_exists(GT_DIR, "test labels folder")

    names, nc, name_of = read_yaml_names_and_nc(DATASET_YAML)
    print(f"\nDataset classes (nc={nc}):")
    for i in range(nc):
        print(f"  {i}: {name_of(i)}")
    print()

    # ---- load model ----
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print(" Model loaded.\n")

    # ---- predict (save txt) ----
    print("Running prediction on test images...\n")

    # clear old pred txts to avoid stale files
    for f in glob.glob(os.path.join(PRED_DIR, "*.txt")):
        os.remove(f)

    predict_kwargs = dict(
        source=IMG_DIR,
        save_txt=True,
        save_conf=True,
        imgsz=IMGSZ,
        project=OUTPUT_ROOT,
        name="predictions",
        exist_ok=True
    )
    if CONF_PRED_SAVE is not None:
        predict_kwargs["conf"] = float(CONF_PRED_SAVE)

    model.predict(**predict_kwargs)

    pred_labels_dir = os.path.join(OUTPUT_ROOT, "predictions", "labels")

    if os.path.exists(pred_labels_dir):
        for f in os.listdir(pred_labels_dir):
            src = os.path.join(pred_labels_dir, f)
            dst = os.path.join(PRED_DIR, f)
            if os.path.isfile(src):
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

    print(f"Prediction labels saved to: {PRED_DIR}\n")

    # ---- evaluate ----
    print("Evaluating model...\n")
    val_results = model.val(
        data=DATASET_YAML,
        split="test",
        project=OUTPUT_ROOT,
        name="eval",
        exist_ok=True
    )

    # Overall metrics: (P, R, mAP50, mAP50-95)
    precision, recall, map50, map5095 = map(float, val_results.mean_results())
    metric_dict = {
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50-95": map5095,
    }
    pd.DataFrame([metric_dict]).to_csv(METRIC_CSV, index=False)

    # Per-class metrics: always write all classes 0..nc-1
    per_class = []
    for cls_id in range(nc):
        cls_name = name_of(cls_id)
        try:
            p, r, ap50, ap5095 = val_results.class_result(cls_id)
            # NaN -> 0.0
            p = 0.0 if p != p else float(p)
            r = 0.0 if r != r else float(r)
            ap50 = 0.0 if ap50 != ap50 else float(ap50)
            ap5095 = 0.0 if ap5095 != ap5095 else float(ap5095)
        except Exception:
            p = r = ap50 = ap5095 = 0.0

        per_class.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "precision": p,
            "recall": r,
            "mAP50": ap50,
            "mAP50-95": ap5095,
        })

    pd.DataFrame(per_class).to_csv(CLASS_METRIC_CSV, index=False)

    with open(METRIC_TXT, "w") as f:
        f.write("=== MODEL EVALUATION METRICS ===\n")
        for k, v in metric_dict.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\n=== PER-CLASS METRICS ===\n")
        for row in per_class:
            f.write(
                f"\nClass {row['class_id']} - {row['class_name']}:\n"
                f"  Precision:  {row['precision']:.4f}\n"
                f"  Recall:     {row['recall']:.4f}\n"
                f"  mAP50:      {row['mAP50']:.4f}\n"
                f"  mAP50-95:   {row['mAP50-95']:.4f}\n"
            )

    print("Metrics saved:")
    print("  -", METRIC_CSV)
    print("  -", CLASS_METRIC_CSV)
    print("  -", METRIC_TXT)
    print()

    # ---- count instances ----
    print("Counting GT vs Pred instances...\n")

    gt_counts = {i: 0 for i in range(nc)}
    pred_counts = {i: 0 for i in range(nc)}

    image_list = list_images(IMG_DIR)
    for img_path in image_list:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt_file = os.path.join(GT_DIR, stem + ".txt")
        pr_file = os.path.join(PRED_DIR, stem + ".txt")

        for cls in load_counts(gt_file):
            if 0 <= cls < nc:
                gt_counts[cls] += 1

        for cls in load_counts(pr_file):
            if 0 <= cls < nc:
                pred_counts[cls] += 1

    df_counts = pd.DataFrame({
        "class_id": list(range(nc)),
        "class_name": [name_of(i) for i in range(nc)],
        "gt_count": [gt_counts[i] for i in range(nc)],
        "pred_count": [pred_counts[i] for i in range(nc)],
    })
    df_counts.to_csv(CLASS_COUNT_PATH, index=False)
    print("Counts saved to:", CLASS_COUNT_PATH, "\n")

    # ---- comparison images ----
    print("Creating comparison images (GT left / Pred right)...\n")

    for img_path in image_list:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            continue

        gt = load_boxes(os.path.join(GT_DIR, stem + ".txt"), pred=False)
        pr = load_boxes(os.path.join(PRED_DIR, stem + ".txt"), pred=True)

        left = draw_boxes(img.copy(), gt, (0, 255, 0), show_conf=False)
        right = draw_boxes(img.copy(), pr, (0, 0, 255), show_conf=True)

        combined = np.hstack([left, right])
        cv2.imwrite(os.path.join(COMP_DIR, f"{stem}_compare.jpg"), combined)

    print("Comparison images saved to:", COMP_DIR)
    print("\n### FULL EVALUATION COMPLETE ###")
    print("All outputs under:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
