"""
Export a unified YOLO dataset from one or more BIIGLE volumes.

What I want this script to do (my own reminder notes):

- Talk to BIIGLE using my username + API token and pull out all images
  from one or more volumes.
- Build ONE big dataset with a global train/val/test split (not per volume),
  so it behaves like a proper ML dataset.
- Convert BIIGLE rectangle annotations to YOLO (cx,cy,w,h) format,
  using a clean mapping from BIIGLE label IDs to class indices.
- Optionally drop images with no annotations (KEEP_EMPTY=0) so I can
  control whether I want background-only images or not.
- Save:
    - images/train, images/val, images/test
    - labels/train, labels/val, labels/test
    - dataset.yaml that Ultralytics YOLO understands
    - stats.json with some basic sanity info
    - problems.tsv with whatever went wrong per image
    - image_manifest.csv so I can trace what went where later
"""

import os
import io
import json
import math
import random
import pathlib
import csv
from typing import Dict, List, Tuple, Any
import requests
from PIL import Image



# Config / environment bits
# These environment variables are what I tweak from outside when I run the script

BASE_URL = os.getenv("BIIGLE_BASE_URL", "https://biigle.de").rstrip("/")

USERNAME = os.getenv("mail")
API_TOKEN = os.getenv("python")

# New: allow multiple volumes via BIIGLE_VOLUME_IDS="123,456,789"
VOLUME_IDS_STR = os.getenv("BIIGLE_VOLUME_IDS", "").strip()
SINGLE_VOLUME_ID = int(os.getenv("BIIGLE_VOLUME_ID", "ID"))

# Where I want the final YOLO dataset to end up
OUT_DIR = pathlib.Path(os.getenv("OUT_DIR", "out_path_to_folder"))

# Fractions for global split
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "0.7"))   # default 70% train
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.15"))      # default 15% val
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))    # seed so the split is reproducible

EMPTY_KEEP_FRAC = float(os.getenv("EMPTY_KEEP_FRAC", "1.0"))  # fraction of empty images to keep (if KEEP_EMPTY=1)

if not (0.0 <= EMPTY_KEEP_FRAC <= 1.0):
    raise SystemExit("ERROR: EMPTY_KEEP_FRAC must be between 0.0 and 1.0.")


# Keep or drop images that have *zero* annotations

KEEP_EMPTY = os.getenv("KEEP_EMPTY", "1").lower() not in ("0", "false", "no")

# Optional global cap on number of images to process (0 means unlimited)

MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))

# Default mapping (can be overridden with LABEL_MAP_JSON)
# This is: BIIGLE label_id -> human-readable class name

DEFAULT_LABEL_ID_TO_NAME: Dict[int, str] = {
    475638: "Fish",
    475635: "Red Algae",
    475636: "Green Algae",
    475637: "Brown Algae",
    475639: "Jellyfish",
    475640: "Hydroids",
    475641: "Barnacles",
    475642: "Bivalves",
}

LABEL_MAP_JSON = os.getenv("LABEL_MAP_JSON", "").strip()
if LABEL_MAP_JSON:
    # If I want to override label mapping from outside (nice for experiments)
    try:
        tmp = json.loads(LABEL_MAP_JSON)
        LABEL_ID_TO_NAME = {int(k): str(v) for k, v in tmp.items()}
    except Exception as e:
        raise SystemExit(f"Failed to parse LABEL_MAP_JSON: {e}")
else:
    LABEL_ID_TO_NAME = DEFAULT_LABEL_ID_TO_NAME.copy()

# Classes in the fixed index order, and mapping from BIIGLE label_id -> class index
CLASSES: List[str] = list(LABEL_ID_TO_NAME.values())
LABEL_ID_TO_IDX: Dict[int, int] = {lid: i for i, lid in enumerate(LABEL_ID_TO_NAME.keys())}


# Quick sanity checks on credentials and dataset split config
if not USERNAME or not API_TOKEN:
    raise SystemExit("ERROR: Set BIIGLE_USERNAME and BIIGLE_API_TOKEN environment variables.")

# Decide which volumes to use
def _resolve_volume_ids() -> List[int]:
    """Figure out which BIIGLE volumes I'm actually exporting from."""
    if VOLUME_IDS_STR:
        # BIIGLE_VOLUME_IDS="123,456" → [123, 456]
        out = []
        for part in VOLUME_IDS_STR.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
        if not out:
            raise SystemExit("ERROR: BIIGLE_VOLUME_IDS is set but no valid volume IDs were parsed.")
        return out
    else:
        # Fallback: single volume
        if not SINGLE_VOLUME_ID:
            raise SystemExit("ERROR: Set either BIIGLE_VOLUME_IDS or BIIGLE_VOLUME_ID to a valid volume id.")
        return [SINGLE_VOLUME_ID]


VOLUME_IDS = _resolve_volume_ids()

if TRAIN_FRAC <= 0 or VAL_FRAC < 0 or TRAIN_FRAC + VAL_FRAC >= 1:
    raise SystemExit("ERROR: Bad TRAIN_FRAC/VAL_FRAC values. Need 0 < TRAIN_FRAC, 0 <= VAL_FRAC, and TRAIN_FRAC+VAL_FRAC < 1.")

TEST_FRAC = 1.0 - TRAIN_FRAC - VAL_FRAC



# BIIGLE API helper section

# (These are the little helper functions I use to talk to BIIGLE)

session = requests.Session()
session.auth = (USERNAME, API_TOKEN)
session.headers.update({"Accept": "application/json"})


def biigle_get(path: str, params: dict | None = None, stream: bool = False) -> requests.Response:
    """Tiny helper: do a GET request against BIIGLE and fail fast if something is wrong."""
    url = f"{BASE_URL}{path}"
    r = session.get(url, params=params, timeout=90, stream=stream)
    r.raise_for_status()
    return r


def fetch_volume_image_ids(volume_id: int) -> List[int]:
    """
    Get all image IDs for a given volume.

    BIIGLE might return either:
    - a plain list
    - or a paginated JSON object with "data" and "links"
    so I handle both shapes here.
    """
    ids: List[int] = []

    r = biigle_get(f"/api/v1/volumes/{volume_id}/files")
    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from /volumes/{volume_id}/files: {e}")

    # Simple list case
    if isinstance(data, list):
        for x in data:
            if isinstance(x, int):
                ids.append(x)
            elif isinstance(x, dict):
                ids.append(int(x.get("id") or x.get("image_id") or x.get("file_id")))
        return ids

    # Paginated shape case
    def extract_ids(items: List[Any]) -> List[int]:
        out: List[int] = []
        for x in items:
            if isinstance(x, int):
                out.append(x)
            else:
                out.append(int(x.get("id") or x.get("image_id") or x.get("file_id")))
        return out

    if isinstance(data, dict) and "data" in data:
        ids.extend(extract_ids(data["data"]))
        page = 2
        while True:
            r = biigle_get(f"/api/v1/volumes/{volume_id}/files", params={"page": page})
            d = r.json()
            items = d.get("data") if isinstance(d, dict) else None
            if not items:
                break
            ids.extend(extract_ids(items))
            page += 1
        return ids

    raise RuntimeError(f"Unexpected response shape for /volumes/{volume_id}/files: {type(data)}")


def download_image(image_id: int) -> Image.Image:
    """Download a single BIIGLE image and decode it into a PIL image."""
    r = biigle_get(f"/api/v1/images/{image_id}/file", stream=True)
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img


def list_annotations(image_id: int) -> List[dict]:
    """Fetch all annotations for a given image_id."""
    r = biigle_get(f"/api/v1/images/{image_id}/annotations")
    return r.json()


# =========================
# Geometry / YOLO helpers
# =========================

def rect_to_aabb(points: List[float]) -> Tuple[float, float, float, float]:
    """Convert a BIIGLE polygon rectangle (list of x,y,x,y,...) into (x1,y1,x2,y2)."""
    xs = points[0::2]
    ys = points[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def to_float_list(seq) -> List[float]:
    """Cast a sequence to a list of floats; if anything fails, return [] so I can skip it cleanly."""
    out: List[float] = []
    for v in seq:
        try:
            out.append(float(v))
        except Exception:
            return []
    return out


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float, float, float, float]:
    """Convert absolute box corners to normalized YOLO (cx,cy,w,h)."""
    cx = ((x1 + x2) / 2.0) / float(W)
    cy = ((y1 + y2) / 2.0) / float(H)
    bw = (x2 - x1) / float(W)
    bh = (y2 - y1) / float(H)
    return cx, cy, bw, bh


def clamp_box(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """
    Clamp the YOLO box to [0,1] and make sure it doesn't drift outside image
    if the center is too close to the border.
    """
    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    w = min(1.0, max(1e-6, w))
    h = min(1.0, max(1e-6, h))

    if cx - w / 2 < 0:
        cx = w / 2
    if cy - h / 2 < 0:
        cy = h / 2
    if cx + w / 2 > 1:
        cx = 1 - w / 2
    if cy + h / 2 > 1:
        cy = 1 - h / 2

    return cx, cy, w, h


# =========================
# Dataset directory helpers
# =========================

def ensure_output_directories() -> None:
    """Make sure the YOLO directory structure exists (images/labels for train/val/test)."""
    for split_name in ("train", "val", "test"):
        (OUT_DIR / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split_name).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml() -> None:
    """
    Write a dataset.yaml that Ultralytics understands.

    This is the file I will point YOLO to later.
    """
    yaml_lines = [
        f"path: {OUT_DIR.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for i, name in enumerate(CLASSES):
        yaml_lines.append(f"  {i}: {name}")
    (OUT_DIR / "dataset.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")


# =========================
# Manifest / splitting logic
# =========================

def collect_all_images() -> List[dict]:
    """
    Collect all image IDs across all configured volumes into a single manifest list.

    Each entry in the manifest is a dict so I can attach extra info later:
    {
        "volume_id": int,
        "image_id": int,
        "split": "train"/"val"/"test" (set later),
        "num_annotations": int (set later),
    }
    """
    manifest: List[dict] = []
    for vid in VOLUME_IDS:
        ids = fetch_volume_image_ids(vid)
        for img_id in ids:
            manifest.append(
                {
                    "volume_id": vid,
                    "image_id": img_id,
                    "split": None,
                    "num_annotations": None,
                }
            )

    # Apply global cap if requested
    if MAX_IMAGES > 0 and len(manifest) > MAX_IMAGES:
        manifest = manifest[:MAX_IMAGES]

    return manifest


def assign_global_splits(manifest: List[dict]) -> None:
    """
    Do a global train/val/test split on the manifest.

    Right now it's a simple random split over all images; if I later want to make
    it grouped by volume/sequence, I can change the logic here.
    """
    random.seed(RANDOM_SEED)
    random.shuffle(manifest)

    n = len(manifest)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    # Whatever is left goes to test
    n_test = n - n_train - n_val

    for i, entry in enumerate(manifest):
        if i < n_train:
            entry["split"] = "train"
        elif i < n_train + n_val:
            entry["split"] = "val"
        else:
            entry["split"] = "test"

    # Just so I can see this when the script runs
    print(f"Total images: {n}")
    print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")
    print(f"Fractions: train={TRAIN_FRAC:.3f}, val={VAL_FRAC:.3f}, test={TEST_FRAC:.3f}")


# =========================
# Core export logic
# =========================

def export_dataset(manifest: List[dict]) -> None:
    """Main loop that actually downloads images, fetches annotations, and writes YOLO files."""
    ensure_output_directories()

    problems_path = OUT_DIR / "problems.tsv"
    prob_f = open(problems_path, "w", newline="", encoding="utf-8")
    prob_writer = csv.writer(prob_f, delimiter="\t")
    prob_writer.writerow(["image_id", "volume_id", "split", "issue", "details"])

    # Stats that I want at the end
    stats: Dict[str, Any] = {
        "written_images": 0,
        "written_boxes": 0,
        "empty_images": 0,
        "skipped_images": 0,
        "unknown_label_ids": {},
        "bad_shapes": 0,
        "images_per_split": {"train": 0, "val": 0, "test": 0},
    }

    # I also want a manifest with paths/split so I can inspect later
    manifest_out_path = OUT_DIR / "image_manifest.csv"
    manifest_out_f = open(manifest_out_path, "w", newline="", encoding="utf-8")
    manifest_writer = csv.writer(manifest_out_f)
    manifest_writer.writerow(
        ["image_id", "volume_id", "split", "num_annotations", "image_relpath", "label_relpath"]
    )

    for entry in manifest:
        img_id = entry["image_id"]
        vol_id = entry["volume_id"]
        split_name = entry["split"]

        # Just in case something went wrong upstream
        if split_name not in ("train", "val", "test"):
            stats["skipped_images"] += 1
            prob_writer.writerow([img_id, vol_id, split_name, "invalid_split", "split not in train/val/test"])
            continue

        # Download image
        try:
            img = download_image(img_id)
        except Exception as e:
            stats["skipped_images"] += 1
            prob_writer.writerow([img_id, vol_id, split_name, "download_failed", str(e)])
            continue

        W, H = img.size

        # Fetch annotations
        try:
            anns = list_annotations(img_id)
        except Exception as e:
            stats["skipped_images"] += 1
            prob_writer.writerow([img_id, vol_id, split_name, "annotation_fetch_failed", str(e)])
            continue

        yolo_lines: List[str] = []

        for ann in anns or []:
            pts = ann.get("points") or []
            pts = to_float_list(pts)
            if len(pts) < 8:  # needs at least 4 (x,y) pairs
                stats["bad_shapes"] += 1
                prob_writer.writerow([img_id, vol_id, split_name, "too_few_points", f"{len(pts)}"])
                continue

            labels = ann.get("labels") or []
            label_id = None
            # I want the newest label that I actually care about
            for lab in reversed(labels):
                lid = lab.get("label_id")
                if lid in LABEL_ID_TO_IDX:
                    label_id = lid
                    break

            if label_id is None:
                # Track unknown label IDs for debugging
                for lab in labels:
                    lid = lab.get("label_id")
                    if lid:
                        stats["unknown_label_ids"][lid] = stats["unknown_label_ids"].get(lid, 0) + 1
                prob_writer.writerow(
                    [img_id, vol_id, split_name, "no_mapped_label", str([l.get('label_id') for l in labels])]
                )
                continue

            x1, y1, x2, y2 = rect_to_aabb(pts)
            if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
                stats["bad_shapes"] += 1
                short_pts = pts[:10]
                prob_writer.writerow(
                    [img_id, vol_id, split_name, "non_finite_coords",
                     f"{short_pts}{'...' if len(pts) > 10 else ''}"]
                )
                continue
            if x2 <= x1 or y2 <= y1:
                stats["bad_shapes"] += 1
                prob_writer.writerow(
                    [img_id, vol_id, split_name, "degenerate_bbox", f"{x1},{y1},{x2},{y2}"]
                )
                continue

            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, W, H)
            cx, cy, bw, bh = clamp_box(cx, cy, bw, bh)
            cls_idx = LABEL_ID_TO_IDX[label_id]

            yolo_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        num_anns = len(yolo_lines)
        entry["num_annotations"] = num_anns  # for manifest

        # If image has no labels and I don't want empty backgrounds, skip it
        if num_anns == 0:
            if not KEEP_EMPTY:
                # Case 1: I explicitly don't want empty images at all
                stats["skipped_images"] += 1
                prob_writer.writerow([img_id, vol_id, split_name, "no_labels_and_dropped", "KEEP_EMPTY=0"])
                continue
            else:
                # Case 2: I keep empties, but maybe only some of them
                # If EMPTY_KEEP_FRAC < 1.0, randomly drop some empty images
                if EMPTY_KEEP_FRAC < 1.0:
                    if random.random() > EMPTY_KEEP_FRAC:
                        stats["skipped_images"] += 1
                        prob_writer.writerow(
                            [img_id, vol_id, split_name, "empty_downsampled", f"EMPTY_KEEP_FRAC={EMPTY_KEEP_FRAC}"]
                        )
                        continue



        img_name = f"{img_id}.jpg"
        img_relpath = f"images/{split_name}/{img_name}"
        lbl_relpath = f"labels/{split_name}/{img_id}.txt"

        img_out = OUT_DIR / img_relpath
        lbl_out = OUT_DIR / lbl_relpath

        try:
            img.save(img_out, format="JPEG", quality=95)
        except Exception as e:
            stats["skipped_images"] += 1
            prob_writer.writerow([img_id, vol_id, split_name, "image_save_failed", str(e)])
            continue

        # Write (possibly empty) label file
        lbl_out.write_text("\n".join(yolo_lines), encoding="utf-8")

        stats["images_per_split"][split_name] += 1
        stats["written_images"] += 1
        stats["written_boxes"] += num_anns
        if num_anns == 0:
            stats["empty_images"] += 1

        # Write manifest line for this image
        manifest_writer.writerow(
            [img_id, vol_id, split_name, num_anns, img_relpath, lbl_relpath]
        )

    prob_f.close()
    manifest_out_f.close()

    # Save stats + class list + dataset.yaml
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (OUT_DIR / "classes.txt").write_text("\n".join(CLASSES), encoding="utf-8")
    write_dataset_yaml()

    print("==== Export complete ====")
    print(f"Output dir: {OUT_DIR.resolve()}")
    print(f"Dataset yaml: {(OUT_DIR / 'dataset.yaml').resolve()}")
    print(f"Problems report: {problems_path.resolve()}")
    print(f"Image manifest: {manifest_out_path.resolve()}")
    print(json.dumps(stats, indent=2))


# =========================
# Main entry point
# =========================

def main() -> None:
    """
    High-level flow I want:

    1) Collect all image IDs from all configured BIIGLE volumes into one manifest.
    2) Do a global train/val/test split on that manifest.
    3) Loop through the manifest and actually download + convert everything to YOLO.
    """
    print(f"Using BIIGLE base URL: {BASE_URL}")
    print(f"Using volumes: {VOLUME_IDS}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Split fractions: train={TRAIN_FRAC}, val={VAL_FRAC}, test={TEST_FRAC}")
    print(f"KEEP_EMPTY = {KEEP_EMPTY} (keep images with no annotations as backgrounds)")
    if MAX_IMAGES > 0:
        print(f"MAX_IMAGES = {MAX_IMAGES} (global cap)")
    else:
        print("MAX_IMAGES = 0 (no global cap)")

    manifest = collect_all_images()
    if not manifest:
        raise SystemExit("No images found in the configured BIIGLE volumes.")

    assign_global_splits(manifest)
    export_dataset(manifest)


if __name__ == "__main__":
    main()
