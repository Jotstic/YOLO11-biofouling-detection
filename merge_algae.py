import os
from pathlib import Path
import shutil
import yaml

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------
SRC = Path("/Users/jot/Documents/MASTEROPPGAVE/datasets/test_dataset")
DST = SRC.parent / "test_dataset1"

if DST.exists():
    raise SystemExit(f"Target already exists: {DST}")

print(f"Creating fixed dataset at: {DST}")
DST.mkdir()

# --------------------------------------------------
# 2. Create YOLO directory structure
# --------------------------------------------------
for sub in ["images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test"]:
    (DST / sub).mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3. Copy ONLY test images + labels into new structure
# --------------------------------------------------
src_images_test = SRC / "images/test"
src_labels_test = SRC / "labels/test"

image_files = list(src_images_test.glob("*.*"))
label_files = list(src_labels_test.glob("*.txt"))

print(f"Found {len(image_files)} test images.")
print(f"Found {len(label_files)} test labels.")

copied = 0

for img in image_files:
    stem = img.stem
    lbl = src_labels_test / f"{stem}.txt"

    if not lbl.exists():
        print(f"[WARN] Missing label for {img.name}")
        continue

    shutil.copy(img, DST / "images/test" / img.name)
    shutil.copy(lbl, DST / "labels/test" / lbl.name)
    copied += 1

print(f"Copied {copied} matched image/label pairs.")

# --------------------------------------------------
# 4. CORRECT CLASS REMAPPING
# --------------------------------------------------

# ORIGINAL CLASSES:
# 0 Fish
# 1 Red Algae
# 2 Green Algae
# 3 Brown Algae
# 4 Jellyfish
# 5 Hydroids
# 6 Bivalves (drop)

mapping = {
    0: 0,  # Fish
    1: 1,  # Red Algae
    2: 2,  # Green Algae → Algae
    3: 2,  # Brown Algae → Algae
    4: 3,  # Jellyfish
    5: 4,  # Hydroids
}

DROP = {6}

print("Applying correct remapping...")

for lbl in (DST / "labels/test").glob("*.txt"):
    lines = lbl.read_text().strip().splitlines()
    new_lines = []

    for line in lines:
        parts = line.split()
        old_id = int(parts[0])

        if old_id in DROP:
            continue

        if old_id not in mapping:
            print(f"[WARN] Unknown class {old_id} in {lbl.name}")
            continue

        parts[0] = str(mapping[old_id])
        new_lines.append(" ".join(parts))

    lbl.write_text("\n".join(new_lines) + "\n")

print("Remapping finished successfully.")

# --------------------------------------------------
# 5. Write dataset.yaml
# --------------------------------------------------
dataset_yaml = {
    "path": str(DST.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {
        0: "Fish",
        1: "Red Algae",
        2: "Algae",
        3: "Jellyfish",
        4: "Hydroids",
    }
}

with open(DST / "dataset.yaml", "w") as f:
    yaml.safe_dump(dataset_yaml, f, sort_keys=False)

print("\n==== DATASET FIXED ====")
print(f"New dataset at: {DST}")
print("Use this dataset.yaml for evaluation.")

