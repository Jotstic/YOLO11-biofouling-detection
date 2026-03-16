# Google Drive + Settings

from google.colab import drive #connecting to google drive in colab, 
drive.mount("/content/gdrive") #Mounts the content g.drive

#Imports dependencies my YOLO pipeline needs, path handlings, file copying, jsonwriting, type hints, image processing, timestamps, and ultralytics.
#Configures run settings and load, train and validate YOLO.
from pathlib import Path 
import os, shutil, json
from typing import Dict
from PIL import Image
from datetime import datetime

from ultralytics import settings, YOLO

DATASET_YAML = "/content/gdrive/MyDrive/Colab Notebooks/datasets/dataset_2/dataset.yaml" #Path to dataset folder. 

# Defines three different folder locations in google drive where the script stores, weights, validations and runs. 
WEIGHTS_DIR = Path("/content/gdrive/MyDrive/Colab Notebooks/weights")          
RUNS_DIR    = Path("/content/gdrive/MyDrive/Colab Notebooks/runs_ultra")      
EVALS_DIR   = Path("/content/gdrive/MyDrive/Colab Notebooks/evals_ultra")     

#Setting my training parameters for easy change if i want to test different factors.
IMG_SIZE = 640          
BATCH    = 24          
EPOCHS   = 100
SEED     = 0

# A configurations dictionary that sets my datasets, caching workers and other sets parameters. 
COMMON_TRAIN_ARGS = dict(
    data=DATASET_YAML,
    project="runs_biogrowth",     
    name=None,                   
    exist_ok=True,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMG_SIZE,
    device=0,
    workers=2,
    cache="ram",
    mosaic=0.0,
    hsv_s=0.2,
    hsv_v=0.1,
    fliplr=0.5,
    patience=20,
    seed=SEED,
    
)

#Decides the YOLO models i want to train and compare. The script will iterate over the different model sizes.
MODELS: Dict[str, str] = {
    "n": "yolo11n.pt",
    "s": "yolo11s.pt",
    "m": "yolo11m.pt",
    "l": "yolo11l.pt",
    "x": "yolo11x.pt",
}

#Ensures the three output directories exists, if they dont they are created. 
for _p in (WEIGHTS_DIR, RUNS_DIR, EVALS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Make Ultralytics write runs to Drive
settings.update({"runs_dir": str(RUNS_DIR)})


# A function that copies a file from the source file path to the destination file path. Created the destinations folder if needed. 
def _copy(src: Path, dst: Path) -> bool:
    src = Path(src); dst = Path(dst)
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True

#A function which replace the destination folder with a fresh copy of source folder. Converts inputs to paths, i destination exist it deletes it.
#Then it copies the entire directory three from src_dir to dst_dir
def _mirror_tree(src_dir: Path, dst_dir: Path):
    src_dir = Path(src_dir); dst_dir = Path(dst_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    print(f"[eval] Mirrored {src_dir} -> {dst_dir}")


# Makes the evaluation plots publication ready as pngs at 300 DPI. Looks for yolo eval images in eval_dir. If file exists it opens it and re saves as png. 
#If original was not PNG deletes the old file after creating the png. Warning if no images are are processed.  
#I chose png because they creates more crisp lines, text and axes. And re saving JPG could cause quality loss. 
def _ensure_high_dpi_pngs(eval_dir: Path, dpi: int = 300):

    eval_dir = Path(eval_dir)
    targets = [
        "confusion_matrix.png",
        "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png",
        "results.png", "labels_correlogram.jpg",
    ]
    for name in targets:
        p = eval_dir / name
        if p.exists():
            try:
                with Image.open(p) as im:
                    out = p.with_suffix(".png")
                    im.save(out, format="PNG", dpi=(dpi, dpi))
                    if p.suffix.lower() != ".png":
                        p.unlink(missing_ok=True)
            except Exception as e:
                print(f"[warn] Could not re-save {p.name} at {dpi} DPI: {e}")


#Creates a folder friendly line for each model size
def _tag_for_size(size_key: str) -> str:
    return f"yolo11-{size_key}"

#Creates a folder friendly weight name for each model size. 
def _weight_name_for_size(size_key: str) -> str:
    return f"yolo11{size_key}_best.pt"


#This snippet initializes an empty dictionary that will later store the evaluation metrics for each trained YOLO model.
ALL_METRICS: Dict[str, dict] = {}

#This entire block train each YOLO model size, evaluates it, then saves the results and records all metrics and metadata. 
#Starts the loop over each YOLO size, creates tag name, nd prints header indicated which size are being trained. 
for size_key, weights_path in MODELS.items():
    tag = _tag_for_size(size_key)
    print(f"\n========== Training {tag} ({weights_path}) ==========")

    #Loads the yolo model, creates a name for it and prepares traing arguments, copying the common settings and inserts the run name. 
    model = YOLO(weights_path)
    run_name = f"{tag}_union_colab"
    train_args = dict(COMMON_TRAIN_ARGS)
    train_args["name"] = run_name

    # Runs training, locates folder and saves weights as the best.pt and the last.pt. 
    results = model.train(**train_args)
    run_dir = Path(results.save_dir)                 
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    # The snippet copies the best and last weight into the central weights folder with clean filenames. 
    out_best = WEIGHTS_DIR / _weight_name_for_size(size_key)
    out_last = WEIGHTS_DIR / f"yolo11{size_key}_last.pt"
    if _copy(best_pt, out_best):
        print(f"[weights] Copied best -> {out_best}")
    if _copy(last_pt, out_last):
        print(f"[weights] Copied last -> {out_last}")

   #This snippet runs a full validation pass from the training model and saves evaluation outputs in a separate eval folder. 
    print(f"\n[eval] Running separate validation for {tag} ...")
    val_res = model.val(
        data=DATASET_YAML,
        split="val",                   
        project="runs_biogrowth_eval", 
        name=f"{tag}_val",             
        plots=True,                    
        save_json=True,               
        verbose=True,
    )
    #Defines where the validation results were saved and where they should be mirrored in the evals folder. 
    eval_src = Path(val_res.save_dir)                 
    eval_dst = EVALS_DIR / tag                       

    #COnverts the eval plots to high dpi pngs nd then copies the entire eval folder into the organized evals irectory. 
    _ensure_high_dpi_pngs(eval_src, dpi=300)
    _mirror_tree(eval_src, eval_dst)

    #Extracts the validation metrics, saves them as a JSON files and stores them in the global results directory. 
    metrics_out = eval_dst / "metrics_summary.json"
    metrics = getattr(val_res, "metrics", None)
    metrics_dict = {}
    if metrics is not None:
        try:
            metrics_dict = dict(metrics)
        except Exception:
            metrics_dict = json.loads(json.dumps(metrics, default=str))
        metrics_out.write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")
        ALL_METRICS[tag] = metrics_dict
        print(f"[eval] Wrote metrics -> {metrics_out}")

    #Builds the metadata folder and saves it to google drive as metadata.json in the models eval folder. 
    meta = {
        "model": f"yolo11{size_key}",
        "tag": tag,
        "train_args": {
            "batch": BATCH,
            "imgsz": IMG_SIZE,
            "epochs": EPOCHS,
            "patience": COMMON_TRAIN_ARGS["patience"],
            "seed": SEED,
            "device": COMMON_TRAIN_ARGS["device"],
            "augment": {
                "mosaic": COMMON_TRAIN_ARGS["mosaic"],
                "hsv_s": COMMON_TRAIN_ARGS["hsv_s"],
                "hsv_v": COMMON_TRAIN_ARGS["hsv_v"],
                "fliplr": COMMON_TRAIN_ARGS["fliplr"],
            },
        },
        "paths": {
            "weights_best": str(out_best),
            "weights_last": str(out_last),
            "train_run_dir": str(run_dir),
            "eval_run_dir": str(eval_src),
            "eval_mirror_dir": str(eval_dst),
            "dataset_yaml": DATASET_YAML,
        },
        "val_metrics": metrics_dict,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (eval_dst / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[eval] Wrote metadata -> {(eval_dst / 'metadata.json')}")

#Saves all the model metrics into one summary JSON file and prints the final output locations for weights, eval results adn teh summary. 
summary_path = EVALS_DIR / "all_models_val_metrics.json"
summary_path.write_text(json.dumps(ALL_METRICS, indent=2), encoding="utf-8")
print("\n==== Done ====")
print("Weights dir:", WEIGHTS_DIR.resolve())
print("Eval root  :", EVALS_DIR.resolve())
print("Summary    :", summary_path.resolve())