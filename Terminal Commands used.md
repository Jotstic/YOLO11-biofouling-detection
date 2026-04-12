
----- CIclass -----

python3 Ciclass.py \
  --data "path_to_dataset" \
  --weights "path_to_weight" \
  --device mps \
  --n_boot 100 \
  --outdir "out_path directory" \
  --csv_name "Name"




----- Automaticannotationtoobiigle -----

BIIGLE_BASE_URL="https://biigle.de" \
BIIGLE_VOLUME_ID=volume_ID \
BIIGLE_USERNAME="username (biigle_mail)" \
BIIGLE_API_TOKEN="API_Token" \
WEIGHTS_PATH="path_to_weight" \
CONF_THRESH=0.50 \
DRY_RUN=0 \
MAX_IMAGES=0 \
python3 "automaticannotationtoobiigle.py"




----- export_dataset_frombiigle -----

export BIIGLE_BASE_URL="https://biigle.de"
export BIIGLE_USERNAME="username (biigle_mail)"
export BIIGLE_API_TOKEN="API_token"
export BIIGLE_VOLUME_IDS=""
export OUT_DIR="path_to_weight"
export TRAIN_FRAC="0.7"
export VAL_FRAC="0.15"
export RANDOM_SEED="42"
export KEEP_EMPTY="1"
export EMPTY_KEEP_FRAC="0.3"
python3 export_dataset_frombiigle.py





----- frameextractor -----

SCRIPT="frameextractor.py"
VIDDIR="path_to_videos"
BASE_OUT="outdirectory_to_frames"

mkdir -p "$BASE_OUT"

python3 "$SCRIPT" run \
  --in "$VIDDIR" \
  --out "$BASE_OUT" \
  --glob "*.mp4" \
  --no-per-video-subfolders \
  --scorer combo \
  --fps-cap 0.5 \
  --top-k-per-sec 1 \
  --laplacian-min 400 \
  --laplacian-max 700 \
  --dct-min-ratio 0.30 \
  --dct-max-ratio 0.70 \
  --pix-fmt yuv444p \
  --jpeg-q 2 \
  --extract-workers 2 \
  --out-w 800 --out-h 600 --resize-mode fit

python3 "$SCRIPT" audit-duplicates \
  "$BASE_OUT" \
  --glob "*.jpg" \
  --hamming 12 \
  --fix \
  --no-dry-run





