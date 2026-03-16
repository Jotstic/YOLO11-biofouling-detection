

from ultralytics import YOLO
import cv2
import time
import statistics
from pathlib import Path

WEIGHTS = [
   r"/Users/jot/Documents/MASTEROPPGAVE/BEST WEIGHTS/yolo11n_Complete_dataset_1_best.pt",
    r"/Users/jot/Documents/MASTEROPPGAVE/BEST WEIGHTS/yolo11s_Complete_dataset_1_best.pt",
    r"/Users/jot/Documents/MASTEROPPGAVE/BEST WEIGHTS/yolo11m_Complete_dataset_1_best.pt",
    r"/Users/jot/Documents/MASTEROPPGAVE/BEST WEIGHTS/yolo11l_Complete_dataset_1_best.pt",
    r"/Users/jot/Documents/MASTEROPPGAVE/BEST WEIGHTS/yolo11x_Complete_dataset_1_best.pt",
]

VIDEO_PATH = r"/Users/jot/Downloads/Video filer/Batch9/tp74back_D18_T1444-1501.mp4"
IMGSZ = 640
WARMUP_FRAMES = 100
MEASURE_FRAMES = 2000
REPEATS = 3

def measure_fps_repeat(weights_path: str, video_path: str, imgsz: int,
                       warmup_frames: int, measure_frames: int, repeats: int):
    weights_path = str(weights_path)
    video_path = str(video_path)

    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    model = YOLO(weights_path)

    fps_runs = []

    for r in range(repeats):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # --- Warmup (not timed) ---
        for _ in range(warmup_frames):
            ret, frame = cap.read()
            if not ret:
                break
            model.predict(frame, imgsz=imgsz, verbose=False)

        # Reset to start for timed section (important for repeatability)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # --- Timed inference ---
        count = 0
        t0 = time.time()
        while count < measure_frames:
            ret, frame = cap.read()
            if not ret:
                break
            model.predict(frame, imgsz=imgsz, verbose=False)
            count += 1

        dt = time.time() - t0
        cap.release()

        fps = 0.0 if (count == 0 or dt == 0) else (count / dt)
        fps_runs.append(fps)

    mean_fps = statistics.mean(fps_runs)
    std_fps = statistics.pstdev(fps_runs) if len(fps_runs) > 1 else 0.0  # population SD
    return fps_runs, mean_fps, std_fps

print("\n=== FPS Benchmark (pure inference) ===")
print(f"Video: {VIDEO_PATH}")
print(f"imgsz={IMGSZ}, warmup={WARMUP_FRAMES}, measure_frames={MEASURE_FRAMES}, repeats={REPEATS}\n")

# Header
print(f"{'Model':50s}  {'Run1':>7s}  {'Run2':>7s}  {'Run3':>7s}  {'Mean':>7s}  {'SD':>7s}")

for w in WEIGHTS:
    runs, mean_fps, sd_fps = measure_fps_repeat(
        w, VIDEO_PATH,
        imgsz=IMGSZ,
        warmup_frames=WARMUP_FRAMES,
        measure_frames=MEASURE_FRAMES,
        repeats=REPEATS
    )

    # Ensure exactly 3 columns even if repeats changed
    r1 = runs[0] if len(runs) > 0 else 0.0
    r2 = runs[1] if len(runs) > 1 else 0.0
    r3 = runs[2] if len(runs) > 2 else 0.0

    print(f"{Path(w).name:50s}  {r1:7.2f}  {r2:7.2f}  {r3:7.2f}  {mean_fps:7.2f}  {sd_fps:7.2f}")





