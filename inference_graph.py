import matplotlib.pyplot as plt

# -----------------------------
# Your results (mean ± SD FPS)
# -----------------------------
fps_mean = {
    "n": 34.11,
    "s": 16.09,
    "m": 6.78,
    "l": 4.88,
    "x": 2.94,
}

fps_sd = {
    "n": 0.11,
    "s": 0.74,
    "m": 0.50,
    "l": 0.32,
    "x": 0.17,
}

map50_95 = {
    "n": 0.186,
    "s": 0.214,
    "m": 0.200,
    "l": 0.188,
    "x": 0.252,
}

# -----------------------------
# Convert FPS -> ms/img
# -----------------------------
def fps_to_ms(fps):
    return 1000.0 / fps

# Error propagation (approx): ms = 1000/fps, so sd_ms ≈ (1000/fps^2) * sd_fps
def fps_sd_to_ms_sd(fps, sd_fps):
    return (1000.0 / (fps ** 2)) * sd_fps

ms_mean = {m: fps_to_ms(fps_mean[m]) for m in fps_mean}
ms_sd   = {m: fps_sd_to_ms_sd(fps_mean[m], fps_sd[m]) for m in fps_mean}

# Sort: lowest latency (fastest) -> highest latency (slowest)
models = sorted(ms_mean.keys(), key=lambda m: ms_mean[m])

x = [ms_mean[m] for m in models]
xerr = [ms_sd[m] for m in models]
y = [map50_95[m] for m in models]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10.5, 4.5))

# Line connecting points (fast -> slow)
plt.plot(x, y, linewidth=2)

# Points with horizontal error bars (latency SD)
plt.errorbar(
    x, y, xerr=xerr,
    fmt='o', capsize=6, elinewidth=3
)

# Labels with slight offset
for m, xi, yi in zip(models, x, y):
    plt.annotate(
        m,
        (xi, yi),
        textcoords="offset points",
        xytext=(6, 3),
        ha="left",
        fontsize=10
    )

plt.xlabel("Latency (ms/img) at imgsz=640 (mean ± SD, n=3)")
plt.ylabel("Detection Performance (mAP50-95)")
plt.title("Inference latency vs detection performance across YOLO11 model sizes")
plt.grid(True, alpha=0.25)

# Let matplotlib auto-scale, just add some breathing room
ax = plt.gca()
ax.margins(x=0.18, y=2.5)

plt.tight_layout()

out_path = "yolo11_imgsz_latency_accuracy_tradeoff.png"
plt.savefig(out_path, dpi=300)
plt.show()

print("Saved figure to:", out_path)

