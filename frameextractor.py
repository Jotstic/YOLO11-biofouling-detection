# Ensures python treats type hints as string, for cleaner and safer future-proof typing
from __future__ import annotations

import asyncio # Calls ffmpeg. runs the frame extraction tasks
import hashlib #Runs the SHA-1 to find duplicated frames
import heapq  #maintains the best frames "top K" per second
import pathlib #cross platform path handling for files and folders
import subprocess #calls ffmpeg to extract signle frames
from dataclasses import dataclass #a container for Candidate and ExtractConfig
from functools import lru_cache #Cache DCT ring mask so they are ccomputed once and then reused.
from typing import Iterable, List, Optional, Sequence, Tuple  #type hints for readability

import numpy as np # Imports Numpy the numerical library, udes for array math (laplacian valies, dct masks etc. )
import typer # Framework for making command lines-
from rich.console import Console #Used for prettier terminal output.

#Imports a progress bars to show a live progress of the frame extraction
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

import cv2 # Imports cv2


# Creates the CLI app, disables shell completion set ups and show help if run with no arguments.
app = typer.Typer(add_completion=False, no_args_is_help=True)


#sets up Rich for pretty printing progress
console = Console()


#Computes a sharpness score, converts to grey and runs a Laplacian filter then returns the variance of the result. Higher variance = sharper, lower=blurrier
def laplacian_score(img_bgr: np.ndarray, ksize: int = 3) -> float:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F, ksize=ksize).var())


# Converts the frame to grayscale and returns the fraction of pixels clipped, either very close to 0 black or 255 white. tol is the margin
def exposure_clip_fraction(img_bgr: np.ndarray, tol: int = 4) -> float:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    total = g.size
    lo = (g <= tol).sum()
    hi = (g >= 255 - tol).sum()
    return float((lo + hi) / max(1, total))


#Computes a 64-bit perceptual hash: downsamples to 32×32, takes an 8×8 DCT block, thresholds coefficients by their median, and packs the resulting 64 booleans into a u64 integer.
def phash_u64(img_bgr: np.ndarray) -> int:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32) / 255.0)[:8, :8]
    med = np.median(dct)
    bits = (dct > med).astype(np.uint8).ravel()[:64]
    v = 0
    for i, b in enumerate(bits): v |= int(b) << i
    return v


#Calculates the hamming distance between two 64 bits integers, (how many bits differ)
def hamming_u64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


#Builds two boolean ring masks over an nxn DCT grid so it can sum mid and high frequency energy separately. 
@lru_cache(maxsize=64)
def _ring_masks(n: int, mid_lo=0.15, mid_hi=0.55, hi_lo=0.65) -> Tuple[np.ndarray, np.ndarray]:
    u = np.arange(n)[:, None]
    v = np.arange(n)[None, :]
    r = np.sqrt((u / n) ** 2 + (v / n) ** 2)
    mid = (r >= mid_lo) & (r <= mid_hi)
    hi  = (r >= hi_lo)
    return mid, hi


#Slides a blockxblock window over the grayscale frame, DCTs for each tile, measures high-freq energy vs mid energy, 
#and returns the 95th percentile of those ratios, and the fraction of tiles above tile_thr- both flagging blocky/compressed frames.
def block_dct_blockiness(gray: np.ndarray,
                         block=16, stride=24,
                         mid_lo=0.15, mid_hi=0.55, hi_lo=0.65,
                         tile_thr=0.55) -> Tuple[float, float]:
    H, W = gray.shape
    mid_mask, hi_mask = _ring_masks(block, mid_lo, mid_hi, hi_lo)
    scores: List[float] = []
    for y in range(0, H - block + 1, stride):
        for x in range(0, W - block + 1, stride):
            tile = gray[y:y+block, x:x+block].astype(np.float32) / 255.0
            G = cv2.dct(tile)
            mid_e = float((G[mid_mask] ** 2).sum())
            hi_e  = float((G[hi_mask]  ** 2).sum())
            s = hi_e / (mid_e + hi_e + 1e-8)
            scores.append(s)
    if not scores:
        return 0.0, 0.0
    arr = np.asarray(scores, np.float32)
    p95  = float(np.percentile(arr, 95))
    frac = float((arr > tile_thr).mean())
    return p95, frac


#Downscales the frame, DCTs it, compares mid freq energy to mid+high energy, and returns that ratio. Higher=smoother, lower = very fine/high freq
def global_dct_detail_ratio(gray: np.ndarray,
                            size=64, mid_lo=0.15, mid_hi=0.55, hi_lo=0.65) -> float:
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    G = cv2.dct(small.astype(np.float32) / 255.0)
    mid_mask, hi_mask = _ring_masks(size, mid_lo, mid_hi, hi_lo)
    mid_e = float((G[mid_mask] ** 2).sum())
    hi_e  = float((G[hi_mask]  ** 2).sum())
    return float(mid_e / (mid_e + hi_e + 1e-8))


#reads the file in 1 MB chunks and updates sha-1 hash, returning the hex digest. This lets you detect byte-for-byte identiccal images. 
def sha1_of_file(p: pathlib.Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# keeps each frame metadata plus a computed score used for ranking and spacing. 
@dataclass
class Candidate:
    sec: int
    t: float
    frame_idx: int
    lap: float
    clip: float
    phash64: int
    score: float = 0.0


#A small config container for how frames are written: JPEG quality, worker count, output size/mode and pixel format that get turnes into ffmpeg flags. 
@dataclass
class ExtractConfig:
    jpeg_q: int = 95
    extract_workers: int = 2
    out_w: Optional[int] = None
    out_h: Optional[int] = None
    resize_mode: str = "exact"
    pix_fmt: str = "yuv420p"


#Skips through the video by steo frames, only decoding frames kept, and yields (frame_idx, time_sec frame)
def _iter_frames(cap: cv2.VideoCapture, step: int) -> Iterable[Tuple[int, float, np.ndarray]]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        if not cap.grab(): break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok: break
            yield idx, idx / fps, frame
        idx += 1


# Samples frames at a capped rate, rejects blocky/over-smooth/blurry ones, scores survivors by sharpness with an exposure penalty 
#Keeps the best K per second , and then inforces a minimum time gap and removes near duplicates by Hamming distance. 
def score_and_sample_video(path: pathlib.Path,
                           fps_cap=1.0, oversample=3.0, top_k_per_sec=1,
                           min_gap_ms=600, phash_distance=14, downscale=1280,
                           lap_min=120.0, lap_ksize=3,
                           block_on=True, block_size=16, block_stride=24,
                           block_tile_thr=0.55, block_p95_max=0.60, block_frac_max=0.35,
                           dct_min_ratio=0.0,
                           dct_max_ratio=1.0,
                           w_clip_pen=0.10,
                           lap_max: float = 0.0
                           ) -> List[Candidate]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scan_fps = min(fps, max(0.1, fps_cap * oversample)) if fps_cap > 0 else fps
    step = max(1, int(round(fps / scan_fps)))

    raw: List[Candidate] = []
    for frame_idx, t, frame in _iter_frames(cap, step):

        if downscale and max(frame.shape[:2]) > downscale:
            s = downscale / max(frame.shape[:2])
            frame_small = cv2.resize(frame, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        else:
            frame_small = frame

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)


        if block_on:
            p95, frac = block_dct_blockiness(
                gray, block=block_size, stride=block_stride,
                mid_lo=0.15, mid_hi=0.55, hi_lo=max(0.65, block_tile_thr),
                tile_thr=block_tile_thr
            )
            if p95 > block_p95_max or frac > block_frac_max:
                continue
            if dct_min_ratio > 0.0:
                if global_dct_detail_ratio(gray) < dct_min_ratio:
                    continue
            if dct_max_ratio < 1.0:
                if global_dct_detail_ratio(gray) > dct_max_ratio:
                    continue



        lap = laplacian_score(frame_small, lap_ksize)
        if lap_min > 0 and lap < lap_min:
            continue
        if lap_max > 0 and lap > lap_max:
            continue

        clip = exposure_clip_fraction(frame_small)
        ph = phash_u64(frame_small)
        raw.append(Candidate(sec=int(t), t=float(t), frame_idx=int(frame_idx),
                             lap=float(lap), clip=float(clip), phash64=int(ph)))

    cap.release()
    if not raw:
        return []


    laps = np.asarray([c.lap for c in raw], float)
    lo, hi = np.percentile(laps, [5, 95])
    if hi <= lo:
        lo, hi = float(laps.min()), float(laps.max())

    def z01(v: float) -> float:
        return float((v - lo) / max(1e-12, hi - lo))

    # composite score = norm_lap * (1 - w_clip_pen * clip)
    for c in raw:
        c.score = max(0.0, z01(c.lap) * (1.0 - w_clip_pen * c.clip))

    # top-K per second (min-heap per bucket)
    buckets: dict[int, List[Tuple[float, Candidate]]] = {}
    for c in raw:
        h = buckets.setdefault(c.sec, [])
        if len(h) < top_k_per_sec:
            heapq.heappush(h, (c.score, c))
        else:
            if c.score > h[0][0]:
                heapq.heapreplace(h, (c.score, c))

    prelim = [c for _, h in buckets.items() for _, c in h]
    prelim.sort(key=lambda c: c.t)

    # spacing + near-duplicate filter (Hamming on pHash)
    picks: List[Candidate] = []
    last_ms = -10**9
    for c in prelim:
        tms = int(round(c.t * 1000))
        if picks and tms - last_ms < min_gap_ms:
            continue
        if any(hamming_u64(p.phash64, c.phash64) <= phash_distance for p in picks[-10:]):
            continue
        picks.append(c)
        last_ms = tms
    return picks



# Extracts one frame at timestamp ts from inp, optionally resizes  crops and pads based on Extraconfig, and saves it as jpeg to out_file
def ffmpeg_extract_one(inp: pathlib.Path, ts: float, out_file: pathlib.Path, e: ExtractConfig) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{ts:.3f}", "-i", str(inp),
        "-frames:v", "1", "-q:v", str(e.jpeg_q),]
    vf_parts: List[str] = []
    w, h = e.out_w, e.out_h
    if w and h:
        if e.resize_mode == "fit":
            vf_parts += [f"scale={w}:{h}:force_original_aspect_ratio=decrease",
                         f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"]
        elif e.resize_mode == "fill":
            vf_parts += [f"scale={w}:{h}:force_original_aspect_ratio=increase",
                         f"crop={w}:{h}"]
        else:
            vf_parts += [f"scale={w}:{h}"]
    elif w and not h:
        vf_parts += [f"scale={w}:-2"]
    elif h and not w:
        vf_parts += [f"scale=-2:{h}"]
    vf_parts.append(f"format={e.pix_fmt}")
    cmd += ["-vf", ",".join(vf_parts)]
    if e.pix_fmt: cmd += ["-pix_fmt", e.pix_fmt]
    cmd += ["-y", str(out_file)]
    subprocess.run(cmd, check=True)


#Creates an async task per choses frame, limits concurrency with a semaphore, runs ffmpeg_extract_one in a thread pool and collects the output file paths. 
async def extract_all_ffmpeg(inp: pathlib.Path, picks: Sequence[Candidate],
                             out_dir: pathlib.Path, stem: str, e: ExtractConfig) -> List[pathlib.Path]:
    saved: List[pathlib.Path] = []
    sem = asyncio.Semaphore(e.extract_workers)
    async def worker(c: Candidate):
        out = out_dir / f"{stem}_t{c.t:06.3f}_f{c.frame_idx}_S{int(round(c.score*1000))}.jpg"
        if out.exists(): saved.append(out); return
        async with sem:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, ffmpeg_extract_one, inp, c.t, out, e)
            except subprocess.CalledProcessError as ex:
                console.print(f"[yellow]ffmpeg failed {inp.name}@{c.t:.3f}s: {ex}")
                return
            saved.append(out)
    await asyncio.gather(*(worker(c) for c in picks))
    return saved


#Finds all maching videos, filters and chooses frames via scoer_and_sampleVideo(), then saves them concurrently with extract all_ffmpeg.
@app.command("run")
def run(
    in_path: pathlib.Path = typer.Option(..., "--in", exists=True, file_okay=False, dir_okay=True),
    out_base: pathlib.Path = typer.Option(..., "--out"),
    subfolder: str = typer.Option("", "--subfolder"),
    glob: str = typer.Option("*.mp4", "--glob"),
    per_video_subfolders: bool = typer.Option(True, "--per-video-subfolders/--no-per-video-subfolders"),
    scorer: str = typer.Option("combo", "--scorer"),  # unused, but KEEP because your CLI passes it
    top_k_per_sec: int = typer.Option(1, "--top-k-per-sec"),
    fps_cap: float = typer.Option(1.0, "--fps-cap"),
    min_gap_ms: int = typer.Option(600, "--min-gap-ms"),
    phash_distance: int = typer.Option(14, "--phash-distance"),
    downscale: int = typer.Option(1280, "--downscale"),
    laplacian_min: float = typer.Option(120.0, "--laplacian-min"),
    laplacian_max: float = typer.Option(0.0, "--laplacian-max"),
    laplacian_ksize: int = typer.Option(3, "--laplacian-ksize"),
    block_on: bool = typer.Option(True, "--block-on/--no-block-on"),
    block_size: int = typer.Option(16, "--block-size"),
    block_stride: int = typer.Option(24, "--block-stride"),
    block_tile_thr: float = typer.Option(0.55, "--block-tile-thr"),
    block_p95_max: float = typer.Option(0.60, "--block-p95-max"),
    block_frac_max: float = typer.Option(0.35, "--block-frac-max"),
    dct_min_ratio: float = typer.Option(0.0, "--dct-min-ratio"),
    dct_max_ratio: float = typer.Option(1.0, "--dct-max-ratio"),
    w_clip_pen: float = typer.Option(0.10, "--w-clip-pen"),
    extract_workers: int = typer.Option(2, "--extract-workers"),
    jpeg_q: int = typer.Option(95, "--jpeg-q"),
    out_w: Optional[int] = typer.Option(None, "--out-w"),
    out_h: Optional[int] = typer.Option(None, "--out-h"),
    resize_mode: str = typer.Option("exact", "--resize-mode"),
    pix_fmt: str = typer.Option("yuv420p", "--pix-fmt"),
    max_per_video: Optional[int] = typer.Option(None, "--max-per-video"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
):
    videos = sorted(in_path.glob(glob))
    if not videos:
        console.print(f"[red]No videos found in {in_path} matching {glob}")
        raise typer.Exit(3)

    out_dir = out_base / subfolder if subfolder else out_base
    out_dir.mkdir(parents=True, exist_ok=True)

    e_cfg = ExtractConfig(
        jpeg_q=jpeg_q, extract_workers=extract_workers,
        out_w=out_w, out_h=out_h, resize_mode=resize_mode, pix_fmt=pix_fmt
    )

    console.rule("[bold cyan]Remora frame extraction (Laplacian + Block-DCT + Global DCT ratio)")

    async def runner():
        total = 0
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(), TimeRemainingColumn()
        ) as progress:
            for vid in videos:
                dest = out_dir / (vid.stem if per_video_subfolders else "")
                dest.mkdir(parents=True, exist_ok=True)
                progress.console.print(f"Scanning: {vid.name}")

                picks = score_and_sample_video(
                    vid,
                    fps_cap=fps_cap, oversample=3.0, top_k_per_sec=top_k_per_sec,
                    min_gap_ms=min_gap_ms, phash_distance=phash_distance, downscale=downscale,
                    lap_min=laplacian_min, lap_ksize=laplacian_ksize,
                    block_on=block_on, block_size=block_size, block_stride=block_stride,
                    block_tile_thr=block_tile_thr, block_p95_max=block_p95_max, block_frac_max=block_frac_max,
                    dct_min_ratio=dct_min_ratio, w_clip_pen=w_clip_pen, lap_max=laplacian_max, dct_max_ratio=dct_max_ratio
                )

                if max_per_video and len(picks) > max_per_video:
                    picks = picks[:max_per_video]
                if dry_run:
                    progress.console.print(f"[yellow]DRY-RUN[/] would keep {len(picks)} frames for {vid.name}")
                    continue
                if not picks:
                    progress.console.print(f"[blue]No frames kept for {vid.name}")
                    continue

                task = progress.add_task(f"Extracting {vid.name}", total=len(picks))
                saved = await extract_all_ffmpeg(vid, picks, dest, vid.stem, e_cfg)
                total += len(saved)
                progress.update(task, advance=len(saved))

            progress.console.rule(f"[green]Done. Saved {total} frames → {out_dir}")

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.")
        raise typer.Exit(13)



#Collect files and finds all images matching glob (default jpg) in path.
#Exact duplicates , hashes every file with SHA-1, if multiple files share the same hash, they are exact copies - grouped together
#Loads each images with opencv and computes a 64 bit pHash, forms clusters where images are within hamming bits of the clusters first item. 
#Prints counts of exact and near-duplicated groups
#Within each group keep the file with the highest suffix in the filename (extrtaction score); removes the rest. 
#Within each near-duplicate cluster, sort by _S### and remove all but the highest score file.
#If dry_run is on, it only prints what would be removed. 
@app.command("audit-duplicates")
def audit_duplicates(
    path: pathlib.Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    glob: str = typer.Option("*.jpg", "--glob"),
    hamming: int = typer.Option(12, "--hamming"),
    fix: bool = typer.Option(False, "--fix/--no-fix"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
):
    imgs = sorted(path.glob(glob))
    if not imgs:
        console.print(f"[red]No images in {path} matching {glob}")
        raise typer.Exit(3)

    console.rule("[bold cyan]Duplicate audit")

    sha_owner: dict[str, pathlib.Path] = {}
    exact_groups: dict[str, List[pathlib.Path]] = {}
    for p in imgs:
        d = sha1_of_file(p)
        if d in sha_owner:
            exact_groups.setdefault(d, []).append(p)
        else:
            sha_owner[d] = p

    if exact_groups:
        extras = sum(len(v) for v in exact_groups.values())
        console.print(f"[yellow]Exact duplicates: {extras} extra files in {len(exact_groups)} groups")
    else:
        console.print("[green]No exact duplicates")

    recs: List[Tuple[pathlib.Path, int, int]] = []
    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        s = 0
        stem = p.stem
        if "_S" in stem:
            try:
                s = int(stem.split("_S")[-1])
            except Exception:
                s = 0
        recs.append((p, phash_u64(img), s))

    clusters: List[List[Tuple[pathlib.Path, int, int]]] = []
    for rec in recs:
        placed = False
        for c in clusters:
            if hamming_u64(rec[1], c[0][1]) <= hamming:
                c.append(rec)
                placed = True
                break
        if not placed:
            clusters.append([rec])

    near_groups = [c for c in clusters if len(c) > 1]
    if near_groups:
        extras = sum(len(c) - 1 for c in near_groups)
        console.print(f"[yellow]Near duplicates (Hamming ≤ {hamming}): {extras} extras in {len(near_groups)} groups")
    else:
        console.print("[green]No near duplicates")

    if fix:
        removed: List[pathlib.Path] = []

        for d, extras_list in exact_groups.items():
            cand = [sha_owner[d]] + extras_list
            best = sorted(
                cand,
                key=lambda p: int(p.stem.split("_S")[-1]) if "_S" in p.stem else 0,
                reverse=True,
            )[0]
            for p in cand:
                if p == best:
                    continue
                if not dry_run:
                    p.unlink(missing_ok=True)
                removed.append(p)

        for c in near_groups:
            c_sorted = sorted(c, key=lambda x: x[2], reverse=True)
            for p, _, _ in c_sorted[1:]:
                if not dry_run:
                    p.unlink(missing_ok=True)
                removed.append(p)

        console.print(f"[green]Planned removals: {len(removed)} files")
        if dry_run:
            console.print("[yellow]Dry-run only. Re-run with --fix --no-dry-run to delete.")




#Required so my terminal commands work. 
if __name__ == "__main__":
    app()
