#!/usr/bin/env python3
"""Exploratory calcium event candidates from HDF5 dF/F, NOT validated astroCaST detection.

Reads a 3D calcium fluorescence movie, computes a per-pixel robust temporal baseline,
thresholds positive transients and labels connected spatiotemporal candidate events.
Downsampling is solely for inexpensive visual screening; it can miss fine processes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as ndi
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def finite_limits(arr, p1=1, p2=99.5):
    v = np.asarray(arr)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [p1, p2])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def read_movie(h5: h5py.File, key: str, skip: int, max_frames: int | None) -> np.ndarray:
    if key not in h5:
        raise KeyError(f"Dataset {key!r} absent; available keys: {list(h5.keys())}")
    ds = h5[key]
    if ds.ndim != 3:
        raise ValueError(f"Expected [time, row, column] movie, received {ds.shape} for {key}")
    count = ds.shape[0] if max_frames is None else min(max_frames, ds.shape[0])
    if count < 5:
        raise ValueError("At least 5 frames are needed")
    # HDF5 reads are bounded: full resolution is not loaded in RAM.
    out = np.empty((count, (ds.shape[1]+skip-1)//skip, (ds.shape[2]+skip-1)//skip), dtype=np.float32)
    for i in range(count):
        out[i] = ds[i, ::skip, ::skip]
    return out


def screen(movie: np.ndarray, threshold: float, sigma_xy: float,
           min_voxels: int, min_frames: int, baseline_step: int = 3):
    if not np.all(np.isfinite(movie)):
        finite = movie[np.isfinite(movie)]
        fill = float(np.median(finite)) if finite.size else 0.0
        movie = np.nan_to_num(movie, nan=fill, posinf=fill, neginf=fill)
    smoothed = ndi.gaussian_filter(movie, sigma=(0, sigma_xy, sigma_xy), mode="reflect")
    sample = smoothed[::baseline_step]
    baseline = np.median(sample, axis=0).astype(np.float32)
    sigma = (1.4826 * np.median(np.abs(sample - baseline), axis=0)).astype(np.float32)
    # Global floor prevents zero-variance pixels from firing on numerical noise.
    valid = sigma[np.isfinite(sigma) & (sigma > 0)]
    floor = max(float(np.percentile(valid, 20)) * 0.5, 1e-6) if valid.size else 1e-6
    denom = np.maximum(sigma, floor)
    zscores = (smoothed - baseline[None]) / denom[None]
    active = zscores >= threshold
    # 3D face/edge adjacency joins contiguous pixels over space and time.
    labels, num_raw = ndi.label(active, structure=ndi.generate_binary_structure(3, 2))
    # Keep only connected components present on multiple distinct frames.
    slices = ndi.find_objects(labels)
    counts = np.bincount(labels.ravel())
    keep = np.zeros(num_raw + 1, dtype=bool)
    rows = []
    for event_id, sl in enumerate(slices, 1):
        if sl is None or counts[event_id] < min_voxels:
            continue
        frame_count = sl[0].stop - sl[0].start
        if frame_count < min_frames:
            continue
        keep[event_id] = True
    del active
    # Compact numbering 1..N without re-labelling spatial shapes.
    remap = np.zeros(num_raw + 1, dtype=np.int32)
    remap[np.flatnonzero(keep)] = np.arange(1, np.count_nonzero(keep)+1, dtype=np.int32)
    labels = remap[labels]
    for original_id in np.flatnonzero(keep):
        sl = slices[original_id - 1]
        region = labels[sl] == remap[original_id]
        coords = np.nonzero(region)
        tt = coords[0] + sl[0].start
        yy = coords[1] + sl[1].start
        xx = coords[2] + sl[2].start
        zvalues = zscores[sl][region]
        rows.append({
            "candidate_id": int(remap[original_id]),
            "first_frame": int(tt.min()),
            "last_frame": int(tt.max()),
            "duration_frames": int(tt.max()-tt.min()+1),
            "voxels": int(region.sum()),
            "centroid_row_reduced": round(float(yy.mean()), 2),
            "centroid_column_reduced": round(float(xx.mean()), 2),
            "peak_robust_z": round(float(zvalues.max()), 2),
        })
    return labels, zscores, rows, floor, int(num_raw)


def figure_preview(h5, movie, zscores, labels, downsample, dest):
    counts = np.count_nonzero(labels, axis=(1, 2))
    t = int(np.argmax(counts)) if counts.any() else len(movie)//2
    if "mc/ch0" in h5:
        base = np.asarray(h5["mc/ch0"][t, ::downsample, ::downsample], dtype=np.float32)
        base_title = "Motion-corrected fluorescence"
    else:
        base = movie[t]
        base_title = "dF/F frame"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    a, b, c = axes
    a.imshow(base, cmap="gray", vmin=finite_limits(base)[0], vmax=finite_limits(base)[1])
    a.set_title(f"{base_title} | frame {t}")
    z = np.clip(zscores[t], 0, 12)
    b.imshow(z, cmap="magma", vmin=0, vmax=12)
    b.set_title("Positive transient | robust Z (clipped)")
    c.imshow(base, cmap="gray", vmin=finite_limits(base)[0], vmax=finite_limits(base)[1])
    overlay = np.ma.masked_where(labels[t] == 0, labels[t])
    c.imshow(overlay, cmap="nipy_spectral", alpha=0.7, interpolation="nearest")
    c.set_title(f"Candidate events | active pixels {counts[t]}")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle("EXPLORATORY ONLY — not validated detections; colors are NOT cell IDs", fontsize=12)
    fig.savefig(dest, dpi=160)
    plt.close(fig)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="ExVivoSuppRaw.h5 path")
    ap.add_argument("--dataset", default="dff/ch0")
    ap.add_argument("--output", default="real_data/exvivo_screen", type=Path)
    ap.add_argument("--downsample", type=int, default=2, help="Spatial stride; >=1")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=5.0, help="Per-pixel robust temporal Z cutoff")
    ap.add_argument("--sigma-xy", type=float, default=0.8)
    ap.add_argument("--min-voxels", type=int, default=20)
    ap.add_argument("--min-frames", type=int, default=2)
    args = ap.parse_args()
    if args.downsample < 1 or args.min_voxels < 1 or args.min_frames < 1 or args.threshold <= 0 or args.sigma_xy < 0:
        ap.error("Invalid threshold, size, frame count, sigma or downsample")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {args.input} :: {args.dataset}; spatial downsample={args.downsample}", flush=True)
    with h5py.File(args.input, "r") as f:
        movie = read_movie(f, args.dataset, args.downsample, args.max_frames)
        print(f"Movie shape: {movie.shape}, RAM for raw sampled stack: {movie.nbytes/2**20:.1f} MiB", flush=True)
        labels, zscores, rows, floor, n_raw = screen(
            movie, args.threshold, args.sigma_xy, args.min_voxels, args.min_frames)
        frame = figure_preview(f, movie, zscores, labels, args.downsample, args.output / "event_screen.png")
    tifffile.imwrite(args.output / "candidate_event_map.tiff", labels, compression="zlib", bigtiff=True)
    cols = ["candidate_id", "first_frame", "last_frame", "duration_frames", "voxels", "centroid_row_reduced", "centroid_column_reduced", "peak_robust_z"]
    with (args.output / "candidate_events.csv").open("w", newline="") as o:
        w = csv.DictWriter(o, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    summary = {
        "status": "exploratory candidates only; unvalidated", "source_file": str(args.input),
        "source_dataset": args.dataset, "movie_shape_reduced": list(movie.shape),
        "downsample_factor": args.downsample, "threshold_robust_z": args.threshold,
        "noise_floor": floor, "raw_components": n_raw, "retained_candidates": len(rows),
        "qc_frame_zero_based": frame,
        "caution": "Not astroCaST detection. No astrocyte cell identity or biology accuracy established. Reduced resolution must not be registered with full-resolution masks without matching resampling."
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", args.output / "event_screen.png", "|", args.output / "candidate_event_map.tiff", "|", args.output / "candidate_events.csv")

if __name__ == "__main__":
    main()
