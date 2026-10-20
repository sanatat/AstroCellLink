#!/usr/bin/env python3
"""Exploratory diagnostics for ONE candidate event from an AstroCellLink threshold screen.

No cell identity inference. This does not execute official astroCaST detection.
Requires numpy, scipy, h5py, tifffile, matplotlib (no pandas).
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import tifffile
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_plots(args):
    results = Path(args.results)
    out = Path(args.output) if args.output else results
    out.mkdir(parents=True, exist_ok=True)
    table_path = results / "candidate_events.csv"
    label_path = results / "candidate_event_map.tiff"
    with open(table_path, newline="") as fh:
        lookup = {int(r["candidate_id"]): r for r in csv.DictReader(fh)}
    if args.event not in lookup:
        raise ValueError(f"Candidate {args.event} not in {table_path}")
    labels = tifffile.imread(label_path)
    if labels.ndim != 3:
        raise ValueError("Expected a [time,row,column] event map")
    on = (labels == args.event)
    if not on.any():
        raise ValueError("Event ID missing from event map")
    tt, yy, xx = np.where(on)
    start, stop = int(tt.min()), int(tt.max())
    footprint = on.any(axis=0)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    H, W = footprint.shape
    s = args.downsample
    area = on.sum(axis=(1, 2)).astype(int)
    components = np.zeros(len(area), dtype=int)
    cy = np.full(len(area), np.nan)
    cx = np.full(len(area), np.nan)
    for t in np.flatnonzero(area):
        mask = on[t, y0:y1, x0:x1]
        components[t] = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))[1]
        ry, rx = np.nonzero(mask)
        cy[t] = float(ry.mean() + y0)
        cx[t] = float(rx.mean() + x0)
    with h5py.File(args.input, "r") as h:
        for name in (args.dataset, args.structural):
            if name not in h:
                raise KeyError(f"Dataset {name!r} not present in HDF5")
        ds = h[args.dataset]
        mc = h[args.structural]
        if ds.ndim != 3 or mc.ndim != 3:
            raise ValueError("HDF5 datasets should be 3D [time,row,column]")
        n, h_full, w_full = ds.shape
        if n != len(area) or (h_full + s - 1) // s != H or (w_full + s - 1) // s != W:
            raise ValueError(
                f"Shape mismatch: event map {labels.shape}, dataset {ds.shape}, downsample {s}. "
                "Do not merge images from different resolutions without matching registration."
            )
        if mc.shape != ds.shape:
            raise ValueError("Structural and signal arrays have different shapes")
        # Apply same integer pixel-stride sampling as exvivo_event_screen.py.
        roi_movie = np.asarray(ds[:, y0*s:y1*s:s, x0*s:x1*s:s], dtype=np.float32)
        msk = footprint[y0:y1, x0:x1]
        assert roi_movie.shape == (n, y1-y0, x1-x0)
        finite_cube = np.where(np.isfinite(roi_movie), roi_movie, np.nan)
        mean_fixed = np.nanmean(finite_cube[:, msk], axis=1)
        # Median can distinguish a few bright pixels from population-wide changes.
        median_fixed = np.nanmedian(finite_cube[:, msk], axis=1)
        # Only as descriptive active-pixel statistic; variable mask makes it biased.
        mean_active = np.full(n, np.nan, dtype=float)
        for t in np.flatnonzero(area):
            mean_active[t] = float(np.nanmean(finite_cube[t][on[t, y0:y1, x0:x1]]))
        # Inspect original brightness around each event, without spatial upsampling.
        pad = args.padding
        ya, yb = max(0, y0-pad), min(H, y1+pad)
        xa, xb = max(0, x0-pad), min(W, x1+pad)
        selected = np.linspace(start, stop, min(6, stop-start+1), dtype=int)
        snapshots = [np.asarray(mc[int(t), ya*s:yb*s:s, xa*s:xb*s:s], dtype=np.float32) for t in selected]

    interval = np.arange(start, stop+1)
    peaks_at = int(interval[np.nanargmax(mean_fixed[interval])])
    t = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(t, mean_fixed, label="Mean dF/F across fixed event footprint", lw=1.8)
    ax.plot(t, median_fixed, label="Median dF/F across fixed footprint", alpha=0.75)
    ax.axvspan(start, stop, color="grey", alpha=0.12, label=f"Candidate interval: {start}-{stop}")
    ax.axvline(peaks_at, ls="--", c="black", alpha=0.4)
    ax.set_ylabel("dF/F (stored units)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    axes[1].step(t, area, where="mid", label="Active pixels assigned to this event")
    axes[1].set_ylabel("Active pixels")
    axes[1].set_xlim(max(0, start-20), min(n-1, stop+20))
    axes[1].grid(alpha=0.2)
    axes[2].plot(t, components, marker=".", lw=1.2)
    axes[2].set_ylabel("2D connected regions")
    axes[2].set_xlabel("Frame index (zero-based; not seconds)")
    axes[2].grid(alpha=0.2)
    fig.suptitle(f"Candidate event {args.event} | temporal diagnostics (NOT cell identity)")
    p_trace = out / f"event_{args.event}_timeseries.png"
    fig.savefig(p_trace, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(selected)//2 + len(selected)%2, figsize=(14, 8), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    for ax, k, base in zip(axes, selected, snapshots):
        finite = base[np.isfinite(base)]
        lo, hi = np.percentile(finite, [1, 99.5]) if finite.size else (0, 1)
        ax.imshow(base, cmap="gray", vmin=lo, vmax=max(lo+1e-12,hi))
        event_mask = on[int(k), ya:yb, xa:xb]
        overlay = np.ma.masked_where(~event_mask, event_mask)
        ax.imshow(overlay, cmap="autumn", vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
        ax.set_title(f"Frame {k} | pixels {area[k]} | regions {components[k]}")
        ax.axis("off")
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle(f"Candidate event {args.event}: morphology snapshots (red = event pixels, not a cell)")
    p_spatial = out / f"event_{args.event}_snapshots.png"
    fig.savefig(p_spatial, dpi=150)
    plt.close(fig)

    p_csv = out / f"event_{args.event}_timeseries.csv"
    with p_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "mean_dff_fixed_footprint", "median_dff_fixed_footprint", "mean_dff_active_pixels", "event_active_pixels", "number_2d_regions", "event_center_row_reduced", "event_center_col_reduced"])
        for k in range(n):
            w.writerow([int(k), float(mean_fixed[k]), float(median_fixed[k]), float(mean_active[k]), int(area[k]), int(components[k]), float(cy[k]), float(cx[k])])
    report = {
        "status": "exploratory, unvalidated event candidates",
        "event_id": args.event,
        "frame_start": start, "frame_end": stop,
        "total_event_voxels": int(on.sum()),
        "footprint_pixels_reduced": int(footprint.sum()),
        "max_frame_area": int(area.max()),
        "frame_of_max_area": int(np.argmax(area)),
        "frame_of_max_fixed_footprint_mean_in_event_window": peaks_at,
        "max_simultaneous_2d_regions": int(components.max()),
        "frames_with_multiple_2d_regions": [int(k) for k in np.flatnonzero(components > 1)],
        "downsample_stride": s,
        "source_dataset": args.dataset,
        "note": "Disconnected islands are not automatically separate cells or separate calcium events. Mean dF/F over fixed footprint is a descriptive intensity signal; not a validated response. Frame rate must be confirmed before converting to seconds."
    }
    (out / f"event_{args.event}_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Saved:", p_trace, p_spatial, p_csv)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="real_data/public_data/ExVivoSuppRaw.h5")
    p.add_argument("--dataset", default="dff/ch0")
    p.add_argument("--structural", default="mc/ch0")
    p.add_argument("--results", default="real_data/exvivo_screen")
    p.add_argument("--output", default=None)
    p.add_argument("--event", type=int, default=34)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--padding", type=int, default=12)
    args = p.parse_args()
    if args.downsample < 1 or args.padding < 0 or args.event < 1:
        p.error("Downsample/event/padding argument invalid")
    make_plots(args)

if __name__ == "__main__":
    main()
