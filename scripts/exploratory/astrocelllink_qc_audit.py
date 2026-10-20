#!/usr/bin/env python3
"""Audit exploratory calcium candidates: do NOT interpret detected labels as cell IDs.

Checks whether high-Z pixels of a real frame were retained in a previous
exvivo_event_screen.py candidate map, and displays retained/missing areas.
This is a QC diagnostic, not ground-truth validation or astroCaST itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
from scipy import ndimage as ndi
import tifffile


def limits(image, lo=1, hi=99.5):
    pixels = image[np.isfinite(image)]
    if pixels.size == 0:
        return 0., 1.
    a, b = np.percentile(pixels, [lo, hi])
    return float(a), float(b if b > a else a + 1)


def rgb_label_image(labels):
    """Distinct colors for event IDs; alpha=0 when background."""
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    for label in np.unique(labels):
        if label <= 0:
            continue
        # Golden ratio sequence cycles well even with sparse event IDs.
        hue = (int(label) * 0.61803398875) % 1.0
        rgb = hsv_to_rgb([hue, 0.83, 0.98])
        rgba[labels == label, :3] = rgb
        rgba[labels == label, 3] = 0.72
    return rgba


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('real_data/public_data/ExVivoSuppRaw.h5'))
    p.add_argument('--results', type=Path, default=Path('real_data/exvivo_screen'))
    p.add_argument('--dataset', default='dff/ch0')
    p.add_argument('--frame', type=int, default=None)
    p.add_argument('--sigma-xy', type=float, default=0.8,
                   help='Must match --sigma-xy in initial event screen (default 0.8)')
    p.add_argument('--baseline-step', type=int, default=3,
                   help='Must match baseline_step from initial event screen (default 3)')
    args = p.parse_args()
    s = json.loads((args.results / 'summary.json').read_text())
    stride = int(s['downsample_factor'])
    threshold = float(s['threshold_robust_z'])
    floor = float(s['noise_floor'])
    labels = np.asarray(tifffile.imread(args.results / 'candidate_event_map.tiff'))
    if labels.ndim != 3:
        p.error('Expected 3D candidate_event_map.tiff')
    t = int(s['qc_frame_zero_based']) if args.frame is None else args.frame
    if not 0 <= t < labels.shape[0]:
        p.error(f'frame {t} outside 0..{labels.shape[0]-1}')
    if args.baseline_step < 1 or args.sigma_xy < 0:
        p.error('invalid filter settings')

    with h5py.File(args.input, 'r') as h5:
        ds = h5[args.dataset]
        if ds.ndim != 3:
            p.error(f'{args.dataset} must be [time,row,column]')
        shape = (min(ds.shape[0], labels.shape[0]),
                 (ds.shape[1] + stride - 1) // stride,
                 (ds.shape[2] + stride - 1) // stride)
        if tuple(labels.shape) != shape:
            p.error(f'Candidate map {labels.shape} disagrees with input {shape}; check inputs/downsampling')
        frames = list(range(0, shape[0], args.baseline_step))
        sample = np.stack([ds[i, ::stride, ::stride] for i in frames]).astype('float32')
        # The original screen smooths each frame independently in XY, not across time.
        sample = ndi.gaussian_filter(sample, sigma=(0, args.sigma_xy, args.sigma_xy), mode='reflect')
        baseline = np.median(sample, axis=0)
        mad = 1.4826 * np.median(np.abs(sample - baseline), axis=0)
        denom = np.maximum(mad, floor)
        signal = np.asarray(ds[t, ::stride, ::stride], dtype=np.float32)
        signal = ndi.gaussian_filter(signal, args.sigma_xy, mode='reflect')
        z = (signal - baseline) / denom
        if 'mc/ch0' in h5:
            structural = np.asarray(h5['mc/ch0'][t, ::stride, ::stride], dtype=np.float32)
        else:
            structural = np.asarray(ds[t, ::stride, ::stride], dtype=np.float32)

    if not np.all(np.isfinite(z)):
        p.error('nonfinite samples in input; rerun with cleaned data, because this diagnostic cannot exactly reproduce initial global fill')
    retained = labels[t] > 0
    active = z >= threshold
    missing = active & ~retained
    covered = active & retained

    label2d, n = ndi.label(active, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(label2d.ravel(), minlength=n+1)
    largest = np.argsort(sizes[1:])[-5:][::-1] + 1
    clusters = []
    for idx in largest:
        if sizes[idx] == 0:
            continue
        pix = label2d == idx
        row, col = ndi.center_of_mass(pix)
        overlapping_ids = np.unique(labels[t][pix & retained]).astype(int)
        clusters.append({
            'size_active_pixels': int(sizes[idx]),
            'centroid_row': round(float(row), 2),
            'centroid_col': round(float(col), 2),
            'kept_fraction': round(float(np.count_nonzero(pix & retained) / sizes[idx]), 4),
            'retained_event_ids': overlapping_ids.tolist()[:20],
        })

    edge_ids = np.unique(np.concatenate((labels[:, 0, :].ravel(), labels[:, -1, :].ravel(),
                                         labels[:, :, 0].ravel(), labels[:, :, -1].ravel())))
    edge_ids = edge_ids[edge_ids != 0]
    summary = {
        'frame': t,
        'threshold_robust_z': threshold,
        'frame_high_z_pixels_before_filter': int(np.count_nonzero(active)),
        'frame_high_z_pixels_retained': int(np.count_nonzero(covered)),
        'frame_high_z_pixels_not_retained': int(np.count_nonzero(missing)),
        'fraction_high_z_pixels_retained': round(float(np.count_nonzero(covered) / max(np.count_nonzero(active), 1)), 4),
        'frame_retained_pixels_total': int(np.count_nonzero(retained)),
        'frame_distinct_retained_event_ids': int(np.unique(labels[t][retained]).size),
        'events_touching_image_border_any_frame': int(edge_ids.size),
        'largest_high_z_regions': clusters,
        'caution': 'Descriptive pixel QC only, not accuracy or biological validation; image center need not equal one cell.',
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / 'diagnostic_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    fig, ax = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    a, b, c, d = ax.flat
    lo, hi = limits(structural)
    for subplot in [a, c]:
        subplot.imshow(structural, cmap='gray', vmin=lo, vmax=hi)
    a.contour(retained.astype(np.uint8), levels=[0.5], colors=['cyan'], linewidths=0.8)
    a.set_title(f'Motion corrected + retained boundaries | frame {t}')
    b.imshow(np.clip(z, 0, 12), cmap='magma', vmin=0, vmax=12)
    b.contour(active.astype(np.uint8), levels=[0.5], colors=['lime'], linewidths=0.6)
    b.set_title(f'Robust Z | lime: above threshold {threshold:g}')
    c.imshow(rgb_label_image(labels[t]))
    c.set_title('Retained candidate IDs | colors NOT cell IDs')
    d.imshow(structural, cmap='gray', vmin=lo, vmax=hi)
    mask = np.zeros((*retained.shape, 4), dtype=np.float32)
    mask[missing] = (1, 0.35, 0.05, 0.7)  # above threshold but discarded
    mask[covered] = (0.1, 0.95, 0.55, 0.65)  # above threshold and retained
    d.imshow(mask)
    d.set_title(f'Green: retained {np.count_nonzero(covered)} | orange: discarded {np.count_nonzero(missing)}')
    for subplot in ax.flat:
        subplot.set_axis_off()
    fig.suptitle('EXPLORATORY QC — not astroCaST; not cell identification', fontsize=13)
    outfile = args.results / f'diagnostic_frame_{t}.png'
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print('Saved:', outfile)


if __name__ == '__main__':
    main()
