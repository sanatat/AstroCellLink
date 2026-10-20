#!/usr/bin/env python3
"""Exploratory spatial-band trace comparison for a previously detected candidate event.

Independent *spatial* regions are derived from the event's pooled footprint,
not from the per-pixel onset colors. Measures empirical fluorescence timing;
it DOES NOT establish a calcium wave, event independence, or cell identities.
Matches the [::downsample, ::downsample] sampling in exvivo_event_screen.py.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def onset_frame(times, signal, start, stop):
    """Return descriptive rising-threshold onset and peak with a pre-event noise guard."""
    pre = signal[times < start]
    if pre.size < 4 or not np.all(np.isfinite(signal)):
        return None, None, None, None
    base = float(np.median(pre))
    mad = float(1.4826 * np.median(np.abs(pre - base)))
    delta = signal - base
    window = (times >= start) & (times <= stop)
    peak = float(np.max(delta[window]))
    if peak <= 0:
        return None, None, base, peak
    threshold = max(0.2 * peak, 3.0 * mad)
    crossing = np.flatnonzero(window & (delta >= threshold))
    onset = int(times[crossing[0]]) if len(crossing) else None
    peak_frame = int(times[np.flatnonzero(window)[int(np.argmax(delta[window]))]])
    return onset, peak_frame, base, peak


def run(args):
    path = Path(args.event_map)
    if path.is_dir():
        path = path / 'event_map.tiff'
    labels = tifffile.imread(path)
    if labels.ndim != 3:
        raise ValueError(f'Expected event_map [time,row,col], got {labels.shape}')
    event = labels == args.event
    if not event.any():
        raise ValueError(f'Event ID {args.event} not present in {path}')
    active_times = np.flatnonzero(np.any(event, axis=(1,2)))
    start, stop = int(active_times[0]), int(active_times[-1])
    footprint = np.any(event, axis=0)
    rr, cc = np.nonzero(footprint)
    y0, y1 = int(rr.min()), int(rr.max())+1
    x0, x1 = int(cc.min()), int(cc.max())+1
    # Boundaries by x quantiles of the footprint, which are defined without onset times.
    boundaries = np.quantile(cc, [1/3, 2/3])
    yygrid, xxgrid = np.indices(footprint.shape)
    regions = [
        footprint & (xxgrid < boundaries[0]),
        footprint & (xxgrid >= boundaries[0]) & (xxgrid < boundaries[1]),
        footprint & (xxgrid >= boundaries[1]),
    ]
    if min(np.count_nonzero(r) for r in regions) < 5:
        raise ValueError('Event footprint too small for three spatial bands')
    names = ['Left', 'Middle', 'Right']
    colors = ['#2375be', '#17975d', '#e37a25']
    lo = max(0, start - 12)
    hi = min(labels.shape[0], stop + 13)
    time = np.arange(lo, hi)

    with h5py.File(args.input, 'r') as hf:
        d = hf[args.dataset]
        if d.ndim != 3 or d.shape[0] != labels.shape[0]:
            raise ValueError(f'Invalid signal dimensions: {d.shape} vs map {labels.shape}')
        stride = args.downsample
        if (d.shape[1] + stride-1)//stride != labels.shape[1] or (d.shape[2] + stride-1)//stride != labels.shape[2]:
            raise ValueError('Spatial mismatch; verify the downsample and dataset registration')
        # Read only the event bounding box; apply same spatial stride as detection.
        cube = np.asarray(d[lo:hi, y0*stride:y1*stride:stride, x0*stride:x1*stride:stride], dtype=np.float32)
        if cube.shape[1:] != (y1-y0, x1-x0):
            raise ValueError('Downsampled movie does not align with the event footprint')
        if args.structural in hf:
            ref_t = min(max(start, (start+stop)//2), len(d)-1)
            bg = np.asarray(hf[args.structural][ref_t, ::stride, ::stride], dtype=np.float32)
        else:
            bg = None

    results = {}
    traces = {}
    for name, region in zip(names, regions):
        m = region[y0:y1, x0:x1]
        v = np.asarray(cube[:, m], dtype=np.float64)
        v[~np.isfinite(v)] = np.nan
        # Fixed footprints, no moving ROI selection; spatial median reduces bright outliers.
        trace = np.nanmedian(v, axis=1)
        valid = np.isfinite(trace)
        if not valid.all():
            if not valid.any():
                raise ValueError(f'All signal samples invalid in {name} region')
            trace[~valid] = np.interp(time[~valid], time[valid], trace[valid])
        trace = gaussian_filter1d(trace, sigma=0.8)
        onset, peak_frame, baseline, amplitude = onset_frame(time, trace, start, stop)
        traces[name] = (trace, baseline, amplitude)
        results[name.lower()] = {
            'n_footprint_pixels': int(m.sum()),
            'onset_frame_20pct_with_noise_guard': onset,
            'peak_frame': peak_frame,
            'baseline_stored_units': baseline,
            'peak_above_baseline_stored_units': amplitude,
        }

    fig, axes = plt.subplots(1,2,figsize=(13,5.5), constrained_layout=True)
    ax = axes[0]
    if bg is not None:
        finite = bg[np.isfinite(bg)]
        if finite.size:
            v0,v1=np.percentile(finite,[2,99])
            ax.imshow(bg, cmap='gray', vmin=v0, vmax=max(v1,v0+1e-6))
    else:
        ax.imshow(footprint, cmap='gray')
    for n, m, color in zip(names, regions, colors):
        overlay = np.ma.masked_where(~m, np.ones(m.shape))
        from matplotlib.colors import ListedColormap
        ax.imshow(overlay, cmap=ListedColormap([color]), vmin=0,vmax=1, alpha=0.72, interpolation='nearest')
        ry,rx=np.nonzero(m)
        ax.text(float(rx.mean()),float(ry.mean()),n[0],color='white',weight='bold',ha='center',va='center',fontsize=13)
    ax.set_xlim(max(0,x0-15),min(footprint.shape[1],x1+15))
    ax.set_ylim(min(footprint.shape[0],y1+15),max(0,y0-15))
    ax.set_title('Fixed spatial bands (not onset-based)')
    ax.set_xlabel('Column, downsampled pixels')
    ax.set_ylabel('Row, downsampled pixels')

    ax=axes[1]
    for name,color in zip(names,colors):
        trace,base,amp=traces[name]
        normalized=(trace-base)/amp if amp is not None and amp>0 else np.full(len(time),np.nan)
        onset=results[name.lower()]['onset_frame_20pct_with_noise_guard']
        ax.plot(time,normalized,label=f'{name} (onset {onset})',color=color,lw=2)
        if onset is not None:
            k=onset-lo
            ax.scatter([onset],[normalized[k]],s=35,color=color,zorder=4)
    ax.axhline(.2,linestyle=':',color='0.4',alpha=.7,label='20% relative peak')
    ax.axvspan(start, stop, color='gray',alpha=.12,label=f'Event {start}-{stop}')
    ax.set_xlabel('Frame index (not seconds)')
    ax.set_ylabel('Baseline-subtracted signal / each band peak')
    ax.set_title('Measured dF/F time courses in fixed regions')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    fig.suptitle(f'Event {args.event} – exploratory timing comparison; NOT proof of a calcium wave')
    fig.savefig(out/f'event_{args.event}_spatial_lags.png',dpi=175)
    plt.close(fig)
    summary={
        'status':'exploratory; no proof of propagation or cell identity',
        'event_id':int(args.event),'event_start_frame':start,'event_last_frame':stop,
        'image_sampling':'pixel stride [::downsample] matches exvivo_event_screen.py',
        'regions':'3 equal-footprint-size x-axis bands; NOT defined by onset values',
        'onset_criterion':'first sample in event duration > max(20% own peak increase,3*pre-event MAD); smoothed signal',
        'caveat':'Amplitude rise can yield apparent lag without physical propagation; check alternate thresholds, raw imaging and independent labels.',
        'bands':results,
    }
    with open(out/f'event_{args.event}_spatial_lags.json','w') as fh:
        json.dump(summary,fh,indent=2)
    print(json.dumps(summary,indent=2))
    print('Plot:',out/f'event_{args.event}_spatial_lags.png')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--event-map',required=True)
    p.add_argument('--event',type=int,default=34)
    p.add_argument('--dataset',default='dff/ch0')
    p.add_argument('--structural',default='mc/ch0')
    p.add_argument('--downsample',type=int,default=2)
    p.add_argument('--output',default='real_data/exvivo_screen')
    a=p.parse_args()
    if a.downsample<1:p.error('--downsample must be positive')
    run(a)
