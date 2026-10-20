#!/usr/bin/env python3
"""Compare mask-derived and fluorescence-derived onset gradients for a candidate event.

Research-only exploratory diagnostic. Event footprint is selected from the
original thresholded event map, so this is NOT an independent biological
validation, calcium-wave test, or astrocyte identity inference.

Requires numpy, scipy, h5py, tifffile, matplotlib. Tested with Python 3.12.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import tifffile


def fit_plane(x, y, onset):
    """Unweighted OLS of t=a+bx*x+by*y in image coordinates (y downward)."""
    t = np.asarray(onset, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(t) < 10 or np.std(t) < 1e-9:
        return None
    xx = x - np.mean(x)
    yy = y - np.mean(y)
    X = np.column_stack((xx, yy, np.ones_like(t)))
    bx, by, intercept = np.linalg.lstsq(X, t, rcond=None)[0]
    predictions = X @ [bx, by, intercept]
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1 - float(np.sum((t - predictions) ** 2)) / ss_tot
    mag = float(np.hypot(bx, by))
    return {'n_pixels': int(len(t)), 'bx_frames_per_reduced_pixel': round(float(bx), 5),
            'by_frames_per_reduced_pixel': round(float(by), 5),
            'r2': round(float(r2), 4),
            'later_direction_degrees_image_xy': round(float(np.degrees(np.arctan2(by, bx)) % 360), 2)
                 if mag > 1e-10 else None}


def analyze(event_map, movie, event_id, *, preframes=30, postframes=7,
            downsample=2, dataset='dff/ch0', sigma=0.8,
            fractions=(0.1, 0.2, 0.3, 0.5), noise_mult=3.0):
    if event_map.ndim != 3 or movie.ndim != 3:
        raise ValueError('Both event map and source movie must be (time, y, x).')
    if event_map.shape[0] != movie.shape[0] or event_map.shape[1:] != tuple(
            (i + downsample - 1) // downsample for i in movie.shape[1:]):
        raise ValueError('Shape mismatch: the event map must align with movie[::, ::downsample, ::downsample].')
    active = event_map == event_id
    frames = np.flatnonzero(np.any(active, axis=(1, 2)))
    if frames.size == 0:
        raise ValueError(f'Event ID {event_id} not found.')
    start, stop = int(frames[0]), int(frames[-1])
    lo = max(0, start - preframes)
    hi = min(movie.shape[0], stop + postframes + 1)
    if start - lo < 10:
        raise ValueError('At least 10 pre-event frames required for per-pixel noise estimation.')
    window = active[start:stop + 1]
    footprint = window.any(axis=0)
    onset_mask = np.argmax(window, axis=0) + start
    yy, xx = np.nonzero(footprint)
    if len(xx) < 20:
        raise ValueError('Event footprint too small (<20 pixels).')
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    # HDF5 data read only for this event bbox and a short time interval.
    sub = np.asarray(movie[lo:hi,
                           y0 * downsample:y1 * downsample:downsample,
                           x0 * downsample:x1 * downsample:downsample], dtype=np.float32)
    if sub.shape != (hi - lo, y1 - y0, x1 - x0):
        raise ValueError(f'Unexpected ROI crop shape: {sub.shape}')
    px_traces = sub[:, yy - y0, xx - x0].astype(np.float64)
    smoothed = gaussian_filter1d(px_traces, sigma=sigma, axis=0) if sigma > 0 else px_traces
    pre = smoothed[:start-lo]
    baseline = np.median(pre, axis=0)
    robust_sigma = 1.4826 * np.median(abs(pre - baseline), axis=0)
    # Conservative floor for vanishing-noise pixels, in the stored image units.
    floor_candidates = robust_sigma[np.isfinite(robust_sigma) & (robust_sigma > 0)]
    noise_floor = max(float(np.median(floor_candidates)) * 0.2, 1e-6) if floor_candidates.size else 1e-6
    robust_sigma = np.maximum(robust_sigma, noise_floor)
    event_delta = smoothed[start-lo:stop-lo+1] - baseline
    peak_rise = np.max(event_delta, axis=0)
    # Require at least two consecutive frames above threshold.
    output = {}
    common_valid = np.isfinite(smoothed).all(axis=0)
    eligible = common_valid & (peak_rise >= noise_mult * robust_sigma)
    for fraction in fractions:
        thresh = np.maximum(float(fraction) * peak_rise, noise_mult * robust_sigma)
        crossed = event_delta >= thresh[None, :]
        consecutive = crossed[:-1] & crossed[1:]
        usable = eligible & np.any(consecutive, axis=0)
        first_local = np.argmax(consecutive, axis=0)
        first_frame = start + first_local
        bandcut = np.quantile(xx, [1/3, 2/3])
        # Bands fixed from original footprint; not selected from onset data.
        thirds = {'left': xx <= bandcut[0],
                  'middle': (xx > bandcut[0]) & (xx <= bandcut[1]),
                  'right': xx > bandcut[1]}
        med = {name: float(np.median(first_frame[usable & mask]))
               if (usable & mask).sum() >= 3 else None
               for name, mask in thirds.items()}
        output[f'{fraction:.2f}'] = {
            'detected_pixels': int(usable.sum()),
            'footprint_pixels': int(len(xx)),
            'detected_fraction': round(float(usable.mean()), 3),
            'median_frames_by_band': med,
            'plane': fit_plane(xx[usable], yy[usable], first_frame[usable]),
            '_onsets': first_frame, '_usable': usable,
        }
    summary = {
        'status': 'EXPLORATORY; not independent biological validation or proof of calcium propagation',
        'event_id': event_id, 'event_start_frame': start, 'event_last_frame': stop,
        'movie_dataset': dataset, 'downsample': downsample,
        'pre_event_frames': int(start-lo), 'temporal_sigma_frames': float(sigma),
        'two_consecutive_frames_required': True, 'noise_guard_multiplier': noise_mult,
        'source': 'per-pixel fluorescence traces in fixed candidate-event footprint',
        'mask_plane': fit_plane(xx, yy, onset_mask[footprint]),
        'fractions': {fraction: {k:v for k,v in d.items() if not k.startswith('_')}
                      for fraction, d in output.items()},
        'limitations': [
            'Event footprint was selected from a thresholded fluorescence candidate map (selection dependence).',
            'Per-pixel thresholds and smoothing can bias first-crossing times; noise is spatially correlated.',
            'R² and angles are descriptive; no p-value or physical velocity is established.',
            'No segmentation ground truth, independent experiments or calcium-wave validation.'
        ]
    }
    aux = {'xx': xx, 'yy': yy, 'fp': footprint, 'mask_onset': onset_mask,
           'smoothed_data': smoothed, 'summary_by_fraction': output, 'bounds': (y0,y1,x0,x1),
           'frame_for_bg': min(stop, max(start, (start + stop) // 2))}
    return summary, aux


def plot_summary(summary, aux, movie, output: Path):
    xx, yy, fp = aux['xx'], aux['yy'], aux['fp']
    y0,y1,x0,x1 = aux['bounds']
    ds = summary['downsample']
    frame = aux['frame_for_bg']
    bg = np.asarray(movie[frame, y0*ds:y1*ds:ds, x0*ds:x1*ds:ds])
    validbg = bg[np.isfinite(bg)]
    low, high = np.percentile(validbg, [2,99.5]) if validbg.size else (0,1)
    if high <= low: high = low + 1
    mask_map = np.full(fp.shape, np.nan)
    mask_map[yy-y0, xx-x0] = aux['mask_onset'][yy,xx]
    frac_key = '0.20' if '0.20' in aux['summary_by_fraction'] else list(aux['summary_by_fraction'])[0]
    info = aux['summary_by_fraction'][frac_key]
    sig_map = np.full(fp.shape, np.nan)
    sig_map[yy[info['_usable']]-y0, xx[info['_usable']]-x0] = info['_onsets'][info['_usable']]
    cmap = plt.colormaps['turbo'].copy()
    cmap.set_bad((0,0,0,0))
    fig, axes = plt.subplots(2, 2, figsize=(12,10), constrained_layout=True)
    for ax, vals, name in [
        (axes[0,0],mask_map,'Event-label first-frame (old method)'),
        (axes[0,1],sig_map,f'Fluorescence first crossing ({float(frac_key)*100:g}% rise, 2 frames)')]:
        ax.imshow(bg, cmap='gray', vmin=low, vmax=high)
        im=ax.imshow(vals,cmap=cmap,vmin=summary['event_start_frame'],vmax=summary['event_last_frame'],alpha=.84)
        ax.set_title(name,fontsize=10)
        ax.set_xlabel('X within crop (reduced px)')
        ax.set_ylabel('Y within crop (reduced px; down)')
    fig.colorbar(im,ax=[axes[0,0],axes[0,1]],label='First activation frame',shrink=.68)
    ax=axes[1,0]
    thr=[]; bxs=[]; bys=[]
    for key, entry in summary['fractions'].items():
        thr.append(int(round(100*float(key))))
        p=entry['plane']
        bxs.append(p['bx_frames_per_reduced_pixel'] if p else np.nan)
        bys.append(p['by_frames_per_reduced_pixel'] if p else np.nan)
    ax.plot(thr,bxs,'o-',label='bx (left–right)')
    ax.plot(thr,bys,'o-',label='by (top–bottom)')
    ax.axhline(0,color='gray',linewidth=.8)
    ax.set_xlabel('Per-pixel rise threshold (%)')
    ax.set_ylabel('Gradient (frames per reduced pixel)')
    ax.set_title('Slope sensitivity (not a wave probability)')
    ax.legend()
    ax=axes[1,1]
    for name in ('left','middle','right'):
        vals = [info['median_frames_by_band'][name] for info in summary['fractions'].values()]
        ax.plot(thr,[np.nan if v is None else v for v in vals],'o-',label=name)
    ax.set_xlabel('Per-pixel rise threshold (%)')
    ax.set_ylabel('Median first-crossing frame')
    ax.set_title('Fixed spatial bands: fluorescence-derived onsets')
    ax.legend()
    fig.suptitle('AstroCellLink | independent trace timing conditional on original event footprint\nNOT biological validation / NO velocity or p-values', fontsize=12)
    fig.savefig(output,dpi=155)
    plt.close(fig)


def run_self_test():
    rng=np.random.default_rng(42)
    T,H,W=90,50,50
    xg,yg=np.meshgrid(np.arange(W),np.arange(H))
    fp=((xg-25)**2/15**2 + (yg-25)**2/13**2)<1
    event = np.zeros((T,H,W),dtype=np.uint16)
    onset_true = 35 + np.rint((xg-10)*0.27).astype(int)
    sig = rng.normal(10,0.13,(T,H,W)).astype(np.float32)
    for y,x in zip(*np.where(fp)):
        s=int(onset_true[y,x]); event[s:s+8,y,x]=7
        delta=np.arange(T)-s
        sig[:,y,x]+=np.where(delta>=0,12*np.exp(-((delta-4)/3.2)**2),0)
    summary,_=analyze(event,sig,7,preframes=25,downsample=1,
                      fractions=(.2,.3,.5))
    p=summary['fractions']['0.20']['plane']
    assert p is not None and p['bx_frames_per_reduced_pixel'] > 0.05, p
    assert p['r2'] > 0.3, p
    assert summary['fractions']['0.20']['detected_pixels']>100
    print('Synthetic sanity test PASS:',json.dumps(p))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,help='Source HDF5 file')
    parser.add_argument('--event-map',type=Path,help='Existing candidate_event_map.tiff')
    parser.add_argument('--event',type=int,default=194)
    parser.add_argument('--dataset',default='dff/ch0')
    parser.add_argument('--downsample',type=int,default=2)
    parser.add_argument('--preframes',type=int,default=30)
    parser.add_argument('--postframes',type=int,default=7)
    parser.add_argument('--sigma',type=float,default=0.8)
    parser.add_argument('--output',type=Path,default=Path('real_data/exvivo_screen'))
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.input or not args.event_map:
        parser.error('--input and --event-map are required unless --self-test')
    if args.downsample<1 or args.sigma<0:
        parser.error('downsample must be >=1, sigma must be >=0')
    labels = tifffile.imread(args.event_map)
    with h5py.File(args.input,'r') as h5:
        if args.dataset not in h5:
            raise KeyError(f'{args.dataset!r} is missing from {args.input}')
        dset=h5[args.dataset]
        report, aux=analyze(labels,dset,args.event,preframes=args.preframes,
                            postframes=args.postframes,downsample=args.downsample,
                            dataset=args.dataset,sigma=args.sigma)
        args.output.mkdir(parents=True,exist_ok=True)
        pic=args.output / f'event_{args.event}_signal_onset_check.png'
        plot_summary(report,aux,dset,pic)
    js=args.output / f'event_{args.event}_signal_onset_check.json'
    with open(js,'w',encoding='utf-8') as f:
        json.dump(report,f,indent=2)
    print(json.dumps({'event_id':args.event,'mask_plane':report['mask_plane'],
                      'fluorescence_20pct':report['fractions']['0.20']},indent=2))
    print(f'Plot: {pic}\nJSON: {js}')


if __name__=='__main__':
    main()
