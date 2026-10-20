#!/usr/bin/env python3
"""AstroCellLink: sensitivity screening of apparent event timing across fixed spatial bands.

Research prototype, NOT astroCaST itself. No biological validation, cell ID,
wave propagation, velocity or probability can be inferred from this output.

Uses event_map label IDs (T,Y,X) and aligned HDF5 fluorescence (T,Y*ds,X*ds).
Regions are left/middle/right thirds by UNIQUE event footprint pixels,
not defined using event onsets. Unlike the earlier single-threshold script,
it sweeps thresholds and temporal smoothing and compares independent
cross-correlation lag estimates. Positive LR lag means right follows left.
"""
import argparse
import csv
import json
import tempfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter1d

FRACTIONS = (0.10, 0.20, 0.30, 0.50)
SMOOTH_SIGMAS = (0.0, 0.8, 1.5)
BAND_NAMES = ('left', 'middle', 'right')


def label_coordinates(labels):
    """Index all non-zero event voxels once; avoids rescanning movie per ID."""
    z, y, x = np.nonzero(labels)
    ids = labels[z, y, x]
    order = np.argsort(ids, kind='stable')
    ids = ids[order]
    z, y, x = z[order], y[order], x[order]
    unique, offsets, counts = np.unique(ids, return_index=True, return_counts=True)
    return {int(event_id): (z[start:start+n], y[start:start+n], x[start:start+n])
            for event_id, start, n in zip(unique, offsets, counts)}


def footprint_bands(coords, min_pixels):
    z, y, x = coords
    y0, y1 = int(y.min()), int(y.max()) + 1
    x0, x1 = int(x.min()), int(x.max()) + 1
    fp = np.zeros((y1-y0, x1-x0), dtype=bool)
    fp[y-y0, x-x0] = True
    yy, xx = np.nonzero(fp)
    if len(xx) < 3*min_pixels:
        raise ValueError(f'footprint_too_small ({len(xx)} pixels)')
    cut1, cut2 = np.quantile(xx, [1/3, 2/3])
    grid_x = np.indices(fp.shape)[1]
    bands = (fp & (grid_x < cut1),
             fp & (grid_x >= cut1) & (grid_x < cut2),
             fp & (grid_x >= cut2))
    sizes = [int(np.sum(m)) for m in bands]
    if min(sizes) < min_pixels:
        raise ValueError(f'unbalanced_or_tiny_band ({sizes})')
    return (y0, y1, x0, x1), fp, bands, sizes


def rising_onset(times, smoothed, start, stop, frac):
    """First crossing of max(frac * peak rise, 3 * pre-event robust SD).

    Reproduces the *criterion* of the prior spatial-lags script; also
    varies sigma. It is descriptive, NOT a validated onset estimator.
    """
    pre = smoothed[times < start]
    valid_ev = (times >= start) & (times <= stop)
    if len(pre) < 8 or not np.isfinite(smoothed).all() or not np.any(valid_ev):
        return None, None, None, None, None
    baseline = float(np.median(pre))
    mad = float(1.4826 * np.median(np.abs(pre - baseline)))
    delta = smoothed - baseline
    peak = float(np.max(delta[valid_ev]))
    if peak <= max(3 * mad, 1e-10):
        return None, None, baseline, peak, mad
    threshold = max(frac * peak, 3.0 * mad)
    crosses = np.flatnonzero(valid_ev & (delta >= threshold))
    onset = int(times[crosses[0]]) if crosses.size else None
    pk = int(times[np.flatnonzero(valid_ev)[np.argmax(delta[valid_ev])]])
    return onset, pk, baseline, peak, mad


def corr_at_lag(ref, target, lag):
    """Corr(ref[t], target[t+lag]); positive => target later."""
    n = len(ref)
    if lag >= 0:
        a, b = ref[:n-lag], target[lag:]
    else:
        a, b = ref[-lag:], target[:n+lag]
    if len(a) < 9 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else None


def xcorr(ref, target, max_lag):
    available = {lag: corr_at_lag(ref, target, lag)
                 for lag in range(-max_lag, max_lag+1)}
    possible = [(lag, c) for lag, c in available.items() if c is not None]
    if not possible:
        return {'lag': None, 'best_correlation': None, 'zero_correlation': None,
                'at_search_boundary': None, 'curve': available}
    lag, best = max(possible, key=lambda pair: pair[1])
    return {'lag': int(lag), 'best_correlation': round(best, 4),
            'zero_correlation': (round(available[0], 4) if available[0] is not None else None),
            'at_search_boundary': bool(abs(lag) == max_lag), 'curve': available}


def read_movie_cube(dset, frame0, frame1, bbox, downsample, movie=None):
    y0, y1, x0, x1 = bbox
    if movie is not None:
        cube = np.asarray(movie[frame0:frame1, y0:y1, x0:x1])
    else:
        cube = np.asarray(dset[frame0:frame1,
                               y0*downsample:y1*downsample:downsample,
                               x0*downsample:x1*downsample:downsample], dtype=np.float32)
    expected = (frame1-frame0, y1-y0, x1-x0)
    if cube.shape != expected:
        raise ValueError(f'cube_shape_mismatch ({cube.shape} vs {expected})')
    return cube


def analyze_event(event_id, coords, dataset, frame_count, downsample, args, movie=None):
    z, y, x = coords
    start, stop = int(z.min()), int(z.max())
    dur = stop-start+1
    if dur < args.min_duration:
        raise ValueError(f'event_short ({dur} frames)')
    if start < 8:
        raise ValueError(f'insufficient_pre_event_baseline (event starts frame {start})')
    bbox, footprint, bands, sizes = footprint_bands(coords, args.min_band_pixels)
    lo = max(0, start-args.preframes)
    hi = min(frame_count, stop+args.postframes+1)
    if start-lo < 8:
        raise ValueError('insufficient_pre_event_samples')
    ts = np.arange(lo, hi)
    cube = read_movie_cube(dataset, lo, hi, bbox, downsample, movie)
    traces = {}
    for name, band in zip(BAND_NAMES, bands):
        vals = np.where(np.isfinite(cube[:, band]), cube[:, band], np.nan)
        with np.errstate(invalid='ignore'):
            trace = np.nanmedian(vals, axis=1).astype(np.float64)
        if not np.isfinite(trace).all():
            raise ValueError(f'nonfinite_band_trace_{name}')
        traces[name] = trace

    rows = []
    for sigma in SMOOTH_SIGMAS:
        signals = {n: (gaussian_filter1d(traces[n], sigma=sigma) if sigma else traces[n])
                   for n in BAND_NAMES}
        for fraction in FRACTIONS:
            ons, peaks, amps = {}, {}, {}
            for n in BAND_NAMES:
                onset, peak, base, amp, mad = rising_onset(ts, signals[n], start, stop, fraction)
                ons[n] = onset
                peaks[n] = peak
                amps[n] = amp
            lag_lr = (ons['right']-ons['left'] if ons['right'] is not None and ons['left'] is not None else None)
            lag_lm = (ons['middle']-ons['left'] if ons['middle'] is not None and ons['left'] is not None else None)
            rows.append({'event_id': event_id, 'sigma': sigma, 'threshold_fraction': fraction,
                         'left_onset': ons['left'], 'middle_onset': ons['middle'], 'right_onset': ons['right'],
                         'middle_minus_left': lag_lm, 'right_minus_left': lag_lr,
                         'left_peak': peaks['left'], 'middle_peak': peaks['middle'], 'right_peak': peaks['right']})

    # Peak-normalized traces for cross correlation; only look near event.
    sig = {n: gaussian_filter1d(traces[n], sigma=0.8) for n in BAND_NAMES}
    segment = (ts >= max(lo, start-5)) & (ts <= min(hi-1, stop+7))
    def normalized(n):
        baseline = np.median(sig[n][ts < start])
        amp = np.max(sig[n][(ts >= start) & (ts <= stop)] - baseline)
        return (sig[n][segment] - baseline) / amp if amp > 0 else np.zeros(segment.sum())
    xc = {key: xcorr(normalized(a), normalized(b), args.max_lag)
          for key, (a, b) in {'LR': ('left', 'right'), 'LM': ('left', 'middle'),
                              'MR': ('middle', 'right')}.items()}

    usable_lags = [r['right_minus_left'] for r in rows if r['right_minus_left'] is not None]
    def median_or_none(values):
        return float(np.median(values)) if values else None
    summary = {'event_id': event_id, 'start_frame': start, 'stop_frame': stop,
               'duration_frames': dur, 'footprint_pixels': int(footprint.sum()),
               'band_footprint_pixels': dict(zip(BAND_NAMES, sizes)),
               'right_minus_left_onset_lag_median_frames': median_or_none(usable_lags),
               'right_minus_left_onset_lag_range_frames': [min(usable_lags), max(usable_lags)] if usable_lags else None,
               'right_minus_left_lag_sign_consistency': (
                   round(max(sum(i < 0 for i in usable_lags), sum(i == 0 for i in usable_lags),
                             sum(i > 0 for i in usable_lags))/len(usable_lags), 3)
                   if usable_lags else None),
               'valid_combinations': len(usable_lags), 'tested_combinations': len(rows),
               'cross_correlation': {k: {key: val for key, val in v.items() if key != 'curve'}
                                     for k, v in xc.items()},
               'note': 'Screening only: onset lags and correlation cannot prove calcium waves or cell identities; positive LR lag means right follows left.'}
    return summary, rows, {'times': ts, 'traces': traces, 'start': start, 'stop': stop, 'xc': xc}


def make_plot(event_id, detail, rows, summary, output):
    times = detail['times']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    colors = {'left':'#2375be', 'middle':'#17975d', 'right':'#e37a25'}
    for n in BAND_NAMES:
        s = gaussian_filter1d(detail['traces'][n], sigma=0.8)
        b = np.median(s[times < detail['start']])
        a = max(np.max(s[(times >= detail['start']) & (times <= detail['stop'])]-b), 1e-8)
        ax[0].plot(times, (s-b)/a, label=n, c=colors[n])
    ax[0].axvspan(detail['start'], detail['stop'], color='gray', alpha=.12)
    ax[0].set(title='Fluorescence traces (fixed footprints)', xlabel='Frame', ylabel='Baseline-subtracted / own peak')
    ax[0].legend()
    ax[0].grid(alpha=.25)
    for sigma in SMOOTH_SIGMAS:
        subset = [r for r in rows if r['sigma']==sigma]
        ax[1].plot([int(100*r['threshold_fraction']) for r in subset],
                   [r['right_minus_left'] if r['right_minus_left'] is not None else np.nan for r in subset],
                   marker='o', label=f'sigma {sigma}')
    ax[1].axhline(0, ls=':', c='gray')
    ax[1].set(title='Right onset minus left onset', xlabel='Threshold (% of rise)', ylabel='Frames (+ means right later)')
    ax[1].legend()
    ax[1].grid(alpha=.25)
    for key,color in [('LR','#ce6630'),('LM','#3565be'),('MR','#38915a')]:
        curve = detail['xc'][key]['curve']
        ax[2].plot(list(curve), [np.nan if a is None else a for a in curve.values()],
                   marker='.', label=key, color=color)
    ax[2].axvline(0,ls=':',color='gray')
    ax[2].set(title='Normalized cross-correlation', xlabel='Lag in frames (+ means second band later)',
              ylabel='Pearson correlation')
    ax[2].legend()
    ax[2].grid(alpha=.25)
    fig.suptitle(f'Event {event_id}: sensitivity analysis ONLY — not biological validation')
    fig.savefig(output, dpi=150)
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(args):
    path = Path(args.event_map)
    if path.is_dir():
        path = path / 'event_map.tiff'
    labels = tifffile.imread(path)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'event-map must be a 3-D integer label movie, got {labels.shape} {labels.dtype}')
    indexed = label_coordinates(labels)
    if not indexed:
        raise ValueError('No nonzero event IDs in event map')
    if not args.all and args.event not in indexed:
        raise ValueError(f'Event {args.event} not in event map')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target_ids = sorted(indexed) if args.all else [args.event]
    with h5py.File(args.input, 'r') as f:
        d = f[args.dataset]
        s = args.downsample
        expected = (d.shape[0], (d.shape[1]+s-1)//s, (d.shape[2]+s-1)//s)
        if d.ndim != 3 or expected != labels.shape:
            raise ValueError(f'Shape mismatch: dataset {d.shape}, stride {s}, event map {labels.shape}')
        summary_rows = []
        detailed = None
        def collect(movie):
            nonlocal detailed
            for i, event_id in enumerate(target_ids, start=1):
                try:
                    summary, rows, aux = analyze_event(event_id, indexed[event_id], d,
                                                       d.shape[0], s, args, movie=movie)
                    summary_rows.append({'event_id': event_id, 'status': 'evaluated',
                                         'reason': '', 'start_frame': summary['start_frame'],
                                         'duration_frames': summary['duration_frames'],
                                         'footprint_pixels': summary['footprint_pixels'],
                                         'valid_combinations': summary['valid_combinations'],
                                         'median_right_minus_left': summary['right_minus_left_onset_lag_median_frames'],
                                         'right_minus_left_range': summary['right_minus_left_onset_lag_range_frames'],
                                         'lag_sign_consistency': summary['right_minus_left_lag_sign_consistency'],
                                         'xcorr_lr_lag': summary['cross_correlation']['LR']['lag'],
                                         'xcorr_lr_corr': summary['cross_correlation']['LR']['best_correlation'],
                                         'xcorr_lr_at_search_boundary': summary['cross_correlation']['LR']['at_search_boundary']})
                    if not args.all:
                        detailed = (summary, rows, aux)
                except (ValueError, IndexError) as exc:
                    summary_rows.append({'event_id': event_id, 'status': 'not_evaluable',
                                         'reason': str(exc), 'start_frame': '',
                                         'duration_frames': '', 'footprint_pixels': '', 'valid_combinations': '',
                                         'median_right_minus_left': '', 'right_minus_left_range': '',
                                         'lag_sign_consistency': '', 'xcorr_lr_lag': '', 'xcorr_lr_corr': '',
                                         'xcorr_lr_at_search_boundary': ''})
                if args.all and (i%50==0 or i==len(target_ids)):
                    print(f'Processed {i}/{len(target_ids)} event IDs', flush=True)

        if args.all:
            # One sequential pass through HDF5. Temporary reduced movie stays on DISK
            # (~71 MB for user's ExVivo: 281x251x251 float32) and is deleted afterward.
            with tempfile.TemporaryDirectory(prefix='astrocelllink_', dir=output) as scratch:
                mm = np.memmap(Path(scratch)/'reduced_float32.dat', mode='w+',
                               dtype=np.float32, shape=labels.shape)
                for t in range(labels.shape[0]):
                    mm[t] = d[t, ::s, ::s]
                mm.flush()
                collect(mm)
                del mm
        else:
            collect(None)

    if args.all:
        dest = output/'population_lag_screen.csv'
        write_csv(dest, summary_rows)
        counts = {'total_event_ids':len(target_ids),
                  'evaluated': sum(x['status']=='evaluated' for x in summary_rows),
                  'not_evaluable': sum(x['status']=='not_evaluable' for x in summary_rows)}
        (output/'population_lag_summary.json').write_text(json.dumps({
            'status':'exploratory descriptive screen; not validated propagation',
            'counts':counts,
            'caveats':['Event labels are candidate components, not independently established cells/events.',
                       'No wave confidence, statistical p-value or speed is calculated.',
                       'Horizontal-band timing is sensitive to amplitude, overlap and segmentation.',
                       'Inspect alternative directions; LR lag alone is not a propagation vector.'],
        }, indent=2))
        print('Summary:', counts)
        print('CSV:', dest)
    else:
        if detailed is None:
            print('Event not evaluable:', summary_rows[0]['reason'])
            return 2
        summary, rows, aux = detailed
        base = output / f'event_{args.event}_robustness'
        (base.with_suffix('.json')).write_text(json.dumps(summary, indent=2, allow_nan=False))
        write_csv(base.with_suffix('.csv'), rows)
        make_plot(args.event, aux, rows, summary, base.with_suffix('.png'))
        print(json.dumps(summary, indent=2, allow_nan=False))
        print('Plot:', base.with_suffix('.png'))
        print('CSV:', base.with_suffix('.csv'))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='HDF5 source movie')
    p.add_argument('--event-map', required=True, help='TIFF event map or .roi directory')
    p.add_argument('--dataset', default='dff/ch0')
    p.add_argument('--downsample', type=int, default=2)
    p.add_argument('--event', type=int, default=34)
    p.add_argument('--all', action='store_true', help='Screen all IDs; DOES NOT identify proven waves')
    p.add_argument('--output', default='real_data/exvivo_screen')
    p.add_argument('--preframes', type=int, default=12)
    p.add_argument('--postframes', type=int, default=12)
    p.add_argument('--min-band-pixels', type=int, default=15)
    p.add_argument('--min-duration', type=int, default=6)
    p.add_argument('--max-lag', type=int, default=8)
    args = p.parse_args()
    if min(args.downsample, args.preframes, args.postframes, args.min_duration, args.min_band_pixels, args.max_lag) < 1:
        p.error('downsample, preframes, postframes, min-duration, min-band-pixels and max-lag must be positive')
    raise SystemExit(main(args))
