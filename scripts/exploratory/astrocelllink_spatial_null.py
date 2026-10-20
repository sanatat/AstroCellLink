#!/usr/bin/env python3
"""Exploratory block-permutation diagnostic for AstroCellLink calcium-event onset patterns.

This is NOT a biological calcium-propagation test, a validated statistical
hypothesis test, a p-value generator, or independent confirmation: the spatial
footprint comes from the very same fluorescence recording being analyzed.

Method: estimate fluorescence-derived pixel onsets using the exact existing
AstroCellLink helper. Group usable pixels by square image-space blocks, compute
one mean onset per block, and compare the observed block-level horizontal slope
to slopes obtained by permuting BLOCK MEANS among fixed block centroids.
Blocks are not independent biological replicates; block means may still be
spatially correlated. Null distributions are diagnostics, not calibrated
significance tests, and must not be advertised as proof of propagation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import h5py
import tifffile

from astrocelllink_signal_onset_validation import analyze


def fit_block_plane(x, y, times):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(times, dtype=float)
    if len(t) < 8 or np.std(t) < 1e-9:
        return None
    center = np.column_stack((x - x.mean(), y - y.mean(), np.ones(len(t))))
    if np.linalg.matrix_rank(center) < 3:
        return None
    b = np.linalg.lstsq(center, t, rcond=None)[0]
    pred = center @ b
    sst = float(np.sum((t - t.mean())**2))
    return (float(b[0]), float(b[1]), float(1 - np.sum((t-pred)**2)/sst))


def block_summaries(x, y, t, k):
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    t = np.asarray(t, dtype=float)
    if len(t) != len(x) or len(t) != len(y):
        raise ValueError('x, y, onset arrays must align')
    xyblocks = np.column_stack((x // k, y // k))
    _, idx = np.unique(xyblocks, axis=0, return_inverse=True)
    n = int(idx.max()) + 1 if len(idx) else 0
    nums = np.bincount(idx, minlength=n).astype(float)
    sx = np.bincount(idx, weights=x, minlength=n) / nums
    sy = np.bincount(idx, weights=y, minlength=n) / nums
    st = np.bincount(idx, weights=t, minlength=n) / nums
    return sx, sy, st, nums


def permutation_diagnostic(x, y, t, block_size=5, draws=2000, seed=0):
    bx, by, bt, bn = block_summaries(x, y, t, block_size)
    obs = fit_block_plane(bx, by, bt)
    if obs is None:
        return {'status':'not_evaluable', 'block_size':block_size,
                'n_blocks':int(len(bt)), 'reason':'fewer than 8 informative blocks or degenerate geometry'}, None
    # Fixed spatial coordinates, randomly exchange block onset means. This destroys
    # directional location association, not all spatial and signal correlations.
    rng = np.random.default_rng(seed)
    null_bx = np.empty(draws, dtype=float)
    for i in range(draws):
        permuted = rng.permutation(bt)
        fit = fit_block_plane(bx, by, permuted)
        null_bx[i] = fit[0]
    abs_fraction = (np.count_nonzero(np.abs(null_bx) >= abs(obs[0])) / draws)
    negative_fraction = (np.count_nonzero(null_bx <= obs[0]) / draws)
    out = {
        'status':'ok', 'block_size_reduced_pixels':int(block_size),
        'n_blocks':int(len(bt)), 'block_pixel_counts_min_median_max':[
            int(np.min(bn)),float(np.median(bn)),int(np.max(bn))],
        'observed_block_level_bx_frames_per_reduced_pixel':round(obs[0],6),
        'observed_block_level_by_frames_per_reduced_pixel':round(obs[1],6),
        'observed_block_level_r2':round(obs[2],4),
        'permutation_count':int(draws),
        'shuffled_block_bx_quantiles_2p5_50_97p5':
            [round(float(a),6) for a in np.quantile(null_bx,[.025,.5,.975])],
        'descriptive_shuffled_fraction_abs_bx_at_least_observed':round(float(abs_fraction),5),
        'descriptive_shuffled_fraction_bx_at_most_observed':round(float(negative_fraction),5),
    }
    return out, null_bx


def run(x, y, t, block_sizes, draws, seed):
    reports = []
    nulls = {}
    for i, k in enumerate(block_sizes):
        res, null = permutation_diagnostic(x, y, t, k, draws, seed + i)
        reports.append(res)
        if null is not None:
            nulls[k] = null
    return reports, nulls


def plot_report(reports, nulls, output, event_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entries = [r for r in reports if r['status']=='ok']
    fig, axs = plt.subplots(1, len(entries), figsize=(5 * len(entries), 4),
                            squeeze=False, constrained_layout=True)
    if not entries:
        plt.close(fig)
        return False
    for ax, r in zip(axs.flat, entries):
        k = r['block_size_reduced_pixels']
        ax.hist(nulls[k], bins=35, alpha=.7, label='Shuffled block means')
        ax.axvline(r['observed_block_level_bx_frames_per_reduced_pixel'],
                   linestyle='--', linewidth=2, label='Observed block slope')
        ax.axvline(0, color='black', linewidth=.7)
        ax.set_title(f'Block {k} | {r["n_blocks"]} blocks')
        ax.set_xlabel('bx (frames / reduced pixel)')
        ax.set_ylabel('Number of shuffles')
        ax.legend(fontsize=8)
    fig.suptitle(f'Event {event_id}: WITHIN-FOOTPRINT BLOCK-SHUFFLE DIAGNOSTIC\n'
                 'NOT independent validation; NOT a biological p-value',fontsize=11)
    fig.savefig(output, dpi=155)
    plt.close(fig)
    return True


def synthetic_tests():
    from astrocelllink_signal_onset_validation import analyze
    rng = np.random.default_rng(73)
    T,H,W = 100,55,55
    yg,xg = np.indices((H,W))
    fp = ((xg-28)**2/13**2 + (yg-28)**2/12**2)<1
    labels = np.zeros((T,H,W), dtype=np.uint16)
    labels[40:66,fp]=194
    for grade in (True, False):
        image = rng.normal(0,0.16,(T,H,W)).astype(np.float32)
        for y,x in zip(*np.nonzero(fp)):
            onset = int(43 + round(.32 * (x-28))) if grade else 43
            delta = np.arange(T)-onset
            pulse = np.where(delta>=0, 10*np.exp(-((delta-5)/4.)**2), 0)
            image[:,y,x] += pulse
        summ, a = analyze(labels, image, 194, downsample=1,
                          preframes=30,postframes=7,fractions=(.2,))
        ff = a['summary_by_fraction']['0.20']
        sel=ff['_usable']
        x,y,t=a['xx'][sel],a['yy'][sel],ff['_onsets'][sel]
        assert len(t)>200,(len(t),grade)
        reps,_=run(x,y,t,[4,5,8],250,42)
        good=[r for r in reps if r['status']=='ok']
        assert len(good)==3,reps
        if grade:
            assert all(r['observed_block_level_bx_frames_per_reduced_pixel'] > .14 for r in good),reps
            assert all(r['descriptive_shuffled_fraction_abs_bx_at_least_observed'] < .08 for r in good),reps
        else:
            assert all(abs(r['observed_block_level_bx_frames_per_reduced_pixel']) < .14 for r in good),reps
        print(('Gradient' if grade else 'Simultaneous')+' synthetic check PASS:',
              [(r['block_size_reduced_pixels'],r['observed_block_level_bx_frames_per_reduced_pixel']) for r in good])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--input',type=Path,help='HDF5 source containing dff/ch0')
    p.add_argument('--event-map',type=Path,help='Existing 3D candidate event TIFF')
    p.add_argument('--event',type=int,default=194)
    p.add_argument('--dataset',default='dff/ch0')
    p.add_argument('--downsample',type=int,default=2)
    p.add_argument('--fraction',type=float,default=.20)
    p.add_argument('--preframes',type=int,default=30)
    p.add_argument('--sigma',type=float,default=.8)
    p.add_argument('--block-sizes',type=int,nargs='+',default=[4,5,8])
    p.add_argument('--permutations',type=int,default=2000)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--output',type=Path,default=Path('real_data/exvivo_screen'))
    args=p.parse_args()
    if args.self_test:
        synthetic_tests()
        return
    if not args.input or not args.event_map:
        p.error('--input and --event-map are required')
    if not .01 <= args.fraction <= .99 or args.permutations<100 or min(args.block_sizes)<2:
        p.error('fraction must be between .01 and .99, permutations >=100, and blocks >=2')
    event_map=tifffile.imread(args.event_map)
    with h5py.File(args.input, 'r') as hf:
        if args.dataset not in hf:
            raise KeyError('Dataset not found: '+args.dataset)
        summ, a=analyze(event_map,hf[args.dataset],args.event,preframes=args.preframes,
                        downsample=args.downsample,sigma=args.sigma,
                        fractions=(args.fraction,),dataset=args.dataset)
    key=f'{args.fraction:.2f}'
    obj=a['summary_by_fraction'][key]
    valid=obj['_usable']
    x=a['xx'][valid]
    y=a['yy'][valid]
    t=obj['_onsets'][valid].astype(float)
    reports, nulls=run(x,y,t,args.block_sizes,args.permutations,args.seed)
    payload={
        'status':'EXPLORATORY block-mean shuffle conditional on event footprint; NOT biological inference',
        'event_id':args.event,
        'source_dataset':args.dataset,
        'threshold_fraction':args.fraction,
        'valid_pixels':int(valid.sum()),
        'total_footprint_pixels':int(len(valid)),
        'reference_pixel_level_plane':summ['fractions'][key]['plane'],
        'block_permutation_diagnostics':reports,
        'important_limitations':[
            'Footprint selected on the same data; selection can induce apparent organized lag.',
            'Exchanging block means assumes location-exchangeability which spatial structure may violate.',
            'Within-block correlations are kept by averaging, but correlations between blocks are not.',
            'Shuffled exceedance fractions are descriptive ranks, NOT calibrated p-values or wave probabilities.',
            'The event was selected after screening many candidates (exploratory selection).',
            'Independent recordings, negative controls, expert annotations and another detector are needed.'
        ]
    }
    args.output.mkdir(exist_ok=True, parents=True)
    jp=args.output/f'event_{args.event}_spatial_null.json'
    pp=args.output/f'event_{args.event}_spatial_null.png'
    cp=args.output/f'event_{args.event}_spatial_null.csv'
    jp.write_text(json.dumps(payload, indent=2), encoding='utf8')
    with cp.open('w', newline='', encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['block_size','n_blocks','observed_bx','observed_by','block_r2',
                                     'shuffled_abs_bx_exceedance_fraction'])
        w.writeheader()
        for r in reports:
            if r['status']=='ok':
                w.writerow({'block_size':r['block_size_reduced_pixels'],
                            'n_blocks':r['n_blocks'],
                            'observed_bx':r['observed_block_level_bx_frames_per_reduced_pixel'],
                            'observed_by':r['observed_block_level_by_frames_per_reduced_pixel'],
                            'block_r2':r['observed_block_level_r2'],
                            'shuffled_abs_bx_exceedance_fraction':r['descriptive_shuffled_fraction_abs_bx_at_least_observed']})
    ok=plot_report(reports,nulls,pp,args.event)
    print('Source:',args.input)
    print('Onset threshold:',args.fraction,'Valid footprint pixels:',len(t))
    for r in reports:
        print(json.dumps(r,ensure_ascii=False))
    print('JSON:',jp,'\nCSV:',cp)
    if ok:
        print('Plot:',pp)

if __name__=='__main__':
    main()
