#!/usr/bin/env python3
"""AstroCellLink research QC: audit the onset threshold guard and spatial-block stability.

This is conditional on the pre-selected candidate-event footprint. Bootstrap
intervals describe within-event spatial sensitivity, NOT independent replication,
p-values, propagation probability, physical velocity, or proof of cell identity.

Requires astrocelllink_signal_onset_validation.py in the same directory and its
numpy/scipy/h5py/tifffile/matplotlib dependencies.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astrocelllink_signal_onset_validation import analyze


def audit_thresholds(aux, start: int, stop: int, preframes: int, noise_mult: float):
    traces = aux['smoothed_data']
    baseline = np.median(traces[:preframes], axis=0)
    robust_sigma = 1.4826 * np.median(np.abs(traces[:preframes] - baseline), axis=0)
    pos = robust_sigma[np.isfinite(robust_sigma) & (robust_sigma > 0)]
    floor = max(float(np.median(pos)) * 0.2, 1e-6) if pos.size else 1e-6
    robust_sigma = np.maximum(robust_sigma, floor)
    rise = traces[preframes:preframes + (stop - start + 1)] - baseline
    peak = np.max(rise, axis=0)
    noise_threshold = noise_mult * robust_sigma
    eligible = np.isfinite(traces).all(axis=0) & (peak >= noise_threshold)
    yy, xx = aux['yy'], aux['xx']
    q1, q2 = np.quantile(xx, [1/3, 2/3])
    region = {'left': xx <= q1, 'middle': (xx > q1) & (xx <= q2), 'right': xx > q2}
    output = {}
    usable_sets = []
    for key, entry in aux['summary_by_fraction'].items():
        f = float(key)
        threshold = np.maximum(f * peak, noise_threshold)
        crossed = rise >= threshold[None, :]
        sustained = np.any(crossed[:-1] & crossed[1:], axis=0)
        usable = eligible & sustained
        reference = entry['_usable']
        if not np.array_equal(usable, reference):
            raise RuntimeError(f'Guard audit not reproduced at fraction {key}: check script version or parameters.')
        usable_sets.append(usable)
        dominated = noise_threshold >= f * peak
        output[key] = {
            'total_footprint_pixels': len(xx),
            'eligible_noise_guard_pixels': int(eligible.sum()),
            'failed_peak_above_noise_guard': int((~eligible).sum()),
            'eligible_failed_two_consecutive_frames': int((eligible & ~sustained).sum()),
            'usable_pixels': int(usable.sum()),
            'usable_noise_guard_controls_threshold': int((usable & dominated).sum()),
            'usable_noise_guard_controls_threshold_fraction': round(float((usable & dominated).sum() / max(1, usable.sum())), 4),
            'missing_pixels_by_region': {k: int((~usable & mask).sum()) for k, mask in region.items()},
        }
    same_set = all(np.array_equal(usable_sets[0], u) for u in usable_sets[1:])
    return output, same_set


def fit_one_block_sample(x: np.ndarray, y: np.ndarray, t: np.ndarray):
    if len(t) < 20 or np.std(t) < 1e-8:
        return None
    X = np.column_stack((x - x.mean(), y - y.mean(), np.ones_like(x)))
    if np.linalg.matrix_rank(X) != 3:
        return None
    b = np.linalg.lstsq(X, t, rcond=None)[0]
    pred = X @ b
    denom = float(np.sum((t - t.mean())**2))
    return (float(b[0]), float(b[1]), 1.0 - float(np.sum((t - pred)**2))/denom)


def spatial_block_bootstrap(xx, yy, first, usable, *, block_size=5, reps=600, rng=None):
    """Resample whole spatial blocks (cluster bootstrap), not independent pixels.

    Blocks are fixed in full-image coordinates. Any image-content-dependent
    selection (event footprint) is held fixed, so inference is conditional.
    """
    if rng is None:
        rng=np.random.default_rng(0)
    x=np.asarray(xx[usable], dtype=float)
    y=np.asarray(yy[usable], dtype=float)
    t=np.asarray(first[usable], dtype=float)
    allx=np.asarray(xx, dtype=float)
    cut1,cut2=np.quantile(allx,[1/3,2/3])
    keys=np.column_stack((np.floor_divide(y.astype(int),block_size), np.floor_divide(x.astype(int),block_size)))
    _, idx = np.unique(keys,axis=0,return_inverse=True)
    groups=[np.flatnonzero(idx==k) for k in range(int(idx.max())+1)] if len(idx) else []
    n_blocks=len(groups)
    original = fit_one_block_sample(x,y,t)
    if n_blocks<8 or original is None:
        return {'status':'insufficient_unique_spatial_blocks','n_blocks':n_blocks,'block_size':block_size,'original':original}
    rec=[]
    for _ in range(reps):
        s=rng.integers(0,n_blocks,size=n_blocks)
        selected=np.concatenate([groups[i] for i in s])
        out=fit_one_block_sample(x[selected],y[selected],t[selected])
        if out is None:
            continue
        l=t[selected][x[selected] <= cut1]
        r=t[selected][x[selected] > cut2]
        if min(len(l),len(r)) < 5:
            continue
        rec.append((*out, float(np.median(l)-np.median(r))))
    if len(rec)<max(50,reps//4):
        return {'status':'too_many_degenerate_draws','n_blocks':n_blocks,'n_valid_draws':len(rec),'block_size':block_size}
    z=np.asarray(rec)
    return {
        'status':'ok', 'block_size_reduced_pixels':block_size,'n_blocks':n_blocks,
        'requested_draws':reps,'valid_draws':len(rec),
        'reference_bx':round(original[0],5),'reference_by':round(original[1],5),
        'bx_quantiles_2p5_50_97p5':np.round(np.quantile(z[:,0],[.025,.5,.975]),5).tolist(),
        'by_quantiles_2p5_50_97p5':np.round(np.quantile(z[:,1],[.025,.5,.975]),5).tolist(),
        'r2_quantiles_2p5_50_97p5':np.round(np.quantile(z[:,2],[.025,.5,.975]),4).tolist(),
        'left_minus_right_onset_median_quantiles_frames':np.round(np.quantile(z[:,3],[.025,.5,.975]),2).tolist(),
        'fraction_draws_bx_negative_descriptive':round(float(np.mean(z[:,0]<0)),4),
        'fraction_draws_left_later_descriptive':round(float(np.mean(z[:,3]>0)),4),
        '_raw_samples':z,
    }


def to_public(z):
    if isinstance(z,dict):
        return {k:to_public(v) for k,v in z.items() if not k.startswith('_')}
    if isinstance(z,list): return [to_public(v) for v in z]
    return z


def plot_audit(aux, audit, boots, output:Path, event_id:int):
    yy,xx=aux['yy'],aux['xx']
    minx,maxx,miny,maxy=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
    first=aux['summary_by_fraction']['0.20']['_onsets']
    usable=aux['summary_by_fraction']['0.20']['_usable']
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    region=np.zeros((maxy-miny,maxx-minx))
    region[yy[~usable]-miny,xx[~usable]-minx]=1
    region[yy[usable]-miny,xx[usable]-minx]=2
    from matplotlib.colors import ListedColormap
    ax[0,0].imshow(region,cmap=ListedColormap(['#101827','#ef8c3b','#18a67d']),vmin=0,vmax=2,interpolation='nearest')
    ax[0,0].set_title('20%: green=valid; orange=missing; black=outside footprint')
    keys=list(audit)
    bx=[]; bmed=[]; bneg=[]; missing=[]; dom=[]
    for key in keys:
        rr=boots[key]['block_5']
        bx.append(rr.get('bx_quantiles_2p5_50_97p5'))
        bmed.append(rr.get('left_minus_right_onset_median_quantiles_frames'))
        bneg.append(rr.get('fraction_draws_bx_negative_descriptive'))
        missing.append(audit[key]['total_footprint_pixels'] - audit[key]['usable_pixels'])
        dom.append(audit[key]['usable_noise_guard_controls_threshold_fraction'])
    f=np.array([100*float(k) for k in keys]); fcolor='#2563eb'
    def ci_panel(axis, rows, title, name):
        valid=[(a,b) for a,b in zip(f,rows) if b is not None]
        if valid:
            xv=np.array([r[0] for r in valid]); yv=np.array([r[1] for r in valid]);
            axis.plot(xv,yv[:,1],'o-',color=fcolor)
            axis.fill_between(xv,yv[:,0],yv[:,2],color=fcolor,alpha=.20)
        axis.axhline(0,color='grey',linestyle='--',linewidth=.8)
        axis.set_xlabel('Threshold (%)')
        axis.set_ylabel(name)
        axis.set_title(title)
    ci_panel(ax[0,1],bx,'Spatial-block bootstrap: bx (block=5)','frames per reduced pixel')
    ci_panel(ax[1,0],bmed,'Spatial-block bootstrap: median left minus right','frames')
    ax[1,1].plot(f,dom,'o-',label='Noise guard dominates usable thresholds')
    ax[1,1].plot(f,np.array(missing)/len(xx),'s-',label='Invalid footprint fraction')
    ax[1,1].set_ylim(-.05,1.05)
    ax[1,1].set_xlabel('Relative rise threshold (%)')
    ax[1,1].set_ylabel('Fraction')
    ax[1,1].set_title('Guard and missing-data audit')
    ax[1,1].legend(fontsize=8)
    fig.suptitle(f'Event {event_id} | conditional exploratory QC; block bootstrap is NOT biological validation',fontsize=12)
    fig.savefig(output,dpi=150)
    plt.close(fig)


def run(event_map, movie, event_id, output, *, downsample=2, sigma=.8, preframes=30, postframes=7, reps=600, seed=20261019, dataset='dff/ch0'):
    report, aux = analyze(event_map,movie,event_id,preframes=preframes,postframes=postframes,downsample=downsample,sigma=sigma,dataset=dataset)
    audit,same=audit_thresholds(aux,report['event_start_frame'],report['event_last_frame'],report['pre_event_frames'],report['noise_guard_multiplier'])
    rng=np.random.default_rng(seed)
    boots={}
    for key,d in aux['summary_by_fraction'].items():
        boots[key]={}
        for size in (4,5,8):
            boots[key][f'block_{size}']=spatial_block_bootstrap(aux['xx'],aux['yy'],d['_onsets'],d['_usable'],block_size=size,reps=reps,rng=rng)
    result={
        'status':'exploratory spatial sensitivity, NOT propagation significance or biological validation',
        'event_id':event_id,'data_source':dataset,'downsample':downsample,
        'same_valid_pixel_membership_at_all_thresholds':same,
        'threshold_guard_audit':audit,
        'spatial_block_bootstrap_by_threshold':to_public(boots),
        'caveats':[
            'Selection bias: the footprint was defined from the same fluorescence movie.',
            'Pixel times are spatially dependent. Spatial-block bootstrap is within-event sensitivity, not independent biological replicates.',
            'Bootstrap intervals are not p-values, and draw sign fractions are not probabilities of physical propagation.',
            'Spatial block-size dependence and missing pixels must be examined.',
            'Comparisons across different event detectors and independent expert labels are still needed.'
        ]
    }
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    jpg=output/f'event_{event_id}_spatial_controls.png'; js=output/f'event_{event_id}_spatial_controls.json'
    plot_audit(aux,audit,boots,jpg,event_id)
    with js.open('w',encoding='utf-8') as f:json.dump(result,f,indent=2)
    return result,jpg,js


def self_test():
    rng=np.random.default_rng(1102)
    T,H,W=100,70,75
    yg,xg=np.indices((H,W));fp=((xg-36)/17)**2+((yg-34)/12)**2<1
    events=np.zeros((T,H,W),dtype=np.uint16)
    movie=rng.normal(8,.1,(T,H,W)).astype('float32')
    true_onset=47 + np.rint((36-xg)*.27).astype(int)
    for y,x in zip(*np.where(fp)):
        s=int(true_onset[y,x]); events[s:s+7,y,x]=194
        tt=np.arange(T)-s
        movie[:,y,x]+=np.where(tt>=0,9*np.exp(-((tt-3.5)/3.1)**2),0)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        result,_,_=run(events,movie,194,Path(d),downsample=1,preframes=25,postframes=7,reps=100)
        rr=result['spatial_block_bootstrap_by_threshold']['0.20']['block_5']
        assert rr['status']=='ok',rr
        assert rr['reference_bx']<-.07,rr
        assert result['threshold_guard_audit']['0.20']['usable_pixels']>150
        print('PASS: simulated right-earlier gradient recovered; guard counts cross-checked')
        print('Synthetic bx:',rr['reference_bx'],'Block CI:',rr['bx_quantiles_2p5_50_97p5'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path)
    p.add_argument('--event-map',type=Path)
    p.add_argument('--dataset',default='dff/ch0')
    p.add_argument('--event',type=int,default=194)
    p.add_argument('--downsample',type=int,default=2)
    p.add_argument('--sigma',type=float,default=.8)
    p.add_argument('--preframes',type=int,default=30)
    p.add_argument('--postframes',type=int,default=7)
    p.add_argument('--bootstrap',type=int,default=600)
    p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--output',type=Path,default=Path('real_data/exvivo_screen'))
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        self_test();return
    if a.input is None or a.event_map is None:
        p.error('Provide --input and --event-map, or --self-test')
    if a.bootstrap<50 or a.downsample<1 or a.sigma<0:
        p.error('bootstrap must be >=50; downsample >=1; sigma >=0')
    ev=tifffile.imread(a.event_map)
    with h5py.File(a.input,'r') as f:
        result,png,js=run(ev,f[a.dataset],a.event,a.output,downsample=a.downsample,sigma=a.sigma,preframes=a.preframes,postframes=a.postframes,reps=a.bootstrap,seed=a.seed,dataset=a.dataset)
    print('Output PNG:',png,'\nOutput JSON:',js)
    print('Same pixels at every threshold:',result['same_valid_pixel_membership_at_all_thresholds'])
    for fraction in result['threshold_guard_audit']:
        guard=result['threshold_guard_audit'][fraction]
        b=result['spatial_block_bootstrap_by_threshold'][fraction]['block_5']
        print(f' {fraction}: usable={guard["usable_pixels"]}; guard-dominates={guard["usable_noise_guard_controls_threshold_fraction"]:.3f}; bx 95%-interval={b.get("bx_quantiles_2p5_50_97p5")}; median LR-interval={b.get("left_minus_right_onset_median_quantiles_frames")}')


if __name__=='__main__':main()
