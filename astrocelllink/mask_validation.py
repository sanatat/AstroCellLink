"""Compare segmented cell masks with independently expert-labelled masks.

These metrics are only meaningful if reference labels are independent.
"""
import csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .io import read_array


def evaluate_masks(pred,reference,*,iou_threshold=.5,output=None):
    if not 0<iou_threshold<=1:raise ValueError('iou_threshold must be in (0,1]')
    a=np.asarray(pred);b=np.asarray(reference)
    if a.ndim!=2 or b.ndim!=2 or a.shape!=b.shape:
        raise ValueError(f'Need 2D matching-size labels, got {a.shape} and {b.shape}')
    if not all(x.dtype.kind in 'uib' and (x>=0).all() for x in (a,b)):
        raise ValueError('Masks must use nonnegative integer labels; background=0')
    ids_a=np.unique(a[a>0]);ids_b=np.unique(b[b>0]);
    n,m=len(ids_a),len(ids_b)
    if max(n,m)>2000: raise ValueError('Too many instances for dense optimal assignment (>2000)')
    ious=np.zeros((n,m),float)
    if n and m:
        id_to_a={int(z):i for i,z in enumerate(ids_a)}
        id_to_b={int(z):j for j,z in enumerate(ids_b)}
        mask=(a>0)&(b>0)
        if mask.any():
            pairs,cs=np.unique(np.stack([a[mask],b[mask]],axis=1),axis=0,return_counts=True)
            sz_a=dict(zip(*np.unique(a[a>0],return_counts=True)))
            sz_b=dict(zip(*np.unique(b[b>0],return_counts=True)))
            for (ida,idb),c in zip(pairs,cs):
                i=id_to_a[int(ida)];j=id_to_b[int(idb)]
                ious[i,j]=c/(sz_a[ida]+sz_b[idb]-c)
    rows=[]
    if n and m:
        ridx,cidx=linear_sum_assignment(-ious)
        for i,j in zip(ridx,cidx):
            if ious[i,j]>=iou_threshold:
                rows.append({'predicted_label':int(ids_a[i]),'reference_label':int(ids_b[j]),'iou':round(float(ious[i,j]),6)})
    tp=len(rows);fp=n-tp;fn=m-tp
    result={'status':'SEGMENTATION VALIDATION only if reference masks are independently expert-labelled',
            'n_predicted':n,'n_reference':m,'matched_at_threshold':tp,
            'false_positives':fp,'false_negatives':fn,'iou_threshold':iou_threshold,
            'precision':tp/n if n else None,'recall':tp/m if m else None,
            'f1':2*tp/(n+m) if (n+m) else None,
            'mean_matched_iou':float(np.mean([x['iou'] for x in rows])) if rows else None,
            'note':'This is mask agreement with supplied reference; data provenance and label quality not independently verified.'}
    if output:
        p=Path(output);p.mkdir(parents=True,exist_ok=True)
        (p/'mask_validation.json').write_text(json.dumps(result,indent=2))
        with (p/'mask_matches.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['predicted_label','reference_label','iou']);w.writeheader();w.writerows(rows)
    return result


def evaluate_files(prediction,reference,**kwargs):
    return evaluate_masks(read_array(prediction),read_array(reference),**kwargs)
