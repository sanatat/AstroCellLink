"""Explicit exploratory structural segmentation and independent mask evaluation.

No network/model downloads; no inferred identity from calcium dF/F. A
fluorescence movie without structural-specific labeling is NOT proof of cell ID.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

from .segmentation import segment_frame
from .io import write_tiff


def load_projection(path, *, dataset=None, stride=1, projection='mean', frame=None):
    """Load 2D structural image or compute 2D projection streaming T,Y,X."""
    if int(stride) != stride or stride < 1:
        raise ValueError('stride must be a positive integer')
    p=Path(path)
    if p.suffix.lower() not in ('.h5', '.hdf5'):
        from .io import read_array
        raw=read_array(p)
        if raw.ndim == 2:
            img=np.asarray(raw[::stride, ::stride], dtype='float32')
        elif raw.ndim == 3:
            if projection=='frame':
                if frame is None or not 0<=frame<len(raw):
                    raise ValueError('frame projection requires --frame within stack')
                img=np.asarray(raw[frame, ::stride, ::stride], dtype='float32')
            elif projection=='mean':
                img=np.asarray(raw[:, ::stride, ::stride], dtype='float64').mean(axis=0).astype('float32')
            else:
                raise ValueError('projection must be mean or frame')
        else:
            raise ValueError(f'Expected 2D or 3D structural input; got {raw.shape}')
    else:
        if not dataset: raise ValueError('For HDF5, --dataset is required')
        with h5py.File(p,'r') as f:
            if dataset not in f: raise KeyError(f'Dataset {dataset} not found')
            src=f[dataset]
            if src.ndim==2:
                img=np.asarray(src[::stride,::stride],dtype='float32')
            elif src.ndim==3:
                if projection=='frame':
                    if frame is None or not 0<=frame<src.shape[0]:
                        raise ValueError('--frame must be within 0..T-1')
                    img=np.asarray(src[frame,::stride,::stride],dtype='float32')
                elif projection=='mean':
                    acc=None
                    for t in range(src.shape[0]):
                        snap=np.asarray(src[t,::stride,::stride],dtype='float64')
                        if not np.all(np.isfinite(snap)):
                            raise ValueError('Structural data contain nonfinite values')
                        if acc is None: acc=np.zeros_like(snap)
                        acc+=snap
                    img=(acc/src.shape[0]).astype('float32')
                else: raise ValueError('projection must be mean or frame')
            else: raise ValueError(f'Expected 2D or (T,Y,X) input; got {src.shape}')
    if not np.all(np.isfinite(img)):
        raise ValueError('Structural data contain nonfinite values')
    return img


def make_segmentation(input_path, *, dataset=None, stride=1, projection='mean',
                      frame=None, min_area=30, min_distance=8, output='segmentation_screen'):
    img=load_projection(input_path,dataset=dataset,stride=stride,projection=projection,frame=frame)
    masks=segment_frame(img,min_area=min_area,min_distance=min_distance)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write_tiff(out/'proposed_instance_masks.tiff',masks)
    boundary=find_boundaries(masks,mode='outer')
    lo,hi=np.percentile(img,[1,99])
    if lo==hi:hi=lo+1
    fig,ax=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    ax[0].imshow(img,cmap='gray',vmin=lo,vmax=hi)
    ax[0].set_title('Input (structural intent not verified)')
    ax[1].imshow(masks,cmap='nipy_spectral',interpolation='nearest')
    ax[1].set_title('Watershed proposed masks')
    ax[2].imshow(img,cmap='gray',vmin=lo,vmax=hi)
    overlay=np.zeros((*img.shape,4),float);overlay[boundary]=[1,.25,.2,.96]
    ax[2].imshow(overlay);ax[2].set_title('Mask boundaries on input')
    for a in ax:a.axis('off')
    fig.suptitle('EXPLORATORY masks — NOT validated astrocyte identity',fontsize=11)
    fig.savefig(out/'proposed_masks_qc.png',dpi=150);plt.close(fig)
    summary={'status':'EXPLORATORY watershed; NOT validated segmentation',
             'source_file':str(input_path),'source_dataset':dataset,
             'source_is_structural_specific':'UNVERIFIED — user must supply/confirm acquisition metadata',
             'projection':projection,'frame':frame,'stride':stride,
             'masks_shape':list(map(int,masks.shape)),
             'proposed_instances':int(np.max(masks)),
             'min_area':min_area,'min_distance':min_distance,
             'warning':'If you use calcium fluorescence or dF/F as morphology, event-dependent features may be misidentified as cells. Independently annotated reference masks required.'}
    (out/'segmentation_screen.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
