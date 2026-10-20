"""Exploratory 2D segmentation screen. NOT cell identification from calcium channel."""
import json
from pathlib import Path
import h5py
import numpy as np
import tifffile
from .segmentation import segment_frame


def screen_morphology(input_file,dataset,output,*,kind='calcium-proxy',mode='mean',stride=2,n_samples=16,min_area=30,min_distance=8):
    if kind not in ('calcium-proxy','independent-morphology'):
        raise ValueError('kind must be calcium-proxy or independent-morphology')
    if mode not in ('mean','max'):
        raise ValueError('mode must be mean or max')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with h5py.File(input_file,'r') as f:
        ds=f[dataset]
        if ds.ndim!=3:raise ValueError('Expected (time,y,x) dataset')
        if n_samples<1 or stride<1:raise ValueError('n_samples and stride must be >=1')
        indices=np.unique(np.linspace(0,ds.shape[0]-1,min(n_samples,ds.shape[0]),dtype=int))
        im=None
        for i,k in enumerate(indices):
            frame=np.asarray(ds[int(k),::stride,::stride],np.float32)
            finite=np.isfinite(frame)
            if not finite.all():
                vals=frame[finite]
                frame=np.where(finite,frame,float(np.median(vals)) if len(vals) else 0)
            if im is None:im=frame.astype(np.float64)
            elif mode=='max':im=np.maximum(im,frame)
            else:im+=frame
    if mode=='mean':im/=len(indices)
    masks=segment_frame(im,min_area=min_area,min_distance=min_distance)
    tifffile.imwrite(output/'candidate_instances.tiff',masks.astype('uint16'),photometric='minisblack')
    tifffile.imwrite(output/'projection.tiff',im.astype('float32'),photometric='minisblack')
    summary={'status':'EXPLORATORY 2D INSTANCE CANDIDATES; not validated astrocyte boundaries',
      'source_kind':kind,'source':str(input_file),'dataset':dataset,'stride':stride,
      'frames_sampled':list(map(int,indices)),'projection':mode,
      'candidate_instances':int(masks.max()),
      'warning':('Calcium activity channel is NOT independent morphology. These masks MUST NOT be reported as cell identities.' if kind=='calcium-proxy'
                else 'Even independently labelled morphology needs expert masks and validation.')}
    (output/'segmentation_screen.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    edges=find_boundaries(masks,mode='outer')
    fig,axs=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    lo,hi=np.percentile(im,[1,99]);
    if hi<=lo:hi=lo+1
    axs[0].imshow(im,cmap='gray',vmin=lo,vmax=hi)
    axs[1].imshow(im,cmap='gray',vmin=lo,vmax=hi)
    rgba=np.zeros((*edges.shape,4),float);rgba[edges]=[1,0.3,0.15,0.9]
    axs[1].imshow(rgba)
    for ax,title in zip(axs,['Fluorescence projection','Watershed candidate contours (NOT cell IDs)']):
        ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle('Visual QC only; independent expert cell masks required',fontsize=11)
    fig.savefig(output/'segmentation_screen.png',dpi=145);plt.close(fig)
    return summary
