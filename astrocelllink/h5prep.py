"""Out-of-core spatial-stride extraction matching the original candidate map.

No interpolation and no attempt to infer pixel size or frame rate.
"""
from pathlib import Path
import json
import h5py
import numpy as np


def prepare_stride(input_path, dataset, output, stride=2, *, overwrite=False):
    if stride < 1:
        raise ValueError('stride must be >=1')
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f'{output} exists. Use a new output path (or --overwrite).')
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, 'r') as src:
        if dataset not in src:
            raise KeyError(f'dataset {dataset!r} not found in {input_path}')
        source = src[dataset]
        if source.ndim != 3:
            raise ValueError(f'Expected (time,y,x), got {source.shape}')
        t,h,w = source.shape
        out_shape = (t, (h+stride-1)//stride, (w+stride-1)//stride)
        with h5py.File(output, 'w') as dst:
            dst.attrs['source_file'] = str(input_path)
            dst.attrs['source_dataset'] = dataset
            dst.attrs['downsample_method'] = f'pixel stride [::{stride},::{stride}] (NO anti-aliasing)'
            dst.attrs['calibration_status'] = 'pixel sizes and fps NOT determined'
            d = dst.create_dataset(dataset, shape=out_shape, dtype='float32',
                     chunks=(1,out_shape[1],out_shape[2]),compression='gzip',compression_opts=3)
            for i in range(t):
                d[i] = np.asarray(source[i, ::stride, ::stride], dtype=np.float32)
    metadata = {'source':str(input_path), 'dataset':dataset,
                'original_shape':[int(t),int(h),int(w)], 'output_shape':list(map(int,out_shape)),
                'stride':int(stride),'output':str(output),
                'warning':'Spatial striding sacrifices resolution. This is an aligned comparison dataset, not an independent replicate.'}
    sidecar=output.with_name(output.stem+'_provenance.json')
    sidecar.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata
