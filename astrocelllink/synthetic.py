"""Reproducible *synthetic* two-channel astrocyte-like benchmark.

NOT a scientifically validated biological calcium imaging dataset.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.ndimage import gaussian_filter
from .io import write_tiff


def simulate(seed=17, n_frames=48, shape=(108,108), n_events=18):
    rng = np.random.default_rng(seed)
    yy,xx = np.indices(shape)
    origin=np.array([[26,26],[26,80],[80,26],[80,80]],dtype=float)
    speeds=np.array([[0.065,0.09],[0.08,-0.09],[-0.07,0.06],[-0.09,-0.06]])
    structural=np.zeros((n_frames,)+shape,dtype=np.float32)
    local_masks=np.zeros((n_frames,)+shape,dtype=np.int32)
    stable_masks=np.zeros_like(local_masks)
    event_map=np.zeros_like(local_masks)
    fluorescence=np.zeros((n_frames,)+shape,dtype=np.float32)
    centers=[]
    for t in range(n_frames):
        labels=rng.permutation(np.arange(1,5))
        cr=[]
        frame=np.full(shape,0.12,dtype=float)
        for i,(pos,vel) in enumerate(zip(origin,speeds)):
            cy,cx=pos + t*vel + np.array([0.5*np.sin(t/5+i),0.5*np.cos(t/6+i)])
            cy,cx=float(cy),float(cx)
            cr.append((cy,cx))
            rad=(yy-cy)**2/11**2+(xx-cx)**2/11**2
            mask=rad<=1
            local_masks[t,mask]=int(labels[i])
            stable_masks[t,mask]=i+1
            frame+=1.3*np.exp(-rad*1.6)
        centers.append(cr)
        structural[t]=gaussian_filter(frame,0.7)+rng.normal(0,0.025,size=shape)
        fluorescence[t]=0.04+rng.normal(0,0.012,size=shape)
    truth=[]
    for e in range(n_events):
        cell=e%4
        start=2+2*e
        stop=min(start+int(rng.integers(3,6)),n_frames)
        if start>=n_frames: break
        offset=rng.integers(-3,4,size=2)
        for t in range(start,stop):
            cy,cx=np.array(centers[t][cell])+offset
            pix=(yy-cy)**2+(xx-cx)**2<=5.0**2
            # Only pixels attributable to the generating cell count.
            pix &= stable_masks[t]==cell+1
            conflict=pix & (event_map[t]!=0)
            pix &= ~conflict
            event_map[t,pix]=e+1
            fluorescence[t,pix]+=float(0.25+0.15*np.sin((t-start+1)*np.pi/(stop-start+1)))
        truth.append({"event_id":e+1,"true_cell_id":cell+1,"first_frame":start,"last_frame":stop-1})
    return structural,local_masks,stable_masks,event_map,fluorescence,truth


def write_demo_data(directory, **kwargs):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    structural,instances,stable,events,fluorescence,truth=simulate(**kwargs)
    import tifffile
    tifffile.imwrite(directory/"structural.tiff",structural,photometric="minisblack",metadata={"axes":"TYX"})
    tifffile.imwrite(directory/"calcium_fluorescence.tiff",fluorescence,photometric="minisblack",metadata={"axes":"TYX"})
    write_tiff(directory/"local_instance_masks.tiff",instances)
    write_tiff(directory/"ground_truth_stable_masks.tiff",stable)
    roi=directory/"sample_video.roi"
    roi.mkdir(exist_ok=True)
    write_tiff(roi/"event_map.tiff",events)
    with open(directory/"ground_truth_events.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(truth[0]));w.writeheader();w.writerows(truth)
    return directory
