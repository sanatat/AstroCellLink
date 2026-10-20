"""Static QC visualization for a single representative frame."""
import numpy as np


def qc_figure(event_map, tracked, assignments, path, structural=None, frame=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    tmax=event_map.shape[0]
    if frame is None:
        nz=(event_map>0).reshape(tmax,-1).sum(axis=1)
        frame=int(np.argmax(nz))
    if not (0 <= frame < tmax):
        raise ValueError("frame outside movie")
    base = structural[frame] if structural is not None and np.asarray(structural).ndim == 3 else structural
    if base is None:
        base=np.asarray(tracked[frame]>0,float)
    base=np.asarray(base,float)
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout="constrained")
    axes[0].imshow(base,cmap="gray")
    axes[0].set_title(f"Structural channel | frame {frame}")
    axes[1].imshow(base,cmap="gray")
    edges=find_boundaries(tracked[frame],mode="outer")
    axes[1].imshow(np.ma.masked_where(~edges,edges),cmap="autumn",alpha=.85)
    for cid in np.unique(tracked[frame]):
        if cid==0:continue
        pts=np.argwhere(tracked[frame]==cid)
        if len(pts):
            cy,cx=np.mean(pts,axis=0)
            axes[1].text(cx,cy,str(cid),color="cyan",weight="bold",ha="center",va="center",fontsize=11,
                         bbox={"facecolor":"black","alpha":.35,"pad":1})
    axes[1].set_title("Tracked cell ID | contour")
    axes[2].imshow(base,cmap="gray")
    ev=event_map[frame]
    overlay=np.ma.masked_where(ev==0,ev)
    axes[2].imshow(overlay,cmap="turbo",alpha=.75,interpolation="nearest")
    assigned={int(r["event_id"]):r for r in assignments}
    for eid in np.unique(ev):
        if eid==0:continue
        pts=np.argwhere(ev==eid)
        cy,cx=np.mean(pts,axis=0)
        row=assigned[int(eid)]
        cell=row["track_id"]
        label=f"E{eid} → C{cell}" if cell is not None else f"E{eid} → ?"
        axes[2].text(cx,cy,label,color="white",ha="center",va="center",fontsize=7,
                     bbox={"facecolor":"black","alpha":.6,"pad":1})
    axes[2].set_title("Calcium event → cell (QC only)")
    for ax in axes:ax.set_axis_off()
    fig.suptitle("AstroCellLink | synthetic-data sanity check; NOT biological validation",fontsize=11)
    fig.savefig(path,dpi=155,bbox_inches="tight")
    plt.close(fig)
