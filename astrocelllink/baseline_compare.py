"""Compare 3D event maps without pretending one is annotation ground truth.

A voxel is a time + pixel index. Matching is unique-pair greedy by space-time IoU.
"""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .io import read_array, labels_3d


def compare_arrays(candidate, upstream, *, min_iou=.10, frame=46, output=None):
    if not 0 <= min_iou <= 1:
        raise ValueError('min_iou must be in [0,1]')
    a=labels_3d(candidate,'candidate')
    b=labels_3d(upstream,'astrocast')
    if a.shape!=b.shape:
        raise ValueError(f'Different shapes: candidate {a.shape}, astroCaST {b.shape}.\n'
            'Run astroCaST on matching-stride dff input (prepare-h5 --stride 2). '
            'Never resize label images silently.')
    n=int(a.shape[0]); counts_a=Counter();counts_b=Counter();cross=Counter()
    foreground_intersection=0; union=0
    for k in range(n):
        aa=a[k].ravel(); bb=b[k].ravel()
        va,ca=np.unique(aa[aa>0],return_counts=True)
        vb,cb=np.unique(bb[bb>0],return_counts=True)
        counts_a.update({int(v):int(c) for v,c in zip(va,ca)})
        counts_b.update({int(v):int(c) for v,c in zip(vb,cb)})
        ma=aa>0;mb=bb>0
        foreground_intersection+=int((ma&mb).sum())
        union+=int((ma|mb).sum())
        both=np.column_stack((aa[ma&mb],bb[ma&mb]))
        if len(both):
            pair,cts=np.unique(both,axis=0,return_counts=True)
            cross.update({(int(x),int(y)):int(c) for (x,y),c in zip(pair,cts)})
    pairs=[]
    for (ida,idb),intersection in cross.items():
        iou=intersection/(counts_a[ida]+counts_b[idb]-intersection)
        if iou>=min_iou:
            pairs.append((iou,intersection,ida,idb))
    pairs.sort(reverse=True)
    used_a=set();used_b=set();rows=[]
    for iou,inter,ida,idb in pairs:
        if ida in used_a or idb in used_b: continue
        used_a.add(ida);used_b.add(idb)
        rows.append({'candidate_id':ida,'astrocast_id':idb,
                     'intersection_voxels':inter,'candidate_voxels':counts_a[ida],
                     'astrocast_voxels':counts_b[idb],'iou_3d':round(float(iou),6)})
    summary={'status':'DETECTOR AGREEMENT ONLY; no expert biological ground truth',
       'shape':list(map(int,a.shape)),'n_candidate_events':len(counts_a),
       'n_astrocast_events':len(counts_b),'matched_pairs_greedy':len(rows),
       'unmatched_candidates':len(counts_a)-len(rows),
       'unmatched_astrocast':len(counts_b)-len(rows),
       'matching_rule':f'one-to-one greedy, 3D IoU >= {min_iou}',
       'foreground_intersection_voxels':foreground_intersection,
       'foreground_union_voxels':union,
       'foreground_jaccard':foreground_intersection/union if union else None,
       'candidate_event_match_fraction':len(rows)/len(counts_a) if counts_a else None,
       'astrocast_event_match_fraction':len(rows)/len(counts_b) if counts_b else None,
       'mean_matched_event_iou':float(np.mean([r['iou_3d'] for r in rows])) if rows else None,
       'warnings':['These are algorithm-agreement measures, NOT precision/recall or correctness.',
         'Matching by greedy IoU is not globally optimal.',
         'Input shapes must be exactly equal; origin/alignment must independently be verified.',
         'Candidate labels can split/merge differently even when foreground overlaps.']}
    if output is not None:
        out=Path(output);out.mkdir(parents=True,exist_ok=True)
        with (out/'event_matches.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['candidate_id','astrocast_id','intersection_voxels','candidate_voxels','astrocast_voxels','iou_3d'])
            w.writeheader();w.writerows(rows)
        (out/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        if not 0<=frame<n: frame=n//2
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fa,fb=a[frame]>0,b[frame]>0
            fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
            axes[0].imshow(fa,cmap='Blues',vmin=0,vmax=1)
            axes[1].imshow(fb,cmap='Oranges',vmin=0,vmax=1)
            rgb=np.zeros((*fa.shape,3),dtype=np.float32)
            rgb[:,:,0]=fa.astype(float);rgb[:,:,2]=fb.astype(float)
            rgb[:,:,1]=(fa&fb).astype(float)
            axes[2].imshow(rgb)
            for ax,title in zip(axes,['Candidate foreground','astroCaST foreground','Red=only candidate; Blue=only astroCaST; white-ish=both']):
                ax.set_title(title,fontsize=9);ax.axis('off')
            fig.suptitle(f'Event-map AGREEMENT, frame {frame}; NOT accuracy',fontsize=11)
            fig.savefig(out/'agreement_preview.png',dpi=150);plt.close(fig)
        except ImportError: pass
    return summary,rows


def compare_files(candidate_path,astrocast_path,*,min_iou=.1,frame=46,output='comparison'):
    return compare_arrays(read_array(candidate_path),read_array(astrocast_path),min_iou=min_iou,frame=frame,output=output)
