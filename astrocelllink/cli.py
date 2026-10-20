"""Command line: astrocelllink demo | run | prepare-h5 | compare-events | segment-structural | evaluate-masks."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .io import read_array
from .pipeline import run_pipeline
from .synthetic import write_demo_data


def evaluate_demo(root,results,assignments,tracks):
    """Synthetic truth: cell track IDs are mapped to generating cell IDs by pixel majority."""
    from collections import Counter
    truth_masks=read_array(root/"ground_truth_stable_masks.tiff")
    track_to_true={}
    for tid in np.unique(tracks):
        if tid == 0: continue
        labels=truth_masks[tracks==tid]
        labels=labels[labels>0]
        if len(labels):track_to_true[int(tid)]=Counter(labels).most_common(1)[0][0]
    with open(root/"ground_truth_events.csv",newline="") as f:
        truth={int(x["event_id"]):int(x["true_cell_id"]) for x in csv.DictReader(f)}
    assigned=[x for x in assignments if x["status"]=="assigned"]
    correct=int(sum(bool(track_to_true.get(int(x["track_id"]))==truth.get(int(x["event_id"]))) for x in assigned))
    out={"kind":"synthetic_only","n_truth":len(truth),"n_assigned":len(assigned),
         "correct_assigned":correct,"assigned_accuracy":correct/len(assigned) if assigned else None,
         "correct_coverage":correct/len(truth) if truth else None,
         "warning":"Synthetic ground truth cannot substitute independent expert annotations."}
    (results/"synthetic_evaluation.json").write_text(json.dumps(out,indent=2))
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(prog="astrocelllink",description="Research prototype: astroCaST event → tracked astrocyte")
    sub=parser.add_subparsers(dest="mode",required=True)
    run=sub.add_parser("run",help="Process astroCaST .roi/event_map.tiff plus morphology or masks")
    run.add_argument("--event-map",required=True,help="Path to astroCaST .roi directory, .tiff, or .npy")
    g=run.add_mutually_exclusive_group(required=True)
    g.add_argument("--instance-masks",help="2D static or 3D per-frame instance-label TIFF/NPY/HDF5")
    g.add_argument("--structural",help="Morphology-channel image; simple watershed baseline only")
    run.add_argument("--masks-already-tracked",action="store_true",help="Do not track; IDs in masks already correspond across time")
    run.add_argument("--output",default="astrocelllink_results")
    run.add_argument("--event-dataset",default=None,help="HDF5 dataset path, if input is .h5")
    run.add_argument("--mask-dataset",default=None)
    run.add_argument("--track-distance",type=float,default=12.0)
    run.add_argument("--track-min-iou",type=float,default=.02)
    run.add_argument("--max-gap",type=int,default=1)
    run.add_argument("--dilate-pixels",type=int,default=0)
    run.add_argument("--min-overlap-fraction",type=float,default=.15)
    run.add_argument("--min-purity",type=float,default=.70)
    run.add_argument("--min-margin",type=float,default=.25)
    run.add_argument("--segmentation-min-area",type=int,default=30)
    run.add_argument("--segmentation-min-distance",type=int,default=8)
    demo=sub.add_parser("demo",help="Create synthetic video, run pipeline, check synthetic truth")
    demo.add_argument("--output",default="demo_output")
    demo.add_argument("--use-auto-segmentation",action="store_true",help="Try watershed instead of supplied synthetic instance masks")
    prep=sub.add_parser("prepare-h5", help="Stream a pixel-stride copy for same-grid detector comparison")
    prep.add_argument("--input",required=True)
    prep.add_argument("--dataset",default="dff/ch0")
    prep.add_argument("--stride",type=int,default=2)
    prep.add_argument("--output",required=True)

    comp=sub.add_parser("compare-events",help="3D voxel/event IoU between candidate and official astroCaST labels (agreement only)")
    comp.add_argument("--candidate",required=True)
    comp.add_argument("--astrocast",required=True)
    comp.add_argument("--min-iou",type=float,default=.10)
    comp.add_argument("--frame",type=int,default=46)
    comp.add_argument("--output",default="comparison_results")

    seg=sub.add_parser("segment-structural",help="Watershed ONLY: prepare provisional masks from morphology (not validated)")
    seg.add_argument("--input",required=True)
    seg.add_argument("--dataset",help="Required for HDF5, e.g. morphology/ch0")
    seg.add_argument("--stride",type=int,default=1)
    seg.add_argument("--projection",choices=("mean","frame"),default="mean")
    seg.add_argument("--frame",type=int)
    seg.add_argument("--min-area",type=int,default=30)
    seg.add_argument("--min-distance",type=int,default=8)
    seg.add_argument("--output",default="provisional_masks")

    hyst=sub.add_parser("hysteresis", help="Experimental spatiotemporal growth from existing event seeds (NOT biological validation)")
    hyst.add_argument("--candidate", required=True, help="Existing seed event-label TIFF or astroCaST .roi folder")
    hyst.add_argument("--movie", required=True, help="Co-registered calcium movie HDF5/TIFF/NPY")
    hyst.add_argument("--dataset", default="dff/ch0", help="Dataset path when movie is HDF5")
    hyst.add_argument("--reference", default=None, help="Optional astroCaST output for agreement AFTER refinement only")
    hyst.add_argument("--output", default="real_data/hysteresis_v03", help="New empty result directory")
    hyst.add_argument("--weak-z", type=float, default=1.0)
    hyst.add_argument("--spatial-radius", type=float, default=2.0)
    hyst.add_argument("--temporal-radius", type=int, default=2)
    hyst.add_argument("--connectivity", type=int, choices=(1,2,3), default=1, help="3D connectivity 6, 18, or 26 neighbors")
    hyst.add_argument("--min-consecutive", type=int, choices=(1,2), default=2)
    hyst.add_argument("--baseline", choices=("global_median","rolling_percentile"), default="global_median")
    hyst.add_argument("--rolling-window", type=int, default=41)
    hyst.add_argument("--rolling-percentile", type=float, default=20.0)
    hyst.add_argument("--frame", type=int, default=180)

    val=sub.add_parser("evaluate-masks",help="Instance IoU/precision/recall against independent expert mask labels")
    val.add_argument("--predicted",required=True)
    val.add_argument("--reference",required=True)
    val.add_argument("--iou",type=float,default=.5)
    val.add_argument("--output",default="mask_evaluation")

    args=parser.parse_args(argv)
    if args.mode=="demo":
        root=Path(args.output)
        raw=write_demo_data(root/"input")
        results=root/"results"
        structural=read_array(raw/"structural.tiff")
        kwargs={"structural":structural} if args.use_auto_segmentation else {"instance_masks":read_array(raw/"local_instance_masks.tiff"),"structural":structural}
        summary,assignments,tracks=run_pipeline(read_array(raw/"sample_video.roi"),output_dir=results,**kwargs)
        truth=evaluate_demo(raw,results,assignments,tracks)
        print(json.dumps({"summary":summary,"synthetic_validation":truth,"output":str(results)},indent=2))
    elif args.mode=="prepare-h5":
        from .h5prep import prepare_stride
        summary=prepare_stride(args.input,args.dataset,args.output,stride=args.stride)
        print(json.dumps(summary,indent=2))
    elif args.mode=="compare-events":
        from .baseline_compare import compare_files
        summary,_=compare_files(args.candidate,args.astrocast,
                 min_iou=args.min_iou,frame=args.frame,output=args.output)
        print(json.dumps(summary,indent=2))
    elif args.mode=="hysteresis":
        from .hysteresis import HysteresisConfig, run_hysteresis_files
        config=HysteresisConfig(weak_z=args.weak_z, spatial_radius=args.spatial_radius,
            temporal_radius=args.temporal_radius, connectivity=args.connectivity,
            min_consecutive=args.min_consecutive, baseline_method=args.baseline,
            rolling_window=args.rolling_window, rolling_percentile=args.rolling_percentile)
        summary=run_hysteresis_files(args.candidate, args.movie, dataset=args.dataset,
            reference=args.reference, output=args.output, config=config, frame=args.frame)
        print(json.dumps(summary, indent=2))
    elif args.mode=="segment-structural":
        from .segmentation_workflow import make_segmentation
        summary=make_segmentation(args.input,dataset=args.dataset,stride=args.stride,
                 projection=args.projection,frame=args.frame,min_area=args.min_area,
                 min_distance=args.min_distance,output=args.output)
        print(json.dumps(summary,indent=2))
    elif args.mode=="evaluate-masks":
        from .mask_validation import evaluate_files
        summary=evaluate_files(args.predicted,args.reference,
                               iou_threshold=args.iou,output=args.output)
        print(json.dumps(summary,indent=2))
    elif args.mode=="run":
        ev=read_array(args.event_map,args.event_dataset)
        st=read_array(args.structural,args.mask_dataset) if args.structural else None
        masks=read_array(args.instance_masks,args.mask_dataset) if args.instance_masks else None
        summary,_,_=run_pipeline(ev,instance_masks=masks,structural=st,
                     already_tracked=args.masks_already_tracked,output_dir=args.output,
                     track_distance=args.track_distance,track_min_iou=args.track_min_iou,
                     max_gap=args.max_gap,dilate_pixels=args.dilate_pixels,
                     min_overlap_fraction=args.min_overlap_fraction,
                     min_purity=args.min_purity,min_margin=args.min_margin,
                     segmentation_min_area=args.segmentation_min_area,
                     segmentation_min_distance=args.segmentation_min_distance)
        print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
