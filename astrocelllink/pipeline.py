"""Reusable end-to-end baseline, IO, measurement and summaries."""
from pathlib import Path
import csv,json
import numpy as np
from .io import read_array, labels_3d, write_tiff
from .segmentation import segment_structural_stack
from .tracking import track_instances, summarize_already_tracked
from .attribution import attribute_events
from .visualize import qc_figure


def _csv(path, rows, default_columns):
    with open(path,"w",newline="",encoding="utf-8") as f:
        fields=list(rows[0]) if rows else default_columns
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(event_map, *, instance_masks=None, structural=None,
                 already_tracked=False, output_dir="results", track_distance=12,
                 track_min_iou=.02, max_gap=1, dilate_pixels=0,
                 min_overlap_fraction=.15, min_purity=.70, min_margin=.25,
                 segmentation_min_area=30, segmentation_min_distance=8):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    events=labels_3d(event_map,"event_map")
    original_struct=None
    if structural is not None:
        original_struct=np.asarray(structural)
        if original_struct.shape not in (events.shape,events.shape[1:]):
            raise ValueError(f"structural {original_struct.shape} must match events {events.shape} or spatial shape")
    if instance_masks is None:
        if structural is None:
            raise ValueError("Need expert instance masks OR structural image for baseline segmentation")
        instance_masks=segment_structural_stack(structural,min_area=segmentation_min_area,
                                                min_distance=segmentation_min_distance)
        source="baseline_structural_segmentation"
    else:
        source="supplied_cell_masks"
    cells=labels_3d(instance_masks,"instance_masks",allow_2d=True,n_frames=events.shape[0])
    if cells.shape!=events.shape:
        raise ValueError(f"Cell masks {cells.shape} do not match event map {events.shape}")
    if already_tracked:
        tracks=cells.copy()
        observations=summarize_already_tracked(tracks)
    else:
        tracks,observations=track_instances(cells,max_distance=track_distance,
                                            min_iou=track_min_iou,max_gap=max_gap)
    assignments=attribute_events(events,tracks,dilate_pixels=dilate_pixels,
                                 min_overlap_fraction=min_overlap_fraction,
                                 min_purity=min_purity,min_margin=min_margin)
    write_tiff(output_dir/"tracked_cells.tiff",tracks)
    _csv(output_dir/"tracks.csv",observations,["frame","track_id","local_label","center_y","center_x","area_px"])
    _csv(output_dir/"event_attribution.csv",assignments,
         ["event_id","track_id","status","candidate_track_id","second_best_track_id", "event_voxels",
          "candidate_overlap_voxels","total_overlap_voxels","overlap_fraction","purity","margin",
          "n_candidate_cells","first_frame","last_frame"])
    counts={s:sum(1 for e in assignments if e["status"]==s) for s in
            ("assigned","no_overlap","weak_overlap","ambiguous")}
    summary={"status":"PROTOTYPE / NOT CLINICALLY OR BIOLOGICALLY VALIDATED",
             "input_source":source,"n_frames":len(events),
             "n_tracks":len(set(int(x) for x in np.unique(tracks) if x>0)),
             "n_events":len(assignments),"events_by_status":counts,
             "assignment_rate":(counts["assigned"]/len(assignments) if assignments else 0),
             "thresholds":{"max_distance":track_distance,"min_iou":track_min_iou,"max_gap":max_gap,
                           "dilate_pixels":dilate_pixels,"min_overlap_fraction":min_overlap_fraction,
                           "min_purity":min_purity,"min_margin":min_margin},
             "note":"Purity/margin are geometric heuristics, not probabilities. Track IDs need biological validation."}
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    qc_figure(events,tracks,assignments,output_dir/"quality_control.png",structural=original_struct)
    return summary,assignments,tracks
