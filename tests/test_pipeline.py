import csv
import json
import numpy as np
import pytest
from astrocelllink.attribution import attribute_events
from astrocelllink.io import read_array, labels_3d
from astrocelllink.tracking import track_instances
from astrocelllink.segmentation import segment_structural_stack
from astrocelllink.pipeline import run_pipeline
from astrocelllink.synthetic import simulate,write_demo_data


def test_tracking_handles_random_frame_local_ids():
    local=np.zeros((4,30,30),int)
    for t in range(4):
        local[t,4:10,4+t:10+t] = 9 if t%2 else 3
        local[t,17:23,18-t:24-t] = 3 if t%2 else 9
    tracked,obs=track_instances(local)
    assert (tracked[:,6,5:8]>0).any()
    assert len(set(np.unique(tracked)) - {0})==2
    assert len(obs)==8
    assert len({int(tracked[t,6,7]) for t in range(4)})==1
    assert len({int(tracked[t,19,19]) for t in range(4)})==1


def test_tracking_can_bridge_single_missing_frame():
    arr=np.zeros((3,20,20),np.int32)
    arr[0,5:10,5:10]=1
    arr[2,5:10,6:11]=17
    tracks,_=track_instances(arr,max_gap=1)
    assert tracks[0,7,7] == tracks[2,7,7]


def test_attribution_assigned():
    ev=np.zeros((3,20,20),int); cell=np.zeros_like(ev)
    cell[:,:,3:10]=5
    ev[0:2,5:8,5:8]=100
    rows=attribute_events(ev,cell)
    assert rows[0]["track_id"]==5
    assert rows[0]["status"]=="assigned"
    assert rows[0]["purity"]==1


def test_attribution_ambiguous():
    ev=np.zeros((2,20,20),int); cell=np.zeros_like(ev)
    cell[:,:,2:9]=2; cell[:,:,9:16]=3
    ev[0,5:10,6:12]=1
    r=attribute_events(ev,cell)[0]
    assert r["status"]=="ambiguous"
    assert r["track_id"] is None
    assert r["n_candidate_cells"] == 2


def test_attribution_no_overlap():
    ev=np.zeros((2,20,20),int); cells=np.zeros_like(ev)
    ev[0,12:15,12:15]=1
    cells[0,2:5,2:5]=1
    row=attribute_events(ev,cells)[0]
    assert row["status"]=="no_overlap"
    assert row["track_id"] is None


def test_attribution_weak_overlap():
    ev=np.zeros((1,20,20),int); cells=np.zeros_like(ev)
    ev[0,2:12,2:12]=1; cells[0,2,2]=1
    row=attribute_events(ev,cells)[0]
    assert row["status"]=="weak_overlap"


def test_attribution_dilation_uses_zero_background_only():
    ev=np.zeros((1,15,15),int); cells=np.zeros_like(ev)
    cells[0,7,4]=10; cells[0,7,10]=20
    ev[0,7,5]=1
    row=attribute_events(ev,cells,dilate_pixels=1)[0]
    assert row["track_id"]==10


def test_segmentation_of_two_soma():
    y,x=np.indices((66,66))
    signal=np.exp(-((y-19)**2+(x-19)**2)/(2*6**2))
    signal+=np.exp(-((y-46)**2+(x-46)**2)/(2*6**2))
    inst=segment_structural_stack(signal,min_area=24,min_distance=9)
    assert len(np.unique(inst[inst>0]))==2


def test_label_validation_rejects_noninteger():
    with pytest.raises(ValueError):
        labels_3d(np.array([[[1.2]]]),"event")


def test_actual_astrocast_roi_directory_format(tmp_path):
    source=write_demo_data(tmp_path/"input",seed=4)
    ev=read_array(source/"sample_video.roi")
    assert ev.ndim==3
    assert ev.max() == 18


def test_end_to_end_synthetic(tmp_path):
    source=write_demo_data(tmp_path/"input",seed=9)
    summary,assigned,tracks=run_pipeline(read_array(source/"sample_video.roi"),
        instance_masks=read_array(source/"local_instance_masks.tiff"),
        structural=read_array(source/"structural.tiff"),output_dir=tmp_path/"results")
    assert summary["n_tracks"]==4
    assert summary["n_events"]==18
    assert summary["events_by_status"]["assigned"]==18
    assert (tmp_path/"results"/"event_attribution.csv").exists()
    assert (tmp_path/"results"/"quality_control.png").stat().st_size>10_000
    assert all(x["track_id"] is not None for x in assigned)


def test_requires_morphology_or_masks(tmp_path):
    with pytest.raises(ValueError,match="Need expert instance masks"):
        run_pipeline(np.zeros((2,5,5),int),output_dir=tmp_path)


def test_shapes_must_match(tmp_path):
    with pytest.raises(ValueError,match="do not match"):
        run_pipeline(np.zeros((2,20,20),int),instance_masks=np.zeros((2,10,10),int),output_dir=tmp_path)


def test_demo_cli_writes_evaluation(tmp_path):
    from astrocelllink.cli import main
    out=tmp_path/"cli_demo"
    main(["demo","--output",str(out)])
    d=json.loads((out/"results"/"synthetic_evaluation.json").read_text())
    assert d["n_truth"]==18
    assert d["correct_assigned"]==18
    assert d["kind"]=="synthetic_only"
