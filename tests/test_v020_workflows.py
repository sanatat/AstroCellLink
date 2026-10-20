import json
from pathlib import Path
import h5py
import numpy as np
import pytest
import tifffile
from astrocelllink.baseline_compare import compare_arrays, compare_files
from astrocelllink.mask_validation import evaluate_masks, evaluate_files
from astrocelllink.h5prep import prepare_stride
from astrocelllink.segmentation_workflow import load_projection, make_segmentation
from astrocelllink.cli import main


def test_prepare_h5_aligns_pixel_stride_and_does_not_overwrite(tmp_path):
    inp=tmp_path/'input.h5'; out=tmp_path/'strided.h5'
    data=np.arange(4*8*7,dtype=np.float32).reshape((4,8,7))
    with h5py.File(inp,'w') as f:f.create_dataset('dff/ch0', data=data)
    ret=prepare_stride(inp,'dff/ch0',out,stride=2)
    assert ret['output_shape']==[4,4,4]
    with h5py.File(out,'r') as f:
        np.testing.assert_array_equal(f['dff/ch0'][:],data[:,::2,::2])
    assert (tmp_path/'strided_provenance.json').exists()
    with pytest.raises(FileExistsError): prepare_stride(inp,'dff/ch0',out,stride=2)


def test_compare_events_has_matching_and_split_unmatched(tmp_path):
    candidate=np.zeros((5,24,24),dtype=np.int32)
    upstream=np.zeros_like(candidate)
    candidate[0:3,3:8,3:8]=12
    candidate[2:4,12:18,12:18]=13
    upstream[0:3,3:8,3:8]=99
    upstream[2:4,14:20,14:20]=100
    summ,matches=compare_arrays(candidate,upstream,min_iou=.1,output=tmp_path)
    assert summ['n_candidate_events']==summ['n_astrocast_events']==2
    assert summ['matched_pairs_greedy']==2
    assert len(matches)==2
    assert summ['foreground_jaccard']<1
    assert (tmp_path/'comparison.json').exists()
    assert (tmp_path/'event_matches.csv').exists()
    assert (tmp_path/'agreement_preview.png').exists()


def test_compare_events_rejects_wrong_grid():
    with pytest.raises(ValueError,match='Different shapes'):
        compare_arrays(np.zeros((4,11,11),int),np.zeros((4,12,12),int))


def test_compare_events_empty_case(tmp_path):
    s,m=compare_arrays(np.zeros((3,12,12),int),np.zeros((3,12,12),int),output=tmp_path)
    assert s['n_candidate_events']==0
    assert m==[]
    assert s['candidate_event_match_fraction'] is None


def test_mask_validation_independent_reference(tmp_path):
    a=np.zeros((30,30),np.int32);b=np.zeros_like(a)
    a[2:9,2:9]=1;a[17:23,17:23]=2
    b[2:9,2:9]=40;b[17:23,17:23]=41
    s=evaluate_masks(a,b,iou_threshold=.5,output=tmp_path)
    assert s['matched_at_threshold']==2
    assert s['precision']==s['recall']==1.0
    assert (tmp_path/'mask_validation.json').exists()
    # False positive cell is reported, not automatically attributed.
    a[11:14,11:14]=3
    s=evaluate_masks(a,b,iou_threshold=.5)
    assert s['false_positives']==1
    assert s['recall']==1


def test_projection_mean_and_watershed_screen(tmp_path):
    y,x=np.indices((60,60))
    structural=np.exp(-((x-18)**2+(y-18)**2)/(2*4**2))
    structural+=np.exp(-((x-43)**2+(y-42)**2)/(2*4**2))
    inp=tmp_path/'structural.h5'
    stack=np.stack([structural,structural*2])
    with h5py.File(inp,'w') as f: f.create_dataset('morph/ch0',data=stack)
    projection=load_projection(inp,dataset='morph/ch0',stride=2)
    np.testing.assert_allclose(projection,1.5*structural[::2,::2],atol=1e-6)
    out=tmp_path/'maskout'
    summary=make_segmentation(inp,dataset='morph/ch0',stride=2,min_area=8,min_distance=5,output=out)
    assert summary['proposed_instances']==2
    assert tuple(tifffile.imread(out/'proposed_instance_masks.tiff').shape)==(30,30)
    assert (out/'proposed_masks_qc.png').exists()
    assert 'NOT validated' in summary['status']


def test_v2_cli_end_to_end(tmp_path):
    raw=tmp_path/'raw.h5'
    source=np.arange(3*12*12,dtype='float32').reshape((3,12,12))
    with h5py.File(raw,'w') as h:h.create_dataset('dff/ch0',data=source)
    small=tmp_path/'small.h5'
    main(['prepare-h5','--input',str(raw),'--output',str(small),'--stride','2'])
    a=np.zeros((3,6,6),dtype=np.int32);a[:,1:4,1:4]=1
    b=a.copy();b[b==1]=12
    ca=tmp_path/'a.tiff';cb=tmp_path/'b.tiff'
    tifffile.imwrite(ca,a,photometric="minisblack");tifffile.imwrite(cb,b,photometric="minisblack")
    out=tmp_path/'compare'
    main(['compare-events','--candidate',str(ca),'--astrocast',str(cb),'--output',str(out)])
    assert json.loads((out/'comparison.json').read_text())['matched_pairs_greedy']==1
    mask1=tmp_path/'p.tiff';mask2=tmp_path/'r.tiff'
    tifffile.imwrite(mask1,a[0]);tifffile.imwrite(mask2,b[0]);mout=tmp_path/'evaluation'
    main(['evaluate-masks','--predicted',str(mask1),'--reference',str(mask2),'--output',str(mout)])
    assert json.loads((mout/'mask_validation.json').read_text())['f1']==1.0
