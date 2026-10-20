"""Synthetic algorithm checks only; these are NOT ex vivo biological validation."""
from pathlib import Path
import json

import h5py
import numpy as np
import pytest
import tifffile
from astrocelllink.hysteresis import (
    HysteresisConfig, calculate_score, refine_events, bounded_corridor,
    run_hysteresis_files,
)
from astrocelllink.cli import main


def make_movie(seed=13):
    rng = np.random.default_rng(seed)
    movie = rng.normal(0, .45, (25, 44, 48)).astype(np.float32)
    seeds = np.zeros_like(movie, dtype=np.int32)
    truth = np.zeros_like(seeds)
    yy, xx = np.indices((44,48))
    for event_id, center, onset, duration in [(12,(11,13),6,8),(91,(32,35),13,8)]:
        y, x = center
        wide = (xx-x)**2 + (yy-y)**2 <= 4**2
        small = (xx-x)**2 + (yy-y)**2 <= 1**2
        for t in range(onset, onset+duration):
            movie[t,wide] += 3.2
            truth[t,wide] = event_id
        seeds[onset+2:onset+duration-2,small] = event_id
    return movie, seeds, truth


def test_refinement_recovers_synthetic_halos_and_temporal_tails():
    movie, seeds, truth = make_movie()
    refined, report = refine_events(movie, seeds,
          HysteresisConfig(weak_z=1.0, spatial_radius=5, temporal_radius=3))
    assert np.all(refined[seeds>0] == seeds[seeds>0])
    assert set(np.unique(refined)) == {0,12,91}
    recovered = np.count_nonzero((refined == truth) & (truth > 0))
    orig_recovered = np.count_nonzero((seeds == truth) & (truth > 0))
    assert recovered > orig_recovered * 3
    assert np.count_nonzero(refined[0]) == 0
    assert report['candidate_event_ids'] == report['refined_event_ids'] == 2
    assert report['added_voxels'] > 0


def test_cannot_expand_into_empty_far_future_or_far_away():
    movie, seeds, _ = make_movie()
    movie[24] += 12.0
    grown, _ = refine_events(movie,seeds,HysteresisConfig(spatial_radius=3,temporal_radius=1))
    assert not grown[24].any()  # no original seed near frame 24
    assert np.all(grown[:,0,0] == 0)
    assert np.all(grown[:,43,0] == 0)


def test_weak_bridge_never_merges_distinct_ids_even_if_ambiguous():
    rng=np.random.default_rng(7)
    movie=rng.normal(0,.1,(13,20,22)).astype('float32')
    seeds=np.zeros_like(movie,dtype='int32')
    seeds[4:7,10,6]=100
    seeds[4:7,10,16]=200
    movie[3:9,9:12,4:19] += 10.0
    refined,summ=refine_events(movie,seeds,HysteresisConfig(weak_z=.5,spatial_radius=8,temporal_radius=2))
    assert np.array_equal(refined[seeds>0],seeds[seeds>0])
    assert set(np.unique(refined)) == {0,100,200}
    assert summ['components_with_competing_seed_ids'] >= 1
    assert np.count_nonzero(refined==100) > 3
    assert np.count_nonzero(refined==200) > 3


def test_empty_input_no_false_discoveries():
    movie=np.ones((6,9,9),dtype=np.float32)
    seeds=np.zeros_like(movie,dtype=np.int32)
    grown,summ=refine_events(movie,seeds)
    assert not grown.any()
    assert summ['candidate_event_ids'] == 0
    assert summ['growth_factor'] is None


def test_flat_movie_does_not_create_halo():
    movie=np.ones((8,14,14),dtype=np.float32)
    seeds=np.zeros_like(movie,dtype=np.int32)
    seeds[3,6,6]=4
    grown,summ=refine_events(movie,seeds)
    np.testing.assert_array_equal(grown,seeds)
    assert summ['added_voxels']==0


def test_disconnected_weak_signal_cannot_teleport_to_it():
    movie,seeds,_=make_movie()
    movie[10,0:3,44:47]+=500
    grown,_=refine_events(movie,seeds,HysteresisConfig(weak_z=.05,spatial_radius=2,temporal_radius=2))
    assert np.count_nonzero(grown[10,0:3,44:47])==0


def test_validation_rejects_shape_nan_and_bad_options():
    movie,seeds,_=make_movie()
    with pytest.raises(ValueError, match='Shape mismatch'):
        refine_events(movie,seeds[:-1])
    movie[0,0,0]=np.nan
    with pytest.raises(ValueError, match='Non-finite'):
        refine_events(movie,seeds)
    with pytest.raises(ValueError, match='connectivity'):
        HysteresisConfig(connectivity=4).validate()


def test_file_cli_and_repeat_output_guard(tmp_path):
    movie,seeds,truth=make_movie()
    p=tmp_path/'candidate.tiff';tifffile.imwrite(p,seeds,photometric='minisblack')
    h=tmp_path/'raw.h5'
    with h5py.File(h,'w') as f: f.create_dataset('dff/ch0',data=movie)
    upstream=tmp_path/'reference.tiff';tifffile.imwrite(upstream,truth,photometric='minisblack')
    result=tmp_path/'output'
    main(['hysteresis','--candidate',str(p),'--movie',str(h),
          '--reference',str(upstream),'--output',str(result),'--spatial-radius','5',
          '--temporal-radius','3','--frame','12'])
    refined=tifffile.imread(result/'refined_event_map.tiff')
    assert refined.shape==seeds.shape
    assert (result/'hysteresis_qc.png').stat().st_size > 5000
    assert (result/'per_event_growth.csv').is_file()
    assert (result/'agreement_before'/'comparison.json').is_file()
    assert (result/'agreement_after'/'comparison.json').is_file()
    meta=json.loads((result/'hysteresis_summary.json').read_text())
    assert meta['candidate_event_ids']==2
    assert 'agreement_with_astrocast' in meta
    ev = meta['agreement_with_astrocast']
    assert ev['newly_added_inside_astrocast_voxels'] + ev['newly_added_outside_astrocast_voxels'] == meta['added_voxels']
    with pytest.raises(FileExistsError, match='Refusing'):
        run_hysteresis_files(p,h,output=result)


def test_rolling_baseline_option_operates():
    movie,seeds,_=make_movie()
    grown,report=refine_events(movie,seeds,HysteresisConfig(
        baseline_method='rolling_percentile',rolling_window=9,rolling_percentile=20.0))
    assert grown.shape==seeds.shape
    assert report['refined_event_ids']==2
    assert np.array_equal(grown[seeds>0],seeds[seeds>0])
