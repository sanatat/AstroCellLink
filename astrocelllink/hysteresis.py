"""Conservative, seed-preserving spatiotemporal hysteresis (EXPLORATORY).

Only an existing candidate event ID can seed growth. Fluorescence-defined support
must be connected in 3D, near some seed in space/time, and optionally persist in
adjacent frames. Multi-source regions are partitioned by seeded watershed and
reported as ambiguous -- not silently merged. This does NOT discover new events
or establish biological accuracy, and uses the same movie as the source seeds.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .io import labels_3d, read_array


@dataclass(frozen=True)
class HysteresisConfig:
    weak_z: float = 1.0
    spatial_radius: float = 2.0
    temporal_radius: int = 2
    connectivity: int = 1
    min_consecutive: int = 2
    sigma_floor_percentile: float = 5.0
    baseline_method: str = "global_median"
    rolling_window: int = 41
    rolling_percentile: float = 20.0

    def validate(self) -> None:
        if not np.isfinite(self.weak_z) or self.weak_z <= 0:
            raise ValueError("weak_z must be positive and finite")
        if not np.isfinite(self.spatial_radius) or self.spatial_radius < 0 or self.spatial_radius > 100:
            raise ValueError("spatial_radius must be in [0,100]")
        if not 0 <= self.temporal_radius <= 30:
            raise ValueError("temporal_radius must be 0..30")
        if self.connectivity not in (1, 2, 3):
            raise ValueError("connectivity must be 1 (6), 2 (18), or 3 (26)")
        if self.min_consecutive not in (1, 2):
            raise ValueError("min_consecutive must be 1 or 2")
        if not 0 <= self.sigma_floor_percentile <= 50:
            raise ValueError("sigma_floor_percentile must be 0..50")
        if self.baseline_method not in ("global_median", "rolling_percentile"):
            raise ValueError("Unsupported baseline_method")
        if self.rolling_window < 3 or self.rolling_window % 2 != 1:
            raise ValueError("rolling_window must be odd and >=3")
        if not 0 < self.rolling_percentile < 100:
            raise ValueError("rolling_percentile must be in (0,100)")


def calculate_score(movie: np.ndarray, config: HysteresisConfig):
    """Robust per-pixel fluorescence z-score, not a calibrated statistical z-test.

    Signal baseline is median over time by default. Optional rolling temporal
    percentile uses the same observed movie (exploratory, not independent).
    Noise scale is global-in-time per-pixel MAD with a spatial percentile floor.
    """
    config.validate()
    m = np.asarray(movie, dtype=np.float32)
    if m.ndim != 3 or min(m.shape) < 2:
        raise ValueError("movie must be a nonempty 3D (time,row,column) stack")
    if not np.isfinite(m).all():
        raise ValueError("Non-finite movie values are not supported; inspect preprocessing")
    median = np.median(m, axis=0).astype(np.float32)
    deviation = np.abs(m - median[None])
    sigma = (1.4826 * np.median(deviation, axis=0)).astype(np.float32)
    positive = sigma[sigma > 0]
    floor = max(float(np.percentile(positive, config.sigma_floor_percentile)), 1e-5) if positive.size else 1e-5
    sigma = np.maximum(sigma, floor)
    del deviation
    if config.baseline_method == "global_median":
        score = (m - median[None]) / sigma[None]
    else:
        # This filter is memory bounded but slower than a global baseline.
        baseline = ndi.percentile_filter(
            m, percentile=config.rolling_percentile,
            size=(config.rolling_window, 1, 1), mode="nearest"
        )
        score = (m - baseline) / sigma[None]
        del baseline
    return np.asarray(score, dtype=np.float32), floor


def temporal_persistence(weak: np.ndarray, min_consecutive: int) -> np.ndarray:
    if min_consecutive == 1:
        return weak
    prev = np.zeros_like(weak)
    prev[1:] = weak[:-1]
    prev[:-1] |= weak[1:]
    return weak & prev


def bounded_corridor(seed: np.ndarray, spatial_radius: float, temporal_radius: int) -> np.ndarray:
    """Proximity to ANY original seed: temporal +-radius; XY Euclidean radius.

    The bound is defined around ORIGINAL seed locations, not new growth; hence
    binary propagation cannot leak indefinitely through a weak-signal bridge.
    """
    seed = np.asarray(seed, dtype=bool)
    if temporal_radius:
        near_time = ndi.maximum_filter1d(seed.astype(np.uint8), size=2*temporal_radius+1, axis=0, mode="constant") > 0
    else:
        near_time = seed
    if spatial_radius == 0:
        return near_time
    corridor = np.empty_like(near_time)
    for frame in range(len(near_time)):
        if np.any(near_time[frame]):
            corridor[frame] = ndi.distance_transform_edt(~near_time[frame]) <= spatial_radius
        else:
            corridor[frame] = False
    return corridor


def _competing_component_stats(reachable: np.ndarray, markers: np.ndarray, structure: np.ndarray) -> dict:
    components, n_components = ndi.label(reachable, structure=structure)
    seed_locations = markers > 0
    if not seed_locations.any():
        return {"reachable_components": 0, "components_with_competing_seed_ids": 0,
                "voxels_in_competing_components": 0}
    pairs = np.column_stack((components[seed_locations], markers[seed_locations]))
    pairs = np.unique(pairs, axis=0)
    component_ids, event_ids_per_component = np.unique(pairs[:, 0], return_counts=True)
    competing = component_ids[event_ids_per_component > 1]
    if len(competing):
        size = np.bincount(components.ravel())
        voxels = int(size[competing].sum())
    else:
        voxels = 0
    return {"reachable_components": int(n_components),
            "components_with_competing_seed_ids": int(len(competing)),
            "voxels_in_competing_components": voxels}


def refine_events(movie: np.ndarray, candidate_labels: np.ndarray, config: HysteresisConfig = HysteresisConfig()):
    """Returns (refined labels, diagnostic summary). Always preserves original labels.

    A positive refined pixel outside seed is conditional on all of:
    robust fluorescence support, temporal persistence, proximity to original
    seeds in space/time, and 3D connectivity to a seed.
    """
    config.validate()
    src = labels_3d(candidate_labels, "candidate_labels")
    if src.shape != np.shape(movie):
        raise ValueError(f"Shape mismatch: candidate={src.shape}, movie={np.shape(movie)}")
    if np.max(src) > np.iinfo(np.int32).max:
        raise ValueError("Event ID too large for int32 labels")
    seed = src > 0
    score, sigma_floor = calculate_score(movie, config)
    weak = np.greater_equal(score, config.weak_z)
    weak = temporal_persistence(weak, config.min_consecutive)
    corridor = bounded_corridor(seed, config.spatial_radius, config.temporal_radius)
    eligible = seed | (weak & corridor)
    del weak, corridor
    structure = ndi.generate_binary_structure(3, config.connectivity)
    reachable = ndi.binary_propagation(seed, structure=structure, mask=eligible)
    del eligible
    if not seed.any():
        refined = np.zeros_like(src)
        ambiguity = {"reachable_components": 0, "components_with_competing_seed_ids": 0,
                     "voxels_in_competing_components": 0}
    else:
        ambiguity = _competing_component_stats(reachable, src, structure)
        # Flood along accessible foreground. Flat elevation gives approximate
        # geodesic nearest-seed partition; original event IDs remain distinct.
        refined = watershed(np.zeros(src.shape, dtype=np.uint8), src,
                            mask=reachable, connectivity=structure).astype(np.int32)
        if not np.array_equal(refined[seed], src[seed]):
            raise AssertionError("Refinement must preserve every original event ID")
        if np.any((refined > 0) & ~reachable):
            raise AssertionError("Unreachable voxels were labeled")
    original_n = int(np.count_nonzero(seed))
    refined_n = int(np.count_nonzero(refined))
    ids_a = np.unique(src[src > 0])
    ids_b = np.unique(refined[refined > 0])
    if not np.array_equal(ids_a, ids_b):
        raise AssertionError("Refinement must neither create nor delete event IDs")
    summary = {
        "status": "EXPLORATORY; seed-dependent hysteresis, NOT biological ground truth",
        "shape": list(map(int, src.shape)),
        "parameters": vars(config),
        "noise_sigma_floor": float(sigma_floor),
        "candidate_event_ids": int(len(ids_a)),
        "refined_event_ids": int(len(ids_b)),
        "original_voxels": original_n,
        "refined_voxels": refined_n,
        "added_voxels": refined_n - original_n,
        "growth_factor": float(refined_n / original_n) if original_n else None,
        **ambiguity,
        "limitations": [
            "Does not discover new event identities; every grown voxel comes from an original seed.",
            "Potential split/merge ambiguities are resolved by a geometric watershed, NOT by biological evidence.",
            "Temporal and spatial bounds reduce uncontrolled growth but cannot guarantee no artificial bridges.",
            "Baseline and noise are estimated from the same movie used for detection; not independently calibrated.",
            "astroCaST is a separate detector agreement reference, not expert annotations.",
            "No biological accuracy, p-value, propagation velocity or calcium-wave claim is supported."
        ]
    }
    return refined, summary


def _save_qc(movie, original, refined, output_path, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = min(max(0, frame), len(movie)-1)
    m = np.asarray(movie[t])
    a = original[t] > 0
    b = refined[t] > 0
    lo, hi = np.percentile(m, [3, 99])
    if hi <= lo: hi = lo + 1
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout="constrained")
    axes[0].imshow(m, cmap="gray", vmin=lo, vmax=hi)
    axes[1].imshow(a, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(b, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(m, cmap="gray", vmin=lo, vmax=hi)
    extra = np.ma.masked_where(~(b & ~a), b)
    axes[3].imshow(extra, cmap="autumn", alpha=.60, vmin=0, vmax=1)
    for ax, title in zip(axes, ["dFF (original)", "Original event seeds", "Hysteresis foreground",
                                 "Newly added voxels (orange)"]):
        ax.set_title(title, fontsize=10); ax.axis("off")
    fig.suptitle(f"AstroCellLink exploratory hysteresis, frame {t}; NOT biological validation", fontsize=11)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _load_movie(movie_path: str, dataset: str) -> np.ndarray:
    path = Path(movie_path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(path, "r") as f:
            if dataset not in f:
                raise ValueError(f"Dataset '{dataset}' is missing. Available top-level groups: {list(f)}")
            return np.asarray(f[dataset][:], dtype=np.float32)
    return np.asarray(read_array(path), dtype=np.float32)


def run_hysteresis_files(candidate: str, movie: str, *, dataset="dff/ch0", output="hysteresis_results",
                         config: HysteresisConfig = HysteresisConfig(), frame=180, reference=None):
    dst = Path(output)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {dst}")
    originals = labels_3d(read_array(candidate), "candidate")
    observed = _load_movie(movie, dataset)
    refined, summary = refine_events(observed, originals, config)
    dst.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dst / "refined_event_map.tiff", refined.astype(np.int32),
                     photometric="minisblack", metadata={"axes": "TYX"}, compression="zlib")
    fields = ["event_id", "original_voxels", "refined_voxels", "added_voxels", "growth_factor"]
    with (dst / "per_event_growth.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        labels, original_counts = np.unique(originals[originals > 0], return_counts=True)
        new_labels, new_counts = np.unique(refined[refined > 0], return_counts=True)
        new_dict = dict(zip(new_labels.tolist(), new_counts.tolist()))
        for event_id, volume in zip(labels.tolist(), original_counts.tolist()):
            grown = int(new_dict.get(event_id, 0))
            writer.writerow({"event_id": event_id, "original_voxels": volume,
                             "refined_voxels": grown, "added_voxels": grown-volume,
                             "growth_factor": round(grown/volume, 6)})
    _save_qc(observed, originals, refined, dst / "hysteresis_qc.png", frame)
    if reference is not None:
        from .baseline_compare import compare_arrays
        target = labels_3d(read_array(reference), "reference_astrocast")
        before, _ = compare_arrays(originals, target, frame=frame, output=dst / "agreement_before")
        after, _ = compare_arrays(refined, target, frame=frame, output=dst / "agreement_after")
        newly_added = int(summary["added_voxels"])
        new_agree = int(after["foreground_intersection_voxels"] - before["foreground_intersection_voxels"])
        summary["agreement_with_astrocast"] = {
            "original": before,
            "refined": after,
            "delta_foreground_jaccard": float(after["foreground_jaccard"] - before["foreground_jaccard"]),
            "newly_added_inside_astrocast_voxels": new_agree,
            "newly_added_outside_astrocast_voxels": newly_added-new_agree,
            "newly_added_astrocast_agreement_fraction": float(new_agree / newly_added) if newly_added else None,
            "caveat": "Agreement, NOT biological accuracy. Do not optimize on the same recording and report an independent test."
        }
    (dst / "hysteresis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary
