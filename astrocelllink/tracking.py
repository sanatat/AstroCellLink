"""Frame-to-frame soma tracking with gated Hungarian assignment.

Includes short gaps, but does not solve mitosis, arborization or touching cells.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackState:
    cy: float
    cx: float
    pixels: np.ndarray
    last_seen: int
    area: int


def _regions(label_frame):
    result = {}
    for n in np.unique(label_frame):
        if n == 0:
            continue
        pix = label_frame == n
        points = np.argwhere(pix)
        if not len(points):
            continue
        cy, cx = np.mean(points, axis=0)
        result[int(n)] = (pix, float(cy), float(cx), int(pix.sum()))
    return result


def track_instances(masks, max_distance=12.0, min_iou=0.02, max_gap=1):
    """Return (T,Y,X) stable ID labels and per-observation table.

    Matching uses centroid-distance gating and IoU-adjusted cost.
    IDs are not reused. max_gap=1 means 1 missing frame is tolerated.
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError("instance masks must have shape (T,Y,X)")
    if not np.issubdtype(masks.dtype, np.integer) or np.any(masks < 0):
        raise ValueError("instance masks must contain nonnegative integers")
    if max_distance <= 0 or not 0 <= min_iou <= 1 or max_gap < 0:
        raise ValueError("invalid tracking thresholds")
    result = np.zeros(masks.shape, dtype=np.int32)
    states = {}
    rows = []
    next_id = 1
    for t, frame in enumerate(masks):
        regions = _regions(frame)
        old_ids = [i for i,s in states.items() if t - s.last_seen - 1 <= max_gap]
        new_ids = list(regions)
        matched = {}
        if old_ids and new_ids:
            cost = np.full((len(old_ids), len(new_ids)), 1e6)
            for i, old_id in enumerate(old_ids):
                prev = states[old_id]
                for j, local_id in enumerate(new_ids):
                    mask, cy, cx, area = regions[local_id]
                    d = np.hypot(cy-prev.cy, cx-prev.cx)
                    if d > max_distance * max(1, t-prev.last_seen):
                        continue
                    intersect = np.count_nonzero(prev.pixels & mask)
                    union = prev.area + area - intersect
                    iou = intersect / union if union else 0
                    # When an object misses a frame, allow centroid-only matching.
                    if iou < min_iou and d > max_distance * 0.40:
                        continue
                    cost[i,j] = d / max_distance + (1 - iou)
            rr, cc = linear_sum_assignment(cost)
            for i,j in zip(rr,cc):
                if cost[i,j] < 1e6:
                    matched[new_ids[j]] = old_ids[i]
        for local_id, (pix, cy, cx, area) in regions.items():
            track_id = matched.get(local_id)
            if track_id is None:
                track_id = next_id
                next_id += 1
            result[t,pix] = track_id
            states[track_id] = TrackState(cy,cx,pix,t,area)
            rows.append({"frame":t,"track_id":track_id,"local_label":local_id,
                         "center_y":round(cy,3),"center_x":round(cx,3),"area_px":area})
        # Remove old tracks so memory does not grow with the number of disconnected objects.
        states = {i:s for i,s in states.items() if t - s.last_seen <= max_gap}
    return result, rows


def summarize_already_tracked(masks):
    """Summarize expert-provided stable-ID masks without relinking."""
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError("masks must be (T,Y,X)")
    rows = []
    for t, frame in enumerate(masks):
        for track_id, (_,cy,cx,area) in _regions(frame).items():
            rows.append({"frame":t,"track_id":track_id,"local_label":track_id,
                         "center_y":round(cy,3),"center_x":round(cx,3),"area_px":area})
    return rows
