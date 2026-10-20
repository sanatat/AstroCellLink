"""Conservative 3D temporal-spatial attribution of astroCaST events to tracked cells."""
import numpy as np
from scipy.ndimage import binary_dilation


def attribute_events(event_map, tracked_masks, *, dilate_pixels=0, min_overlap_fraction=0.15,
                     min_purity=0.70, min_margin=0.25):
    """Assign event labels (T,Y,X) to stable cell IDs (T,Y,X).

    Quality is a heuristic, NOT a calibrated probability. Events that touch
    multiple cells are marked ambiguous; non-overlapping are unassigned.
    Per-event memory usage scales with event-local frames, not the full movie.
    """
    events = np.asarray(event_map)
    cells = np.asarray(tracked_masks)
    if events.ndim != 3 or events.shape != cells.shape:
        raise ValueError(f"Event-map and masks must have identical 3D shapes: {events.shape}, {cells.shape}")
    if not np.issubdtype(events.dtype, np.integer) or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("events and cells require nonnegative integer labels")
    if np.any(events < 0) or np.any(cells < 0):
        raise ValueError("negative labels not permitted")
    if not (0 <= min_overlap_fraction <= 1 and 0 <= min_purity <= 1 and 0 <= min_margin <= 1):
        raise ValueError("fractions must lie in [0,1]")
    if dilate_pixels < 0 or int(dilate_pixels) != dilate_pixels:
        raise ValueError("dilate_pixels must be a nonnegative integer")

    unique_events = np.unique(events)
    unique_events = unique_events[unique_events > 0]
    if not len(unique_events):
        return []
    # For high event counts, avoid scanning the entire stack once per event:
    # assign each occupied voxel to one tracked mask by ID pair histogram.
    overlaps = {}
    total = {}
    frame_min = {}
    frame_max = {}
    for t in range(len(events)):
        ev = events[t]
        cm = cells[t]
        active = ev > 0
        if not active.any():
            continue
        ids, counts = np.unique(ev[active], return_counts=True)
        for event_id,n in zip(ids,counts):
            event_id=int(event_id)
            total[event_id]=total.get(event_id,0)+int(n)
            frame_min[event_id]=min(t,frame_min.get(event_id,t))
            frame_max[event_id]=max(t,frame_max.get(event_id,t))
        if dilate_pixels:
            # Dilate cell masks into unlabeled pixels ONLY. Ambiguous dilation is
            # represented as ID=0 to avoid arbitrary tie-breaking.
            expanded = cm.copy()
            eligible = cm == 0
            fill_count = np.zeros(cm.shape, dtype=np.uint8)
            fill_id = np.zeros(cm.shape, dtype=np.int32)
            for cell_id in np.unique(cm):
                if cell_id == 0:
                    continue
                dilated = binary_dilation(cm == cell_id, iterations=int(dilate_pixels)) & eligible
                fill_count[dilated] = np.minimum(fill_count[dilated].astype(np.int16) + 1, 255)
                fill_id[dilated] = cell_id
            expanded[eligible & (fill_count == 1)] = fill_id[eligible & (fill_count == 1)]
            cm = expanded
        pair_e = ev[active & (cm > 0)].astype(np.int64)
        pair_c = cm[active & (cm > 0)].astype(np.int64)
        if pair_e.size:
            pair_unique, pair_counts = np.unique(np.column_stack((pair_e,pair_c)),axis=0,return_counts=True)
            for (event_id,cell_id),n in zip(pair_unique,pair_counts):
                key=(int(event_id),int(cell_id))
                overlaps[key]=overlaps.get(key,0)+int(n)
    by_event={}
    for (event_id,cell_id),value in overlaps.items():
        by_event.setdefault(event_id,[]).append((cell_id,value))
    rows=[]
    for event_id in unique_events:
        event_id=int(event_id)
        voxel_count=total[event_id]
        ranks=sorted(by_event.get(event_id,[]),key=lambda a:(-a[1],a[0]))
        best_id,best=(ranks[0] if ranks else (None,0))
        second_id,second=(ranks[1] if len(ranks)>1 else (None,0))
        overlap_count=sum(v for _,v in ranks)
        overlap_fraction=best/voxel_count
        purity=best/overlap_count if overlap_count else 0.0
        margin=(best-second)/overlap_count if overlap_count else 0.0
        if not ranks:
            status="no_overlap"
        elif overlap_fraction < min_overlap_fraction:
            status="weak_overlap"
        elif purity < min_purity or margin < min_margin:
            status="ambiguous"
        else:
            status="assigned"
        rows.append({"event_id":event_id, "track_id":best_id if status == "assigned" else None,
                     "status":status,"candidate_track_id":best_id,
                     "second_best_track_id":second_id,
                     "event_voxels":voxel_count,"candidate_overlap_voxels":best,
                     "total_overlap_voxels":overlap_count,
                     "overlap_fraction":round(overlap_fraction,5),"purity":round(purity,5),
                     "margin":round(margin,5),"n_candidate_cells":len(ranks),
                     "first_frame":frame_min[event_id],"last_frame":frame_max[event_id]})
    return rows
