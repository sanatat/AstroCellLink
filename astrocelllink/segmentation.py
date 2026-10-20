"""Conservative, non-deep-learning soma segmentation baseline.

IMPORTANT: calcium activity maps are not sufficient to infer cell identity reliably.
Supply structural fluorescence images or expert/manual instance masks.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def segment_frame(frame, min_area=30, blur=1.2, min_distance=8, threshold_scale=1.0):
    frame = np.asarray(frame, dtype=float)
    if frame.ndim != 2 or not np.all(np.isfinite(frame)):
        raise ValueError("Structural frame must be a finite 2D image")
    smooth = gaussian(frame, sigma=blur, preserve_range=True)
    if smooth.max() <= smooth.min():
        return np.zeros(frame.shape, np.int32)
    threshold = threshold_otsu(smooth) * threshold_scale
    foreground = smooth > threshold
    regions,_ = ndi.label(foreground)
    sizes = np.bincount(regions.ravel())
    foreground = (regions > 0) & (sizes[regions] >= min_area)
    if not foreground.any():
        return np.zeros(frame.shape, np.int32)
    distance = ndi.distance_transform_edt(foreground)
    maxima = peak_local_max(distance, min_distance=min_distance, labels=foreground, exclude_border=False)
    markers = np.zeros(frame.shape, np.int32)
    for i, (y, x) in enumerate(maxima, start=1):
        markers[y, x] = i
    if not maxima.size:
        markers, _ = ndi.label(foreground)
    separated = watershed(-distance, markers, mask=foreground)
    sizes = np.bincount(separated.ravel())
    separated = np.where((separated > 0) & (sizes[separated] >= min_area), separated, 0)
    out = np.zeros(separated.shape, np.int32)
    for i, val in enumerate(np.unique(separated[separated > 0]), start=1):
        out[separated == val] = i
    return out


def segment_structural_stack(images, **kwargs):
    """Segment 2D morphology or 3D (T,Y,X) stack; frame labels are NOT stable IDs."""
    im = np.asarray(images)
    if im.ndim == 2:
        return segment_frame(im, **kwargs)
    if im.ndim == 3:
        return np.stack([segment_frame(f, **kwargs) for f in im])
    raise ValueError("Structural images must have shape (Y,X) or (T,Y,X)")
