"""Read image stacks and astroCaST ROI event maps without importing astroCaST."""
from pathlib import Path
import numpy as np
import tifffile


def read_array(path, dataset=None):
    """Load .npy, TIFF or HDF5 numeric arrays. HDF5 requires --dataset."""
    path = Path(path)
    if path.is_dir():
        # astroCaST normally creates video.roi/event_map.tiff.
        for name in ("event_map.tiff", "event_map.tif", "event_map.npy"):
            if (path / name).is_file():
                path = path / name
                break
        else:
            suffix = " (TileDB output needs conversion to TIFF via astroCaST)" if (path / "event_map.tdb").exists() else ""
            raise FileNotFoundError(f"No event_map.tiff or event_map.npy in {path}{suffix}")
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        if dataset is None:
            raise ValueError("HDF5 files require --dataset /path/to/dataset")
        import h5py
        with h5py.File(path, "r") as f:
            return f[dataset][()]
    raise ValueError(f"Unsupported file format: {path.suffix}")


def labels_3d(arr, name, *, allow_2d=False, n_frames=None):
    """Require nonnegative integer label images; allow a fixed 2D mask for all frames."""
    arr = np.asarray(arr)
    if allow_2d and arr.ndim == 2:
        if n_frames is None:
            raise ValueError("n_frames needed to broadcast static 2D masks")
        arr = np.broadcast_to(arr, (n_frames,) + arr.shape)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be (frames, rows, columns), got {arr.shape}")
    if arr.dtype.kind not in "uib":
        # e.g. astroCaST may write integer labels in float TIFF.
        if not np.all(np.isfinite(arr)) or np.any(arr != np.floor(arr)):
            raise ValueError(f"{name} contains non-integer/nonfinite values")
    if np.any(arr < 0):
        raise ValueError(f"{name} has negative labels")
    return arr.astype(np.int32, copy=False)


def write_tiff(path, array):
    """Write labels with safe integer precision."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array, dtype=np.int32), photometric="minisblack", metadata={"axes": "TYX"})
