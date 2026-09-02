from __future__ import annotations

from glob import glob
from pathlib import Path
import re

import numpy as np

from .volume_io import load_volume


def as_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.bool_:
        return mask
    return np.asarray(mask) > 0


def load_mask(path: str | Path, file_type: str | None = None, key: str | None = None) -> np.ndarray:
    return as_bool_mask(load_volume(path, file_type=file_type, key=key))


def load_mask_slice_stack(
    path_pattern: str | Path,
    key: str,
    threshold: float = 0.5,
) -> np.ndarray:
    """Load numbered 2D NPZ masks as one boolean ``[Z, Y, X]`` volume.

    The spine masks are stored as ``axial_000.npz``, ``axial_001.npz``, ... .
    Failing on missing/duplicated indices is intentional: silently stacking an
    incomplete mask would shift the ROI relative to the GT volume.
    """
    pattern = str(path_pattern)
    paths = [Path(path) for path in glob(pattern)]
    if not paths:
        raise FileNotFoundError(pattern)

    indexed_paths = sorted((_slice_index(path), path) for path in paths)
    indices = [index for index, _ in indexed_paths]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate mask slice indices for pattern: {pattern}")
    expected = list(range(indices[0], indices[-1] + 1))
    if indices[0] != 0 or indices != expected:
        missing = sorted(set(range(0, indices[-1] + 1)) - set(indices))
        raise ValueError(
            f"Mask slice indices must be contiguous from 0 for {pattern}; "
            f"first={indices[0]}, last={indices[-1]}, missing={missing[:10]}"
        )

    slices: list[np.ndarray] = []
    slice_shape: tuple[int, int] | None = None
    for _, path in indexed_paths:
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"Key '{key}' not found in {path}. Available keys: {data.files}")
            arr = np.asarray(data[key])
        if arr.ndim != 2:
            raise ValueError(f"Mask slice must be 2D: {path} has shape {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Mask slice contains NaN/Inf: {path}")
        if np.any((arr != 0) & (arr != 1)):
            values = np.unique(arr)
            raise ValueError(f"Mask slice is not binary: {path}, values={values[:10].tolist()}")
        if slice_shape is None:
            slice_shape = tuple(int(v) for v in arr.shape)
        elif tuple(arr.shape) != slice_shape:
            raise ValueError(
                f"Inconsistent mask slice shape: {path} has {arr.shape}, expected {slice_shape}"
            )
        slices.append(np.asarray(arr > float(threshold), dtype=np.bool_))

    mask = np.stack(slices, axis=0)
    if not mask.any():
        raise ValueError(f"Empty mask volume loaded from {pattern}")
    return mask


def _slice_index(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse numeric slice index from {path.name}")
    return int(match.group(1))


def resize_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(target_shape):
        return as_bool_mask(mask)
    from scipy.ndimage import zoom

    factors = [t / s for t, s in zip(target_shape, mask.shape)]
    resized = zoom(mask.astype(np.uint8), zoom=factors, order=0)
    return as_bool_mask(resized)


def get_2d_bbox(mask2d: np.ndarray, padding: int = 8) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask2d)
    if len(ys) == 0:
        return None
    y1 = max(int(ys.min()) - padding, 0)
    y2 = min(int(ys.max()) + padding + 1, mask2d.shape[0])
    x1 = max(int(xs.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding + 1, mask2d.shape[1])
    return y1, y2, x1, x2
