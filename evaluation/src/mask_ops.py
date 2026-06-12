from __future__ import annotations

from pathlib import Path

import numpy as np

from .volume_io import load_volume


def as_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.bool_:
        return mask
    return np.asarray(mask) > 0


def load_mask(path: str | Path, file_type: str | None = None, key: str | None = None) -> np.ndarray:
    return as_bool_mask(load_volume(path, file_type=file_type, key=key))


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

