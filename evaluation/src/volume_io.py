from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def infer_file_type(path: str | Path, file_type: str | None = None) -> str:
    if file_type:
        return file_type.lower().lstrip(".")
    text = str(path).lower()
    if text.endswith(".nii.gz"):
        return "nii.gz"
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix


def load_volume(
    path: str | Path,
    file_type: str | None = None,
    key: str | None = None,
    read_backend: str | None = None,
) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    kind = infer_file_type(path, file_type)

    if kind == "npy":
        arr = np.load(path, allow_pickle=False)
    elif kind == "npz":
        arr = _load_npz(path, key)
    elif kind in {"h5", "hdf5"}:
        arr = _load_h5(path, key)
    elif kind in {"nii", "nii.gz"}:
        arr = _load_nifti(path, read_backend=read_backend)
    elif kind in {"mha", "mhd"}:
        arr = _load_sitk(path)
    elif kind in {"pt", "pth"}:
        arr = _load_torch(path, key)
    else:
        raise ValueError(f"Unsupported file_type '{kind}' for {path}")

    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        return arr
    arr = arr.astype(np.float32, copy=False)
    return sanitize_finite(arr)


def sanitize_finite(arr: np.ndarray, min_value: float = 0.0, max_value: float | None = None) -> np.ndarray:
    if np.isfinite(arr).all():
        return arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if max_value is None:
        max_value = float(finite.max()) if finite.size else 0.0
    return np.nan_to_num(arr, nan=0.0, posinf=max_value, neginf=min_value).astype(np.float32)


def _load_npz(path: Path, key: str | None) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())
    if key:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {keys}")
        return data[key]
    if len(keys) == 1:
        return data[keys[0]]
    preferred = ["vol_pred", "volume", "ct", "arr_0"]
    for candidate in preferred:
        if candidate in data:
            return data[candidate]
    raise ValueError(f"NPZ file {path} has multiple keys; configure key explicitly. Available keys: {keys}")


def _load_h5(path: Path, key: str | None) -> np.ndarray:
    if not key:
        raise ValueError(f"HDF5 file {path} requires an explicit key")
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read HDF5 volumes. Install h5py.") from exc

    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(f.keys())}")
        return f[key][()]


def _load_nifti(path: Path, read_backend: str | None = None) -> np.ndarray:
    backend = (read_backend or "SimpleITK").lower()
    if backend in {"nibabel", "nib"}:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise RuntimeError("nibabel is required to read this NIfTI file.") from exc
        return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    return _load_sitk(path)


def _load_sitk(path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required to read this medical image format.") from exc
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


def _load_torch(path: Path, key: str | None) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to read .pt/.pth volumes.") from exc
    obj: Any = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if key:
            if key not in obj:
                raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(obj.keys())}")
            obj = obj[key]
        elif len(obj) == 1:
            obj = next(iter(obj.values()))
        else:
            raise ValueError(f"Torch file {path} has multiple keys; configure key explicitly.")
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().numpy()
    return np.asarray(obj)
