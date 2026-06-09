from pathlib import Path
import re

import numpy as np

_AXIAL_SLICE_RE = re.compile(r"^axial_(\d+)\.(?:npz|npy)$")


def context_channel_count(slice_context_radius, use_mask_conditioning=False):
    slice_channels = 2 * int(slice_context_radius) + 1
    return slice_channels * (2 if use_mask_conditioning else 1)


def clamped_context_indices(center_idx, total_slices, slice_context_radius):
    radius = int(slice_context_radius)
    if radius < 0:
        raise ValueError("--slice-context-radius must be non-negative.")
    if total_slices < 1:
        raise ValueError("Cannot build context from an empty slice volume.")
    return [min(max(center_idx + offset, 0), total_slices - 1) for offset in range(-radius, radius + 1)]


def build_context_stack(volume_xyz, center_idx, slice_context_radius):
    volume_xyz = np.asarray(volume_xyz, dtype=np.float32)
    if volume_xyz.ndim != 3:
        raise ValueError(f"Expected volume shape [H, W, Z], got {volume_xyz.shape}.")
    indices = clamped_context_indices(center_idx, volume_xyz.shape[2], slice_context_radius)
    return np.stack([volume_xyz[:, :, idx] for idx in indices], axis=0).astype(np.float32)


def build_conditioning_channels(volume_xyz, center_idx, slice_context_radius, mask_volume_xyz=None):
    slice_stack = build_context_stack(volume_xyz, center_idx, slice_context_radius)
    if mask_volume_xyz is None:
        return slice_stack
    mask_stack = build_context_stack(mask_volume_xyz, center_idx, slice_context_radius)
    return np.concatenate([slice_stack, mask_stack], axis=0).astype(np.float32)


def axial_slice_index(path_or_name, fallback=None):
    name = Path(path_or_name).name
    match = _AXIAL_SLICE_RE.match(name)
    if match is None:
        if fallback is None:
            raise ValueError(f"Cannot parse axial slice index from {name!r}.")
        return fallback
    return int(match.group(1))


def build_context_stack_from_indices(volume_xyz, slice_indices):
    volume_xyz = np.asarray(volume_xyz, dtype=np.float32)
    if volume_xyz.ndim != 3:
        raise ValueError(f"Expected volume shape [H, W, Z], got {volume_xyz.shape}.")
    if volume_xyz.shape[2] < 1:
        raise ValueError("Cannot build context from an empty slice volume.")
    indices = [min(max(int(idx), 0), volume_xyz.shape[2] - 1) for idx in slice_indices]
    return np.stack([volume_xyz[:, :, idx] for idx in indices], axis=0).astype(np.float32)


def _read_mha_xyz(path):
    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise ImportError("SimpleITK is required to read .mha/.mhd mask files.") from exc
    image = sitk.ReadImage(str(path))
    volume_zyx = sitk.GetArrayFromImage(image)
    return np.transpose(volume_zyx, (2, 1, 0))


def load_mask_volume(path, target_shape=None, npz_key=None):
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".mha") or suffixes.endswith(".mhd"):
        mask = _read_mha_xyz(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as data:
            key = npz_key if npz_key is not None else data.files[0]
            mask = data[key]
    elif path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        raise ValueError(f"Unsupported mask file type: {path}")

    mask = (mask > 0).astype(np.uint8)
    if target_shape is not None:
        target_shape = tuple(target_shape)
        if len(target_shape) != mask.ndim:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match target volume shape {target_shape}. "
                "Rank mismatch."
            )
        shape_mismatch = any(
            expected is not None and actual != expected
            for actual, expected in zip(mask.shape, target_shape)
        )
    else:
        shape_mismatch = False
    if shape_mismatch:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} does not match target volume shape {target_shape}. "
            "For .mha files this loader converts SimpleITK ZYX arrays to repository XYZ order."
        )
    return mask


def resolve_mask_path(case_dir, explicit_mask_path=None, mask_relpath="mask/ct_file.mha", require_mask=False):
    if explicit_mask_path:
        path = Path(explicit_mask_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing mask file: {path}")
        return path

    path = Path(case_dir) / mask_relpath
    if path.exists():
        return path
    if require_mask:
        raise FileNotFoundError(f"Missing mask file: {path}")
    return None
