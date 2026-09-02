#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a SliceFixer volume archive from completed distributed shards."
    )
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--output_name", default="vol_pred_slicefixer.npz")
    parser.add_argument("--key", default="vol_pred")
    parser.add_argument("--backup_suffix", default=".corrupt")
    return parser.parse_args()


def rebuild_from_shards(case_dir: Path, output_name: str, key: str, backup_suffix: str) -> Path:
    shard_paths = sorted((case_dir / "shards").glob("shard_rank*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_rank*.npz files under {case_dir / 'shards'}")

    reconstructed: dict[int, np.ndarray] = {}
    slice_shape: tuple[int, int] | None = None
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            if "indices" not in shard.files or "slices" not in shard.files:
                raise KeyError(f"Missing indices/slices in {shard_path}")
            indices = np.asarray(shard["indices"], dtype=np.int64)
            slices = np.asarray(shard["slices"], dtype=np.float32)
        if slices.ndim != 3 or len(indices) != len(slices):
            raise ValueError(
                f"Invalid shard shapes in {shard_path}: indices={indices.shape}, slices={slices.shape}"
            )
        for index, slice_array in zip(indices.tolist(), slices):
            if index in reconstructed:
                raise ValueError(f"Duplicate reconstructed slice index {index}")
            if not np.isfinite(slice_array).all():
                raise ValueError(f"NaN/Inf in {shard_path} slice {index}")
            if slice_shape is None:
                slice_shape = tuple(int(v) for v in slice_array.shape)
            elif tuple(slice_array.shape) != slice_shape:
                raise ValueError(
                    f"Inconsistent slice shape at index {index}: {slice_array.shape} != {slice_shape}"
                )
            reconstructed[int(index)] = slice_array

    ordered_indices = sorted(reconstructed)
    if not ordered_indices:
        raise ValueError("No reconstructed slices found")
    expected = list(range(ordered_indices[-1] + 1))
    if ordered_indices != expected:
        missing = sorted(set(expected) - set(ordered_indices))
        raise ValueError(f"Missing reconstructed slices: {missing[:10]}")

    volume = np.stack([reconstructed[index] for index in expected], axis=-1).astype(np.float32)
    output_path = case_dir / output_name
    temp_path = case_dir / f".{output_name}.rebuild.tmp.npz"
    if temp_path.exists():
        temp_path.unlink()
    np.savez_compressed(temp_path, **{key: volume})
    with np.load(temp_path, allow_pickle=False) as check:
        if key not in check.files:
            raise KeyError(f"Rebuilt archive is missing key '{key}'")
        restored = np.asarray(check[key])
        if restored.shape != volume.shape or restored.dtype != np.float32:
            raise ValueError(
                f"Rebuilt archive validation failed: shape={restored.shape}, dtype={restored.dtype}"
            )
        if not np.isfinite(restored).all():
            raise ValueError("Rebuilt archive contains NaN/Inf")

    if output_path.exists():
        backup_path = output_path.with_name(output_path.name + backup_suffix)
        if backup_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {backup_path}")
        output_path.replace(backup_path)
        print(f"Preserved original archive: {backup_path}", flush=True)
    os.replace(temp_path, output_path)
    print(
        f"Rebuilt {output_path}: shape={volume.shape}, dtype={volume.dtype}, "
        f"shards={len(shard_paths)}",
        flush=True,
    )
    return output_path


def main() -> None:
    args = parse_args()
    rebuild_from_shards(
        Path(args.case_dir),
        output_name=args.output_name,
        key=args.key,
        backup_suffix=args.backup_suffix,
    )


if __name__ == "__main__":
    main()
