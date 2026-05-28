import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np


def _find_volume_paths(case_path):
    case_path = Path(case_path)
    coarse = next(
        (p for p in (case_path / "vol_pred.npz", case_path / "vol_pred.npy") if p.exists()),
        None,
    )
    target = next(
        (
            p
            for p in (
                case_path / "volume_gt.npz",
                case_path / "volume_gt.npy",
                case_path / "vol_gt.npz",
                case_path / "vol_gt.npy",
            )
            if p.exists()
        ),
        None,
    )
    return coarse, target


def _npz_array_info(path):
    path = Path(path)
    with np.load(path) as archive:
        key = path.stem if path.stem in archive.files else archive.files[0]
    with zipfile.ZipFile(path) as archive, archive.open(f"{key}.npy") as handle:
        version = np.lib.format.read_magic(handle)
        shape, _, dtype = np.lib.format._read_array_header(
            handle, version, max_header_size=10000
        )
    return key, tuple(shape), np.dtype(dtype)


class TemporaryVolumeCache:
    """Materialize compressed case volumes as temporary mmap-friendly .npy files."""

    def __init__(self, cache_root, run_id):
        self.run_dir = Path(cache_root) / run_id

    def _cached_case_dir(self, block_name, case_path):
        return self.run_dir / block_name / Path(case_path).name

    def _required_bytes(self, case_paths):
        required = 0
        for case_path in case_paths:
            for source_path in _find_volume_paths(case_path):
                if source_path is not None and source_path.suffix == ".npz":
                    _, shape, dtype = _npz_array_info(source_path)
                    required += int(np.prod(shape)) * dtype.itemsize
        return required

    def materialize_block(self, block_name, case_paths):
        started = time.perf_counter()
        block_dir = self.run_dir / block_name
        shutil.rmtree(block_dir, ignore_errors=True)
        block_dir.mkdir(parents=True, exist_ok=True)

        required = self._required_bytes(case_paths)
        free = shutil.disk_usage(self.run_dir).free
        if required > free:
            raise RuntimeError(
                f"Volume cache needs {required / (1024 ** 3):.2f} GiB but only "
                f"{free / (1024 ** 3):.2f} GiB is free in {self.run_dir}. "
                "Reduce --volume_cache_cases_per_block or choose another --volume_cache_dir."
            )

        cached_bytes = 0
        for case_path in case_paths:
            coarse_path, gt_path = _find_volume_paths(case_path)
            if coarse_path is None or gt_path is None:
                continue
            case_dir = self._cached_case_dir(block_name, case_path)
            case_dir.mkdir(parents=True, exist_ok=True)
            for source_path, output_name in (
                (coarse_path, "vol_pred.npy"),
                (gt_path, "volume_gt.npy"),
            ):
                if source_path.suffix != ".npz":
                    continue
                key, _, _ = _npz_array_info(source_path)
                output_path = case_dir / output_name
                temporary_path = output_path.with_suffix(".npy.tmp")
                with np.load(source_path) as archive, open(temporary_path, "wb") as handle:
                    np.save(handle, archive[key], allow_pickle=False)
                os.replace(temporary_path, output_path)
                cached_bytes += output_path.stat().st_size

        ready_path = block_dir / ".ready"
        ready_path.write_text("ready\n", encoding="ascii")
        return {
            "seconds": time.perf_counter() - started,
            "bytes": cached_bytes,
            "cases": len(case_paths),
        }

    def path_overrides(self, block_name, case_paths):
        block_dir = self.run_dir / block_name
        if not (block_dir / ".ready").exists():
            raise RuntimeError(f"Volume cache block is not ready: {block_dir}")
        overrides = {}
        for case_path in case_paths:
            coarse_path, gt_path = _find_volume_paths(case_path)
            if coarse_path is None or gt_path is None:
                continue
            case_dir = self._cached_case_dir(block_name, case_path)
            cached_coarse = case_dir / "vol_pred.npy"
            cached_gt = case_dir / "volume_gt.npy"
            overrides[str(case_path)] = {
                "coarse_path": str(cached_coarse if cached_coarse.exists() else coarse_path),
                "gt_path": str(cached_gt if cached_gt.exists() else gt_path),
            }
        return overrides

    def cleanup_block(self, block_name):
        shutil.rmtree(self.run_dir / block_name, ignore_errors=True)

    def cleanup(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)
