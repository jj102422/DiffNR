#!/usr/bin/env python3
"""Read-only preflight audit for the 09-aligned spine-Dice experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml


SLICE_NUMBER = re.compile(r"axial_(\d+)$")
PE_CONFIG = Path("/root/epfs/PerX2CT-jym/configs/PerX2CT_global_w_zoomin.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/epfs/data"))
    parser.add_argument("--info-json", type=Path, default=Path("/root/epfs/DiffNR/info.json"))
    parser.add_argument(
        "--pe-frequency-checkpoint",
        type=Path,
        default=Path(
            "/root/epfs/PerX2CT-jym/logs/PerX2CT/"
            "freqmask_static6__20260701_070730/checkpoints/last.ckpt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/root/epfs/DiffNR/outputs/"
            "perx2ct_pefreq_static6_slicefixer_2p5d_gtmaskcond_spinehead_dicebce_fromscratch/"
            "preflight"
        ),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    parser.add_argument(
        "--checksum-mode",
        choices=("full", "metadata"),
        default="full",
        help="full hashes file bytes; metadata hashes resolved path, size, and mtime.",
    )
    return parser.parse_args()


def canonical_stem(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".npz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def numbered_files(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.glob("axial_*"):
        match = SLICE_NUMBER.fullmatch(canonical_stem(path))
        if match is None:
            continue
        number = int(match.group(1))
        if number in result:
            raise ValueError(f"Duplicate axial slice {number} in {directory}")
        result[number] = path
    if not result:
        raise FileNotFoundError(f"No axial slices found in {directory}")
    numbers = sorted(result)
    expected = list(range(numbers[0], numbers[-1] + 1))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        raise ValueError(f"Non-contiguous slices in {directory}; missing={missing[:20]}")
    return result


def checksum(path: Path, mode: str) -> str:
    digest = hashlib.sha256()
    if mode == "full":
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    else:
        stat = path.stat()
        digest.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def read_npz(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive:
            raise KeyError(f"Missing key {key!r} in {path}; found={archive.files}")
        value = np.squeeze(np.asarray(archive[key]))
    if value.ndim != 2:
        raise ValueError(f"Expected 2D {key} in {path}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite values in {path}:{key}")
    return value


def read_nifti_slice(path: Path) -> np.ndarray:
    value = np.squeeze(np.asarray(nib.load(str(path)).dataobj))
    if value.ndim != 2:
        raise ValueError(f"Expected 2D mask_pred in {path}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite values in {path}")
    return value


def require_binary(value: np.ndarray, path: Path) -> None:
    unique = np.unique(value)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"Mask is not binary in {path}; unique sample={unique[:20].tolist()}")


def validate_upstream_config(checkpoint: Path) -> dict[str, object]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing PE-frequency checkpoint: {checkpoint}")
    if not PE_CONFIG.is_file():
        raise FileNotFoundError(f"Missing PE-frequency config: {PE_CONFIG}")
    config = yaml.safe_load(PE_CONFIG.read_text(encoding="utf-8"))
    nerf = (
        config["model"]["params"]["metadata"]["encoder_params"]["params"]
        ["main_model_of_encoder"]["params"]["cfg"]["nerf_params"]["cfg"]
    )
    freq_mask = nerf["freq_mask"]
    actual = {
        "multires": nerf.get("multires"),
        "enabled": freq_mask.get("enabled"),
        "num_freq_visible": freq_mask.get("num_freq_visible"),
        "anneal_iters": freq_mask.get("anneal_iters"),
    }
    expected = {
        "multires": 10,
        "enabled": True,
        "num_freq_visible": 6,
        "anneal_iters": 0,
    }
    if actual != expected:
        raise ValueError(f"Expected upstream static-6 settings {expected}, found {actual}")
    return {
        "config": str(PE_CONFIG),
        "checkpoint": str(checkpoint),
        "multires": 10,
        "num_freq_visible": 6,
        "anneal_iters": 0,
    }


def main() -> None:
    args = parse_args()
    split_data = json.loads(args.info_json.read_text(encoding="utf-8"))
    cases: list[tuple[str, str]] = []
    for split in args.splits:
        ids = split_data.get(split)
        if not isinstance(ids, list):
            raise ValueError(f"Missing list split {split!r} in {args.info_json}")
        cases.extend((split, str(case_id)) for case_id in ids)
    duplicated = len(cases) - len({case_id for _, case_id in cases})
    if duplicated:
        raise ValueError(f"The requested splits contain {duplicated} duplicate case ids")

    upstream = validate_upstream_config(args.pe_frequency_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_dir / "input_file_checksums.csv.gz"
    cases_path = args.output_dir / "case_audit.csv"
    failures_path = args.output_dir / "failed_cases.csv"
    summary_path = args.output_dir / "preflight_summary.json"
    fieldnames = ["split", "case_id", "kind", "slice", "path", "bytes", "sha256"]
    case_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    with gzip.open(inventory_path, "wt", encoding="utf-8", newline="") as compressed:
        writer = csv.DictWriter(compressed, fieldnames=fieldnames)
        writer.writeheader()
        for case_index, (split, case_id) in enumerate(cases, start=1):
            try:
                case_dir = args.dataset_root / case_id
                groups = {
                    "pred": numbered_files(case_dir / "pred"),
                    "gt": numbered_files(case_dir / "gt"),
                    "mask": numbered_files(case_dir / "mask"),
                    "mask_pred": numbered_files(case_dir / "mask_pred"),
                }
                indices = set(groups["pred"])
                for kind, files in groups.items():
                    if set(files) != indices:
                        missing = sorted(indices - set(files))
                        extra = sorted(set(files) - indices)
                        raise ValueError(
                            f"{kind} index mismatch; missing={missing[:10]}, extra={extra[:10]}"
                        )

                shape = None
                mask_voxels = 0
                pred_mask_voxels = 0
                for index in sorted(indices):
                    arrays = {
                        "pred": read_npz(groups["pred"][index], "vol_pred"),
                        "gt": read_npz(groups["gt"][index], "ct"),
                        "mask": read_npz(groups["mask"][index], "gt_mask"),
                        "mask_pred": read_nifti_slice(groups["mask_pred"][index]),
                    }
                    shapes = {kind: value.shape for kind, value in arrays.items()}
                    if len(set(shapes.values())) != 1:
                        raise ValueError(f"Slice {index} shape mismatch: {shapes}")
                    if shape is None:
                        shape = next(iter(shapes.values()))
                    elif shape != next(iter(shapes.values())):
                        raise ValueError(f"Inconsistent in-plane shape at slice {index}: {shapes}")
                    require_binary(arrays["mask"], groups["mask"][index])
                    require_binary(arrays["mask_pred"], groups["mask_pred"][index])
                    mask_voxels += int(np.count_nonzero(arrays["mask"] >= 0.5))
                    pred_mask_voxels += int(np.count_nonzero(arrays["mask_pred"] >= 0.5))
                    for kind, files in groups.items():
                        path = files[index]
                        writer.writerow(
                            {
                                "split": split,
                                "case_id": case_id,
                                "kind": kind,
                                "slice": index,
                                "path": str(path),
                                "bytes": path.stat().st_size,
                                "sha256": checksum(path, args.checksum_mode),
                            }
                        )

                for view in (1, 2):
                    xray_path = case_dir / f"{case_id}_xray_{view}.pt"
                    feature = torch.load(xray_path, map_location="cpu", weights_only=True)
                    if isinstance(feature, dict):
                        tensor_values = [value for value in feature.values() if torch.is_tensor(value)]
                        if len(tensor_values) != 1:
                            raise ValueError(f"Cannot identify one tensor in {xray_path}")
                        feature = tensor_values[0]
                    if tuple(feature.shape) != (1, 768) or not torch.isfinite(feature).all():
                        raise ValueError(f"Invalid RAD-DINO feature {xray_path}: shape={feature.shape}")
                    writer.writerow(
                        {
                            "split": split,
                            "case_id": case_id,
                            "kind": f"xray_{view}",
                            "slice": "",
                            "path": str(xray_path),
                            "bytes": xray_path.stat().st_size,
                            "sha256": checksum(xray_path, args.checksum_mode),
                        }
                    )
                if mask_voxels == 0:
                    raise ValueError("GT spine mask is empty over the complete volume")
                case_rows.append(
                    {
                        "split": split,
                        "case_id": case_id,
                        "slices": len(indices),
                        "height": shape[0],
                        "width": shape[1],
                        "gt_mask_voxels": mask_voxels,
                        "pred_mask_voxels": pred_mask_voxels,
                        "status": "ok",
                    }
                )
                print(f"[{case_index}/{len(cases)}] ok {split} {case_id} slices={len(indices)}", flush=True)
            except Exception as exc:
                failures.append(
                    {"split": split, "case_id": case_id, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"[{case_index}/{len(cases)}] FAILED {split} {case_id}: {exc}", flush=True)

    with cases_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "split", "case_id", "slices", "height", "width",
            "gt_mask_voxels", "pred_mask_voxels", "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(case_rows)
    with failures_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "case_id", "error"])
        writer.writeheader()
        writer.writerows(failures)
    summary = {
        "status": "ok" if not failures and len(case_rows) == len(cases) else "failed",
        "requested_cases": len(cases),
        "validated_cases": len(case_rows),
        "failed_cases": len(failures),
        "split_counts": {split: len(split_data[split]) for split in args.splits},
        "checksum_mode": args.checksum_mode,
        "inventory": str(inventory_path),
        "upstream_pe_frequency": upstream,
        "slice_context_radius": 2,
        "mask_context_radius": 2,
        "conditioning_channels": 10,
        "train_mask_condition": "mask/gt_mask",
        "deployment_mask_condition": "mask_pred",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
