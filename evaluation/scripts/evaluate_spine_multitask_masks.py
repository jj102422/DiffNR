#!/usr/bin/env python3
"""Evaluate SliceFixer's internal spine mask against GT and input mask_pred."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info-json", type=Path, default=Path("/root/epfs/DiffNR/info.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/epfs/data"))
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path(
            "/root/epfs/test/"
            "10_perx2ct_pefreq_slicefixer_gtmasktrain_predmasktest_spinedice_ckpt100000"
        ),
    )
    parser.add_argument(
        "--spacing-reference-root",
        type=Path,
        default=Path("/root/epfs/test/gt_mask_new"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/epfs/test/evaluation_spine_dice_ablation/mask_metrics"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--surface-tolerance-mm", type=float, default=2.0)
    return parser.parse_args()


def slice_index(path: Path) -> int:
    stem = path.name.removesuffix(".nii.gz").removesuffix(".npz")
    try:
        return int(stem.rsplit("_", 1)[1])
    except Exception as exc:
        raise ValueError(f"Cannot parse axial slice number from {path}") from exc


def stack_gt_mask(directory: Path) -> np.ndarray:
    paths = sorted(directory.glob("axial_*.npz"), key=slice_index)
    if not paths:
        raise FileNotFoundError(f"No GT mask slices in {directory}")
    indices = [slice_index(path) for path in paths]
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"Non-contiguous GT mask slices in {directory}")
    slices = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if "gt_mask" not in archive:
                raise KeyError(f"Missing gt_mask in {path}")
            value = np.squeeze(np.asarray(archive["gt_mask"]))
        if value.ndim != 2:
            raise ValueError(f"Invalid GT mask slice shape {value.shape} in {path}")
        slices.append(value >= 0.5)
    return np.stack(slices, axis=-1)


def stack_input_mask(directory: Path) -> np.ndarray:
    paths = sorted(directory.glob("axial_*.nii.gz"), key=slice_index)
    if not paths:
        raise FileNotFoundError(f"No input mask_pred slices in {directory}")
    indices = [slice_index(path) for path in paths]
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"Non-contiguous mask_pred slices in {directory}")
    slices = []
    for path in paths:
        value = np.squeeze(np.asarray(nib.load(str(path)).dataobj))
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f"Invalid mask_pred slice {path}: shape={value.shape}")
        slices.append(value >= 0.5)
    return np.stack(slices, axis=-1)


def load_internal_mask(path: Path, threshold: float) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "mask_prob" not in archive:
            raise KeyError(f"Missing mask_prob in {path}; found={archive.files}")
        probability = np.squeeze(np.asarray(archive["mask_prob"]))
    if probability.ndim != 3 or not np.isfinite(probability).all():
        raise ValueError(f"Invalid internal mask probability {path}: shape={probability.shape}")
    return probability >= threshold


def overlap_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    intersection = int(np.count_nonzero(prediction & target))
    pred_count = int(np.count_nonzero(prediction))
    target_count = int(np.count_nonzero(target))
    dice_denominator = pred_count + target_count
    dice = 1.0 if dice_denominator == 0 else 2.0 * intersection / dice_denominator
    precision = 1.0 if pred_count == 0 and target_count == 0 else intersection / max(pred_count, 1)
    recall = 1.0 if target_count == 0 and pred_count == 0 else intersection / max(target_count, 1)
    return {"dice": dice, "precision": precision, "recall": recall}


def surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    tolerance_mm: float,
) -> tuple[float, float]:
    if not prediction.any() and not target.any():
        return 0.0, 1.0
    if not prediction.any() or not target.any():
        return float("inf"), 0.0
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = prediction ^ ndimage.binary_erosion(prediction, structure=structure, border_value=0)
    target_surface = target ^ ndimage.binary_erosion(target, structure=structure, border_value=0)
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing_xyz)
    distance_to_prediction = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_xyz)
    pred_distances = distance_to_target[pred_surface]
    target_distances = distance_to_prediction[target_surface]
    all_distances = np.concatenate((pred_distances, target_distances))
    hd95 = float(np.percentile(all_distances, 95))
    surface_dice = float(
        (
            np.count_nonzero(pred_distances <= tolerance_mm)
            + np.count_nonzero(target_distances <= tolerance_mm)
        )
        / (pred_distances.size + target_distances.size)
    )
    return hd95, surface_dice


def prefixed_metrics(
    prefix: str,
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    tolerance_mm: float,
) -> dict[str, float]:
    metrics = overlap_metrics(prediction, target)
    hd95, surface_dice = surface_metrics(prediction, target, spacing_xyz, tolerance_mm)
    return {
        f"{prefix}_dice": metrics["dice"],
        f"{prefix}_precision": metrics["precision"],
        f"{prefix}_recall": metrics["recall"],
        f"{prefix}_hd95_mm": hd95,
        f"{prefix}_surface_dice_2mm": surface_dice,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")
    cases = json.loads(args.info_json.read_text(encoding="utf-8")).get("test", [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for position, case_id in enumerate(cases, start=1):
        try:
            gt = stack_gt_mask(args.dataset_root / case_id / "mask")
            input_mask = stack_input_mask(args.dataset_root / case_id / "mask_pred")
            internal = load_internal_mask(
                args.prediction_root / case_id / "spine_mask_prob.npz",
                args.threshold,
            )
            if not (gt.shape == input_mask.shape == internal.shape):
                raise ValueError(
                    f"Strict shape mismatch: GT={gt.shape}, input={input_mask.shape}, internal={internal.shape}"
                )
            spacing_image = sitk.ReadImage(
                str(args.spacing_reference_root / case_id / "ct_file.mha")
            )
            spacing_xyz = tuple(float(value) for value in spacing_image.GetSpacing())
            row: dict[str, object] = {
                "case_id": case_id,
                "shape": "x".join(str(value) for value in gt.shape),
                "spacing_xyz_mm": "x".join(f"{value:.6g}" for value in spacing_xyz),
                "gt_voxels": int(gt.sum()),
                "input_voxels": int(input_mask.sum()),
                "internal_voxels": int(internal.sum()),
            }
            row.update(
                prefixed_metrics(
                    "internal_vs_gt", internal, gt, spacing_xyz, args.surface_tolerance_mm
                )
            )
            row.update(
                prefixed_metrics(
                    "input_vs_gt", input_mask, gt, spacing_xyz, args.surface_tolerance_mm
                )
            )
            row["internal_vs_input_dice"] = overlap_metrics(internal, input_mask)["dice"]
            input_error = input_mask != gt
            internal_error = internal != gt
            row["corrected_input_error_voxels"] = int(np.count_nonzero(input_error & ~internal_error))
            row["remaining_input_error_voxels"] = int(np.count_nonzero(input_error & internal_error))
            row["new_error_voxels"] = int(np.count_nonzero(~input_error & internal_error))
            row["net_error_reduction_voxels"] = int(input_error.sum() - internal_error.sum())
            rows.append(row)
            print(f"[{position}/{len(cases)}] ok {case_id}", flush=True)
        except Exception as exc:
            failures.append({"case_id": case_id, "stage": "mask_metrics", "error": str(exc)})
            print(f"[{position}/{len(cases)}] FAILED {case_id}: {exc}", flush=True)

    per_case_fields = list(rows[0]) if rows else ["case_id"]
    write_csv(args.output_dir / "per_case_mask_metrics.csv", rows, per_case_fields)
    write_csv(
        args.output_dir / "failed_cases.csv",
        failures,
        ["case_id", "stage", "error"],
    )
    summary_rows = []
    excluded = {"case_id", "shape", "spacing_xyz_mm"}
    for metric in per_case_fields:
        if metric in excluded:
            continue
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "N": int(finite.size),
            }
        )
    write_csv(
        args.output_dir / "summary_mask_metrics.csv",
        summary_rows,
        ["metric", "mean", "std", "N"],
    )
    if failures or len(rows) != len(cases):
        raise SystemExit(f"Mask evaluation incomplete: success={len(rows)}, failures={len(failures)}")


if __name__ == "__main__":
    main()
