#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.src.align import (  # noqa: E402
    apply_axis_order_to_zyx,
    apply_rot90,
    prepare_case_for_metric,
    reshape_with_placeholders,
    resolve_gt_path,
    resolve_mask_path,
    resolve_pred_path,
    restore_intensity,
    squeeze_volume,
)
from evaluation.src.config_io import load_yaml, resolve_model_name  # noqa: E402
from evaluation.src.mask_ops import as_bool_mask, resize_mask_to_shape  # noqa: E402
from evaluation.src.visualization import choose_mask_slices, save_visual_check  # noqa: E402
from evaluation.src.volume_io import load_volume  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Diet alignment candidates for one model/case.")
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--model_diet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--output_dir", default="/root/epfs/test/evaluation_alignment_candidates")
    parser.add_argument("--axis_mode", choices=["auto", "diet", "permutations"], default="auto")
    parser.add_argument("--z_step", type=int, default=10)
    parser.add_argument("--num_slices", type=int, default=9)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--min_mask_pixels", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_cfg = load_yaml(args.eval_config)
    model_diet_all = load_yaml(args.model_diet)
    model_name = resolve_model_name(args.model, model_diet_all)
    model_cfg = model_diet_all["models"][model_name]

    gt, mask = load_gt_and_mask(args.case_id, global_cfg)
    pred_raw = load_pred_raw(args.case_id, model_cfg, gt.shape)
    ct_min = float(global_cfg["canonical"]["ct_min"])
    ct_max = float(global_cfg["canonical"]["ct_max"])
    gt = np.clip(gt.astype(np.float32), ct_min, ct_max)
    mask = resize_mask_to_shape(mask, gt.shape)
    slices = choose_mask_slices(mask, num_slices=args.num_slices)
    if not slices:
        raise ValueError(f"Empty mask for {args.case_id}")

    rows: list[dict[str, Any]] = []
    for axis_candidate in axis_candidates(pred_raw, model_cfg["pred"], args.axis_mode, model_name):
        axis_vol = apply_axis_candidate(pred_raw, model_cfg["pred"], axis_candidate)
        for k, flip_axes in itertools.product(range(4), flip_candidates()):
            vol = apply_rot90(axis_vol, {"k": k, "axes": ["Y", "X"]})
            for axis_name in flip_axes:
                axis = {"Z": 0, "Y": 1, "X": 2}[axis_name]
                vol = np.flip(vol, axis=axis)

            z_candidates = build_z_candidates(vol.shape[0], gt.shape[0], model_cfg.get("align_to_gt", {}), args.z_step)
            for z_candidate in z_candidates:
                score = score_candidate(
                    gt=gt,
                    pred_axis=vol,
                    mask=mask,
                    slices=slices,
                    pred_cfg=model_cfg["pred"],
                    align_cfg=model_cfg.get("align_to_gt", {}),
                    z_candidate=z_candidate,
                    ct_min=ct_min,
                    ct_max=ct_max,
                    min_mask_pixels=args.min_mask_pixels,
                )
                rows.append(
                    {
                        "model": model_name,
                        "case_id": args.case_id,
                        **axis_candidate,
                        "rot90_k": k,
                        "flip_axes": "+".join(flip_axes),
                        **z_candidate,
                        **score,
                    }
                )

    rows.sort(key=lambda row: row["score"])
    out_dir = Path(args.output_dir) / model_name / args.case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "candidate_scores.csv"
    write_scores(scores_path, rows)

    for rank, row in enumerate(rows[: args.top_k], start=1):
        candidate_cfg = build_candidate_model_cfg(model_cfg, row)
        gt_full, pred_full, mask_full, _ = prepare_case_for_metric(args.case_id, model_name, global_cfg, candidate_cfg)
        out_path = out_dir / (
            f"rank{rank:02d}_score{row['score']:.5f}_mae{row['mae_norm']:.5f}_ncc{row['ncc']:.3f}_"
            f"axis{row['axis_id']}_rot{row['rot90_k']}_flip{row['flip_axes'] or 'none'}_"
            f"z{row['z_mode']}{row['z_offset'] if row['z_offset'] != '' else ''}.png"
        )
        save_visual_check(
            gt_full,
            pred_full,
            mask_full,
            out_path,
            num_slices=int(global_cfg.get("runtime", {}).get("visual_check_num_slices", 5)),
            ct_min=ct_min,
            ct_max=ct_max,
        )

    best = rows[0]
    print(f"[ok] wrote {scores_path}")
    print(
        "best: "
        f"axis={best['axis_id']} transpose={best['transpose']} raw_axis_order={best['raw_axis_order']} "
        f"rot90={best['rot90_k']} flip={best['flip_axes'] or 'none'} "
        f"z_mode={best['z_mode']} z_offset={best['z_offset']} "
        f"score={best['score']:.6f} mae_norm={best['mae_norm']:.6f} ncc={best['ncc']:.4f}"
    )


def load_gt_and_mask(case_id: str, global_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    gt_path = resolve_gt_path(case_id, global_cfg)
    gt = load_volume(
        gt_path,
        file_type=global_cfg["dataset"].get("eval_gt_type"),
        key=global_cfg["dataset"].get("eval_gt_h5_key"),
    )
    gt = apply_axis_order_to_zyx(gt, global_cfg["dataset"].get("eval_gt_axis_order", "ZYX"))

    mask_path = resolve_mask_path(case_id, global_cfg)
    mask = load_volume(
        mask_path,
        file_type=global_cfg["dataset"].get("eval_mask_type"),
        key=global_cfg["dataset"].get("eval_mask_key"),
    )
    mask = apply_axis_order_to_zyx(mask, global_cfg["dataset"].get("eval_mask_axis_order", "ZYX"))
    return gt.astype(np.float32, copy=False), as_bool_mask(mask)


def load_pred_raw(case_id: str, model_cfg: dict[str, Any], gt_shape: tuple[int, int, int]) -> np.ndarray:
    pred_cfg = model_cfg["pred"]
    pred_path = resolve_pred_path(case_id, model_cfg)
    pred = load_volume(pred_path, file_type=pred_cfg.get("file_type"), key=pred_cfg.get("key"))
    if pred_cfg.get("squeeze", True):
        pred = squeeze_volume(pred, remove_channel_dim=pred_cfg.get("remove_channel_dim", True))
    if pred_cfg.get("reshape_to") is not None:
        pred = reshape_with_placeholders(pred, pred_cfg["reshape_to"], gt_shape)
    if pred.ndim != 3:
        raise ValueError(f"Expected a 3D prediction after squeeze/reshape, got {pred.shape}")
    return pred


def axis_candidates(pred: np.ndarray, pred_cfg: dict[str, Any], axis_mode: str, model_name: str) -> list[dict[str, Any]]:
    use_permutations = axis_mode == "permutations"
    if axis_mode == "auto":
        use_permutations = model_name.lower() == "x2ct" or len(set(pred.shape)) == 1
    if use_permutations:
        return [
            {"axis_id": "".join(map(str, perm)), "transpose": list(perm), "raw_axis_order": ""}
            for perm in itertools.permutations([0, 1, 2])
        ]
    transpose = pred_cfg.get("transpose")
    raw_axis_order = pred_cfg.get("raw_axis_order", "ZYX") if transpose is None else ""
    return [{"axis_id": "diet", "transpose": transpose if transpose is not None else "", "raw_axis_order": raw_axis_order}]


def apply_axis_candidate(pred: np.ndarray, pred_cfg: dict[str, Any], candidate: dict[str, Any]) -> np.ndarray:
    if candidate["transpose"] != "":
        return np.transpose(pred, tuple(candidate["transpose"]))
    return apply_axis_order_to_zyx(pred, candidate["raw_axis_order"] or pred_cfg.get("raw_axis_order", "ZYX"))


def flip_candidates() -> list[tuple[str, ...]]:
    axes = ["X", "Y", "Z"]
    out: list[tuple[str, ...]] = []
    for count in range(len(axes) + 1):
        out.extend(tuple(combo) for combo in itertools.combinations(axes, count))
    return out


def build_z_candidates(pred_z: int, gt_z: int, align_cfg: dict[str, Any], z_step: int) -> list[dict[str, Any]]:
    strategy = align_cfg.get("strategy", "assert_same_shape")
    if strategy == "resize_to_gt_shape":
        return [{"z_mode": "resize", "z_offset": ""}]
    if pred_z == gt_z:
        return [{"z_mode": "center", "z_offset": ""}]
    total = abs(gt_z - pred_z)
    offsets = {0, total, total // 2}
    step = max(1, int(z_step))
    offsets.update(range(0, total + 1, step))
    return [{"z_mode": "offset", "z_offset": offset} for offset in sorted(offsets)]


def score_candidate(
    gt: np.ndarray,
    pred_axis: np.ndarray,
    mask: np.ndarray,
    slices: list[int],
    pred_cfg: dict[str, Any],
    align_cfg: dict[str, Any],
    z_candidate: dict[str, Any],
    ct_min: float,
    ct_max: float,
    min_mask_pixels: int,
) -> dict[str, Any]:
    mae_vals: list[float] = []
    ncc_vals: list[float] = []
    valid_pixels = 0
    for z in slices:
        mask_s = mask[z]
        pixels = int(mask_s.sum())
        if pixels < min_mask_pixels:
            continue
        pred_s = extract_aligned_slice(pred_axis, z, gt.shape, align_cfg, z_candidate)
        pred_s = restore_intensity(pred_s.astype(np.float32, copy=False), pred_cfg)
        pred_s = np.clip(pred_s, ct_min, ct_max)
        gt_s = gt[z]
        diff = np.abs(pred_s[mask_s] - gt_s[mask_s])
        mae_vals.append(float(np.mean(diff) / max(ct_max - ct_min, 1.0)))
        ncc_vals.append(masked_ncc(gt_s, pred_s, mask_s))
        valid_pixels += pixels
    if not mae_vals:
        return {"score": math.inf, "mae_norm": math.inf, "ncc": float("nan"), "valid_slices": 0, "valid_pixels": 0}
    ncc = float(np.nanmean(ncc_vals)) if np.isfinite(ncc_vals).any() else 0.0
    mae_norm = float(np.mean(mae_vals))
    return {
        "score": mae_norm - 0.1 * ncc,
        "mae_norm": mae_norm,
        "ncc": ncc,
        "valid_slices": len(mae_vals),
        "valid_pixels": valid_pixels,
    }


def extract_aligned_slice(
    pred: np.ndarray,
    z: int,
    gt_shape: tuple[int, int, int],
    align_cfg: dict[str, Any],
    z_candidate: dict[str, Any],
) -> np.ndarray:
    strategy = align_cfg.get("strategy", "assert_same_shape")
    if strategy == "resize_to_gt_shape":
        src_z = int(round(z * (pred.shape[0] - 1) / max(gt_shape[0] - 1, 1)))
        pred_s = pred[src_z]
        return resize_2d(pred_s, gt_shape[1:])

    if pred.shape[0] < gt_shape[0]:
        offset = int(z_candidate["z_offset"] or 0)
        src_z = z - offset
        if src_z < 0 or src_z >= pred.shape[0]:
            return np.zeros(gt_shape[1:], dtype=np.float32)
    elif pred.shape[0] > gt_shape[0]:
        offset = int(z_candidate["z_offset"] or 0)
        src_z = z + offset
    else:
        src_z = z
    pred_s = pred[src_z]
    return resize_2d(pred_s, gt_shape[1:])


def resize_2d(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if tuple(img.shape) == tuple(target_shape):
        return img.astype(np.float32, copy=False)
    from scipy.ndimage import zoom

    factors = [target_shape[0] / img.shape[0], target_shape[1] / img.shape[1]]
    return zoom(img.astype(np.float32), zoom=factors, order=1).astype(np.float32)


def masked_ncc(gt_s: np.ndarray, pred_s: np.ndarray, mask_s: np.ndarray) -> float:
    g = gt_s[mask_s].astype(np.float32)
    p = pred_s[mask_s].astype(np.float32)
    g = g - float(g.mean())
    p = p - float(p.mean())
    denom = float(np.sqrt(np.sum(g * g) * np.sum(p * p)))
    if denom <= 1.0e-6:
        return float("nan")
    return float(np.sum(g * p) / denom)


def build_candidate_model_cfg(model_cfg: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(model_cfg)
    pred_cfg = cfg["pred"]
    if row["transpose"] != "":
        pred_cfg["transpose"] = parse_transpose(row["transpose"])
        pred_cfg["raw_axis_order"] = "ZYX"
    else:
        pred_cfg["transpose"] = None
        pred_cfg["raw_axis_order"] = row["raw_axis_order"] or pred_cfg.get("raw_axis_order", "ZYX")
    pred_cfg["rot90"] = {"k": int(row["rot90_k"]), "axes": ["Y", "X"]} if int(row["rot90_k"]) % 4 else None
    pred_cfg["flip_axes"] = row["flip_axes"].split("+") if row["flip_axes"] else []

    align_cfg = cfg.setdefault("align_to_gt", {})
    if row["z_mode"] == "resize":
        align_cfg["strategy"] = "resize_to_gt_shape"
        align_cfg.pop("z_mode", None)
        align_cfg.pop("z_offset", None)
    elif row["z_mode"] == "offset":
        align_cfg["strategy"] = "center_crop_or_pad_z"
        align_cfg["z_mode"] = "offset"
        align_cfg["z_offset"] = int(row["z_offset"])
    return cfg


def parse_transpose(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, tuple):
        return [int(v) for v in value]
    text = str(value).strip()
    if text.startswith("["):
        return [int(v.strip()) for v in text.strip("[]").split(",") if v.strip()]
    return [int(ch) for ch in text if ch.isdigit()]


def write_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "case_id",
        "axis_id",
        "transpose",
        "raw_axis_order",
        "rot90_k",
        "flip_axes",
        "z_mode",
        "z_offset",
        "score",
        "mae_norm",
        "ncc",
        "valid_slices",
        "valid_pixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
