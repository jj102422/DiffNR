from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from .lpips_metric import compute_lpips_volume


def normalize_for_metric(volume: np.ndarray, ct_min: float, ct_max: float) -> np.ndarray:
    if ct_max <= ct_min:
        raise ValueError(f"Invalid CT range: ct_min={ct_min}, ct_max={ct_max}")
    volume = np.clip(volume.astype(np.float32), ct_min, ct_max)
    return ((volume - ct_min) / (ct_max - ct_min)).astype(np.float32)


def compute_mask_psnr(gt_norm: np.ndarray, pred_norm: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    diff = pred_norm[mask] - gt_norm[mask]
    mse = float(np.mean(diff**2))
    if mse <= eps:
        return float("inf"), mse
    return float(10.0 * np.log10(1.0 / mse)), mse


def compute_masked_ssim_slice_map(
    gt_norm: np.ndarray,
    pred_norm: np.ndarray,
    mask: np.ndarray,
    data_range: float = 1.0,
    min_mask_pixels_per_slice: int = 100,
    gaussian_weights: bool = False,
) -> tuple[float, int]:
    scores: list[float] = []
    for z in range(gt_norm.shape[0]):
        m = mask[z]
        if int(m.sum()) < int(min_mask_pixels_per_slice):
            continue
        win_size = _ssim_win_size(gt_norm[z].shape)
        if win_size is None:
            continue
        _, ssim_map = structural_similarity(
            gt_norm[z],
            pred_norm[z],
            data_range=data_range,
            full=True,
            gaussian_weights=gaussian_weights,
            win_size=win_size,
        )
        scores.append(float(np.mean(ssim_map[m])))
    if not scores:
        return float("nan"), 0
    return float(np.mean(scores)), len(scores)


def metric(GT: np.ndarray, pred: np.ndarray, mask: np.ndarray, Diet: dict) -> dict:
    if GT.ndim != 3 or pred.ndim != 3 or mask.ndim != 3:
        raise AssertionError("GT, pred, and mask must be 3D arrays")
    if GT.shape != pred.shape or GT.shape != mask.shape:
        raise AssertionError(f"Shape mismatch: GT={GT.shape}, pred={pred.shape}, mask={mask.shape}")
    mask = mask.astype(bool, copy=False)
    if not np.isfinite(GT).all() or not np.isfinite(pred).all():
        raise AssertionError("GT and pred must be finite")
    mask_voxels = int(mask.sum())
    if mask_voxels == 0:
        raise ValueError("Empty mask")

    canonical = Diet["canonical"]
    metric_cfg = Diet["metric"]
    ct_min = float(canonical["ct_min"])
    ct_max = float(canonical["ct_max"])

    GT = np.clip(GT.astype(np.float32), ct_min, ct_max)
    pred = np.clip(pred.astype(np.float32), ct_min, ct_max)
    GT_norm = normalize_for_metric(GT, ct_min, ct_max)
    pred_norm = normalize_for_metric(pred, ct_min, ct_max)

    mae_raw = float(np.mean(np.abs(pred[mask] - GT[mask])))
    mae_norm = float(np.mean(np.abs(pred_norm[mask] - GT_norm[mask])))
    mae_primary = metric_cfg.get("mae", {}).get("mae_primary", "raw")
    MAE = mae_raw if mae_primary == "raw" else mae_norm

    PSNR, mse = compute_mask_psnr(
        GT_norm,
        pred_norm,
        mask,
        eps=float(metric_cfg.get("psnr", {}).get("eps", 1e-8)),
    )

    if metric_cfg.get("ssim", {}).get("enabled", True):
        ssim_cfg = metric_cfg.get("ssim", {})
        SSIM, valid_ssim_slices = compute_masked_ssim_slice_map(
            GT_norm,
            pred_norm,
            mask,
            data_range=float(ssim_cfg.get("data_range", 1.0)),
            min_mask_pixels_per_slice=int(ssim_cfg.get("min_mask_pixels_per_slice", 100)),
            gaussian_weights=bool(ssim_cfg.get("gaussian_weights", False)),
        )
    else:
        SSIM, valid_ssim_slices = float("nan"), 0

    if metric_cfg.get("lpips", {}).get("enabled", True):
        lpips_cfg = metric_cfg.get("lpips", {})
        LPIPS, valid_lpips_slices = compute_lpips_volume(
            GT_norm,
            pred_norm,
            mask,
            lpips_runner=Diet.get("_runtime", {}).get("lpips_runner"),
            resize_hw=tuple(lpips_cfg.get("resize_hw", [256, 256])),
            min_mask_pixels_per_slice=int(lpips_cfg.get("min_mask_pixels_per_slice", 100)),
            bbox_padding=int(lpips_cfg.get("bbox_padding", 8)),
        )
    else:
        LPIPS, valid_lpips_slices = float("nan"), 0

    return {
        "MAE": float(MAE),
        "MAE_raw": mae_raw,
        "MAE_norm": mae_norm,
        "PSNR": PSNR,
        "SSIM": SSIM,
        "LPIPS": LPIPS,
        "mask_voxels": mask_voxels,
        "mse_norm_mask": mse,
        "valid_ssim_slices": int(valid_ssim_slices),
        "valid_lpips_slices": int(valid_lpips_slices),
    }


def _ssim_win_size(shape: tuple[int, int]) -> int | None:
    side = min(shape)
    if side < 3:
        return None
    size = min(7, side)
    if size % 2 == 0:
        size -= 1
    return max(size, 3)
