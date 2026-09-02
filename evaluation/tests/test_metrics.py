from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.src.metrics import compute_masked_ssim_slice_map, metric


class MockLPIPS:
    batch_size = 4

    def prepare_slice(self, img2d_norm, bbox, resize_hw=(256, 256)):
        return (img2d_norm, bbox)

    def forward_batch(self, gt_list, pred_list):
        scores = []
        for (gt, _), (pred, _) in zip(gt_list, pred_list):
            scores.append(float(np.mean(np.abs(gt - pred))))
        return scores


def diet(lpips_enabled=True, mask_outside=False):
    return {
        "canonical": {"ct_min": 0.0, "ct_max": 1.0},
        "metric": {
            "mae": {"mae_primary": "raw"},
            "psnr": {"eps": 1e-12},
            "ssim": {
                "enabled": True,
                "data_range": 1.0,
                "min_mask_pixels_per_slice": 1,
                "mask_outside": mask_outside,
            },
            "lpips": {
                "enabled": lpips_enabled,
                "resize_hw": [8, 8],
                "min_mask_pixels_per_slice": 1,
                "bbox_padding": 0,
                "mask_outside": mask_outside,
            },
        },
        "_runtime": {"lpips_runner": MockLPIPS()},
    }


def test_metric_identity():
    rng = np.random.default_rng(7)
    gt = rng.random((3, 8, 8), dtype=np.float32)
    mask = np.ones_like(gt, dtype=bool)
    out = metric(gt, gt.copy(), mask, diet())
    assert out["MAE"] == 0.0
    assert out["MAE_raw"] == 0.0
    assert math.isinf(out["PSNR"])
    assert out["SSIM"] == pytest.approx(1.0)
    assert out["LPIPS"] == pytest.approx(0.0)


def test_mask_mae_psnr_only_uses_mask():
    gt = np.zeros((1, 8, 8), dtype=np.float32)
    pred = np.zeros_like(gt)
    pred[:, :4, :4] = 0.5
    pred[:, 4:, 4:] = 1.0
    mask = np.zeros_like(gt, dtype=bool)
    mask[:, :4, :4] = True
    out = metric(gt, pred, mask, diet(lpips_enabled=False))
    assert out["MAE"] == pytest.approx(0.5)
    assert out["mse_norm_mask"] == pytest.approx(0.25)
    assert out["PSNR"] == pytest.approx(10.0 * np.log10(1.0 / 0.25))


def test_ssim_full_map_mask_average_skips_small_masks():
    gt = np.ones((2, 8, 8), dtype=np.float32)
    pred = gt.copy()
    mask = np.zeros_like(gt, dtype=bool)
    mask[0, :2, :2] = True
    mask[1, :, :] = True
    score, valid = compute_masked_ssim_slice_map(gt, pred, mask, min_mask_pixels_per_slice=10)
    assert valid == 1
    assert score == pytest.approx(1.0)


def test_empty_mask_raises():
    gt = np.zeros((1, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        metric(gt, gt.copy(), np.zeros_like(gt, dtype=bool), diet(lpips_enabled=False))


def test_all_roi_metrics_ignore_errors_outside_exact_mask():
    gt = np.zeros((1, 16, 16), dtype=np.float32)
    pred = np.ones_like(gt)
    mask = np.zeros_like(gt, dtype=bool)
    mask[:, 4:12, 4:12] = True
    pred[mask] = gt[mask]

    out = metric(gt, pred, mask, diet(lpips_enabled=True, mask_outside=True))

    assert out["MAE"] == 0.0
    assert math.isinf(out["PSNR"])
    assert out["SSIM"] == pytest.approx(1.0)
    assert out["LPIPS"] == pytest.approx(0.0)


def test_cropped_strict_roi_ssim_matches_full_zero_background_map():
    rng = np.random.default_rng(19)
    gt = rng.random((1, 32, 32), dtype=np.float32)
    pred = rng.random((1, 32, 32), dtype=np.float32)
    mask = np.zeros_like(gt, dtype=bool)
    mask[:, 8:25, 9:24] = True
    gt_zero = np.where(mask, gt, 0.0)
    pred_zero = np.where(mask, pred, 0.0)

    full_score, full_valid = compute_masked_ssim_slice_map(
        gt_zero,
        pred_zero,
        mask,
        min_mask_pixels_per_slice=1,
        mask_outside=False,
    )
    cropped_score, cropped_valid = compute_masked_ssim_slice_map(
        gt,
        pred,
        mask,
        min_mask_pixels_per_slice=1,
        mask_outside=True,
    )

    assert cropped_valid == full_valid == 1
    assert cropped_score == pytest.approx(full_score, abs=1.0e-7)
