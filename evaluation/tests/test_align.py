from __future__ import annotations

import numpy as np
import pytest

from evaluation.src.align import (
    align_pred_to_gt,
    apply_axis_transform,
    center_crop_or_pad_z,
    center_crop_or_pad_to_shape,
    inverse_intensity_from_cfg,
    restore_intensity,
    reshape_with_placeholders,
)
from evaluation.src.mask_ops import resize_mask_to_shape


def test_center_crop_z():
    pred = np.zeros((512, 8, 8), dtype=np.float32)
    pred[:, 0, 0] = np.arange(512)
    cropped = center_crop_or_pad_z(pred, target_z=482)
    assert cropped.shape == (482, 8, 8)
    assert cropped[0, 0, 0] == 15
    assert cropped[-1, 0, 0] == 496


def test_center_pad_z():
    pred = np.ones((3, 2, 2), dtype=np.float32)
    padded = center_crop_or_pad_z(pred, target_z=5)
    assert padded.shape == (5, 2, 2)
    assert np.all(padded[0] == 0)
    assert np.all(padded[1:4] == 1)
    assert np.all(padded[4] == 0)


def test_offset_pad_z():
    pred = np.ones((3, 2, 2), dtype=np.float32)
    padded = center_crop_or_pad_z(pred, target_z=6, z_mode="offset", z_offset=2)
    assert padded.shape == (6, 2, 2)
    assert np.all(padded[:2] == 0)
    assert np.all(padded[2:5] == 1)
    assert np.all(padded[5] == 0)


def test_offset_crop_z():
    pred = np.zeros((6, 2, 2), dtype=np.float32)
    pred[:, 0, 0] = np.arange(6)
    cropped = center_crop_or_pad_z(pred, target_z=3, z_mode="offset", z_offset=2)
    assert cropped.shape == (3, 2, 2)
    assert cropped[:, 0, 0].tolist() == [2, 3, 4]


def test_resize_mask_nearest():
    mask = np.zeros((1, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    resized = resize_mask_to_shape(mask, (1, 4, 4))
    assert resized.dtype == np.bool_
    assert resized.shape == (1, 4, 4)
    assert resized.sum() == 4


def test_reshape_with_gt_placeholder():
    arr = np.arange(2 * 3 * 4)
    out = reshape_with_placeholders(arr, [2, 3, "{gt_z}"], (4, 9, 9))
    assert out.shape == (2, 3, 4)


def test_reshape_element_mismatch_raises():
    with pytest.raises(ValueError):
        reshape_with_placeholders(np.arange(10), [2, 3, "{gt_z}"], (4, 9, 9))


def test_transpose_and_flip_from_diet():
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    out = apply_axis_transform(arr, {"transpose": [2, 1, 0], "flip_axes": ["Z"]})
    expected = np.flip(np.transpose(arr, (2, 1, 0)), axis=0)
    assert np.array_equal(out, expected)


def test_rot90_from_diet():
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    out = apply_axis_transform(arr, {"raw_axis_order": "ZYX", "rot90": {"k": 1, "axes": ["Y", "X"]}})
    expected = np.rot90(arr, k=1, axes=(1, 2))
    assert np.array_equal(out, expected)


def test_inverse_slicefixer_output_intensity_transform():
    pred = np.array([-0.25, 0.0, 0.5, 1.0, 1.25], dtype=np.float32)
    out = restore_intensity(pred, {"intensity_transform": "inverse_slicefixer_output"})
    assert out.tolist() == [0.0, 0.0, 1500.0, 3000.0, 3000.0]


def test_spec_inverse_formula_2500_minus_1024():
    pred = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    out = inverse_intensity_from_cfg(
        pred,
        {
            "type": "affine",
            "formula": "pred_hu = clip(pred_norm, 0, 1) * 2500 - 1024",
            "clip_hu": [-1024, 1476],
        },
    )
    assert out.tolist() == [-1024.0, -1024.0, 226.0, 1476.0, 1476.0]


def test_center_crop_or_pad_to_shape_uses_fill_value():
    pred = np.ones((1, 2, 3), dtype=np.float32)
    out = center_crop_or_pad_to_shape(pred, (3, 4, 3), pad_value=-1024.0)
    assert out.shape == (3, 4, 3)
    assert np.all(out[1, 1:3, :] == 1)
    assert out[0, 0, 0] == -1024.0


def test_assert_same_shape_mismatch_raises():
    pred = np.zeros((2, 2, 2), dtype=np.float32)
    gt = np.zeros((3, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        align_pred_to_gt(pred, gt, {"strategy": "assert_same_shape"})
