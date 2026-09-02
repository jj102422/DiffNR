import numpy as np

from evaluation.scripts.evaluate_spine_multitask_masks import (
    overlap_metrics,
    surface_metrics,
)


def test_identical_mask_metrics_are_perfect():
    mask = np.zeros((12, 12, 8), dtype=bool)
    mask[3:8, 2:9, 2:6] = True
    overlap = overlap_metrics(mask, mask)
    hd95, surface_dice = surface_metrics(mask, mask, (1.0, 1.0, 2.0), 2.0)
    assert overlap == {"dice": 1.0, "precision": 1.0, "recall": 1.0}
    assert hd95 == 0.0
    assert surface_dice == 1.0


def test_one_voxel_shift_respects_physical_spacing():
    target = np.zeros((12, 12, 8), dtype=bool)
    prediction = np.zeros_like(target)
    target[4:7, 4:7, 3:5] = True
    prediction[5:8, 4:7, 3:5] = True
    hd95, surface_dice = surface_metrics(
        prediction,
        target,
        spacing_xyz=(2.5, 1.0, 1.0),
        tolerance_mm=2.0,
    )
    assert hd95 == 2.5
    assert 0.0 < surface_dice < 1.0


def test_empty_masks_have_defined_metrics():
    empty = np.zeros((4, 4, 4), dtype=bool)
    assert overlap_metrics(empty, empty)["dice"] == 1.0
    assert surface_metrics(empty, empty, (1.0, 1.0, 1.0), 2.0) == (0.0, 1.0)
