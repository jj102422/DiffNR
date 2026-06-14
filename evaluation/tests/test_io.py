from __future__ import annotations

import numpy as np
import pytest

from evaluation.src.align import StageError, prepare_case_for_metric
from evaluation.src.config_io import read_test_list
from evaluation.src.volume_io import load_volume


def test_npy_and_npz_loading(tmp_path):
    npy = tmp_path / "a.npy"
    np.save(npy, np.array([1, 2], dtype=np.float32))
    assert load_volume(npy).tolist() == [1.0, 2.0]

    npz = tmp_path / "a.npz"
    np.savez(npz, vol_pred=np.array([3, 4], dtype=np.float32))
    assert load_volume(npz, key="vol_pred").tolist() == [3.0, 4.0]


def test_npz_multiple_keys_requires_key(tmp_path):
    path = tmp_path / "multi.npz"
    np.savez(path, a=np.array([1]), b=np.array([2]))
    with pytest.raises(ValueError):
        load_volume(path)


def test_h5_missing_key_raises(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("ct", data=np.zeros((1, 2, 2), dtype=np.float32))
    with pytest.raises(KeyError):
        load_volume(path, file_type="h5", key="missing")


def test_nan_inf_sanitized(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.array([np.nan, np.inf, -np.inf, 2.0], dtype=np.float32))
    arr = load_volume(path)
    assert np.isfinite(arr).all()


def test_test_list_preserves_dotted_case_ids(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("1.3.6.1.4.1.14519\n/path/to/case0051.npy\n", encoding="utf-8")
    assert read_test_list(path) == ["1.3.6.1.4.1.14519", "case0051"]


def test_missing_pred_reports_stage(tmp_path):
    case = "case001"
    gt_dir = tmp_path / "gt" / case
    gt_dir.mkdir(parents=True)
    np.save(gt_dir / "ct.npy", np.zeros((1, 8, 8), dtype=np.float32))
    np.save(tmp_path / "mask.npy", np.ones((1, 8, 8), dtype=np.uint8))
    global_cfg = {
        "dataset": {
            "eval_gt_root": str(tmp_path / "gt"),
            "eval_gt_type": "npy",
            "eval_gt_h5_name": "ct.npy",
            "eval_gt_h5_key": None,
            "eval_gt_axis_order": "ZYX",
            "eval_mask_root": str(tmp_path),
            "eval_mask_pattern": "mask.npy",
            "eval_mask_axis_order": "ZYX",
        },
        "canonical": {"ct_min": 0.0, "ct_max": 1.0, "clip_before_metric": True},
    }
    model_cfg = {
        "pred_root": str(tmp_path / "pred"),
        "pred_pattern": "{case_id}.npy",
        "pred": {"file_type": "npy", "raw_axis_order": "ZYX", "normalized": False},
        "align_to_gt": {"strategy": "assert_same_shape"},
    }
    with pytest.raises(StageError) as exc:
        prepare_case_for_metric(case, "m", global_cfg, model_cfg)
    assert exc.value.stage == "load_pred"


def test_prepare_case_accepts_model_diet_spec_schema(tmp_path):
    case = "case001"
    gt = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    pred_xyz = np.transpose(gt / 2500.0, (2, 1, 0))
    mask = np.ones_like(gt, dtype=np.uint8)
    np.save(tmp_path / "gt.npy", gt)
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "pred.npy", pred_xyz)

    global_cfg = {
        "dataset": {
            "eval_gt_root": str(tmp_path),
            "eval_gt_type": "npy",
            "eval_gt_pattern": "gt.npy",
            "eval_gt_axis_order": "ZYX",
            "eval_mask_root": str(tmp_path),
            "eval_mask_type": "npy",
            "eval_mask_pattern": "mask.npy",
            "eval_mask_axis_order": "ZYX",
        },
        "eval": {
            "reference_space": "canonical_gt",
            "intensity": {"clip_for_metric": [0, 2500], "norm_min": 0, "norm_max": 2500},
            "mask": {"binarize_threshold": 0},
        },
    }
    model_cfg = {
        "pred_path_pattern": str(tmp_path / "pred.npy"),
        "pred_format": "npy",
        "read_backend": "numpy",
        "pred_axis_order_in_file": "XYZ",
        "transpose_order": [2, 1, 0],
        "flip_axes": [],
        "inverse_intensity": {"type": "affine", "formula": "pred_hu = clip(pred, 0, 1) * 2500"},
        "spatial_align": {"method": "identity", "interpolation": "linear"},
        "allow_metric": True,
        "alignment_quality": "verified",
    }

    gt_out, pred_out, mask_out, debug = prepare_case_for_metric(case, "m", global_cfg, model_cfg)
    assert np.array_equal(gt_out, gt)
    assert np.array_equal(pred_out, gt)
    assert mask_out.dtype == np.bool_
    assert debug["pred_after_axis_shape"] == "2x2x2"
    assert debug["alignment_quality"] == "verified"
