from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.scripts.evaluate_all import validate_complete_matrix
from evaluation.scripts.repair_slicefixer_volume_from_shards import rebuild_from_shards


def test_rebuild_slicefixer_volume_from_shards_preserves_original(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez_compressed(
        shard_dir / "shard_rank000.npz",
        indices=np.array([0, 2], dtype=np.int64),
        slices=np.stack(
            [np.zeros((2, 2), dtype=np.float32), np.full((2, 2), 2.0, dtype=np.float32)]
        ),
    )
    np.savez_compressed(
        shard_dir / "shard_rank001.npz",
        indices=np.array([1], dtype=np.int64),
        slices=np.ones((1, 2, 2), dtype=np.float32),
    )
    output = tmp_path / "vol_pred_slicefixer.npz"
    output.write_bytes(b"corrupt")

    rebuilt = rebuild_from_shards(tmp_path, output.name, "vol_pred", ".corrupt")

    assert rebuilt == output
    assert (tmp_path / "vol_pred_slicefixer.npz.corrupt").read_bytes() == b"corrupt"
    with np.load(output, allow_pickle=False) as data:
        volume = data["vol_pred"]
    assert volume.shape == (2, 2, 3)
    assert volume[0, 0].tolist() == [0.0, 1.0, 2.0]


def test_validate_complete_matrix_accepts_exact_product():
    rows = [
        {"model": "a", "case_id": "1"},
        {"model": "b", "case_id": "1"},
        {"model": "a", "case_id": "2"},
        {"model": "b", "case_id": "2"},
    ]
    validate_complete_matrix(rows, ["a", "b"], ["1", "2"])
