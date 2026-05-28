import tempfile
import unittest
from pathlib import Path

import numpy as np

from slicefixer.volume_cache import TemporaryVolumeCache


class TemporaryVolumeCacheTest(unittest.TestCase):
    def test_materializes_slices_and_preserves_sources_on_cleanup(self):
        with tempfile.TemporaryDirectory() as source_root, tempfile.TemporaryDirectory() as cache_root:
            case_path = Path(source_root) / "case_a"
            case_path.mkdir()
            coarse = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
            target = coarse + 100.0
            np.savez_compressed(case_path / "vol_pred.npz", vol_pred=coarse)
            np.savez_compressed(case_path / "volume_gt.npz", volume_gt=target)

            cache = TemporaryVolumeCache(cache_root, "test_run")
            stats = cache.materialize_block("train_block", [str(case_path)])
            paths = cache.path_overrides("train_block", [str(case_path)])[str(case_path)]

            cached_coarse = np.load(paths["coarse_path"], mmap_mode="r")
            cached_target = np.load(paths["gt_path"], mmap_mode="r")
            np.testing.assert_array_equal(cached_coarse[:, :, 3], coarse[:, :, 3])
            np.testing.assert_array_equal(cached_target[:, :, 2], target[:, :, 2])
            self.assertEqual(stats["cases"], 1)
            self.assertGreater(stats["bytes"], 0)

            cache.cleanup_block("train_block")
            self.assertTrue((case_path / "vol_pred.npz").exists())
            self.assertTrue((case_path / "volume_gt.npz").exists())


if __name__ == "__main__":
    unittest.main()
