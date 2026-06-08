import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/root/epfs/data")
    parser.add_argument("--info-json", default="/root/epfs/DiffNR/info.json")
    parser.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--hu-min", type=float, default=0.0)
    parser.add_argument("--hu-max", type=float, default=2500.0)
    parser.add_argument("--processed-max-threshold", type=float, default=0.01)
    parser.add_argument("--tiny-repair-threshold", type=float, default=1e-5)
    return parser.parse_args()


def process_one(path_str, hu_min, hu_max, processed_max_threshold, tiny_repair_threshold):
    path = Path(path_str)
    try:
        with np.load(path) as archive:
            files = list(archive.files)
            if "ct" not in files:
                return ("missing_ct_key", path_str, None)
            arrays = {key: archive[key] for key in files}

        ct = arrays["ct"].astype(np.float32, copy=False)
        old_min = float(np.nanmin(ct))
        old_max = float(np.nanmax(ct))
        if (
            np.isfinite(old_max)
            and np.isfinite(old_min)
            and 0.0 < old_max < tiny_repair_threshold
        ):
            arrays["ct"] = (ct * (hu_max - hu_min)).astype(np.float32)
            fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=str(path.parent))
            os.close(fd)
            try:
                np.savez_compressed(tmp_name, **arrays)
                os.replace(tmp_name, path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            new_ct = arrays["ct"]
            return (
                "tiny_repaired",
                path_str,
                (old_min, old_max, float(np.nanmin(new_ct)), float(np.nanmax(new_ct))),
            )

        if (
            np.isfinite(old_max)
            and np.isfinite(old_min)
            and old_max <= processed_max_threshold
            and old_min >= -processed_max_threshold
        ):
            return ("already_processed", path_str, (old_min, old_max))

        arrays["ct"] = ((ct - hu_min) / (hu_max - hu_min)).astype(np.float32)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=str(path.parent))
        os.close(fd)
        try:
            np.savez_compressed(tmp_name, **arrays)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        new_ct = arrays["ct"]
        return (
            "normalized_now",
            path_str,
            (old_min, old_max, float(np.nanmin(new_ct)), float(np.nanmax(new_ct))),
        )
    except Exception as exc:
        return ("bad_files", path_str, repr(exc))


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    info = json.loads(Path(args.info_json).read_text())
    case_ids = []
    for split in args.splits:
        case_ids.extend(info.get(split, []))
    case_ids = list(dict.fromkeys(case_ids))

    paths = []
    for case_id in case_ids:
        gt_dir = dataset_root / case_id / "gt"
        if gt_dir.is_dir():
            paths.extend(sorted(gt_dir.glob("axial_*.npz")))
    paths = [str(path) for path in paths]
    if args.max_files is not None:
        paths = paths[: args.max_files]

    stats = {
        "already_processed": 0,
        "tiny_repaired": 0,
        "normalized_now": 0,
        "missing_ct_key": 0,
        "bad_files": 0,
    }
    examples = {key: [] for key in stats}
    started = time.time()
    total = len(paths)
    print(f"total_gt_slices={total} workers={args.workers} splits={','.join(args.splits)}", flush=True)

    worker = partial(
        process_one,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        processed_max_threshold=args.processed_max_threshold,
        tiny_repair_threshold=args.tiny_repair_threshold,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(worker, paths, chunksize=args.chunksize)
        for done, result in enumerate(results, 1):
            key, path, detail = result
            stats[key] += 1
            if len(examples[key]) < 5:
                examples[key].append((path, detail))
            if done % args.progress_every == 0 or done == total:
                elapsed = time.time() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                print(
                    f"progress={done}/{total} elapsed={elapsed:.1f}s "
                    f"rate={rate:.1f}/s eta={eta/60:.1f}min stats={stats}",
                    flush=True,
                )

    print("final_stats=", stats, flush=True)
    print("examples=", examples, flush=True)


if __name__ == "__main__":
    main()
