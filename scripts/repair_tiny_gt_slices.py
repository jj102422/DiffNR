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
    parser = argparse.ArgumentParser(
        description=(
            "Repair GT axial slices that were normalized twice. "
            "Slices with 0 < max(ct) < repair_max_threshold are multiplied by scale_factor."
        )
    )
    parser.add_argument("--dataset-root", default="/root/epfs/data")
    parser.add_argument("--info-json", default="/root/epfs/DiffNR/info.json")
    parser.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--repair-max-threshold", type=float, default=0.01)
    parser.add_argument("--scale-factor", type=float, default=2500.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_savez(path, arrays):
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def process_one(path_str, repair_max_threshold, scale_factor, dry_run):
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
        if not np.isfinite(old_min) or not np.isfinite(old_max):
            return ("nonfinite", path_str, (old_min, old_max))

        if 0.0 < old_max < repair_max_threshold:
            repaired = (ct * scale_factor).astype(np.float32)
            new_min = float(np.nanmin(repaired))
            new_max = float(np.nanmax(repaired))
            if not dry_run:
                arrays["ct"] = repaired
                atomic_savez(path, arrays)
            return ("would_repair" if dry_run else "repaired", path_str, (old_min, old_max, new_min, new_max))

        return ("already_ok", path_str, (old_min, old_max))
    except Exception as exc:
        return ("bad_files", path_str, repr(exc))


def collect_paths(dataset_root, info_json, splits, max_files):
    info = json.loads(Path(info_json).read_text())
    case_ids = []
    for split in splits:
        case_ids.extend(info.get(split, []))
    case_ids = list(dict.fromkeys(case_ids))

    paths = []
    for case_id in case_ids:
        gt_dir = Path(dataset_root) / case_id / "gt"
        if gt_dir.is_dir():
            paths.extend(sorted(gt_dir.glob("axial_*.npz")))
    paths = [str(path) for path in paths]
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def main():
    args = parse_args()
    paths = collect_paths(args.dataset_root, args.info_json, args.splits, args.max_files)
    stats = {
        "already_ok": 0,
        "would_repair": 0,
        "repaired": 0,
        "missing_ct_key": 0,
        "nonfinite": 0,
        "bad_files": 0,
    }
    examples = {key: [] for key in stats}
    started = time.time()
    total = len(paths)
    print(
        f"total_gt_slices={total} workers={args.workers} splits={','.join(args.splits)} "
        f"dry_run={args.dry_run} repair=0<max<{args.repair_max_threshold} scale={args.scale_factor}",
        flush=True,
    )

    worker = partial(
        process_one,
        repair_max_threshold=args.repair_max_threshold,
        scale_factor=args.scale_factor,
        dry_run=args.dry_run,
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
