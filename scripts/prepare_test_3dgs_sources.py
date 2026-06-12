#!/usr/bin/env python
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from r2_gaussian.data_generator.synthetic_dataset.generate_data import (  # noqa: E402
    generate_two_view_projections,
    save_case,
)


REQUIRED_SOURCE_FILES = (
    "meta_data.json",
    "volume_gt.npy",
    "proj_train/0000.npy",
    "proj_train/0001.npy",
    "proj_test/0000.npy",
    "proj_test/0001.npy",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare 3DGS source folders only for info.json test cases."
    )
    parser.add_argument("--info-json", default="/root/epfs/DiffNR/info.json")
    parser.add_argument("--data-root", default="/root/epfs/data")
    parser.add_argument("--output-root", default="/root/epfs/test")
    parser.add_argument(
        "--scanner",
        default=str(REPO_ROOT / "r2_gaussian/data_generator/scanner/cone_beam.yml"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--gt-key", default="ct")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-init", action="store_true")
    parser.add_argument("--init-recon-method", default="fdk")
    parser.add_argument("--init-n-points", type=int, default=50000)
    parser.add_argument("--init-density-thresh", type=float, default=0.01)
    parser.add_argument("--init-density-rescale", type=float, default=0.15)
    return parser.parse_args()


def load_npz_slice(path: Path, preferred_key: str) -> np.ndarray:
    with np.load(path) as data:
        key = preferred_key if preferred_key in data.files else data.files[0]
        return data[key].astype(np.float32)


def load_case_volume(case_data_dir: Path, gt_key: str) -> np.ndarray:
    gt_dir = case_data_dir / "gt"
    paths = sorted(gt_dir.glob("axial_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No GT axial slices under {gt_dir}")
    slices = [np.clip(load_npz_slice(path, gt_key), 0.0, None) for path in paths]
    return np.stack(slices, axis=-1).astype(np.float32)


def source_is_complete(case_dir: Path, case_id: str, require_init: bool) -> bool:
    for rel in REQUIRED_SOURCE_FILES:
        if not (case_dir / rel).exists():
            return False
    if require_init and not (case_dir / f"init_{case_id}.npy").exists():
        return False
    if not (case_dir / f"{case_id}_xray_1.pt").exists():
        return False
    if not (case_dir / f"{case_id}_xray_2.pt").exists():
        return False
    return True


def copy_xray_features(case_data_dir: Path, case_out_dir: Path, case_id: str):
    for idx in (1, 2):
        src = case_data_dir / f"{case_id}_xray_{idx}.pt"
        dst = case_out_dir / f"{case_id}_xray_{idx}.pt"
        if not src.exists():
            raise FileNotFoundError(f"Missing xray feature: {src}")
        shutil.copy2(src, dst)


def write_source(case_data_dir: Path, case_out_dir: Path, case_id: str, scanner_cfg: dict, gt_key: str):
    volume = load_case_volume(case_data_dir, gt_key)
    cfg = copy.deepcopy(scanner_cfg)
    projs, angles_rad, volume_used = generate_two_view_projections(
        volume,
        cfg,
        angles_deg=[0.0, 90.0],
    )
    save_case(str(case_out_dir), case_id, volume_used, projs, angles_rad, cfg)
    copy_xray_features(case_data_dir, case_out_dir, case_id)


def build_init(case_out_dir: Path, case_id: str, args):
    init_path = case_out_dir / f"init_{case_id}.npy"
    if init_path.exists() and not args.force:
        return
    if init_path.exists():
        init_path.unlink()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "r2_gaussian/data_generator/initialize_pcd.py"),
        "--data",
        str(case_out_dir),
        "--output",
        str(init_path),
        "--recon_method",
        args.init_recon_method,
        "--n_points",
        str(args.init_n_points),
        "--density_thresh",
        str(args.init_density_thresh),
        "--density_rescale",
        str(args.init_density_rescale),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def main():
    args = parse_args()
    info = json.loads(Path(args.info_json).read_text())
    case_ids = info.get(args.split, [])
    if not case_ids:
        raise ValueError(f"No cases found in split {args.split!r}: {args.info_json}")

    with open(args.scanner, "r", encoding="utf-8") as f:
        scanner_cfg = yaml.safe_load(f)

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    done = 0
    generated = 0
    for index, case_id in enumerate(case_ids, start=1):
        case_data_dir = data_root / case_id
        case_out_dir = output_root / case_id
        if not case_data_dir.is_dir():
            raise FileNotFoundError(f"Missing test case data directory: {case_data_dir}")

        if source_is_complete(case_out_dir, case_id, require_init=not args.skip_init) and not args.force:
            print(f"[{index:03d}/{len(case_ids):03d}] skip complete {case_id}", flush=True)
            done += 1
            continue

        print(f"[{index:03d}/{len(case_ids):03d}] prepare 3DGS source {case_id}", flush=True)
        case_out_dir.mkdir(parents=True, exist_ok=True)
        write_source(case_data_dir, case_out_dir, case_id, scanner_cfg, args.gt_key)
        if not args.skip_init:
            build_init(case_out_dir, case_id, args)
        generated += 1

    print(
        f"Prepared test 3DGS sources: skipped_complete={done}, generated_or_refreshed={generated}",
        flush=True,
    )


if __name__ == "__main__":
    main()
