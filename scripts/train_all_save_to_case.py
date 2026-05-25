# Script to train all cases and save only pickle + vol_pred.npy into case folder.

import argparse
import glob
import numpy as np
import os
import os.path as osp
import shutil
import subprocess
from pathlib import Path


def find_latest_iteration(point_cloud_dir: str) -> str:
    if not osp.isdir(point_cloud_dir):
        raise FileNotFoundError(f"Missing point_cloud dir: {point_cloud_dir}")
    candidates = glob.glob(osp.join(point_cloud_dir, "iteration_*"))
    if not candidates:
        raise FileNotFoundError(f"No iteration folders under {point_cloud_dir}")

    def iter_num(path: str) -> int:
        base = osp.basename(path)
        try:
            return int(base.split("_")[-1])
        except Exception:
            return -1

    candidates.sort(key=iter_num)
    return candidates[-1]


def main(args):
    source_path = args.source
    output_root = args.output
    device = args.device
    config_path = args.config
    ckpt_path = args.ckpt
    organ_type = args.organ_type
    sd_turbo_path = args.sd_turbo_path
    keep_output = args.keep_output
    skip_existing = args.skip_existing
    fix_init = args.fix_init
    init_recon_method = args.init_recon_method
    init_n_points = args.init_n_points
    init_density_thresh = args.init_density_thresh
    init_density_rescale = args.init_density_rescale
    init_random_density_max = args.init_random_density_max

    case_paths = sorted(glob.glob(osp.join(source_path, "*")))
    if len(case_paths) == 0:
        raise ValueError(f"{source_path} find no folder!")

    for case_path in case_paths:
        case_name = osp.basename(case_path)
        case_output_path = osp.join(output_root, case_name)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        existing_vol_pred_npy = Path(case_path) / "vol_pred.npy"
        existing_vol_pred_npz = Path(case_path) / "vol_pred.npz"
        if skip_existing and (existing_vol_pred_npy.exists() or existing_vol_pred_npz.exists()):
            print(f"Skip {case_name}: vol_pred already exists.")
            continue

        init_path = osp.join(case_path, f"init_{case_name}.npy")
        init_missing = not osp.exists(init_path)
        init_empty = False
        if not init_missing:
            try:
                init_empty = osp.getsize(init_path) == 0
            except OSError:
                init_empty = True

        if fix_init and (init_missing or init_empty):
            if init_empty and osp.exists(init_path):
                try:
                    os.remove(init_path)
                except OSError:
                    pass
            init_cmd = [
                "python",
                "r2_gaussian/data_generator/initialize_pcd.py",
                "--data",
                case_path,
                "--output",
                init_path,
                "--recon_method",
                init_recon_method,
                "--n_points",
                str(init_n_points),
                "--density_thresh",
                str(init_density_thresh),
                "--density_rescale",
                str(init_density_rescale),
                "--random_density_max",
                str(init_random_density_max),
            ]
            print(f"Rebuilding init for {case_name} ...")
            subprocess.run(init_cmd, check=True, env=env)

        cmd = [
            "python",
            "train_DiffNR.py",
            "-s",
            case_path,
            "-m",
            case_output_path,
            "--slicefixer_model_path",
            ckpt_path,
            "--organ_type",
            organ_type,
            "--sd_turbo_path",
            sd_turbo_path,
        ]
        if config_path:
            cmd += ["--config", config_path]

        print(f"Training {case_name} ...")
        subprocess.run(cmd, check=True, env=env)

        point_cloud_dir = osp.join(case_output_path, "point_cloud")
        latest_iter_dir = find_latest_iteration(point_cloud_dir)

        src_vol_pred = osp.join(latest_iter_dir, "vol_pred.npy")
        if not osp.exists(src_vol_pred):
            raise FileNotFoundError(f"Missing vol_pred.npy in {latest_iter_dir}")

        dst_vol_pred = Path(case_path) / "vol_pred.npz"
        vol_pred = np.load(src_vol_pred, mmap_mode="r")
        np.savez_compressed(dst_vol_pred, vol_pred=vol_pred)
        stale_npy = Path(case_path) / "vol_pred.npy"
        if stale_npy.exists():
            stale_npy.unlink()
        if Path(init_path).exists():
            Path(init_path).unlink()
        print(f"Saved compressed vol_pred to {dst_vol_pred}")

        if not keep_output:
            shutil.rmtree(case_output_path, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=str, help="Path to CT dataset root.")
    parser.add_argument("--output", required=True, type=str, help="Temp output root.")
    parser.add_argument("--config", default=None, type=str, help="Path to config.")
    parser.add_argument("--device", default=0, type=int, help="GPU device.")
    parser.add_argument(
        "--ckpt",
        default="checkpoints/slicefixer/model.pkl",
        type=str,
        help="Path to SliceFixer checkpoint.",
    )
    parser.add_argument(
        "--sd_turbo_path",
        default="stabilityai/sd-turbo",
        type=str,
        help="Path or HF model id for SD-Turbo.",
    )
    parser.add_argument("--organ_type", default="Chest", type=str, help="CT organ type")
    parser.add_argument(
        "--keep_output",
        action="store_true",
        help="Keep training output folder under --output.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip cases that already have point_cloud.pickle and vol_pred.npy.",
    )
    parser.add_argument(
        "--fix_init",
        action="store_true",
        help="Rebuild init_*.npy if missing or empty.",
    )
    parser.add_argument("--init_recon_method", default="fdk", type=str)
    parser.add_argument("--init_n_points", default=50000, type=int)
    parser.add_argument("--init_density_thresh", default=0.05, type=float)
    parser.add_argument("--init_density_rescale", default=0.15, type=float)
    parser.add_argument("--init_random_density_max", default=1.0, type=float)

    main(parser.parse_args())
