# Script to train all cases and save only pickle + vol_pred.npy into case folder.
# Support distributed training using torchrun

import argparse
import glob
import os
import os.path as osp
import shutil
import subprocess
import torch


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
    use_torchrun = args.use_torchrun
    nproc_per_node = args.nproc_per_node
    nnodes = args.nnodes
    node_rank = args.node_rank
    master_addr = args.master_addr
    master_port = args.master_port
    train_batch_size = args.train_batch_size
    start_idx = args.start_idx
    end_idx = args.end_idx

    case_paths = sorted(glob.glob(osp.join(source_path, "*")))
    if len(case_paths) == 0:
        raise ValueError(f"{source_path} find no folder!")
    
    # Support case range
    if start_idx is not None and end_idx is not None:
        case_paths = case_paths[start_idx:end_idx]
        print(f"Processing cases from index {start_idx} to {end_idx}: {len(case_paths)} cases")

    for case_idx, case_path in enumerate(case_paths):
        case_name = osp.basename(case_path)
        case_output_path = osp.join(output_root, case_name)
        
        existing_pickle = osp.join(case_path, "point_cloud.pickle")
        existing_vol_pred = osp.join(case_path, "vol_pred.npy")
        if skip_existing and osp.exists(existing_pickle) and osp.exists(existing_vol_pred):
            print(f"[{case_idx+1}/{len(case_paths)}] Skip {case_name}: outputs already exist.")
            continue

        # Handle initialization if needed
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
            print(f"[{case_idx+1}/{len(case_paths)}] Rebuilding init for {case_name} ...")
            subprocess.run(init_cmd, check=True)

        # Prepare training command
        if use_torchrun:
            # Use torchrun for distributed training
            cmd = [
                "torchrun",
                "--nproc_per_node", str(nproc_per_node),
                "--nnodes", str(nnodes),
                "--node_rank", str(node_rank),
                "--master_addr", master_addr,
                "--master_port", str(master_port),
                "train_DiffNR.py",
                "-s", case_path,
                "-m", case_output_path,
                "--slicefixer_model_path", ckpt_path,
                "--organ_type", organ_type,
                "--sd_turbo_path", sd_turbo_path,
                "--train_batch_size", str(train_batch_size),
            ]
        else:
            # Use single GPU training
            cmd = [
                "python",
                "train_DiffNR.py",
                "-s", case_path,
                "-m", case_output_path,
                "--slicefixer_model_path", ckpt_path,
                "--organ_type", organ_type,
                "--sd_turbo_path", sd_turbo_path,
                "--train_batch_size", str(train_batch_size),
            ]
        
        if config_path:
            cmd += ["--config", config_path]

        print(f"[{case_idx+1}/{len(case_paths)}] Training {case_name} ...")
        if use_torchrun:
            print(f"  Using torchrun: nproc_per_node={nproc_per_node}, batch_size={train_batch_size}")
        subprocess.run(cmd, check=True)

        point_cloud_dir = osp.join(case_output_path, "point_cloud")
        latest_iter_dir = find_latest_iteration(point_cloud_dir)

        src_pickle = osp.join(latest_iter_dir, "point_cloud.pickle")
        src_vol_pred = osp.join(latest_iter_dir, "vol_pred.npy")
        if not osp.exists(src_pickle) or not osp.exists(src_vol_pred):
            raise FileNotFoundError(
                f"Missing outputs in {latest_iter_dir}: point_cloud.pickle / vol_pred.npy"
            )

        dst_pickle = osp.join(case_path, "point_cloud.pickle")
        dst_vol_pred = osp.join(case_path, "vol_pred.npy")
        shutil.copy2(src_pickle, dst_pickle)
        shutil.copy2(src_vol_pred, dst_vol_pred)
        print(f"  Saved outputs to {case_path}")

        if not keep_output:
            shutil.rmtree(case_output_path, ignore_errors=True)
    
    print(f"Completed processing {len(case_paths)} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all cases with distributed training support"
    )
    parser.add_argument("--source", required=True, type=str, help="Path to CT dataset root.")
    parser.add_argument("--output", required=True, type=str, help="Temp output root.")
    parser.add_argument("--config", default=None, type=str, help="Path to config.")
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
        "--train_batch_size", 
        type=int, 
        default=1,
        help="Batch size for training (number of viewpoints per iteration)"
    )
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
    
    # Distributed training arguments
    parser.add_argument(
        "--use_torchrun",
        action="store_true",
        help="Use torchrun for distributed training"
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Number of GPUs per node"
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes"
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Rank of current node"
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Master node address for distributed training"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master node port for distributed training"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Start case index (for parallel processing across multiple scripts)"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End case index (for parallel processing across multiple scripts)"
    )

    main(parser.parse_args())
