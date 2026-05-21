import os
import os.path as osp
import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import gpu
import numpy as np
import yaml
import plotly.graph_objects as go
import scipy.ndimage.interpolation
from tigre.utilities import CTnoise
import json
import matplotlib.pyplot as plt
import tigre.algorithms as algs
import argparse
import open3d as o3d
import pickle
import copy
import torch

try:
    import SimpleITK as sitk
except Exception:
    sitk = None

import sys

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry_tigre, recon_volume


def load_mha_volume(mha_path):
    if sitk is None:
        raise ImportError(
            "SimpleITK is required to read .mha files. Please install SimpleITK."
        )
    image = sitk.ReadImage(mha_path)
    volume_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    volume_xyz = np.transpose(volume_zyx, (2, 1, 0))
    
    # Normalize CT volume: clip at 0, normalize by fixed range 3000
    # Following the pattern from process_dcm: clip(0, 3000) → [0, 3000] range → normalize to [0, 1]
    volume_xyz = np.clip(volume_xyz, 0.0, None)  # Clip left at 0, no right bound
    volume_xyz = volume_xyz / 3000.0  # Fixed normalization range: 3000
    volume_xyz = np.clip(volume_xyz, 0.0, 1.0)  # Ensure [0, 1] range
    
    return volume_xyz




def calc_nDetector(DSD, DSO, nVoxel, dVoxel):
    nVoxel_W = nVoxel[0]
    nDetector_W = np.round(nVoxel_W * DSD / (DSO - nVoxel_W * dVoxel[0] / 2)).astype(
        int
    )

    nVoxel_H = nVoxel[0]
    nDetector_H = np.round(nVoxel_H * DSD / (DSO - nVoxel_H * dVoxel[0] / 2)).astype(
        int
    )
    return [int(nDetector_H), int(nDetector_W)]


def update_scanner_for_volume(scanner_cfg, vol, auto_nvoxel=True, auto_ndetector=True):
    if auto_nvoxel:
        scanner_cfg["nVoxel"] = [int(x) for x in vol.shape]
        if "sVoxel" in scanner_cfg:
            scanner_cfg["dVoxel"] = (
                np.array(scanner_cfg["sVoxel"]) / np.array(scanner_cfg["nVoxel"])
            ).tolist()
    if auto_ndetector and "dVoxel" in scanner_cfg:
        scanner_cfg["nDetector"] = calc_nDetector(
            scanner_cfg["DSD"],
            scanner_cfg["DSO"],
            scanner_cfg["nVoxel"],
            scanner_cfg["dVoxel"],
        )
        if "sDetector" in scanner_cfg:
            scanner_cfg["dDetector"] = (
                np.array(scanner_cfg["sDetector"]) / np.array(scanner_cfg["nDetector"])
            ).tolist()


def generate_two_view_projections(vol, scanner_cfg, angles_deg):
    target_shape = tuple(int(x) for x in scanner_cfg.get("nVoxel", vol.shape))
    if vol.shape != target_shape:
        if any(v > t for v, t in zip(vol.shape, target_shape)):
            raise ValueError(
                f"Volume shape {vol.shape} exceeds target nVoxel {target_shape}."
            )
        padded = np.zeros(target_shape, dtype=vol.dtype)
        start = [(t - v) // 2 for t, v in zip(target_shape, vol.shape)]
        end = [s + v for s, v in zip(start, vol.shape)]
        padded[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = vol
        vol = padded
    geo = get_geometry_tigre(scanner_cfg)
    angles_rad = np.deg2rad(np.array(angles_deg, dtype=np.float32))
    angles_rad = angles_rad + scanner_cfg.get("startAngle", 0.0) / 180 * np.pi
    projs = tigre.Ax(np.transpose(vol, (2, 1, 0)).copy(), geo, angles_rad)[:, ::-1, :]
    if scanner_cfg.get("noise", False):
        projs = CTnoise.add(
            projs,
            Poisson=float(scanner_cfg.get("possion_noise", 0)),
            Gaussian=np.array(scanner_cfg.get("gaussian_noise", [0, 0])),
        )
        projs[projs < 0.0] = 0.0
    return projs, angles_rad, vol


def save_case(case_save_path, case_name, vol, projs, angles_rad, scanner_cfg):
    os.makedirs(case_save_path, exist_ok=True)
    np.save(osp.join(case_save_path, "volume_gt.npy"), vol)

    proj_train_dir = osp.join(case_save_path, "proj_train")
    proj_test_dir = osp.join(case_save_path, "proj_test")
    os.makedirs(proj_train_dir, exist_ok=True)
    os.makedirs(proj_test_dir, exist_ok=True)

    file_path_dict = {"proj_train": [], "proj_test": []}

    for idx in range(projs.shape[0]):
        proj = projs[idx]
        npy_name = f"{idx:04d}.npy"
        np.save(osp.join(proj_train_dir, npy_name), proj)
        np.save(osp.join(proj_test_dir, npy_name), proj)

        file_path_dict["proj_train"].append(
            {"file_path": osp.join("proj_train", npy_name), "angle": float(angles_rad[idx])}
        )
        file_path_dict["proj_test"].append(
            {"file_path": osp.join("proj_test", npy_name), "angle": float(angles_rad[idx])}
        )

    torch.save(
        torch.from_numpy(projs[0]).float(),
        osp.join(case_save_path, f"{case_name}_xray_1.pt"),
    )
    torch.save(
        torch.from_numpy(projs[1]).float(),
        osp.join(case_save_path, f"{case_name}_xray_2.pt"),
    )

    meta = {
        "scanner": scanner_cfg,
        "vol": "volume_gt.npy",
        "bbox": [[-1, -1, -1], [1, 1, 1]],
        "proj_train": file_path_dict["proj_train"],
        "proj_test": file_path_dict["proj_test"],
    }
    with open(osp.join(case_save_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


def main(args):
    """Assume CT is in a unit cube. We synthesize two fixed-view X-ray projections."""
    scanner_cfg_path = args.scanner
    output_path = args.output

    with open(scanner_cfg_path, "r") as handle:
        scanner_cfg = yaml.safe_load(handle)

    angles_deg = [0.0, 90.0]

    if args.data_root:
        case_dirs = [
            osp.join(args.data_root, d)
            for d in sorted(os.listdir(args.data_root))
            if osp.isdir(osp.join(args.data_root, d))
        ]
        for case_dir in case_dirs:
            case_name = osp.basename(case_dir)
            ct_path = osp.join(case_dir, args.ct_name)
            if not osp.exists(ct_path):
                print(f"Skip {case_name}: missing {args.ct_name}")
                continue
            print(f"Generate data for case {case_name}")
            vol = load_mha_volume(ct_path)
            update_scanner_for_volume(
                scanner_cfg,
                vol,
                auto_nvoxel=False,
                auto_ndetector=False,
            )
            projs, angles_rad, vol_used = generate_two_view_projections(
                vol, scanner_cfg, angles_deg
            )
            case_save_path = case_dir
            save_case(
                case_save_path, case_name, vol_used, projs, angles_rad, scanner_cfg
            )
            print(f"Generate data for case {case_name} complete!")
        return

    vol_path = args.vol
    if vol_path.endswith(".mha"):
        vol = load_mha_volume(vol_path)
    else:
        vol = np.load(vol_path).astype(np.float32)
    vol_name = osp.basename(vol_path).split(".")[0]
    if args.output:
        case_name = osp.basename(osp.normpath(args.output))
    else:
        case_name = vol_name

    print(f"Generate data for case {case_name}")
    update_scanner_for_volume(
        scanner_cfg,
        vol,
        auto_nvoxel=False,
        auto_ndetector=False,
    )
    projs, angles_rad, vol_used = generate_two_view_projections(
        vol, scanner_cfg, angles_deg
    )
    case_save_path = output_path
    save_case(case_save_path, case_name, vol_used, projs, angles_rad, scanner_cfg)
    print(f"Generate data for case {case_name} complete!")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Data generator parameters")
    
    parser.add_argument("--vol", default="data_generator/volume_gt/0_chest.npy", type=str, help="Path to volume (.npy or .mha).")
    parser.add_argument("--scanner", default="data_generator/scanner/cone_beam.yml", type=str, help="Path to scanner configuration.")
    parser.add_argument("--output", default="data/cone_ntrain_50_angle_360", type=str, help="Path to output.")
    parser.add_argument("--n_train", default=50, type=int, help="(Unused) Number of projections for training.")
    parser.add_argument("--n_test", default=100, type=int, help="(Unused) Number of projections for evaluation.")
    parser.add_argument("--data_root", default="", type=str, help="Root folder containing case directories with ct_file.mha.")
    parser.add_argument("--ct_name", default="ct_file.mha", type=str, help="CT filename inside each case directory.")
    parser.add_argument("--fixed_angles", default="", type=str, help="(Unused) Comma-separated angles in degrees.")
    parser.add_argument("--n_views", default=50, type=int, help="(Unused) Number of evenly spaced views.")
    parser.add_argument("--disable_auto_nvoxel", action="store_true", help="Disable auto nVoxel update from volume shape.")
    parser.add_argument("--disable_auto_ndetector", action="store_true", help="Disable auto nDetector update from volume shape.")
    # fmt: on

    args = parser.parse_args()
    main(args)