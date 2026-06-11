import os
import atexit
from pathlib import Path
import zipfile
import json
import pickle
import random
import sys
import time
import lpips
import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
# from pix2pix_turbo import Pix2Pix_Turbo
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "slicefixer")) # 确保能索引到 DiffNR 里的 slicefixer 模块
from SliceFixer import SliceFixer
from conditioning_utils import (
    axial_slice_index,
    build_context_stack,
    build_context_stack_from_indices,
    context_channel_count,
    clamped_context_indices,
    load_mask_volume,
)
from intensity_utils import volume_to_slicefixer
from my_utils.training_utils import parse_args_paired_training
from volume_cache import TemporaryVolumeCache
from r2_gaussian.utils.loss_utils import ssim as standard_ssim

try:
    sys.path.append(str(REPO_ROOT / "r2_gaussian/submodules/fused-ssim"))
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


def ssim_loss_and_value(pred, target):
    pred_01 = (pred.float() + 1.0) * 0.5
    target_01 = (target.float() + 1.0) * 0.5
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(pred_01, target_01)
    else:
        ssim_value = standard_ssim(pred_01, target_01)
    return 1.0 - ssim_value, ssim_value


class ConditionalDiscriminator(nn.Module):
    def __init__(self, discriminator, condition_channels=3, image_channels=3):
        super().__init__()
        in_channels = condition_channels + image_channels
        self.base_discriminator = discriminator
        self.condition_adapter = nn.Conv2d(in_channels, image_channels, kernel_size=1)
        self._init_as_target_passthrough(condition_channels, image_channels)

    @property
    def cv_ensemble(self):
        return self.base_discriminator.cv_ensemble

    def _init_as_target_passthrough(self, condition_channels, image_channels):
        with torch.no_grad():
            self.condition_adapter.weight.zero_()
            self.condition_adapter.bias.zero_()
            for channel in range(image_channels):
                self.condition_adapter.weight[channel, condition_channels + channel, 0, 0] = 1.0

    def forward(self, images, *args, **kwargs):
        if images.shape[1] != self.condition_adapter.in_channels:
            raise ValueError(
                f"Conditional discriminator expected {self.condition_adapter.in_channels} channels, "
                f"got {images.shape[1]}."
            )
        return self.base_discriminator(self.condition_adapter(images), *args, **kwargs)


def make_conditional_disc_input(x_src, x_img):
    return torch.cat([x_src, x_img], dim=1)


def make_disc_input(args, x_src, x_img):
    if args.disable_conditional_gan:
        return x_img
    return make_conditional_disc_input(x_src, x_img)


def normalize_to_255(img):
    # Visualization only: model tensors are in s-domain, where background is -1.
    img = display_image_tensor(img).detach().cpu().float()
    if img.numel() == 0:
        return torch.zeros_like(img, dtype=torch.uint8)
    
    non_zero_mask = img > -1.0 + 1e-6
    if non_zero_mask.sum() > 0:
        # 如果有非零值，对非零值进行分位数统计
        non_zero_vals = img[non_zero_mask]
        lo = torch.quantile(non_zero_vals, 0.01)
        hi = torch.quantile(non_zero_vals, 0.99)
    else:
        # 如果全是零，直接返回黑图
        return torch.zeros_like(img, dtype=torch.uint8)
    
    if torch.isclose(hi, lo):
        lo = non_zero_vals.min()
        hi = non_zero_vals.max()
    if torch.isclose(hi, lo):
        return torch.zeros_like(img, dtype=torch.uint8)
    
    img = img.clamp(lo, hi)
    img = (img - lo) / (hi - lo + 1e-8) * 255.0
    return img.clamp(0, 255).to(torch.uint8)


def display_image_tensor(img):
    if img.ndim == 3 and img.shape[0] not in (1, 3):
        slice_channels = img.shape[0] // 2 if img.shape[0] % 2 == 0 else img.shape[0]
        center_channel = slice_channels // 2
        return img[center_channel : center_channel + 1].repeat(3, 1, 1)
    if img.ndim == 3 and img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    return img


def unique_parameters(params):
    # 收集优化器参数时按对象 id 去重，避免同一个参数被重复加入参数组。
    unique = []
    seen = set()
    for param in params:
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        unique.append(param)
    return unique


def set_parameters_requires_grad(params, requires_grad):
    for param in params:
        param.requires_grad_(requires_grad)


def _npz_key(path):
    stem = Path(path).stem
    with np.load(path) as data:
        return stem if stem in data.files else data.files[0]


def _load_array_file(path, key=None):
    if path.endswith(".npz"):
        with np.load(path) as data:
            array_key = key or (Path(path).stem if Path(path).stem in data.files else data.files[0])
            return data[array_key]
    return np.load(path)


def get_volume_info(path):
    if path.endswith(".npz"):
        key = _npz_key(path)
        with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as handle:
            version = np.lib.format.read_magic(handle)
            shape, _, dtype = np.lib.format._read_array_header(
                handle,
                version,
                max_header_size=10000,
            )
        return tuple(shape), dtype, key

    arr = np.load(path, mmap_mode="r")
    return arr.shape, arr.dtype, None


def _slice_files(slice_dir):
    paths = {}
    for suffix in ("*.npz", "*.npy"):
        for path in Path(slice_dir).glob(suffix):
            if path.is_file():
                paths[path.name] = str(path)
    return paths


def _slice_manifest_cache_valid(payload, case_paths, use_mask_conditioning, mask_relpath, require_mask_conditioning):
    return (
        payload.get("version") == 1
        and payload.get("case_paths") == [str(p) for p in case_paths]
        and payload.get("use_mask_conditioning") == bool(use_mask_conditioning)
        and payload.get("mask_relpath") == str(mask_relpath)
        and payload.get("require_mask_conditioning") == bool(require_mask_conditioning)
    )


def _build_slice_manifest(case_paths, use_mask_conditioning=False, mask_relpath="mask", require_mask_conditioning=False):
    manifest = {}
    for case_path in case_paths:
        case_path = str(case_path)
        pred_dir = os.path.join(case_path, "pred")
        gt_dir = os.path.join(case_path, "gt")
        if not os.path.isdir(pred_dir) or not os.path.isdir(gt_dir):
            continue
        pred_files = _slice_files(pred_dir)
        gt_files = _slice_files(gt_dir)
        slice_names = sorted(set(pred_files) & set(gt_files))
        if not slice_names:
            continue

        mask_dir = None
        if use_mask_conditioning:
            candidate_mask_dir = Path(case_path) / mask_relpath
            if candidate_mask_dir.is_dir():
                mask_files = _slice_files(candidate_mask_dir)
                missing_masks = [name for name in slice_names if name not in mask_files]
                if missing_masks:
                    if require_mask_conditioning:
                        continue
                else:
                    mask_dir = str(candidate_mask_dir)
            elif require_mask_conditioning:
                continue

        first_slice_name = slice_names[0]
        first_coarse_shape, _, _ = get_volume_info(pred_files[first_slice_name])
        first_gt_shape, _, _ = get_volume_info(gt_files[first_slice_name])
        if len(first_coarse_shape) != 2 or len(first_gt_shape) != 2 or first_coarse_shape != first_gt_shape:
            continue

        if use_mask_conditioning and mask_dir is not None:
            first_mask_path = os.path.join(mask_dir, first_slice_name)
            first_mask_shape, _, _ = get_volume_info(first_mask_path)
            if len(first_mask_shape) != 2 or tuple(first_mask_shape) != tuple(first_coarse_shape):
                if require_mask_conditioning:
                    continue
                mask_dir = None

        manifest[case_path] = {
            "slice_names": slice_names,
            "pred_dir": pred_dir,
            "gt_dir": gt_dir,
            "mask_dir": mask_dir,
        }
    return manifest


def load_or_build_slice_manifest(
    cache_path,
    case_paths,
    use_mask_conditioning=False,
    mask_relpath="mask",
    require_mask_conditioning=False,
):
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if _slice_manifest_cache_valid(
                payload,
                case_paths,
                use_mask_conditioning,
                mask_relpath,
                require_mask_conditioning,
            ):
                return payload["cases"], True
        except Exception:
            pass

    manifest = _build_slice_manifest(
        case_paths,
        use_mask_conditioning=use_mask_conditioning,
        mask_relpath=mask_relpath,
        require_mask_conditioning=require_mask_conditioning,
    )
    payload = {
        "version": 1,
        "case_paths": [str(p) for p in case_paths],
        "use_mask_conditioning": bool(use_mask_conditioning),
        "mask_relpath": str(mask_relpath),
        "require_mask_conditioning": bool(require_mask_conditioning),
        "cases": manifest,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, cache_path)
    return manifest, False


def load_slice_manifest(cache_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    with cache_path.open("rb") as handle:
        return pickle.load(handle).get("cases", {})


def load_volume_slice(path, slice_idx, key=None, mmap_cache=None):
    if path.endswith(".npz"):
        with np.load(path) as data:
            array_key = key or (Path(path).stem if Path(path).stem in data.files else data.files[0])
            return data[array_key][:, :, slice_idx]

    if mmap_cache is not None:
        arr = mmap_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            mmap_cache[path] = arr
    else:
        arr = np.load(path, mmap_mode="r")
    return arr[:, :, slice_idx]


class MedicalCTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        case_paths,
        tokenizer,
        prompt,
        use_xray_conditioning=False,
        slice_context_radius=0,
        use_mask_conditioning=False,
        mask_relpath="mask/ct_file.mha",
        mask_key=None,
        require_mask_conditioning=False,
        volume_path_overrides=None,
        slice_case_manifest=None,
    ):
        self.case_paths = case_paths
        self.caption = prompt
        self.use_xray_conditioning = use_xray_conditioning
        self.slice_context_radius = int(slice_context_radius)
        if self.slice_context_radius < 0:
            raise ValueError("--slice_context_radius must be non-negative.")
        self.use_mask_conditioning = use_mask_conditioning
        self.mask_relpath = mask_relpath
        self.mask_key = mask_key
        self.require_mask_conditioning = require_mask_conditioning
        self.input_ids = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        self.index_map = []
        self._volume_info = {}
        self._slice_case_info = {}
        self._mmap_cache = {}
        self.volume_path_overrides = volume_path_overrides or {}
        self.slice_case_manifest = slice_case_manifest or {}

        for case_path in self.case_paths:
            case_name = os.path.basename(case_path.rstrip(os.sep))
            case_conditioning = {}
            if self.use_xray_conditioning:
                xray_path1 = os.path.join(case_path, f"{case_name}_xray_1.pt")
                xray_path2 = os.path.join(case_path, f"{case_name}_xray_2.pt")
                if not os.path.exists(xray_path1) or not os.path.exists(xray_path2):
                    continue
                xray_feat1 = torch.load(xray_path1, map_location="cpu").float()
                xray_feat2 = torch.load(xray_path2, map_location="cpu").float()
                if tuple(xray_feat1.shape) != (1, 768) or tuple(xray_feat2.shape) != (1, 768):
                    raise ValueError(
                        f"Expected RAD-DINO CLS features [1, 768] for {case_name}; "
                        f"got {tuple(xray_feat1.shape)} and {tuple(xray_feat2.shape)}. "
                        "Run scripts/extract_rad_dino_xray_features.py first."
                    )
                case_conditioning["xray_feat1"] = xray_feat1
                case_conditioning["xray_feat2"] = xray_feat2
            self._volume_info[case_path] = case_conditioning

            pred_dir = os.path.join(case_path, "pred")
            gt_dir = os.path.join(case_path, "gt")
            slice_manifest = self.slice_case_manifest.get(str(case_path))
            if slice_manifest is not None:
                slice_names = slice_manifest["slice_names"]
                if not slice_names:
                    continue
                pred_dir = slice_manifest["pred_dir"]
                gt_dir = slice_manifest["gt_dir"]
                mask_dir = slice_manifest.get("mask_dir")
                pred_files = {name: os.path.join(pred_dir, name) for name in slice_names}
                gt_files = {name: os.path.join(gt_dir, name) for name in slice_names}
                mask_files = (
                    {name: os.path.join(mask_dir, name) for name in slice_names}
                    if self.use_mask_conditioning and mask_dir is not None
                    else None
                )
                case_conditioning["mask_files"] = mask_files
                self._volume_info[case_path] = case_conditioning
                self._slice_case_info[case_path] = {
                    "slice_names": slice_names,
                    "pred_files": pred_files,
                    "gt_files": gt_files,
                    "mask_files": mask_files,
                }
                for slice_pos, slice_name in enumerate(slice_names):
                    self.index_map.append(
                        {
                            "case_path": case_path,
                            "mode": "slice",
                            "coarse_path": pred_files[slice_name],
                            "coarse_key": None,
                            "gt_path": gt_files[slice_name],
                            "gt_key": None,
                            "slice_name": slice_name,
                            "slice_pos": slice_pos,
                        }
                    )
                continue

            if os.path.isdir(pred_dir) and os.path.isdir(gt_dir):
                pred_files = _slice_files(pred_dir)
                gt_files = _slice_files(gt_dir)
                slice_names = sorted(set(pred_files) & set(gt_files))
                if not slice_names:
                    continue
                mask_files = None
                if self.use_mask_conditioning:
                    mask_dir = Path(case_path) / self.mask_relpath
                    if mask_dir.is_dir():
                        mask_files = _slice_files(mask_dir)
                    elif self.require_mask_conditioning:
                        continue

                first_slice_name = slice_names[0]
                first_coarse_shape, _, _ = get_volume_info(pred_files[first_slice_name])
                first_gt_shape, _, _ = get_volume_info(gt_files[first_slice_name])
                if len(first_coarse_shape) != 2 or len(first_gt_shape) != 2:
                    continue
                if first_coarse_shape != first_gt_shape:
                    continue
                if self.use_mask_conditioning and mask_files is not None:
                    missing_masks = [name for name in slice_names if name not in mask_files]
                    if missing_masks:
                        if self.require_mask_conditioning:
                            continue
                        mask_files = None
                    else:
                        first_mask_shape, _, _ = get_volume_info(mask_files[slice_names[0]])
                        if len(first_mask_shape) != 2 or tuple(first_mask_shape) != tuple(first_coarse_shape):
                            if self.require_mask_conditioning:
                                continue
                            mask_files = None
                case_conditioning["mask_files"] = mask_files
                self._volume_info[case_path] = case_conditioning
                self._slice_case_info[case_path] = {
                    "slice_names": slice_names,
                    "pred_files": pred_files,
                    "gt_files": gt_files,
                    "mask_files": mask_files,
                }
                for slice_pos, slice_name in enumerate(slice_names):
                    coarse_path = pred_files[slice_name]
                    gt_path = gt_files[slice_name]
                    self.index_map.append(
                        {
                            "case_path": case_path,
                            "mode": "slice",
                            "coarse_path": coarse_path,
                            "coarse_key": None,
                            "gt_path": gt_path,
                            "gt_key": None,
                            "slice_name": slice_name,
                            "slice_pos": slice_pos,
                        }
                    )
                continue

            override = self.volume_path_overrides.get(str(case_path), {})
            coarse_path = override.get("coarse_path")
            gt_path = override.get("gt_path")
            if coarse_path is None:
                coarse_candidates = [
                    os.path.join(case_path, "vol_pred.npz"),
                    os.path.join(case_path, "vol_pred.npy"),
                ]
                coarse_path = next((p for p in coarse_candidates if os.path.exists(p)), None)
            if gt_path is None:
                gt_candidates = [
                    os.path.join(case_path, "volume_gt.npz"),
                    os.path.join(case_path, "volume_gt.npy"),
                    os.path.join(case_path, "vol_gt.npz"),
                    os.path.join(case_path, "vol_gt.npy"),
                ]
                gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)
            if coarse_path is None or gt_path is None:
                continue

            coarse_shape, _, coarse_key = get_volume_info(coarse_path)
            gt_shape, _, gt_key = get_volume_info(gt_path)
            if len(coarse_shape) != 3 or len(gt_shape) != 3:
                continue
            if coarse_shape != gt_shape:
                continue
            mask_volume = self._load_case_mask(case_path, coarse_shape)
            if self.use_mask_conditioning and mask_volume is None and self.require_mask_conditioning:
                continue
            case_conditioning["mask_volume"] = mask_volume
            self._volume_info[case_path] = case_conditioning
            # 这里的 volume_gt.npy / vol_pred.npy 已经在数据生成阶段转成 XYZ 顺序，
            # 因此 axis 2 才是 axial 方向；索引范围也要跟着改成 shape[2]。
            for slice_idx in range(coarse_shape[2]):
                self.index_map.append(
                    {
                        "case_path": case_path,
                        "mode": "volume",
                        "coarse_path": coarse_path,
                        "coarse_key": coarse_key,
                        "gt_path": gt_path,
                        "gt_key": gt_key,
                        "slice_idx": slice_idx,
                        "total_slices": coarse_shape[2],
                    }
                )

    def _load_case_mask(self, case_path, target_shape, min_slices=None):
        if not self.use_mask_conditioning:
            return None
        mask_path = Path(case_path) / self.mask_relpath
        if mask_path.is_dir():
            return None
        if not mask_path.exists():
            return None
        mask_volume = load_mask_volume(mask_path, target_shape=target_shape, npz_key=self.mask_key)
        if min_slices is not None and mask_volume.shape[2] < min_slices:
            raise ValueError(
                f"Mask shape {tuple(mask_volume.shape)} has fewer axial slices than "
                f"the paired slice set ({min_slices}) for {case_path}."
            )
        return mask_volume

    def _zero_mask_stack(self, coarse_stack):
        return np.zeros_like(coarse_stack, dtype=np.float32)

    def _load_slice_mode_context(self, sample_info):
        case_path = sample_info["case_path"]
        slice_info = self._slice_case_info[case_path]
        slice_names = slice_info["slice_names"]
        indices = clamped_context_indices(
            sample_info["slice_pos"],
            len(slice_names),
            self.slice_context_radius,
        )
        coarse_stack = np.stack(
            [
                _load_array_file(slice_info["pred_files"][slice_names[i]])
                for i in indices
            ],
            axis=0,
        ).astype(np.float32)
        if self.use_mask_conditioning:
            mask_files = slice_info.get("mask_files")
            if mask_files is not None:
                mask_stack = np.stack(
                    [
                        (_load_array_file(mask_files[slice_names[i]], self.mask_key) > 0).astype(np.float32)
                        for i in indices
                    ],
                    axis=0,
                )
            else:
                mask_stack = self._zero_mask_stack(coarse_stack)
            coarse_stack = np.concatenate([coarse_stack, mask_stack], axis=0).astype(np.float32)
        return coarse_stack

    def _load_volume_mode_context(self, sample_info):
        indices = clamped_context_indices(
            sample_info["slice_idx"],
            sample_info["total_slices"],
            self.slice_context_radius,
        )
        coarse_stack = np.stack(
            [
                load_volume_slice(
                    sample_info["coarse_path"],
                    slice_idx,
                    sample_info["coarse_key"],
                    self._mmap_cache,
                )
                for slice_idx in indices
            ],
            axis=0,
        ).astype(np.float32)
        mask_volume = self._volume_info[sample_info["case_path"]].get("mask_volume")
        if self.use_mask_conditioning:
            mask_stack = (
                build_context_stack(mask_volume, sample_info["slice_idx"], self.slice_context_radius)
                if mask_volume is not None
                else self._zero_mask_stack(coarse_stack)
            )
            coarse_stack = np.concatenate([coarse_stack, mask_stack], axis=0).astype(np.float32)
        return coarse_stack

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        sample_info = self.index_map[idx]
        case_path = sample_info["case_path"]
        volume_info = self._volume_info[case_path]
        if sample_info["mode"] == "slice":
            coarse_stack = self._load_slice_mode_context(sample_info)
            coarse_slice = _load_array_file(sample_info["coarse_path"], sample_info["coarse_key"])
            gt_slice = _load_array_file(sample_info["gt_path"], sample_info["gt_key"])
        else:
            # axial slice：axis 2 对应 Z 方向，因此这里沿最后一个维度取切片。
            coarse_stack = self._load_volume_mode_context(sample_info)
            coarse_slice = load_volume_slice(
                sample_info["coarse_path"],
                sample_info["slice_idx"],
                sample_info["coarse_key"],
                self._mmap_cache,
            )
            gt_slice = load_volume_slice(
                sample_info["gt_path"],
                sample_info["slice_idx"],
                sample_info["gt_key"],
                self._mmap_cache,
            )

        # 这里需要 copy 一份，避免 mmap 读出的只读 numpy 数组直接转 tensor 时触发警告。
        coarse_v = torch.from_numpy(coarse_stack.copy()).float()
        center_coarse_v = torch.from_numpy(coarse_slice.copy()).float().unsqueeze(0)
        gt_v = torch.from_numpy(gt_slice.copy()).float().unsqueeze(0)

        coarse_tensor = volume_to_slicefixer(coarse_v)
        gt_tensor = volume_to_slicefixer(gt_v).repeat(3, 1, 1)

        sample = {
            "sample_index": torch.tensor(idx, dtype=torch.long),
            "conditioning_pixel_values": coarse_tensor,
            "output_pixel_values": gt_tensor,
            "caption": self.caption,
            "input_ids": self.input_ids,
            "coarse_v_max_before_clip": center_coarse_v.max(),
            "gt_v_max_before_clip": gt_v.max(),
            "coarse_v_upper_clipped_ratio": (center_coarse_v > 1.0).float().mean(),
            "gt_v_upper_clipped_ratio": (gt_v > 1.0).float().mean(),
        }
        if self.use_xray_conditioning:
            sample["xray_feat1"] = volume_info["xray_feat1"]
            sample["xray_feat2"] = volume_info["xray_feat2"]
        return sample


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    conditioning_in_channels = context_channel_count(
        args.slice_context_radius,
        use_mask_conditioning=args.use_mask_conditioning,
    )
    if accelerator.is_main_process:
        print(
            f"SliceFixer conditioning channels: {conditioning_in_channels} "
            f"(slice_context_radius={args.slice_context_radius}, "
            f"use_mask_conditioning={args.use_mask_conditioning})"
        )

    net_pix2pix = SliceFixer(
        pretrained_name=None,
        pretrained_path=args.slicefixer_pretrained_path,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        sd_turbo_path=args.pretrained_model_name_or_path,
        use_xray_conditioning=args.use_xray_conditioning,
        conditioning_in_channels=conditioning_in_channels,
    )
    net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        base_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc = (
            base_disc
            if args.disable_conditional_gan
            else ConditionalDiscriminator(base_disc, condition_channels=conditioning_in_channels)
        )
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    if args.lambda_clipsim > 0:
        net_clip, _ = clip.load("ViT-B/32", device="cuda")
        net_clip.requires_grad_(False)
        net_clip.eval()
    else:
        net_clip = None

    net_lpips.requires_grad_(False)

    # 只把需要训练的 LoRA / skip / conv 参数放进优化器，随后再做一次去重。
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())
    if args.use_xray_conditioning:
        layers_to_opt += list(net_pix2pix.fusion_adapter.parameters())
    layers_to_opt += list(net_pix2pix.input_adapter.parameters())
    # 防止 conv_in、skip_conv 或 LoRA 参数在不同收集路径中被重复加入。
    layers_to_opt = unique_parameters(layers_to_opt)

    # 【FP16 梯度修复】确保所有可训练参数保持 FP32，避免 GradScaler unscale 崩溃
    for name, param in net_pix2pix.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    info_path = args.info_json
    if not os.path.isabs(info_path):
        info_path = os.path.join(os.getcwd(), info_path)
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing info.json: {info_path}")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    train_ids = info.get(args.train_split, [])
    val_ids = info.get(args.val_split, [])
    if not val_ids:
        val_ids = info.get("test", [])

    def to_case_paths(case_ids):
        return [os.path.join(args.dataset_folder, cid) for cid in case_ids]

    train_case_paths = to_case_paths(train_ids)
    val_case_paths = to_case_paths(val_ids)

    manifest_dir = Path(args.output_dir) / "slice_manifests"
    train_manifest_path = manifest_dir / f"{args.train_split}.pkl"
    val_manifest_path = manifest_dir / f"{args.val_split}.pkl"
    if accelerator.is_main_process:
        start_time = time.time()
        train_manifest, train_manifest_cached = load_or_build_slice_manifest(
            train_manifest_path,
            train_case_paths,
            use_mask_conditioning=args.use_mask_conditioning,
            mask_relpath=args.mask_relpath,
            require_mask_conditioning=args.require_mask_conditioning,
        )
        val_manifest, val_manifest_cached = load_or_build_slice_manifest(
            val_manifest_path,
            val_case_paths,
            use_mask_conditioning=args.use_mask_conditioning,
            mask_relpath=args.mask_relpath,
            require_mask_conditioning=args.require_mask_conditioning,
        )
        print(
            "Slice manifest ready: "
            f"train_cases={len(train_manifest)} ({'cached' if train_manifest_cached else 'built'}), "
            f"val_cases={len(val_manifest)} ({'cached' if val_manifest_cached else 'built'}), "
            f"elapsed={time.time() - start_time:.1f}s"
        )
    accelerator.wait_for_everyone()
    train_slice_manifest = load_slice_manifest(train_manifest_path)
    val_slice_manifest = load_slice_manifest(val_manifest_path)

    prompt = "high quality medical CT slice, clear anatomical structures"
    tokenizer = net_pix2pix.tokenizer
    use_volume_cache = args.use_volume_cache and not args.disable_volume_cache
    if args.volume_cache_cases_per_block < 1:
        raise ValueError("--volume_cache_cases_per_block must be at least 1.")
    if args.num_samples_eval < 1:
        raise ValueError("--num_samples_eval must be at least 1.")
    volume_cache = None
    val_cache_stats = None
    if use_volume_cache:
        run_id = Path(args.output_dir).name
        volume_cache = TemporaryVolumeCache(args.volume_cache_dir, run_id)
        if accelerator.is_main_process:
            volume_cache.cleanup()
            atexit.register(volume_cache.cleanup)
            val_cache_stats = volume_cache.materialize_block("validation", val_case_paths)
        accelerator.wait_for_everyone()
        val_overrides = volume_cache.path_overrides("validation", val_case_paths)
    else:
        val_overrides = None

    def dataloader_kwargs(num_workers, allow_persistent_workers=True):
        kwargs = {
            "num_workers": num_workers,
            "pin_memory": args.pin_memory,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = args.persistent_workers and allow_persistent_workers
            if args.prefetch_factor is not None:
                kwargs["prefetch_factor"] = args.prefetch_factor
        return kwargs

    dataset_val = MedicalCTDataset(
        case_paths=val_case_paths,
        tokenizer=tokenizer,
        prompt=prompt,
        use_xray_conditioning=args.use_xray_conditioning,
        slice_context_radius=args.slice_context_radius,
        use_mask_conditioning=args.use_mask_conditioning,
        mask_relpath=args.mask_relpath,
        mask_key=args.mask_key,
        require_mask_conditioning=args.require_mask_conditioning,
        volume_path_overrides=val_overrides,
        slice_case_manifest=val_slice_manifest,
    )
    val_sample_count = min(args.num_samples_eval, len(dataset_val))
    if val_sample_count == 0:
        raise ValueError("Validation split contains no readable volume slices.")
    dataset_val = torch.utils.data.Subset(dataset_val, range(val_sample_count))
    dl_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        **dataloader_kwargs(args.val_dataloader_num_workers),
    )

    def build_train_dataloader(case_paths, volume_path_overrides=None):
        dataset = MedicalCTDataset(
            case_paths=case_paths,
            tokenizer=tokenizer,
            prompt=prompt,
            use_xray_conditioning=args.use_xray_conditioning,
            slice_context_radius=args.slice_context_radius,
            use_mask_conditioning=args.use_mask_conditioning,
            mask_relpath=args.mask_relpath,
            mask_key=args.mask_key,
            require_mask_conditioning=args.require_mask_conditioning,
            volume_path_overrides=volume_path_overrides,
            slice_case_manifest=train_slice_manifest,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            **dataloader_kwargs(
                args.dataloader_num_workers,
                allow_persistent_workers=not use_volume_cache,
            ),
        )
        return accelerator.prepare(dataloader)

    dl_train = None
    if not use_volume_cache:
        dl_train = build_train_dataloader(train_case_paths)

    # Prepare everything with our `accelerator`.
    net_pix2pix, net_disc, optimizer, optimizer_disc, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, lr_scheduler, lr_scheduler_disc
    )
    disc_trainable_params = [param for param in net_disc.parameters() if param.requires_grad]
    dl_val = accelerator.prepare(dl_val)
    net_lpips = accelerator.prepare(net_lpips)
    if net_clip is not None:
        net_clip = accelerator.prepare(net_clip)
    # renorm with image net statistics
    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move all networks to weight_dtype (only change non-trainable params for net_pix2pix and net_disc)
    # 【FP16 梯度修复】由于 Accelerate 的 unscale 操作在 fp16 混合精度下会崩溃，
    # 我们仅将不参与训练的参数(requires_grad=False)转为 fp16 节省显存，
    # 强制让所有参与训练的参数(requires_grad=True)保持为 float32。
    net_lpips.to(dtype=weight_dtype)
    if net_clip is not None:
        net_clip.to(dtype=weight_dtype)
    
    for name, param in net_pix2pix.named_parameters():
        if not param.requires_grad:
            param.data = param.data.to(weight_dtype)
            
    for name, param in net_disc.named_parameters():
        if not param.requires_grad:
            param.data = param.data.to(weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config["effective_batch_size"] = (
            accelerator.num_processes
            * args.train_batch_size
            * args.gradient_accumulation_steps
        )
        init_kwargs = {"wandb": {"name": args.tracker_run_name}} if args.tracker_run_name else {}
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=args.initial_global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # compute the reference stats for FID tracking
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)

        ref_stats = get_folder_features(os.path.join(args.dataset_folder, "test_B"), model=feat_model, num_workers=0, num=None,
                shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)

    constant_prompt_tokens = tokenizer(
        prompt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.cuda()
    constant_clip_tokens = (
        clip.tokenize([prompt], truncate=True).cuda()
        if args.lambda_clipsim > 0
        else None
    )

    effective_batch_size = (
        accelerator.num_processes
        * args.train_batch_size
        * args.gradient_accumulation_steps
    )
    if accelerator.is_main_process:
        print(f"Effective batch size: {effective_batch_size}")

    pending_cache_logs = {}

    def iter_epoch_batches(epoch):
        if not use_volume_cache:
            yield from dl_train
            return

        shuffled_cases = list(train_case_paths)
        rng = random.Random((args.seed or 0) + epoch)
        rng.shuffle(shuffled_cases)
        for block_index in range(0, len(shuffled_cases), args.volume_cache_cases_per_block):
            block_cases = shuffled_cases[
                block_index : block_index + args.volume_cache_cases_per_block
            ]
            block_name = f"train_epoch_{epoch}_block_{block_index // args.volume_cache_cases_per_block}"
            if accelerator.is_main_process:
                stats = volume_cache.materialize_block(block_name, block_cases)
                pending_cache_logs.update(
                    {
                        "cache/train_block_prepare_seconds": stats["seconds"],
                        "cache/block_bytes": stats["bytes"],
                        "cache/block_cases": stats["cases"],
                    }
                )
            accelerator.wait_for_everyone()
            overrides = volume_cache.path_overrides(block_name, block_cases)
            block_loader = build_train_dataloader(block_cases, overrides)
            yield from block_loader
            del block_loader
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                cleanup_started = time.perf_counter()
                volume_cache.cleanup_block(block_name)
                pending_cache_logs["cache/train_block_cleanup_seconds"] = (
                    time.perf_counter() - cleanup_started
                )
            accelerator.wait_for_everyone()

    # start the training loop
    global_step = args.initial_global_step
    train_started = time.perf_counter()
    last_optimizer_step_time = train_started
    epoch = 0
    while global_step < args.max_train_steps:
        for step, batch in enumerate(iter_epoch_batches(epoch)):
            l_acc = [net_pix2pix, net_disc]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"].cuda(non_blocking=True)
                x_tgt = batch["output_pixel_values"].cuda(non_blocking=True)
                xray_feat1 = batch["xray_feat1"].cuda(non_blocking=True) if args.use_xray_conditioning else None
                xray_feat2 = batch["xray_feat2"].cuda(non_blocking=True) if args.use_xray_conditioning else None
                B, C, H, W = x_src.shape
                prompt_tokens = constant_prompt_tokens.expand(B, -1)
                # forward pass
                x_tgt_pred = net_pix2pix(
                    x_src,
                    prompt_tokens=prompt_tokens,
                    xray_feat1=xray_feat1,
                    xray_feat2=xray_feat2,
                    deterministic=True,
                )
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = torch.tensor(0.0, device=x_tgt_pred.device)
                if args.lambda_lpips > 0:
                    loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips

                # CLIP alignment loss
                loss_clipsim = torch.tensor(0.0, device=x_tgt_pred.device)
                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(
                        x_tgt_pred_renorm,
                        (224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )
                    caption_tokens = constant_clip_tokens.expand(B, -1)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100) * args.lambda_clipsim

                loss_ssim = torch.tensor(0.0, device=x_tgt_pred.device)
                if args.lambda_ssim > 0:
                    ssim_loss, _ = ssim_loss_and_value(x_tgt_pred, x_tgt)
                    loss_ssim = ssim_loss * args.lambda_ssim

                gan_enabled = args.lambda_gan > 0 and global_step >= args.gan_warmup_steps

                # GAN loss for generator
                loss_gan = torch.tensor(0.0, device=x_tgt_pred.device)
                if gan_enabled:
                    if args.disable_conditional_gan:
                        set_parameters_requires_grad(disc_trainable_params, False)
                        try:
                            fake_for_g = make_disc_input(args, x_src, x_tgt_pred)
                            loss_gan = net_disc(fake_for_g, for_G=True).mean() * args.lambda_gan
                        finally:
                            set_parameters_requires_grad(disc_trainable_params, True)
                    else:
                        fake_for_g = make_disc_input(args, x_src, x_tgt_pred)
                        loss_gan = net_disc(fake_for_g, for_G=True).mean() * args.lambda_gan

                # Total generator loss (paper formula)
                loss = loss_l2 + loss_lpips + loss_clipsim + loss_gan + loss_ssim

                accelerator.backward(loss, retain_graph=False)
                
                # [修复 NaN 与 Unscale 报错]
                # 1. 重新启用梯度裁剪 (clip_grad_norm_)，避免模型在训练扩散和 GAN 时梯度爆炸产生 NaN 问题。
                # 2. 移除了之前错误屏蔽 fp16 裁剪的逻辑。
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Discriminator loss: fake image vs real image
                """
                # [修复 NaN 与 Unscale 报错]
                # 这是修复最核心死机Bug的地方。
                # 原逻辑分别对 real 和 fake 调用了 backward() 并执行了两次 step()，
                # 导致混合精度的 GradScaler 状态异常并直接报错或产生 NaN。
                # 正确的做法：真实和生成的 loss 相加，一个 batch 内只执行一次统一的 backward 和 step。
                # real vs fake
                lossD = torch.tensor(0.0, device=x_tgt_pred.device)
                if gan_enabled:
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    if args.disable_conditional_gan:
                        real_for_d = make_disc_input(args, x_src.detach(), x_tgt.detach())
                        lossD_real = net_disc(real_for_d, for_real=True).mean() * args.lambda_gan
                        accelerator.backward(lossD_real)

                        fake_for_d = make_disc_input(args, x_src.detach(), x_tgt_pred.detach())
                        lossD_fake = net_disc(fake_for_d, for_real=False).mean() * args.lambda_gan
                        accelerator.backward(lossD_fake)
                    else:
                        real_for_d = make_disc_input(args, x_src.detach(), x_tgt.detach())
                        fake_for_d = make_disc_input(args, x_src.detach(), x_tgt_pred.detach())
                        lossD_real = net_disc(real_for_d, for_real=True).mean() * args.lambda_gan
                        lossD_fake = net_disc(fake_for_d, for_real=False).mean() * args.lambda_gan
                        lossD = lossD_real + lossD_fake
                        accelerator.backward(lossD)
                    lossD = lossD_real.detach() + lossD_fake.detach()

                    # [修复 NaN 错误] 同样重新启用判别器的梯度裁剪，控制判别器的更新幅度
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)

                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {}
                if accelerator.is_main_process:
                    # log all the losses
                    logs["lossG"] = loss.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["gan_active"] = float(gan_enabled)
                    logs["input_v_max_before_clip"] = batch["coarse_v_max_before_clip"].max().item()
                    logs["target_v_max_before_clip"] = batch["gt_v_max_before_clip"].max().item()
                    logs["input_v_upper_clipped_ratio"] = batch["coarse_v_upper_clipped_ratio"].mean().item()
                    logs["target_v_upper_clipped_ratio"] = batch["gt_v_upper_clipped_ratio"].mean().item()
                    logs["pred_s_max"] = x_tgt_pred.max().detach().item()
                    logs["target_s_max"] = x_tgt.max().detach().item()
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    if args.lambda_ssim > 0:
                        logs["loss_ssim"] = loss_ssim.detach().item()
                    if args.lambda_gan > 0:
                        logs["loss_gan"] = loss_gan.detach().item()
                    current_time = time.perf_counter()
                    logs["config/effective_batch_size"] = effective_batch_size
                    logs["timing/train_step_seconds"] = current_time - last_optimizer_step_time
                    logs["throughput/train_steps_per_second"] = global_step / (
                        current_time - train_started
                    )
                    if val_cache_stats is not None:
                        logs["cache/val_prepare_seconds"] = val_cache_stats["seconds"]
                        val_cache_stats = None
                    logs.update(pending_cache_logs)
                    pending_cache_logs.clear()
                    progress_bar.set_postfix(**logs)

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        # 训练阶段把当前 batch 的 input / target / output 三个子图发到 wandb。
                        log_dict = {
                            "train/input": [wandb.Image(normalize_to_255(x_src[idx]), caption=f"input_{idx}") for idx in range(B)],
                            "train/target": [wandb.Image(normalize_to_255(x_tgt[idx]), caption=f"target(GT)_{idx}") for idx in range(B)],
                            "train/output": [wandb.Image(normalize_to_255(x_tgt_pred[idx]), caption=f"output_{idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                # Validation runs on every rank; rank 0 only aggregates and records.
                if global_step % args.eval_freq == 1:
                    validation_started = time.perf_counter()
                    fid_dir = os.path.join(args.output_dir, "eval", f"fid_{global_step}")
                    if accelerator.is_main_process and args.track_val_fid:
                        os.makedirs(fid_dir, exist_ok=True)
                    accelerator.wait_for_everyone()

                    local_metrics = []
                    val_preview = None
                    net_pix2pix.eval()
                    with torch.inference_mode():
                        for val_step, batch_val in enumerate(dl_val):
                            val_src = batch_val["conditioning_pixel_values"].cuda(non_blocking=True)
                            val_tgt = batch_val["output_pixel_values"].cuda(non_blocking=True)
                            val_xray_feat1 = batch_val["xray_feat1"].cuda(non_blocking=True) if args.use_xray_conditioning else None
                            val_xray_feat2 = batch_val["xray_feat2"].cuda(non_blocking=True) if args.use_xray_conditioning else None
                            val_prompt_tokens = constant_prompt_tokens.expand(val_src.shape[0], -1)
                            val_pred = net_pix2pix(
                                val_src,
                                prompt_tokens=val_prompt_tokens,
                                xray_feat1=val_xray_feat1,
                                xray_feat2=val_xray_feat2,
                                deterministic=True,
                            )
                            val_ssim_loss, val_ssim = ssim_loss_and_value(val_pred, val_tgt)
                            metric_values = [
                                F.mse_loss(val_pred.float(), val_tgt.float(), reduction="mean"),
                                net_lpips(val_pred.float(), val_tgt.float()).mean(),
                                val_ssim_loss,
                                val_ssim,
                            ]
                            if args.lambda_clipsim > 0:
                                pred_renorm = t_clip_renorm(val_pred * 0.5 + 0.5)
                                pred_renorm = F.interpolate(
                                    pred_renorm, (224, 224), mode="bilinear", align_corners=False
                                )
                                caption_tokens = constant_clip_tokens.expand(val_src.shape[0], -1)
                                clipsim, _ = net_clip(pred_renorm, caption_tokens)
                                metric_values.append(clipsim.mean())
                            local_metrics.append(torch.stack(metric_values))
                            sample_index = int(batch_val["sample_index"][0].item())
                            if val_preview is None or sample_index > val_preview[0]:
                                val_preview = (
                                    sample_index,
                                    display_image_tensor(val_src[0]),
                                    display_image_tensor(val_tgt[0]),
                                    display_image_tensor(val_pred[0]),
                                )
                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(val_pred[0].cpu() * 0.5 + 0.5)
                                outf = os.path.join(
                                    fid_dir,
                                    f"rank_{accelerator.process_index}_val_{val_step}.png",
                                )
                                output_pil.save(outf)

                    metrics = torch.stack(local_metrics)
                    metrics = accelerator.gather_for_metrics(metrics).float().cpu().numpy()
                    preview_indices = accelerator.gather(
                        torch.tensor([val_preview[0]], device=val_src.device)
                    ).cpu()
                    preview_images = accelerator.gather(
                        torch.stack(val_preview[1:]).unsqueeze(0)
                    ).cpu()
                    net_pix2pix.train()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        logs["val/l2"] = float(np.mean(metrics[:, 0]))
                        logs["val/lpips"] = float(np.mean(metrics[:, 1]))
                        logs["val/ssim_loss"] = float(np.mean(metrics[:, 2]))
                        logs["val/ssim"] = float(np.mean(metrics[:, 3]))
                        if args.lambda_clipsim > 0:
                            logs["val/clipsim"] = float(np.mean(metrics[:, 4]))
                        if args.track_val_fid:
                            curr_stats = get_folder_features(fid_dir, model=feat_model, num_workers=0, num=None,
                                    shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                    mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
                            logs["val/clean_fid"] = fid_from_feats(ref_stats, curr_stats)
                        preview = preview_images[int(torch.argmax(preview_indices).item())]
                        logs["val/input"] = [wandb.Image(normalize_to_255(preview[0]), caption="val_input")]
                        logs["val/target"] = [wandb.Image(normalize_to_255(preview[1]), caption="val_target")]
                        logs["val/output"] = [wandb.Image(normalize_to_255(preview[2]), caption="val_output")]
                        logs["timing/validation_seconds"] = (
                            time.perf_counter() - validation_started
                        )

                if accelerator.is_main_process:
                    accelerator.log(logs, step=global_step)
                    last_optimizer_step_time = time.perf_counter()
            if global_step >= args.max_train_steps:
                break
        epoch += 1
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and volume_cache is not None:
        volume_cache.cleanup()
    if accelerator.is_main_process and global_step > 0:
        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
        accelerator.unwrap_model(net_pix2pix).save_model(outf)
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
