#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch.distributed as dist
import sys
import numpy as np
import cv2
import yaml
import einops
import random
from datetime import timedelta
from torch import nn
from tqdm import tqdm
from argparse import ArgumentParser
from random import randint
from functools import partial
from PIL import Image
from torchvision.transforms import ToTensor
import time
import torchvision.transforms.functional as TF
from torchvision import transforms

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss, ssim3d, metric_vol_loss
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice

from slicefixer.SliceFixer import SliceFixer
from slicefixer.intensity_utils import volume_to_slicefixer, slicefixer_to_volume

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False
print(f"Fused SSIM available: {FUSED_SSIM_AVAILABLE}")

# ============== Distributed Training Utilities ==============
def init_distributed_mode():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30)
        )

    return rank, world_size, gpu

def is_main_process():
    """Check if current process is main process."""
    return int(os.environ.get("RANK", 0)) == 0

def print_rank0(*args, **kwargs):
    """Print only from main process."""
    if is_main_process():
        print(*args, **kwargs)

def synchronize():
    """Synchronize between all processes."""
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.barrier(device_ids=[torch.cuda.current_device()])

def get_rank():
    """Get rank of current process."""
    return int(os.environ.get("RANK", 0))

def get_world_size():
    """Get total number of processes."""
    return int(os.environ.get("WORLD_SIZE", 1))

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def all_reduce_gaussian_grads(gaussians):
    """SUM gradients so every rank applies the same Gaussian optimizer step."""
    if not is_distributed():
        return
    for param in [gaussians._xyz, gaussians._density, gaussians._scaling, gaussians._rotation]:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

def get_diffusion_cache_path(opt, model_path):
    if getattr(opt, "diffusion_cache_path", ""):
        return opt.diffusion_cache_path
    return osp.join(
        model_path,
        "diffusion_cache",
        f"enhanced_volume_iter{opt.diffusion_start_iter}.pt",
    )

def load_checkpoint_on_current_device(checkpoint):
    current_device = torch.cuda.current_device()
    return torch.load(
        checkpoint,
        map_location=lambda storage, loc: storage.cuda(current_device),
        weights_only=False,
    )

def move_gaussians_to_current_device(gaussians):
    device = torch.device("cuda", torch.cuda.current_device())
    for param in [gaussians._xyz, gaussians._density, gaussians._scaling, gaussians._rotation]:
        if param.device != device:
            param.data = param.data.to(device)
        if param.grad is not None and param.grad.device != device:
            param.grad.data = param.grad.data.to(device)

    for attr in ["max_radii2D", "xyz_gradient_accum", "denom"]:
        value = getattr(gaussians, attr, None)
        if torch.is_tensor(value) and value.device != device:
            setattr(gaussians, attr, value.to(device))

    if gaussians.optimizer is not None:
        for state in gaussians.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.device != device:
                    state[key] = value.to(device)

@torch.no_grad()
def initialize_slicefixer(model_path=None, sd_turbo_path=None, use_fp16=False):
    try:
        if model_path and os.path.exists(model_path):
            print(f"Loading SliceFixer model from: {model_path}")
            model = SliceFixer(pretrained_path=model_path, sd_turbo_path=sd_turbo_path)
        else:
            raise ValueError("model_path is not provided or does not exist")

        model.set_eval()
        if use_fp16:
            model.half()
            print("Using FP16 precision")
        print("SliceFixer model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading SliceFixer model: {e}")
        return None

@torch.no_grad()
def enhance_slice_with_slicefixer(slicefixer, slice_data, prompt, xray_feat1, xray_feat2, use_fp16=False):
    try:
        original_hw = slice_data.shape[-2:]
        slice_v = torch.clamp(slice_data.float().cuda().unsqueeze(0), 0.0, 1.0)
        slice_v = F.interpolate(slice_v, size=(512, 512), mode="bilinear", align_corners=False)
        c_t = volume_to_slicefixer(slice_v).repeat(1, 3, 1, 1)
        if use_fp16:
            c_t = c_t.half()

        output_s = slicefixer(
            c_t,
            prompt=prompt,
            xray_feat1=xray_feat1,
            xray_feat2=xray_feat2
        )
        output_v = slicefixer_to_volume(output_s)
        output_v = torch.mean(output_v, dim=1, keepdim=True)
        return F.interpolate(output_v, size=original_hw, mode="bilinear", align_corners=False)

    except Exception as e:
        print(f"Error in SliceFixer enhancement: {e}")
        return torch.clamp(slice_data.float().cuda().unsqueeze(0), 0.0, 1.0)


# Extract slices from predicted CT volume during training
def extract_slices(vol_pred):
    #print(f"Volume shape: {vol_pred.shape}")
    #print(f"Global min: {vol_pred.min().item():.4f}, max: {vol_pred.max().item():.4f}")
    #print(f"Mean: {vol_pred.mean().item():.4f}, std: {vol_pred.std().item():.4f}")
    slices = [vol_pred[..., i][None] for i in range(vol_pred.shape[2])]
    # Optionally, print per-slice statistics
    #for i, s in enumerate(slices):
    #    s_min = s.min().item()
    #    s_max = s.max().item()
    #    print(f"Slice {i}: min={s_min:.4f}, max={s_max:.4f}")
    return slices


def normalize_for_slicefixer_domain(volume):
    return torch.clamp(volume.float(), 0.0, 1.0)

def generate_diffusion_enhanced_volume(
    gaussians,
    scanner_cfg,
    pipe,
    slicefixer,
    prompt,
    xray_feat1,
    xray_feat2,
    organ_type,
):
    print("Generating SliceFixer enhanced volume...")
    start_time = time.time()

    with torch.no_grad():
        vol_pred = query(
            gaussians,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"],
            pipe,
        )["vol"]

    slices = extract_slices(vol_pred)
    diffusion_enhanced_slices = []

    if organ_type == "Chest":
        print("Processing Chest CT with SliceFixer")
    elif organ_type == "Tooth":
        print("Processing Tooth CT with SliceFixer")

    for idx, slice_data in tqdm(enumerate(slices), total=len(slices), desc="SliceFixer enhancing slices"):
        enhanced_slice_tensor = enhance_slice_with_slicefixer(
            slicefixer,
            slice_data,
            prompt,
            xray_feat1=xray_feat1,
            xray_feat2=xray_feat2,
            use_fp16=False,
        )

        enhanced_slice_clean = enhanced_slice_tensor.squeeze(0)  # [1, H, W]
        diffusion_enhanced_slices.append(enhanced_slice_clean.cpu())

    diffusion_enhanced_volume = torch.stack(diffusion_enhanced_slices, dim=0)  # [N, 1, H, W]
    diffusion_enhanced_volume = diffusion_enhanced_volume.squeeze(1)  # [N, H, W]
    diffusion_enhanced_volume = diffusion_enhanced_volume.permute(1, 2, 0).contiguous()  # [H, W, N]

    processing_time = time.time() - start_time
    print(f"SliceFixer enhancement completed in {processing_time:.2f}s for {len(slices)} slices")
    print(f"Average time per slice: {processing_time/len(slices):.3f}s")
    return diffusion_enhanced_volume

def get_rank_z_bounds(n_z, rank, world_size):
    base = n_z // world_size
    remainder = n_z % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end

def query_z_slab(gaussians, scanner_cfg, pipe, z_start, z_end):
    n_voxel = torch.tensor(scanner_cfg["nVoxel"], dtype=torch.long)
    s_voxel = torch.tensor(scanner_cfg["sVoxel"], dtype=torch.float32)
    off_origin = torch.tensor(scanner_cfg["offOrigin"], dtype=torch.float32)
    d_voxel = s_voxel / n_voxel.float()

    slab_n_voxel = n_voxel.clone()
    slab_n_voxel[2] = z_end - z_start
    slab_s_voxel = s_voxel.clone()
    slab_s_voxel[2] = d_voxel[2] * slab_n_voxel[2].float()

    min_bbox = off_origin - s_voxel / 2
    slab_center = off_origin.clone()
    slab_center[2] = min_bbox[2] + (z_start + z_end) * 0.5 * d_voxel[2]

    return query(
        gaussians,
        slab_center,
        slab_n_voxel,
        slab_s_voxel,
        pipe,
    )["vol"]

def _fused_ssim_sum_for_axis(target, pred, axis):
    ssim_sum = torch.zeros((), device=pred.device, dtype=pred.dtype)
    count = torch.zeros((), device=pred.device, dtype=pred.dtype)
    n_slice = target.shape[axis]
    for i in range(n_slice):
        if axis == 0:
            target_slice = target[i, :, :]
            pred_slice = pred[i, :, :]
        elif axis == 1:
            target_slice = target[:, i, :]
            pred_slice = pred[:, i, :]
        elif axis == 2:
            target_slice = target[:, :, i]
            pred_slice = pred[:, :, i]
        else:
            raise NotImplementedError

        if target_slice.max() > 0:
            ssim_sum = ssim_sum + fused_ssim(pred_slice[None, None], target_slice[None, None])
            count = count + 1
    return ssim_sum, count

def slab_diffusion_ssim_loss(
    diffusion_enhanced_volume,
    vol_pred_for_diffusion,
    core_start,
    core_end,
    halo_start,
    lambda_diffusion_ssim,
):
    if not FUSED_SSIM_AVAILABLE:
        raise RuntimeError("diffusion_parallel_mode=slab requires fused_ssim for differentiable slab SSIM")

    local_core_start = core_start - halo_start
    local_core_end = core_end - halo_start
    target_slab = diffusion_enhanced_volume[:, :, halo_start : halo_start + vol_pred_for_diffusion.shape[2]]
    target_slab = target_slab.to(vol_pred_for_diffusion.device, non_blocking=True)

    axis_sums = []
    axis_counts = []
    for axis in [0, 1]:
        axis_sum, axis_count = _fused_ssim_sum_for_axis(
            target_slab,
            vol_pred_for_diffusion,
            axis,
        )
        axis_sums.append(axis_sum)
        axis_counts.append(axis_count.detach())

    target_core = target_slab[:, :, local_core_start:local_core_end]
    pred_core = vol_pred_for_diffusion[:, :, local_core_start:local_core_end]
    axis_sum, axis_count = _fused_ssim_sum_for_axis(target_core, pred_core, 2)
    axis_sums.append(axis_sum)
    axis_counts.append(axis_count.detach())

    global_counts = []
    for axis_count in axis_counts:
        global_count = axis_count.clone()
        if is_distributed():
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        global_counts.append(torch.clamp(global_count, min=1.0))

    local_ssim_part = torch.zeros((), device=vol_pred_for_diffusion.device, dtype=vol_pred_for_diffusion.dtype)
    for axis_sum, global_count in zip(axis_sums, global_counts):
        local_ssim_part = local_ssim_part + axis_sum / global_count
    local_ssim_part = local_ssim_part / 3.0

    detached_sums = torch.stack([axis_sum.detach() for axis_sum in axis_sums])
    detached_counts = torch.stack(axis_counts)
    if is_distributed():
        dist.all_reduce(detached_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(detached_counts, op=dist.ReduceOp.SUM)
    global_ssim = torch.mean(detached_sums / torch.clamp(detached_counts, min=1.0))

    return lambda_diffusion_ssim * (1.0 - local_ssim_part), global_ssim

def ssim3d(img1, img2):
    """Metrics for volume. img1 must be GT."""
    ssims = []
    for axis in [0, 1, 2]:
        results = []
        count = 0
        n_slice = img1.shape[axis]
        for i in range(n_slice):
            if axis == 0:
                slice1 = img1[i, :, :]
                slice2 = img2[i, :, :]
            elif axis == 1:
                slice1 = img1[:, i, :]
                slice2 = img2[:, i, :]
            elif axis == 2:
                slice1 = img1[:, :, i]
                slice2 = img2[:, :, i]
            else:
                raise NotImplementedError
            if slice1.max() > 0:
                result = fused_ssim(slice2[None, None], slice1[None, None])
                count += 1
            else:
                result = torch.tensor(0.0, device=img1.device)
            results.append(result)

        results = torch.stack(results)
        mean_results = torch.sum(results) / count
        ssims.append(mean_results)
    return torch.mean(torch.stack(ssims)), ssims

def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    model_path,  # slicefixer_path
    organ_type,
    sd_turbo_path,
    train_batch_size=1,
):
    first_iter = 0
    rank = get_rank()
    world_size = get_world_size()

    if is_main_process():
        print(f"Distributed Training: Rank={rank}, World_Size={world_size}, Batch_Size={train_batch_size}")



    # Set up dataset
    scene = Scene(dataset, shuffle=False)


    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None

    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world

    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    # Initialize gaussians
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = load_checkpoint_on_current_device(checkpoint)
        gaussians.restore(model_params, opt)
        move_gaussians_to_current_device(gaussians)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv: # 3D tv loss
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    use_diffusion = opt.lambda_diffusion_ssim > 0
    use_tiny_volume = opt.diffusion_tv_size > 0
    use_diffusion_l1 = opt.lambda_diffusion_l1 > 0
    slab_parallel = use_diffusion and opt.diffusion_parallel_mode == "slab"
    rank0_primary_loss_only = slab_parallel and world_size > 1
    slicefixer = None
    diffusion_cache_path = get_diffusion_cache_path(opt, scene.model_path)

    if slab_parallel and world_size < 2:
        raise ValueError("diffusion_parallel_mode=slab must be launched with torchrun and world_size >= 2.")
    if opt.diffusion_parallel_mode not in ["none", "slab"]:
        raise ValueError(f"Unsupported diffusion_parallel_mode: {opt.diffusion_parallel_mode}")

    if use_diffusion:
        print(f"Initializing SliceFixer model. lambda_diffusion_ssim: {opt.lambda_diffusion_ssim}, lambda_diffusion_l1: {opt.lambda_diffusion_l1}, tiny volume used: {use_tiny_volume}, parallel mode: {opt.diffusion_parallel_mode}")

        if not slab_parallel or is_main_process():
            # Initialize SliceFixer only on ranks that may generate the cache.
            slicefixer = initialize_slicefixer(
                model_path=model_path,
                sd_turbo_path=sd_turbo_path,
                use_fp16=False,
            )

            case_id = os.path.basename(dataset.source_path)
            xray_feat_path1 = os.path.join(dataset.source_path, f"{case_id}_xray_1.pt")
            xray_feat_path2 = os.path.join(dataset.source_path, f"{case_id}_xray_2.pt")
            xray_feat1 = torch.load(xray_feat_path1)
            xray_feat2 = torch.load(xray_feat_path2)
            if tuple(xray_feat1.shape) != (1, 768) or tuple(xray_feat2.shape) != (1, 768):
                raise ValueError(
                    f"Expected RAD-DINO CLS features [1, 768] for {case_id}; "
                    f"got {tuple(xray_feat1.shape)} and {tuple(xray_feat2.shape)}."
                )
            xray_feat1 = xray_feat1.float().unsqueeze(1).cuda()
            xray_feat2 = xray_feat2.float().unsqueeze(1).cuda()
            print(f"load xray_feat1 from {xray_feat_path1}, shape: {xray_feat1.shape}")
            print(f"load xray_feat2 from {xray_feat_path2}, shape: {xray_feat2.shape}")

        if slicefixer is None and (not slab_parallel or is_main_process()):
            print("Warning: Failed to load SliceFixer model, diffusion enhancement disabled")
            use_diffusion = False
            slab_parallel = False
            rank0_primary_loss_only = False

    if slab_parallel and world_size > 1 and opt.densify_until_iter > first_iter:
        raise ValueError("diffusion_parallel_mode=slab requires densification to be finished before distributed resume.")

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None

    if is_main_process():
        progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
        progress_bar.update(first_iter)
    else:
        progress_bar = None
    first_iter += 1

    # 用于存储增强后的体积数据
    diffusion_enhanced_volume = None
    if use_diffusion and first_iter > opt.diffusion_start_iter:
        if osp.exists(diffusion_cache_path):
            diffusion_enhanced_volume = torch.load(diffusion_cache_path, map_location="cpu")
            if not slab_parallel:
                diffusion_enhanced_volume = diffusion_enhanced_volume.cuda()
            print_rank0(f"Loaded SliceFixer enhanced volume cache from {diffusion_cache_path}, shape: {tuple(diffusion_enhanced_volume.shape)}")
        elif slab_parallel:
            if is_main_process():
                prompt = "high quality medical CT slice, clear anatomical structures"
                diffusion_enhanced_volume = generate_diffusion_enhanced_volume(
                    gaussians,
                    scanner_cfg,
                    pipe,
                    slicefixer,
                    prompt,
                    xray_feat1,
                    xray_feat2,
                    organ_type,
                )
                os.makedirs(osp.dirname(diffusion_cache_path), exist_ok=True)
                torch.save(diffusion_enhanced_volume, diffusion_cache_path)
                print(f"Saved SliceFixer enhanced volume cache to {diffusion_cache_path}")
            synchronize()
            if diffusion_enhanced_volume is None:
                diffusion_enhanced_volume = torch.load(diffusion_cache_path, map_location="cpu")

    if slab_parallel and is_main_process():
        z_start, z_end = get_rank_z_bounds(int(scanner_cfg["nVoxel"][2]), rank, world_size)
        print(f"Using slab diffusion parallel mode with world_size={world_size}; rank0 core z range: [{z_start}, {z_end})")

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        compute_primary_loss = not rank0_primary_loss_only or is_main_process()

        # Get batch of cameras for training
        viewpoint_cams = []
        batch_size = 0
        if compute_primary_loss:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()

            # Sample batch of viewpoints
            batch_size = min(train_batch_size, len(viewpoint_stack))
            for _ in range(batch_size):
                if len(viewpoint_stack) == 0:
                    viewpoint_stack = scene.getTrainCameras().copy()
                idx = randint(0, len(viewpoint_stack) - 1)
                viewpoint_cams.append(viewpoint_stack.pop(idx))

        # Render X-ray projections and compute loss
        loss = {"total": torch.zeros((), device="cuda")}
        last_viewspace_point_tensor = None
        last_visibility_filter = None

        for viewpoint_cam in viewpoint_cams:
            # Render X-ray projection
            render_pkg = render(viewpoint_cam, gaussians, pipe)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            # Compute loss
            gt_image = viewpoint_cam.original_image.cuda()
            render_loss = l1_loss(image, gt_image)
            loss["render"] = loss.get("render", 0.0) + render_loss / batch_size
            loss["total"] += render_loss / batch_size

            if opt.lambda_dssim > 0:
                if FUSED_SSIM_AVAILABLE:
                    loss_dssim = 1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                else:
                    loss_dssim = 1.0 - ssim(image, gt_image)
                loss["dssim"] = loss.get("dssim", 0.0) + loss_dssim / batch_size
                loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim / batch_size

            # Store visibility stats (will accumulate max_radii2D in no_grad block)
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            # Save the last viewspace tensor and visibility filter for densification stats
            last_viewspace_point_tensor = viewspace_point_tensor
            last_visibility_filter = visibility_filter

         # 3D TV loss
        if use_tv and compute_primary_loss:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        if use_diffusion:
            if iteration == opt.diffusion_start_iter:
                if not slab_parallel or is_main_process():
                    prompt = "high quality medical CT slice, clear anatomical structures"
                    diffusion_enhanced_volume = generate_diffusion_enhanced_volume(
                        gaussians,
                        scanner_cfg,
                        pipe,
                        slicefixer,
                        prompt,
                        xray_feat1,
                        xray_feat2,
                        organ_type,
                    )

                    os.makedirs(osp.dirname(diffusion_cache_path), exist_ok=True)
                    torch.save(diffusion_enhanced_volume, diffusion_cache_path)
                    if not slab_parallel:
                        diffusion_enhanced_volume = diffusion_enhanced_volume.cuda()

                    print(f"Saved SliceFixer enhanced volume cache to {diffusion_cache_path}")

                if slab_parallel:
                    synchronize()
                    if diffusion_enhanced_volume is None:
                        diffusion_enhanced_volume = torch.load(diffusion_cache_path, map_location="cpu")

            if iteration > opt.diffusion_start_iter and iteration % 10 == 0 and diffusion_enhanced_volume is not None:
                if slab_parallel:
                    n_z = int(scanner_cfg["nVoxel"][2])
                    core_start, core_end = get_rank_z_bounds(n_z, rank, world_size)
                    halo_start = max(0, core_start - opt.diffusion_slab_halo)
                    halo_end = min(n_z, core_end + opt.diffusion_slab_halo)
                    vol_pred = query_z_slab(
                        gaussians,
                        scanner_cfg,
                        pipe,
                        halo_start,
                        halo_end,
                    )
                    vol_pred_for_diffusion = normalize_for_slicefixer_domain(vol_pred)
                    diffusion_loss, diffusion_ssim_global = slab_diffusion_ssim_loss(
                        diffusion_enhanced_volume,
                        vol_pred_for_diffusion,
                        core_start,
                        core_end,
                        halo_start,
                        opt.lambda_diffusion_ssim,
                    )
                    loss["diffusion_ssim"] = diffusion_loss
                    loss["diffusion_ssim_global"] = opt.lambda_diffusion_ssim * (1.0 - diffusion_ssim_global.detach())
                    loss["total"] = loss["total"] + loss["diffusion_ssim"]
                elif not use_tiny_volume:
                    vol_pred = query(
                        gaussians,
                        scanner_cfg["offOrigin"],
                        scanner_cfg["nVoxel"],
                        scanner_cfg["sVoxel"],
                        pipe,
                    )["vol"]
                    vol_pred_for_diffusion = normalize_for_slicefixer_domain(vol_pred)

                    if opt.lambda_diffusion_ssim > 0:

                        if FUSED_SSIM_AVAILABLE:
                            diffusion_ssim, _ = ssim3d(diffusion_enhanced_volume, vol_pred_for_diffusion)
                        else:
                            diffusion_ssim, _ = metric_vol_loss(diffusion_enhanced_volume, vol_pred_for_diffusion, "ssim")
                        diffusion_ssim_loss = 1.0 - diffusion_ssim
                        loss["diffusion_ssim"] = opt.lambda_diffusion_ssim * diffusion_ssim_loss
                        loss["total"] = loss["total"] + loss["diffusion_ssim"]
                    else:
                        print("not using diffusion ssim loss")
                    if use_diffusion_l1:
                        diffusion_l1_loss = l1_loss(vol_pred_for_diffusion, diffusion_enhanced_volume)
                        loss["diffusion_l1"] = opt.lambda_diffusion_l1 * diffusion_l1_loss
                        loss["total"] = loss["total"] + loss["diffusion_l1"]
                else:
                    # tiny volume处理
                    diff_tv_vol_size = opt.diffusion_tv_size
                    diff_tv_vol_nVoxel = torch.tensor([diff_tv_vol_size, diff_tv_vol_size, diff_tv_vol_size])
                    diff_tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * diff_tv_vol_nVoxel

                    # Randomly get the tiny volume center
                    diff_tv_vol_center = (bbox[0] + diff_tv_vol_sVoxel / 2) + (bbox[1] - diff_tv_vol_sVoxel - bbox[0]) * torch.rand(3)

                    vol_pred = query(
                        gaussians,
                        diff_tv_vol_center,
                        diff_tv_vol_nVoxel,
                        diff_tv_vol_sVoxel,
                        pipe,
                    )["vol"]
                    vol_pred_for_diffusion = normalize_for_slicefixer_domain(vol_pred)

                    # 从完整增强体积中提取对应的小体积区域
                    min_bbox_tv = diff_tv_vol_center - diff_tv_vol_sVoxel / 2
                    max_bbox_tv = diff_tv_vol_center + diff_tv_vol_sVoxel / 2
                    min_bbox_full = torch.tensor(scanner_cfg["offOrigin"]) - torch.tensor(scanner_cfg["sVoxel"]) / 2
                    max_bbox_full = torch.tensor(scanner_cfg["offOrigin"]) + torch.tensor(scanner_cfg["sVoxel"]) / 2
                    full_range = max_bbox_full - min_bbox_full
                    min_idx = (min_bbox_tv - min_bbox_full) / full_range * torch.tensor(scanner_cfg["nVoxel"])
                    max_idx = (max_bbox_tv - min_bbox_full) / full_range * torch.tensor(scanner_cfg["nVoxel"])

                    # 确保索引在有效范围内
                    min_idx = torch.clamp(min_idx, 0, torch.tensor(scanner_cfg["nVoxel"]) - 1).int()
                    max_idx = torch.clamp(max_idx, min_idx + 1, torch.tensor(scanner_cfg["nVoxel"])).int()

                    try:
                        tv_diffusion_enhanced_tensor = diffusion_enhanced_volume[
                            min_idx[0]:max_idx[0],
                            min_idx[1]:max_idx[1],
                            min_idx[2]:max_idx[2]
                        ]

                        if FUSED_SSIM_AVAILABLE:
                            diffusion_ssim, _ = ssim3d(tv_diffusion_enhanced_tensor, vol_pred_for_diffusion)
                        else:
                            diffusion_ssim, _ = metric_vol_loss(tv_diffusion_enhanced_tensor, vol_pred_for_diffusion, "ssim")
                        loss_diffusion_ssim = 1.0 - diffusion_ssim
                        loss["diffusion_ssim"] = opt.lambda_diffusion_ssim * loss_diffusion_ssim
                        loss["total"] = loss["total"] + loss["diffusion_ssim"]

                        if use_diffusion_l1:
                            diffusion_l1_loss = l1_loss(vol_pred_for_diffusion, tv_diffusion_enhanced_tensor)
                            loss["diffusion_l1"] = opt.lambda_diffusion_l1 * diffusion_l1_loss
                            loss["total"] = loss["total"] + loss["diffusion_l1"]
                    except Exception as e:
                        print(f"Error in tiny volume diffusion loss calculation: {e}")

        if loss["total"].requires_grad:
            loss["total"].backward()
        if slab_parallel and world_size > 1:
            all_reduce_gaussian_grads(gaussians)

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control
            if last_viewspace_point_tensor is not None and last_visibility_filter is not None:
                gaussians.add_densification_stats(last_viewspace_point_tensor, last_visibility_filter)
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if is_main_process() and (iteration in saving_iterations or iteration == opt.iterations):
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints
            if is_main_process() and iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0 and is_main_process():
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss['total'].item():.1e}",
                            "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                        }
                    )
                    progress_bar.update(10)
            if iteration == opt.iterations and is_main_process():
                if progress_bar is not None:
                    progress_bar.close()

            # Logging
            if is_main_process():
                metrics = {}
                for l in loss:
                    metrics["loss_" + l] = loss[l].item()
                for param_group in gaussians.optimizer.param_groups:
                    metrics[f"lr_{param_group['name']}"] = param_group["lr"]
                training_report(
                    tb_writer,
                    iteration,
                    metrics,
                    iter_start.elapsed_time(iter_end),
                    testing_iterations,
                    scene,
                    lambda x, y: render(x, y, pipe),
                    queryfunc,
                )

def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        vol_pred = queryFunc(scene.gaussians)["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_two_slice(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 11_000, 12_000, 13_500, 15_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[13_500, 15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--slicefixer_model_path", type=str, default=None, help="Path to trained SliceFixer model")
    parser.add_argument("--organ_type", type=str, default="Chest")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training (number of viewpoints per iteration)")
    parser.add_argument(
        "--sd_turbo_path",
        type=str,
        default=None,
        help="Path or HF model id for SD-Turbo (fallback: env SD_TURBO_PATH, then stabilityai/sd-turbo)",
    )

    # Use parse_known_args to handle torch.distributed.launch's --local-rank parameter
    args, unknown_args = parser.parse_known_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize distributed training
    rank, world_size, gpu = init_distributed_mode()

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.cuda.set_device(gpu)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print_rank0(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer (only on rank 0)
    tb_writer = prepare_output_and_logger(args) if is_main_process() else None

    print_rank0("Optimizing " + str(args.slicefixer_model_path))
    print_rank0(f"Distributed Training Mode: rank={rank}, world_size={world_size}, gpu={gpu}, batch_size={args.train_batch_size}")

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.slicefixer_model_path,
        args.organ_type,
        args.sd_turbo_path,
        train_batch_size=args.train_batch_size,
    )

    # Clean up distributed training
    if world_size > 1:
        dist.destroy_process_group()

    # All done
    print_rank0("Training complete.")
