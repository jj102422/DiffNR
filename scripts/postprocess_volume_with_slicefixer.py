import argparse
import math
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "slicefixer"))

from SliceFixer import SliceFixer
from conditioning_utils import build_context_stack, context_channel_count
from intensity_utils import slicefixer_to_volume, volume_to_slicefixer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SliceFixer post-processing on a reconstructed 3D volume."
    )
    parser.add_argument("--checkpoint", required=True, help="SliceFixer .pkl checkpoint.")
    parser.add_argument("--input-volume", required=True, help="Input vol_pred .npy or .npz.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-source", required=True, help="Case directory containing xray feature .pt files.")
    parser.add_argument("--xray-feature-1", default=None)
    parser.add_argument("--xray-feature-2", default=None)
    parser.add_argument("--input-key", default="vol_pred")
    parser.add_argument("--gt-volume", default=None, help="Optional GT volume .npy or .npz for metrics/NIfTI.")
    parser.add_argument("--gt-key", default=None)
    parser.add_argument("--sd-turbo-path", default="/root/epfs/sd-turbo")
    parser.add_argument("--prompt", default="high quality medical CT slice, clear anatomical structures")
    parser.add_argument("--slice-context-radius", type=int, default=0)
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--preview-every", type=int, default=0)
    parser.add_argument("--skip-nifti", action="store_true", help="Skip writing pred.nii.gz and gt.nii.gz.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def load_array(path, key=None):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            array_key = key if key is not None and key in data.files else data.files[0]
            return data[array_key].astype(np.float32)
    return np.load(path).astype(np.float32)


def save_nifti(path, volume):
    image = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), affine=np.eye(4, dtype=np.float32))
    image.header.set_data_dtype(np.float32)
    nib.save(image, str(path))


def get_distributed_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return rank, local_rank, world_size


def resolve_xray_features(case_source, explicit_1=None, explicit_2=None):
    if explicit_1 and explicit_2:
        return Path(explicit_1), Path(explicit_2)

    case_source = Path(case_source)
    case_id = case_source.name
    candidates = [
        (case_source / f"{case_id}_xray_1.pt", case_source / f"{case_id}_xray_2.pt"),
        (case_source / f"{case_id}_xray_raw_1.pt", case_source / f"{case_id}_xray_raw_2.pt"),
    ]
    for path1, path2 in candidates:
        if path1.exists() and path2.exists():
            return path1, path2

    glob1 = sorted(case_source.glob("*_xray_1.pt")) + sorted(case_source.glob("*_xray_raw_1.pt"))
    glob2 = sorted(case_source.glob("*_xray_2.pt")) + sorted(case_source.glob("*_xray_raw_2.pt"))
    if glob1 and glob2:
        return glob1[0], glob2[0]

    raise FileNotFoundError(f"Missing xray feature pair under {case_source}")


def load_xray_feature(path, device):
    feat = torch.load(path, map_location=device).float()
    if tuple(feat.shape) == (1, 768):
        return feat.unsqueeze(0)
    if tuple(feat.shape) == (1, 1, 768):
        return feat
    raise ValueError(f"Expected xray feature shape (1, 768) or (1, 1, 768), got {tuple(feat.shape)} from {path}")


def to_uint8(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    lo = float(np.nanpercentile(arr, 1))
    hi = float(np.nanpercentile(arr, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def make_preview(input_v, output_v, gt_v, title):
    panels = [("input", input_v), ("output", output_v)]
    if gt_v is not None:
        panels.extend([("gt", gt_v), ("abs diff", np.abs(output_v - gt_v))])
    images = [Image.fromarray(to_uint8(arr)).convert("RGB") for _, arr in panels]
    w, h = images[0].size
    label_h = 24
    canvas = Image.new("RGB", (w * len(images), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, ((label, _), img) in enumerate(zip(panels, images)):
        canvas.paste(img, (i * w, label_h))
        draw.text((i * w + 6, 5), f"{title} {label}", fill=(0, 0, 0))
    return canvas


def metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(1.0 / math.sqrt(mse))
    return mse, mae, psnr


def write_metrics(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("slice,input_mse,output_mse,input_mae,output_mae,input_psnr,output_psnr\n")
        for row in rows:
            f.write(",".join([str(row[0]), *[f"{x:.8g}" for x in row[1:]]]) + "\n")


def read_metrics(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 7:
                rows.append((int(parts[0]), *[float(x) for x in parts[1:]]))
    return rows


def main():
    args = parse_args()
    rank, local_rank, world_size = get_distributed_info()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)

    input_volume = load_array(args.input_volume, args.input_key)
    if input_volume.ndim != 3:
        raise ValueError(f"Expected 3D input volume, got shape={input_volume.shape}")
    if args.max_slices is not None:
        input_volume = input_volume[:, :, : args.max_slices]

    gt_volume = load_array(args.gt_volume, args.gt_key) if args.gt_volume else None
    if gt_volume is not None:
        gt_volume = gt_volume[:, :, : input_volume.shape[2]]
        if gt_volume.shape != input_volume.shape:
            raise ValueError(f"GT shape {gt_volume.shape} does not match input shape {input_volume.shape}")

    out_dir = Path(args.output_dir)
    shard_dir = out_dir / "shards"
    preview_dir = out_dir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    xray_path1, xray_path2 = resolve_xray_features(args.case_source, args.xray_feature_1, args.xray_feature_2)
    conditioning_in_channels = context_channel_count(args.slice_context_radius, use_mask_conditioning=False)
    print(
        f"rank={rank}/{world_size} device={device} input_shape={input_volume.shape} "
        f"conditioning_channels={conditioning_in_channels} checkpoint={args.checkpoint}",
        flush=True,
    )

    model = SliceFixer(
        pretrained_path=args.checkpoint,
        sd_turbo_path=args.sd_turbo_path,
        conditioning_in_channels=conditioning_in_channels,
    ).to(device)
    model.set_eval()
    if args.fp16:
        model.half()
    xray_feat1 = load_xray_feature(xray_path1, device)
    xray_feat2 = load_xray_feature(xray_path2, device)

    z_count = input_volume.shape[2]
    slice_indices = [idx for idx in range(z_count) if idx % world_size == rank]
    outputs = []
    metric_rows = []
    started = time.time()
    with torch.inference_mode():
        progress = tqdm(slice_indices, desc=f"SliceFixer volume rank{rank}", disable=rank != 0)
        for z_idx in progress:
            pred_v = input_volume[:, :, z_idx]
            original_hw = pred_v.shape[-2:]
            conditioning_v = build_context_stack(input_volume, z_idx, args.slice_context_radius)
            pred_tensor = torch.from_numpy(conditioning_v).float().to(device).unsqueeze(0)
            pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
            pred_512 = F.interpolate(pred_tensor, size=(512, 512), mode="bilinear", align_corners=False)
            c_t = volume_to_slicefixer(pred_512)
            if args.fp16:
                c_t = c_t.half()
            out_s = model(
                c_t,
                prompt=args.prompt,
                xray_feat1=xray_feat1,
                xray_feat2=xray_feat2,
                deterministic=True,
            )
            out_v = slicefixer_to_volume(out_s).mean(dim=1, keepdim=True)
            out_v = F.interpolate(out_v, size=original_hw, mode="bilinear", align_corners=False)
            out_np = out_v.squeeze().float().cpu().numpy().astype(np.float32)
            outputs.append((z_idx, out_np))

            gt_v = gt_volume[:, :, z_idx] if gt_volume is not None else None
            if gt_v is not None:
                mse, mae, psnr = metrics(out_np, gt_v)
                in_mse, in_mae, in_psnr = metrics(pred_v, gt_v)
                metric_rows.append((z_idx, in_mse, mse, in_mae, mae, in_psnr, psnr))
            if args.preview_every > 0 and z_idx % args.preview_every == 0:
                preview = make_preview(pred_v, out_np, gt_v, f"slice {z_idx:04d}")
                preview.save(preview_dir / f"slice_{z_idx:04d}.png")

    shard_indices = np.array([idx for idx, _ in outputs], dtype=np.int64)
    shard_slices = (
        np.stack([out for _, out in outputs], axis=0).astype(np.float32)
        if outputs
        else np.empty((0, 0, 0), dtype=np.float32)
    )
    np.savez_compressed(shard_dir / f"shard_rank{rank:03d}.npz", indices=shard_indices, slices=shard_slices)
    if metric_rows:
        write_metrics(shard_dir / f"metrics_rank{rank:03d}.csv", metric_rows)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        ordered_slices = [None] * z_count
        for shard_path in sorted(shard_dir.glob("shard_rank*.npz")):
            with np.load(shard_path) as shard:
                for idx, out_slice in zip(shard["indices"], shard["slices"]):
                    ordered_slices[int(idx)] = out_slice.astype(np.float32)
        missing = [idx for idx, out_slice in enumerate(ordered_slices) if out_slice is None]
        if missing:
            raise RuntimeError(f"Missing reconstructed slices: {missing[:10]}... total={len(missing)}")
        volume = np.stack(ordered_slices, axis=-1).astype(np.float32)
        np.save(out_dir / "vol_pred_slicefixer.npy", volume)
        np.savez_compressed(out_dir / "vol_pred_slicefixer.npz", vol_pred=volume)
        np.save(out_dir / "input_vol_pred.npy", input_volume.astype(np.float32))
        if not args.skip_nifti:
            save_nifti(out_dir / "pred.nii.gz", volume)
        if gt_volume is not None and not args.skip_nifti:
            save_nifti(out_dir / "gt.nii.gz", gt_volume.astype(np.float32))

        all_rows = []
        for metrics_path in sorted(shard_dir.glob("metrics_rank*.csv")):
            all_rows.extend(read_metrics(metrics_path))
        if all_rows:
            all_rows.sort(key=lambda row: row[0])
            write_metrics(out_dir / "metrics.csv", all_rows)
            arr = np.array([row[1:] for row in all_rows], dtype=np.float64)
            print(
                "mean metrics: "
                f"input_mse={arr[:,0].mean():.6g} output_mse={arr[:,1].mean():.6g} "
                f"input_mae={arr[:,2].mean():.6g} output_mae={arr[:,3].mean():.6g} "
                f"input_psnr={arr[:,4].mean():.3f} output_psnr={arr[:,5].mean():.3f}",
                flush=True,
            )
        print(f"saved_volume={out_dir / 'vol_pred_slicefixer.npz'}", flush=True)

    elapsed = time.time() - started
    print(
        f"rank={rank} local_slices={len(outputs)} elapsed={elapsed:.1f}s "
        f"avg={elapsed / max(len(outputs), 1):.3f}s/slice",
        flush=True,
    )
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
