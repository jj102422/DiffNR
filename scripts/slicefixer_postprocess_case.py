import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "slicefixer"))

from SliceFixer import SliceFixer
from conditioning_utils import (
    build_context_stack,
    build_context_stack_from_indices,
    clamped_context_indices,
    context_channel_count,
    load_mask_volume,
    resolve_mask_path,
)
from intensity_utils import slicefixer_to_volume, volume_to_slicefixer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SliceFixer post-processing on one case's pred/axial_*.npz slices."
    )
    parser.add_argument("--checkpoint", required=True, help="SliceFixer .pkl checkpoint.")
    parser.add_argument("--dataset-root", default="/root/epfs/data")
    parser.add_argument("--info-json", default="/root/epfs/DiffNR/info.json")
    parser.add_argument("--case-id", default=None, help="Case id. Defaults to the first id from --split.")
    parser.add_argument("--split", default="test", help="Split used when --case-id is omitted.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sd-turbo-path", default="/root/epfs/sd-turbo")
    parser.add_argument("--prompt", default="high quality medical CT slice, clear anatomical structures")
    parser.add_argument("--pred-key", default="vol_pred")
    parser.add_argument("--gt-key", default="ct")
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--save-slices", action="store_true", help="Save per-slice npz files.")
    parser.add_argument("--preview-every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true", help="Run SliceFixer inference in fp16 to reduce VRAM.")
    parser.add_argument(
        "--slice-context-radius",
        type=int,
        default=0,
        help="Number of neighboring axial slices to add on each side. Use 2 for five-slice 2.5D input.",
    )
    parser.add_argument(
        "--mask-path",
        default=None,
        help="Optional segmentation mask slice directory or volume file. Defaults to <case_dir>/mask when present.",
    )
    parser.add_argument("--mask-relpath", default="mask")
    parser.add_argument("--mask-key", default=None, help="NPZ key when --mask-path points to a .npz mask.")
    parser.add_argument("--require-mask", action="store_true", help="Fail if no mask is found.")
    return parser.parse_args()


def load_npz_key(path, preferred_key):
    with np.load(path) as data:
        key = preferred_key if preferred_key in data.files else data.files[0]
        return data[key].astype(np.float32)


def slice_files(slice_dir):
    paths = {}
    for suffix in ("*.npz", "*.npy"):
        for path in Path(slice_dir).glob(suffix):
            if path.is_file():
                paths[path.name] = path
    return paths


def load_xray_feature(path, device):
    feat = torch.load(path, map_location=device).float()
    if tuple(feat.shape) == (1, 768):
        return feat.unsqueeze(0)
    if tuple(feat.shape) == (1, 1, 768):
        return feat
    raise ValueError(f"Expected xray feature shape (1, 768) or (1, 1, 768), got {tuple(feat.shape)} from {path}")


def get_distributed_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return rank, local_rank, world_size


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
    panels = [
        ("input", input_v),
        ("output", output_v),
        ("gt", gt_v),
        ("abs diff", np.abs(output_v - gt_v)),
    ]
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


def write_metrics_csv(path, rows):
    with path.open("w", encoding="utf-8") as f:
        f.write("slice,input_mse,output_mse,input_mae,output_mae,input_psnr,output_psnr\n")
        for row in rows:
            f.write(",".join([row[0], *[f"{x:.8g}" for x in row[1:]]]) + "\n")


def read_metrics_csv(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 7:
                rows.append((parts[0], *[float(x) for x in parts[1:]]))
    return rows


def save_nifti(path, volume):
    image = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), affine=np.eye(4, dtype=np.float32))
    image.header.set_data_dtype(np.float32)
    nib.save(image, str(path))


def load_gt_volume(gt_dir, slice_names, gt_key):
    slices = []
    missing = []
    for slice_name in slice_names:
        gt_path = gt_dir / slice_name
        if not gt_path.exists():
            missing.append(slice_name)
            continue
        slices.append(load_npz_key(gt_path, gt_key))
    if missing:
        raise FileNotFoundError(f"Missing GT slices: {missing[:10]}... total={len(missing)}")
    return np.stack(slices, axis=-1).astype(np.float32)


def main():
    args = parse_args()
    rank, local_rank, world_size = get_distributed_info()
    dataset_root = Path(args.dataset_root)
    if args.case_id is None:
        info = json.loads(Path(args.info_json).read_text())
        case_ids = info.get(args.split, [])
        if not case_ids:
            raise ValueError(f"No case ids found for split {args.split!r} in {args.info_json}")
        case_id = case_ids[0]
    else:
        case_id = args.case_id

    case_dir = dataset_root / case_id
    pred_dir = case_dir / "pred"
    gt_dir = case_dir / "gt"
    xray_path1 = case_dir / f"{case_id}_xray_1.pt"
    xray_path2 = case_dir / f"{case_id}_xray_2.pt"
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Missing pred dir: {pred_dir}")
    if not xray_path1.exists() or not xray_path2.exists():
        raise FileNotFoundError(f"Missing RAD-DINO xray features for {case_id}")

    pred_paths = sorted(pred_dir.glob("axial_*.npz"))
    if args.max_slices is not None:
        pred_paths = pred_paths[: args.max_slices]
    if not pred_paths:
        raise ValueError(f"No axial_*.npz files found in {pred_dir}")

    out_dir = Path(args.output_dir) / case_id
    slice_out_dir = out_dir / "slices"
    preview_dir = out_dir / "preview"
    shard_dir = out_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    if args.save_slices:
        slice_out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)

    pred_items = [(idx, path) for idx, path in enumerate(pred_paths) if idx % world_size == rank]
    print(
        f"rank={rank}/{world_size} local_rank={local_rank} device={device} "
        f"slices={len(pred_items)}/{len(pred_paths)}",
        flush=True,
    )
    pred_volume = np.stack([load_npz_key(path, args.pred_key) for path in pred_paths], axis=-1).astype(np.float32)
    mask_path = resolve_mask_path(
        case_dir,
        explicit_mask_path=args.mask_path,
        mask_relpath=args.mask_relpath,
        require_mask=args.require_mask,
    )
    mask_volume = None
    mask_files = None
    if mask_path is not None:
        if Path(mask_path).is_dir():
            mask_files = slice_files(mask_path)
            missing_masks = [path.name for path in pred_paths if path.name not in mask_files]
            if missing_masks:
                raise FileNotFoundError(
                    f"Missing mask slices under {mask_path}: {missing_masks[:10]}... total={len(missing_masks)}"
                )
            print(f"Loaded segmentation mask slices: {mask_path}, slices={len(mask_files)}", flush=True)
        else:
            mask_volume = load_mask_volume(mask_path, target_shape=(*pred_volume.shape[:2], None), npz_key=args.mask_key)
            if mask_volume.shape[2] < pred_volume.shape[2]:
                raise ValueError(
                    f"Mask shape {tuple(mask_volume.shape)} has fewer axial slices than "
                    f"the prediction volume ({pred_volume.shape[2]})."
                )
            print(f"Loaded segmentation mask volume: {mask_path}, shape={mask_volume.shape}", flush=True)

    conditioning_in_channels = context_channel_count(
        args.slice_context_radius,
        use_mask_conditioning=mask_volume is not None or mask_files is not None,
    )
    print(
        f"Loading SliceFixer checkpoint: {args.checkpoint} "
        f"conditioning_channels={conditioning_in_channels} "
        f"slice_context_radius={args.slice_context_radius} "
        f"use_mask={mask_volume is not None or mask_files is not None}",
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
        print(f"rank={rank} using fp16 inference", flush=True)
    xray_feat1 = load_xray_feature(xray_path1, device)
    xray_feat2 = load_xray_feature(xray_path2, device)

    outputs = []
    metric_rows = []
    started = time.time()
    with torch.inference_mode():
        progress = tqdm(pred_items, desc=f"SliceFixer {case_id} rank{rank}", disable=rank != 0)
        for global_idx, pred_path in progress:
            pred_v = pred_volume[:, :, global_idx]
            original_hw = pred_v.shape[-2:]
            slice_stack = build_context_stack(pred_volume, global_idx, args.slice_context_radius)
            if mask_volume is None and mask_files is None:
                conditioning_v = slice_stack
            else:
                context_positions = clamped_context_indices(
                    global_idx,
                    len(pred_paths),
                    args.slice_context_radius,
                )
                if mask_files is not None:
                    mask_stack = np.stack(
                        [
                            (load_npz_key(mask_files[pred_paths[pos].name], args.mask_key) > 0).astype(np.float32)
                            for pos in context_positions
                        ],
                        axis=0,
                    )
                else:
                    mask_stack = build_context_stack_from_indices(mask_volume, context_positions)
                conditioning_v = np.concatenate([slice_stack, mask_stack], axis=0).astype(np.float32)
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
            outputs.append((global_idx, out_np))

            if args.save_slices:
                np.savez_compressed(slice_out_dir / pred_path.name, vol_pred=out_np)

            gt_path = gt_dir / pred_path.name
            if gt_path.exists():
                gt_v = load_npz_key(gt_path, args.gt_key)
                mse, mae, psnr = metrics(out_np, gt_v)
                in_mse, in_mae, in_psnr = metrics(pred_v, gt_v)
                metric_rows.append((pred_path.name, in_mse, mse, in_mae, mae, in_psnr, psnr))
                if args.preview_every > 0 and global_idx % args.preview_every == 0:
                    preview = make_preview(pred_v, out_np, gt_v, pred_path.stem)
                    preview.save(preview_dir / f"{pred_path.stem}.png")

    if outputs:
        shard_indices = np.array([idx for idx, _ in outputs], dtype=np.int64)
        shard_slices = np.stack([out for _, out in outputs], axis=0).astype(np.float32)
    else:
        shard_indices = np.empty((0,), dtype=np.int64)
        shard_slices = np.empty((0, 0, 0), dtype=np.float32)
    np.savez_compressed(shard_dir / f"shard_rank{rank:03d}.npz", indices=shard_indices, slices=shard_slices)

    if metric_rows:
        write_metrics_csv(shard_dir / f"metrics_rank{rank:03d}.csv", metric_rows)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        ordered_slices = [None] * len(pred_paths)
        for shard_path in sorted(shard_dir.glob("shard_rank*.npz")):
            with np.load(shard_path) as shard:
                for idx, out_slice in zip(shard["indices"], shard["slices"]):
                    ordered_slices[int(idx)] = out_slice.astype(np.float32)
        missing = [idx for idx, out_slice in enumerate(ordered_slices) if out_slice is None]
        if missing:
            raise RuntimeError(f"Missing reconstructed slices: {missing[:10]}... total={len(missing)}")
        volume = np.stack(ordered_slices, axis=-1).astype(np.float32)
        np.savez_compressed(out_dir / "vol_pred_slicefixer.npz", vol_pred=volume)
        save_nifti(out_dir / "pred.nii.gz", volume)
        gt_volume = load_gt_volume(gt_dir, [path.name for path in pred_paths], args.gt_key)
        save_nifti(out_dir / "gt.nii.gz", gt_volume)

        all_metric_rows = []
        for metrics_path in sorted(shard_dir.glob("metrics_rank*.csv")):
            all_metric_rows.extend(read_metrics_csv(metrics_path))
        if all_metric_rows:
            all_metric_rows.sort(key=lambda row: row[0])
            write_metrics_csv(out_dir / "metrics.csv", all_metric_rows)
            arr = np.array([row[1:] for row in all_metric_rows], dtype=np.float64)
            print(
                "mean metrics: "
                f"input_mse={arr[:,0].mean():.6g} output_mse={arr[:,1].mean():.6g} "
                f"input_mae={arr[:,2].mean():.6g} output_mae={arr[:,3].mean():.6g} "
                f"input_psnr={arr[:,4].mean():.3f} output_psnr={arr[:,5].mean():.3f}",
                flush=True,
            )

    elapsed = time.time() - started
    print(
        f"rank={rank} case_id={case_id} local_slices={len(outputs)} "
        f"elapsed={elapsed:.1f}s avg={elapsed / max(len(outputs), 1):.3f}s/slice",
        flush=True,
    )
    if rank == 0:
        print(f"saved_volume={out_dir / 'vol_pred_slicefixer.npz'}", flush=True)
        print(f"saved_pred_nii={out_dir / 'pred.nii.gz'}", flush=True)
        print(f"saved_gt_nii={out_dir / 'gt.nii.gz'}", flush=True)
        print(f"preview_dir={preview_dir}", flush=True)

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
