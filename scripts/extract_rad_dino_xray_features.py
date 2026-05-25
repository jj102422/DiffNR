import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rad_dino import RadDino
from tqdm import tqdm


FEATURE_SHAPE = (1, 768)


def projection_to_image(projection: np.ndarray) -> Image.Image:
    projection = np.asarray(projection, dtype=np.float32)
    low = float(projection.min())
    high = float(projection.max())
    if high > low:
        projection = (projection - low) / (high - low)
    else:
        projection = np.zeros_like(projection)
    pixels = np.rint(projection * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")


def load_projection(case_dir: Path, feature_path: Path, view_index: int, force: bool):
    if feature_path.exists() and not force:
        value = torch.load(feature_path, map_location="cpu")
        if tuple(value.shape) == FEATURE_SHAPE:
            return None
        if value.ndim == 2:
            return value.numpy()

    raw_path = case_dir / f"{case_dir.name}_xray_raw_{view_index + 1}.pt"
    if raw_path.exists():
        value = torch.load(raw_path, map_location="cpu")
        if value.ndim != 2:
            raise ValueError(f"Expected raw 2D projection at {raw_path}, got {tuple(value.shape)}")
        return value.numpy()

    projection_path = case_dir / "proj_train" / f"{view_index:04d}.npy"
    if not projection_path.exists():
        raise FileNotFoundError(f"Missing raw projection for {case_dir.name}: {projection_path}")
    return np.load(projection_path)


def atomic_torch_save(value: torch.Tensor, path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp_path)
    os.replace(tmp_path, path)


def main(args):
    device = torch.device(args.device)
    encoder = RadDino().eval().to(device)
    case_dirs = sorted(path for path in Path(args.data_root).iterdir() if path.is_dir())
    if args.limit is not None:
        case_dirs = case_dirs[: args.limit]

    converted = skipped = failed = 0
    for case_dir in tqdm(case_dirs, desc="RAD-DINO features"):
        case_name = case_dir.name
        try:
            output_paths = [
                case_dir / f"{case_name}_xray_1.pt",
                case_dir / f"{case_name}_xray_2.pt",
            ]
            raw_projections = [
                load_projection(case_dir, output_paths[0], 0, args.force),
                load_projection(case_dir, output_paths[1], 1, args.force),
            ]
            if raw_projections[0] is None and raw_projections[1] is None:
                skipped += 1
                continue
            if any(value is None for value in raw_projections):
                raw_projections = [
                    np.load(case_dir / "proj_train" / "0000.npy"),
                    np.load(case_dir / "proj_train" / "0001.npy"),
                ]

            for projection, output_path in zip(raw_projections, output_paths):
                image = projection_to_image(projection)
                feature = encoder.extract_cls_token(image).detach().cpu().float()
                if tuple(feature.shape) != FEATURE_SHAPE:
                    raise ValueError(f"Unexpected RAD-DINO feature shape {tuple(feature.shape)}")
                atomic_torch_save(feature, output_path)
            converted += 1
        except Exception as exc:
            failed += 1
            print(f"ERROR {case_name}: {exc}")

    print(f"cases={len(case_dirs)} converted={converted} skipped={skipped} failed={failed}")
    if failed:
        raise RuntimeError(f"RAD-DINO feature extraction failed for {failed} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace raw biplanar projections with RAD-DINO CLS features.")
    parser.add_argument("--data-root", default="/root/epfs/data")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N cases for validation.")
    parser.add_argument("--force", action="store_true", help="Re-extract features from proj_train even if features exist.")
    main(parser.parse_args())
