from __future__ import annotations

import numpy as np

from .mask_ops import get_2d_bbox


class LPIPSMetric:
    def __init__(
        self,
        net: str = "alex",
        device: str = "cuda",
        batch_size: int = 16,
        cpu_num_threads: int = 4,
    ):
        try:
            import torch
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS is enabled but dependencies are missing. Install torch and lpips, "
                "or set metric.lpips.enabled=false."
            ) from exc

        self.torch = torch
        # Hundreds of small ROI resizes are much slower with the host's large
        # default thread pool because scheduling dominates each operation.
        torch.set_num_threads(max(1, int(cpu_num_threads)))
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()
        self.batch_size = int(batch_size)

    @classmethod
    def from_config(cls, lpips_cfg: dict) -> "LPIPSMetric":
        return cls(
            net=lpips_cfg.get("backbone", "alex"),
            device=lpips_cfg.get("device", "cuda"),
            batch_size=lpips_cfg.get("batch_size", 16),
            cpu_num_threads=lpips_cfg.get("cpu_num_threads", 4),
        )

    def prepare_slice(self, img2d_norm: np.ndarray, bbox, resize_hw=(256, 256)):
        return prepare_lpips_slice(img2d_norm, bbox, resize_hw=resize_hw, torch_module=self.torch)

    def forward_batch(self, gt_list, pred_list) -> list[float]:
        torch = self.torch
        with torch.no_grad():
            gt = torch.stack(gt_list, dim=0).to(self.device)
            pred = torch.stack(pred_list, dim=0).to(self.device)
            dist = self.model(gt, pred)
        return dist.detach().flatten().cpu().numpy().astype(float).tolist()


def prepare_lpips_slice(img2d_norm: np.ndarray, bbox, resize_hw=(256, 256), torch_module=None):
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("torch is required to prepare LPIPS slices.") from exc

    y1, y2, x1, x2 = bbox
    crop = img2d_norm[y1:y2, x1:x2].astype(np.float32, copy=False)
    crop_t = torch_module.from_numpy(crop).float()[None, None, :, :]
    crop_t = torch_module.nn.functional.interpolate(
        crop_t,
        size=tuple(resize_hw),
        mode="bilinear",
        align_corners=False,
    )
    crop_t = crop_t.repeat(1, 3, 1, 1)
    crop_t = crop_t * 2.0 - 1.0
    return crop_t[0]


def compute_lpips_volume(
    gt_norm: np.ndarray,
    pred_norm: np.ndarray,
    mask: np.ndarray,
    lpips_runner,
    resize_hw=(256, 256),
    min_mask_pixels_per_slice: int = 100,
    bbox_padding: int = 8,
    mask_outside: bool = False,
    background_value: float = 0.0,
    normalization_range: tuple[float, float] | None = None,
) -> tuple[float, int]:
    if lpips_runner is None:
        raise RuntimeError("LPIPS runner is required when metric.lpips.enabled=true")

    gt_batch = []
    pred_batch = []
    scores: list[float] = []
    valid_slices = 0
    for z in range(gt_norm.shape[0]):
        m = mask[z]
        if int(m.sum()) < int(min_mask_pixels_per_slice):
            continue
        bbox = get_2d_bbox(m, padding=int(bbox_padding))
        if bbox is None:
            continue
        gt_slice = gt_norm[z]
        pred_slice = pred_norm[z]
        if mask_outside:
            y1, y2, x1, x2 = bbox
            m = m[y1:y2, x1:x2]
            gt_slice = gt_slice[y1:y2, x1:x2]
            pred_slice = pred_slice[y1:y2, x1:x2]
            bbox = (0, gt_slice.shape[0], 0, gt_slice.shape[1])
        if normalization_range is not None:
            norm_min, norm_max = normalization_range
            scale = float(norm_max) - float(norm_min)
            gt_slice = ((np.clip(gt_slice, norm_min, norm_max) - norm_min) / scale).astype(np.float32)
            pred_slice = ((np.clip(pred_slice, norm_min, norm_max) - norm_min) / scale).astype(np.float32)
        if mask_outside:
            gt_slice = np.where(m, gt_slice, float(background_value))
            pred_slice = np.where(m, pred_slice, float(background_value))
        gt_batch.append(lpips_runner.prepare_slice(gt_slice, bbox, resize_hw=resize_hw))
        pred_batch.append(lpips_runner.prepare_slice(pred_slice, bbox, resize_hw=resize_hw))
        valid_slices += 1
        if len(gt_batch) == lpips_runner.batch_size:
            scores.extend(lpips_runner.forward_batch(gt_batch, pred_batch))
            gt_batch.clear()
            pred_batch.clear()

    if gt_batch:
        scores.extend(lpips_runner.forward_batch(gt_batch, pred_batch))
    if not scores:
        return float("nan"), 0
    return float(np.mean(scores)), int(valid_slices)
