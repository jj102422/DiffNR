from __future__ import annotations

from pathlib import Path

import numpy as np


def save_visual_check(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    num_slices: int = 5,
    ct_min: float = 0.0,
    ct_max: float = 2500.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slices = choose_mask_slices(mask, num_slices=num_slices)
    if not slices:
        raise ValueError("Cannot save visual check for empty mask")

    fig, axes = plt.subplots(len(slices), 4, figsize=(12, 3 * len(slices)), squeeze=False)
    for row_idx, z in enumerate(slices):
        gt_s = gt[z]
        pred_s = pred[z]
        err_s = np.abs(pred_s - gt_s)
        mask_s = mask[z]
        panels = [
            (gt_s, "GT", ct_min, ct_max),
            (pred_s, "pred", ct_min, ct_max),
            (err_s, "abs error", 0.0, max(float(err_s.max()), 1.0)),
            (gt_s, "mask overlay", ct_min, ct_max),
        ]
        for col_idx, (img, title, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx][col_idx]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            if col_idx == 3:
                overlay = np.ma.masked_where(~mask_s, mask_s)
                ax.imshow(overlay, cmap="autumn", alpha=0.35)
            ax.set_title(f"z={z} {title}")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_three_plane_check(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    ct_min: float = -1024.0,
    ct_max: float = 1000.0,
    title_extra: str = "",
    overlay_mask: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z, y, x = choose_center_indices(mask)
    err = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
    mask_panels = [mask[z], mask[:, y, :], mask[:, :, x]]
    rows = [
        ("GT", [gt[z], gt[:, y, :], gt[:, :, x]], ct_min, ct_max),
        ("pred", [pred[z], pred[:, y, :], pred[:, :, x]], ct_min, ct_max),
        ("abs error", [err[z], err[:, y, :], err[:, :, x]], 0.0, max(float(np.nanpercentile(err, 99)), 1.0)),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), squeeze=False)
    for col, title in enumerate([f"axial z={z}", f"coronal y={y}", f"sagittal x={x}"]):
        axes[0][col].set_title(title, fontsize=10)
    for row_idx, (label, panels, vmin, vmax) in enumerate(rows):
        for col_idx, img in enumerate(panels):
            ax = axes[row_idx][col_idx]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
            if overlay_mask:
                mask_panel = mask_panels[col_idx]
                if np.any(mask_panel) and not np.all(mask_panel):
                    ax.contour(mask_panel.astype(np.float32), levels=[0.5], colors=["#ff8c00"], linewidths=0.6)
            ax.axis("off")
            if col_idx == 0:
                ax.text(-0.08, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=9)
    if title_extra:
        fig.suptitle(title_extra, fontsize=10)
        fig.tight_layout(rect=(0.08, 0.02, 1, 0.96))
    else:
        fig.tight_layout(rect=(0.08, 0.02, 1, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def choose_mask_slices(mask: np.ndarray, num_slices: int = 5) -> list[int]:
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    valid = np.where(counts > 0)[0]
    if valid.size == 0:
        return []
    picks = {int(valid[len(valid) // 2]), int(valid[np.argmax(counts[valid])])}
    if num_slices > len(picks):
        positions = np.linspace(0, len(valid) - 1, num=min(num_slices, len(valid)), dtype=int)
        picks.update(int(valid[pos]) for pos in positions)
    return sorted(picks)[:num_slices]


def choose_center_indices(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    center = np.array(mask.shape) // 2 if coords.size == 0 else np.round(np.median(coords, axis=0)).astype(int)
    return tuple(max(0, min(int(idx), int(size) - 1)) for idx, size in zip(center, mask.shape))
