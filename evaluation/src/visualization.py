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
