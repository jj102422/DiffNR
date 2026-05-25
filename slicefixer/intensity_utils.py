import torch


def volume_to_slicefixer(volume_v: torch.Tensor) -> torch.Tensor:
    """Convert normalized CT volume values to the SliceFixer model domain."""
    return 2.0 * torch.clamp_min(volume_v, 0.0) - 1.0


def slicefixer_to_volume(volume_s: torch.Tensor) -> torch.Tensor:
    """Convert SliceFixer output back to the normalized CT volume domain."""
    return (torch.clamp_min(volume_s, -1.0) + 1.0) / 2.0
