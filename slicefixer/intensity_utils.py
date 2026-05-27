import torch


def volume_to_slicefixer(volume_v: torch.Tensor) -> torch.Tensor:
    """Convert saved volume values to SliceFixer's bounded model domain."""
    return 2.0 * torch.clamp(volume_v, 0.0, 1.0) - 1.0


def slicefixer_to_volume(volume_s: torch.Tensor) -> torch.Tensor:
    """Convert bounded SliceFixer output back to normalized volume values."""
    return (torch.clamp(volume_s, -1.0, 1.0) + 1.0) / 2.0
