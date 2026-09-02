import torch

from slicefixer.conditioning_utils import (
    build_context_stack,
    clamped_context_indices,
    context_channel_count,
)
from slicefixer.spine_multitask import (
    SpineMaskHead,
    binary_mask_metrics,
    linear_warmup_weight,
    soft_dice_loss,
    spine_segmentation_loss,
)


def test_2p5d_ct_and_mask_conditioning_has_ten_channels():
    assert context_channel_count(2, use_mask_conditioning=True, mask_context_radius=2) == 10


def test_boundary_context_indices_are_clamped():
    assert clamped_context_indices(0, total_slices=4, slice_context_radius=2) == [0, 0, 0, 1, 2]
    assert clamped_context_indices(3, total_slices=4, slice_context_radius=2) == [1, 2, 3, 3, 3]
    volume = torch.arange(4).reshape(1, 1, 4).numpy()
    assert build_context_stack(volume, 0, 2)[:, 0, 0].tolist() == [0, 0, 0, 1, 2]


def test_spine_head_output_shape_and_state_round_trip():
    first = SpineMaskHead(16)
    features = torch.randn(2, 16, 32, 24)
    assert first(features).shape == (2, 1, 32, 24)
    state = first.state_dict()
    second = SpineMaskHead(16)
    second.load_state_dict(state)
    assert torch.equal(first.conv.weight, second.conv.weight)
    assert torch.equal(first.conv.bias, second.conv.bias)


def test_exact_mask_has_near_zero_dice_loss_and_perfect_metrics():
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = torch.where(target > 0, torch.tensor(30.0), torch.tensor(-30.0))
    assert soft_dice_loss(logits, target).item() < 1.0e-6
    metrics = binary_mask_metrics(logits, target)
    assert metrics["dice"].item() == 1.0
    assert metrics["precision"].item() == 1.0
    assert metrics["recall"].item() == 1.0


def test_empty_target_skips_dice_but_bce_has_gradient():
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    combined, dice, bce = spine_segmentation_loss(logits, target, bce_weight=0.5)
    assert dice.item() == 0.0
    assert bce.item() > 0.0
    combined.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() > 0


def test_spine_gradient_reaches_shared_decoder_path():
    shared_decoder = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1)
    head = SpineMaskHead(8)
    features = shared_decoder(torch.randn(2, 4, 16, 16))
    logits = head(features)
    target = torch.zeros_like(logits)
    target[:, :, 4:12, 5:11] = 1
    loss, _, _ = spine_segmentation_loss(logits, target)
    loss.backward()
    assert shared_decoder.weight.grad is not None
    assert torch.count_nonzero(shared_decoder.weight.grad).item() > 0


def test_spine_lambda_linear_warmup():
    assert linear_warmup_weight(0, 0.1, 10_000) == 0.0
    assert linear_warmup_weight(5_000, 0.1, 10_000) == 0.05
    assert linear_warmup_weight(10_000, 0.1, 10_000) == 0.1
    assert linear_warmup_weight(50_000, 0.1, 10_000) == 0.1
