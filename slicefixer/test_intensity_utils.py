import torch

from slicefixer.intensity_utils import inverse_slicefixer_output


def test_inverse_slicefixer_output_returns_clipped_hu():
    pred = torch.tensor([-0.25, 0.0, 0.5, 1.0, 1.25])
    out = inverse_slicefixer_output(pred)
    assert torch.equal(out, torch.tensor([0.0, 0.0, 1500.0, 3000.0, 3000.0]))
