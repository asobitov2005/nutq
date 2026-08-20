from __future__ import annotations

import pytest
import torch

from nutq_asr.modules import AcousticMemoryCompressor, GatedProjector


@pytest.mark.parametrize("mode", ["fixed", "soft"])
def test_compressor_shape_and_padding(mode: str) -> None:
    compressor = AcousticMemoryCompressor(mode, ratio=3, blank_token_id=4)
    states = torch.randn(2, 7, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]])
    logits = torch.randn(2, 7, 5, requires_grad=True)

    output = compressor(states, mask, logits)

    assert output.hidden_states.shape == (2, 3, 8)
    assert output.attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    output.hidden_states.sum().backward()
    assert states.grad is not None
    if mode == "soft":
        assert logits.grad is not None


def test_gated_projector_changes_hidden_size() -> None:
    projector = GatedProjector(8, 12, expansion=1, dropout=0.0)
    assert projector(torch.randn(2, 4, 8)).shape == (2, 4, 12)
