from __future__ import annotations

import torch

from nutq_asr.modules import TDTHead


def test_tdt_shapes_and_loss(tiny_x_config) -> None:
    head = TDTHead(tiny_x_config)
    states = torch.randn(2, 8, 16, requires_grad=True)
    lengths = torch.tensor([8, 6])
    labels = torch.tensor([[3, 4, 5], [6, 7, -100]])

    output = head(states, lengths, labels)
    loss = head.loss(output, labels)

    assert output.token_logits.shape == (2, 4, 4, 17)
    assert output.duration_logits.shape == (2, 4, 4, 3)
    assert torch.isfinite(loss)
    loss.backward()
    assert states.grad is not None
