"""Projection between independently pretrained speech and text components."""

from __future__ import annotations

import torch
from torch import nn


class GatedProjector(nn.Module):
    """A normalized SwiGLU-style residual-free dimension bridge."""

    def __init__(
        self, input_size: int, output_size: int, expansion: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        inner_size = output_size * expansion
        self.input_norm = nn.LayerNorm(input_size)
        self.value = nn.Linear(input_size, inner_size, bias=False)
        self.gate = nn.Linear(input_size, inner_size, bias=False)
        self.output = nn.Linear(inner_size, output_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(output_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(hidden_states)
        projected = self.value(normalized) * torch.nn.functional.silu(self.gate(normalized))
        return self.output_norm(self.dropout(self.output(projected)))
