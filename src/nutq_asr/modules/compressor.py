"""CTC-guided acoustic memory compression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CompressorOutput:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor


class AcousticMemoryCompressor(nn.Module):
    """Compress encoder frames while preserving a differentiable path.

    ``fixed`` performs masked window pooling. ``soft`` places pooling kernels uniformly in
    cumulative non-blank CTC mass, concentrating memory around likely acoustic symbols.
    A salience floor prevents an untrained CTC head from erasing low-confidence evidence.
    """

    def __init__(
        self,
        mode: str,
        ratio: int,
        blank_token_id: int,
        temperature: float = 0.08,
        salience_floor: float = 0.05,
    ) -> None:
        super().__init__()
        if mode not in {"none", "fixed", "soft"}:
            raise ValueError(f"Unsupported compressor mode: {mode}")
        if ratio < 1:
            raise ValueError("ratio must be at least 1")
        self.mode = mode
        self.ratio = ratio
        self.blank_token_id = blank_token_id
        self.temperature = temperature
        self.salience_floor = salience_floor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        ctc_logits: torch.Tensor,
    ) -> CompressorOutput:
        batch_size, sequence_length, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length), dtype=torch.long, device=hidden_states.device
            )
        else:
            attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.long)

        if self.mode == "none" or self.ratio == 1:
            return CompressorOutput(hidden_states, attention_mask)
        if self.mode == "fixed":
            return self._fixed_pool(hidden_states, attention_mask)
        return self._soft_ctc_pool(hidden_states, attention_mask, ctc_logits)

    def _fixed_pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> CompressorOutput:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        output_length = (sequence_length + self.ratio - 1) // self.ratio
        padded_length = output_length * self.ratio
        padding = padded_length - sequence_length

        if padding:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, padding))
            attention_mask = torch.nn.functional.pad(attention_mask, (0, padding))

        states = hidden_states.view(batch_size, output_length, self.ratio, hidden_size)
        mask = attention_mask.view(batch_size, output_length, self.ratio)
        weights = mask.unsqueeze(-1).to(dtype=states.dtype)
        pooled = (states * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        pooled_mask = mask.any(dim=2).long()
        return CompressorOutput(pooled, pooled_mask)

    def _soft_ctc_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        ctc_logits: torch.Tensor,
    ) -> CompressorOutput:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        valid_lengths = attention_mask.sum(dim=-1)
        output_lengths = torch.div(
            valid_lengths + self.ratio - 1, self.ratio, rounding_mode="floor"
        ).clamp_min(1)
        max_output_length = int(output_lengths.max().item())

        probabilities = ctc_logits.float().softmax(dim=-1)
        salience = 1.0 - probabilities[..., self.blank_token_id]
        salience = (salience + self.salience_floor) * attention_mask
        cumulative = torch.cumsum(salience, dim=1)
        total = cumulative[:, -1:].clamp_min(torch.finfo(cumulative.dtype).eps)
        positions = (cumulative - 0.5 * salience) / total

        slot_ids = torch.arange(max_output_length, device=hidden_states.device)
        slot_mask = slot_ids.unsqueeze(0) < output_lengths.unsqueeze(1)
        centers = (slot_ids.to(positions.dtype) + 0.5).unsqueeze(0)
        centers = centers / output_lengths.unsqueeze(1).to(positions.dtype)

        distance = positions.unsqueeze(1) - centers.unsqueeze(2)
        widths = (self.temperature / output_lengths.clamp_min(1).to(positions.dtype)).view(
            batch_size, 1, 1
        )
        weights = torch.exp(-0.5 * (distance / widths.clamp_min(1e-4)).square())
        weights = weights * salience.unsqueeze(1) * slot_mask.unsqueeze(2)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        pooled = torch.einsum("bst,bth->bsh", weights.to(hidden_states.dtype), hidden_states)
        pooled = pooled * slot_mask.unsqueeze(-1).to(pooled.dtype)
        return CompressorOutput(
            pooled.view(batch_size, max_output_length, hidden_size), slot_mask.long()
        )
