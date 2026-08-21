"""Token-and-Duration Transducer head used by NUTQ-X."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers.loss.loss_tdt import tdt_loss

from ..configuration_nutq import NutqConfig


@dataclass
class TDTOutput:
    token_logits: torch.Tensor
    duration_logits: torch.Tensor
    logit_lengths: torch.Tensor


class TDTHead(nn.Module):
    """A byte TDT predictor/joiner with learned temporal subsampling.

    UTF-8 bytes keep the transducer vocabulary compact. The accuracy fallback uses the
    regular Whisper BPE decoder, so this fast path does not constrain AR output coverage.
    """

    def __init__(self, config: NutqConfig) -> None:
        super().__init__()
        hidden_size = config.tdt_hidden_size
        strides = []
        remaining = config.tdt_subsampling_factor
        input_size = config.d_model
        while remaining > 1:
            strides.append(
                nn.Sequential(
                    nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.LayerNorm(hidden_size),
                )
            )
            input_size = hidden_size
            remaining //= 2
        self.subsampling = nn.ModuleList(strides)
        self.encoder_projection = nn.Linear(input_size, hidden_size)
        self.embedding = nn.Embedding(config.tdt_bos_token_id + 1, hidden_size)
        self.predictor = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=config.tdt_num_layers,
            batch_first=True,
        )
        self.joint_norm = nn.LayerNorm(hidden_size)
        self.joint_head = nn.Linear(hidden_size, config.tdt_vocab_size + len(config.tdt_durations))
        self.config = config

    @staticmethod
    def _subsample_lengths(lengths: torch.Tensor, stages: int) -> torch.Tensor:
        for _ in range(stages):
            lengths = (lengths + 1) // 2
        return lengths

    def encode(
        self, encoder_hidden_states: torch.Tensor, encoder_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = encoder_hidden_states.transpose(1, 2)
        for layer in self.subsampling:
            hidden_states = layer[0](hidden_states).transpose(1, 2)  # type: ignore[index]
            hidden_states = layer[2](layer[1](hidden_states)).transpose(1, 2)  # type: ignore[index]
        hidden_states = self.encoder_projection(hidden_states.transpose(1, 2))
        lengths = self._subsample_lengths(encoder_lengths, len(self.subsampling))
        return hidden_states, lengths.clamp(max=hidden_states.shape[1])

    def _predict_training(self, labels: torch.Tensor) -> torch.Tensor:
        safe_labels = labels.masked_fill(labels.lt(0), 0)
        bos = labels.new_full((labels.shape[0], 1), self.config.tdt_bos_token_id)
        predictor_inputs = torch.cat([bos, safe_labels], dim=1)
        prediction, _ = self.predictor(self.embedding(predictor_inputs))
        return prediction

    def join(
        self, encoded: torch.Tensor, predicted: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint = torch.tanh(encoded[:, :, None, :] + predicted[:, None, :, :])
        logits = self.joint_head(self.joint_norm(joint))
        split = self.config.tdt_vocab_size
        return logits[..., :split], logits[..., split:]

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_lengths: torch.Tensor,
        labels: torch.Tensor,
    ) -> TDTOutput:
        encoded, logit_lengths = self.encode(encoder_hidden_states, encoder_lengths)
        predicted = self._predict_training(labels)
        token_logits, duration_logits = self.join(encoded, predicted)
        return TDTOutput(token_logits, duration_logits, logit_lengths)

    def loss(self, output: TDTOutput, labels: torch.Tensor) -> torch.Tensor:
        target_lengths = labels.ne(-100).sum(dim=-1)
        safe_labels = labels.masked_fill(labels.lt(0), 0)
        return tdt_loss(
            token_logits=output.token_logits,
            duration_logits=output.duration_logits,
            targets=safe_labels,
            logit_lengths=output.logit_lengths,
            target_lengths=target_lengths,
            blank_token_id=self.config.tdt_blank_token_id,
            durations=list(self.config.tdt_durations),
            sigma=self.config.tdt_sigma,
            reduction="mean",
        )

    @torch.inference_mode()
    def greedy_decode(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_lengths: torch.Tensor,
    ) -> tuple[list[list[int]], list[float]]:
        """Decode UTF-8 bytes and return mean emitted-token confidence per utterance."""
        encoded, lengths = self.encode(encoder_hidden_states, encoder_lengths)
        hypotheses: list[list[int]] = []
        confidences: list[float] = []
        durations = self.config.tdt_durations
        for row in range(encoded.shape[0]):
            state: tuple[torch.Tensor, torch.Tensor] | None = None
            token = torch.tensor(
                [[self.config.tdt_bos_token_id]], device=encoded.device, dtype=torch.long
            )
            predicted, state = self.predictor(self.embedding(token), state)
            prediction = predicted[:, -1:, :]
            hypothesis: list[int] = []
            token_confidences: list[float] = []
            time_index = 0
            symbols_at_time = 0
            while time_index < int(lengths[row]):
                token_logits, duration_logits = self.join(
                    encoded[row : row + 1, time_index : time_index + 1], prediction
                )
                token_log_probs = token_logits[0, 0, 0].float().log_softmax(dim=-1)
                duration_log_probs = duration_logits[0, 0, 0].float().log_softmax(dim=-1)
                token_id = int(token_log_probs.argmax())
                duration_index = int(duration_log_probs.argmax())
                duration = durations[duration_index]

                if token_id == self.config.tdt_blank_token_id and duration == 0:
                    duration_index = int(duration_log_probs[1:].argmax()) + 1
                    duration = durations[duration_index]
                elif token_id != self.config.tdt_blank_token_id:
                    hypothesis.append(token_id)
                    token_confidences.append(float(token_log_probs[token_id].exp()))
                    token = torch.tensor([[token_id]], device=encoded.device, dtype=torch.long)
                    predicted, state = self.predictor(self.embedding(token), state)
                    prediction = predicted[:, -1:, :]
                    symbols_at_time += 1

                if duration > 0:
                    time_index += duration
                    symbols_at_time = 0
                elif symbols_at_time >= self.config.tdt_max_symbols_per_step:
                    time_index += 1
                    symbols_at_time = 0

            hypotheses.append(hypothesis)
            confidences.append(
                sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
            )
        return hypotheses, confidences
