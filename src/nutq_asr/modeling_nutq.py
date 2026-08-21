"""PyTorch implementation of NUTQ-S, NUTQ-M, and NUTQ-X."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from .configuration_nutq import NUTQ_PRESETS, NutqConfig
from .modules import TDTHead


@dataclass
class NutqSeq2SeqLMOutput(Seq2SeqLMOutput):
    """Whisper output augmented with independently measurable ASR losses."""

    ar_loss: torch.Tensor | None = None
    ctc_loss: torch.Tensor | None = None
    tdt_loss: torch.Tensor | None = None
    ctc_logits: torch.Tensor | None = None


class NutqForConditionalGeneration(WhisperForConditionalGeneration):
    """Four-layer Whisper ASR with byte CTC and an optional NUTQ-X TDT path."""

    config_class = NutqConfig
    base_model_prefix = "model"

    def __init__(self, config: NutqConfig) -> None:
        super().__init__(config)
        self.ctc_head = nn.Linear(config.d_model, config.ctc_vocab_size)
        self.tdt_head = TDTHead(config) if config.tdt_enabled else None
        self.ctc_head.apply(self._init_weights)
        if self.tdt_head is not None:
            self.tdt_head.apply(self._init_weights)

    @staticmethod
    def _encoder_lengths(
        attention_mask: torch.Tensor | None,
        encoder_steps: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.full((batch_size,), encoder_steps, dtype=torch.long, device=device)
        # Whisper conv1 has stride one and conv2 has stride two.
        lengths = (attention_mask.to(device=device, dtype=torch.long).sum(dim=-1) + 1) // 2
        return lengths.clamp(max=encoder_steps)

    def _ctc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor | None,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor | None:
        if labels is None:
            return None
        label_mask = labels.ne(-100)
        targets = labels.masked_select(label_mask)
        target_lengths = label_mask.sum(dim=-1)
        return F.ctc_loss(
            logits.float().log_softmax(dim=-1).transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
            blank=self.config.ctc_blank_token_id,
            reduction="mean",
            zero_infinity=True,
        )

    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: BaseModelOutput | tuple[torch.Tensor, ...] | None = None,
        past_key_values: Any | None = None,
        decoder_inputs_embeds: torch.Tensor | None = None,
        decoder_position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        ctc_labels: torch.Tensor | None = None,
        tdt_labels: torch.Tensor | None = None,
        auxiliary_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> NutqSeq2SeqLMOutput | tuple[torch.Tensor, ...]:
        requested_return_dict = self.config.use_return_dict if return_dict is None else return_dict
        outputs = super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            labels=labels,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.encoder_last_hidden_state
        encoder_lengths = self._encoder_lengths(
            attention_mask,
            hidden_states.shape[1],
            hidden_states.shape[0],
            hidden_states.device,
        )
        ctc_logits = self.ctc_head(hidden_states) if ctc_labels is not None else None
        if auxiliary_mask is None:
            auxiliary_mask = torch.ones(
                hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device
            )
        else:
            auxiliary_mask = auxiliary_mask.to(device=hidden_states.device, dtype=torch.bool)
        ctc_loss = None
        if (
            ctc_logits is not None
            and ctc_labels is not None
            and self.config.ctc_loss_weight
            and auxiliary_mask.any()
        ):
            ctc_loss = self._ctc_loss(
                ctc_logits[auxiliary_mask],
                ctc_labels[auxiliary_mask],
                encoder_lengths[auxiliary_mask],
            )

        tdt_loss = None
        if tdt_labels is not None and self.config.tdt_loss_weight and auxiliary_mask.any():
            if self.tdt_head is None:
                raise ValueError("tdt_labels were provided to a NUTQ variant without a TDT head")
            selected_labels = tdt_labels[auxiliary_mask]
            tdt_output = self.tdt_head(
                hidden_states[auxiliary_mask],
                encoder_lengths[auxiliary_mask],
                selected_labels,
            )
            tdt_loss = self.tdt_head.loss(tdt_output, selected_labels)

        ar_loss = outputs.loss
        loss = None
        for value, weight in (
            (ar_loss, self.config.ar_loss_weight),
            (ctc_loss, self.config.ctc_loss_weight),
            (tdt_loss, self.config.tdt_loss_weight),
        ):
            if value is not None and weight:
                weighted = value * weight
                loss = weighted if loss is None else loss + weighted

        result = NutqSeq2SeqLMOutput(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=hidden_states,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            ar_loss=ar_loss,
            ctc_loss=ctc_loss,
            tdt_loss=tdt_loss,
            ctc_logits=ctc_logits,
        )
        return result if requested_return_dict else result.to_tuple()

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        inputs = super().prepare_inputs_for_generation(*args, **kwargs)  # type: ignore[misc]
        for key in ("ctc_labels", "tdt_labels", "auxiliary_mask"):
            inputs.pop(key, None)
        return inputs

    @classmethod
    def from_pretrained_variant(
        cls,
        variant: str = "s",
        backbone_name_or_path: str | None = None,
        **config_overrides: Any,
    ) -> NutqForConditionalGeneration:
        """Initialize a NUTQ preset from a pretrained multilingual Whisper checkpoint."""
        variant = variant.lower()
        if variant not in NUTQ_PRESETS:
            choices = ", ".join(sorted(NUTQ_PRESETS))
            raise ValueError(f"variant must be one of: {choices}")
        backbone = backbone_name_or_path or str(NUTQ_PRESETS[variant]["backbone"])
        source = WhisperForConditionalGeneration.from_pretrained(backbone, use_safetensors=True)
        values = source.config.to_dict()
        for key in ("model_type", "architectures", "_name_or_path", "decoder_layers"):
            values.pop(key, None)
        values.update(config_overrides)
        config = NutqConfig(variant=variant, backbone_name=backbone, **values)
        model = cls(config)

        model.model.encoder.load_state_dict(source.model.encoder.state_dict())
        model.model.decoder.embed_tokens.load_state_dict(
            source.model.decoder.embed_tokens.state_dict()
        )
        model.model.decoder.embed_positions.load_state_dict(
            source.model.decoder.embed_positions.state_dict()
        )
        model.model.decoder.layer_norm.load_state_dict(source.model.decoder.layer_norm.state_dict())
        model.proj_out.load_state_dict(source.proj_out.state_dict())

        source_layers = source.model.decoder.layers
        layer_indices = torch.linspace(0, len(source_layers) - 1, steps=4).round().long().tolist()
        for destination, source_index in zip(
            model.model.decoder.layers, layer_indices, strict=True
        ):
            destination.load_state_dict(source_layers[source_index].state_dict())
        model.generation_config = copy.deepcopy(source.generation_config)
        model.config.decoder_layers = 4
        return model

    def set_trainable(self, stage: str, encoder_unfreeze_layers: int = 4) -> None:
        """Select a reproducible training stage without semantic special cases."""
        choices = {"heads", "decoder", "top", "full"}
        if stage not in choices:
            raise ValueError(f"stage must be one of: {', '.join(sorted(choices))}")
        for parameter in self.parameters():
            parameter.requires_grad = stage == "full"
        if stage == "full":
            return
        for parameter in self.ctc_head.parameters():
            parameter.requires_grad = True
        if self.tdt_head is not None:
            for parameter in self.tdt_head.parameters():
                parameter.requires_grad = True
        if stage in {"decoder", "top"}:
            for parameter in self.model.decoder.parameters():
                parameter.requires_grad = True
            for parameter in self.proj_out.parameters():
                parameter.requires_grad = True
        if stage == "top":
            if not 1 <= encoder_unfreeze_layers <= len(self.model.encoder.layers):
                raise ValueError("encoder_unfreeze_layers is outside the encoder depth")
            for layer in self.model.encoder.layers[-encoder_unfreeze_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.model.encoder.layer_norm.parameters():
                parameter.requires_grad = True

    def freeze_pretrained_components(self) -> None:
        """Compatibility alias for the heads-only stage."""
        self.set_trainable("heads")

    def unfreeze_all(self) -> None:
        self.set_trainable("full")

    @torch.inference_mode()
    def ctc_greedy_decode(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[list[list[int]], list[float]]:
        encoder = self.model.encoder(input_features, attention_mask=attention_mask)
        hidden_states = encoder.last_hidden_state
        lengths = self._encoder_lengths(
            attention_mask,
            hidden_states.shape[1],
            hidden_states.shape[0],
            hidden_states.device,
        )
        probabilities = self.ctc_head(hidden_states).float().softmax(dim=-1)
        hypotheses: list[list[int]] = []
        confidences: list[float] = []
        for row, length in enumerate(lengths.tolist()):
            frame_probabilities, frame_tokens = probabilities[row, :length].max(dim=-1)
            sequence: list[int] = []
            emitted_confidence: list[float] = []
            previous = self.config.ctc_blank_token_id
            for token, confidence in zip(
                frame_tokens.tolist(), frame_probabilities.tolist(), strict=True
            ):
                if token != self.config.ctc_blank_token_id and token != previous:
                    sequence.append(token)
                    emitted_confidence.append(confidence)
                previous = token
            hypotheses.append(sequence)
            confidences.append(
                sum(emitted_confidence) / len(emitted_confidence) if emitted_confidence else 0.0
            )
        return hypotheses, confidences

    @torch.inference_mode()
    def tdt_greedy_decode(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[list[list[int]], list[float]]:
        if self.tdt_head is None:
            raise ValueError("TDT decoding is available only when config.tdt_enabled is true")
        encoder = self.model.encoder(input_features, attention_mask=attention_mask)
        hidden_states = encoder.last_hidden_state
        lengths = self._encoder_lengths(
            attention_mask,
            hidden_states.shape[1],
            hidden_states.shape[0],
            hidden_states.device,
        )
        return self.tdt_head.greedy_decode(hidden_states, lengths)
