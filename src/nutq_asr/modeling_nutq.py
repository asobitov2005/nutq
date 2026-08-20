"""PyTorch implementation of NUTQ."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, T5ForConditionalGeneration, WhisperModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, ModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from .configuration_nutq import NutqConfig
from .modules import AcousticMemoryCompressor, GatedProjector


@dataclass
class NutqEncoderOutput(BaseModelOutput):
    ctc_logits: torch.Tensor | None = None
    ctc_attention_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


@dataclass
class NutqModelOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    past_key_values: Any | None = None
    decoder_hidden_states: tuple[torch.Tensor, ...] | None = None
    decoder_attentions: tuple[torch.Tensor, ...] | None = None
    cross_attentions: tuple[torch.Tensor, ...] | None = None
    encoder_last_hidden_state: torch.Tensor | None = None
    encoder_hidden_states: tuple[torch.Tensor, ...] | None = None
    encoder_attentions: tuple[torch.Tensor, ...] | None = None
    ctc_logits: torch.Tensor | None = None
    ctc_attention_mask: torch.Tensor | None = None
    encoder_attention_mask: torch.Tensor | None = None


@dataclass
class NutqSeq2SeqLMOutput(Seq2SeqLMOutput):
    ctc_loss: torch.Tensor | None = None
    decoder_loss: torch.Tensor | None = None
    ctc_logits: torch.Tensor | None = None


class NutqPreTrainedModel(PreTrainedModel):
    config_class = NutqConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class NutqAcousticEncoder(nn.Module):
    """Whisper encoder plus CTC head, compressor, and hidden-size bridge."""

    main_input_name = "input_features"

    def __init__(self, config: NutqConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = WhisperEncoder(config.encoder)
        self.ctc_head = nn.Linear(config.encoder.d_model, config.ctc_vocab_size)
        self.compressor = AcousticMemoryCompressor(
            mode=config.compressor_mode,
            ratio=config.compression_ratio,
            blank_token_id=config.ctc_blank_token_id,
            temperature=config.compressor_temperature,
            salience_floor=config.compressor_salience_floor,
        )
        self.projector = GatedProjector(
            config.encoder.d_model,
            config.decoder.d_model,
            expansion=config.projector_expansion,
            dropout=config.projector_dropout,
        )

    @staticmethod
    def _encoder_mask(
        input_mask: torch.Tensor | None, output_length: int, device: torch.device
    ) -> torch.Tensor | None:
        if input_mask is None:
            return None
        input_lengths = input_mask.to(device=device, dtype=torch.long).sum(dim=-1)
        # Whisper's first convolution keeps length; the second uses stride 2.
        output_lengths = ((input_lengths + 1) // 2).clamp(max=output_length)
        steps = torch.arange(output_length, device=device).unsqueeze(0)
        return (steps < output_lengths.unsqueeze(1)).long()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = True,
        **_: Any,
    ) -> NutqEncoderOutput | tuple:
        outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        frame_mask = self._encoder_mask(
            attention_mask, outputs.last_hidden_state.shape[1], outputs.last_hidden_state.device
        )
        ctc_logits = self.ctc_head(outputs.last_hidden_state)
        compressed = self.compressor(outputs.last_hidden_state, frame_mask, ctc_logits)
        projected = self.projector(compressed.hidden_states)

        result = NutqEncoderOutput(
            last_hidden_state=projected,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            ctc_logits=ctc_logits,
            ctc_attention_mask=frame_mask,
            attention_mask=compressed.attention_mask,
        )
        if return_dict is False:
            return tuple(value for value in result.values() if value is not None)
        return result


class NutqModel(NutqPreTrainedModel):
    def __init__(self, config: NutqConfig) -> None:
        super().__init__(config)
        self.acoustic_encoder = NutqAcousticEncoder(config)
        self.shared = nn.Embedding(config.decoder.vocab_size, config.decoder.d_model)
        decoder_config = copy.deepcopy(config.decoder)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.add_cross_attention = True
        decoder_config.num_layers = config.decoder.num_decoder_layers
        self.decoder = T5Stack(decoder_config)
        self.decoder.set_input_embeddings(self.shared)
        self.post_init()

    def get_encoder(self) -> NutqAcousticEncoder:
        return self.acoustic_encoder

    def get_decoder(self) -> T5Stack:
        return self.decoder

    def get_input_embeddings(self) -> nn.Embedding:
        return self.shared

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.shared = value
        self.decoder.set_input_embeddings(value)

    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: NutqEncoderOutput | tuple | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        decoder_inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> NutqModelOutput | tuple:
        del inputs_embeds
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if encoder_outputs is None:
            if input_features is None:
                raise ValueError("input_features or encoder_outputs must be provided")
            encoder_outputs = self.acoustic_encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        elif not isinstance(encoder_outputs, NutqEncoderOutput):
            encoder_outputs = NutqEncoderOutput(last_hidden_state=encoder_outputs[0])

        encoder_attention_mask = encoder_outputs.attention_mask
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        result = NutqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            ctc_logits=encoder_outputs.ctc_logits,
            ctc_attention_mask=encoder_outputs.ctc_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        if not return_dict:
            return tuple(value for value in result.values() if value is not None)
        return result


class NutqForConditionalGeneration(NutqPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {
        "lm_head.weight": "model.shared.weight",
        "model.decoder.embed_tokens.weight": "model.shared.weight",
    }

    def __init__(self, config: NutqConfig) -> None:
        super().__init__(config)
        self.model = NutqModel(config)
        self.lm_head = nn.Linear(config.decoder.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_encoder(self) -> NutqAcousticEncoder:
        return self.model.get_encoder()

    def get_decoder(self) -> T5Stack:
        return self.model.get_decoder()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.config.decoder_start_token_id is None:
            raise ValueError("decoder_start_token_id must be set")
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[..., 1:] = input_ids[..., :-1].clone()
        shifted[..., 0] = self.config.decoder_start_token_id
        shifted.masked_fill_(shifted == -100, self.config.pad_token_id)
        return shifted

    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: NutqEncoderOutput | tuple | None = None,
        past_key_values: Any | None = None,
        labels: torch.Tensor | None = None,
        ctc_labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> NutqSeq2SeqLMOutput | tuple:
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)
        # Decoder caches are for autoregressive generation. Keeping them enabled for a
        # full teacher-forced sequence wastes memory and can mix self/cross cache lengths.
        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        sequence_output = outputs.last_hidden_state
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.config.decoder.d_model**-0.5)
        logits = self.lm_head(sequence_output)

        decoder_loss = None
        if labels is not None:
            decoder_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        ctc_loss = self._compute_ctc_loss(outputs, ctc_labels)
        loss = decoder_loss
        if ctc_loss is not None:
            weighted_ctc = self.config.ctc_loss_weight * ctc_loss
            loss = weighted_ctc if loss is None else loss + weighted_ctc

        result = NutqSeq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            ctc_loss=ctc_loss,
            decoder_loss=decoder_loss,
            ctc_logits=outputs.ctc_logits,
        )
        if not return_dict:
            values = tuple(
                value for key, value in result.items() if key != "loss" and value is not None
            )
            return ((loss,) + values) if loss is not None else values
        return result

    def _compute_ctc_loss(
        self, outputs: NutqModelOutput, ctc_labels: torch.Tensor | None
    ) -> torch.Tensor | None:
        if ctc_labels is None or outputs.ctc_logits is None:
            return None
        label_mask = ctc_labels.ne(-100)
        target_lengths = label_mask.sum(dim=-1)
        targets = ctc_labels.masked_select(label_mask)
        if outputs.ctc_attention_mask is None:
            input_lengths = torch.full(
                (outputs.ctc_logits.shape[0],),
                outputs.ctc_logits.shape[1],
                dtype=torch.long,
                device=outputs.ctc_logits.device,
            )
        else:
            input_lengths = outputs.ctc_attention_mask.sum(dim=-1)
        return F.ctc_loss(
            outputs.ctc_logits.float().log_softmax(dim=-1).transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
            blank=self.config.ctc_blank_token_id,
            reduction="mean",
            zero_infinity=True,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: NutqEncoderOutput | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_features": None,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
        }

    @classmethod
    def from_pretrained_components(
        cls,
        encoder_name_or_path: str = "openai/whisper-small",
        decoder_name_or_path: str = "google/byt5-small",
        **config_overrides: Any,
    ) -> NutqForConditionalGeneration:
        """Initialize NUTQ from separately pretrained Whisper and ByT5 checkpoints."""
        # Component initialization accepts only safetensors. Besides avoiding arbitrary
        # pickle loading, this remains safe on PyTorch versions that block vulnerable
        # ``torch.load`` releases.
        whisper = WhisperModel.from_pretrained(encoder_name_or_path, use_safetensors=True)
        byt5 = T5ForConditionalGeneration.from_pretrained(
            decoder_name_or_path, use_safetensors=True
        )
        config = NutqConfig(
            encoder_config=whisper.config,
            decoder_config=byt5.config,
            **config_overrides,
        )
        model = cls(config)
        model.model.acoustic_encoder.encoder.load_state_dict(whisper.encoder.state_dict())
        model.model.shared.load_state_dict(byt5.shared.state_dict())
        model.model.decoder.load_state_dict(byt5.decoder.state_dict())
        model.lm_head.load_state_dict(byt5.lm_head.state_dict())
        model.tie_weights()
        return model

    def freeze_pretrained_components(self) -> None:
        """Freeze backbone blocks while adapting the bridge and cross-attention."""
        for parameter in self.model.acoustic_encoder.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.model.decoder.parameters():
            parameter.requires_grad = False
        for block in self.model.decoder.block:
            if len(block.layer) > 1:
                for parameter in block.layer[1].parameters():
                    parameter.requires_grad = True
        for parameter in self.model.decoder.final_layer_norm.parameters():
            parameter.requires_grad = True
        for parameter in self.model.shared.parameters():
            parameter.requires_grad = False
        for parameter in self.lm_head.parameters():
            parameter.requires_grad = False

    def unfreeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True
