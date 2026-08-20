"""Configuration for NUTQ models."""

from __future__ import annotations

from transformers import PretrainedConfig, T5Config, WhisperConfig


class NutqConfig(PretrainedConfig):
    """Configuration for a NUTQ encoder-decoder model.

    NUTQ composes a Whisper encoder configuration and a T5 decoder configuration. The
    nested configurations are serialized into the regular Transformers ``config.json``.
    """

    model_type = "nutq"
    is_composition = True

    def __init__(
        self,
        encoder_config: dict | WhisperConfig | None = None,
        decoder_config: dict | T5Config | None = None,
        ctc_vocab_size: int | None = None,
        ctc_blank_token_id: int | None = None,
        ctc_loss_weight: float = 0.3,
        compressor_mode: str = "soft",
        compression_ratio: int = 6,
        compressor_temperature: float = 0.08,
        compressor_salience_floor: float = 0.05,
        projector_expansion: int = 2,
        projector_dropout: float = 0.1,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        encoder = self._coerce_encoder_config(encoder_config)
        decoder = self._coerce_decoder_config(decoder_config)

        decoder.is_decoder = True
        decoder.is_encoder_decoder = False
        decoder.add_cross_attention = True
        # T5 stores encoder depth in ``num_layers`` and decoder depth separately. NUTQ
        # keeps only the decoder, so generation cache allocation must see decoder depth.
        decoder.num_layers = decoder.num_decoder_layers

        vocab_size = decoder.vocab_size
        blank_id = vocab_size if ctc_blank_token_id is None else ctc_blank_token_id
        ctc_size = max(vocab_size + 1, blank_id + 1) if ctc_vocab_size is None else ctc_vocab_size

        if compressor_mode not in {"none", "fixed", "soft"}:
            raise ValueError("compressor_mode must be one of: none, fixed, soft")
        if compression_ratio < 1:
            raise ValueError("compression_ratio must be at least 1")
        if not ctc_loss_weight >= 0.0:
            raise ValueError("ctc_loss_weight must be non-negative")
        if not 0 <= blank_id < ctc_size:
            raise ValueError("ctc_blank_token_id must be inside the CTC vocabulary")

        kwargs.setdefault("is_encoder_decoder", True)
        kwargs.setdefault("pad_token_id", decoder.pad_token_id)
        kwargs.setdefault("eos_token_id", decoder.eos_token_id)
        kwargs.setdefault(
            "decoder_start_token_id",
            getattr(decoder, "decoder_start_token_id", decoder.pad_token_id),
        )
        kwargs.setdefault("tie_word_embeddings", decoder.tie_word_embeddings)
        # Transformers 5 validates composite configs during ``super().__init__`` and calls
        # ``get_text_config``/``to_dict``. Nested and NUTQ-specific fields must therefore
        # already exist at that point.
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.ctc_vocab_size = ctc_size
        self.ctc_blank_token_id = blank_id
        self.ctc_loss_weight = ctc_loss_weight
        self.compressor_mode = compressor_mode
        self.compression_ratio = compression_ratio
        self.compressor_temperature = compressor_temperature
        self.compressor_salience_floor = compressor_salience_floor
        self.projector_expansion = projector_expansion
        self.projector_dropout = projector_dropout
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @staticmethod
    def _coerce_encoder_config(config: dict | WhisperConfig | None) -> WhisperConfig:
        if config is None:
            return WhisperConfig()
        if isinstance(config, WhisperConfig):
            return config
        config = dict(config)
        config.pop("model_type", None)
        return WhisperConfig(**config)

    @staticmethod
    def _coerce_decoder_config(config: dict | T5Config | None) -> T5Config:
        if config is None:
            return T5Config()
        if isinstance(config, T5Config):
            return config
        config = dict(config)
        config.pop("model_type", None)
        return T5Config(**config)

    def to_dict(self) -> dict:
        output = super().to_dict()
        output["encoder_config"] = self.encoder.to_dict()
        output["decoder_config"] = self.decoder.to_dict()
        # Use explicit nested names on disk and retain aliases in memory for Transformers.
        output.pop("encoder", None)
        output.pop("decoder", None)
        return output
