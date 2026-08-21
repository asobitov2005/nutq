"""Configuration for the NUTQ model family."""

from __future__ import annotations

from typing import Any

from transformers import WhisperConfig

NUTQ_PRESETS: dict[str, dict[str, Any]] = {
    "s": {
        "backbone": "openai/whisper-small",
        "display_name": "NUTQ-S",
        "tdt_enabled": False,
    },
    "m": {
        "backbone": "openai/whisper-medium",
        "display_name": "NUTQ-M",
        "tdt_enabled": False,
    },
    "x": {
        "backbone": "openai/whisper-large-v3-turbo",
        "display_name": "NUTQ-X",
        "tdt_enabled": True,
    },
}


class NutqConfig(WhisperConfig):
    """Whisper-compatible configuration with byte CTC and optional TDT heads.

    NUTQ-S and NUTQ-M use a four-layer Whisper decoder plus an auxiliary UTF-8 byte
    CTC objective. NUTQ-X adds a byte Token-and-Duration Transducer (TDT) fast path;
    the four-layer Whisper decoder remains the accuracy-oriented fallback.
    """

    model_type = "nutq"

    def __init__(
        self,
        variant: str = "s",
        backbone_name: str | None = None,
        ctc_vocab_size: int = 257,
        ctc_blank_token_id: int = 256,
        ctc_loss_weight: float = 0.3,
        ar_loss_weight: float = 1.0,
        tdt_enabled: bool | None = None,
        tdt_vocab_size: int = 257,
        tdt_blank_token_id: int = 256,
        tdt_bos_token_id: int = 257,
        tdt_hidden_size: int = 512,
        tdt_num_layers: int = 2,
        tdt_subsampling_factor: int = 4,
        tdt_durations: list[int] | tuple[int, ...] = (0, 1, 2, 3, 4),
        tdt_sigma: float = 0.05,
        tdt_loss_weight: float = 0.3,
        tdt_max_symbols_per_step: int = 10,
        auto_fallback_threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        variant = variant.lower()
        if variant not in NUTQ_PRESETS:
            choices = ", ".join(sorted(NUTQ_PRESETS))
            raise ValueError(f"variant must be one of: {choices}")
        preset = NUTQ_PRESETS[variant]
        if tdt_enabled is None:
            tdt_enabled = bool(preset["tdt_enabled"])
        if kwargs.get("decoder_layers", 4) != 4:
            raise ValueError("NUTQ checkpoints use exactly four Whisper decoder layers")
        if ctc_vocab_size <= ctc_blank_token_id:
            raise ValueError("ctc_blank_token_id must be inside ctc_vocab_size")
        if tdt_vocab_size <= tdt_blank_token_id:
            raise ValueError("tdt_blank_token_id must be inside tdt_vocab_size")
        if tdt_bos_token_id < tdt_vocab_size:
            raise ValueError("tdt_bos_token_id must not collide with an emitted TDT token")
        if tdt_subsampling_factor not in {1, 2, 4, 8}:
            raise ValueError("tdt_subsampling_factor must be one of: 1, 2, 4, 8")
        durations = tuple(int(value) for value in tdt_durations)
        if not durations or durations[0] != 0 or any(value < 0 for value in durations):
            raise ValueError("tdt_durations must begin with zero and contain non-negative values")
        if len(set(durations)) != len(durations):
            raise ValueError("tdt_durations must not contain duplicates")
        if min(ctc_loss_weight, ar_loss_weight, tdt_loss_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if auto_fallback_threshold is not None and not 0.0 <= auto_fallback_threshold <= 1.0:
            raise ValueError("auto_fallback_threshold must be between zero and one")

        kwargs["decoder_layers"] = 4
        super().__init__(**kwargs)
        self.variant = variant
        self.backbone_name = backbone_name or str(preset["backbone"])
        self.ctc_vocab_size = ctc_vocab_size
        self.ctc_blank_token_id = ctc_blank_token_id
        self.ctc_loss_weight = ctc_loss_weight
        self.ar_loss_weight = ar_loss_weight
        self.tdt_enabled = tdt_enabled
        self.tdt_vocab_size = tdt_vocab_size
        self.tdt_blank_token_id = tdt_blank_token_id
        self.tdt_bos_token_id = tdt_bos_token_id
        self.tdt_hidden_size = tdt_hidden_size
        self.tdt_num_layers = tdt_num_layers
        self.tdt_subsampling_factor = tdt_subsampling_factor
        self.tdt_durations = durations
        self.tdt_sigma = tdt_sigma
        self.tdt_loss_weight = tdt_loss_weight
        self.tdt_max_symbols_per_step = tdt_max_symbols_per_step
        self.auto_fallback_threshold = auto_fallback_threshold

    @property
    def display_name(self) -> str:
        return str(NUTQ_PRESETS[self.variant]["display_name"])
