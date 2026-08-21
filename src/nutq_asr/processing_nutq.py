"""Whisper audio/BPE processing plus universal UTF-8 auxiliary targets."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperProcessor

from .configuration_nutq import NUTQ_PRESETS


class NutqProcessor(WhisperProcessor):
    """Whisper processor with deterministic UTF-8 byte helpers for CTC and TDT."""

    @staticmethod
    def encode_bytes(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode_bytes(token_ids: list[int]) -> str:
        values = bytes(token_id for token_id in token_ids if 0 <= token_id <= 255)
        return values.decode("utf-8", errors="replace")

    def set_target_prefix(self, language: str | None, task: str = "transcribe") -> None:
        """Set Whisper supervision tokens for one language/task pair."""
        if task not in {"transcribe", "translate"}:
            raise ValueError("task must be 'transcribe' or 'translate'")
        if hasattr(self.tokenizer, "set_prefix_tokens"):
            self.tokenizer.set_prefix_tokens(
                language=language,
                task=task,
                predict_timestamps=False,
            )

    @classmethod
    def from_pretrained_variant(
        cls,
        variant: str = "s",
        backbone_name_or_path: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
        **feature_extractor_kwargs: Any,
    ) -> NutqProcessor:
        variant = variant.lower()
        if variant not in NUTQ_PRESETS:
            choices = ", ".join(sorted(NUTQ_PRESETS))
            raise ValueError(f"variant must be one of: {choices}")
        backbone = backbone_name_or_path or str(NUTQ_PRESETS[variant]["backbone"])
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            backbone, **feature_extractor_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        processor = cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
        processor.set_target_prefix(language, task)
        return processor
