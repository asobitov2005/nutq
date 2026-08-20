"""Audio and byte-token processing for NUTQ."""

from __future__ import annotations

from typing import Any

from transformers import ByT5Tokenizer, ProcessorMixin, WhisperFeatureExtractor


class NutqProcessor(ProcessorMixin):
    """Bundle a Whisper feature extractor with a ByT5 byte tokenizer."""

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "ByT5Tokenizer"

    def __init__(
        self, feature_extractor: WhisperFeatureExtractor, tokenizer: ByT5Tokenizer
    ) -> None:
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        audio: Any | None = None,
        text: str | list[str] | None = None,
        sampling_rate: int = 16_000,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if audio is None and text is None:
            raise ValueError("At least one of audio or text must be provided")
        output: dict[str, Any] = {}
        if audio is not None:
            audio_kwargs = dict(kwargs.pop("audio_kwargs", {}))
            audio_kwargs.setdefault("return_attention_mask", True)
            output.update(
                self.feature_extractor(
                    audio,
                    sampling_rate=sampling_rate,
                    return_tensors=return_tensors,
                    **audio_kwargs,
                )
            )
        if text is not None:
            text_kwargs = dict(kwargs.pop("text_kwargs", {}))
            tokenized = self.tokenizer(text, return_tensors=return_tensors, **text_kwargs)
            if audio is None:
                output.update(tokenized)
            else:
                output["labels"] = tokenized["input_ids"]
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected processor arguments: {unexpected}")
        return output

    def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> str:
        return self.tokenizer.decode(*args, **kwargs)

    @classmethod
    def from_pretrained_components(
        cls,
        encoder_name_or_path: str = "openai/whisper-small",
        decoder_name_or_path: str = "google/byt5-small",
        **feature_extractor_kwargs: Any,
    ) -> NutqProcessor:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            encoder_name_or_path, **feature_extractor_kwargs
        )
        tokenizer = ByT5Tokenizer.from_pretrained(decoder_name_or_path)
        return cls(feature_extractor, tokenizer)
