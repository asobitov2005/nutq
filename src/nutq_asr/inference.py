"""High-level NUTQ transcription API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr
import torch

from .modeling_nutq import NutqForConditionalGeneration
from .processing_nutq import NutqProcessor


class NutqTranscriber:
    """Load a NUTQ checkpoint and transcribe files or waveform arrays."""

    def __init__(
        self,
        model: NutqForConditionalGeneration,
        processor: NutqProcessor,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        compile_model: bool = False,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        compile_model: bool = False,
        **kwargs: Any,
    ) -> NutqTranscriber:
        model = NutqForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        processor = NutqProcessor.from_pretrained(model_name_or_path)
        return cls(model, processor, device=device, dtype=dtype, compile_model=compile_model)

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        sampling_rate: int | None = None,
        **generation_kwargs: Any,
    ) -> str:
        waveform, rate = self._load_audio(audio, sampling_rate)
        target_rate = self.processor.feature_extractor.sampling_rate
        if rate != target_rate:
            waveform = soxr.resample(waveform, rate, target_rate)
            rate = target_rate
        inputs = self.processor(
            audio=waveform,
            sampling_rate=rate,
            return_tensors="pt",
            audio_kwargs={"padding": True},
        )
        model_inputs = {
            key: value.to(device=self.device, non_blocking=True)
            for key, value in inputs.items()
            if key in {"input_features", "attention_mask"}
        }
        model_inputs["input_features"] = model_inputs["input_features"].to(self.dtype)
        generation_kwargs.setdefault("max_new_tokens", 512)
        with torch.inference_mode():
            token_ids = self.model.generate(**model_inputs, **generation_kwargs)
        return self.processor.decode(token_ids[0], skip_special_tokens=True)

    @staticmethod
    def _load_audio(
        audio: str | Path | np.ndarray, sampling_rate: int | None
    ) -> tuple[np.ndarray, int]:
        if isinstance(audio, (str, Path)):
            waveform, rate = sf.read(audio, dtype="float32", always_2d=False)
        else:
            if sampling_rate is None:
                raise ValueError("sampling_rate is required when audio is an array")
            waveform = np.asarray(audio, dtype=np.float32)
            rate = sampling_rate
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1:
            raise ValueError("audio must be mono or a channels-last 2D waveform")
        return waveform, rate
