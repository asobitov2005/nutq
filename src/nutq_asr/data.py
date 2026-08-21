"""Dataset preprocessing and dynamic padding for NUTQ."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr
import torch

from .processing_nutq import NutqProcessor


@dataclass
class NutqDataCollator:
    """Pad acoustic features and create Whisper, byte CTC, and optional TDT targets."""

    processor: NutqProcessor
    tdt_enabled: bool = False
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        acoustic = [
            {key: feature[key] for key in ("input_features", "attention_mask") if key in feature}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(acoustic, return_tensors="pt")

        prefix_tokens = getattr(self.processor.tokenizer, "prefix_tokens", ())
        decoder_start_id = (
            prefix_tokens[0] if prefix_tokens else self.processor.tokenizer.bos_token_id
        )
        label_features = []
        for feature in features:
            token_ids = list(feature["labels"])
            if token_ids and token_ids[0] == decoder_start_id:
                token_ids = token_ids[1:]
            label_features.append({"input_ids": token_ids})
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), self.label_pad_token_id
        )
        batch["labels"] = labels

        auxiliary_mask = torch.tensor(
            [bool(feature.get("auxiliary_enabled", True)) for feature in features], dtype=torch.bool
        )
        byte_sequences = [
            list(feature["byte_labels"]) if enabled else []
            for feature, enabled in zip(features, auxiliary_mask.tolist(), strict=True)
        ]
        max_length = max((len(sequence) for sequence in byte_sequences), default=0)
        byte_labels = torch.full(
            (len(features), max_length), self.label_pad_token_id, dtype=torch.long
        )
        for row, sequence in enumerate(byte_sequences):
            if sequence:
                byte_labels[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        batch["ctc_labels"] = byte_labels
        batch["auxiliary_mask"] = auxiliary_mask
        if self.tdt_enabled:
            batch["tdt_labels"] = byte_labels.clone()
        return batch


def prepare_dataset_example(
    example: dict[str, Any],
    processor: NutqProcessor,
    audio_column: str = "audio",
    text_column: str = "text",
    max_audio_seconds: float = 30.0,
    max_label_length: int = 448,
    long_audio_policy: str = "reject",
    language_column: str | None = None,
    task_column: str | None = None,
    default_language: str | None = None,
    default_task: str = "transcribe",
) -> dict[str, Any]:
    """Convert one Datasets audio record to model-ready Python lists."""
    array, sampling_rate = decode_audio_record(example[audio_column])
    target_rate = processor.feature_extractor.sampling_rate
    if sampling_rate != target_rate:
        array = soxr.resample(array, sampling_rate, target_rate)
        sampling_rate = target_rate
    max_samples = int(max_audio_seconds * sampling_rate)
    if len(array) > max_samples:
        if long_audio_policy == "truncate":
            array = array[:max_samples]
        elif long_audio_policy == "reject":
            raise ValueError(
                f"audio is {len(array) / sampling_rate:.2f}s; maximum is {max_audio_seconds:.2f}s"
            )
        else:
            raise ValueError("long_audio_policy must be 'reject' or 'truncate'")
    language = (
        str(example[language_column])
        if language_column and example.get(language_column) is not None
        else default_language
    )
    task = (
        str(example[task_column])
        if task_column and example.get(task_column) is not None
        else default_task
    )
    processor.set_target_prefix(language, task)
    processed = processor(
        audio=array,
        text=example[text_column],
        sampling_rate=sampling_rate,
        audio_kwargs={"padding": "max_length", "truncation": True},
        text_kwargs={"truncation": True, "max_length": max_label_length},
    )
    return {
        "input_features": processed["input_features"][0],
        "attention_mask": processed["attention_mask"][0],
        "labels": processed["labels"],
        "byte_labels": processor.encode_bytes(str(example[text_column])),
        "auxiliary_enabled": task == "transcribe",
        "input_length": len(array),
    }


def decode_audio_record(audio: Any) -> tuple[np.ndarray, int]:
    """Decode a Datasets Audio value without requiring TorchCodec."""
    if isinstance(audio, dict) and audio.get("array") is not None:
        array = np.asarray(audio["array"], dtype=np.float32)
        sampling_rate = int(audio["sampling_rate"])
    else:
        source: Any
        if isinstance(audio, dict) and audio.get("bytes") is not None:
            source = io.BytesIO(audio["bytes"])
        elif isinstance(audio, dict) and audio.get("path") is not None:
            source = audio["path"]
        elif isinstance(audio, (str, Path)):
            source = audio
        else:
            raise ValueError("audio must contain array, bytes, or path data")
        array, sampling_rate = sf.read(source, dtype="float32", always_2d=False)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("decoded audio must be a non-empty mono waveform")
    if not np.isfinite(array).all():
        raise ValueError("decoded audio contains non-finite samples")
    return np.ascontiguousarray(array, dtype=np.float32), sampling_rate
