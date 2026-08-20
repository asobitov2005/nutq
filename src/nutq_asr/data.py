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
    """Pad acoustic features and create decoder and CTC targets."""

    processor: NutqProcessor
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        acoustic = [
            {key: feature[key] for key in ("input_features", "attention_mask") if key in feature}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(acoustic, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), self.label_pad_token_id
        )
        batch["labels"] = labels

        special_ids = set(self.processor.tokenizer.all_special_ids)
        ctc_sequences = [
            [token_id for token_id in feature["labels"] if token_id not in special_ids]
            for feature in features
        ]
        max_length = max((len(sequence) for sequence in ctc_sequences), default=0)
        ctc_labels = torch.full(
            (len(features), max_length), self.label_pad_token_id, dtype=torch.long
        )
        for row, sequence in enumerate(ctc_sequences):
            if sequence:
                ctc_labels[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        batch["ctc_labels"] = ctc_labels
        return batch


def prepare_dataset_example(
    example: dict[str, Any],
    processor: NutqProcessor,
    audio_column: str = "audio",
    text_column: str = "text",
    max_audio_seconds: float = 30.0,
    max_label_length: int = 1024,
) -> dict[str, Any]:
    """Convert one Datasets audio record to model-ready Python lists."""
    array, sampling_rate = decode_audio_record(example[audio_column])
    target_rate = processor.feature_extractor.sampling_rate
    if sampling_rate != target_rate:
        array = soxr.resample(array, sampling_rate, target_rate)
        sampling_rate = target_rate
    max_samples = int(max_audio_seconds * sampling_rate)
    if len(array) > max_samples:
        array = array[:max_samples]
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
