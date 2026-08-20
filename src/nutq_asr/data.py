"""Dataset preprocessing and dynamic padding for NUTQ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    audio = example[audio_column]
    array = audio["array"]
    sampling_rate = audio["sampling_rate"]
    max_samples = int(max_audio_seconds * sampling_rate)
    if len(array) > max_samples:
        array = array[:max_samples]
    processed = processor(
        audio=array,
        text=example[text_column],
        sampling_rate=sampling_rate,
        audio_kwargs={"padding": False},
        text_kwargs={"truncation": True, "max_length": max_label_length},
    )
    return {
        "input_features": processed["input_features"][0],
        "attention_mask": processed["attention_mask"][0],
        "labels": processed["labels"],
    }
