"""Evaluate a NUTQ checkpoint with the training data pipeline."""

from __future__ import annotations

import argparse
import functools
import json

import numpy as np
import torch
from datasets import Audio, load_dataset
from jiwer import cer, wer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import NutqDataCollator, prepare_dataset_example
from ..modeling_nutq import NutqForConditionalGeneration
from ..processing_nutq import NutqProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a NUTQ checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    processor = NutqProcessor.from_pretrained(args.model)
    model = NutqForConditionalGeneration.from_pretrained(args.model).to(device, dtype=dtype).eval()
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    dataset = dataset.cast_column(
        args.audio_column, Audio(sampling_rate=processor.feature_extractor.sampling_rate)
    )
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    transform = functools.partial(
        prepare_dataset_example,
        processor=processor,
        audio_column=args.audio_column,
        text_column=args.text_column,
    )
    dataset = dataset.map(transform, remove_columns=dataset.column_names)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=NutqDataCollator(processor),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    references: list[str] = []
    predictions: list[str] = []
    for batch in tqdm(loader, desc="Evaluating"):
        labels = batch.pop("labels")
        batch.pop("ctc_labels")
        references.extend(
            processor.batch_decode(
                np.where(labels.numpy() == -100, processor.tokenizer.pad_token_id, labels.numpy()),
                skip_special_tokens=True,
            )
        )
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key in {"input_features", "attention_mask"}
        }
        batch["input_features"] = batch["input_features"].to(dtype)
        with torch.inference_mode():
            generated = model.generate(
                **batch, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens
            )
        predictions.extend(processor.batch_decode(generated.cpu(), skip_special_tokens=True))
    print(
        json.dumps(
            {"wer": 100 * wer(references, predictions), "cer": 100 * cer(references, predictions)}
        )
    )


if __name__ == "__main__":
    main()
