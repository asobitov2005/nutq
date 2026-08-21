"""Evaluate a NUTQ checkpoint with the training data pipeline."""

from __future__ import annotations

import argparse
import functools
import json
import time

import numpy as np
import torch
from datasets import Audio, load_dataset
from jiwer import cer, wer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

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
    parser.add_argument("--max-new-tokens", type=int, default=448)
    parser.add_argument("--strategy", choices=["ar", "ctc", "tdt", "auto"], default="ar")
    parser.add_argument("--fallback-threshold", type=float, default=None)
    parser.add_argument(
        "--language", default=None, help="Whisper language name/code; omit to detect"
    )
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    processor = NutqProcessor.from_pretrained(args.model)
    model = NutqForConditionalGeneration.from_pretrained(args.model)
    model = model.to(device=device, dtype=dtype)  # type: ignore[call-arg]
    model.eval()
    if args.strategy in {"tdt", "auto"} and not model.config.tdt_enabled:
        raise ValueError("tdt/auto evaluation requires a NUTQ-X checkpoint")
    configured_threshold = (
        args.fallback_threshold
        if args.fallback_threshold is not None
        else model.config.auto_fallback_threshold
    )
    if args.strategy == "auto" and configured_threshold is None:
        raise ValueError("auto evaluation requires a calibrated auto_fallback_threshold")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    dataset = dataset.cast_column(
        args.audio_column,
        Audio(sampling_rate=processor.feature_extractor.sampling_rate, decode=False),
    )
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    transform = functools.partial(
        prepare_dataset_example,
        processor=processor,
        audio_column=args.audio_column,
        text_column=args.text_column,
        long_audio_policy="reject",
        default_language=args.language,
        default_task=args.task,
    )
    dataset = dataset.map(transform, remove_columns=dataset.column_names)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=NutqDataCollator(processor, tdt_enabled=model.config.tdt_enabled),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    references: list[str] = []
    predictions: list[str] = []
    total_audio_seconds = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generation_options = {
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "task": args.task,
    }
    if args.language is not None:
        generation_options["language"] = args.language
    for batch in tqdm(loader, desc="Evaluating"):
        labels = batch.pop("labels")
        batch.pop("ctc_labels")
        batch.pop("tdt_labels", None)
        batch.pop("auxiliary_mask")
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
        total_audio_seconds += (
            float(batch["attention_mask"].sum().item())
            * processor.feature_extractor.hop_length
            / processor.feature_extractor.sampling_rate
        )
        with torch.inference_mode():
            if args.strategy == "ctc":
                byte_ids, _ = model.ctc_greedy_decode(**batch)
                batch_predictions = [processor.decode_bytes(value) for value in byte_ids]
            elif args.strategy in {"tdt", "auto"}:
                byte_ids, confidence = model.tdt_greedy_decode(**batch)
                batch_predictions = [processor.decode_bytes(value) for value in byte_ids]
                if args.strategy == "auto":
                    threshold = float(configured_threshold)
                    fallback = [
                        index for index, value in enumerate(confidence) if value < threshold
                    ]
                    if fallback:
                        selected = {key: value[fallback] for key, value in batch.items()}
                        generated = model.generate(
                            **selected,
                            **generation_options,
                        )
                        corrected = processor.batch_decode(
                            generated.cpu(), skip_special_tokens=True
                        )
                        for index, text in zip(fallback, corrected, strict=True):
                            batch_predictions[index] = text
            else:
                generated = model.generate(**batch, **generation_options)
                batch_predictions = processor.batch_decode(
                    generated.cpu(), skip_special_tokens=True
                )
        predictions.extend(batch_predictions)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    normalizer = BasicTextNormalizer()
    normalized_references = [normalizer(value) for value in references]
    normalized_predictions = [normalizer(value) for value in predictions]
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    )
    print(
        json.dumps(
            {
                "strategy": args.strategy,
                "wer": 100 * wer(normalized_references, normalized_predictions),
                "cer": 100 * cer(normalized_references, normalized_predictions),
                "raw_wer": 100 * wer(references, predictions),
                "raw_cer": 100 * cer(references, predictions),
                "rtf": elapsed / total_audio_seconds if total_audio_seconds else None,
                "audio_seconds": total_audio_seconds,
                "wall_seconds": elapsed,
                "peak_vram_gib": peak_vram,
            }
        )
    )


if __name__ == "__main__":
    main()
