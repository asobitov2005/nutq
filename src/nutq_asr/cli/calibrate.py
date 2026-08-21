"""Calibrate the NUTQ-X TDT-to-AR confidence gate on held-out data."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..calibration import select_fallback_threshold
from ..data import NutqDataCollator, prepare_dataset_example
from ..modeling_nutq import NutqForConditionalGeneration
from ..processing_nutq import NutqProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the NUTQ-X confidence router")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=448)
    parser.add_argument(
        "--language", default=None, help="Whisper language name/code; omit to detect"
    )
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--max-relative-wer-increase", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("calibration.json"))
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
    if not model.config.tdt_enabled:
        raise ValueError("calibration requires a NUTQ-X checkpoint with a trained TDT head")

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
        collate_fn=NutqDataCollator(processor, tdt_enabled=True),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    references: list[str] = []
    tdt_drafts: list[str] = []
    ar_transcripts: list[str] = []
    confidences: list[float] = []
    generation_options = {"max_new_tokens": args.max_new_tokens, "num_beams": 1, "task": args.task}
    if args.language is not None:
        generation_options["language"] = args.language
    for batch in tqdm(loader, desc="Calibrating"):
        labels = batch.pop("labels")
        batch.pop("ctc_labels")
        batch.pop("tdt_labels")
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
        with torch.inference_mode():
            byte_ids, batch_confidences = model.tdt_greedy_decode(**batch)
            generated = model.generate(**batch, **generation_options)
        tdt_drafts.extend(processor.decode_bytes(value) for value in byte_ids)
        confidences.extend(batch_confidences)
        ar_transcripts.extend(processor.batch_decode(generated.cpu(), skip_special_tokens=True))

    selected, curve = select_fallback_threshold(
        references,
        tdt_drafts,
        ar_transcripts,
        confidences,
        max_relative_wer_increase=args.max_relative_wer_increase,
    )
    result = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "samples": len(references),
        "max_relative_wer_increase": args.max_relative_wer_increase,
        "selected": {
            "threshold": selected.threshold,
            "wer": 100.0 * selected.wer,
            "fallback_rate": selected.fallback_rate,
        },
        "curve": [
            {
                "threshold": point.threshold,
                "wer": 100.0 * point.wer,
                "fallback_rate": point.fallback_rate,
            }
            for point in curve
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["selected"]))


if __name__ == "__main__":
    main()
