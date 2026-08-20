"""Train NUTQ with Transformers Trainer and Accelerate backends."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Audio, load_dataset
from jiwer import cer, wer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed

from ..data import NutqDataCollator, prepare_dataset_example
from ..modeling_nutq import NutqForConditionalGeneration
from ..processing_nutq import NutqProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a NUTQ speech recognition model")
    parser.add_argument("--config", type=Path, default=Path("configs/nutq-180m.yaml"))
    parser.add_argument("--dataset", required=True, help="Hub dataset ID or local dataset script")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--train-files", nargs="*", default=None)
    parser.add_argument("--eval-files", nargs="*", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Continue from a NUTQ checkpoint")
    parser.add_argument("--resume-trainer-state", default=None)
    parser.add_argument("--stage", choices=["bridge", "full"], default="bridge")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", nargs="*", default=["tensorboard"])
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section in ("model", "data", "training"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Configuration requires a '{section}' mapping")
    return config


def _load_splits(args: argparse.Namespace, sampling_rate: int) -> dict[str, Any]:
    data_files = None
    if args.train_files or args.eval_files:
        data_files = {}
        if args.train_files:
            data_files[args.train_split] = args.train_files
        if args.eval_files:
            data_files[args.eval_split] = args.eval_files
    common = {
        "path": args.dataset,
        "name": args.dataset_config,
        "data_files": data_files,
        "streaming": args.streaming,
    }
    train = load_dataset(**common, split=args.train_split)
    evaluation = load_dataset(**common, split=args.eval_split)
    # Decode with SoundFile in our preprocessing function. This avoids coupling dataset
    # ingestion to a particular TorchCodec/CUDA build.
    train = train.cast_column("audio", Audio(sampling_rate=sampling_rate, decode=False))
    evaluation = evaluation.cast_column("audio", Audio(sampling_rate=sampling_rate, decode=False))
    if args.max_train_samples is not None:
        train = (
            train.take(args.max_train_samples)
            if args.streaming
            else train.select(range(min(args.max_train_samples, len(train))))
        )
    if args.max_eval_samples is not None:
        evaluation = (
            evaluation.take(args.max_eval_samples)
            if args.streaming
            else evaluation.select(range(min(args.max_eval_samples, len(evaluation))))
        )
    return {"train": train, "validation": evaluation}


def _prepare_splits(
    datasets: dict[str, Any], processor: NutqProcessor, data_config: dict[str, Any]
) -> dict[str, Any]:
    transform = functools.partial(
        prepare_dataset_example,
        processor=processor,
        audio_column=data_config.get("audio_column", "audio"),
        text_column=data_config.get("text_column", "text"),
        max_audio_seconds=float(data_config.get("max_audio_seconds", 30.0)),
        max_label_length=int(data_config.get("max_label_length", 1024)),
    )
    prepared = {}
    for split, dataset in datasets.items():
        remove_columns = list(dataset.column_names)
        prepared[split] = dataset.map(transform, remove_columns=remove_columns)
    return prepared


def _metrics(processor: NutqProcessor):
    def compute(eval_prediction: Any) -> dict[str, float]:
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(
            eval_prediction.label_ids == -100,
            processor.tokenizer.pad_token_id,
            eval_prediction.label_ids,
        )
        predicted_text = processor.batch_decode(predictions, skip_special_tokens=True)
        reference_text = processor.batch_decode(labels, skip_special_tokens=True)
        return {
            "wer": 100.0 * wer(reference_text, predicted_text),
            "cer": 100.0 * cer(reference_text, predicted_text),
        }

    return compute


def _training_arguments(config: dict[str, Any], report_to: list[str]) -> Seq2SeqTrainingArguments:
    values = dict(config)
    values.update(
        {
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "predict_with_generate": True,
            "generation_max_length": values.pop("generation_max_length", 512),
            "load_best_model_at_end": True,
            "metric_for_best_model": "wer",
            "greater_is_better": False,
            "remove_unused_columns": False,
            "label_names": ["labels", "ctc_labels"],
            "report_to": report_to,
            "dataloader_pin_memory": True,
            "ddp_find_unused_parameters": False,
        }
    )
    return Seq2SeqTrainingArguments(**values)


def main() -> None:
    args = build_parser().parse_args()
    config = _load_config(args.config)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    model_config = dict(config["model"])
    encoder_name = model_config.pop("encoder_name")
    decoder_name = model_config.pop("decoder_name")
    if args.checkpoint:
        model = NutqForConditionalGeneration.from_pretrained(args.checkpoint)
        processor = NutqProcessor.from_pretrained(args.checkpoint)
    else:
        model = NutqForConditionalGeneration.from_pretrained_components(
            encoder_name, decoder_name, **model_config
        )
        processor = NutqProcessor.from_pretrained_components(encoder_name, decoder_name)
    if args.stage == "bridge":
        model.freeze_pretrained_components()
    else:
        model.unfreeze_all()

    sampling_rate = int(config["data"].get("sampling_rate", 16_000))
    raw_datasets = _load_splits(args, sampling_rate)
    datasets = _prepare_splits(raw_datasets, processor, config["data"])
    training_args = _training_arguments(config["training"], args.report_to)
    collator = NutqDataCollator(processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        compute_metrics=_metrics(processor),
        processing_class=processor,
    )
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainer.is_world_process_zero():
        print(
            json.dumps(
                {"parameters": total, "trainable_parameters": trainable, "stage": args.stage}
            )
        )
    trainer.train(resume_from_checkpoint=args.resume_trainer_state)
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
