from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..inference import NutqTranscriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe audio with a NUTQ checkpoint")
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--strategy", choices=["ar", "ctc", "tdt", "auto"], default="ar")
    parser.add_argument("--fallback-threshold", type=float, default=None)
    parser.add_argument(
        "--language", default=None, help="Whisper language name/code; omit to detect"
    )
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=448)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dtype = getattr(torch, args.dtype) if args.dtype else None
    transcriber = NutqTranscriber.from_pretrained(
        args.model, device=args.device, dtype=dtype, compile_model=args.compile
    )
    for path in args.audio:
        generation_options = {
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "task": args.task,
        }
        if args.language is not None:
            generation_options["language"] = args.language
        text = transcriber.transcribe(
            path,
            strategy=args.strategy,
            fallback_threshold=args.fallback_threshold,
            **generation_options,
        )
        if args.as_json:
            print(json.dumps({"audio": str(path), "text": text}, ensure_ascii=False))
        else:
            print(f"{path}: {text}")


if __name__ == "__main__":
    main()
