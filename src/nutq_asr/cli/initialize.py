"""Create an initial NUTQ checkpoint from pretrained components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..modeling_nutq import NutqForConditionalGeneration
from ..processing_nutq import NutqProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a NUTQ checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", default="openai/whisper-small")
    parser.add_argument("--decoder", default="google/byt5-small")
    parser.add_argument("--compressor", choices=["none", "fixed", "soft"], default="soft")
    parser.add_argument("--compression-ratio", type=int, default=6)
    parser.add_argument("--ctc-loss-weight", type=float, default=0.3)
    args = parser.parse_args()

    model = NutqForConditionalGeneration.from_pretrained_components(
        args.encoder,
        args.decoder,
        compressor_mode=args.compressor,
        compression_ratio=args.compression_ratio,
        ctc_loss_weight=args.ctc_loss_weight,
    )
    processor = NutqProcessor.from_pretrained_components(args.encoder, args.decoder)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    processor.save_pretrained(args.output)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"output": str(args.output), "parameters": parameters}))


if __name__ == "__main__":
    main()
