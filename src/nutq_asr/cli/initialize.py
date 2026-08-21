"""Create an initialized NUTQ-S, NUTQ-M, or NUTQ-X checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..modeling_nutq import NutqForConditionalGeneration
from ..processing_nutq import NutqProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a NUTQ model family checkpoint")
    parser.add_argument("--variant", choices=["s", "m", "x"], default="m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default=None, help="Override the preset Whisper checkpoint")
    parser.add_argument("--language", default=None, help="Optional Whisper language prefix")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--ctc-loss-weight", type=float, default=0.3)
    parser.add_argument("--tdt-loss-weight", type=float, default=0.3)
    args = parser.parse_args()

    model = NutqForConditionalGeneration.from_pretrained_variant(
        args.variant,
        args.backbone,
        ctc_loss_weight=args.ctc_loss_weight,
        tdt_loss_weight=args.tdt_loss_weight,
    )
    processor = NutqProcessor.from_pretrained_variant(
        args.variant,
        args.backbone,
        language=args.language,
        task=args.task,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    processor.save_pretrained(args.output)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "variant": model.config.display_name,
                "parameters": parameters,
                "trainable_parameters": trainable,
                "backbone": model.config.backbone_name,
            }
        )
    )


if __name__ == "__main__":
    main()
