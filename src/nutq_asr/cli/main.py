"""Unified, low-friction NUTQ command line."""

from __future__ import annotations

import sys
from collections.abc import Callable


def _commands() -> dict[str, Callable[[], None]]:
    # Lazy imports keep `nutq doctor` fast and avoid importing dataset tooling for inference.
    from .doctor import main as doctor
    from .evaluate import main as evaluate
    from .initialize import main as initialize
    from .train import main as train
    from .transcribe import main as transcribe

    return {
        "doctor": doctor,
        "init": initialize,
        "train": train,
        "eval": evaluate,
        "transcribe": transcribe,
    }


def _print_help() -> None:
    print(
        """NUTQ — compact multilingual speech recognition

Usage:
  nutq doctor [--require-gpu]
  nutq init --output CHECKPOINT
  nutq train --dataset DATASET [OPTIONS]
  nutq eval --model CHECKPOINT --dataset DATASET [OPTIONS]
  nutq transcribe AUDIO... --model CHECKPOINT [OPTIONS]

Run `nutq COMMAND --help` for command-specific options.
"""
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_help()
        return
    command = sys.argv[1]
    commands = _commands()
    if command not in commands:
        available = ", ".join(commands)
        raise SystemExit(f"Unknown command '{command}'. Available commands: {available}")
    sys.argv = [f"{sys.argv[0]} {command}", *sys.argv[2:]]
    commands[command]()


if __name__ == "__main__":
    main()
