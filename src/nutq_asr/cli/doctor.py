"""Report whether the current machine is ready for NUTQ."""

from __future__ import annotations

import argparse
import json
import platform
from typing import Any

import torch
import transformers


def system_report() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpus: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpus": gpus,
    }
    if cuda_available:
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "vram_gib": round(properties.total_memory / 2**30, 2),
                }
            )
    else:
        report["bf16_supported"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check NUTQ GPU and software readiness")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = system_report()
    print(json.dumps(report, indent=2))
    if args.require_gpu and not report["cuda_available"]:
        raise SystemExit("CUDA GPU not found. Install a CUDA-enabled PyTorch build.")


if __name__ == "__main__":
    main()
