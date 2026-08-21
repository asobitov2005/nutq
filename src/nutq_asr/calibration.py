"""Data-driven confidence calibration for the NUTQ-X TDT/AR router."""

from __future__ import annotations

from dataclasses import dataclass

from jiwer import wer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


@dataclass(frozen=True)
class CalibrationPoint:
    threshold: float
    wer: float
    fallback_rate: float


def select_fallback_threshold(
    references: list[str],
    tdt_drafts: list[str],
    ar_transcripts: list[str],
    confidences: list[float],
    max_relative_wer_increase: float = 0.02,
) -> tuple[CalibrationPoint, list[CalibrationPoint]]:
    """Choose the cheapest confidence gate inside a declared AR-relative WER budget."""
    size = len(references)
    if not size or not (len(tdt_drafts) == len(ar_transcripts) == len(confidences) == size):
        raise ValueError("calibration inputs must have the same non-zero length")
    if max_relative_wer_increase < 0:
        raise ValueError("max_relative_wer_increase must be non-negative")
    normalizer = BasicTextNormalizer()
    normalized_references = [normalizer(value) for value in references]
    normalized_tdt = [normalizer(value) for value in tdt_drafts]
    normalized_ar = [normalizer(value) for value in ar_transcripts]
    ar_wer = wer(normalized_references, normalized_ar)
    allowed_wer = ar_wer * (1.0 + max_relative_wer_increase)
    thresholds = sorted({0.0, 1.0, *(max(0.0, min(1.0, value)) for value in confidences)})
    curve = []
    for threshold in thresholds:
        fallback = [value < threshold for value in confidences]
        transcripts = [
            ar if use_ar else draft
            for draft, ar, use_ar in zip(normalized_tdt, normalized_ar, fallback, strict=True)
        ]
        point = CalibrationPoint(
            threshold=threshold,
            wer=wer(normalized_references, transcripts),
            fallback_rate=sum(fallback) / size,
        )
        curve.append(point)
    eligible = [point for point in curve if point.wer <= allowed_wer + 1e-12]
    if not eligible:
        raise RuntimeError("no threshold satisfies the requested AR-relative WER budget")
    selected = min(eligible, key=lambda point: (point.fallback_rate, point.wer, point.threshold))
    return selected, curve
