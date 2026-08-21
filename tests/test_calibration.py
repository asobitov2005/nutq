from __future__ import annotations

from nutq_asr.calibration import select_fallback_threshold


def test_calibration_selects_data_driven_gate() -> None:
    references = ["one two", "three four", "five six"]
    drafts = ["one two", "wrong", "wrong"]
    ar = references.copy()
    confidence = [0.9, 0.4, 0.2]

    selected, curve = select_fallback_threshold(
        references, drafts, ar, confidence, max_relative_wer_increase=0.0
    )

    assert selected.threshold > 0.4
    assert selected.fallback_rate == 2 / 3
    assert selected.wer == 0.0
    assert len(curve) >= 3
