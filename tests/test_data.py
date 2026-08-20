from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from nutq_asr.data import decode_audio_record


def test_decode_audio_bytes_to_mono() -> None:
    stream = io.BytesIO()
    stereo = np.stack(
        [np.linspace(-0.5, 0.5, 160, dtype=np.float32), np.zeros(160, dtype=np.float32)],
        axis=1,
    )
    sf.write(stream, stereo, 16_000, format="WAV", subtype="FLOAT")

    waveform, sampling_rate = decode_audio_record({"bytes": stream.getvalue(), "path": None})

    assert sampling_rate == 16_000
    assert waveform.shape == (160,)
    assert waveform.dtype == np.float32
    np.testing.assert_allclose(waveform, stereo.mean(axis=1), atol=1e-6)


def test_decode_audio_rejects_non_finite_array() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        decode_audio_record({"array": [0.0, float("nan")], "sampling_rate": 16_000})
