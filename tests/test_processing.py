from __future__ import annotations

from transformers import AutoProcessor, ByT5Tokenizer, WhisperFeatureExtractor

from nutq_asr import NutqConfig, NutqProcessor


def test_processor_auto_class_round_trip(tmp_path, tiny_config: NutqConfig) -> None:
    processor = NutqProcessor(
        WhisperFeatureExtractor(feature_size=8, num_mel_bins=8, chunk_length=1),
        ByT5Tokenizer(),
    )
    tiny_config.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)

    restored = AutoProcessor.from_pretrained(tmp_path)

    assert isinstance(restored, NutqProcessor)
    assert restored.feature_extractor.sampling_rate == 16_000
    assert restored.decode(restored.tokenizer.encode("nutq"), skip_special_tokens=True) == "nutq"
