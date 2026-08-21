from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoProcessor, PreTrainedTokenizerFast, WhisperFeatureExtractor

from nutq_asr import NutqConfig, NutqProcessor


def test_processor_auto_class_round_trip(tmp_path, tiny_config: NutqConfig) -> None:
    backend = Tokenizer(
        WordLevel({"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "nutq": 4}, "<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    processor = NutqProcessor(
        feature_extractor=WhisperFeatureExtractor(feature_size=8, num_mel_bins=8, chunk_length=1),
        tokenizer=tokenizer,
    )
    tiny_config.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)

    restored = AutoProcessor.from_pretrained(tmp_path)

    assert isinstance(restored, NutqProcessor)
    assert restored.feature_extractor.sampling_rate == 16_000
    assert restored.decode_bytes(restored.encode_bytes("nutq — нутқ")) == "nutq — нутқ"
