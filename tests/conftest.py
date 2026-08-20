from __future__ import annotations

import pytest
from transformers import T5Config, WhisperConfig

from nutq_asr import NutqConfig


@pytest.fixture
def tiny_config() -> NutqConfig:
    encoder = WhisperConfig(
        vocab_size=32,
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=32,
        max_source_positions=8,
        max_target_positions=8,
    )
    decoder = T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        d_kv=8,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        tie_word_embeddings=True,
    )
    return NutqConfig(
        encoder_config=encoder,
        decoder_config=decoder,
        ctc_vocab_size=33,
        ctc_blank_token_id=32,
        compression_ratio=2,
        compressor_mode="soft",
        projector_expansion=1,
        projector_dropout=0.0,
    )
