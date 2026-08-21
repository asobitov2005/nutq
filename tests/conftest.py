from __future__ import annotations

import pytest

from nutq_asr import NutqConfig


def _config(**overrides) -> NutqConfig:
    values = {
        "variant": "s",
        "vocab_size": 32,
        "num_mel_bins": 8,
        "d_model": 16,
        "encoder_layers": 2,
        "encoder_attention_heads": 2,
        "encoder_ffn_dim": 32,
        "decoder_layers": 4,
        "decoder_attention_heads": 2,
        "decoder_ffn_dim": 32,
        "max_source_positions": 8,
        "max_target_positions": 8,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "decoder_start_token_id": 1,
        "ctc_vocab_size": 17,
        "ctc_blank_token_id": 16,
    }
    values.update(overrides)
    return NutqConfig(**values)


@pytest.fixture
def tiny_config() -> NutqConfig:
    return _config()


@pytest.fixture
def tiny_x_config() -> NutqConfig:
    return _config(
        variant="x",
        tdt_enabled=True,
        tdt_vocab_size=17,
        tdt_blank_token_id=16,
        tdt_bos_token_id=17,
        tdt_hidden_size=16,
        tdt_num_layers=1,
        tdt_subsampling_factor=2,
        tdt_durations=(0, 1, 2),
    )
