from __future__ import annotations

from nutq_asr import NutqConfig


def test_config_round_trip(tmp_path, tiny_config: NutqConfig) -> None:
    tiny_config.save_pretrained(tmp_path)
    restored = NutqConfig.from_pretrained(tmp_path)

    assert restored.model_type == "nutq"
    assert restored.encoder.d_model == 16
    assert restored.decoder.d_model == 16
    assert restored.ctc_blank_token_id == 32


def test_config_rejects_invalid_compressor(tiny_config: NutqConfig) -> None:
    config = tiny_config.to_dict()
    config["compressor_mode"] = "semantic-if-statements"
    try:
        NutqConfig(**config)
    except ValueError as error:
        assert "compressor_mode" in str(error)
    else:
        raise AssertionError("invalid compressor mode was accepted")
