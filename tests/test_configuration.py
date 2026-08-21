from __future__ import annotations

import pytest

from nutq_asr import NutqConfig


def test_config_round_trip(tmp_path, tiny_config: NutqConfig) -> None:
    tiny_config.save_pretrained(tmp_path)
    restored = NutqConfig.from_pretrained(tmp_path)

    assert restored.model_type == "nutq"
    assert restored.variant == "s"
    assert restored.decoder_layers == 4
    assert restored.ctc_blank_token_id == 16


def test_config_rejects_unknown_variant(tiny_config: NutqConfig) -> None:
    values = tiny_config.to_dict()
    values["variant"] = "phrase-hardcoded-edition"
    with pytest.raises(ValueError, match="variant"):
        NutqConfig(**values)


def test_config_rejects_invalid_tdt_durations(tiny_config: NutqConfig) -> None:
    values = tiny_config.to_dict()
    values["tdt_durations"] = [1, 2, 3]
    with pytest.raises(ValueError, match="tdt_durations"):
        NutqConfig(**values)
