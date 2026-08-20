from __future__ import annotations

import torch

from nutq_asr import NutqConfig, NutqForConditionalGeneration


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_features": torch.randn(2, 8, 16),
        "attention_mask": torch.tensor([[1] * 16, [1] * 12 + [0] * 4], dtype=torch.long),
        "labels": torch.tensor([[3, 4, 5, 1, -100], [6, 7, 1, -100, -100]]),
        "ctc_labels": torch.tensor([[3, 4, 5, -100], [6, 7, -100, -100]]),
    }


def test_forward_backward(tiny_config: NutqConfig) -> None:
    model = NutqForConditionalGeneration(tiny_config)
    output = model(**_batch())

    assert output.logits.shape == (2, 5, 32)
    assert output.ctc_logits.shape == (2, 8, 33)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.ctc_loss is not None and torch.isfinite(output.ctc_loss)
    output.loss.backward()
    assert model.model.acoustic_encoder.projector.output.weight.grad is not None


def test_save_and_load(tmp_path, tiny_config: NutqConfig) -> None:
    model = NutqForConditionalGeneration(tiny_config).eval()
    model.save_pretrained(tmp_path)
    restored = NutqForConditionalGeneration.from_pretrained(tmp_path).eval()

    batch = _batch()
    with torch.no_grad():
        original = model(**batch).logits
        loaded = restored(**batch).logits
    torch.testing.assert_close(original, loaded)


def test_greedy_generation(tiny_config: NutqConfig) -> None:
    model = NutqForConditionalGeneration(tiny_config).eval()
    batch = _batch()
    with torch.no_grad():
        tokens = model.generate(
            input_features=batch["input_features"][:1],
            attention_mask=batch["attention_mask"][:1],
            max_new_tokens=3,
        )
    assert tokens.shape[0] == 1
    assert tokens.shape[1] <= 4
