from __future__ import annotations

import torch

from nutq_asr import NutqConfig, NutqForConditionalGeneration


def _batch(include_tdt: bool = False) -> dict[str, torch.Tensor]:
    batch = {
        "input_features": torch.randn(2, 8, 16),
        "attention_mask": torch.tensor([[1] * 16, [1] * 12 + [0] * 4], dtype=torch.long),
        "labels": torch.tensor([[3, 4, 5, 2, -100], [6, 7, 2, -100, -100]]),
        "ctc_labels": torch.tensor([[3, 4, 5, -100], [6, 7, -100, -100]]),
    }
    if include_tdt:
        batch["tdt_labels"] = batch["ctc_labels"].clone()
    return batch


def test_forward_backward(tiny_config: NutqConfig) -> None:
    model = NutqForConditionalGeneration(tiny_config)
    output = model(**_batch())

    assert output.logits.shape == (2, 5, 32)
    assert output.ctc_logits.shape == (2, 8, 17)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.ctc_loss is not None and torch.isfinite(output.ctc_loss)
    output.loss.backward()
    assert model.ctc_head.weight.grad is not None
    assert model.model.decoder.layers[0].fc1.weight.grad is not None


def test_x_tdt_forward_backward(tiny_x_config: NutqConfig) -> None:
    model = NutqForConditionalGeneration(tiny_x_config)
    batch = _batch(include_tdt=True)
    batch["auxiliary_mask"] = torch.tensor([True, False])
    output = model(**batch)

    assert output.tdt_loss is not None and torch.isfinite(output.tdt_loss)
    output.loss.backward()
    assert model.tdt_head is not None
    assert model.tdt_head.joint_head.weight.grad is not None


def test_translation_examples_skip_monotonic_losses(tiny_x_config: NutqConfig) -> None:
    model = NutqForConditionalGeneration(tiny_x_config)
    batch = _batch(include_tdt=True)
    batch["auxiliary_mask"] = torch.tensor([False, False])
    output = model(**batch)

    assert output.ar_loss is not None
    assert output.ctc_loss is None
    assert output.tdt_loss is None
    assert torch.isfinite(output.loss)


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


def test_ctc_and_tdt_greedy_paths(tiny_x_config: NutqConfig) -> None:
    model = NutqForConditionalGeneration(tiny_x_config).eval()
    batch = _batch()
    ctc_tokens, ctc_confidence = model.ctc_greedy_decode(
        batch["input_features"], batch["attention_mask"]
    )
    tdt_tokens, tdt_confidence = model.tdt_greedy_decode(
        batch["input_features"], batch["attention_mask"]
    )
    assert len(ctc_tokens) == len(ctc_confidence) == 2
    assert len(tdt_tokens) == len(tdt_confidence) == 2
