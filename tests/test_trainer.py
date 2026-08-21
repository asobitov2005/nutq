from __future__ import annotations

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from nutq_asr import NutqConfig, NutqForConditionalGeneration


class TinySpeechDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        token = 3 + index
        return {
            "input_features": torch.randn(8, 16),
            "attention_mask": torch.ones(16, dtype=torch.long),
            "labels": torch.tensor([token, token + 1, 1, -100]),
            "ctc_labels": torch.tensor([token, token + 1, -100]),
        }


def test_seq2seq_trainer_one_step(tmp_path, tiny_config: NutqConfig) -> None:
    model = NutqForConditionalGeneration(tiny_config)
    model.set_trainable("heads")
    before = model.ctc_head.weight.detach().clone()
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        max_steps=1,
        per_device_train_batch_size=2,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels", "ctc_labels"],
    )
    trainer = Seq2SeqTrainer(model=model, args=arguments, train_dataset=TinySpeechDataset())

    result = trainer.train()

    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert not torch.equal(before, model.ctc_head.weight)
