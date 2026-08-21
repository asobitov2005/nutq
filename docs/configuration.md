# Configuration

NUTQ recipes contain `model`, `data`, and `training` sections. Start from one of:

```text
configs/nutq-s.yaml
configs/nutq-m.yaml
configs/nutq-x.yaml
```

## Model

Common fields:

| Field | Description |
|---|---|
| `variant` | `s`, `m`, or `x` |
| `backbone_name` | Local path or Hugging Face Whisper checkpoint |
| `language` | Optional Whisper language prefix; `null` enables detection |
| `task` | `transcribe` or `translate` |
| `ar_loss_weight` | Whisper decoder loss weight |
| `ctc_loss_weight` | Byte CTC loss weight |

NUTQ-X fields:

| Field | Description |
|---|---|
| `tdt_loss_weight` | TDT loss weight |
| `tdt_hidden_size` | Predictor and joiner width |
| `tdt_num_layers` | LSTM predictor depth |
| `tdt_subsampling_factor` | Post-encoder reduction: 1, 2, 4, or 8 |
| `tdt_durations` | Supported frame durations, beginning with zero |
| `tdt_sigma` | TDT logit under-normalization constant |
| `auto_fallback_threshold` | Optional pre-calibrated routing threshold |

All inherited `WhisperConfig` arguments are accepted by `NutqConfig`. Changing dimensions
while loading pretrained weights requires shape-compatible checkpoints.

## Data

| Field | Description |
|---|---|
| `sampling_rate` | Target audio rate; presets use 16 kHz |
| `max_audio_seconds` | Maximum training clip length |
| `max_label_length` | Maximum AR label length |
| `long_audio_policy` | `reject` or explicit `truncate` |
| `audio_column` | Dataset audio column |
| `text_column` | Dataset transcript column |
| `language_column` | Optional per-example Whisper language field |
| `task_column` | Optional per-example `transcribe`/`translate` field |
| `default_language` | Language used when no per-example value exists |
| `default_task` | Task used when no per-example value exists |

See [Multilingual data and decoding](multilingual.md) for single-language, mixed-language,
code-switch, and translation manifests.

## Training

The `training` section is passed to `Seq2SeqTrainingArguments`. This keeps optimizer,
precision, batching, logging, checkpointing, distributed, and reporting options compatible
with Transformers. Refer to the Transformers API for the complete field list.

Common overrides:

```yaml
training:
  output_dir: outputs/experiment
  learning_rate: 0.00008
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  bf16: true
  tf32: true
  gradient_checkpointing: true
  group_by_length: true
  save_steps: 1000
  eval_steps: 1000
```

## CLI

```bash
nutq COMMAND --help
```

The CLI controls dataset locations, splits, checkpoint resume, training stage, device,
precision, decoding strategy, beam size, and evaluation limits. Architecture and Trainer
options remain in versioned YAML so experiments are reproducible.
