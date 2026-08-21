# Training guide

## Data contract

Every example must provide a non-empty audio recording and an exact transcript. Audio is
decoded with SoundFile, converted to mono, resampled to 16 kHz, and checked for non-finite
values. Recordings over 30 seconds are rejected by default; set `long_audio_policy: truncate`
only when deliberate truncation matches the transcript policy.

Before training:

- remove corrupt, empty, clipped, and transcript-mismatched records;
- deduplicate audio and near-duplicate transcripts across splits;
- split by speaker and recording source;
- version manifests, normalization rules, and dataset revisions;
- balance languages/domains with a sampling policy rather than phrase exceptions;
- build silence, noise, numeral, entity, code-switch, and long-form test slices.

For mixed-language data, set `language_column` and store a valid Whisper language name/code
in every manifest record. See [Multilingual data](multilingual.md).

## Initialization

```bash
nutq init --variant m --output checkpoints/nutq-m-init
```

S/M select four decoder blocks from a deeper Whisper teacher and need distillation or paired
ASR training. X copies the existing four Turbo decoder blocks, preserving the pretrained AR
path before its new heads are trained.

## Stages

| Stage | Trainable parameters | Intended use |
|---|---|---|
| `heads` | CTC and optional TDT | First NUTQ-X alignment stage |
| `decoder` | heads + four-layer AR decoder | First S/M stage |
| `top` | decoder/heads + last N encoder blocks | Domain/acoustic adaptation |
| `full` | complete network | Final large-data stage only |

Example S/M schedule:

```bash
nutq train --config configs/nutq-m.yaml --dataset json \
  --train-files /data/train.jsonl --eval-files /data/valid.jsonl --stage decoder

nutq train --config configs/nutq-m-top.yaml --dataset json \
  --train-files /data/train.jsonl --eval-files /data/valid.jsonl \
  --checkpoint outputs/nutq-m/checkpoint-best --stage top \
  --encoder-unfreeze-layers 4
```

Example X first stage:

```bash
nutq train --config configs/nutq-x.yaml --dataset json \
  --train-files /data/train.jsonl --eval-files /data/valid.jsonl --stage heads
```

Do not enable `auto` routing until CTC/TDT have been trained and the threshold has been
calibrated on a separate validation set.

## Loss schedule

The YAML defaults are starting points, not accuracy claims:

```text
λ_ar  = 1.0
λ_ctc = 0.3
λ_tdt = 0.3 (X only)
```

Track each loss separately. Change one objective at a time and keep total training data,
updates, seed, and decoding constant. If CTC/TDT dominates gradients, warm up the auxiliary
weight or clip gradients based on measured norms.

Recommended distillation extension for S/M:

```text
L_total = L_hard + λ_kd KL(student || teacher) + λ_ctc L_ctc
```

Use Whisper large-v3 or the matching full Whisper decoder as a teacher. Teacher-generated
labels must not leak evaluation recordings into training.

## GPU utilization

No flag guarantees 100% utilization. Measure the bottleneck:

- maximize micro-batch until close to the VRAM limit;
- use BF16 on supported GPUs and TF32 for eligible FP32 matrix operations;
- keep pinned memory, persistent workers, and local NVMe data enabled;
- use duration grouping to reduce batch imbalance;
- use gradient accumulation for effective batch, not instantaneous utilization;
- use gradient checkpointing only when memory is the limiter;
- use FSDP/ZeRO for M/X full-network optimizer states;
- profile before writing Triton kernels.

The 16 GB examples in the YAML are conservative. NUTQ-X full training will usually require
sharding/offload or a larger accelerator even at batch one.

## Reproducibility

Record the git commit, backbone revision, manifest hash, split construction, seed, YAML,
precision, GPU, CUDA/PyTorch/Transformers versions, optimizer, effective batch, decoding
parameters, normalized/raw WER/CER, RTF, peak VRAM, and wall-clock time.
