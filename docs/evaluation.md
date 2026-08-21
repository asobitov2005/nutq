# Evaluation protocol

## Accuracy

Report both text-normalized and raw metrics:

- WER and CER;
- substitutions, deletions, and insertions;
- numeral/entity exactness;
- hallucination rate on silence and non-speech noise;
- language and domain slices;
- code-switch and long-form deletion rates.

The CLI uses the same Whisper basic normalizer for every compared system and also returns raw
WER/CER. Do not compare a normalized score from one model with a raw score from another.

```bash
nutq eval --model outputs/nutq-m --dataset ORG/DATA --split test --strategy ar
nutq eval --model outputs/nutq-x --dataset ORG/DATA --split test --strategy tdt
```

## Speed

Report the same hardware, precision, batch, audio distribution, and warm-up policy. Minimum
metrics are wall time, total audio seconds, real-time factor (`wall/audio`), peak VRAM, and
fallback rate for auto routing. Separate cold-start latency from warmed steady state.

## Routing calibration

Sweep the TDT confidence threshold only on validation data. For every threshold, calculate:

- corpus WER/CER after replacing low-confidence TDT drafts with AR outputs;
- percentage and audio duration sent to AR;
- complete pipeline RTF;
- WER change relative to always-AR.

Select a threshold from a declared constraint, such as the lowest fallback rate whose WER is
within a chosen relative tolerance of always-AR. Freeze it before running the test set. A
threshold chosen on test WER invalidates the result.

```bash
nutq calibrate --model outputs/nutq-x --dataset ORG/VALIDATION \
  --split validation --max-relative-wer-increase 0.02 \
  --output calibration.json

nutq eval --model outputs/nutq-x --dataset ORG/TEST --split test \
  --strategy auto --fallback-threshold VALUE_FROM_CALIBRATION
```

## Baseline matrix

| Model | AR WER | CTC WER | TDT WER | Auto WER | RTF | VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Whisper small | | | | | | |
| NUTQ-S | | | n/a | n/a | | |
| Whisper medium | | | | | | |
| NUTQ-M | | | n/a | n/a | | |
| large-v3-turbo | | | | | | |
| NUTQ-X | | | | | | |

No row may be filled from synthetic input, a training split, or a different normalization.
