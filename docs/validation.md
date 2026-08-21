# Validation record

This file records implementation checks, not speech-recognition accuracy.

## Current source release

The automated suite covers:

- S/M/X configuration validation and serialization;
- four-layer Whisper AR forward/backward and greedy generation;
- byte CTC loss and greedy collapse;
- NUTQ-X token/duration logits, anti-diagonal TDT loss, gradients, and greedy decoding;
- checkpoint save/load parity;
- UTF-8 byte round trips across Latin and Cyrillic text;
- Transformers AutoProcessor registration;
- audio byte decoding and invalid-sample rejection;
- one real Seq2SeqTrainer optimization step;
- CLI dispatch and accelerator reporting.

Parameter counts were computed from the exact model classes on meta tensors:

| Variant | Parameters | New auxiliary parameters |
|---|---:|---:|
| NUTQ-S | 166,330,625 | 197,633 |
| NUTQ-M | 428,228,865 | 263,425 |
| NUTQ-X | 816,695,559 | 7,817,479 |

The counts are structural checks. They are not evidence of WER, throughput, robustness, or
production readiness.

## Required checkpoint validation

Before publishing weights, add a dated record containing:

- exact checkpoint and dataset revisions;
- speaker/source-disjoint split construction;
- normalized/raw WER and CER against unmodified Whisper baselines;
- CTC, TDT, AR, and calibrated-auto results where applicable;
- batch-one and batched RTF, peak VRAM, GPU, precision, and warm-up method;
- silence/noise hallucination and long-form deletion tests;
- known failed domains and language slices.

Until that record exists, README language must continue to describe NUTQ as an untrained
research implementation.
