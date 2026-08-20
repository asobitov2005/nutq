# Validation record

This document separates implementation verification from ASR quality evaluation. None of
the numbers below is a WER or accuracy claim.

## Automated tests

On 2026-08-20, the package passed 12 tests on Python 3.12, PyTorch 2.13 CPU, and
Transformers 5.15.1. The suite covers:

- config serialization and invalid configuration rejection;
- fixed and soft CTC compression gradients;
- combined decoder/CTC forward and backward;
- checkpoint save/load parity;
- multi-layer greedy generation with KV cache;
- Transformers AutoModel and AutoProcessor registration;
- audio bytes decoding and invalid-sample validation;
- one real `Seq2SeqTrainer` optimizer step.

A two-record public LibriSpeech dummy sample was also decoded and collated into acoustic
features of shape `[2, 80, 3000]`, a frame mask of `[2, 3000]`, and independent decoder/CTC
byte labels.

## RTX 4080 smoke tests

Hardware/software: NVIDIA GeForce RTX 4080 16 GB, driver 580.173.02, PyTorch 2.5.1+cu121,
CUDA 12.1 runtime, and Transformers 5.15.1.

Tiny-model BF16 forward, combined-loss backward, and two-layer greedy generation passed.
The real `NUTQ-180M` component initialization then passed with safetensors-only loading.

For one random 30-second-shaped tensor with five seconds marked valid:

- parameters: 179,290,497;
- CTC frames: `[1, 1500, 385]`;
- compressed memory: `[1, 42, 1472]`;
- three decoder positions: `[1, 3, 384]`;
- single cold forward: 0.224 seconds;
- reported peak allocated VRAM: 394.53 MiB.

One bridge-stage forward/backward step also passed:

- trainable parameters in bridge stage: 18,207,425;
- projector and cross-attention gradients: finite;
- single cold step: 0.262 seconds;
- reported peak allocated VRAM: 424.18 MiB.

These timing and memory readings used synthetic input, batch size one, no warmup, no data
loading, and a concurrently used GPU. They are smoke measurements, not throughput or
real-time-factor benchmarks. The bridge is untrained, so its loss and generated tokens have
no speech-recognition meaning.

