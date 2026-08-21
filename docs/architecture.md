# Architecture

## Design goals

NUTQ prioritizes transcription accuracy, then reduces latency without forcing every example
through the fastest decoder. The design keeps a modality-aligned Whisper encoder/decoder
interface and learns auxiliary monotonic paths from paired audio and text.

The current family keeps Whisper byte-BPE for AR output and limits raw bytes to compact
CTC/TDT heads. Encoder states reach the decoder directly, preserving the pretrained
audio-to-text interface.

## Shared AR and CTC paths

For encoder states `h_t`, the CTC head produces logits over 256 byte values plus blank:

```text
p_ctc(z_t | x) = softmax(W_ctc h_t)
```

The AR decoder is a four-layer Whisper decoder initialized from the matching backbone. Its
cross-attention consumes the original encoder states without late compression or a hidden-
size projector.

S/M training minimizes:

```text
L = λ_ar L_ar + λ_ctc L_ctc
```

CTC is an auxiliary monotonic alignment objective and an independently measurable fast
draft. It does not change the AR decoder context.

## NUTQ-X TDT path

NUTQ-X additionally reduces the final encoder sequence by four with learned stride-two
convolutions. A recurrent prediction network represents previous emitted bytes. The joiner
combines acoustic and predictor states and emits two independently normalized distributions:

```text
P(token, duration | t, u) = P_token(token | t, u) · P_duration(duration | t, u)
```

Blank emissions cannot use duration zero. Nonblank emissions may use zero and emit multiple
symbols at one acoustic position, bounded during greedy inference to prevent an infinite
loop. Training uses durations `[0, 1, 2, 3, 4]` and the TDT forward loss:

```text
L = λ_ar L_ar + λ_ctc L_ctc + λ_tdt L_tdt
```

Reference: [Token-and-Duration Transducer](https://arxiv.org/abs/2304.06795).

## Confidence routing

NUTQ-X exposes three independent hypotheses:

1. CTC greedy bytes;
2. TDT greedy bytes with mean emitted-token confidence;
3. Whisper AR transcript.

In `auto` mode, a threshold selects the TDT draft for high-confidence utterances and runs
the AR decoder otherwise. The threshold must be calibrated on a held-out set against a
declared WER/latency objective. It must not depend on named phrases, document titles,
locations, expected answers, or language-specific regex branches.

The current fallback retranscribes the complete uncertain utterance. Span-only correction
requires a trained conditioning interface and is intentionally not faked with string rules.

## Required ablations

- unmodified Whisper backbone;
- four-layer AR without CTC;
- four-layer AR with CTC;
- X with CTC but no TDT loss;
- X TDT-only decoding;
- X auto routing at every reported fallback rate;
- S, M, and X under the same normalized evaluation protocol.

Paper speedups do not multiply. Measure the complete system, including frontend, encoder,
decoder, data transfer, and routing overhead.
