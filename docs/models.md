# Model family

## NUTQ-S

NUTQ-S is the deployable baseline. It copies the Whisper-small encoder, token embedding,
positional embedding, LM projection, final decoder normalization, and four maximally spaced
decoder blocks. A 257-class byte CTC head is new.

| Property | Value |
|---|---|
| Encoder | 12 layers, width 768, 12 attention heads |
| Decoder | 4 layers, width 768 |
| AR tokenizer | Whisper multilingual byte-BPE |
| CTC alphabet | 256 UTF-8 byte values + blank |
| Parameters | 166,330,625 |

Use S for edge/server deployment experiments and as the minimum baseline every larger
variant must beat at a justified latency cost.

## NUTQ-M

NUTQ-M is the default accuracy/speed research model. It uses the same objectives as S but
copies the stronger Whisper-medium acoustic encoder.

| Property | Value |
|---|---|
| Encoder | 24 layers, width 1024, 16 attention heads |
| Decoder | 4 layers, width 1024 |
| AR tokenizer | Whisper multilingual byte-BPE |
| CTC alphabet | 256 UTF-8 byte values + blank |
| Parameters | 428,228,865 |

Its four selected decoder layers are an initialization, not a distilled checkpoint. Train
and compare it against the unmodified Whisper-medium and a domain-adapted Whisper baseline.

## NUTQ-X

NUTQ-X is the flagship hybrid. It initializes its AR path from large-v3-turbo and adds:

- a full-rate byte CTC head;
- learned 4× temporal reduction after the Whisper encoder;
- a two-layer byte predictor;
- independent token and duration logits with durations `[0, 1, 2, 3, 4]`;
- greedy TDT inference and a validation-calibrated AR fallback.

| Property | Value |
|---|---|
| Encoder | 32 layers, width 1280, 20 attention heads |
| AR decoder | 4 layers, width 1280 |
| TDT predictor | 2 layers, width 512, UTF-8 bytes |
| TDT reduction | 4× after the encoder |
| Parameters | 816,695,559 |

The TDT head predicts a token and a duration distribution independently at every lattice
state, and training uses the anti-diagonal TDT forward algorithm supplied by Transformers.
The fast path is label-autoregressive; duration prediction makes it skip acoustic frames,
but does not make it fully non-autoregressive.

## Initialization policy

For source decoders deeper than four layers, NUTQ chooses four indices evenly across depth.
This is deterministic architectural initialization, not a semantic routing rule. The
selection must be compared with alternative layer selections or teacher distillation before
claiming it is optimal.

All component downloads require safetensors. Newly added heads use the backbone's standard
initialization and therefore have no transcription ability before training.
