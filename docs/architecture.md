# Architecture

## Research hypothesis

Speech encoders produce far more time steps than a transcript needs. Fixed subsampling is
cheap but spends the same memory on silence and speech. Hard CTC spike selection is smaller
but can permanently drop evidence when the CTC head is uncertain. NUTQ tests a middle path:
soft pooling kernels are distributed over cumulative non-blank CTC probability.

For encoder frame `t`, let `b_t` be the CTC blank probability. Salience is
`s_t = (1 - b_t + floor) * mask_t`. The normalized cumulative mass defines an acoustic
coordinate. Output slot centers are uniform in that coordinate, and Gaussian weights pool
nearby frames. The floor keeps a gradient and evidence path before the CTC head is useful.

The default six-times compression maps Whisper's 1,500 encoder positions for a 30-second
clip to at most 250 cross-attention memory positions.

## Loss

Training minimizes:

```text
L = L_decoder_cross_entropy + λ_ctc · L_ctc
```

The decoder and CTC labels use ByT5's byte vocabulary. Special decoder tokens are excluded
from the CTC target by tokenizer metadata; no language phrases or named examples are
hardcoded.

## Initialization

- Acoustic encoder: the encoder portion of `openai/whisper-small`.
- Decoder, byte embedding and LM head: the decoder portion of `google/byt5-small`.
- Cross-attention: transferred with the ByT5 decoder and adapted to projected speech.
- CTC head and gated projector: newly initialized.

## Required ablations

Before claiming that CTC guidance helps, train identical runs with:

1. no compression;
2. fixed masked pooling;
3. soft CTC-mass pooling;
4. soft pooling without the auxiliary CTC loss.

Report WER, CER, peak VRAM, training tokens/second, real-time factor, and memory positions per
audio second. The hypothesis fails if soft pooling does not beat fixed pooling at comparable
latency and training budget.

## Known risks

- Byte sequences are longer than BPE sequences, increasing autoregressive decoding cost.
- A text-trained cross-attention block must adapt to projected acoustic states.
- CTC can be overconfident; the salience floor reduces but does not eliminate this risk.
- Whisper's reference frontend is chunked to 30 seconds; streaming needs explicit state and
  overlap handling, not merely smaller chunks.

