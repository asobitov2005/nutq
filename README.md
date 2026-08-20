# NUTQ

**Compact multilingual speech recognition, byte by byte.**

NUTQ is an experimental encoder-decoder speech recognition architecture. It connects a
pretrained Whisper speech encoder to a byte-level ByT5 decoder through CTC-guided,
compressed acoustic memory. The goal is a multilingual model that is materially smaller
than speech-to-LLM systems while keeping open-vocabulary transcription and native Unicode
coverage.

> [!IMPORTANT]
> NUTQ is currently an untrained research implementation. There is no released checkpoint
> and no benchmark claim yet. Results must be measured on speaker- and source-disjoint test
> sets before the architecture is described as accurate.

## Architecture

```text
16 kHz audio
    │
Whisper acoustic encoder
    │
auxiliary byte-level CTC head ──► CTC loss / alignment signal
    │
CTC-guided acoustic memory compressor
    │
gated hidden-size projector
    │ cross-attention
ByT5 byte decoder ───────────────► UTF-8 transcript
```

The planned `NUTQ-190M` initialization uses the encoder portion of
`openai/whisper-small` and the decoder portion of `google/byt5-small`. Neither pretrained
model is redistributed by this repository; weights are fetched from their original model
repositories when requested.

## Project status

- [x] Package and research contract
- [ ] Core Transformers model
- [ ] Data collator and streaming dataset pipeline
- [ ] Accelerate training and distributed configurations
- [ ] Generation, transcription, and evaluation CLIs
- [ ] Shape, serialization, and smoke tests
- [ ] Public trained checkpoint and reproducible benchmarks

## License

Apache-2.0. See [LICENSE](LICENSE).

