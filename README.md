# NUTQ

**Compact multilingual speech recognition, byte by byte.**

NUTQ is an experimental encoder-decoder automatic speech recognition architecture. It
connects a pretrained Whisper acoustic encoder to a byte-level ByT5 decoder through
CTC-guided compressed acoustic memory. It is designed to retain multilingual Unicode
coverage without putting every audio frame into a large language model context.

> [!IMPORTANT]
> NUTQ is currently an untrained research implementation. There is no released accuracy
> claim yet. The code runs end to end; meaningful transcripts require training and a
> speaker/source-disjoint evaluation.

## NUTQ-180M

The default configuration has exactly **179,290,497 parameters before freezing**:

| Component | Parameters | Initialization |
|---|---:|---|
| Whisper acoustic encoder | 88,154,112 | `openai/whisper-small` encoder |
| ByT5 decoder and byte embedding | 81,980,288 | `google/byt5-small` decoder |
| Gated projector | 8,860,032 | new |
| Byte CTC head | 296,065 | new |

```text
16 kHz audio
    │
Whisper acoustic encoder
    │
auxiliary byte CTC head ───────► alignment signal + CTC loss
    │
soft CTC-mass memory pooling ──► ~6x fewer acoustic positions
    │
gated 768 → 1472 projector
    │ cross-attention
4-layer ByT5 byte decoder ─────► UTF-8 transcript
```

See [Architecture](docs/architecture.md) for the algorithm and research risks.

## Install

Python 3.10+ and PyTorch 2.4+ are required. Install PyTorch for the target CUDA version
first, then NUTQ:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[train]'
```

For CPU-only development, use PyTorch's `/whl/cpu` index. For contributors:

```bash
pip install -e '.[dev]'
ruff check .
pytest
```

## Initialize a checkpoint

This copies only the Whisper encoder and ByT5 decoder weights. The CTC head, projector,
and modality adaptation still need training.

```bash
nutq-init --output checkpoints/nutq-180m-init
```

Python API:

```python
from nutq_asr import NutqForConditionalGeneration, NutqProcessor

model = NutqForConditionalGeneration.from_pretrained_components()
processor = NutqProcessor.from_pretrained_components()
model.save_pretrained("checkpoints/nutq-180m-init")
processor.save_pretrained("checkpoints/nutq-180m-init")
```

NUTQ registers with Transformers when `nutq_asr` is imported:

```python
import nutq_asr  # registers the custom architecture
from transformers import AutoModelForSpeechSeq2Seq

model = AutoModelForSpeechSeq2Seq.from_pretrained("checkpoints/nutq-180m-init")
```

## Dataset format

Training requires paired audio and exact transcripts. JSON/JSONL files work directly:

```json
{"audio": "/data/audio/000001.wav", "text": "Exact transcript."}
{"audio": "/data/audio/000002.flac", "text": "Keyingi transkript."}
```

Keep train, validation, and test speakers and recording sources disjoint. A duplicate clip
or speaker across splits makes WER look better without improving the model.

## Train

Stage 1 trains CTC, the projector, decoder cross-attention, and decoder normalization while
freezing the transferred backbones:

```bash
accelerate launch --config_file configs/accelerate/1gpu.yaml \
  -m nutq_asr.cli.train \
  --config configs/nutq-180m.yaml \
  --dataset json \
  --train-files /data/train.jsonl \
  --eval-files /data/validation.jsonl \
  --stage bridge
```

Stage 2 unfreezes the full network. Point `--checkpoint` at the best stage-1 model:

```bash
accelerate launch --config_file configs/accelerate/1gpu.yaml \
  -m nutq_asr.cli.train \
  --config configs/nutq-180m.yaml \
  --dataset json \
  --train-files /data/train.jsonl \
  --eval-files /data/validation.jsonl \
  --checkpoint outputs/nutq-180m/checkpoint-best \
  --stage full
```

Hub datasets use the same CLI. A small pipeline check can use
`--dataset openslr/librispeech_asr --dataset-config clean --max-train-samples 16
--max-eval-samples 8`; it is a smoke test, not a benchmark.

For four GPUs, switch to `configs/accelerate/fsdp-4gpu.yaml`. DeepSpeed ZeRO-2 settings are
in `configs/deepspeed/zero2.json`. See [Training](docs/training.md) before a long run.

## Inference and evaluation

```bash
nutq-transcribe sample.wav --model outputs/nutq-180m --dtype bfloat16 --compile

nutq-eval \
  --model outputs/nutq-180m \
  --dataset openslr/librispeech_asr \
  --dataset-config clean \
  --split test.clean
```

Or use the library:

```python
from nutq_asr.inference import NutqTranscriber

asr = NutqTranscriber.from_pretrained("outputs/nutq-180m", device="cuda")
print(asr.transcribe("sample.wav", num_beams=1))
```

## Performance direction

The current reference backend is PyTorch with SDPA, BF16, KV caching and optional
`torch.compile`. Native serving is intentionally staged: first a Triton Inference Server
Python backend, then an exported/fused runtime, and finally custom Triton/CUDA kernels only
for measured bottlenecks. See [Performance roadmap](docs/performance.md).

## License

Apache-2.0. NUTQ does not redistribute pretrained component weights; review each model and
dataset license before releasing a derivative checkpoint.

