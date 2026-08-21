# NUTQ

[![CI](https://github.com/asobitov2005/nutq/actions/workflows/ci.yml/badge.svg)](https://github.com/asobitov2005/nutq/actions/workflows/ci.yml)

NUTQ is a Transformers-compatible multilingual ASR model family built on Whisper encoders.
It provides training, evaluation, and inference commands for three presets.

> NUTQ currently publishes source code and initialization recipes. Trained NUTQ checkpoints
> and benchmark results have not been released yet.

## Models

| Model | Encoder | Decoder | Extra head | Parameters |
|---|---|---|---|---:|
| NUTQ-S | Whisper small | 4-layer Whisper | byte CTC | 166.3M |
| NUTQ-M | Whisper medium | 4-layer Whisper | byte CTC | 428.2M |
| NUTQ-X | Whisper large-v3/Turbo | 4-layer Whisper | byte CTC + TDT | 816.7M |

S and M are compact AR/CTC models. X adds a Token-and-Duration Transducer fast path and can
fall back to its AR decoder when TDT confidence is low. See [Models](docs/models.md).

## Install

Python 3.10+ is required. Install the correct PyTorch build for the target CUDA version,
then install NUTQ:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[train]'
nutq doctor --require-gpu
```

## Quickstart

Initialize a model:

```bash
nutq init --variant m --output checkpoints/nutq-m-init
```

Train it on JSON/JSONL manifests:

```json
{"audio": "/data/000001.wav", "text": "Exact transcript.", "language": "english", "task": "transcribe"}
{"audio": "/data/000002.flac", "text": "Aniq transkript.", "language": "uzbek", "task": "transcribe"}
```

```bash
nutq train \
  --config configs/nutq-m.yaml \
  --dataset json \
  --train-files /data/train.jsonl \
  --eval-files /data/validation.jsonl \
  --stage decoder
```

Transcribe with a trained checkpoint:

```bash
nutq transcribe audio.wav --model outputs/nutq-m --strategy ar --dtype bfloat16
```

Evaluate:

```bash
nutq eval --model outputs/nutq-m --dataset ORG/DATASET --split test --strategy ar
```

## Python API

```python
from nutq_asr import NutqForConditionalGeneration, NutqProcessor
from nutq_asr.inference import NutqTranscriber

model = NutqForConditionalGeneration.from_pretrained_variant("m")
processor = NutqProcessor.from_pretrained_variant("m", language="uzbek")

asr = NutqTranscriber.from_pretrained("outputs/nutq-m", device="cuda")
text = asr.transcribe("audio.wav", strategy="ar")
```

Importing `nutq_asr` registers the model with Transformers:

```python
import nutq_asr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model = AutoModelForSpeechSeq2Seq.from_pretrained("outputs/nutq-m")
processor = AutoProcessor.from_pretrained("outputs/nutq-m")
```

## NUTQ-X decoding

```bash
nutq transcribe audio.wav --model outputs/nutq-x --strategy tdt

nutq calibrate --model outputs/nutq-x --dataset ORG/VALIDATION \
  --split validation --output calibration.json

nutq transcribe audio.wav --model outputs/nutq-x --strategy auto \
  --fallback-threshold CALIBRATED_VALUE
```

`ctc`, `tdt`, and `auto` require trained auxiliary heads. `auto` requires a threshold selected
on held-out validation data.

## Configuration

Ready-to-edit recipes are in [`configs/`](configs). Model, data, and Trainer settings can be
overridden in YAML or through the Python configuration API. See
[Configuration](docs/configuration.md) and [Training](docs/training.md).

## Documentation

- [Models](docs/models.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Multilingual data](docs/multilingual.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [Performance](docs/performance.md)
- [Validation status](docs/validation.md)

## Development

```bash
pip install -e '.[dev]'
ruff check .
mypy src
pytest
```

## License

Apache-2.0. Pretrained backbone weights and datasets retain their own licenses and terms.
