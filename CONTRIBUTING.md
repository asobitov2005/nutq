# Contributing to NUTQ

NUTQ is an early research project. Before a large change, open an issue describing the
hypothesis, expected impact, data, and evaluation protocol.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
ruff check .
pytest
```

Do not report quality improvements from training loss alone. Include WER/CER, the exact
dataset revision and split, decoding parameters, hardware, random seed, and a baseline.

