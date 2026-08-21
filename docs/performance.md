# Performance roadmap

## Reference backend

The correctness implementation uses PyTorch/Transformers with BF16 or FP32, scaled-dot-
product attention, AR KV caching, inference mode, optional `torch.compile`, batched dataset
evaluation, and explicit CTC/TDT/AR strategies.

NUTQ-X TDT training uses Transformers' anti-diagonal PyTorch TDT loss. Profile it before a
long run; an optimized CUDA/Numba loss backend is a separate parity-tested optimization.

## Phase 1: remove fixed-window waste

Whisper currently requires 3,000 mel positions and produces 1,500 encoder positions for
every item. Duration grouping helps batching but a two-second recording still pays the
30-second encoder cost. Dynamic length requires:

1. slicing positional embeddings to the post-convolution sequence length;
2. propagating a correct bidirectional encoder mask;
3. passing the reduced mask into decoder cross-attention;
4. distillation/fine-tuning after changing the pretrained execution distribution;
5. parity and WER tests at multiple durations.

This must be implemented as an architecture change, not by merely disabling padding.

## Phase 2: optimized Python serving

- compile stable AR and encoder graphs separately;
- use static KV cache where supported;
- batch by strategy and duration;
- keep a warmed model pool for S/M/X;
- export Prometheus-compatible latency, fallback, and error metrics;
- serve through Triton Inference Server's Python backend only after reference parity.

## Phase 3: exported/fused runtime

Export the encoder, CTC head, TDT subsampler/joiner, and one AR decoder step independently.
Compare ONNX Runtime/TensorRT against PyTorch using transcript equality, logit error,
first-token latency, per-token latency, complete RTF, and peak memory.

Only implement custom Triton/CUDA kernels for profiler-proven bottlenecks. Likely candidates
are TDT joint/loss materialization and small-step decoder launch overhead—not the already
optimized matrix multiplications.

## Phase 4: compression

Evaluate weight-only INT8/INT4 and activation quantization separately for encoder, AR, and
TDT. A release requires per-language/noise WER regression tables, not only lower VRAM.

## Non-goals

- claiming native/TensorRT support before numerical parity;
- multiplying isolated paper speedups;
- calling independent chunks streaming while resetting all state;
- selecting routing thresholds with semantic phrase rules;
- quantizing before a trained FP16/BF16 accuracy baseline exists.
