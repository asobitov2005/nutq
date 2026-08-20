# Performance roadmap

## Available now: PyTorch reference

- BF16/FP32 model execution;
- PyTorch scaled-dot-product attention selected by Transformers;
- autoregressive KV caching during generation;
- `torch.inference_mode()` and optional `torch.compile(mode="reduce-overhead")`;
- dynamic audio resampling and a Python CLI/API.

This path is the correctness oracle for optimized backends.

## Phase 1: production Python serving

Package the reference runtime as a Triton Inference Server Python backend with dynamic
batching, explicit audio/input limits, health metrics, and a pinned execution environment.
This improves scheduling and operations; it does not by itself fuse NUTQ kernels.

## Phase 2: exported graph

Export and validate encoder, compressor/projector, and one decoder step separately. Compare
ONNX Runtime/TensorRT against PyTorch on transcript equality, maximum logit error, first-token
latency, steady-state token latency, throughput, VRAM, and real-time factor. Generation stays
host-orchestrated until an engine-native loop is proven correct.

## Phase 3: measured native kernels

Profile representative batch/length distributions. Candidate fusion boundaries are soft
CTC pooling, gated projection, and decoder-step launch overhead. Implement a Triton kernel
only when a stable bottleneck is demonstrated. Keep a pure PyTorch fallback and numerical
tests for every fused operation.

## Non-goals for the first release

- claiming TensorRT support before export parity tests pass;
- a custom CUDA kernel that is slower than SDPA or vendor libraries;
- fake streaming that resets all state on each chunk;
- quantization without per-language and noisy-speech regression evaluation.

