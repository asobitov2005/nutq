# Training guide

## Data contract

Every record needs audio readable by Datasets and one exact transcript. Audio is decoded and
resampled to 16 kHz. NUTQ truncates examples beyond 30 seconds in the current release.

Use speaker- and source-disjoint splits. Preserve punctuation/casing consistently, record
the normalization policy, and version the manifest. Remove corrupt audio, empty transcripts,
transcript/audio mismatches, and train/test duplicates through a data quality pipeline.

## Two-stage schedule

`bridge` is the safe first stage. It trains the new CTC head and projector plus transferred
decoder cross-attention and normalization. This establishes the modality bridge without
immediately moving every pretrained weight.

`full` unfreezes all parameters. Start from the best bridge checkpoint, lower the learning
rate (typically 2e-5 to 5e-5), and compare against a run that keeps the acoustic encoder
frozen. The repository does not hardcode a best learning rate because it depends on batch,
data diversity, and training hours.

## GPU utilization

No flag guarantees 100% GPU utilization. Measure it and remove the actual bottleneck:

- increase per-device batch size until close to the VRAM limit;
- use BF16 on supported GPUs and TF32 for FP32 matrix multiplications;
- keep data-loader workers, pinned memory, and persistent workers enabled;
- store data on local NVMe or use streaming with sufficient network throughput;
- bucket examples by duration in a future sampler to reduce padding;
- use gradient accumulation for effective batch size, not higher instantaneous GPU use;
- use gradient checkpointing only when VRAM is the limiter because recomputation costs time;
- profile before adding custom kernels.

For a 16 GB RTX 4080, begin bridge training at micro-batch 1 or 2 with BF16 and gradient
checkpointing. Raise it only after observing peak allocated VRAM. Full training has larger
optimizer/gradient memory and may require optimizer sharding, CPU offload, or a larger GPU.

## Reproducible reporting

Record the git commit, component revisions, dataset revision, manifest hash, split
construction, seed, full YAML, GPU, CUDA/PyTorch versions, precision, decoding parameters,
WER/CER, and wall-clock time. Training loss alone is not an ASR quality result.

