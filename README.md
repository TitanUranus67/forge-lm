# ForgeLM

[![CI](https://github.com/TitanUranus67/forge-lm/actions/workflows/ci.yml/badge.svg)](https://github.com/TitanUranus67/forge-lm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

ForgeLM is a from-scratch GPT-style language model implementation in C# and
.NET 10. The repository includes a byte-level BPE tokenizer, transformer,
AdamW optimizer, deterministic data preparation, resumable checkpoints, text
generation, and custom CPU, NVIDIA CUDA, and Windows Direct3D 12 backends.

[ComputeSharp](https://github.com/Sergio0694/ComputeSharp) powers the optional
Windows D3D12 backend, [ILGPU](https://github.com/m4rs-mt/ILGPU) powers CUDA on
Linux and Windows, and
[Parquet.Net](https://github.com/aloneguid/parquet-dotnet) reads FineWeb parquet
shards during data preparation.

## Forge-98M

The first released model is **Forge-98M**, a 97,934,592-parameter base language
model trained on 1.024 billion FineWeb-Edu tokens. It is an experimental text
completion model, not an instruction-following assistant.

- [Download the checkpoint and tokenizer](https://github.com/TitanUranus67/forge-lm/releases/latest)
- [Read the model card](MODEL_CARD.md)
- Checkpoint SHA-256: `563a112a7b8ab61e95e5ffe47968896362d59831532d7369ed008be015828481`

## Build & test

```sh
dotnet build ForgeLM.slnx
dotnet run --project tests/LLM.Core.Tests   # CUDA/D3D12 tests skip cleanly when unavailable
```

## Quickstart

The CLI has seven subcommands (tokenizer training is folded into `prepare`):

```sh
# 1. Download tiny-shakespeare, train a BPE tokenizer, encode to train/val bins
dotnet run --project src/LLM.Cli -- prepare --out data/shakes --merges 2000

# 1b. Or build a real corpus: download N FineWeb-Edu (sample-10BT) shards,
#     isolate validation by URL, train tokenizer, and stream-encode (takes hours)
dotnet run --project src/LLM.Cli -- prepare-fineweb --out data/forge --shards 3 --merges 16000

# Build a deterministic weighted dataset from compatible prepared sources
dotnet run --project src/LLM.Cli -- prepare-mixture \
    --manifest docs/forge-next-mixture.example.json --out data/forge-next/mixed

# Train Forge-98M for the fixed 1.024B-token budget; benchmark the Vast GPU first
# and override batch/accum/matmul with its winning configuration
dotnet run -c Release --project src/LLM.Cli -- train --preset forge-98m \
    --backend cuda --data data/forge --tokens 1024000000 --warmup-tokens 4096000 \
    --lr 3e-4 --minlr 3e-5 --logevery 16 --valevery 320 --valbatches 50 \
    --saveevery 320

# 2. Train a small GPT (checkpoint written to out/model.bin)
dotnet run --project src/LLM.Cli -- train --data data/shakes --steps 5000 --out out/model.bin

# 2b. Explicit NVIDIA CUDA (Linux or Windows)
dotnet run --project src/LLM.Cli -- train --data data/shakes --steps 5000 --out out/model.bin --backend cuda

# 2c. Explicit Windows D3D12 compute shaders
dotnet run --project src/LLM.Cli -- train --data data/shakes --steps 5000 --out out/model.bin --backend gpu

# 3. Generate text from the checkpoint
dotnet run --project src/LLM.Cli -- generate --model out/model.bin --tokenizer data/shakes \
    --prompt "Once upon a time" --tokens 200 --temperature 0.7 --topk 30 \
    --repetition-penalty 1.1 --no-repeat-ngram 4

# 4. Or chat with it interactively (base model: it continues your text, it won't answer questions)
dotnet run --project src/LLM.Cli -- chat --model out/model.bin --tokenizer data/shakes

# Benchmark the full training shape without reading data or writing a checkpoint
dotnet run -c Release --project src/LLM.Cli -- benchmark --backend cuda \
    --preset forge-220m --matmul-precision fp32 --batch 1 --accum 32 --steps 3
```

All flags are optional where a default exists; run any command with `--help` for
the full list (model size, learning-rate schedule, sampling knobs, resuming from
a checkpoint with `--init`, etc.). `train`, `generate` and `chat` accept
`--backend auto|cpu|gpu|cuda` (default `auto`). Auto tries CUDA first, then
D3D12 on Windows, and always falls back to CPU. Explicit accelerator choices
fail loudly if unavailable. The selected backend and device are printed at startup.

- `prepare` accepts a local file or an http(s) URL (default: tiny-shakespeare),
  trains or loads a tokenizer (`tokenizer.json`), encodes the corpus, and writes
  a 90/10 split of raw little-endian uint16 token files (`train.bin`, `val.bin`).
- `prepare-fineweb` builds a large-scale corpus from
  [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  (`sample-10BT` config) by default; `--dataset fineweb` selects unfiltered
  FineWeb. It lists parquet shards via the Hugging Face API, downloads the
  first `--shards N` (default 10) into `<out>/shards/<dataset>/` (existing
  complete shards are skipped, so interrupted runs resume), extracts the `text`
  column into `<out>/corpus.txt` with a document-length index at
  `<out>/corpus.idx` via
  [Parquet.Net](https://github.com/aloneguid/parquet-dotnet) (CLI-only
  dependency). A stable URL hash assigns whole documents to train or validation,
  preventing the same URL from crossing splits. The tokenizer trains on a
  deterministic corpus-wide sample of training documents (`--toktrainmb`, default
  200 MB), and BPE merges cannot cross document boundaries. Preparation then
  stream-encodes every document followed by EOS — a multi-GB
  corpus never has to fit in memory. Encoding is the slow
  phase (tens of minutes to hours depending on merges/corpus size); progress in
  MB is printed per chunk. Corpus, tokenizer, and bin artifacts are published
  transactionally with provenance manifests; stale or incomplete artifacts are
  never silently reused. `--rebuild true` regenerates derived artifacts while
  retaining downloaded shards.
- `prepare-mixture` combines two or more prepared sources that use byte-identical
  tokenizers. A versioned JSON specification supplies source directories, positive
  weights, and train/validation token targets. The mixer deterministically balances
  emitted tokens while copying complete EOS-terminated documents, refuses insufficient
  or stale inputs, and records provenance in `.forge-mixture.json`. FineWeb preparation
  accepts `--tokenizer` so every source can share the tokenizer trained on the primary
  corpus. `--exclude-index` removes stable document identities already present in a
  prior `corpus.idx`, avoiding overlap when a derived corpus is mixed with its parent.
  See [the next-run plan](docs/forge-next-run.md) for the reviewed 80/20 mix.
- `train` runs a batched training loop. Each physical `[B*T, C]` pass contains
  `--batch N` sequences (default 8); the CLI defaults to `--accum 16`, averaging
  16 physical-batch gradients before clipping and one Adam/LR update. `--tokens`
  and `--warmup-tokens` convert desired token budgets to optimizer-update counts,
  preventing accumulation from silently multiplying total corpus exposure or
  warmup length. Training uses warmup + cosine LR decay and periodic validation.
  Validation is the
  forward-only mean over 50 deterministic physical-size batches by default
  (`--valbatches`/`--valseed`), evaluated in microbatches so it does not raise
  peak VRAM or perturb the training sampler. Validation cadence is independent
  of logging cadence. On an interactive
  console an in-place progress bar shows percent, step, loss, rolling tok/s,
  ETA and elapsed time (it falls back to plain log lines when output is piped);
  pressing `p` pauses (resume, save+resume, or save+quit). Checkpoints are written
  at the end, every `--saveevery N` steps, and on Ctrl+C (the run finishes its
  current step, then saves) — an interrupted run is never lost. Training
  checkpoints preserve the global step, Adam moments, LR schedule, and complete
  no-replacement sampler position, so `--init` continues the same trajectory instead
  of restarting warmup or repeating data. Saves include a SHA-256 trailer, bind the
  checkpoint to its tokenizer and training data, and rotate the previous generation
  to `<checkpoint>.bak`. Startup hashes the input files before an optimizer update.
- `generate` loads a checkpoint and samples autoregressively with temperature,
  top-k filtering, an opt-in repetition penalty, and an opt-in no-repeat n-gram
  constraint. The raw defaults leave both repetition controls disabled so model
  evaluations remain comparable. An incremental UTF-8 decoder preserves characters whose
  bytes span token boundaries. Generation stops when EOS is sampled or when
  the `--tokens` safety limit is reached.
- `chat` is an interactive REPL over a checkpoint: each line you type is appended
  to a rolling context the model continues. `/reset` clears context, `/quit` exits.
- `benchmark` performs one unmeasured warmup update and then times synthetic
  full-shape training. It is the preflight tool for choosing a physical batch on
  the exact GPU that will host a run; it never reads data or writes a checkpoint.
  `forge-98m`, `forge-220m`, and `forge-320m` presets are shared with `train`, so
  benchmark and production shapes cannot drift. See the
  [next-run plan](docs/forge-next-run.md) for the reviewed larger-model matrix.

## Architecture

- **BPE tokenizer** (`Tokenizer/BpeTokenizer.cs`) — byte-level, GPT-2 style.
  Ids 0–255 are raw bytes, learned merges produce ids 256+, and every tokenizer
  appends a dedicated EOS id. Encoding is rank-greedy; arbitrary bytes remain
  representable and valid UTF-8 text round-trips losslessly.
- **GPT model** (`Model/`) — GPT-2 architecture: learned token + positional
  embeddings, pre-LN transformer blocks (multi-head causal self-attention +
  GELU MLP), final LayerNorm, and an output projection tied to the token embedding
  table. Training is batched: B
  sequences of length T are stacked row-wise into `[B*T, C]` tensors and
  processed in one pass (attention never crosses sequence boundaries); the
  (sequence, head) attention slots are packed into slot-contiguous tensors and
  processed with batched kernels. Inference runs one sequence at a time; there
  is no KV cache.
- **Training** (`Training/`) — AdamW (decoupled weight decay on 2-D params),
  averaged gradient accumulation, linear-warmup + cosine-decay in optimizer-update
  units, global gradient-norm clipping, and a checkpointable O(1)-memory affine
  sampler that visits every complete context window once before reshuffling.
- **CPU backend** (`Tensor/CpuBackend.cs`) — all math through an
  `ITensorBackend` interface; the CPU implementation uses `Vector<T>` SIMD in
  the matmul inner loops. Large matmuls parallelize over output rows with
  `Parallel.For` and pooled buffers (no steady-state allocation); small ones
  run sequentially.
- **D3D12 backend** (`Tensor/Gpu/`) — a second `ITensorBackend` implementation
  running D3D12 compute shaders written in C# via ComputeSharp (Windows only;
  the rest of the library stays cross-platform). One shader per kernel, fp32
  throughout; matmuls use 16×16 groupshared tiles (~950 GFLOPS on an RTX 2080
  vs ~30 for the naive one-thread-per-output version), attention uses batched
  3-D-dispatched matmuls over packed (sequence, head) slots, and the embedding
  backward scatter-add uses compare-exchange atomics on a bit-cast int copy
  (HLSL has no float `InterlockedAdd` on typed buffers). Tensors stay resident
  on the GPU: each `Tensor` gets a device allocation cached in
  `Tensor.DeviceResource`, sub-allocated from shared 64 MB arenas because
  ComputeSharp caps live buffer objects at 2048. Two interface hooks keep host
  and device coherent: kernels never update `Tensor.Data` (device is
  authoritative until `EnsureHostCurrent`), and host-side writes (optimizer,
  gradient clipping, checkpoint load) must call `InvalidateDeviceCache`.
  Measured at the Forge-98M shape (dmodel 768, 12 layers, ctx 512): see below.
- **CUDA backend** (`Tensor/Cuda/`) — an ILGPU implementation for NVIDIA GPUs
  on Linux and Windows. It covers the complete training interface: tiled fp32
  matmuls, batched attention, normalization, softmax/cross-entropy, embedding
  scatter atomics, AdamW, gradient reductions, and host/device synchronization.
  Tensor storage is sub-allocated from reusable 64 MB CUDA arenas and remains
  device-resident; tests enforce numerical agreement with CPU, full-model
  gradients, overfit behavior, aliasing, and a steady-state allocation plateau.
- **Checkpoints** (`Checkpoint/Checkpoint.cs`) — the `FORGEMODEL1` custom binary
  format stores the tied-model architecture, cumulative step, schedule/configuration,
  complete sampler position, Adam age and moments, input identities, and a SHA-256
  checksum. Loading validates the exact payload size, architecture, and parameter registry.
  Checkpoints are backend-agnostic: train on GPU, generate on CPU or vice versa.

## Accelerator backend notes

- `--backend cuda` requires an NVIDIA driver visible to the process and works
  on Linux or Windows. ILGPU JIT-compiles the C# kernels to CUDA PTX at backend
  startup, so the first launch takes longer than subsequent operations.
- `--backend gpu` requires Windows with a D3D12 device (any recent NVIDIA/AMD/Intel driver;
  WARP software devices also work for correctness testing). Without one, the
  GPU tests skip and `--backend gpu` exits with a clear error.
- ForgeLM targets .NET 10. Use the SDK version selected by `global.json` or a
  newer compatible .NET 10 feature band.
- Reference throughput on an RTX 2080 8 GB, Forge-98M scale
  (`--dmodel 768 --layers 12 --heads 12 --ctx 512 --batch 8`, 4096 tokens/step):
  CPU ≈ 100 tok/s, GPU ≈ 950 tok/s (~9×). Device memory usage is ~5.5 GB.
- A controlled 100-step RTX 2080 comparison at the previous untied 110M shape
  (`768/12/12`, ctx 512, batch 4, fp32) reached 1,002 tok/s with CUDA versus
  922 tok/s with D3D12. Loss and validation values matched at every logged gate.
- Checkpoints are backend-agnostic, so a checkpoint created by the D3D12 or CPU
  backend can resume directly under CUDA on a Linux server.

For a self-contained Linux publish and cloud resume checklist, see
[docs/linux-cuda.md](docs/linux-cuda.md).

## Data and licensing

ForgeLM source code and the released Forge-98M weights are licensed under
[Apache-2.0](LICENSE). Forge-98M was trained on
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), which
is distributed under ODC-By 1.0 and derived from Common Crawl. Exact source
shards, preparation settings, and artifact hashes are recorded in
[data/forge/SOURCE.md](data/forge/SOURCE.md). Users remain responsible for
reviewing generated output and complying with applicable dataset and content
terms.

## Project layout

```
src/LLM.Core/     the library (tokenizer, tensors, backends, model, training, inference, checkpoints)
src/LLM.Cli/      the command-line front end (prepare / train / generate / chat)
tests/LLM.Core.Tests/   hand-rolled test harness (dotnet run)
```
