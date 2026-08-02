# LLM_Scratch

A mini-GPT built from scratch in C# (.NET 9). Everything (BPE tokenizer,
transformer, AdamW, SIMD tensor backend, checkpoint format) is implemented on
top of the Base Class Library. The core library has exactly one NuGet
dependency: [ComputeSharp](https://github.com/Sergio0694/ComputeSharp) 3.2.0,
which powers the optional D3D12 GPU backend. The CLI additionally uses
[Parquet.Net](https://github.com/aloneguid/parquet-dotnet) to read FineWeb
parquet shards for `prepare-fineweb`.

## Build & test

```sh
dotnet build
dotnet run --project tests/LLM.Core.Tests   # 87 tests (GPU tests skip cleanly without a D3D12 device)
```

## Quickstart

The CLI has five subcommands (tokenizer training is folded into `prepare`):

```sh
# 1. Download tiny-shakespeare, train a BPE tokenizer, encode to train/val bins
dotnet run --project src/LLM.Cli -- prepare --out data/shakes --merges 2000

# 1b. Or build a real corpus: download N FineWeb (sample-10BT) shards from
#     Hugging Face, extract text, train tokenizer, stream-encode (takes hours)
dotnet run --project src/LLM.Cli -- prepare-fineweb --out data/fineweb --shards 10 --merges 16000

# 2. Train a small GPT (checkpoint written to out/model.bin)
dotnet run --project src/LLM.Cli -- train --data data/shakes --steps 5000 --out out/model.bin

# 2b. Or train on the GPU (D3D12 compute shaders)
dotnet run --project src/LLM.Cli -- train --data data/shakes --steps 5000 --out out/model.bin --backend gpu

# 3. Generate text from the checkpoint
dotnet run --project src/LLM.Cli -- generate --model out/model.bin --tokenizer data/shakes \
    --prompt "Once upon a time" --tokens 200 --temperature 0.8 --topk 40

# 4. Or chat with it interactively (base model: it continues your text, it won't answer questions)
dotnet run --project src/LLM.Cli -- chat --model out/model.bin --tokenizer data/shakes
```

All flags are optional where a default exists; run any command with `--help` for
the full list (model size, learning-rate schedule, sampling knobs, resuming from
a checkpoint with `--init`, etc.). `train`, `generate` and `chat` accept
`--backend cpu|gpu` (default `cpu`); the selected backend is printed at startup.

- `prepare` accepts a local file or an http(s) URL (default: tiny-shakespeare),
  trains or loads a tokenizer (`tokenizer.json`), encodes the corpus, and writes
  a 90/10 split of raw little-endian uint16 token files (`train.bin`, `val.bin`).
- `prepare-fineweb` builds a large-scale corpus from
  [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (`sample-10BT`
  config): it lists the parquet shards via the Hugging Face API, downloads the
  first `--shards N` (default 10, ~2.1 GB each) into `<out>/shards/` (existing
  complete shards are skipped, so interrupted runs resume), extracts the `text`
  column into `<out>/corpus.txt` via
  [Parquet.Net](https://github.com/aloneguid/parquet-dotnet) (CLI-only
  dependency), trains the tokenizer on the first `--toktrainmb` MB only
  (default 200), then stream-encodes the full corpus in 50 MB newline-aligned
  chunks — a multi-GB corpus never has to fit in memory. Encoding is the slow
  phase (tens of minutes to hours depending on merges/corpus size); progress in
  MB is printed per chunk. Corpus, tokenizer, and bin artifacts are published
  transactionally with provenance manifests; stale or incomplete artifacts are
  never silently reused. `--rebuild true` regenerates derived artifacts while
  retaining downloaded shards.
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
  current step, then saves) — an interrupted run is never lost. V2+ training
  checkpoints preserve the global step, Adam moments, LR schedule, and sampler RNG,
  so `--init` continues the same trajectory instead of restarting warmup. Legacy V1
  model-only checkpoints still load; `--resume-step` can provide their known global
  scheduler position during the one-time upgrade. New V3 saves add a SHA-256 trailer,
  bind the checkpoint to its tokenizer and training data, and rotate the previous
  generation to `<checkpoint>.bak`. Loading V1/V2 remains supported; the next save
  upgrades it. Startup hashes the input files before an optimizer update.
- `generate` loads a checkpoint and samples autoregressively with temperature
  and top-k filtering. An incremental UTF-8 decoder preserves characters whose
  bytes span token boundaries.
- `chat` is an interactive REPL over a checkpoint: each line you type is appended
  to a rolling context the model continues. `/reset` clears context, `/quit` exits.

## Architecture

- **BPE tokenizer** (`Tokenizer/BpeTokenizer.cs`) — byte-level, GPT-2 style.
  Ids 0–255 are raw bytes; learned merges produce ids 256+. Encoding is
  rank-greedy; arbitrary bytes remain representable and valid UTF-8 text
  round-trips losslessly.
- **GPT model** (`Model/`) — GPT-2 architecture: learned token + positional
  embeddings, pre-LN transformer blocks (multi-head causal self-attention +
  GELU MLP), final LayerNorm, untied output head. Training is batched: B
  sequences of length T are stacked row-wise into `[B*T, C]` tensors and
  processed in one pass (attention never crosses sequence boundaries); the
  (sequence, head) attention slots are packed into slot-contiguous tensors and
  processed with batched kernels. Inference runs one sequence at a time; there
  is no KV cache.
- **Training** (`Training/`) — AdamW (decoupled weight decay on 2-D params),
  averaged gradient accumulation, linear-warmup + cosine-decay in optimizer-update
  units, global gradient-norm clipping, and a checkpointable random-offset data
  loader over raw uint16 token files, including files beyond 2.147 billion tokens.
- **CPU backend** (`Tensor/CpuBackend.cs`) — all math through an
  `ITensorBackend` interface; the CPU implementation uses `Vector<T>` SIMD in
  the matmul inner loops. Large matmuls parallelize over output rows with
  `Parallel.For` and pooled buffers (no steady-state allocation); small ones
  run sequentially.
- **GPU backend** (`Tensor/Gpu/`) — a second `ITensorBackend` implementation
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
  Measured at GPT-1 scale (dmodel 768, 12 layers, ctx 512, batch 8): see below.
- **Checkpoints** (`Checkpoint/Checkpoint.cs`) — versioned custom binary format.
  `LLMSCRATCH1` is the legacy model-only format. `LLMSCRATCH2` adds cumulative step,
  schedule/configuration, sampler RNG, Adam age, and first/second moment tensors to
  the model weights. `LLMSCRATCH3` adds input identities and a SHA-256 checksum.
  Loading validates the exact payload size and parameter registry.
  Checkpoints are backend-agnostic: train on GPU, generate on CPU or vice versa.

## GPU backend notes

- Requires Windows with a D3D12 device (any recent NVIDIA/AMD/Intel driver;
  WARP software devices also work for correctness testing). Without one, the
  GPU tests skip and `--backend gpu` exits with a clear error.
- Runtime: the ComputeSharp source generators rely on `[UnsafeAccessor]`,
  which hits a JIT bug in .NET 9.0.0–9.0.2 (`MissingFieldException` on
  dispatch). The app and test projects set `RollForward=LatestMajor`, so any
  newer runtime (.NET 9.0.3+ servicing or .NET 10) is used automatically when
  installed. The CPU backend is unaffected on any runtime.
- Reference throughput on an RTX 2080 8 GB, GPT-1 scale
  (`--dmodel 768 --layers 12 --heads 12 --ctx 512 --batch 8`, 4096 tokens/step):
  CPU ≈ 100 tok/s, GPU ≈ 950 tok/s (~9×). Device memory usage is ~5.5 GB.

## Project layout

```
src/LLM.Core/     the library (tokenizer, tensors, backends, model, training, inference, checkpoints)
src/LLM.Cli/      the command-line front end (prepare / train / generate / chat)
tests/LLM.Core.Tests/   hand-rolled test harness (dotnet run)
```
