# LLM_Scratch

A mini-GPT built from scratch in pure C# (.NET 9). **Zero NuGet dependencies** —
everything (BPE tokenizer, transformer, AdamW, SIMD tensor backend, checkpoint
format) is implemented on top of the Base Class Library only.

## Build & test

```sh
dotnet build
dotnet run --project tests/LLM.Core.Tests   # 54 tests
```

## Quickstart

The CLI has three subcommands (tokenizer training is folded into `prepare`):

```sh
# 1. Download tiny-shakespeare, train a BPE tokenizer, encode to train/val bins
dotnet run --project src/LLM.Cli -- prepare --out data/shakes --merges 2000

# 2. Train a small GPT (checkpoint written to out/model.bin)
dotnet run --project src/LLM.Cli -- train --data data/shakes --steps 5000 --out out/model.bin

# 3. Generate text from the checkpoint
dotnet run --project src/LLM.Cli -- generate --model out/model.bin --tokenizer data/shakes \
    --prompt "Once upon a time" --tokens 200 --temperature 0.8 --topk 40

# 4. Or chat with it interactively (base model: it continues your text, it won't answer questions)
dotnet run --project src/LLM.Cli -- chat --model out/model.bin --tokenizer data/shakes
```

All flags are optional where a default exists; run any command with `--help` for
the full list (model size, learning-rate schedule, sampling knobs, resuming from
a checkpoint with `--init`, etc.).

- `prepare` accepts a local file or an http(s) URL (default: tiny-shakespeare),
  trains or loads a tokenizer (`tokenizer.json`), encodes the corpus, and writes
  a 90/10 split of raw little-endian uint16 token files (`train.bin`, `val.bin`).
- `train` runs a batched training loop: each optimizer step processes `--batch N`
  sequences (default 8) at once as `[B*T, C]` tensors, with warmup + cosine LR
  decay, gradient clipping, and periodic validation loss. Checkpoints are written
  at the end, every `--saveevery N` steps, and on Ctrl+C (the run finishes its
  current step, then saves) — an interrupted run is never lost. `--init` resumes
  from a checkpoint.
- `generate` loads a checkpoint and samples autoregressively with temperature
  and top-k filtering.
- `chat` is an interactive REPL over a checkpoint: each line you type is appended
  to a rolling context the model continues. `/reset` clears context, `/quit` exits.

## Architecture

- **BPE tokenizer** (`Tokenizer/BpeTokenizer.cs`) — byte-level, GPT-2 style.
  Ids 0–255 are raw bytes; learned merges produce ids 256+. Encoding is
  rank-greedy; any byte sequence round-trips losslessly.
- **GPT model** (`Model/`) — GPT-2 architecture: learned token + positional
  embeddings, pre-LN transformer blocks (multi-head causal self-attention +
  GELU MLP), final LayerNorm, untied output head. Training is batched: B
  sequences of length T are stacked row-wise into `[B*T, C]` tensors and
  processed in one pass (attention never crosses sequence boundaries).
  Inference runs one sequence at a time; there is no KV cache.
- **Training** (`Training/`) — AdamW (decoupled weight decay on 2-D params),
  linear-warmup + cosine-decay LR schedule, global gradient-norm clipping, and
  a random-offset data loader over raw uint16 token files.
- **CPU backend** (`Tensor/CpuBackend.cs`) — all math through an
  `ITensorBackend` interface; the CPU implementation uses `Vector<T>` SIMD in
  the matmul inner loops. Large matmuls parallelize over output rows with
  `Parallel.For` and pooled buffers (no steady-state allocation); small ones
  (attention scores) run sequentially while batched attention parallelizes
  across (sequence, head) slots instead.
- **Checkpoints** (`Checkpoint/Checkpoint.cs`) — custom binary format:
  `LLMSCRATCH1` magic, a JSON header (model config + parameter names), then raw
  little-endian float32 weights. Loading validates everything and fails loud.

## Project layout

```
src/LLM.Core/     the library (tokenizer, tensors, model, training, inference, checkpoints)
src/LLM.Cli/      the command-line front end (prepare / train / generate / chat)
tests/LLM.Core.Tests/   hand-rolled test harness (dotnet run)
```

## Road to GPU

Every tensor operation goes through `ITensorBackend`
(`src/LLM.Core/Tensor/ITensorBackend.cs`): matmuls, layernorm, softmax, GELU,
embeddings, cross-entropy, attention helpers. A GPU backend is "just" another
implementation of that interface — `GptModel`, `Trainer`, and `Sampler` are
backend-agnostic, so swapping `new CpuBackend()` for a CUDA/Metal/DirectML
implementation at the call sites is the whole integration.
