# Linux NVIDIA cloud deployment

The CUDA backend runs the full Forge training path on Linux through ILGPU.
Checkpoints are backend-agnostic and contain the model name, complete optimizer
state, scheduler position, sampler position, and data identities.

## 1. Publish for Linux

From the repository root on Linux:

```sh
dotnet publish src/LLM.Cli -c Release -r linux-x64 --self-contained true \
  -o publish/linux-x64
```

Or from PowerShell on Windows:

```powershell
dotnet publish src/LLM.Cli -c Release -r linux-x64 --self-contained true `
  -o publish/linux-x64
```

The RID-specific build excludes the Windows-only ComputeSharp backend. The
self-contained output does not require .NET to be installed on the server.

## 2. Copy the required files

Copy these items to the server:

```text
publish/linux-x64/       application
data/forge/tokenizer.json
data/forge/train.bin
data/forge/val.bin
```

The FineWeb-Edu corpus text and parquet shards are not needed for training once the
three prepared data files exist. Preserve their bytes exactly: checkpoints
verify the tokenizer and train/validation identities before resuming.

Allow enough free disk for the data plus checkpoint rotation. A save can
temporarily coexist with the current checkpoint and its `.bak` generation.

## 3. Verify the server GPU

```sh
nvidia-smi
chmod +x publish/linux-x64/LLM.Cli
```

The process must see an NVIDIA driver (`libcuda.so`). On a container host, start
the container with NVIDIA GPU access. `CUDA_VISIBLE_DEVICES` can be used to expose
one selected GPU; the backend opens visible device zero.

Run the production-shape benchmark before committing to a physical batch:

```sh
./publish/linux-x64/LLM.Cli benchmark \
  --backend cuda \
  --batch 4 --accum 16 --steps 3
```

Repeat with `--batch 8 --accum 8` and, when VRAM headroom permits,
`--batch 16 --accum 4`. Also compare the default custom matmul with
`--matmul-precision fp32`. Keep 32,768 tokens per optimizer update.

## 4. Start Forge-98M from scratch

```sh
./publish/linux-x64/LLM.Cli train \
  --backend cuda \
  --preset forge-98m \
  --data data/forge \
  --tokens 1024000000 \
  --warmup-tokens 4096000 \
  --lr 6e-4 \
  --minlr 6e-5 \
  --batch 4 \
  --accum 16 \
  --logevery 16 \
  --valevery 320 \
  --valbatches 50 \
  --saveevery 320 \
  --out out/forge-98m.bin
```

Replace batch, accumulation, and matmul precision with the winning benchmark
configuration. The first startup hashes the three prepared input files, and the
first CUDA backend creation JIT-compiles its kernels. Both are one-time costs.

To resume later, run the same command with `--init out/forge-98m.bin`; checkpointed
training settings become the defaults, so architecture and schedule flags may be
omitted.

For an interruptible cloud instance, send `Ctrl+C` while the process is attached.
Training finishes the current optimizer step and saves before exiting. Copy the
new `forge-98m.bin` off-instance and verify its SHA-256 before terminating ephemeral storage.
