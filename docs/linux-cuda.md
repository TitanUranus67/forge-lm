# Linux NVIDIA cloud deployment

The CUDA backend runs the full training path on Linux through ILGPU. Checkpoints
are backend-agnostic, so the current Windows/D3D12 checkpoint can be copied to a
Linux NVIDIA instance and resumed without conversion.

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
data/fineweb/tokenizer.json
data/fineweb/train.bin
data/fineweb/val.bin
out/gpt1.bin             checkpoint
```

The FineWeb corpus text and parquet shards are not needed for training once the
three prepared data files exist. Preserve their bytes exactly: V3 checkpoints
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

Run a quick load check:

```sh
./publish/linux-x64/LLM.Cli generate \
  --backend cuda \
  --model out/gpt1.bin \
  --tokenizer data/fineweb \
  --tokens 0
```

Startup should print `backend: cuda` followed by the NVIDIA device name and VRAM.

## 4. Resume the current training run

```sh
./publish/linux-x64/LLM.Cli train \
  --backend cuda \
  --data data/fineweb \
  --logevery 16 \
  --valevery 320 \
  --valbatches 50 \
  --saveevery 320 \
  --init out/gpt1.bin \
  --out out/gpt1.bin
```

The checkpoint restores the model, cumulative optimizer step, Adam moments,
learning-rate schedule, and training sampler. The first Linux startup hashes the
three prepared input files, and the first CUDA backend creation JIT-compiles its
kernels. Both are one-time startup costs for that process.

For an interruptible cloud instance, send `Ctrl+C` while the process is attached.
Training finishes the current optimizer step and saves before exiting. Copy the
new `gpt1.bin` off-instance before terminating ephemeral storage.
