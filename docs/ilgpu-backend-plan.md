# Plan: ILGPU (CUDA) backend + safe automatic selection

## Goal
Add a third tensor backend, `CudaBackend`, built on ILGPU (C# → CUDA PTX JIT), so the
project runs on Linux/NVIDIA cloud GPUs (vast.ai et al.). Add an `auto` selection mode
that prefers CUDA on this machine only if it benches ≥ the D3D12 `GpuBackend`, while
always falling back to D3D12 or CPU when CUDA is unavailable.

## Key facts (verified)
- **ILGPU 1.5.3** (NuGet, MIT) targets .NET Standard 2.1 / .NET 5+ → works on net9.0.
  One package only. **NOT** ILGPU.Algorithms — it has known device-compat issues on
  newer cards (RTX 40-series) and we already have our own reductions.
- NuGet restore on this machine needs `--source https://api.nuget.org/v3/index.json`
  (flaky private feed).
- Kernel inventory to port (28 public methods on `GpuBackend`): 19 `ITensorBackend`
  kernels + `Zero`, `AdamWStep`, batched matmuls (NN/NT/TN), `PackHeads`/`UnpackHeads`,
  `CopyBlock`, `SumSquares`, plus the sync pair (`InvalidateDeviceCache`/`EnsureHostCurrent`).
- Architecture patterns to reuse (src/LLM.Core/Tensor/Gpu/GpuBackend.cs): per-Tensor
  device-resident buffers via `Tensor.DeviceResource`, host/device dirty flags
  (`DeviceCurrent`/`HostStale`), size-bucketed arena allocator with free lists +
  weak-ref reclaim (BucketOf — note: geometric series with alignment folded into rungs,
  fixed-point invariant, covered by `BucketOf_IdempotentAlignedAndBounded`).
- ILGPU equivalents for the D3D12 primitives: `MemoryBuffer1D<float>` + `ArrayView`
  **subviews** (arena carving works), `Atomic.Add` (SumRows, EmbeddingBackward CAS
  scatter), group/shared memory + `Group.Barrier` (tiled 16×16 matmuls), async stream
  launches (sequential ordering on one stream; downloads synchronize).
- Tests: `tests/LLM.Core.Tests/GpuBackendTests.cs` is the template — per-kernel
  CPU-vs-GPU comparisons (fwd ≤1e-4, bwd ≤1e-2), residency/sync tests, aliasing tests,
  full-model batched gradient check, overfit smoke, skip-cleanly-if-no-device.
- Bench reference numbers (RTX 2080, 110M config 768/12/12/512 batch 4): cpu ~96 tok/s,
  d3d12 ~790–950 tok/s.

## Milestone 1 — Scaffolding + allocator + trivial kernels
New: `src/LLM.Core/Tensor/Cuda/{CudaBackend.cs,CudaKernels.cs}`.
1. Add ILGPU 1.5.3 to LLM.Core only (`dotnet add src/LLM.Core package ILGPU --version 1.5.3 --source https://api.nuget.org/v3/index.json`).
2. `CudaBackend : ITensorBackend, IDisposable`: Context + CudaAccelerator(0), one
   accelerator stream, device name/memory properties.
3. Port the allocator: same bucketed arena design over `MemoryBuffer1D<float>`
   (carve via subviews), same `BucketOf` (share or duplicate verbatim — fixed-point
   invariant test must pass either way), free lists, weak-ref reclaim, gen-0/full GC
   backstops, `LLM_GPU_STATS`-style profiling hooks.
4. Same sync model: `InvalidateDeviceCache`/`EnsureHostCurrent` (identical semantics;
   kernels leave device authoritative, downloads on demand).
5. Trivial kernels first (Fill/Zero, Copy, CopyBlock, AddInPlace, Scale, Transpose) +
   `tests/LLM.Core.Tests/CudaBackendTests.cs` with skip-if-no-CUDA and the first
   kernel-vs-CPU comparisons. All existing 87 tests keep passing.

## Milestone 2 — Full kernel port + validation
1. Elementwise/reduction kernels: AddBias, SumRows (atomics), GeluForward/Backward,
   EmbeddingForward, EmbeddingBackward (CAS scatter), CausalMask, SoftmaxForward/
   Backward (one thread/row), LayerNorm fwd/bwd, SumSquares, CrossEntropy fwd/bwd —
   preserving the **in-place aliasing contract** (CE probs/dLogits may alias logits;
   target logit captured before writes; elementwise backward).
2. Matmul NN/NT/TN: tiled 16×16 with group-shared memory (port the D3D12 shader
   logic directly), plus BatchedMatMulNN/NT/TN and PackHeads/UnpackHeads for attention.
3. AdamWStep + Zero on device.
4. Mirror the GPU test suite into CudaBackendTests: every kernel vs CpuBackend
   (same tolerances), residency tests (incl. the stale-without-invalidate behavior),
   CE aliasing, BucketOf property test, full-model batched gradient check (tiny config),
   200-step overfit smoke. Suite: 87 + ~16 = 103+ tests, ALL PASSED, 0 warnings.

## Milestone 3 — CLI + bench gate
1. `--backend auto|cpu|gpu|cuda` on train/generate/chat (default stays `cpu` until M4);
   startup line prints device + VRAM for cuda too. README updated.
2. Bench on the RTX 2080 at 110M config (768/12/12/512, batch 4, 100 steps after JIT
   warmup, steady-state
   last-line tok/s): cpu vs gpu vs cuda. **Gate: cuda ≥ gpu × 0.95** to flip the default
   auto preference order in M4; also confirm device committed memory plateaus (bucketed
   allocator doing its job, no creep over 100 steps).
3. Also smoke `generate --backend cuda` (sampler path) and checkpoint save/load round-trip
   with cuda tensors resident.

## Milestone 4 — Safe automatic selection + commit
1. Default `--backend` becomes `auto` for train/generate/chat. Selection is capability-
   checked and never fails merely because an optional accelerator is absent:
   - if the M3 performance gate passes: CUDA → D3D12 → CPU;
   - if it fails on Windows: D3D12 → CUDA → CPU;
   - on Linux: CUDA → CPU.
   Explicit `--backend cuda|gpu` remains fail-loud instead of silently changing devices.
2. Add selection tests with injectable capability probes so CUDA-absent, D3D12-absent,
   and CPU-only machines all have deterministic coverage. A fresh command with no
   `--backend` must succeed on a CPU-only host.
3. Commit: "Add ILGPU CUDA backend with automatic fallback". Keep GpuBackend in the
   tree and tested — it is a supported backend, not dead code.

## Explicit non-goals / follow-ups
- No fp16/TF32/tensor cores (fp32 throughout, like the rest of the project).
- No ILGPU.Algorithms, no multi-GPU, no OpenCL path (CUDA only; OpenCL is a maybe-later).
- vast.ai deployment is a FOLLOW-UP after this lands: linux-x64 publish, driver check,
  `prepare-fineweb` on-instance or upload bins, interruptible-instance checkpoint loop.
  Not part of this plan, but this plan is the unblock for it.
- Guard: before starting, verify no other agent/process is editing the tree (a wedged
  background agent caused file-lock chaos earlier in this project).

## Verification per milestone
- M1: 87 + new trivial-kernel tests green.
- M2: full suite (103+) green incl. cuda gradient check + overfit.
- M3: bench numbers reported (cpu/gpu/cuda), plateau confirmed.
- M4: suite green, commit made, fresh commands with no `--backend` select the fastest
  available validated backend and still run on a CPU-only host.
