# CUDA training performance plan

**Status: cuBLAS/TF32 was rejected for the production GPU. A one-readback CUDA
global gradient norm is implemented and is awaiting its matched cloud benchmark.**

## Goal

Increase end-to-end training throughput and lower cost per trained token without
changing the model, token budget, optimizer schedule, checkpoint semantics, or
validation quality. Optimizations are developed and proven against separate local
checkpoints before the current cloud executable is eligible to be replaced.

The current remote executable must not be replaced mid-run. Performance work will
remain isolated from that process and each optimization will be tested, benchmarked,
and committed independently.

## Implementation record - 2026-08-02

The first implementation slice prioritizes the largest likely compute win while
retaining the original CUDA kernel as the unchanged default:

1. Add selectable cuBLAS SGEMM for NN, NT, TN, and contiguous batched attention
   matmuls, bound to ILGPU's default CUDA stream.
2. Add `--matmul-precision custom|fp32|tf32`. `custom` remains the default during
   validation, `fp32` selects strict cuBLAS FP32, and `tf32` is explicit and requires
   CUDA compute capability 8.0 or newer.
3. Prove FP32 locally, then benchmark FP32 and TF32 on the modern cloud GPU before
   considering any change to the live training process.

The initial matched RTX 2080 production-shape run used the 110M-parameter model,
batch 4, accumulation 16, and 32,768 tokens/update for six optimizer updates:

| Matmul path | Final reported throughput | Training wall time | Final train/val loss |
| --- | ---: | ---: | ---: |
| Custom ILGPU FP32 | 1,025 tok/s | 3:11 | 8.9707 / 8.9499 |
| cuBLAS FP32 | 3,154 tok/s | 1:02 | 8.9707 / 8.9499 |

This short result is a 3.08x local throughput improvement with identical displayed
losses. The 98-test suite, real-CUDA model/gradient checks, allocator checks, Linux
self-contained publish, and custom-kernel-checkpoint-to-cuBLAS resume test pass. It is
still only a local promotion signal. The cloud benchmark below completed the next
gate and rejected both optimized modes for the production run. The RTX 2080 is SM
7.5 and cannot execute TF32, so its local TF32 test correctly verifies fail-loud
capability gating.

The other queued improvements remain recorded below: device-side loss accumulation,
one-readback gradient norm, 16 GB batch sizing, remaining launch fusion, data
prefetch, validation/checkpoint timing, and BF16 only after the FP32/TF32 path is
settled.

### RTX 5070 Ti promotion benchmark - 2026-08-02

The live trainer was stopped cleanly at global step 4,401, and its complete checkpoint
was frozen and SHA-256 verified locally before benchmarking. Each candidate resumed
from that exact state with validation and periodic saves disabled during timing. The
host used NVIDIA cuBLAS 12.9.2.10; an initial result from Ubuntu's obsolete cuBLAS
11.7 package was discarded.

| Matmul path | Updates | Training time | Measured throughput | Loss at step 4,417 |
| --- | ---: | ---: | ---: | ---: |
| Custom ILGPU FP32 | 20 | 5:15 | 2,081 tok/s | 4.8393 |
| cuBLAS FP32 | 17 | 4:26 | 2,094 tok/s | 4.8392 |
| cuBLAS TF32 | 18 | 5:12 | 1,890 tok/s | 4.8392 |

The original long-running custom process averaged 2,132 tok/s including its normal
validation and checkpoint cadence. On this GPU, cuBLAS FP32 is effectively tied with
the custom kernel in the short control and slightly slower than the established
long-run rate. TF32 is materially slower. Neither cuBLAS mode qualifies for promotion
to the production run. The local RTX 2080 improvement therefore does not generalize
to the Blackwell cloud GPU, and the next work should target synchronization/readback
costs before revisiting library matmuls or cuBLASLt.

## Current baseline

The first production run on an RTX 5070 Ti establishes the initial baseline:

- Model: 768 width, 12 layers, 12 heads, context 512, 110,434,688 parameters.
- Physical batch: 4 sequences (2,048 tokens).
- Gradient accumulation: 16 physical batches (32,768 tokens/update).
- Steady training throughput: approximately 2,140-2,220 tokens/second.
- Device memory: approximately 4.8 GB of 16 GB.
- Observed SM utilization: bursty, 29-100% over a 30-second sample, approximately
  65% mean; the Vast dashboard may show lower smoothed snapshots.
- Host process: approximately one fully occupied CPU core.
- CUDA path: FP32 ILGPU kernels, tiled 16x16 project-owned matmuls, no cuBLAS,
  TF32, FP16, BF16, or tensor-core path.

The two clearest synchronization costs are:

1. Cross-entropy copies one loss value per row back to the host for every physical
   batch. With accumulation 16, this forces at least 16 device/host synchronization
   points per optimizer update, plus 50 readbacks per validation evaluation.
2. Gradient clipping calls `SumSquares` once per parameter tensor and each call
   copies a scalar to the host. This serializes what should be one device-side
   global-norm operation.

Periodic deterministic validation and 1.3 GB checkpoint writes also affect
end-to-end throughput, but they are reliability and quality features and should
only be changed after their cost is measured separately.

## Rules for every optimization

1. Use the same GPU host, model shape, dataset, seed, and effective 32,768
   tokens/update when comparing before and after.
2. Warm up ILGPU JIT compilation before measuring. Record at least 100 optimizer
   updates and report median/steady tokens per second, optimizer-update time,
   GPU utilization, CPU utilization, peak VRAM, and wall time spent validating
   and saving.
3. Keep a short end-to-end benchmark that includes normal validation and checkpoint
   cadence in addition to the kernel-only/steady training measurement.
4. Run the complete test suite and real-CUDA tests. Compare fixed-seed losses,
   gradients, parameter updates, validation means, checkpoint save/load, and resumed
   training against the pre-change backend within explicitly documented tolerances.
5. Confirm device allocations plateau and that repeated training, validation, and
   save cycles do not leak host or GPU memory.
6. Commit one completed task at a time. Do not keep a risky optimization merely
   because it is theoretically faster: retain it only if correctness holds and the
   measured end-to-end result justifies its complexity.

## Task 1 - Add a repeatable performance harness

Create a benchmark mode or script that runs the production model shape without
changing normal training behavior. It should:

- Separate JIT/startup, data sampling, forward/backward, gradient clipping, AdamW,
  validation, and checkpoint timing.
- Sample GPU SM utilization, clocks, power, VRAM, and throttling state at one-second
  intervals.
- Emit machine-readable output so results from multiple commits can be compared.
- Include both a short correctness run and a 100-update steady-state run.
- Record the GPU model, driver, ILGPU version, command line, commit SHA, and effective
  tokens/update with every result.

**Acceptance:** Running the harness twice on the same idle GPU should produce
steady-state throughput within 3%, and enabling measurement must not alter the
fixed-seed loss trajectory.

**Commit:** `Add repeatable CUDA training benchmarks`

## Task 2 - Remove per-microbatch loss readbacks

Keep cross-entropy loss reduction on the device:

- Have CUDA cross-entropy produce a device-side loss sum and valid-token count.
- Reset one reusable loss accumulator at the start of an optimizer update.
- Accumulate all 16 physical-batch losses on the device.
- Copy only the final sum/count to the host once per optimizer update.
- During validation, accumulate all 50 batches and perform one final readback.
- Preserve the exact definition of mean loss and ignored-target handling.
- Avoid allocating a new host `float[]` for every forward pass.

The CPU and D3D12 backends may keep immediate values internally, but the public
training abstraction should expose the same accumulation semantics across backends.

**Acceptance:** Training and validation losses match the old implementation within
the existing CUDA tolerance. Host readbacks fall from at least 16 to one per training
update and from 50 to one per validation evaluation. Keep the change only if the
benchmark shows a measurable improvement without additional memory growth.

**Commit:** `Accumulate CUDA loss before host readback`

## Task 3 - Compute global gradient norm with one readback

**Implementation status:** CUDA now performs block partial sum-of-squares reductions
for every gradient tensor, finishes their global reduction on the device, and reads
back one scalar per optimizer update. CPU and D3D12 retain the interface's existing
fallback behavior. The implementation reuses its partial buffer, avoids per-update
managed allocations, rejects non-finite norms, and has direct clipped, unclipped,
empty-input, numerical-accuracy, and readback-count coverage.

The matched six-update RTX 2080 production-shape benchmark increased final reported
throughput from 1,025 to 1,045 tok/s (about 2.0%), with the same displayed final
train/validation losses of 8.9707 / 8.9499. All 101 tests pass. This is a valid but
small local gain; the fixed-checkpoint RTX 5070 Ti benchmark remains the promotion
gate.

Replace the per-parameter synchronization loop with a device-side global reduction:

- Clear one reusable device scalar before clipping.
- Launch per-tensor/block partial reductions that accumulate into device memory.
- Finish the global sum-of-squares on the device.
- Copy one scalar to the CPU, decide whether clipping is required, then launch scale
  kernels asynchronously if necessary.
- Handle empty tensors, zero norm, non-finite values, and very large gradients
  explicitly.

Prefer block partials followed by a final reduction over high-contention atomics if
profiling shows the latter is slower or numerically unstable.

**Acceptance:** Clipped and unclipped test cases agree with CPU results within the
existing tolerance, non-finite gradients cannot silently pass, and clipping performs
one host readback per optimizer update.

**Commit:** `Reduce CUDA global gradient norm on device`

## Task 4 - Tune physical batch size for 16 GB GPUs

The current batch of 4 was selected for an 8 GB RTX 2080, while the rented 5070 Ti
uses only about 4.8 GB. Benchmark configurations that preserve exactly 32,768
tokens per optimizer update:

| Physical batch | Accumulation | Tokens/update |
| ---: | ---: | ---: |
| 4 | 16 | 32,768 |
| 8 | 8 | 32,768 |
| 16 | 4 | 32,768 |

Test batch 8 first. Attempt batch 16 only if measured peak allocation leaves safe
headroom; an out-of-memory-prone configuration is not acceptable. Keep at least
1.5-2 GB free for JIT, validation, checkpoint synchronization, allocator variation,
and driver overhead.

Changing physical batch grouping can cause small floating-point differences even
with the same effective batch, so compare short-run convergence rather than demand
bitwise identity. Do not change the global step, warmup, decay, or token accounting.

**Acceptance:** Select the fastest stable configuration that stays within the memory
headroom rule and shows no material short-run loss regression. Document the selected
command; do not silently make a hardware-specific value the universal default.

**Commit:** `Document and validate 16 GB CUDA batch sizing`

## Task 5 - Replace project matmuls with an optimized CUDA library path

**Implementation status:** cuBLAS FP32 is implemented behind the explicit `fp32`
selection and passed the numerical gates. Its RTX 5070 Ti throughput was effectively
tied with the short custom control and below the established long-run custom rate,
so it remains non-default and was rejected for the production run.

The project-owned tiled matmul was valuable for establishing a correct portable
backend, but matrix multiplication should use NVIDIA's optimized implementation:

- Add a cuBLAS-backed FP32 implementation for NN, NT, TN, and strided-batched
  attention matmuls.
- Bind the cuBLAS handle to the same CUDA stream used by ILGPU so kernel ordering and
  device-residency rules remain correct.
- Reuse existing device allocations; do not stage matrices through host memory.
- Carefully map the project's row-major layouts and transpose flags to cuBLAS.
- Keep the custom ILGPU matmul as a tested fallback when cuBLAS is unavailable.
- Add an explicit diagnostic identifying which matmul implementation is active.

Start with strict FP32 behavior. Tensor-core precision modes belong in the next task
and must not be enabled implicitly as part of this change.

**Acceptance:** Every matmul variant, attention path, full-model gradient test, and
short overfit test passes. Make cuBLAS the default only if it is stable on both the
RTX 2080 and a modern cloud GPU and materially improves end-to-end throughput.

**Commit:** `Add cuBLAS FP32 matrix multiplication`

## Task 6 - Add an explicit TF32/tensor-core mode

**Implementation status:** explicit mode selection and SM 8.0+ capability gating
are implemented. A fixed-state RTX 5070 Ti run with cuBLAS 12.9 followed the FP32
loss trajectory but was about 9% slower than the matched custom control, so TF32 is
rejected for production in its current SGEMM math-mode form.

After the FP32 cuBLAS path is stable, evaluate TF32 as a separate, opt-in precision
mode:

- Add an explicit CLI/config choice such as `--matmul-precision fp32|tf32`.
- Keep weights, optimizer moments, reductions, and checkpoints in FP32.
- Limit reduced precision to eligible matmuls.
- Record numerical tolerances and the exact NVIDIA/cuBLAS math mode used.
- Run a fixed-seed short training comparison long enough to reveal loss drift, not
  merely a single forward-pass comparison.

**Acceptance:** A meaningful throughput/cost improvement, no NaNs or instability,
and no material validation-loss regression. FP32 remains available and TF32 should
not become the default until the training-quality evidence supports it.

**Commit:** `Add opt-in TF32 CUDA matmuls`

## Task 7 - Reduce remaining small-kernel launch overhead

Re-profile after Tasks 2-6. Only optimize the largest remaining measured costs.
Candidates include:

- Multi-tensor zeroing and gradient scaling.
- Multi-tensor AdamW updates.
- Combined parameter housekeeping through compact device-side descriptors.
- Kernel fusion where adjacent elementwise operations have compatible lifetimes.
- Reuse of target/index buffers without redundant host copies.

Avoid a broad rewrite. Implement each candidate separately and discard it if its
effect is lost in end-to-end noise.

**Acceptance:** Each retained subtask passes numerical and checkpoint-resume tests
and improves steady-state optimizer-update time by at least the benchmark noise
floor.

**Commits:** One focused commit per retained fusion or multi-tensor operation.

## Task 8 - Remove host data-pipeline bubbles if they remain

Only pursue this if profiling still shows the GPU waiting on batch preparation after
device synchronization has been fixed:

- Prepare the next physical batch while the current one executes.
- Use double-buffered reusable arrays rather than per-step allocation.
- Evaluate pinned/page-locked transfer buffers and asynchronous copies.
- Preserve the checkpointed sampler position and deterministic sample order.
- Bound the prefetch queue so pause, Ctrl+C, and save-and-quit remain prompt and
  checkpoint state always describes the next unconsumed batch.

**Acceptance:** Identical sampled token offsets for a fixed seed, correct resume
behavior at every interruption point, and a measurable reduction in GPU idle gaps.

**Commit:** `Prefetch CUDA training batches safely`

## Task 9 - Measure validation and checkpoint overhead separately

Do not weaken validation or checkpoint safety merely to improve the headline rate.
Instead:

- Report training, validation, checkpoint serialization, disk flush, and off-instance
  copy time separately.
- Benefit from Task 2's single validation-loss readback before changing cadence.
- Evaluate whether checkpoint host readback and disk I/O can be pipelined safely from
  an immutable snapshot.
- Retain atomic save/rotation, SHA-256 verification, optimizer state, scheduler state,
  RNG state, and data identities.
- If asynchronous snapshots require a second 1.3 GB state copy, include their RAM,
  VRAM, and consistency cost in the decision.

**Acceptance:** No reduction in recovery guarantees. Any cadence recommendation must
state the maximum work at risk and compare saved cost with that risk.

**Commit:** `Report validation and checkpoint overhead` before any optional follow-up.

## Task 10 - Consider BF16 mixed precision last

BF16/FP16 training is a separate architectural project, not a quick utilization fix.
Only begin after the FP32/TF32 path is measured and stable. A complete design must
cover:

- FP32 master weights and Adam moments.
- BF16/FP16 activation and matmul storage.
- Loss scaling and non-finite detection where required.
- LayerNorm, softmax, reductions, and embedding accumulation precision.
- Backend-neutral FP32 checkpoints and resume compatibility.
- Convergence testing beyond a tiny overfit smoke.

This task should have its own design review and milestone plan before implementation.

## Recommended execution order

1. Finish, download, verify, and close the current cloud run.
2. Task 1: benchmark harness.
3. Task 2: loss accumulation/readback.
4. Task 3: global gradient-norm reduction.
5. Task 4: physical batch sweep.
6. Task 5: cuBLAS FP32.
7. Task 6: opt-in TF32.
8. Re-profile and select only justified work from Tasks 7-9.
9. Treat BF16 as a new project if FP32/TF32 results leave enough value on the table.

The first combined target is sustained average utilization above 75% and at least a
25% end-to-end throughput improvement on the same RTX 5070 Ti class without a loss,
resume, or checkpoint regression. This is a target rather than an excuse to combine
changes: every result remains attributable to one reviewed commit.
