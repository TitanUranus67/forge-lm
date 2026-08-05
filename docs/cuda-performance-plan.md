# CUDA training performance plan

**Status: the first cloud run is complete and its instance is gone. Device-side
loss accumulation and the one-readback CUDA global gradient norm are in Forge.
cuBLAS FP32 and TF32 remain selectable. The first cuBLAS implementation was not
faster on the tested RTX 5070 Ti because its attention path submitted one GEMM per
batch/head slot. True strided-batched cuBLAS attention was promoted on the Forge-220M
RTX 5090 run after numerical and same-host performance gates.**

## Goal

Increase end-to-end training throughput and lower cost per trained token without
changing the selected Forge architecture, fixed 1.024B-token budget, optimizer
schedule, checkpoint semantics, or validation quality. Each optimization is tested,
benchmarked, and committed independently before it is used for a paid cloud run.

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

The remaining queued improvements are recorded below: cloud-GPU batch sizing,
remaining launch fusion, data prefetch, validation/checkpoint timing, and BF16 only
as a separately designed project.

### RTX 5090 strided-batched attention promotion - 2026-08-05

Live profiling of Forge-220M showed one host thread saturated while GPU utilization
oscillated between 7% and 82%. The cuBLAS path's nominally batched operation was a
host loop over all `batch * heads` slots. At batch 4, 16 heads, 16 layers, and 8
accumulation passes, the seven forward/backward attention matmuls submitted 57,344
individual GEMMs per optimizer update.

ILGPU 1.5 does not expose `cublasSgemmStridedBatched`, so Forge now has a narrow
native binding for that one operation. It reuses ILGPU's cuBLAS handle, stream,
device-resident allocations, pointer mode, and math mode. Single-matrix projections
continue through ILGPU's ordinary GEMM wrapper.

The complete 121-test suite passed on an RTX 2080, including NN/NT/TN batched
matmuls with and without accumulation, full-model gradients, overfit, allocation,
checkpoint-resume, and CPU comparisons. The Linux candidate then produced matching
loss on the production RTX 5090:

| Same-host Forge-220M benchmark | Old slot loop | Strided batch | Gain |
| --- | ---: | ---: | ---: |
| 5 measured updates | 4,717 tok/s | 5,332 tok/s | 13.0% |
| 10 measured updates, reverse order | 5,080 tok/s | 5,409 tok/s | 6.5% |
| Combined measured tokens/time | 4,953 tok/s | 5,383 tok/s | 8.7% |

The production trainer was checkpointed at global/Adam step 5,439, the old binary
was retained on the instance as a rollback, and training resumed from the exact
checkpoint with the promoted binary. A stale completion watchdog then destroyed
that instance after mistaking the clean benchmark stop for the end of training.
The verified checkpoint was recovered locally before destruction and resumed at
step 5,439; the first update and fixed validation produced loss 4.3708 / val 4.3263
and atomically saved a new step-5,440 checkpoint.

The first replacement 5090 host was rejected after a clean 16-update interval ran
at only 2,064 tok/s. The same published binary and production shape benchmarked at
5,639 tok/s on a Ryzen 9 9950X3D / RTX 5090 host, 2.73x faster and slightly cheaper.
This host-level benchmark is now a required recovery gate: matching the GPU name is
not sufficient when the training loop is sensitive to host submission latency.
After the signal-handling gate advanced and atomically saved step 5,442, production
resumed on that host and reached step 5,456 at 5,385 tok/s with loss 4.3280.

Remote shutdown also now registers SIGTERM through .NET's POSIX signal API. The
background launcher inherits SIGINT as ignored under `nohup`, so SIGTERM is the
reliable automation path for finishing the current optimizer update and publishing
an atomic checkpoint.

### RTX 5090 attention housekeeping fusion - 2026-08-05

Increasing the physical batch was tested first on the same Ryzen 9 9950X3D / RTX
5090 host. Batch 8 with accumulation 4 preserved 32,768 tokens/update but used about
26.5 GB instead of 15.5 GB and reached 5,088 tok/s, versus 5,384 tok/s for batch 4
with accumulation 8 in the reverse control. The larger physical batch was therefore
rejected; it was 5.5% slower despite using substantially more VRAM.

The next candidate fused attention's Q/K/V head packing, Q/K/V gradient unpacking,
scale-plus-causal-mask-plus-softmax, and scaled softmax backward operations. At the
Forge-220M production geometry this removes approximately 1,408 small CUDA launches
per optimizer update without changing model, optimizer, sampler, or checkpoint
semantics. The complete 121-test suite passed, including CPU-composed comparisons
for every fused helper, CUDA model/gradient agreement, overfit, allocation, and
checkpoint tests.

The first A/B attempt was discarded after discovering that the remote watchdog had
restarted production during the measurements. After stopping both the outer
watchdog and trainer and verifying an idle GPU, two alternating ten-update pairs
produced matching deterministic loss:

| Exclusive same-host pair | Existing binary | Fused attention | Gain |
| --- | ---: | ---: | ---: |
| Baseline then candidate | 5,373 tok/s | 5,430 tok/s | 1.1% |
| Candidate then baseline | 5,361 tok/s | 5,451 tok/s | 1.7% |
| Combined measured tokens/time | 5,367 tok/s | 5,441 tok/s | 1.4% |

The gain is modest but repeated in both orders. The old binary remains on the host
as a rollback, and production resumed from the exact global/Adam step 5,840
checkpoint with the fused binary. Its first two normal 16-update production logs
reached 5,563 and 5,690 tok/s with train losses 4.2524 and 4.3407.

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

## Historical first-run baseline

The completed first production run on an RTX 5070 Ti established this baseline:

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

**Implementation status:** a built-in `benchmark` command now constructs the fresh
97.9M-parameter shape with synthetic tokens, performs an unmeasured JIT/allocation
warmup update, and reports measured wall time, throughput, and loss without touching
training data or checkpoints. It is sufficient for the physical-batch preflight.
Hardware telemetry, phase-separated timings, JSON output, and the 100-update
repeatability gate remain follow-up instrumentation rather than launch blockers.

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

**Implementation status:** CUDA now atomically accumulates cross-entropy NLL into a
reusable device scalar across all microbatches and performs one final readback per
training update or validation evaluation. The valid-target count remains exact on
the host. CPU and D3D12 expose identical begin/end semantics. Direct CPU and real-CUDA
tests verify the combined mean; the CUDA test also proves two losses cause one
readback. Training, validation, and checkpoint-resume tests pass.

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
train/validation losses of 8.9707 / 8.9499. All 101 tests pass.

The fixed-step-4,401 RTX 5070 Ti promotion run then completed the same 20-update
sample used by the previous custom-kernel control:

| Gradient norm path | Updates | Training time | Measured throughput | Loss at step 4,417 |
| --- | ---: | ---: | ---: | ---: |
| Per-tensor scalar readbacks | 20 | 5:15 | 2,081 tok/s | 4.8393 |
| One global scalar readback | 20 | 5:03 | 2,163 tok/s | 4.8393 |

This is a 3.9% improvement over the matched control and 1.45% above the established
2,132 tok/s production average. An instrumented run measured the new clipping phase
at about 2 ms/update after warmup. The optimization therefore passes its promotion
gate. At 2,163 tok/s, the 879,788,032 tokens remaining after step 4,401 are estimated
at 113.0 hours (4.71 days) and $14.23 at $0.1259259259/hour, excluding normal
validation/checkpoint overhead.

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

**Local preflight result (RTX 2080, 8 GB, 2026-08-03):** after the competing game was
closed, the fresh tied model with cuBLAS FP32 measured 3,127 tok/s at batch 4 / accum
16 and 1,276 tok/s at batch 8 / accum 8. Batch 16 / accum 4 entered severe memory
pressure and its run was terminated. Batch 4 remains the safe 8 GB choice. These
numbers do not select the Vast configuration: repeat the same sweep on the exact
cloud GPU before starting its full run, where 16-24 GB may favor a larger batch.

The batch of 4 was originally selected for an 8 GB RTX 2080, while the rented 5070 Ti
used only about 4.8 GB. Benchmark configurations that preserve exactly 32,768
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

1. Publish the tested Forge commit to the selected Vast instance.
2. Run the Task 4 sweep at batch/accum 4/16, 8/8, and 16/4 where memory permits;
   compare custom and cuBLAS FP32 on that exact GPU.
3. Prepare three FineWeb-Edu shards into `data/forge`, then verify the tokenizer,
   EOS-delimited splits, manifests, and reported token counts.
4. Run a short fixed-seed pilot with normal validation and checkpoint cadence.
   Verify loss movement, generation, checkpoint reload, and exact resume.
5. Start the fixed 1.024B-token Forge-98M run only after the pilot passes. Copy and
   hash-verify checkpoints off-instance throughout the run and before destruction.
6. Re-profile afterward and select only justified work from Tasks 7-9. Treat BF16
   as a new project if the measured upside warrants its complexity.

Throughput and utilization targets must be established against the selected Vast
GPU's own repeatable baseline. Every retained change remains attributable to one
reviewed commit; utilization alone is not a reason to accept a slower path.
