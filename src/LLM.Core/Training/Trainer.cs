namespace LLM.Core.Training
{
    using System.Diagnostics;
    using LLM.Core.Model;
    using LLM.Core.Tensor;

    /// <summary>Hyperparameters for one <see cref="Trainer.Train"/> run.</summary>
    public sealed record TrainOptions
    {
        /// <summary>Total optimizer steps.</summary>
        public int Steps { get; init; } = 1000;
        /// <summary>Peak learning rate (reached at the end of warmup).</summary>
        public float MaxLr { get; init; } = 3e-3f;
        /// <summary>Cosine-decay floor learning rate.</summary>
        public float MinLr { get; init; } = 3e-4f;
        /// <summary>Linear warmup steps.</summary>
        public int WarmupSteps { get; init; } = 100;
        /// <summary>Decoupled AdamW weight decay (2-D tensors only).</summary>
        public float WeightDecay { get; init; } = 0.1f;
        /// <summary>Global gradient-norm clip; &lt;= 0 disables clipping.</summary>
        public float GradClip { get; init; } = 1.0f;
        /// <summary>Sequence length per step; 0 means the model's ContextLength.</summary>
        public int ContextLength { get; init; }
        /// <summary>Sequences per optimizer step (true batching: one [B*T, C] pass per step).</summary>
        public int BatchSize { get; init; } = 8;
        /// <summary>Physical batches accumulated and averaged before one optimizer update.</summary>
        public int AccumulationSteps { get; init; } = 1;
        /// <summary>RNG seed for data sampling (and therefore reproducibility).</summary>
        public int Seed { get; init; } = 1337;
        /// <summary>Emit a log record every this many steps.</summary>
        public int LogEvery { get; init; } = 50;
        /// <summary>Evaluate validation loss every this many steps; 0 disables it.</summary>
        public int ValEvery { get; init; } = 0;
        /// <summary>Number of deterministic physical-size val batches averaged per evaluation.</summary>
        public int ValBatches { get; init; } = 50;
        /// <summary>Seed defining the fixed validation sample set (independent of training RNG).</summary>
        public int ValSeed { get; init; } = 424242;
        /// <summary>Invoke the save callback every this many steps; 0 disables it.</summary>
        public int SaveEvery { get; init; } = 0;
    }

    /// <summary>One progress record emitted to the Train callback.</summary>
    public sealed record TrainLog(int Step, float Lr, float TrainLoss, float? ValLoss, TimeSpan Elapsed);

    /// <summary>Returned by <see cref="Trainer.Train"/> when the run finishes.</summary>
    public sealed record TrainSummary(float FinalTrainLoss, float? FinalValLoss, int Steps, TimeSpan Elapsed);

    /// <summary>Steering command returned by the control hook passed to <see cref="Trainer.Train"/>.</summary>
    public enum TrainCommand
    {
        /// <summary>Keep training.</summary>
        Continue,
        /// <summary>Stop after the current step, exactly as if cancelled (the caller still saves).</summary>
        SaveAndQuit,
    }

    /// <summary>
    /// Batched training loop. Each optimizer step runs
    /// <see cref="TrainOptions.AccumulationSteps"/> physical batches of
    /// <see cref="TrainOptions.BatchSize"/> sequences, averages their gradients,
    /// clips the global norm once, and applies one AdamW update on the
    /// warmup+cosine learning-rate schedule.
    /// </summary>
    public static class Trainer
    {
        /// <summary>
        /// Trains <paramref name="model"/> in place. <paramref name="onLog"/> is
        /// invoked at step 0 and then every <see cref="TrainOptions.LogEvery"/>
        /// steps; a val loss (averaged over a fixed set of
        /// <see cref="TrainOptions.ValBatches"/> physical-size batches) is included
        /// every <see cref="TrainOptions.ValEvery"/> steps
        /// when a val loader is given. <paramref name="onSave"/> is invoked with the
        /// model and complete resumable state every <see cref="TrainOptions.SaveEvery"/>
        /// steps (never on the final step — the caller saves then). Cancelling
        /// <paramref name="cancel"/> stops after the current step; the summary's
        /// Steps reflects how many steps actually ran. <paramref name="controlHook"/>
        /// is invoked once per step (after the optimizer step, before logging) with
        /// the 1-based step number; returning <see cref="TrainCommand.SaveAndQuit"/>
        /// stops the run exactly like cancellation.
        /// </summary>
        public static TrainSummary Train(GptModel model, DataLoader train, DataLoader? val,
            TrainOptions opts, Action<TrainLog>? onLog = null,
            Action<GptModel, TrainingState>? onSave = null, CancellationToken cancel = default,
            Func<int, TrainCommand>? controlHook = null, TrainingState? state = null)
        {
            if (val is null && opts.ValEvery != 0)
                throw new ArgumentException("ValEvery is set but no val DataLoader was given.", nameof(val));
            if (opts.BatchSize < 1)
                throw new ArgumentException("BatchSize must be >= 1.", nameof(opts));
            if (opts.AccumulationSteps < 1)
                throw new ArgumentException("AccumulationSteps must be >= 1.", nameof(opts));
            if (opts.ValBatches < 1)
                throw new ArgumentException("ValBatches must be >= 1.", nameof(opts));
            int ctx = opts.ContextLength > 0 ? opts.ContextLength : model.Config.ContextLength;
            int batch = opts.BatchSize;
            int accumulation = opts.AccumulationSteps;
            state ??= TrainingState.CreateNew(model.Backend, opts);
            state.RequireCompatible(opts);
            TrainingRandom rng = state.DataRandom;
            AdamW adam = state.Optimizer;
            int[] inputs = new int[batch * ctx], targets = new int[batch * ctx];
            int[] seqInputs = new int[ctx], seqTargets = new int[ctx];
            var sw = Stopwatch.StartNew();
            bool prof = Environment.GetEnvironmentVariable("LLM_GPU_STATS") == "1";

            float lastTrain = 0f, lastVal = float.NaN;
            int stepsRun = state.GlobalStep;
            for (int step = state.GlobalStep; step < opts.Steps; step++)
            {
                if (cancel.IsCancellationRequested) break;
                float lr = LrSchedule.GetLr(step, opts.Steps, opts.MaxLr, opts.MinLr, opts.WarmupSteps);

                long p0 = prof ? Stopwatch.GetTimestamp() : 0;
                model.Params.ZeroGrads();
                long p1 = prof ? Stopwatch.GetTimestamp() : 0;
                double trainLossSum = 0;
                for (int microStep = 0; microStep < accumulation; microStep++)
                {
                    for (int b = 0; b < batch; b++)
                    {
                        train.Sample(rng, ctx, seqInputs, seqTargets);
                        Array.Copy(seqInputs, 0, inputs, b * ctx, ctx);
                        Array.Copy(seqTargets, 0, targets, b * ctx, ctx);
                    }
                    trainLossSum += model.ForwardBackward(inputs, targets, batch);
                }
                lastTrain = (float)(trainLossSum / accumulation);
                if (accumulation > 1)
                    ScaleGradients(model.Params, model.Backend, 1f / accumulation);
                long p2 = prof ? Stopwatch.GetTimestamp() : 0;
                ClipGradNorm(model.Params, model.Backend, opts.GradClip);
                long p3 = prof ? Stopwatch.GetTimestamp() : 0;
                adam.Step(model.Params, lr, weightDecay: opts.WeightDecay);
                stepsRun = step + 1;
                state.GlobalStep = stepsRun;
                if (prof)
                {
                    long p4 = Stopwatch.GetTimestamp();
                    double Ms(long a, long b) => (b - a) * 1000.0 / Stopwatch.Frequency;
                    Console.Error.WriteLine($"[step {step + 1}] zero {Ms(p0, p1):F0}ms  fwdbwd {Ms(p1, p2):F0}ms  clip {Ms(p2, p3):F0}ms  adam {Ms(p3, p4):F0}ms");
                    model.Backend.DumpStats($"step {step + 1}");
                }

                if (controlHook?.Invoke(step + 1) == TrainCommand.SaveAndQuit) break;

                if (step == 0 || (step + 1) % opts.LogEvery == 0 || step + 1 == opts.Steps)
                {
                    float? valLoss = null;
                    if (val is not null && opts.ValEvery > 0 &&
                        (step == 0 || (step + 1) % opts.ValEvery == 0 || step + 1 == opts.Steps))
                    {
                        valLoss = EvalLoss(model, val, ctx, batch, opts.ValBatches, opts.ValSeed);
                        lastVal = valLoss.Value;
                    }
                    onLog?.Invoke(new TrainLog(step + 1, lr, lastTrain, valLoss, sw.Elapsed));
                }

                if (onSave is not null && opts.SaveEvery > 0 && stepsRun < opts.Steps && stepsRun % opts.SaveEvery == 0)
                    onSave(model, state);
            }
            sw.Stop();
            return new TrainSummary(lastTrain, float.IsNaN(lastVal) ? null : lastVal, stepsRun, sw.Elapsed);
        }

        /// <summary>
        /// Forward-only mean over a fixed validation set. Batches are evaluated one at
        /// a time so increasing sample count does not increase peak activation memory.
        /// A fresh dedicated RNG makes every evaluation directly comparable and keeps
        /// validation from perturbing the training sampler trajectory.
        /// </summary>
        private static float EvalLoss(GptModel model, DataLoader data, int ctx, int batch,
            int valBatches, int valSeed)
        {
            var rng = new TrainingRandom(valSeed);
            int[] inputs = new int[batch * ctx], targets = new int[batch * ctx];
            int[] seqInputs = new int[ctx], seqTargets = new int[ctx];
            double lossSum = 0;
            for (int evalBatch = 0; evalBatch < valBatches; evalBatch++)
            {
                for (int b = 0; b < batch; b++)
                {
                    data.Sample(rng, ctx, seqInputs, seqTargets);
                    Array.Copy(seqInputs, 0, inputs, b * ctx, ctx);
                    Array.Copy(seqTargets, 0, targets, b * ctx, ctx);
                }
                lossSum += model.EvaluateLoss(inputs, targets, batch);
            }
            return (float)(lossSum / valBatches);
        }

        /// <summary>Scales all gradients so the global L2 norm does not exceed maxNorm.</summary>
        private static void ClipGradNorm(Parameters p, ITensorBackend backend, float maxNorm)
        {
            if (maxNorm <= 0f) return;
            double sumSq = 0;
            foreach (string name in p.Names)
                sumSq += backend.SumSquares(p.Grad(name)); // device-side reduction when supported
            float norm = MathF.Sqrt((float)sumSq);
            if (norm <= maxNorm || norm == 0f) return;
            float scale = maxNorm / norm;
            foreach (string name in p.Names)
                backend.Scale(p.Grad(name), scale);
        }

        private static void ScaleGradients(Parameters p, ITensorBackend backend, float scale)
        {
            foreach (string name in p.Names)
                backend.Scale(p.Grad(name), scale);
        }
    }
}
