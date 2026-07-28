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
        /// <summary>RNG seed for data sampling (and therefore reproducibility).</summary>
        public int Seed { get; init; } = 1337;
        /// <summary>Emit a log record every this many steps.</summary>
        public int LogEvery { get; init; } = 50;
        /// <summary>Evaluate validation loss every this many steps; 0 disables it.</summary>
        public int ValEvery { get; init; } = 0;
        /// <summary>Number of val sequences averaged per evaluation.</summary>
        public int ValSamples { get; init; } = 5;
        /// <summary>Invoke the save callback every this many steps; 0 disables it.</summary>
        public int SaveEvery { get; init; } = 0;
    }

    /// <summary>One progress record emitted to the Train callback.</summary>
    public sealed record TrainLog(int Step, float Lr, float TrainLoss, float? ValLoss, TimeSpan Elapsed);

    /// <summary>Returned by <see cref="Trainer.Train"/> when the run finishes.</summary>
    public sealed record TrainSummary(float FinalTrainLoss, float? FinalValLoss, int Steps, TimeSpan Elapsed);

    /// <summary>
    /// Minimal single-sequence training loop. Each step samples ONE sequence of
    /// ContextLength tokens (no batching, no gradient accumulation — the model
    /// processes [T, C] tensors without a batch dimension), runs forward/backward,
    /// clips the global gradient norm, and applies one AdamW update on the
    /// warmup+cosine learning-rate schedule.
    /// </summary>
    public static class Trainer
    {
        /// <summary>
        /// Trains <paramref name="model"/> in place. <paramref name="onLog"/> is
        /// invoked at step 0 and then every <see cref="TrainOptions.LogEvery"/>
        /// steps; a val loss (averaged over <see cref="TrainOptions.ValSamples"/>
        /// sequences) is included every <see cref="TrainOptions.ValEvery"/> steps
        /// when a val loader is given. <paramref name="onSave"/> is invoked with the
        /// model and completed-step count every <see cref="TrainOptions.SaveEvery"/>
        /// steps (never on the final step — the caller saves then). Cancelling
        /// <paramref name="cancel"/> stops after the current step; the summary's
        /// Steps reflects how many steps actually ran.
        /// </summary>
        public static TrainSummary Train(GptModel model, DataLoader train, DataLoader? val,
            TrainOptions opts, Action<TrainLog>? onLog = null,
            Action<GptModel, int>? onSave = null, CancellationToken cancel = default)
        {
            if (val is null && opts.ValEvery != 0)
                throw new ArgumentException("ValEvery is set but no val DataLoader was given.", nameof(val));
            int ctx = opts.ContextLength > 0 ? opts.ContextLength : model.Config.ContextLength;
            var rng = new Random(opts.Seed);
            var adam = new AdamW();
            int[] inputs = new int[ctx], targets = new int[ctx];
            var sw = Stopwatch.StartNew();

            float lastTrain = 0f, lastVal = float.NaN;
            int stepsRun = 0;
            for (int step = 0; step < opts.Steps; step++)
            {
                if (cancel.IsCancellationRequested) break;
                float lr = LrSchedule.GetLr(step, opts.Steps, opts.MaxLr, opts.MinLr, opts.WarmupSteps);

                train.Sample(rng, ctx, inputs, targets);
                model.Params.ZeroGrads();
                lastTrain = model.ForwardBackward(inputs, targets);
                ClipGradNorm(model.Params, opts.GradClip);
                adam.Step(model.Params, lr, weightDecay: opts.WeightDecay);
                stepsRun = step + 1;

                if (step == 0 || (step + 1) % opts.LogEvery == 0 || step + 1 == opts.Steps)
                {
                    float? valLoss = null;
                    if (val is not null && opts.ValEvery > 0 &&
                        (step == 0 || (step + 1) % opts.ValEvery == 0 || step + 1 == opts.Steps))
                    {
                        valLoss = EvalLoss(model, val, rng, ctx, opts.ValSamples);
                        lastVal = valLoss.Value;
                    }
                    onLog?.Invoke(new TrainLog(step + 1, lr, lastTrain, valLoss, sw.Elapsed));
                }

                if (onSave is not null && opts.SaveEvery > 0 && stepsRun < opts.Steps && stepsRun % opts.SaveEvery == 0)
                    onSave(model, stepsRun);
            }
            sw.Stop();
            return new TrainSummary(lastTrain, float.IsNaN(lastVal) ? null : lastVal, stepsRun, sw.Elapsed);
        }

        /// <summary>Mean loss of <paramref name="samples"/> random sequences; leaves grads dirty.</summary>
        private static float EvalLoss(GptModel model, DataLoader data, Random rng, int ctx, int samples)
        {
            int[] inputs = new int[ctx], targets = new int[ctx];
            float sum = 0f;
            for (int i = 0; i < samples; i++)
            {
                data.Sample(rng, ctx, inputs, targets);
                model.Params.ZeroGrads();
                sum += model.ForwardBackward(inputs, targets);
            }
            return sum / samples;
        }

        /// <summary>Scales all gradients so the global L2 norm does not exceed maxNorm.</summary>
        private static void ClipGradNorm(Parameters p, float maxNorm)
        {
            if (maxNorm <= 0f) return;
            double sumSq = 0;
            foreach (string name in p.Names)
                foreach (float g in p.Grad(name).Data)
                    sumSq += (double)g * g;
            float norm = MathF.Sqrt((float)sumSq);
            if (norm <= maxNorm || norm == 0f) return;
            float scale = maxNorm / norm;
            foreach (string name in p.Names)
            {
                Tensor g = p.Grad(name);
                for (int i = 0; i < g.Length; i++) g.Data[i] *= scale;
            }
        }
    }
}
