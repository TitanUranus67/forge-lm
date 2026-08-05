
namespace LLM.Core.Tests
{
    using LLM.Core.Model;
    using LLM.Core.Tensor;
    using LLM.Core.Training;
    using Tensor = LLM.Core.Tensor.Tensor;

    /// <summary>
    /// Tests for batched training (GptModel.ForwardBackward with a batch dimension):
    /// equivalence with the single-sequence path (loss = mean of per-sequence losses,
    /// grads = mean of per-sequence grads, B=1 identical), a finite-difference
    /// gradient check through the batched path, a batched Trainer run, and a
    /// fixed-batch overfit smoke test that would catch attention leaking across
    /// sequence boundaries.
    /// </summary>
    public static class BatchedModelTests
    {
        private static readonly CpuBackend B = new();

        /// <summary>Small config for most tests: 7776 parameters.</summary>
        private static ModelConfig Small => new(VocabSize: 32, ContextLength: 8, DModel: 16, NLayers: 2, NHeads: 2);

        /// <summary>Even tinier config for the gradient check: 1140 parameters.</summary>
        private static ModelConfig Tiny => new(VocabSize: 12, ContextLength: 6, DModel: 8, NLayers: 1, NHeads: 2);

        /// <summary>Deterministic batch of B random sequences, flattened sequence-major.</summary>
        private static (int[] Inputs, int[] Targets) MakeBatch(int batch, int ctx, int vocab, int seed)
        {
            var rng = new Random(seed);
            int[] inputs = new int[batch * ctx], targets = new int[batch * ctx];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = rng.Next(vocab);
                targets[i] = rng.Next(vocab);
            }
            return (inputs, targets);
        }

        private static Dictionary<string, float[]> SnapshotGrads(GptModel model)
        {
            var snap = new Dictionary<string, float[]>();
            foreach (string name in model.Params.Names)
                snap[name] = (float[])model.Params.Grad(name).Data.Clone();
            return snap;
        }

        [Test]
        public static void BatchVsSingle_LossAndGradsAreMean()
        {
            var model = new GptModel(Small, B, new Random(11));
            const int batch = 3;
            int ctx = Small.ContextLength;
            var (inputs, targets) = MakeBatch(batch, ctx, Small.VocabSize, 99);

            // reference: three independent single-sequence forward/backward passes
            float lossSum = 0f;
            var gradSum = new Dictionary<string, float[]>();
            for (int b = 0; b < batch; b++)
            {
                int[] inB = inputs.AsSpan(b * ctx, ctx).ToArray();
                int[] tgB = targets.AsSpan(b * ctx, ctx).ToArray();
                model.Params.ZeroGrads();
                lossSum += model.ForwardBackward(inB, tgB);
                foreach (string name in model.Params.Names)
                {
                    Tensor g = model.Params.Grad(name);
                    if (!gradSum.TryGetValue(name, out float[]? acc))
                        gradSum[name] = acc = new float[g.Length];
                    for (int i = 0; i < g.Length; i++) acc[i] += g.Data[i];
                }
            }

            model.Params.ZeroGrads();
            float batchLoss = model.ForwardBackward(inputs, targets, batch);
            Check.Near(batchLoss, lossSum / batch, 1e-5f,
                $"batched loss {batchLoss:G6} equals mean of single-sequence losses {lossSum / batch:G6}");

            foreach (string name in model.Params.Names)
            {
                float[] acc = gradSum[name];
                var mean = new float[acc.Length];
                for (int i = 0; i < mean.Length; i++) mean[i] = acc[i] / batch;
                Check.SpanNear(model.Params.Grad(name).Data, mean, 1e-4f,
                    $"batched grad equals mean of single-sequence grads: {name}");
            }
        }

        [Test]
        public static void GradientCheck_BatchedModel()
        {
            var config = Tiny;
            var model = new GptModel(config, B, new Random(123));
            const int batch = 2, ctx = 4;
            var (inputs, targets) = MakeBatch(batch, ctx, config.VocabSize, 77);
            const float eps = 1e-3f, tol = 1e-2f;

            model.Params.ZeroGrads();
            float loss = model.ForwardBackward(inputs, targets, batch);
            Check.True(loss > 0f && float.IsFinite(loss), "batched loss is positive and finite");
            Dictionary<string, float[]> analytic = SnapshotGrads(model);

            float LossFn()
            {
                model.Params.ZeroGrads();
                return model.ForwardBackward(inputs, targets, batch);
            }

            float maxErr = 0f;
            string worst = "";
            foreach (string name in model.Params.Names)
            {
                Tensor w = model.Params.Weight(name);
                float[] g = analytic[name];
                for (int i = 0; i < w.Length; i++)
                {
                    float orig = w.Data[i];
                    w.Data[i] = orig + eps;
                    float hi = LossFn();
                    w.Data[i] = orig - eps;
                    float lo = LossFn();
                    w.Data[i] = orig;
                    float numeric = (hi - lo) / (2f * eps);
                    float err = Math.Abs(numeric - g[i]) / Math.Max(1f, Math.Abs(numeric));
                    if (err > maxErr) { maxErr = err; worst = $"{name}[{i}] numeric={numeric:G6} analytic={g[i]:G6}"; }
                }
            }
            Check.True(maxErr < tol, $"max batched grad error {maxErr:G4} < {tol} (worst: {worst})");
            Console.WriteLine($"    batched grad-check max rel/abs err {maxErr:G4} (worst: {worst})");
        }

        [Test]
        public static void BatchOfOne_MatchesSingleSequencePath()
        {
            var model = new GptModel(Small, B, new Random(21));
            int[] inputs = { 3, 1, 4, 1, 5, 9, 2, 6 };
            int[] targets = { 1, 4, 1, 5, 9, 2, 6, 5 };

            model.Params.ZeroGrads();
            float single = model.ForwardBackward(inputs, targets);
            Dictionary<string, float[]> g1 = SnapshotGrads(model);

            model.Params.ZeroGrads();
            float batched = model.ForwardBackward(inputs, targets, batch: 1);
            Check.Near(batched, single, 1e-6f, $"B=1 batched loss {batched:G6} == single-sequence loss {single:G6}");
            foreach (string name in model.Params.Names)
                Check.SpanNear(model.Params.Grad(name).Data, g1[name], 1e-6f, $"B=1 batched grad == single-sequence grad: {name}");
        }

        [Test]
        public static void GradientScale_ScalesGradientsWithoutChangingLoss()
        {
            var full = new GptModel(Small, B, new Random(22));
            var scaled = new GptModel(Small, B, new Random(22));
            var (inputs, targets) = MakeBatch(batch: 2, ctx: Small.ContextLength,
                vocab: Small.VocabSize, seed: 101);

            full.Params.ZeroGrads();
            float fullLoss = full.ForwardBackward(inputs, targets, batch: 2);
            scaled.Params.ZeroGrads();
            float scaledLoss = scaled.ForwardBackward(inputs, targets, batch: 2, gradientScale: 0.25f);

            Check.Near(scaledLoss, fullLoss, 0f, "gradient scaling does not change reported loss");
            foreach (string name in full.Params.Names)
            {
                float[] expected = full.Params.Grad(name).Data.Select(x => x * 0.25f).ToArray();
                Check.SpanNear(scaled.Params.Grad(name).Data, expected, 2e-6f,
                    $"gradient scaling applies before backward: {name}");
            }
        }

        [Test]
        public static void AttentionActivationCache_MatchesRecomputation()
        {
            var recomputed = new GptModel(Small, B, new Random(71));
            var cached = new GptModel(Small, B, new Random(71),
                cacheAttentionActivations: true);
            var (inputs, targets) = MakeBatch(batch: 3, ctx: Small.ContextLength,
                vocab: Small.VocabSize, seed: 72);

            recomputed.Params.ZeroGrads();
            float recomputedLoss = recomputed.ForwardBackward(inputs, targets, batch: 3);
            cached.Params.ZeroGrads();
            float cachedLoss = cached.ForwardBackward(inputs, targets, batch: 3);

            Check.True(!recomputed.CachesAttentionActivations,
                "attention recomputation remains the default");
            Check.True(cached.CachesAttentionActivations,
                "attention activation caching is explicitly enabled");
            Check.Near(cachedLoss, recomputedLoss, 0f,
                "attention activation caching preserves loss");
            foreach (string name in recomputed.Params.Names)
                Check.SpanNear(cached.Params.Grad(name).Data,
                    recomputed.Params.Grad(name).Data, 0f,
                    $"attention activation caching preserves gradient: {name}");
        }

        [Test]
        public static void Trainer_BatchedTrainLossDrops()
        {
            // Repetitive data: next token is fully predictable (i -> (i+1) mod 16).
            string path = Path.GetTempFileName();
            try
            {
                using (var bw = new BinaryWriter(File.Create(path)))
                    for (int i = 0; i < 4096; i++) bw.Write((ushort)(i % 16));

                var train = new DataLoader(path);
                var val = new DataLoader(path);
                var model = new GptModel(Small, B, new Random(3));
                var opts = new TrainOptions
                {
                    Steps = 300,
                    MaxLr = 3e-3f,
                    MinLr = 3e-4f,
                    WarmupSteps = 10,
                    WeightDecay = 0.1f,
                    GradClip = 1.0f,
                    BatchSize = 4,
                    Seed = 123,
                    LogEvery = 50,
                    ValEvery = 100,
                    ValBatches = 2,
                };

                var logs = new List<TrainLog>();
                TrainSummary summary = Trainer.Train(model, train, val, opts, logs.Add);

                float initial = logs[0].TrainLoss;
                Check.True(summary.Steps == 300, "summary reports step count");
                Check.True(summary.FinalValLoss is not null, "val loss was evaluated");
                Check.True(summary.FinalTrainLoss < 0.25f * initial,
                    $"batched train loss {initial:F3} -> {summary.FinalTrainLoss:F3}, expected < {0.25f * initial:F3}");
                Console.WriteLine($"    batched trainer loss {initial:F4} -> {summary.FinalTrainLoss:F4} " +
                                  $"(val {summary.FinalValLoss:F4})");
            }
            finally { File.Delete(path); }
        }

        [Test]
        public static void Overfit_FixedBatchLossGoesToNearZero()
        {
            // One FIXED batch of 4 sequences, 200 plain AdamW steps: the loss must
            // collapse. Attention leaking across sequence boundaries would prevent it.
            // (AdamW's per-step move is ~lr, so the lr is larger than typical training.)
            // The sequences are structured (targets = next token of a cyclic pattern),
            // so the mapping context -> target is consistent and fully memorizable.
            var config = Tiny;
            var model = new GptModel(config, B, new Random(5));
            const int batch = 4;
            int ctx = config.ContextLength;
            int[] inputs = new int[batch * ctx], targets = new int[batch * ctx];
            for (int b = 0; b < batch; b++)
                for (int t = 0; t < ctx; t++)
                {
                    inputs[b * ctx + t] = (b * 3 + t) % config.VocabSize;
                    targets[b * ctx + t] = (b * 3 + t + 1) % config.VocabSize;
                }
            var adam = new AdamW();

            model.Params.ZeroGrads();
            float initial = model.ForwardBackward(inputs, targets, batch);
            for (int step = 0; step < 200; step++)
            {
                model.Params.ZeroGrads();
                model.ForwardBackward(inputs, targets, batch);
                adam.Step(model.Params, lr: 3e-2f, weightDecay: 0f);
            }
            model.Params.ZeroGrads();
            float final = model.ForwardBackward(inputs, targets, batch);
            Check.True(final < 0.05f,
                $"overfit fixed batch: loss {initial:F3} -> {final:F4}, expected near zero");
            Console.WriteLine($"    batched overfit loss {initial:F4} -> {final:F4} after 200 AdamW steps (batch {batch})");
        }
    }
}
