
namespace LLM.Core.Tests
{
    using LLM.Core.Model;
    using LLM.Core.Tensor;
    using Tensor = LLM.Core.Tensor.Tensor;

    /// <summary>
    /// Tests for the GPT model: forward shape/determinism, init statistics, a full-model
    /// finite-difference gradient check over every parameter tensor, loss-at-init sanity,
    /// and an overfit smoke test for the composed training path.
    /// </summary>
    public static class ModelTests
    {
        private static readonly CpuBackend B = new();

        /// <summary>Small config for most tests: 7776 parameters.</summary>
        private static ModelConfig Small => new(VocabSize: 32, ContextLength: 8, DModel: 16, NLayers: 2, NHeads: 2);

        /// <summary>Even tinier config for the gradient check: 1140 parameters.</summary>
        private static ModelConfig Tiny => new(VocabSize: 12, ContextLength: 6, DModel: 8, NLayers: 1, NHeads: 2);

        [Test]
        public static void Forward_OutputShape()
        {
            var model = new GptModel(Small, B, new Random(1));
            int[] tokens = { 1, 2, 3, 4 };
            Tensor logits = model.Forward(tokens);
            Check.True(logits.Shape.Length == 2 && logits.Shape[0] == tokens.Length && logits.Shape[1] == Small.VocabSize,
                $"logits shape [{string.Join(",", logits.Shape)}] should be [{tokens.Length},{Small.VocabSize}]");

            Tensor last = model.ForwardLast(tokens);
            Check.True(last.Shape.Length == 2 && last.Shape[0] == 1 && last.Shape[1] == Small.VocabSize,
                $"ForwardLast shape [{string.Join(",", last.Shape)}] should be [1,{Small.VocabSize}]");
            for (int v = 0; v < Small.VocabSize; v++)
                Check.Near(last.Data[v], logits.Data[(tokens.Length - 1) * Small.VocabSize + v], 0f, $"ForwardLast row equals last logits row [{v}]");

            // VD + CD + L*(12D^2+13D) + 2D + (D+1)*V = 512 + 128 + 2*3280 + 32 + 544
            Check.True(model.Params.Count == 7776, $"param count {model.Params.Count} should be 7776");
        }

        [Test]
        public static void Forward_DeterministicPerSeed()
        {
            int[] tokens = { 3, 1, 4, 1, 5 };
            Tensor a = new GptModel(Small, B, new Random(42)).Forward(tokens);
            Tensor b = new GptModel(Small, B, new Random(42)).Forward(tokens);
            Tensor c = new GptModel(Small, B, new Random(43)).Forward(tokens);
            Check.SpanNear(a.Data, b.Data, 0f, "same seed -> identical logits");
            bool anyDiff = false;
            for (int i = 0; i < a.Length; i++) anyDiff |= a.Data[i] != c.Data[i];
            Check.True(anyDiff, "different seed -> different logits");
        }

        [Test]
        public static void Init_ResidualProjectionsUseScaledStd()
        {
            // GPT-2 init: residual-stream projections use std 0.02/sqrt(2*NLayers) = 0.01 here.
            var model = new GptModel(Small, B, new Random(7));
            float expectedProj = 0.02f / MathF.Sqrt(2f * Small.NLayers);

            foreach (string name in new[] { "blocks.0.attn.qkv.w", "blocks.0.attn.proj.w", "blocks.1.mlp.fc.w", "blocks.1.mlp.proj.w" })
            {
                Tensor w = model.Params.Weight(name);
                double sum = 0, sq = 0;
                foreach (float x in w.Data) { sum += x; sq += (double)x * x; }
                double std = Math.Sqrt(sq / w.Length - (sum / w.Length) * (sum / w.Length));
                double expected = name.Contains("proj") ? expectedProj : 0.02;
                Check.True(std > expected * 0.7 && std < expected * 1.3,
                    $"{name} sample std {std:F4} should be ~{expected:F4}");
            }

            Tensor lnW = model.Params.Weight("blocks.0.ln1.w");
            foreach (float x in lnW.Data) Check.Near(x, 1f, 0f, "layernorm weight initialized to 1");
            Tensor lnB = model.Params.Weight("ln_f.b");
            foreach (float x in lnB.Data) Check.Near(x, 0f, 0f, "layernorm bias initialized to 0");
        }

        [Test]
        public static void GradientCheck_FullModel()
        {
            var config = Tiny;
            var model = new GptModel(config, B, new Random(123));
            int[] inputs = { 1, 2, 3, 4 };
            int[] targets = { 2, 3, 4, 5 };
            const float eps = 1e-3f, tol = 1e-2f;

            model.Params.ZeroGrads();
            float loss = model.ForwardBackward(inputs, targets);
            Check.True(loss > 0f && float.IsFinite(loss), "loss is positive and finite");

            float LossFn()
            {
                Tensor logits = model.Forward(inputs);
                var probs = new Tensor(inputs.Length, config.VocabSize);
                return B.CrossEntropyForward(logits.Data, targets, probs.Data, inputs.Length, config.VocabSize, -1);
            }

            float maxErr = 0f;
            string worst = "";
            foreach (string name in model.Params.Names)
            {
                Tensor w = model.Params.Weight(name);
                Tensor g = model.Params.Grad(name);
                for (int i = 0; i < w.Length; i++)
                {
                    float orig = w.Data[i];
                    w.Data[i] = orig + eps;
                    float hi = LossFn();
                    w.Data[i] = orig - eps;
                    float lo = LossFn();
                    w.Data[i] = orig;
                    float numeric = (hi - lo) / (2f * eps);
                    // relative-or-absolute error, matching the kernel grad checks
                    float err = Math.Abs(numeric - g.Data[i]) / Math.Max(1f, Math.Abs(numeric));
                    if (err > maxErr) { maxErr = err; worst = $"{name}[{i}] numeric={numeric:G6} analytic={g.Data[i]:G6}"; }
                }
            }
            Check.True(maxErr < tol, $"max grad error {maxErr:G4} < {tol} (worst: {worst})");
            Console.WriteLine($"    grad-check max rel/abs err {maxErr:G4} (worst: {worst})");
        }

        [Test]
        public static void Loss_AtInitNearLnVocab()
        {
            var config = Small;
            var model = new GptModel(config, B, new Random(9));
            var rng = new Random(10);
            int[] inputs = new int[8], targets = new int[8];
            for (int i = 0; i < 8; i++) { inputs[i] = rng.Next(config.VocabSize); targets[i] = rng.Next(config.VocabSize); }
            model.Params.ZeroGrads();
            float loss = model.ForwardBackward(inputs, targets);
            float expected = MathF.Log(config.VocabSize);
            Check.Near(loss, expected, 0.5f, $"random-init loss ~ ln(V) = {expected:F3}");
        }

        [Test]
        public static void Overfit_SmallBatchLossDrops()
        {
            var config = Tiny;
            var model = new GptModel(config, B, new Random(5));
            int[] inputs = { 0, 1, 2, 3, 4, 5 };
            int[] targets = { 1, 2, 3, 4, 5, 0 };
            const float lr = 0.05f;

            model.Params.ZeroGrads();
            float initial = model.ForwardBackward(inputs, targets);
            for (int step = 0; step < 200; step++)
            {
                model.Params.ZeroGrads();
                model.ForwardBackward(inputs, targets);
                foreach (string name in model.Params.Names)
                {
                    Tensor w = model.Params.Weight(name);
                    Tensor g = model.Params.Grad(name);
                    for (int i = 0; i < w.Length; i++)
                        w.Data[i] -= lr * g.Data[i];
                }
            }
            model.Params.ZeroGrads();
            float final = model.ForwardBackward(inputs, targets);
            Check.True(final < 0.2f * initial,
                $"overfit: loss {initial:F3} -> {final:F3}, expected final < {0.2f * initial:F3}");
            Console.WriteLine($"    overfit loss {initial:F4} -> {final:F4} after 200 SGD steps");
        }
    }
}
