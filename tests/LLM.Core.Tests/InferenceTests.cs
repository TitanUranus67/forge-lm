
namespace LLM.Core.Tests
{
    using LLM.Core.Inference;
    using LLM.Core.Model;
    using LLM.Core.Tensor;

    /// <summary>
    /// Tests for sampling: greedy/top-k/temperature behavior, the categorical
    /// distribution against an analytic softmax, and the autoregressive
    /// Generate loop (count, bounds, sliding window past the context length).
    /// </summary>
    public static class InferenceTests
    {
        private static readonly CpuBackend B = new();

        private static ModelConfig Small => new(VocabSize: 32, ContextLength: 8, DModel: 16, NLayers: 2, NHeads: 2);

        [Test]
        public static void Sample_ZeroTemperatureIsArgmax()
        {
            float[] logits = { 0.1f, 2.5f, -1f, 2.4f };
            var rng = new Random(1);
            for (int i = 0; i < 20; i++)
                Check.True(Sampler.Sample(logits, temperature: 0f, topK: 0, rng) == 1, "temperature=0 -> argmax");
            Check.True(Sampler.Greedy(logits) == 1, "Greedy picks the max");
        }

        [Test]
        public static void Sample_TopK1IsArgmax()
        {
            float[] logits = { -3f, 0.5f, 0.4f, 0.6f };
            var rng = new Random(2);
            for (int i = 0; i < 20; i++)
                Check.True(Sampler.Sample(logits, temperature: 1f, topK: 1, rng) == 3, "topK=1 -> argmax");
        }

        [Test]
        public static void Sample_DominantLogitWinsAlmostAlways()
        {
            float[] logits = { 10f, 0f, 0f, 0f };
            var rng = new Random(3);
            int hits = 0;
            const int n = 1000;
            for (int i = 0; i < n; i++)
                if (Sampler.Sample(logits, 1f, 0, rng) == 0) hits++;
            Check.True(hits > 0.95 * n, $"dominant logit sampled {hits}/{n} times, expected >95%");
        }

        [Test]
        public static void Sample_DistributionMatchesSoftmax()
        {
            float[] logits = { 1f, 2f, 3f };
            var expected = new float[3];
            float sum = 0f;
            for (int i = 0; i < 3; i++) { expected[i] = MathF.Exp(logits[i]); sum += expected[i]; }
            for (int i = 0; i < 3; i++) expected[i] /= sum;

            var rng = new Random(4);
            var counts = new int[3];
            const int n = 20000;
            for (int i = 0; i < n; i++) counts[Sampler.Sample(logits, 1f, 0, rng)]++;
            for (int i = 0; i < 3; i++)
                Check.Near(counts[i] / (float)n, expected[i], 0.03f, $"sample frequency[{i}] ~ softmax");
        }

        [Test]
        public static void Sample_TopKZeroAndAboveVocabMeansNoFiltering()
        {
            float[] logits = { 0f, 0f, 0f, 5f }; // without filtering, index 3 dominates anyway
            var rng = new Random(5);
            for (int i = 0; i < 10; i++)
            {
                int a = Sampler.Sample(logits, 1f, 0, rng);
                int b = Sampler.Sample(logits, 1f, 99, rng);
                Check.True((uint)a < 4 && (uint)b < 4, "sampled ids in range");
            }
        }

        [Test]
        public static void Generate_YieldsRequestedCountInRange()
        {
            var model = new GptModel(Small, B, new Random(6));
            var outTokens = Sampler.Generate(model, new[] { 1, 2 }, maxNewTokens: 5,
                temperature: 1f, topK: 0, new Random(7)).ToList();
            Check.True(outTokens.Count == 5, $"expected 5 tokens, got {outTokens.Count}");
            foreach (int t in outTokens)
                Check.True((uint)t < (uint)Small.VocabSize, $"token {t} in [0,{Small.VocabSize})");
        }

        [Test]
        public static void Generate_SlidesWindowPastContextLength()
        {
            var model = new GptModel(Small, B, new Random(8));
            int[] prompt = Enumerable.Range(0, 12).Select(i => i % Small.VocabSize).ToArray(); // longer than ctx=8
            var outTokens = Sampler.Generate(model, prompt, maxNewTokens: 6,
                temperature: 0f, topK: 0, new Random(9)).ToList();
            Check.True(outTokens.Count == 6, "generation works with a prompt longer than ContextLength");
        }

        [Test]
        public static void Generate_StopsAtEos()
        {
            var model = new GptModel(Small, B, new Random(10));
            // Greedy decoding is deterministic, so find which token the model
            // would emit first and declare it the eos: generation must stop at 1.
            int first = Sampler.Generate(model, new[] { 3, 1 }, 1, 0f, 0, new Random(11)).First();
            var outTokens = Sampler.Generate(model, new[] { 3, 1 }, maxNewTokens: 10,
                temperature: 0f, topK: 0, new Random(11), eosId: first).ToList();
            Check.True(outTokens.Count == 1 && outTokens[0] == first,
                $"generation stops after yielding eos (got {outTokens.Count} tokens)");
        }

        [Test]
        public static void InvalidSamplingOptionsFailLoudly()
        {
            bool emptyThrew = false;
            try { Sampler.Sample(ReadOnlySpan<float>.Empty, 1f, 0, new Random(1)); }
            catch (ArgumentException) { emptyThrew = true; }
            Check.True(emptyThrew, "sampling an empty logit vector throws");

            bool temperatureThrew = false;
            try { Sampler.Sample(new[] { 1f }, float.NaN, 0, new Random(1)); }
            catch (ArgumentException) { temperatureThrew = true; }
            Check.True(temperatureThrew, "non-finite temperature throws");

            var model = new GptModel(Small, B, new Random(2));
            bool countThrew = false;
            try { _ = Sampler.Generate(model, new[] { 1 }, -1, 1f, 0, new Random(1)).ToList(); }
            catch (ArgumentOutOfRangeException) { countThrew = true; }
            Check.True(countThrew, "negative generation length throws");
        }
    }
}
