namespace LLM.Core.Inference
{
    using LLM.Core.Model;
    using LLM.Core.Tensor;

    /// <summary>
    /// Token sampling from a logit vector: repetition controls, temperature scaling,
    /// top-k filtering, softmax, and categorical sampling — plus an autoregressive
    /// <see cref="Generate"/> loop over <see cref="GptModel.ForwardLast"/>.
    /// </summary>
    public static class Sampler
    {
        /// <summary>Index of the largest logit (ties resolve to the first).</summary>
        public static int Greedy(ReadOnlySpan<float> logits)
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++)
                if (logits[i] > logits[best]) best = i;
            return best;
        }

        /// <summary>
        /// Samples one token id from <paramref name="logits"/>. The logits are
        /// divided by <paramref name="temperature"/> (&lt;= 0 means argmax),
        /// then filtered to the <paramref name="topK"/> highest values
        /// (k &lt;= 0 or &gt;= V disables filtering), then softmaxed and sampled
        /// categorically.
        /// </summary>
        public static int Sample(ReadOnlySpan<float> logits, float temperature, int topK, Random rng)
        {
            if (logits.IsEmpty) throw new ArgumentException("Need at least one logit.", nameof(logits));
            ArgumentNullException.ThrowIfNull(rng);
            if (!float.IsFinite(temperature))
                throw new ArgumentException("Temperature must be finite.", nameof(temperature));
            if (temperature <= 0f) return Greedy(logits);
            int v = logits.Length;

            var probs = new float[v];
            logits.CopyTo(probs);
            for (int i = 0; i < v; i++) probs[i] /= temperature;

            if (topK > 0 && topK < v)
            {
                // threshold = k-th largest scaled logit
                var sorted = (float[])probs.Clone();
                Array.Sort(sorted);
                float threshold = sorted[v - topK];
                for (int i = 0; i < v; i++)
                    if (probs[i] < threshold) probs[i] = float.NegativeInfinity;
            }

            float max = float.NegativeInfinity;
            for (int i = 0; i < v; i++) if (probs[i] > max) max = probs[i];
            float sum = 0f;
            for (int i = 0; i < v; i++)
            {
                probs[i] = MathF.Exp(probs[i] - max);
                sum += probs[i];
            }

            float u = (float)rng.NextDouble() * sum;
            float acc = 0f;
            for (int i = 0; i < v; i++)
            {
                acc += probs[i];
                if (u < acc) return i;
            }
            return v - 1; // rounding fallback
        }

        /// <summary>
        /// Autoregressive generation: feeds the prompt (kept as a sliding window
        /// truncated to the model's ContextLength from the left), samples the next
        /// token from the last position's logits, appends it, and repeats.
        /// Yields exactly <paramref name="maxNewTokens"/> tokens, or fewer if
        /// <paramref name="eosId"/> is sampled (the eos token itself is yielded).
        /// </summary>
        public static IEnumerable<int> Generate(GptModel model, IReadOnlyList<int> promptTokens,
            int maxNewTokens, float temperature, int topK, Random rng, int? eosId = null,
            float repetitionPenalty = 1f, int noRepeatNgramSize = 0)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(promptTokens);
            ArgumentNullException.ThrowIfNull(rng);
            if (promptTokens.Count == 0) throw new ArgumentException("Need at least one prompt token.");
            if (maxNewTokens < 0) throw new ArgumentOutOfRangeException(nameof(maxNewTokens));
            if (eosId is int validatedEos && (uint)validatedEos >= (uint)model.Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(eosId), "EOS token must be inside the model vocabulary.");
            ValidateRepetitionOptions(repetitionPenalty, noRepeatNgramSize);
            var window = new List<int>(promptTokens);
            for (int n = 0; n < maxNewTokens; n++)
            {
                if (window.Count > model.Config.ContextLength)
                    window.RemoveRange(0, window.Count - model.Config.ContextLength);
                Tensor logits = model.ForwardLast(window);
                ReadOnlySpan<float> samplingLogits = logits.Data;
                float[]? adjusted = null;
                if (repetitionPenalty != 1f || noRepeatNgramSize > 0)
                {
                    adjusted = (float[])logits.Data.Clone();
                    ApplyRepetitionControls(adjusted, window, repetitionPenalty, noRepeatNgramSize);
                    samplingLogits = adjusted;
                }
                int next = Sample(samplingLogits, temperature, topK, rng);
                yield return next;
                window.Add(next);
                if (eosId is int eos && next == eos) yield break;
            }
        }

        internal static void ApplyRepetitionControls(Span<float> logits, IReadOnlyList<int> history,
            float repetitionPenalty, int noRepeatNgramSize)
        {
            ValidateRepetitionOptions(repetitionPenalty, noRepeatNgramSize);
            ArgumentNullException.ThrowIfNull(history);

            if (repetitionPenalty != 1f)
            {
                var seen = new HashSet<int>();
                foreach (int token in history)
                {
                    if ((uint)token >= (uint)logits.Length || !seen.Add(token)) continue;
                    logits[token] = logits[token] < 0f
                        ? logits[token] * repetitionPenalty
                        : logits[token] / repetitionPenalty;
                }
            }

            if (noRepeatNgramSize <= 0 || history.Count < noRepeatNgramSize - 1) return;
            float[] beforeBan = logits.ToArray();
            if (noRepeatNgramSize == 1)
            {
                foreach (int token in history)
                    if ((uint)token < (uint)logits.Length) logits[token] = float.NegativeInfinity;
            }
            else
            {
                int prefixLength = noRepeatNgramSize - 1;
                int suffixStart = history.Count - prefixLength;
                for (int start = 0; start + noRepeatNgramSize <= history.Count; start++)
                {
                    bool matches = true;
                    for (int j = 0; j < prefixLength; j++)
                    {
                        if (history[start + j] == history[suffixStart + j]) continue;
                        matches = false;
                        break;
                    }
                    if (!matches) continue;
                    int banned = history[start + prefixLength];
                    if ((uint)banned < (uint)logits.Length)
                        logits[banned] = float.NegativeInfinity;
                }
            }

            bool anyAvailable = false;
            for (int i = 0; i < logits.Length; i++)
            {
                if (float.IsNegativeInfinity(logits[i])) continue;
                anyAvailable = true;
                break;
            }
            if (!anyAvailable)
                beforeBan.CopyTo(logits);
        }

        private static void ValidateRepetitionOptions(float repetitionPenalty, int noRepeatNgramSize)
        {
            if (!float.IsFinite(repetitionPenalty) || repetitionPenalty < 1f)
                throw new ArgumentOutOfRangeException(nameof(repetitionPenalty),
                    "Repetition penalty must be finite and >= 1.");
            if (noRepeatNgramSize < 0)
                throw new ArgumentOutOfRangeException(nameof(noRepeatNgramSize),
                    "No-repeat n-gram size must be >= 0.");
        }
    }
}
