namespace LLM.Core.Training
{
    using LLM.Core.Model;
    using LLM.Core.Tensor;

    /// <summary>
    /// AdamW optimizer (Loshchilov &amp; Hutter): adaptive moments with decoupled
    /// weight decay. First/second moment buffers are allocated lazily per parameter
    /// tensor on the first <see cref="Step"/> call. Following nanoGPT, weight decay
    /// is skipped for 1-D tensors (biases and LayerNorm gains).
    /// </summary>
    public sealed class AdamW
    {
        private readonly Dictionary<string, (float[] M, float[] V)> _state = new();
        private int _t;

        /// <summary>Number of <see cref="Step"/> calls so far (drives bias correction).</summary>
        public int StepCount => _t;

        /// <summary>
        /// One AdamW update over every parameter in <paramref name="p"/> using the
        /// gradients currently stored in the registry:
        ///   m = beta1*m + (1-beta1)*g,  v = beta2*v + (1-beta2)*g^2  (bias-corrected),
        ///   theta -= lr * wd * theta   (2-D tensors only, applied decoupled),
        ///   theta -= lr * mHat / (sqrt(vHat) + eps).
        /// </summary>
        public void Step(Parameters p, float lr, float beta1 = 0.9f, float beta2 = 0.95f,
            float eps = 1e-8f, float weightDecay = 0.1f)
        {
            _t++;
            float bc1 = 1f - MathF.Pow(beta1, _t);
            float bc2 = 1f - MathF.Pow(beta2, _t);

            foreach (string name in p.Names)
            {
                Tensor w = p.Weight(name);
                Tensor g = p.Grad(name);
                if (!_state.TryGetValue(name, out var st))
                {
                    st = (new float[w.Length], new float[w.Length]);
                    _state.Add(name, st);
                }
                else if (st.M.Length != w.Length)
                {
                    throw new InvalidOperationException($"Parameter '{name}' changed shape between steps.");
                }

                bool decay = weightDecay != 0f && w.Rank > 1;
                float[] m = st.M, v = st.V;
                for (int i = 0; i < w.Length; i++)
                {
                    float gi = g.Data[i];
                    m[i] = beta1 * m[i] + (1f - beta1) * gi;
                    v[i] = beta2 * v[i] + (1f - beta2) * gi * gi;
                    if (decay) w.Data[i] -= lr * weightDecay * w.Data[i];
                    float mHat = m[i] / bc1;
                    float vHat = v[i] / bc2;
                    w.Data[i] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
                }
            }
        }
    }
}
