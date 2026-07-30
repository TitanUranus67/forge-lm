namespace LLM.Core.Training
{
    using LLM.Core.Model;
    using LLM.Core.Tensor;

    /// <summary>
    /// AdamW optimizer (Loshchilov &amp; Hutter): adaptive moments with decoupled
    /// weight decay. First/second moment buffers are allocated lazily per parameter
    /// tensor on the first <see cref="Step"/> call. Following nanoGPT, weight decay
    /// is skipped for 1-D tensors (biases and LayerNorm gains).
    /// With a backend attached, the update itself runs through
    /// <see cref="ITensorBackend.AdamWStep"/> — fully on device for device backends,
    /// with the moment buffers kept as device-resident tensors.
    /// </summary>
    public sealed class AdamW
    {
        private readonly Dictionary<string, (Tensor M, Tensor V)> _state = new();
        private readonly ITensorBackend? _backend;
        private int _t;

        /// <summary>
        /// Creates the optimizer. <paramref name="backend"/> runs the per-parameter
        /// update (and keeps moment buffers device-resident for device backends);
        /// pass null for host-only use.
        /// </summary>
        public AdamW(ITensorBackend? backend = null) => _backend = backend;

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
                    st = (new Tensor(w.Shape), new Tensor(w.Shape));
                    if (_backend is not null)
                    {
                        _backend.Zero(st.M); // zero on device too: no zero-upload on first use
                        _backend.Zero(st.V);
                    }
                    _state.Add(name, st);
                }
                else if (st.M.Length != w.Length)
                {
                    throw new InvalidOperationException($"Parameter '{name}' changed shape between steps.");
                }

                if (_backend is not null)
                {
                    _backend.AdamWStep(w, g, st.M, st.V, lr, beta1, beta2, eps, weightDecay, _t);
                    continue;
                }

                bool decay = weightDecay != 0f && w.Rank > 1;
                float[] m = st.M.Data, v = st.V.Data;
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
