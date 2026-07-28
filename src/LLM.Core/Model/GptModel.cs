
namespace LLM.Core.Model
{
    using LLM.Core.Tensor;

    /// <summary>
    /// GPT language model (GPT-2 architecture, pre-LN, learned positional embeddings,
    /// untied output head). Sequences are processed one at a time as [T, C] tensors —
    /// no batch dimension, no KV cache. All parameter gradients accumulate into the
    /// <see cref="Parameters"/> registry; the caller zeroes them.
    ///
    /// Parameter names:
    ///   tok_emb, pos_emb,
    ///   blocks.{i}.ln1.{w,b}, blocks.{i}.attn.qkv.{w,b}, blocks.{i}.attn.proj.{w,b},
    ///   blocks.{i}.ln2.{w,b}, blocks.{i}.mlp.fc.{w,b}, blocks.{i}.mlp.proj.{w,b},
    ///   ln_f.{w,b}, head.{w,b}
    /// </summary>
    public sealed class GptModel
    {
        private const int IgnoreIndex = -1;

        private readonly ITensorBackend _b;
        private readonly int[] _positions; // [0..ContextLength) for the positional embedding
        private readonly Embedding _tokEmb, _posEmb;
        private readonly TransformerBlock[] _blocks;
        private readonly LayerNorm _lnF;
        private readonly Linear _head;

        // caches for Backward
        private int[]? _lastTokens;
        private Tensor? _logits;

        public GptModel(ModelConfig config, ITensorBackend backend, Random rng)
        {
            Config = config;
            _b = backend;
            var p = new Parameters();
            int d = config.DModel;
            float std = 0.02f;
            float residStd = 0.02f / MathF.Sqrt(2f * config.NLayers); // GPT-2 residual-projection init

            // embeddings
            p.Add("tok_emb", config.VocabSize, d).FillNormal(rng, std);
            p.Add("pos_emb", config.ContextLength, d).FillNormal(rng, std);
            _tokEmb = new Embedding(backend, p.Weight("tok_emb"), p.Grad("tok_emb"));
            _posEmb = new Embedding(backend, p.Weight("pos_emb"), p.Grad("pos_emb"));

            // transformer blocks
            _blocks = new TransformerBlock[config.NLayers];
            for (int i = 0; i < config.NLayers; i++)
            {
                string pre = $"blocks.{i}.";
                LayerNorm MakeLn(string name)
                {
                    p.Add(pre + name + ".w", d).Fill(1f);
                    p.Add(pre + name + ".b", d); // zero
                    return new LayerNorm(backend, p.Weight(pre + name + ".w"), p.Grad(pre + name + ".w"),
                        p.Weight(pre + name + ".b"), p.Grad(pre + name + ".b"));
                }
                Linear MakeLinear(string name, int inDim, int outDim, float wStd)
                {
                    p.Add(pre + name + ".w", inDim, outDim).FillNormal(rng, wStd);
                    p.Add(pre + name + ".b", outDim); // zero
                    return new Linear(backend, p.Weight(pre + name + ".w"), p.Grad(pre + name + ".w"),
                        p.Weight(pre + name + ".b"), p.Grad(pre + name + ".b"));
                }

                var ln1 = MakeLn("ln1");
                var attn = new MultiHeadAttention(backend, config.NHeads, d,
                    MakeLinear("attn.qkv", d, 3 * d, std),
                    MakeLinear("attn.proj", d, d, residStd));
                var ln2 = MakeLn("ln2");
                var mlp = new Mlp(backend,
                    MakeLinear("mlp.fc", d, config.MlpHidden, std),
                    MakeLinear("mlp.proj", config.MlpHidden, d, residStd));
                _blocks[i] = new TransformerBlock(backend, ln1, attn, ln2, mlp);
            }

            // final layernorm + untied output head
            p.Add("ln_f.w", d).Fill(1f);
            p.Add("ln_f.b", d);
            _lnF = new LayerNorm(backend, p.Weight("ln_f.w"), p.Grad("ln_f.w"), p.Weight("ln_f.b"), p.Grad("ln_f.b"));
            p.Add("head.w", d, config.VocabSize).FillNormal(rng, std);
            p.Add("head.b", config.VocabSize);
            _head = new Linear(backend, p.Weight("head.w"), p.Grad("head.w"), p.Weight("head.b"), p.Grad("head.b"));

            _positions = new int[config.ContextLength];
            for (int i = 0; i < _positions.Length; i++) _positions[i] = i;

            Params = p;
        }

        /// <summary>Named parameter registry (weights + gradients), in registration order.</summary>
        public Parameters Params { get; }

        /// <summary>Model hyperparameters.</summary>
        public ModelConfig Config { get; }

        /// <summary>
        /// Forward pass over one sequence: returns logits [T, Vocab] and caches all
        /// intermediates needed by a subsequent backward pass.
        /// </summary>
        public Tensor Forward(IReadOnlyList<int> tokens)
        {
            int[] toks = ValidateTokens(tokens);
            _lastTokens = toks;

            Tensor x = _tokEmb.Forward(toks);
            _b.AddInPlace(x.Data, _posEmb.Forward(_positions.AsSpan(0, toks.Length)).Data);
            foreach (TransformerBlock block in _blocks)
                x = block.Forward(x);
            _logits = _head.Forward(_lnF.Forward(x));
            return _logits;
        }

        /// <summary>
        /// Full training step on one (inputs, targets) pair: forward, mean cross-entropy
        /// (ignoreIndex = -1), backward accumulating every parameter gradient.
        /// Gradients are NOT zeroed here — call <see cref="Parameters.ZeroGrads"/> first.
        /// Returns the loss.
        /// </summary>
        public float ForwardBackward(IReadOnlyList<int> inputs, IReadOnlyList<int> targets)
        {
            if (inputs.Count != targets.Count)
                throw new ArgumentException($"inputs ({inputs.Count}) and targets ({targets.Count}) must have equal length.");
            Tensor logits = Forward(inputs);
            int t = logits.Shape[0], v = Config.VocabSize;
            int[] tgt = ValidateTargets(targets);

            var probs = new Tensor(t, v);
            float loss = _b.CrossEntropyForward(logits.Data, tgt, probs.Data, t, v, IgnoreIndex);
            var dLogits = new Tensor(t, v);
            _b.CrossEntropyBackward(probs.Data, tgt, dLogits.Data, t, v, IgnoreIndex);
            Backward(dLogits);
            return loss;
        }

        /// <summary>
        /// Inference convenience: full forward, then slice out the logits of the last
        /// position as a [1, Vocab] tensor.
        /// </summary>
        public Tensor ForwardLast(IReadOnlyList<int> tokens)
        {
            Tensor logits = Forward(tokens);
            int t = logits.Shape[0], v = Config.VocabSize;
            var last = new Tensor(1, v);
            Array.Copy(logits.Data, (t - 1) * v, last.Data, 0, v);
            return last;
        }

        /// <summary>Backpropagates dLogits through head, final LN, blocks and embeddings.</summary>
        private void Backward(Tensor dLogits)
        {
            if (_lastTokens is null) throw new InvalidOperationException("Forward must run before Backward.");
            Tensor dX = _lnF.Backward(_head.Backward(dLogits));
            for (int i = _blocks.Length - 1; i >= 0; i--)
                dX = _blocks[i].Backward(dX);
            // x = tok_emb[tokens] + pos_emb[positions]: the residual splits the gradient.
            _posEmb.Backward(dX);
            _tokEmb.Backward(dX);
        }

        private int[] ValidateTokens(IReadOnlyList<int> tokens)
        {
            if (tokens.Count == 0) throw new ArgumentException("Need at least one token.");
            if (tokens.Count > Config.ContextLength)
                throw new ArgumentException($"Sequence length {tokens.Count} exceeds ContextLength {Config.ContextLength}.");
            var toks = new int[tokens.Count];
            for (int i = 0; i < toks.Length; i++)
            {
                int id = tokens[i];
                if ((uint)id >= (uint)Config.VocabSize)
                    throw new ArgumentException($"Token id {id} out of range [0,{Config.VocabSize}).");
                toks[i] = id;
            }
            return toks;
        }

        private int[] ValidateTargets(IReadOnlyList<int> targets)
        {
            var tgt = new int[targets.Count];
            for (int i = 0; i < tgt.Length; i++)
            {
                int id = targets[i];
                if (id != IgnoreIndex && (uint)id >= (uint)Config.VocabSize)
                    throw new ArgumentException($"Target id {id} out of range [0,{Config.VocabSize}) (or {IgnoreIndex} to ignore).");
                tgt[i] = id;
            }
            return tgt;
        }
    }
}
