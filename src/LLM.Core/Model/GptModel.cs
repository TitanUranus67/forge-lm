
namespace LLM.Core.Model
{
    using LLM.Core.Tensor;

    /// <summary>
    /// GPT language model (GPT-2 architecture, pre-LN, learned positional embeddings,
    /// tied token-embedding/output weights). Training runs batched: B sequences of length T are stacked
    /// row-wise into [B*T, C] tensors (sequence b occupies rows b*T..(b+1)*T) and
    /// processed in one pass — attention never crosses sequence boundaries. Inference
    /// uses the single-sequence path (B = 1); there is no KV cache. All parameter
    /// gradients accumulate into the <see cref="Parameters"/> registry; the caller zeroes them.
    ///
    /// Parameter names:
    ///   tok_emb, pos_emb,
    ///   blocks.{i}.ln1.{w,b}, blocks.{i}.attn.qkv.{w,b}, blocks.{i}.attn.proj.{w,b},
    ///   blocks.{i}.ln2.{w,b}, blocks.{i}.mlp.fc.{w,b}, blocks.{i}.mlp.proj.{w,b},
    ///   ln_f.{w,b}
    /// </summary>
    public sealed class GptModel
    {
        private const int IgnoreIndex = -1;

        private readonly ITensorBackend _b;
        private readonly int[] _positions; // [0..ContextLength) for the positional embedding
        private readonly Embedding _tokEmb, _posEmb;
        private readonly TransformerBlock[] _blocks;
        private readonly LayerNorm _lnF;
        private readonly TiedOutputProjection _head;

        // caches for Backward
        private int[]? _lastTokens;

        // cached positional indices for the batched path: [0..T) repeated B times
        private int[]? _batchPositions;
        private int _batchPosT = -1, _batchPosBatch = -1;

        public GptModel(ModelConfig config, ITensorBackend backend, Random rng)
        {
            Config = config;
            _b = backend;
            var p = new Parameters(backend);
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

            // final layernorm + output projection tied to the token embedding table
            p.Add("ln_f.w", d).Fill(1f);
            p.Add("ln_f.b", d);
            _lnF = new LayerNorm(backend, p.Weight("ln_f.w"), p.Grad("ln_f.w"), p.Weight("ln_f.b"), p.Grad("ln_f.b"));
            _head = new TiedOutputProjection(backend, p.Weight("tok_emb"), p.Grad("tok_emb"));

            _positions = new int[config.ContextLength];
            for (int i = 0; i < _positions.Length; i++) _positions[i] = i;

            Params = p;
        }

        /// <summary>Named parameter registry (weights + gradients), in registration order.</summary>
        public Parameters Params { get; }

        /// <summary>The tensor backend this model runs on.</summary>
        public ITensorBackend Backend => _b;

        /// <summary>Model hyperparameters.</summary>
        public ModelConfig Config { get; }

        /// <summary>
        /// Forward pass over one sequence: returns logits [T, Vocab] and caches all
        /// intermediates needed by a subsequent backward pass.
        /// </summary>
        public Tensor Forward(IReadOnlyList<int> tokens)
        {
            int[] toks = ValidateTokens(tokens);
            Tensor logits = ForwardCore(toks, batch: 1);
            _b.EnsureHostCurrent(logits); // callers read logits.Data directly
            return logits;
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
            return ForwardBackwardCore(ValidateTokens(inputs), ValidateTargets(targets), batch: 1);
        }

        /// <summary>
        /// Full training step on a batch of <paramref name="batch"/> equal-length
        /// sequences, flattened sequence-major: sequence b occupies rows
        /// b*T..(b+1)*T of <paramref name="inputs"/>/<paramref name="targets"/>
        /// (T = inputs.Length / batch). One forward/backward over [B*T, C] tensors;
        /// the loss and gradients are the mean over all B*T positions (which, all
        /// sequences having equal length, equals the mean of per-sequence means).
        /// Gradients are NOT zeroed here — call <see cref="Parameters.ZeroGrads"/> first.
        /// Returns the loss.
        /// </summary>
        public float ForwardBackward(int[] inputs, int[] targets, int batch)
        {
            ValidateBatch(inputs, targets, batch);
            return ForwardBackwardCore(ValidateTokenIds(inputs), ValidateTargets(targets), batch);
        }

        /// <summary>
        /// Forward-only mean cross-entropy for a flattened batch. This follows the
        /// same numerical path as <see cref="ForwardBackward(int[],int[],int)"/> but
        /// does not calculate or modify parameter gradients.
        /// </summary>
        public float EvaluateLoss(int[] inputs, int[] targets, int batch)
        {
            ValidateBatch(inputs, targets, batch);
            int[] validInputs = ValidateTokenIds(inputs);
            int[] validTargets = ValidateTargets(targets);
            Tensor logits = ForwardCore(validInputs, batch);
            int rows = logits.Shape[0], v = Config.VocabSize;
            return _b.CrossEntropyForward(logits, validTargets, logits, rows, v, IgnoreIndex);
        }

        private void ValidateBatch(int[] inputs, int[] targets, int batch)
        {
            if (batch < 1)
                throw new ArgumentOutOfRangeException(nameof(batch), "batch must be >= 1.");
            if (inputs.Length != targets.Length)
                throw new ArgumentException($"inputs ({inputs.Length}) and targets ({targets.Length}) must have equal length.");
            if (inputs.Length % batch != 0)
                throw new ArgumentException($"inputs length {inputs.Length} is not a multiple of batch {batch}.");
            int t = inputs.Length / batch;
            if (t == 0)
                throw new ArgumentException("Need at least one token per sequence.");
            if (t > Config.ContextLength)
                throw new ArgumentException($"Sequence length {t} exceeds ContextLength {Config.ContextLength}.");
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

        /// <summary>Batched forward: tokens [B*T] (sequence-major) -&gt; logits [B*T, Vocab].</summary>
        private Tensor ForwardCore(int[] tokens, int batch)
        {
            _lastTokens = tokens;
            int t = tokens.Length / batch;

            Tensor x = _tokEmb.Forward(tokens);
            _b.AddInPlace(x, _posEmb.Forward(BatchPositions(batch, t)));
            foreach (TransformerBlock block in _blocks)
                x = block.Forward(x, batch);
            return _head.Forward(_lnF.Forward(x)); // logits; NOT cached — training consumes them immediately
        }

        /// <summary>Batched training step: forward, mean cross-entropy over all B*T positions, backward.</summary>
        private float ForwardBackwardCore(int[] inputs, int[] targets, int batch)
        {
            Tensor logits = ForwardCore(inputs, batch);
            int rows = logits.Shape[0], v = Config.VocabSize;

            // Alias everything onto the logits buffer: it is dead after the CE forward
            // (backward uses the LN-cached x, not y), and at 16k vocab each [rows,v]
            // tensor is ~266 MB — two extra copies would blow the 8 GB VRAM budget.
            // Both CE kernels are row read-then-write and elementwise respectively,
            // so in-place aliasing is safe.
            Tensor probs = logits;
            float loss = _b.CrossEntropyForward(logits, targets, probs, rows, v, IgnoreIndex);
            Tensor dLogits = logits;
            _b.CrossEntropyBackward(probs, targets, dLogits, rows, v, IgnoreIndex);
            Backward(dLogits);
            return loss;
        }

        /// <summary>Positional indices [0..T) repeated B times; rebuilt only when (B, T) changes.</summary>
        private int[] BatchPositions(int batch, int t)
        {
            if (_batchPositions is null || _batchPosT != t || _batchPosBatch != batch)
            {
                var pos = new int[batch * t];
                for (int b = 0; b < batch; b++)
                    Array.Copy(_positions, 0, pos, b * t, t);
                _batchPositions = pos;
                _batchPosT = t;
                _batchPosBatch = batch;
            }
            return _batchPositions;
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
            return ValidateTokenIds(tokens);
        }

        /// <summary>Copies a flattened (possibly batched) token array, range-checking every id.</summary>
        private int[] ValidateTokenIds(IReadOnlyList<int> tokens)
        {
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
