
namespace LLM.Core.Model
{
    using System.Threading.Tasks;
    using LLM.Core.Tensor;

    /// <summary>
    /// Dense layer y = x @ W + b with W [in,out] row-major, x [T,in] (one sequence,
    /// no batch dim). Backward accumulates parameter gradients and returns dX.
    /// </summary>
    public sealed class Linear
    {
        private readonly ITensorBackend _backend;
        private readonly Tensor _w, _b, _dW, _dB;
        private Tensor? _x; // cached forward input

        public Linear(ITensorBackend backend, Tensor w, Tensor dW, Tensor b, Tensor dB)
        {
            _backend = backend; _w = w; _dW = dW; _b = b; _dB = dB;
        }

        public int In => _w.Shape[0];
        public int Out => _w.Shape[1];

        /// <summary>Forward: x [T,in] -&gt; y [T,out]. Caches x for Backward unless
        /// <paramref name="cacheX"/> is false (the caller then passes x explicitly to
        /// <see cref="Backward(Tensor, Tensor)"/> — used when the input is recomputed
        /// instead of cached to save device memory).</summary>
        public Tensor Forward(Tensor x, bool cacheX = true)
        {
            _x = cacheX ? x : null;
            int t = x.Shape[0];
            var y = new Tensor(t, Out);
            _backend.MatMulNN(x, _w, y, t, In, Out);
            _backend.AddBias(y, _b, t, Out);
            return y;
        }

        /// <summary>Backward: dY [T,out] -&gt; dX [T,in]; accumulates dW, dB. Releases the cached forward input.</summary>
        public Tensor Backward(Tensor dY)
        {
            Tensor x = _x ?? throw new InvalidOperationException("Forward must run before Backward.");
            return Backward(dY, x);
        }

        /// <summary>Backward with an explicitly supplied forward input (e.g. a recomputed activation).</summary>
        public Tensor Backward(Tensor dY, Tensor x)
        {
            _x = null; // release the activation for the allocator right after its last use
            int t = x.Shape[0];
            _backend.MatMulTN(x, dY, _dW, In, t, Out, accumulate: true);
            _backend.SumRows(dY, _dB, t, Out);
            var dX = new Tensor(t, In);
            _backend.MatMulNT(dY, _w, dX, t, Out, In);
            return dX;
        }
    }

    /// <summary>
    /// Vocabulary projection that reuses the token embedding table. The shared
    /// table is stored [V,D], so forward uses x * table^T and backward adds the
    /// output-projection gradient into the same tensor used by embedding lookup.
    /// </summary>
    public sealed class TiedOutputProjection
    {
        private readonly ITensorBackend _backend;
        private readonly Tensor _table, _dTable;
        private Tensor? _x;

        public TiedOutputProjection(ITensorBackend backend, Tensor table, Tensor dTable)
        {
            _backend = backend;
            _table = table;
            _dTable = dTable;
        }

        public int VocabSize => _table.Shape[0];
        public int DModel => _table.Shape[1];

        public Tensor Forward(Tensor x)
        {
            _x = x;
            var logits = new Tensor(x.Shape[0], VocabSize);
            _backend.MatMulNT(x, _table, logits, x.Shape[0], DModel, VocabSize);
            return logits;
        }

        public Tensor Backward(Tensor dLogits)
        {
            Tensor x = _x ?? throw new InvalidOperationException("Forward must run before Backward.");
            _x = null;
            int rows = x.Shape[0];
            _backend.MatMulTN(dLogits, x, _dTable, VocabSize, rows, DModel, accumulate: true);
            var dX = new Tensor(rows, DModel);
            _backend.MatMulNN(dLogits, _table, dX, rows, VocabSize, DModel);
            return dX;
        }
    }

    /// <summary>
    /// Row-wise layer normalization y = (x - mean)/sqrt(var+eps) * w + b over rows of x [T,C].
    /// Caches x, mean and rstd for Backward.
    /// </summary>
    public sealed class LayerNorm
    {
        private const float Eps = 1e-5f;

        private readonly ITensorBackend _backend;
        private readonly Tensor _w, _b, _dW, _dB;
        private Tensor? _x, _mean, _rstd;

        public LayerNorm(ITensorBackend backend, Tensor w, Tensor dW, Tensor b, Tensor dB)
        {
            _backend = backend; _w = w; _dW = dW; _b = b; _dB = dB;
        }

        public int C => _w.Shape[0];

        /// <summary>Forward: x [T,C] -&gt; y [T,C]. Caches x, mean, rstd.</summary>
        public Tensor Forward(Tensor x)
        {
            _x = x;
            int t = x.Shape[0];
            var y = new Tensor(t, C);
            _mean = new Tensor(t);
            _rstd = new Tensor(t);
            _backend.LayerNormForward(x, _w, _b, y, _mean, _rstd, t, C, Eps);
            return y;
        }

        /// <summary>Backward: dY [T,C] -&gt; dX [T,C]; accumulates dW, dB. Releases the forward caches.</summary>
        public Tensor Backward(Tensor dY)
        {
            if (_x is null || _mean is null || _rstd is null)
                throw new InvalidOperationException("Forward must run before Backward.");
            int t = _x.Shape[0];
            var dX = new Tensor(t, C);
            _backend.LayerNormBackward(dY, _x, _w, _mean, _rstd, dX, _dW, _dB, t, C);
            _x = _mean = _rstd = null; // release activations right after their last use
            return dX;
        }
    }

    /// <summary>
    /// Causal multi-head self-attention (GPT-2 style): fused QKV projection, per-head
    /// scaled dot-product attention with causal mask, head concatenation, output
    /// projection. Input/output are [B*T, DModel]: B independent sequences of T rows
    /// stacked row-wise (B = 1 is plain single-sequence inference). Attention never
    /// crosses sequence boundaries. The (sequence, head) slots are packed into single
    /// slot-contiguous tensors ([B*H*T, ...]) and processed with the batched kernels,
    /// so one layer costs a constant number of backend calls regardless of B*H.
    /// </summary>
    public sealed class MultiHeadAttention
    {
        private readonly ITensorBackend _b;
        private readonly int _nHeads, _headDim, _dModel;
        private readonly float _invScale;

        // forward cache: the fused QKV projection output [B*T, 3D]. The packed
        // Q/K/V and attention probs are NOT cached — Backward rebuilds them from
        // this one tensor (3 packs + the scores matmul + softmax), which cuts the
        // per-layer activation cache by ~4x (one [B*T,3D] instead of three packed
        // [B*H*T,HD] plus the [B*H*T,T] probs) at ~1% extra compute.
        private Tensor? _qkv;
        private int _t, _batch = 1;

        public MultiHeadAttention(ITensorBackend backend, int nHeads, int dModel, Linear qkv, Linear proj)
        {
            _b = backend;
            _nHeads = nHeads;
            _dModel = dModel;
            _headDim = dModel / nHeads;
            _invScale = 1f / MathF.Sqrt(_headDim);
            Qkv = qkv;
            Proj = proj;
        }

        /// <summary>Fused QKV projection [DModel, 3*DModel].</summary>
        public Linear Qkv { get; }

        /// <summary>Output projection [DModel, DModel].</summary>
        public Linear Proj { get; }

        /// <summary>
        /// Forward: x [B*T,D] -&gt; y [B*T,D]; every block of T consecutive rows is an
        /// independent sequence. Caches the fused QKV projection for Backward (packed
        /// Q/K/V and probs are rebuilt from it there).
        /// </summary>
        public Tensor Forward(Tensor x, int batch = 1)
        {
            int rows = x.Shape[0];
            if (batch < 1 || rows % batch != 0)
                throw new ArgumentException($"Row count {rows} must be a multiple of batch size {batch}.");
            int t = rows / batch;
            _t = t;
            _batch = batch;
            Tensor qkv = Qkv.Forward(x); // [B*T, 3D]

            int slots = batch * _nHeads;
            int hd = _headDim;
            var q = new Tensor(slots * t, hd);
            var k = new Tensor(slots * t, hd);
            var v = new Tensor(slots * t, hd);
            _b.PackHeads(qkv, q, batch, t, _nHeads, hd, 0);
            _b.PackHeads(qkv, k, batch, t, _nHeads, hd, _dModel);
            _b.PackHeads(qkv, v, batch, t, _nHeads, hd, 2 * _dModel);

            var probs = new Tensor(slots * t, t); // scores -> softmax in place
            _b.BatchedMatMulNT(q, k, probs, slots, t, hd, t);
            _b.Scale(probs, _invScale);
            _b.CausalMask(probs, t);
            _b.SoftmaxForward(probs, slots * t, t);

            var ctx = new Tensor(slots * t, hd);
            _b.BatchedMatMulNN(probs, v, ctx, slots, t, t, hd);

            var concat = new Tensor(rows, _dModel);
            _b.UnpackHeads(ctx, concat, batch, t, _nHeads, hd, 0);

            _qkv = qkv;
            return Proj.Forward(concat);
        }

        /// <summary>Backward: dY [B*T,D] -&gt; dX [B*T,D]; accumulates into Qkv/Proj parameter grads.</summary>
        public Tensor Backward(Tensor dY)
        {
            if (_qkv is null)
                throw new InvalidOperationException("Forward must run before Backward.");
            int t = _t, batch = _batch;
            int slots = batch * _nHeads;
            int hd = _headDim;
            Tensor dConcat = Proj.Backward(dY); // [B*T,D]

            // rebuild the packed forward caches from the fused QKV projection
            var q = new Tensor(slots * t, hd);
            var k = new Tensor(slots * t, hd);
            var v = new Tensor(slots * t, hd);
            _b.PackHeads(_qkv, q, batch, t, _nHeads, hd, 0);
            _b.PackHeads(_qkv, k, batch, t, _nHeads, hd, _dModel);
            _b.PackHeads(_qkv, v, batch, t, _nHeads, hd, 2 * _dModel);
            var probs = new Tensor(slots * t, t);
            _b.BatchedMatMulNT(q, k, probs, slots, t, hd, t);
            _b.Scale(probs, _invScale);
            _b.CausalMask(probs, t);
            _b.SoftmaxForward(probs, slots * t, t);
            _qkv = null; // release the forward cache right after the rebuild

            var dCtx = new Tensor(slots * t, hd);
            _b.PackHeads(dConcat, dCtx, batch, t, _nHeads, hd, 0);

            var dProbs = new Tensor(slots * t, t);
            _b.BatchedMatMulNT(dCtx, v, dProbs, slots, t, hd, t);
            var dV = new Tensor(slots * t, hd);
            _b.BatchedMatMulTN(probs, dCtx, dV, slots, t, t, hd);

            var dScores = new Tensor(slots * t, t);
            _b.SoftmaxBackward(dProbs, probs, dScores, slots * t, t);
            _b.Scale(dScores, _invScale);

            var dQ = new Tensor(slots * t, hd);
            _b.BatchedMatMulNN(dScores, k, dQ, slots, t, t, hd);
            var dK = new Tensor(slots * t, hd);
            _b.BatchedMatMulTN(dScores, q, dK, slots, t, t, hd);

            var dQkv = new Tensor(dY.Shape[0], 3 * _dModel);
            _b.Zero(dQkv); // the three unpacks below fully cover it — keep the device copy authoritative
            _b.UnpackHeads(dQ, dQkv, batch, t, _nHeads, hd, 0);
            _b.UnpackHeads(dK, dQkv, batch, t, _nHeads, hd, _dModel);
            _b.UnpackHeads(dV, dQkv, batch, t, _nHeads, hd, 2 * _dModel);
            return Qkv.Backward(dQkv);
        }
    }

    /// <summary>
    /// GPT feed-forward network: Linear(D-&gt;4D) -&gt; GELU (tanh approx) -&gt; Linear(4D-&gt;D).
    /// </summary>
    public sealed class Mlp
    {
        private readonly ITensorBackend _b;
        private Tensor? _fcOut; // GELU input, cached for backward

        public Mlp(ITensorBackend backend, Linear fc, Linear proj)
        {
            _b = backend;
            Fc = fc;
            Proj = proj;
        }

        /// <summary>Expansion projection [D, 4D].</summary>
        public Linear Fc { get; }

        /// <summary>Contraction projection [4D, D].</summary>
        public Linear Proj { get; }

        /// <summary>Forward: x [T,D] -&gt; y [T,D]. Caches only the pre-GELU activations;
        /// the GELU output is recomputed from them in Backward (halves the cached MLP memory).</summary>
        public Tensor Forward(Tensor x)
        {
            _fcOut = Fc.Forward(x);
            int t = x.Shape[0];
            var h = new Tensor(t, Fc.Out);
            _b.GeluForward(_fcOut, h);
            return Proj.Forward(h, cacheX: false); // h dies here; Backward recomputes it
        }

        /// <summary>Backward: dY [T,D] -&gt; dX [T,D]; accumulates into Fc/Proj parameter grads. Releases the activation cache.</summary>
        public Tensor Backward(Tensor dY)
        {
            if (_fcOut is null) throw new InvalidOperationException("Forward must run before Backward.");
            var h = new Tensor(_fcOut.Shape[0], _fcOut.Shape[1]);
            _b.GeluForward(_fcOut, h); // recompute the GELU output Proj saw in Forward
            Tensor dH = Proj.Backward(dY, h);
            var dA = new Tensor(dH.Shape[0], dH.Shape[1]);
            _b.GeluBackward(dH, _fcOut, dA);
            _fcOut = null; // release right after its last use
            return Fc.Backward(dA);
        }
    }

    /// <summary>
    /// Token embedding lookup: out[t,:] = table[tokens[t],:]. The table is [V,D].
    /// Backward scatter-adds gradients into the table's gradient tensor.
    /// </summary>
    public sealed class Embedding
    {
        private readonly ITensorBackend _b;
        private readonly Tensor _table, _dTable;
        private int[]? _indices;

        public Embedding(ITensorBackend backend, Tensor table, Tensor dTable)
        {
            _b = backend; _table = table; _dTable = dTable;
        }

        public int D => _table.Shape[1];

        /// <summary>Forward: indices [T] -&gt; out [T,D]. Caches the indices.</summary>
        public Tensor Forward(ReadOnlySpan<int> indices)
        {
            _indices = indices.ToArray();
            var output = new Tensor(_indices.Length, D);
            _b.EmbeddingForward(_table, _indices, output, D);
            return output;
        }

        /// <summary>Backward: dOut [T,D]; accumulates into the table gradient.</summary>
        public void Backward(Tensor dOut)
        {
            if (_indices is null) throw new InvalidOperationException("Forward must run before Backward.");
            _b.EmbeddingBackward(dOut, _indices, _dTable, D);
        }
    }

    /// <summary>
    /// Pre-LN transformer block: x += Attn(LN1(x)); x += Mlp(LN2(x)).
    /// Residual adds are handled by the block; backward mirrors the forward graph.
    /// </summary>
    public sealed class TransformerBlock
    {
        private readonly ITensorBackend _b;
        private Tensor? _x2; // residual midpoint after the attention sub-block

        public TransformerBlock(ITensorBackend backend, LayerNorm ln1, MultiHeadAttention attn, LayerNorm ln2, Mlp mlp)
        {
            _b = backend;
            Ln1 = ln1; Attn = attn; Ln2 = ln2; Mlp = mlp;
        }

        public LayerNorm Ln1 { get; }
        public MultiHeadAttention Attn { get; }
        public LayerNorm Ln2 { get; }
        public Mlp Mlp { get; }

        /// <summary>Forward: x [B*T,D] -&gt; y [B*T,D] (B independent sequences of T rows). Caches the midpoint residual x2.</summary>
        public Tensor Forward(Tensor x, int batch = 1)
        {
            Tensor a = Attn.Forward(Ln1.Forward(x), batch);
            _b.AddInPlace(a, x); // a += x: the sub-block output becomes the residual midpoint
            _x2 = a;
            Tensor m = Mlp.Forward(Ln2.Forward(_x2));
            _b.AddInPlace(m, _x2); // m += x2: the MLP output becomes the block output
            return m;
        }

        /// <summary>Backward: dY [B*T,D] -&gt; dX [B*T,D]; accumulates all sub-block parameter grads. Releases the residual cache.</summary>
        public Tensor Backward(Tensor dY)
        {
            if (_x2 is null) throw new InvalidOperationException("Forward must run before Backward.");
            _x2 = null; // not needed by the backward math — release the midpoint residual early
            // y = x2 + mlp(ln2(x2))
            var dX2 = new Tensor(dY.Shape);
            _b.Copy(dY, dX2);
            _b.AddInPlace(dX2, Ln2.Backward(Mlp.Backward(dY)));
            // x2 = x + attn(ln1(x))
            var dX = new Tensor(dY.Shape);
            _b.Copy(dX2, dX);
            _b.AddInPlace(dX, Ln1.Backward(Attn.Backward(dX2)));
            return dX;
        }
    }
}
