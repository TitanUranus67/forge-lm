
namespace LLM.Core.Model
{
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

        /// <summary>Forward: x [T,in] -&gt; y [T,out]. Caches x for Backward.</summary>
        public Tensor Forward(Tensor x)
        {
            _x = x;
            int t = x.Shape[0];
            var y = new Tensor(t, Out);
            _backend.MatMulNN(x.Data, _w.Data, y.Data, t, In, Out);
            _backend.AddBias(y.Data, _b.Data, t, Out);
            return y;
        }

        /// <summary>Backward: dY [T,out] -&gt; dX [T,in]; accumulates dW, dB.</summary>
        public Tensor Backward(Tensor dY)
        {
            Tensor x = _x ?? throw new InvalidOperationException("Forward must run before Backward.");
            int t = x.Shape[0];
            _backend.MatMulTN(x.Data, dY.Data, _dW.Data, In, t, Out, accumulate: true);
            _backend.SumRows(dY.Data, _dB.Data, t, Out);
            var dX = new Tensor(t, In);
            _backend.MatMulNT(dY.Data, _w.Data, dX.Data, t, Out, In);
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
            _backend.LayerNormForward(x.Data, _w.Data, _b.Data, y.Data, _mean.Data, _rstd.Data, t, C, Eps);
            return y;
        }

        /// <summary>Backward: dY [T,C] -&gt; dX [T,C]; accumulates dW, dB.</summary>
        public Tensor Backward(Tensor dY)
        {
            if (_x is null || _mean is null || _rstd is null)
                throw new InvalidOperationException("Forward must run before Backward.");
            int t = _x.Shape[0];
            var dX = new Tensor(t, C);
            _backend.LayerNormBackward(dY.Data, _x.Data, _w.Data, _mean.Data, _rstd.Data, dX.Data, _dW.Data, _dB.Data, t, C);
            return dX;
        }
    }

    /// <summary>
    /// Causal multi-head self-attention (GPT-2 style): fused QKV projection, per-head
    /// scaled dot-product attention with causal mask, head concatenation, output
    /// projection. Input/output are [T, DModel]; heads are processed with a simple loop.
    /// </summary>
    public sealed class MultiHeadAttention
    {
        private readonly ITensorBackend _b;
        private readonly int _nHeads, _headDim, _dModel;
        private readonly float _invScale;

        // per-head forward caches
        private Tensor[]? _q, _k, _v, _probs;
        private int _t;

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

        /// <summary>Forward: x [T,D] -&gt; y [T,D]. Caches per-head Q, K, V and attention probs.</summary>
        public Tensor Forward(Tensor x)
        {
            int t = x.Shape[0];
            _t = t;
            Tensor qkv = Qkv.Forward(x); // [T, 3D]

            _q = new Tensor[_nHeads];
            _k = new Tensor[_nHeads];
            _v = new Tensor[_nHeads];
            _probs = new Tensor[_nHeads];
            var concat = new Tensor(t, _dModel);

            for (int h = 0; h < _nHeads; h++)
            {
                int hd = _headDim;
                var q = new Tensor(t, hd);
                var k = new Tensor(t, hd);
                var v = new Tensor(t, hd);
                SliceHead(qkv, q, h * hd);
                SliceHead(qkv, k, _dModel + h * hd);
                SliceHead(qkv, v, 2 * _dModel + h * hd);

                var probs = new Tensor(t, t); // scores -> softmax in place
                _b.MatMulNT(q.Data, k.Data, probs.Data, t, hd, t);
                _b.Scale(probs.Data, _invScale);
                _b.CausalMask(probs.Data, t);
                _b.SoftmaxForward(probs.Data, t, t);

                var ctx = new Tensor(t, hd);
                _b.MatMulNN(probs.Data, v.Data, ctx.Data, t, t, hd);
                MergeHead(concat, ctx, h * hd);

                _q[h] = q; _k[h] = k; _v[h] = v; _probs[h] = probs;
            }

            return Proj.Forward(concat);
        }

        /// <summary>Backward: dY [T,D] -&gt; dX [T,D]; accumulates into Qkv/Proj parameter grads.</summary>
        public Tensor Backward(Tensor dY)
        {
            if (_q is null || _k is null || _v is null || _probs is null)
                throw new InvalidOperationException("Forward must run before Backward.");
            int t = _t;
            int hd = _headDim;
            Tensor dConcat = Proj.Backward(dY); // [T,D]
            var dQkv = new Tensor(t, 3 * _dModel);

            for (int h = 0; h < _nHeads; h++)
            {
                var dCtx = new Tensor(t, hd);
                SliceHead(dConcat, dCtx, h * hd);

                var dProbs = new Tensor(t, t);
                _b.MatMulNT(dCtx.Data, _v[h].Data, dProbs.Data, t, hd, t);
                var dV = new Tensor(t, hd);
                _b.MatMulTN(_probs[h].Data, dCtx.Data, dV.Data, t, t, hd);

                var dScores = new Tensor(t, t);
                _b.SoftmaxBackward(dProbs.Data, _probs[h].Data, dScores.Data, t, t);
                _b.Scale(dScores.Data, _invScale);

                var dQ = new Tensor(t, hd);
                _b.MatMulNN(dScores.Data, _k[h].Data, dQ.Data, t, t, hd);
                var dK = new Tensor(t, hd);
                _b.MatMulTN(dScores.Data, _q[h].Data, dK.Data, t, t, hd);

                MergeHead(dQkv, dQ, h * hd);
                MergeHead(dQkv, dK, _dModel + h * hd);
                MergeHead(dQkv, dV, 2 * _dModel + h * hd);
            }

            return Qkv.Backward(dQkv);
        }

        /// <summary>dst[t, 0..cols) = src[t, colOffset..colOffset+cols) — extract one head's columns.</summary>
        private static void SliceHead(Tensor src, Tensor dst, int colOffset)
        {
            int t = dst.Shape[0], cols = dst.Shape[1], srcCols = src.Shape[1];
            for (int i = 0; i < t; i++)
                Array.Copy(src.Data, i * srcCols + colOffset, dst.Data, i * cols, cols);
        }

        /// <summary>dst[t, colOffset..colOffset+cols) = src[t, 0..cols) — write one head's columns back.</summary>
        private static void MergeHead(Tensor dst, Tensor src, int colOffset)
        {
            int t = src.Shape[0], cols = src.Shape[1], dstCols = dst.Shape[1];
            for (int i = 0; i < t; i++)
                Array.Copy(src.Data, i * cols, dst.Data, i * dstCols + colOffset, cols);
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

        /// <summary>Forward: x [T,D] -&gt; y [T,D]. Caches the pre-GELU activations.</summary>
        public Tensor Forward(Tensor x)
        {
            _fcOut = Fc.Forward(x);
            int t = x.Shape[0];
            var h = new Tensor(t, Fc.Out);
            _b.GeluForward(_fcOut.Data, h.Data);
            return Proj.Forward(h);
        }

        /// <summary>Backward: dY [T,D] -&gt; dX [T,D]; accumulates into Fc/Proj parameter grads.</summary>
        public Tensor Backward(Tensor dY)
        {
            if (_fcOut is null) throw new InvalidOperationException("Forward must run before Backward.");
            Tensor dH = Proj.Backward(dY);
            var dA = new Tensor(dH.Shape[0], dH.Shape[1]);
            _b.GeluBackward(dH.Data, _fcOut.Data, dA.Data);
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
            _b.EmbeddingForward(_table.Data, _indices, output.Data, D);
            return output;
        }

        /// <summary>Backward: dOut [T,D]; accumulates into the table gradient.</summary>
        public void Backward(Tensor dOut)
        {
            if (_indices is null) throw new InvalidOperationException("Forward must run before Backward.");
            _b.EmbeddingBackward(dOut.Data, _indices, _dTable.Data, D);
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

        /// <summary>Forward: x [T,D] -&gt; y [T,D]. Caches the midpoint residual x2.</summary>
        public Tensor Forward(Tensor x)
        {
            Tensor a = Attn.Forward(Ln1.Forward(x));
            _x2 = x.Clone();
            _b.AddInPlace(_x2.Data, a.Data);
            Tensor m = Mlp.Forward(Ln2.Forward(_x2));
            var y = _x2.Clone();
            _b.AddInPlace(y.Data, m.Data);
            return y;
        }

        /// <summary>Backward: dY [T,D] -&gt; dX [T,D]; accumulates all sub-block parameter grads.</summary>
        public Tensor Backward(Tensor dY)
        {
            if (_x2 is null) throw new InvalidOperationException("Forward must run before Backward.");
            // y = x2 + mlp(ln2(x2))
            var dX2 = dY.Clone();
            _b.AddInPlace(dX2.Data, Ln2.Backward(Mlp.Backward(dY)).Data);
            // x2 = x + attn(ln1(x))
            var dX = dX2.Clone();
            _b.AddInPlace(dX.Data, Ln1.Backward(Attn.Backward(dX2)).Data);
            return dX;
        }
    }
}
