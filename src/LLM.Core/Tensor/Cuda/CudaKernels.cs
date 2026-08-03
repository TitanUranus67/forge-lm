using ILGPU;

namespace LLM.Core.Tensor.Cuda;

/// <summary>ILGPU kernels used by <see cref="CudaBackend"/>.</summary>
public static class CudaKernels
{
    public readonly struct UnaryArgs
    {
        public readonly ArrayView<float> X;
        public readonly int Length;
        public UnaryArgs(ArrayView<float> x, int length) { X = x; Length = length; }
    }

    public readonly struct BinaryArgs
    {
        public readonly ArrayView<float> Source;
        public readonly ArrayView<float> Destination;
        public readonly int Length;
        public BinaryArgs(ArrayView<float> source, ArrayView<float> destination, int length)
        { Source = source; Destination = destination; Length = length; }
    }

    public readonly struct CopyBlockArgs
    {
        public readonly ArrayView<float> Source, Destination;
        public readonly int SourceRow, SourceCol, DestinationRow, DestinationCol;
        public readonly int SourceCols, DestinationCols, Cols, Length;
        public CopyBlockArgs(ArrayView<float> source, ArrayView<float> destination,
            int sourceRow, int sourceCol, int destinationRow, int destinationCol,
            int sourceCols, int destinationCols, int cols, int length)
        {
            Source = source; Destination = destination;
            SourceRow = sourceRow; SourceCol = sourceCol;
            DestinationRow = destinationRow; DestinationCol = destinationCol;
            SourceCols = sourceCols; DestinationCols = destinationCols;
            Cols = cols; Length = length;
        }
    }

    public readonly struct ScaleArgs
    {
        public readonly ArrayView<float> X;
        public readonly float Factor;
        public readonly int Length;
        public ScaleArgs(ArrayView<float> x, float factor, int length)
        { X = x; Factor = factor; Length = length; }
    }

    public readonly struct AddBiasArgs
    {
        public readonly ArrayView<float> Y, Bias;
        public readonly int Cols, Length;
        public AddBiasArgs(ArrayView<float> y, ArrayView<float> bias, int cols, int length)
        { Y = y; Bias = bias; Cols = cols; Length = length; }
    }

    public readonly struct TransposeArgs
    {
        public readonly ArrayView<float> X, Output;
        public readonly int Rows, Cols, Length;
        public TransposeArgs(ArrayView<float> x, ArrayView<float> output, int rows, int cols)
        { X = x; Output = output; Rows = rows; Cols = cols; Length = rows * cols; }
    }

    public readonly struct GeluBackwardArgs
    {
        public readonly ArrayView<float> DOut, X, DX;
        public readonly int Length;
        public GeluBackwardArgs(ArrayView<float> dOut, ArrayView<float> x, ArrayView<float> dX, int length)
        { DOut = dOut; X = x; DX = dX; Length = length; }
    }

    public readonly struct CausalMaskArgs
    {
        public readonly ArrayView<float> Scores;
        public readonly int T, Length;
        public CausalMaskArgs(ArrayView<float> scores, int t, int length)
        { Scores = scores; T = t; Length = length; }
    }

    public readonly struct HeadPackArgs
    {
        public readonly ArrayView<float> Source, Destination;
        public readonly int MatrixCols, ColBase, T, HeadDim, Heads, Length;
        public HeadPackArgs(ArrayView<float> source, ArrayView<float> destination,
            int matrixCols, int colBase, int t, int headDim, int heads, int length)
        {
            Source = source; Destination = destination; MatrixCols = matrixCols;
            ColBase = colBase; T = t; HeadDim = headDim; Heads = heads; Length = length;
        }
    }

    public readonly struct SumRowsArgs
    {
        public readonly ArrayView<float> DY, DBias;
        public readonly int Rows, Cols;
        public SumRowsArgs(ArrayView<float> dY, ArrayView<float> dBias, int rows, int cols)
        { DY = dY; DBias = dBias; Rows = rows; Cols = cols; }
    }

    public readonly struct LayerNormForwardArgs
    {
        public readonly ArrayView<float> X, W, B, Output, Mean, Rstd;
        public readonly int Rows, Cols;
        public readonly float Eps;
        public LayerNormForwardArgs(ArrayView<float> x, ArrayView<float> w, ArrayView<float> b,
            ArrayView<float> output, ArrayView<float> mean, ArrayView<float> rstd,
            int rows, int cols, float eps)
        { X = x; W = w; B = b; Output = output; Mean = mean; Rstd = rstd; Rows = rows; Cols = cols; Eps = eps; }
    }

    public readonly struct LayerNormBackwardDxArgs
    {
        public readonly ArrayView<float> DOut, X, W, Mean, Rstd, DX;
        public readonly int Rows, Cols;
        public LayerNormBackwardDxArgs(ArrayView<float> dOut, ArrayView<float> x, ArrayView<float> w,
            ArrayView<float> mean, ArrayView<float> rstd, ArrayView<float> dX, int rows, int cols)
        { DOut = dOut; X = x; W = w; Mean = mean; Rstd = rstd; DX = dX; Rows = rows; Cols = cols; }
    }

    public readonly struct LayerNormBackwardParamsArgs
    {
        public readonly ArrayView<float> DOut, X, Mean, Rstd, DW, DB;
        public readonly int Rows, Cols;
        public LayerNormBackwardParamsArgs(ArrayView<float> dOut, ArrayView<float> x,
            ArrayView<float> mean, ArrayView<float> rstd, ArrayView<float> dW, ArrayView<float> dB,
            int rows, int cols)
        { DOut = dOut; X = x; Mean = mean; Rstd = rstd; DW = dW; DB = dB; Rows = rows; Cols = cols; }
    }

    public readonly struct SoftmaxBackwardArgs
    {
        public readonly ArrayView<float> DOut, Softmax, DX;
        public readonly int Rows, Cols;
        public SoftmaxBackwardArgs(ArrayView<float> dOut, ArrayView<float> softmax, ArrayView<float> dX, int rows, int cols)
        { DOut = dOut; Softmax = softmax; DX = dX; Rows = rows; Cols = cols; }
    }

    public readonly struct EmbeddingArgs
    {
        public readonly ArrayView<float> Source, Destination;
        public readonly ArrayView<int> Indices;
        public readonly int D, Length;
        public EmbeddingArgs(ArrayView<float> source, ArrayView<int> indices,
            ArrayView<float> destination, int d, int length)
        { Source = source; Indices = indices; Destination = destination; D = d; Length = length; }
    }

    public readonly struct CrossEntropyForwardArgs
    {
        public readonly ArrayView<float> Logits, Probs, Nll;
        public readonly ArrayView<int> Targets;
        public readonly int T, V, IgnoreIndex, AccumulateLoss;
        public CrossEntropyForwardArgs(ArrayView<float> logits, ArrayView<int> targets,
            ArrayView<float> probs, ArrayView<float> nll, int t, int v, int ignoreIndex, int accumulateLoss)
        { Logits = logits; Targets = targets; Probs = probs; Nll = nll; T = t; V = v; IgnoreIndex = ignoreIndex; AccumulateLoss = accumulateLoss; }
    }

    public readonly struct CrossEntropyBackwardArgs
    {
        public readonly ArrayView<float> Probs, DLogits;
        public readonly ArrayView<int> Targets;
        public readonly int V, IgnoreIndex, Length;
        public readonly float Scale;
        public CrossEntropyBackwardArgs(ArrayView<float> probs, ArrayView<int> targets,
            ArrayView<float> dLogits, int v, int ignoreIndex, float scale, int length)
        { Probs = probs; Targets = targets; DLogits = dLogits; V = v; IgnoreIndex = ignoreIndex; Scale = scale; Length = length; }
    }

    public readonly struct MatMulArgs
    {
        public readonly ArrayView<float> A, B, Y;
        public readonly int M, K, N, Slots, Mode, Accumulate;
        public MatMulArgs(ArrayView<float> a, ArrayView<float> b, ArrayView<float> y,
            int m, int k, int n, int slots, int mode, bool accumulate)
        { A = a; B = b; Y = y; M = m; K = k; N = n; Slots = slots; Mode = mode; Accumulate = accumulate ? 1 : 0; }
    }

    public readonly struct SumSquaresArgs
    {
        public readonly ArrayView<float> X, Result;
        public readonly int Length;
        public SumSquaresArgs(ArrayView<float> x, ArrayView<float> result, int length)
        { X = x; Result = result; Length = length; }
    }

    public readonly struct SumSquaresPartialsArgs
    {
        public readonly ArrayView<float> X, Partials;
        public readonly int Length, PartialOffset, NumGroups;
        public SumSquaresPartialsArgs(ArrayView<float> x, ArrayView<float> partials,
            int length, int partialOffset, int numGroups)
        { X = x; Partials = partials; Length = length; PartialOffset = partialOffset; NumGroups = numGroups; }
    }

    public readonly struct ReduceSumArgs
    {
        public readonly ArrayView<float> Input, Result;
        public readonly int Length;
        public ReduceSumArgs(ArrayView<float> input, ArrayView<float> result, int length)
        { Input = input; Result = result; Length = length; }
    }

    public readonly struct AdamWArgs
    {
        public readonly ArrayView<float> W, G, M, V;
        public readonly float Lr, LrWd, Beta1, OneBeta1, Beta2, OneBeta2, Bc1, Bc2, Eps;
        public readonly int Decay, Length;
        public AdamWArgs(ArrayView<float> w, ArrayView<float> g, ArrayView<float> m, ArrayView<float> v,
            float lr, float lrWd, float beta1, float oneBeta1, float beta2, float oneBeta2,
            float bc1, float bc2, float eps, bool decay, int length)
        {
            W = w; G = g; M = m; V = v; Lr = lr; LrWd = lrWd;
            Beta1 = beta1; OneBeta1 = oneBeta1; Beta2 = beta2; OneBeta2 = oneBeta2;
            Bc1 = bc1; Bc2 = bc2; Eps = eps; Decay = decay ? 1 : 0; Length = length;
        }
    }

    public static void AddInPlace(Index1D index, BinaryArgs a)
    { int i = index; if (i < a.Length) a.Destination[i] += a.Source[i]; }

    public static void Copy(Index1D index, BinaryArgs a)
    { int i = index; if (i < a.Length) a.Destination[i] = a.Source[i]; }

    public static void CopyBlock(Index1D index, CopyBlockArgs a)
    {
        int i = index;
        if (i >= a.Length) return;
        int r = i / a.Cols, c = i % a.Cols;
        a.Destination[(a.DestinationRow + r) * a.DestinationCols + a.DestinationCol + c] =
            a.Source[(a.SourceRow + r) * a.SourceCols + a.SourceCol + c];
    }

    public static void Scale(Index1D index, ScaleArgs a)
    { int i = index; if (i < a.Length) a.X[i] *= a.Factor; }

    public static void Fill(Index1D index, ScaleArgs a)
    { int i = index; if (i < a.Length) a.X[i] = a.Factor; }

    public static void AddBias(Index1D index, AddBiasArgs a)
    { int i = index; if (i < a.Length) a.Y[i] += a.Bias[i % a.Cols]; }

    public static void Transpose(Index1D index, TransposeArgs a)
    { int i = index; if (i < a.Length) a.Output[(i % a.Cols) * a.Rows + i / a.Cols] = a.X[i]; }

    public static void GeluForward(Index1D index, BinaryArgs a)
    {
        int i = index;
        if (i >= a.Length) return;
        float x = a.Source[i];
        float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        a.Destination[i] = 0.5f * x * (1f + MathF.Tanh(u));
    }

    public static void GeluBackward(Index1D index, GeluBackwardArgs a)
    {
        int i = index;
        if (i >= a.Length) return;
        float x = a.X[i];
        float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        float t = MathF.Tanh(u);
        float du = 0.7978845608028654f * (1f + 3f * 0.044715f * x * x);
        a.DX[i] = a.DOut[i] * 0.5f * (1f + t + x * (1f - t * t) * du);
    }

    public static void CausalMask(Index1D index, CausalMaskArgs a)
    {
        int i = index;
        if (i < a.Length && i % a.T > (i / a.T) % a.T)
            a.Scores[i] = float.NegativeInfinity;
    }

    public static void PackHeads(Index1D index, HeadPackArgs a)
    {
        int i = index;
        if (i >= a.Length) return;
        int slot = i / (a.T * a.HeadDim), r = i % (a.T * a.HeadDim);
        int row = r / a.HeadDim, d = r % a.HeadDim;
        int seq = slot / a.Heads, head = slot % a.Heads;
        a.Destination[i] = a.Source[(seq * a.T + row) * a.MatrixCols + a.ColBase + head * a.HeadDim + d];
    }

    public static void UnpackHeads(Index1D index, HeadPackArgs a)
    {
        int i = index;
        if (i >= a.Length) return;
        int slot = i / (a.T * a.HeadDim), r = i % (a.T * a.HeadDim);
        int row = r / a.HeadDim, d = r % a.HeadDim;
        int seq = slot / a.Heads, head = slot % a.Heads;
        a.Destination[(seq * a.T + row) * a.MatrixCols + a.ColBase + head * a.HeadDim + d] = a.Source[i];
    }

    public static void SumRows(Index1D index, SumRowsArgs a)
    {
        int c = index;
        if (c >= a.Cols) return;
        float sum = 0f;
        for (int r = 0; r < a.Rows; r++) sum += a.DY[r * a.Cols + c];
        a.DBias[c] += sum;
    }

    public static void LayerNormForward(Index1D index, LayerNormForwardArgs a)
    {
        int r = index;
        if (r >= a.Rows) return;
        int o = r * a.Cols;
        float sum = 0f;
        for (int c = 0; c < a.Cols; c++) sum += a.X[o + c];
        float mean = sum / a.Cols;
        float variance = 0f;
        for (int c = 0; c < a.Cols; c++)
        { float d = a.X[o + c] - mean; variance += d * d; }
        float rstd = 1f / MathF.Sqrt(variance / a.Cols + a.Eps);
        a.Mean[r] = mean; a.Rstd[r] = rstd;
        for (int c = 0; c < a.Cols; c++)
            a.Output[o + c] = (a.X[o + c] - mean) * rstd * a.W[c] + a.B[c];
    }

    public static void LayerNormBackwardDx(Index1D index, LayerNormBackwardDxArgs a)
    {
        int r = index;
        if (r >= a.Rows) return;
        float mean = a.Mean[r], rstd = a.Rstd[r];
        int o = r * a.Cols;
        float sumDxhat = 0f, sumDxhatXhat = 0f;
        for (int c = 0; c < a.Cols; c++)
        {
            float xhat = (a.X[o + c] - mean) * rstd;
            float dxhat = a.DOut[o + c] * a.W[c];
            a.DX[o + c] = dxhat;
            sumDxhat += dxhat; sumDxhatXhat += dxhat * xhat;
        }
        float m1 = sumDxhat / a.Cols, m2 = sumDxhatXhat / a.Cols;
        for (int c = 0; c < a.Cols; c++)
        {
            float xhat = (a.X[o + c] - mean) * rstd;
            a.DX[o + c] = rstd * (a.DX[o + c] - m1 - xhat * m2);
        }
    }

    public static void LayerNormBackwardParams(Index1D index, LayerNormBackwardParamsArgs a)
    {
        int c = index;
        if (c >= a.Cols) return;
        float dw = 0f, db = 0f;
        for (int r = 0; r < a.Rows; r++)
        {
            int i = r * a.Cols + c;
            float xhat = (a.X[i] - a.Mean[r]) * a.Rstd[r];
            dw += a.DOut[i] * xhat; db += a.DOut[i];
        }
        a.DW[c] += dw; a.DB[c] += db;
    }

    public static void SoftmaxForward(UnaryArgs a)
    {
        int row = Grid.IdxX, lane = Group.IdxX;
        ArrayView<float> reduction = SharedMemory.Allocate<float>(256);
        int o = row * a.Length;
        float max = float.NegativeInfinity;
        for (int c = lane; c < a.Length; c += 256) max = MathF.Max(max, a.X[o + c]);
        reduction[lane] = max;
        Group.Barrier();
        for (int s = 128; s > 0; s >>= 1)
        { if (lane < s) reduction[lane] = MathF.Max(reduction[lane], reduction[lane + s]); Group.Barrier(); }
        max = reduction[0];
        float sum = 0f;
        for (int c = lane; c < a.Length; c += 256)
        { float e = MathF.Exp(a.X[o + c] - max); a.X[o + c] = e; sum += e; }
        reduction[lane] = sum;
        Group.Barrier();
        for (int s = 128; s > 0; s >>= 1)
        { if (lane < s) reduction[lane] += reduction[lane + s]; Group.Barrier(); }
        float inv = 1f / reduction[0];
        for (int c = lane; c < a.Length; c += 256) a.X[o + c] *= inv;
    }

    public static void SoftmaxBackward(SoftmaxBackwardArgs a)
    {
        int row = Grid.IdxX, lane = Group.IdxX;
        ArrayView<float> reduction = SharedMemory.Allocate<float>(256);
        int o = row * a.Cols;
        float dot = 0f;
        for (int c = lane; c < a.Cols; c += 256) dot += a.DOut[o + c] * a.Softmax[o + c];
        reduction[lane] = dot;
        Group.Barrier();
        for (int s = 128; s > 0; s >>= 1)
        { if (lane < s) reduction[lane] += reduction[lane + s]; Group.Barrier(); }
        dot = reduction[0];
        for (int c = lane; c < a.Cols; c += 256)
            a.DX[o + c] = a.Softmax[o + c] * (a.DOut[o + c] - dot);
    }

    public static void EmbeddingForward(Index1D index, EmbeddingArgs a)
    {
        int i = index;
        if (i >= a.Length) return;
        int token = i / a.D, col = i % a.D;
        a.Destination[i] = a.Source[a.Indices[token] * a.D + col];
    }

    public static void EmbeddingBackward(Index1D index, EmbeddingArgs a)
    {
        int i = index;
        if (i >= a.Length) return;
        int token = i / a.D, col = i % a.D;
        Atomic.Add(ref a.Destination[a.Indices[token] * a.D + col], a.Source[i]);
    }

    public static void CrossEntropyForward(Index1D index, CrossEntropyForwardArgs a)
    {
        int t = index;
        if (t >= a.T) return;
        int o = t * a.V, target = a.Targets[t];
        float targetLogit = target == a.IgnoreIndex ? 0f : a.Logits[o + target];
        float max = a.Logits[o];
        for (int c = 1; c < a.V; c++) max = MathF.Max(max, a.Logits[o + c]);
        float sum = 0f;
        for (int c = 0; c < a.V; c++)
        { float e = MathF.Exp(a.Logits[o + c] - max); a.Probs[o + c] = e; sum += e; }
        float inv = 1f / sum;
        for (int c = 0; c < a.V; c++) a.Probs[o + c] *= inv;
        float nll = target == a.IgnoreIndex ? 0f : MathF.Log(sum) + max - targetLogit;
        if (a.AccumulateLoss != 0) Atomic.Add(ref a.Nll[0], nll);
        else a.Nll[t] = nll;
    }

    public static void CrossEntropyBackward(Index1D index, CrossEntropyBackwardArgs a)
    {
        int i = index;
        if (i >= a.Length) return;
        int target = a.Targets[i / a.V];
        a.DLogits[i] = target == a.IgnoreIndex ? 0f :
            a.Probs[i] * a.Scale - (i % a.V == target ? a.Scale : 0f);
    }

    public static void MatMul(MatMulArgs a)
    {
        int tx = Group.IdxX, ty = Group.IdxY;
        int col = Grid.GlobalIndex.X, row = Grid.GlobalIndex.Y, slot = Grid.IdxZ;
        ArrayView<float> tileA = SharedMemory.Allocate<float>(256);
        ArrayView<float> tileB = SharedMemory.Allocate<float>(256);
        int tile = ty * 16 + tx;
        float sum = 0f;
        int blockRow = Grid.IdxY * 16;

        for (int k0 = 0; k0 < a.K; k0 += 16)
        {
            if (a.Mode == 2)
                tileA[tile] = k0 + ty < a.K && blockRow + tx < a.M
                    ? a.A[(slot * a.K + k0 + ty) * a.M + blockRow + tx] : 0f;
            else
                tileA[tile] = row < a.M && k0 + tx < a.K
                    ? a.A[(slot * a.M + row) * a.K + k0 + tx] : 0f;

            if (a.Mode == 1)
                tileB[tile] = k0 + ty < a.K && col < a.N
                    ? a.B[(slot * a.N + col) * a.K + k0 + ty] : 0f;
            else
                tileB[tile] = k0 + ty < a.K && col < a.N
                    ? a.B[(slot * a.K + k0 + ty) * a.N + col] : 0f;

            Group.Barrier();
            for (int kk = 0; kk < 16; kk++)
                sum += a.Mode == 2
                    ? tileA[kk * 16 + ty] * tileB[kk * 16 + tx]
                    : tileA[ty * 16 + kk] * tileB[kk * 16 + tx];
            Group.Barrier();
        }

        if (row >= a.M || col >= a.N) return;
        int output = (slot * a.M + row) * a.N + col;
        if (a.Accumulate != 0) a.Y[output] += sum;
        else a.Y[output] = sum;
    }

    public static void SumSquares(SumSquaresArgs a)
    {
        int lane = Group.IdxX;
        ArrayView<float> reduction = SharedMemory.Allocate<float>(256);
        float sum = 0f;
        for (int i = lane; i < a.Length; i += 256)
        { float x = a.X[i]; sum += x * x; }
        reduction[lane] = sum;
        Group.Barrier();
        for (int s = 128; s > 0; s >>= 1)
        { if (lane < s) reduction[lane] += reduction[lane + s]; Group.Barrier(); }
        if (lane == 0) a.Result[0] = reduction[0];
    }

    public static void SumSquaresPartials(SumSquaresPartialsArgs a)
    {
        int group = Grid.IdxX, lane = Group.IdxX;
        ArrayView<float> reduction = SharedMemory.Allocate<float>(256);
        float sum = 0f;
        for (int i = group * 256 + lane; i < a.Length; i += a.NumGroups * 256)
        { float x = a.X[i]; sum += x * x; }
        reduction[lane] = sum;
        Group.Barrier();
        for (int s = 128; s > 0; s >>= 1)
        { if (lane < s) reduction[lane] += reduction[lane + s]; Group.Barrier(); }
        if (lane == 0) a.Partials[a.PartialOffset + group] = reduction[0];
    }

    public static void ReduceSum(ReduceSumArgs a)
    {
        int lane = Group.IdxX;
        ArrayView<float> reduction = SharedMemory.Allocate<float>(256);
        float sum = 0f;
        for (int i = lane; i < a.Length; i += 256) sum += a.Input[i];
        reduction[lane] = sum;
        Group.Barrier();
        for (int s = 128; s > 0; s >>= 1)
        { if (lane < s) reduction[lane] += reduction[lane + s]; Group.Barrier(); }
        if (lane == 0) a.Result[0] = reduction[0];
    }

    public static void AdamW(Index1D index, AdamWArgs a)
    {
        int i = index;
        if (i >= a.Length) return;
        float g = a.G[i];
        float m = a.Beta1 * a.M[i] + a.OneBeta1 * g;
        float v = a.Beta2 * a.V[i] + a.OneBeta2 * g * g;
        a.M[i] = m; a.V[i] = v;
        float w = a.W[i];
        if (a.Decay != 0) w -= a.LrWd * w;
        a.W[i] = w - a.Lr * (m / a.Bc1) / (MathF.Sqrt(v / a.Bc2) + a.Eps);
    }
}
