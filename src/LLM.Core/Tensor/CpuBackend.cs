using System.Buffers;
using System.Numerics;
using System.Threading.Tasks;

namespace LLM.Core.Tensor;

/// <summary>
/// Pure-managed CPU implementation of <see cref="ITensorBackend"/>.
/// Large matmuls parallelize over output rows with <see cref="Parallel.For"/> and use
/// <see cref="Vector{T}"/> SIMD in the inner loops; small matmuls (attention scores etc.)
/// run sequentially directly on the input spans — below the parallel threshold the
/// pooled-copy + Parallel.For overhead costs more than the compute, and callers
/// (e.g. batched attention) already parallelize over independent slots at a higher level.
/// Everything else is scalar Span-based code with no allocations in hot paths.
/// </summary>
/// <remarks>
/// Spans cannot be captured by the <see cref="Parallel.For"/> lambda (ref-like
/// types in closures), so the large-matmul entry points copy their operands into
/// pooled arrays first. Rents come from <see cref="ArrayPool{T}.Shared"/>, so
/// steady-state calls do not allocate; the copy is O(MK+KN+MN) against the
/// O(M*N*K) compute it enables.
/// </remarks>
public class CpuBackend : ITensorBackend
{
    private static readonly float GeluC = MathF.Sqrt(2f / MathF.PI); // sqrt(2/pi)
    private const float GeluCoeff = 0.044715f;

    /// <summary>Matmuls with M*K*N at or below this run sequentially (no pooling, no Parallel.For).</summary>
    private const long SmallMatMulWork = 8_000_000;

    // ---- Matmul ------------------------------------------------------------

    /// <inheritdoc/>
    public void MatMulNN(Tensor a, Tensor b, Tensor y, int M, int K, int N, bool accumulate = false)
        => MatMulNN(a.AsReadOnlySpan(), b.AsReadOnlySpan(), y.AsSpan(), M, K, N, accumulate);

    /// <inheritdoc/>
    public void MatMulTN(Tensor a, Tensor b, Tensor y, int M, int K, int N, bool accumulate = false)
        => MatMulTN(a.AsReadOnlySpan(), b.AsReadOnlySpan(), y.AsSpan(), M, K, N, accumulate);

    /// <inheritdoc/>
    public void MatMulNT(Tensor a, Tensor b, Tensor y, int M, int K, int N, bool accumulate = false)
        => MatMulNT(a.AsReadOnlySpan(), b.AsReadOnlySpan(), y.AsSpan(), M, K, N, accumulate);

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void MatMulNN(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> y, int M, int K, int N, bool accumulate)
    {
        if ((long)M * K * N <= SmallMatMulWork)
        {
            MatMulRowKernelSeq(a, b, y, M, K, N, accumulate, aIsTransposed: false);
            return;
        }
        MatMulRowKernel(a, b, y, M, K, N, accumulate, aIsTransposed: false);
    }

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void MatMulTN(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> y, int M, int K, int N, bool accumulate)
    {
        if ((long)M * K * N <= SmallMatMulWork)
        {
            MatMulRowKernelSeq(a, b, y, M, K, N, accumulate, aIsTransposed: true);
            return;
        }
        MatMulRowKernel(a, b, y, M, K, N, accumulate, aIsTransposed: true);
    }

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void MatMulNT(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> y, int M, int K, int N, bool accumulate)
    {
        if ((long)M * K * N <= SmallMatMulWork)
        {
            // y[m,n] = dot(a[m,:], b[n,:]); both rows contiguous.
            for (int m = 0; m < M; m++)
            {
                ReadOnlySpan<float> aRow = a.Slice(m * K, K);
                Span<float> yRow = y.Slice(m * N, N);
                for (int n = 0; n < N; n++)
                {
                    float d = Dot(aRow, b.Slice(n * K, K));
                    if (accumulate) yRow[n] += d;
                    else yRow[n] = d;
                }
            }
            return;
        }

        // y[m,n] = dot(a[m,:], b[n,:]); both rows contiguous.
        var pool = ArrayPool<float>.Shared;
        float[] aArr = Rent(pool, a, M * K);
        float[] bArr = Rent(pool, b, N * K);
        float[] yArr = Rent(pool, y, M * N);
        try
        {
            Parallel.For(0, M, m =>
            {
                int aOff = m * K, yOff = m * N;
                for (int n = 0; n < N; n++)
                {
                    float d = Dot(aArr, aOff, bArr, n * K, K);
                    if (accumulate) yArr[yOff + n] += d;
                    else yArr[yOff + n] = d;
                }
            });
            yArr.AsSpan(0, M * N).CopyTo(y);
        }
        finally
        {
            pool.Return(aArr);
            pool.Return(bArr);
            pool.Return(yArr);
        }
    }

    /// <summary>Sequential twin of <see cref="MatMulRowKernel"/> for small matmuls: no pooling, no Parallel.For.</summary>
    private static void MatMulRowKernelSeq(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> y,
        int M, int K, int N, bool accumulate, bool aIsTransposed)
    {
        for (int m = 0; m < M; m++)
        {
            Span<float> yRow = y.Slice(m * N, N);
            if (!accumulate) yRow.Clear();
            for (int k = 0; k < K; k++)
            {
                float s = aIsTransposed ? a[k * M + m] : a[m * K + k];
                AddScaled(yRow, b.Slice(k * N, N), s);
            }
        }
    }

    /// <summary>
    /// y[m,:] (+)= sum_k a_k * b[k,:], where a_k is a[m,k] (NN) or a[k,m] (TN).
    /// Row-contiguous in both b and y, so the inner loop is a SIMD FMA over N.
    /// </summary>
    private static void MatMulRowKernel(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> y,
        int M, int K, int N, bool accumulate, bool aIsTransposed)
    {
        var pool = ArrayPool<float>.Shared;
        float[] aArr = Rent(pool, a, M * K);
        float[] bArr = Rent(pool, b, K * N);
        float[] yArr = Rent(pool, y, M * N);
        try
        {
            Parallel.For(0, M, m =>
            {
                int yOff = m * N;
                if (!accumulate) Array.Clear(yArr, yOff, N);
                for (int k = 0; k < K; k++)
                {
                    float s = aIsTransposed ? aArr[k * M + m] : aArr[m * K + k];
                    AddScaled(yArr, yOff, bArr, k * N, s, N);
                }
            });
            yArr.AsSpan(0, M * N).CopyTo(y);
        }
        finally
        {
            pool.Return(aArr);
            pool.Return(bArr);
            pool.Return(yArr);
        }
    }

    /// <summary>Rents a pooled array of at least <paramref name="length"/> and copies <paramref name="src"/> into it.</summary>
    private static float[] Rent(ArrayPool<float> pool, ReadOnlySpan<float> src, int length)
    {
        float[] arr = pool.Rent(length);
        src.CopyTo(arr);
        return arr;
    }

    /// <summary>dst[dstOff+i] += scale * src[srcOff+i] for i in [0,length), SIMD over the row.</summary>
    private static void AddScaled(float[] dst, int dstOff, float[] src, int srcOff, float scale, int length)
    {
        int n = 0, w = Vector<float>.Count;
        Vector<float> vs = new(scale);
        for (; n <= length - w; n += w)
            (new Vector<float>(dst, dstOff + n) + vs * new Vector<float>(src, srcOff + n)).CopyTo(dst, dstOff + n);
        for (; n < length; n++)
            dst[dstOff + n] += scale * src[srcOff + n];
    }

    /// <summary>dst[i] += scale * src[i], SIMD. Span twin used by the sequential kernels.</summary>
    private static void AddScaled(Span<float> dst, ReadOnlySpan<float> src, float scale)
    {
        int n = 0, w = Vector<float>.Count;
        Vector<float> vs = new(scale);
        for (; n <= dst.Length - w; n += w)
            (new Vector<float>(dst.Slice(n, w)) + vs * new Vector<float>(src.Slice(n, w))).CopyTo(dst.Slice(n, w));
        for (; n < dst.Length; n++)
            dst[n] += scale * src[n];
    }

    /// <summary>SIMD dot product of two equal-length array segments.</summary>
    private static float Dot(float[] x, int xOff, float[] y, int yOff, int length)
    {
        int i = 0, w = Vector<float>.Count;
        Vector<float> acc = Vector<float>.Zero;
        for (; i <= length - w; i += w)
            acc += new Vector<float>(x, xOff + i) * new Vector<float>(y, yOff + i);
        float sum = Vector.Sum(acc);
        for (; i < length; i++)
            sum += x[xOff + i] * y[yOff + i];
        return sum;
    }

    /// <summary>SIMD dot product of two equal-length spans. Used by the sequential kernels.</summary>
    private static float Dot(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        int i = 0, w = Vector<float>.Count;
        Vector<float> acc = Vector<float>.Zero;
        for (; i <= x.Length - w; i += w)
            acc += new Vector<float>(x.Slice(i, w)) * new Vector<float>(y.Slice(i, w));
        float sum = Vector.Sum(acc);
        for (; i < x.Length; i++)
            sum += x[i] * y[i];
        return sum;
    }

    // ---- Batched matmul (packed independent slots) --------------------------------

    /// <inheritdoc/>
    public void BatchedMatMulNN(Tensor a, Tensor b, Tensor y, int slots, int M, int K, int N, bool accumulate = false)
    {
        for (int s = 0; s < slots; s++)
            MatMulNN(a.AsReadOnlySpan().Slice(s * M * K, M * K), b.AsReadOnlySpan().Slice(s * K * N, K * N),
                y.AsSpan().Slice(s * M * N, M * N), M, K, N, accumulate);
    }

    /// <inheritdoc/>
    public void BatchedMatMulNT(Tensor a, Tensor b, Tensor y, int slots, int M, int K, int N, bool accumulate = false)
    {
        for (int s = 0; s < slots; s++)
            MatMulNT(a.AsReadOnlySpan().Slice(s * M * K, M * K), b.AsReadOnlySpan().Slice(s * N * K, N * K),
                y.AsSpan().Slice(s * M * N, M * N), M, K, N, accumulate);
    }

    /// <inheritdoc/>
    public void BatchedMatMulTN(Tensor a, Tensor b, Tensor y, int slots, int M, int K, int N, bool accumulate = false)
    {
        for (int s = 0; s < slots; s++)
            MatMulTN(a.AsReadOnlySpan().Slice(s * K * M, K * M), b.AsReadOnlySpan().Slice(s * K * N, K * N),
                y.AsSpan().Slice(s * M * N, M * N), M, K, N, accumulate);
    }

    // ---- Attention head packing -----------------------------------------------------

    /// <inheritdoc/>
    public void PackHeads(Tensor src, Tensor dst, int batch, int T, int nHeads, int headDim, int colBase)
    {
        int srcCols = src.Cols;
        for (int seq = 0; seq < batch; seq++)
            for (int h = 0; h < nHeads; h++)
            {
                int s = seq * nHeads + h;
                for (int t = 0; t < T; t++)
                    src.AsReadOnlySpan().Slice((seq * T + t) * srcCols + colBase + h * headDim, headDim)
                        .CopyTo(dst.AsSpan().Slice((s * T + t) * headDim, headDim));
            }
    }

    /// <inheritdoc/>
    public void UnpackHeads(Tensor src, Tensor dst, int batch, int T, int nHeads, int headDim, int colBase)
    {
        int dstCols = dst.Cols;
        for (int seq = 0; seq < batch; seq++)
            for (int h = 0; h < nHeads; h++)
            {
                int s = seq * nHeads + h;
                for (int t = 0; t < T; t++)
                    src.AsReadOnlySpan().Slice((s * T + t) * headDim, headDim)
                        .CopyTo(dst.AsSpan().Slice((seq * T + t) * dstCols + colBase + h * headDim, headDim));
            }
    }

    // ---- Elementwise / rows -------------------------------------------------

    /// <inheritdoc/>
    public void AddBias(Tensor y, Tensor bias, int rows, int cols)
        => AddBias(y.AsSpan(), bias.AsReadOnlySpan(), rows, cols);

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void AddBias(Span<float> y, ReadOnlySpan<float> bias, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
            AddInPlace(y.Slice(r * cols, cols), bias);
    }

    /// <inheritdoc/>
    public void SumRows(Tensor dY, Tensor dBias, int rows, int cols)
        => SumRows(dY.AsReadOnlySpan(), dBias.AsSpan(), rows, cols);

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void SumRows(ReadOnlySpan<float> dY, Span<float> dBias, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> row = dY.Slice(r * cols, cols);
            for (int c = 0; c < cols; c++)
                dBias[c] += row[c];
        }
    }

    /// <inheritdoc/>
    public void AddInPlace(Tensor dst, Tensor src)
        => AddInPlace(dst.AsSpan(), src.AsReadOnlySpan());

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void AddInPlace(Span<float> dst, ReadOnlySpan<float> src)
    {
        int i = 0, w = Vector<float>.Count;
        for (; i <= dst.Length - w; i += w)
            (new Vector<float>(dst.Slice(i, w)) + new Vector<float>(src.Slice(i, w))).CopyTo(dst.Slice(i, w));
        for (; i < dst.Length; i++)
            dst[i] += src[i];
    }

    /// <inheritdoc/>
    public void Copy(Tensor src, Tensor dst)
        => src.AsReadOnlySpan().CopyTo(dst.AsSpan());

    /// <inheritdoc/>
    public void CopyBlock(Tensor src, Tensor dst, int srcRow, int srcCol, int dstRow, int dstCol, int rows, int cols)
    {
        int srcCols = src.Cols, dstCols = dst.Cols;
        for (int r = 0; r < rows; r++)
            src.AsReadOnlySpan().Slice((srcRow + r) * srcCols + srcCol, cols)
                .CopyTo(dst.AsSpan().Slice((dstRow + r) * dstCols + dstCol, cols));
    }

    /// <inheritdoc/>
    public void Scale(Tensor x, float factor)
        => Scale(x.AsSpan(), factor);

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void Scale(Span<float> x, float factor)
    {
        int i = 0, w = Vector<float>.Count;
        Vector<float> f = new(factor);
        for (; i <= x.Length - w; i += w)
            (new Vector<float>(x.Slice(i, w)) * f).CopyTo(x.Slice(i, w));
        for (; i < x.Length; i++)
            x[i] *= factor;
    }

    /// <inheritdoc/>
    public void Transpose(Tensor x, Tensor output, int rows, int cols)
        => Transpose(x.AsReadOnlySpan(), output.AsSpan(), rows, cols);

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void Transpose(ReadOnlySpan<float> x, Span<float> output, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                output[c * rows + r] = x[r * cols + c];
    }

    // ---- LayerNorm -----------------------------------------------------------

    /// <inheritdoc/>
    public void LayerNormForward(Tensor x, Tensor w, Tensor b,
        Tensor output, Tensor mean, Tensor rstd, int rows, int cols, float eps)
        => LayerNormForward(x.AsReadOnlySpan(), w.AsReadOnlySpan(), b.AsReadOnlySpan(),
            output.AsSpan(), mean.AsSpan(), rstd.AsSpan(), rows, cols, eps);

    /// <inheritdoc/>
    public void LayerNormBackward(Tensor dOut, Tensor x, Tensor w,
        Tensor mean, Tensor rstd,
        Tensor dX, Tensor dW, Tensor dB, int rows, int cols)
        => LayerNormBackward(dOut.AsReadOnlySpan(), x.AsReadOnlySpan(), w.AsReadOnlySpan(),
            mean.AsReadOnlySpan(), rstd.AsReadOnlySpan(),
            dX.AsSpan(), dW.AsSpan(), dB.AsSpan(), rows, cols);

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void LayerNormForward(ReadOnlySpan<float> x, ReadOnlySpan<float> w, ReadOnlySpan<float> b,
        Span<float> output, Span<float> mean, Span<float> rstd, int rows, int cols, float eps)
    {
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> xr = x.Slice(r * cols, cols);
            Span<float> outR = output.Slice(r * cols, cols);

            float sum = 0f;
            for (int c = 0; c < cols; c++) sum += xr[c];
            float mu = sum / cols;

            float varSum = 0f;
            for (int c = 0; c < cols; c++) { float d = xr[c] - mu; varSum += d * d; }
            float rs = 1f / MathF.Sqrt(varSum / cols + eps); // biased variance

            mean[r] = mu;
            rstd[r] = rs;
            for (int c = 0; c < cols; c++)
                outR[c] = (xr[c] - mu) * rs * w[c] + b[c];
        }
    }

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void LayerNormBackward(ReadOnlySpan<float> dOut, ReadOnlySpan<float> x, ReadOnlySpan<float> w,
        ReadOnlySpan<float> mean, ReadOnlySpan<float> rstd,
        Span<float> dX, Span<float> dW, Span<float> dB, int rows, int cols)
    {
        float invN = 1f / cols;
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> dRow = dOut.Slice(r * cols, cols);
            ReadOnlySpan<float> xRow = x.Slice(r * cols, cols);
            Span<float> dxRow = dX.Slice(r * cols, cols);
            float mu = mean[r], rs = rstd[r];

            // dxhat = dOut * w; accumulate parameter grads.
            float sumDxhat = 0f, sumDxhatXhat = 0f;
            for (int c = 0; c < cols; c++)
            {
                float xhat = (xRow[c] - mu) * rs;
                float dxhat = dRow[c] * w[c];
                dW[c] += dRow[c] * xhat;
                dB[c] += dRow[c];
                dxRow[c] = dxhat; // stash dxhat; converted to dX in the second pass
                sumDxhat += dxhat;
                sumDxhatXhat += dxhat * xhat;
            }

            float m1 = sumDxhat * invN, m2 = sumDxhatXhat * invN;
            for (int c = 0; c < cols; c++)
            {
                float xhat = (xRow[c] - mu) * rs;
                dxRow[c] = rs * (dxRow[c] - m1 - xhat * m2);
            }
        }
    }

    // ---- Softmax --------------------------------------------------------------

    /// <inheritdoc/>
    public void SoftmaxForward(Tensor x, int rows, int cols)
        => SoftmaxForward(x.AsSpan(), rows, cols);

    /// <inheritdoc/>
    public void SoftmaxBackward(Tensor dOut, Tensor softmaxOut, Tensor dX, int rows, int cols)
        => SoftmaxBackward(dOut.AsReadOnlySpan(), softmaxOut.AsReadOnlySpan(), dX.AsSpan(), rows, cols);

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void SoftmaxForward(Span<float> x, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            Span<float> row = x.Slice(r * cols, cols);
            float max = row[0];
            for (int c = 1; c < cols; c++) max = MathF.Max(max, row[c]);
            float sum = 0f;
            for (int c = 0; c < cols; c++) { row[c] = MathF.Exp(row[c] - max); sum += row[c]; }
            float inv = 1f / sum;
            for (int c = 0; c < cols; c++) row[c] *= inv;
        }
    }

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void SoftmaxBackward(ReadOnlySpan<float> dOut, ReadOnlySpan<float> softmaxOut, Span<float> dX, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> dRow = dOut.Slice(r * cols, cols);
            ReadOnlySpan<float> sRow = softmaxOut.Slice(r * cols, cols);
            Span<float> dxRow = dX.Slice(r * cols, cols);
            float dot = 0f;
            for (int c = 0; c < cols; c++) dot += dRow[c] * sRow[c];
            for (int c = 0; c < cols; c++) dxRow[c] = sRow[c] * (dRow[c] - dot);
        }
    }

    // ---- GELU ------------------------------------------------------------------

    /// <inheritdoc/>
    public void GeluForward(Tensor x, Tensor output)
        => GeluForward(x.AsReadOnlySpan(), output.AsSpan());

    /// <inheritdoc/>
    public void GeluBackward(Tensor dOut, Tensor x, Tensor dX)
        => GeluBackward(dOut.AsReadOnlySpan(), x.AsReadOnlySpan(), dX.AsSpan());

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void GeluForward(ReadOnlySpan<float> x, Span<float> output)
    {
        for (int i = 0; i < x.Length; i++)
            output[i] = Gelu(x[i]);
    }

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void GeluBackward(ReadOnlySpan<float> dOut, ReadOnlySpan<float> x, Span<float> dX)
    {
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            float u = GeluC * (v + GeluCoeff * v * v * v);
            float t = MathF.Tanh(u);
            float du = GeluC * (1f + 3f * GeluCoeff * v * v);
            dX[i] = dOut[i] * 0.5f * (1f + t + v * (1f - t * t) * du);
        }
    }

    /// <inheritdoc/>
    public void InvalidateDeviceCache(Tensor t) { } // host-only backend: no device cache to invalidate

    /// <inheritdoc/>
    public void EnsureHostCurrent(Tensor t) { } // host-only backend: Data is always current

    private static float Gelu(float v)
    {
        float u = GeluC * (v + GeluCoeff * v * v * v);
        return 0.5f * v * (1f + MathF.Tanh(u));
    }

    // ---- Embedding ---------------------------------------------------------------

    /// <inheritdoc/>
    public void EmbeddingForward(Tensor table, int[] indices, Tensor output, int D)
        => EmbeddingForward(table.AsReadOnlySpan(), indices, output.AsSpan(), D);

    /// <inheritdoc/>
    public void EmbeddingBackward(Tensor dOut, int[] indices, Tensor dTable, int D)
        => EmbeddingBackward(dOut.AsReadOnlySpan(), indices, dTable.AsSpan(), D);

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void EmbeddingForward(ReadOnlySpan<float> table, int[] indices, Span<float> output, int D)
    {
        for (int t = 0; t < indices.Length; t++)
            table.Slice(indices[t] * D, D).CopyTo(output.Slice(t * D, D));
    }

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void EmbeddingBackward(ReadOnlySpan<float> dOut, int[] indices, Span<float> dTable, int D)
    {
        for (int t = 0; t < indices.Length; t++)
            AddInPlace(dTable.Slice(indices[t] * D, D), dOut.Slice(t * D, D));
    }

    // ---- Attention helpers ---------------------------------------------------------

    /// <inheritdoc/>
    public void CausalMask(Tensor scores, int T)
        => CausalMask(scores.AsSpan(), T);

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void CausalMask(Span<float> scores, int T)
    {
        for (int o = 0; o < scores.Length; o += T * T)
            for (int i = 0; i < T; i++)
                for (int j = i + 1; j < T; j++)
                    scores[o + i * T + j] = float.NegativeInfinity;
    }

    // ---- Cross-entropy --------------------------------------------------------------

    /// <inheritdoc/>
    public float CrossEntropyForward(Tensor logits, int[] targets, Tensor probs, int T, int V, int ignoreIndex)
        => CrossEntropyForward(logits.AsReadOnlySpan(), targets, probs.AsSpan(), T, V, ignoreIndex);

    /// <inheritdoc/>
    public void CrossEntropyBackward(Tensor probs, int[] targets, Tensor dLogits, int T, int V, int ignoreIndex)
        => CrossEntropyBackward(probs.AsReadOnlySpan(), targets, dLogits.AsSpan(), T, V, ignoreIndex);

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static float CrossEntropyForward(ReadOnlySpan<float> logits, int[] targets, Span<float> probs, int T, int V, int ignoreIndex)
    {
        float totalLoss = 0f;
        int count = 0;
        for (int t = 0; t < T; t++)
        {
            ReadOnlySpan<float> logitRow = logits.Slice(t * V, V);
            Span<float> probRow = probs.Slice(t * V, V);

            // probs may alias logits (in-place softmax): capture the target logit before any writes.
            bool ignored = targets[t] == ignoreIndex;
            float targetLogit = ignored ? 0f : logitRow[targets[t]];
            float max = logitRow[0];
            for (int v = 1; v < V; v++) max = MathF.Max(max, logitRow[v]);
            float sum = 0f;
            for (int v = 0; v < V; v++) { probRow[v] = MathF.Exp(logitRow[v] - max); sum += probRow[v]; }
            float inv = 1f / sum;
            for (int v = 0; v < V; v++) probRow[v] *= inv;

            if (!ignored)
            {
                totalLoss += MathF.Log(sum) + max - targetLogit; // -log softmax, stable
                count++;
            }
        }
        return count > 0 ? totalLoss / count : 0f;
    }

    /// <summary>Span twin of the public kernel; the body operates purely on spans.</summary>
    private static void CrossEntropyBackward(ReadOnlySpan<float> probs, int[] targets, Span<float> dLogits, int T, int V, int ignoreIndex)
    {
        int count = 0;
        for (int t = 0; t < T; t++)
            if (targets[t] != ignoreIndex) count++;
        if (count == 0)
        {
            dLogits.Clear();
            return;
        }

        float scale = 1f / count;
        for (int t = 0; t < T; t++)
        {
            ReadOnlySpan<float> probRow = probs.Slice(t * V, V);
            Span<float> dRow = dLogits.Slice(t * V, V);
            if (targets[t] == ignoreIndex)
            {
                dRow.Clear();
                continue;
            }
            for (int v = 0; v < V; v++)
                dRow[v] = probRow[v] * scale;
            dRow[targets[t]] -= scale;
        }
    }
}
