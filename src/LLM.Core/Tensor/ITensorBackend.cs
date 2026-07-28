namespace LLM.Core.Tensor;

/// <summary>
/// Raw numerical kernels (forward + backward) that all model code calls into.
/// The CPU implementation lives in CpuBackend; a GPU backend can implement the
/// same contract later. All arrays are row-major; shapes are passed explicitly
/// so kernels stay allocation-free and Span-friendly.
///
/// Conventions:
///  - Matrices are [rows, cols] row-major.
///  - "accumulate" means += into the destination instead of overwrite (used for
///    gradient accumulation into parameter grads).
///  - Backward kernels take the forward inputs/outputs needed to reconstruct
///    the local gradient; layer classes are responsible for caching them.
/// </summary>
public interface ITensorBackend
{
    // ---- Matmul ------------------------------------------------------------
    /// <summary>y[m,n] = sum_k a[m,k]*b[k,n].  a:[M,K], b:[K,N], y:[M,N]</summary>
    void MatMulNN(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> y, int M, int K, int N, bool accumulate = false);
    /// <summary>y[m,n] = sum_k a[m,k]*b[n,k].  a:[M,K], b:[N,K], y:[M,N]</summary>
    void MatMulNT(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> y, int M, int K, int N, bool accumulate = false);
    /// <summary>y[m,n] = sum_k a[k,m]*b[k,n].  a:[K,M], b:[K,N], y:[M,N]</summary>
    void MatMulTN(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> y, int M, int K, int N, bool accumulate = false);

    // ---- Elementwise / rows -------------------------------------------------
    /// <summary>y[r,c] += bias[c] for every row r. y:[rows,cols], bias:[cols]</summary>
    void AddBias(Span<float> y, ReadOnlySpan<float> bias, int rows, int cols);
    /// <summary>dBias[c] = sum_r dY[r,c]. dY:[rows,cols], dBias:[cols]. Accumulates.</summary>
    void SumRows(ReadOnlySpan<float> dY, Span<float> dBias, int rows, int cols);
    /// <summary>dst[i] += src[i]</summary>
    void AddInPlace(Span<float> dst, ReadOnlySpan<float> src);
    /// <summary>x[i] *= factor</summary>
    void Scale(Span<float> x, float factor);
    /// <summary>out[r,c] = x[c,r]. x:[rows,cols], out:[cols,rows]</summary>
    void Transpose(ReadOnlySpan<float> x, Span<float> output, int rows, int cols);

    // ---- LayerNorm -----------------------------------------------------------
    /// <summary>
    /// Row-wise: out = (x - mean)/sqrt(var+eps) * w + b.
    /// x,w,b,out: [rows,cols],[cols],[cols],[rows,cols]; mean,rstd:[rows] receive cached stats.
    /// </summary>
    void LayerNormForward(ReadOnlySpan<float> x, ReadOnlySpan<float> w, ReadOnlySpan<float> b,
        Span<float> output, Span<float> mean, Span<float> rstd, int rows, int cols, float eps);
    /// <summary>Accumulates dW and dB; dX is overwritten.</summary>
    void LayerNormBackward(ReadOnlySpan<float> dOut, ReadOnlySpan<float> x, ReadOnlySpan<float> w,
        ReadOnlySpan<float> mean, ReadOnlySpan<float> rstd,
        Span<float> dX, Span<float> dW, Span<float> dB, int rows, int cols);

    // ---- Softmax --------------------------------------------------------------
    /// <summary>Row-wise softmax, in place. x:[rows,cols]</summary>
    void SoftmaxForward(Span<float> x, int rows, int cols);
    /// <summary>dX = s * (dOut - sum(dOut*s)) row-wise; s = softmax output. All [rows,cols].</summary>
    void SoftmaxBackward(ReadOnlySpan<float> dOut, ReadOnlySpan<float> softmaxOut, Span<float> dX, int rows, int cols);

    // ---- GELU ------------------------------------------------------------------
    /// <summary>Tanh-approximation GELU, elementwise.</summary>
    void GeluForward(ReadOnlySpan<float> x, Span<float> output);
    void GeluBackward(ReadOnlySpan<float> dOut, ReadOnlySpan<float> x, Span<float> dX);

    // ---- Embedding ---------------------------------------------------------------
    /// <summary>out[t,:] = table[idx[t],:]. table:[V,D], out:[T,D]</summary>
    void EmbeddingForward(ReadOnlySpan<float> table, ReadOnlySpan<int> indices, Span<float> output, int D);
    /// <summary>dTable[idx[t],:] += dOut[t,:]. Accumulates.</summary>
    void EmbeddingBackward(ReadOnlySpan<float> dOut, ReadOnlySpan<int> indices, Span<float> dTable, int D);

    // ---- Attention helpers ---------------------------------------------------------
    /// <summary>Sets scores[i,j] = -inf for j > i (causal mask). scores:[T,T]</summary>
    void CausalMask(Span<float> scores, int T);

    // ---- Cross-entropy --------------------------------------------------------------
    /// <summary>
    /// Mean cross-entropy over T positions. logits:[T,V]; writes softmax probs into probs:[T,V].
    /// Positions with target == ignoreIndex are excluded from the mean.
    /// </summary>
    float CrossEntropyForward(ReadOnlySpan<float> logits, ReadOnlySpan<int> targets, Span<float> probs, int T, int V, int ignoreIndex);
    /// <summary>dLogits = (probs - onehot(target)) / count. probs from CrossEntropyForward.</summary>
    void CrossEntropyBackward(ReadOnlySpan<float> probs, ReadOnlySpan<int> targets, Span<float> dLogits, int T, int V, int ignoreIndex);
}
