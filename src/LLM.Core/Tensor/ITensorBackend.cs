namespace LLM.Core.Tensor;

/// <summary>
/// Raw numerical kernels (forward + backward) that all model code calls into.
/// The CPU implementation lives in CpuBackend; a GPU backend can implement the
/// same contract later. Kernels take whole tensors (never sub-views); tensor
/// storage is row-major and shapes are passed explicitly as ints so dispatch
/// bounds stay explicit for device backends. Index arrays (token ids, targets)
/// stay host-side plain int[] — backends upload them per call as needed.
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
    void MatMulNN(Tensor a, Tensor b, Tensor y, int M, int K, int N, bool accumulate = false);
    /// <summary>y[m,n] = sum_k a[m,k]*b[n,k].  a:[M,K], b:[N,K], y:[M,N]</summary>
    void MatMulNT(Tensor a, Tensor b, Tensor y, int M, int K, int N, bool accumulate = false);
    /// <summary>y[m,n] = sum_k a[k,m]*b[k,n].  a:[K,M], b:[K,N], y:[M,N]</summary>
    void MatMulTN(Tensor a, Tensor b, Tensor y, int M, int K, int N, bool accumulate = false);

    // ---- Batched matmul (packed independent slots) --------------------------------
    // Attention runs S = batch * heads independent small matmuls per layer. These
    // variants take slot-packed operands (slot s occupies rows s*M..(s+1)*M of a/y)
    // and apply the operation per slot, so a device backend pays ONE dispatch for
    // the whole layer instead of one per slot.

    /// <summary>Per slot s: y[s,m,n] = sum_k a[s,m,k]*b[s,k,n]. a:[S*M,K], b:[S*K,N], y:[S*M,N]</summary>
    void BatchedMatMulNN(Tensor a, Tensor b, Tensor y, int slots, int M, int K, int N, bool accumulate = false);
    /// <summary>Per slot s: y[s,m,n] = sum_k a[s,m,k]*b[s,n,k]. a:[S*M,K], b:[S*N,K], y:[S*M,N]</summary>
    void BatchedMatMulNT(Tensor a, Tensor b, Tensor y, int slots, int M, int K, int N, bool accumulate = false);
    /// <summary>Per slot s: y[s,m,n] = sum_k a[s,k,m]*b[s,k,n]. a:[S*K,M], b:[S*K,N], y:[S*M,N]</summary>
    void BatchedMatMulTN(Tensor a, Tensor b, Tensor y, int slots, int M, int K, int N, bool accumulate = false);

    // ---- Attention head packing -----------------------------------------------------

    /// <summary>
    /// Packs (sequence, head) slices of src into slot-contiguous form:
    /// dst[s*T + t, d] = src[seq*T + t, colBase + h*headDim + d] where s = seq*nHeads + h,
    /// for seq in [0,batch), h in [0,nHeads). src row width is taken from src.Shape[^1];
    /// dst is [batch*nHeads*T, headDim].
    /// </summary>
    void PackHeads(Tensor src, Tensor dst, int batch, int T, int nHeads, int headDim, int colBase);

    /// <summary>
    /// Inverse of <see cref="PackHeads"/>: dst[seq*T + t, colBase + h*headDim + d] = src[s*T + t, d].
    /// dst row width is taken from dst.Shape[^1]; src is [batch*nHeads*T, headDim].
    /// </summary>
    void UnpackHeads(Tensor src, Tensor dst, int batch, int T, int nHeads, int headDim, int colBase);

    // ---- Elementwise / rows -------------------------------------------------
    /// <summary>y[r,c] += bias[c] for every row r. y:[rows,cols], bias:[cols]</summary>
    void AddBias(Tensor y, Tensor bias, int rows, int cols);
    /// <summary>dBias[c] = sum_r dY[r,c]. dY:[rows,cols], dBias:[cols]. Accumulates.</summary>
    void SumRows(Tensor dY, Tensor dBias, int rows, int cols);
    /// <summary>dst[i] += src[i]</summary>
    void AddInPlace(Tensor dst, Tensor src);
    /// <summary>dst[i] = src[i] (full-tensor copy; dst and src have equal length).</summary>
    void Copy(Tensor src, Tensor dst);
    /// <summary>
    /// Strided 2-D block copy between row-major matrices: dst[dstRow+r, dstCol+c] = src[srcRow+r, srcCol+c]
    /// for r in [0,rows), c in [0,cols). Column offsets are in elements of each tensor's own row
    /// width (Shape[^1]); both tensors must have room for the block.
    /// </summary>
    void CopyBlock(Tensor src, Tensor dst, int srcRow, int srcCol, int dstRow, int dstCol, int rows, int cols);
    /// <summary>x[i] *= factor</summary>
    void Scale(Tensor x, float factor);
    /// <summary>out[r,c] = x[c,r]. x:[rows,cols], out:[cols,rows]</summary>
    void Transpose(Tensor x, Tensor output, int rows, int cols);

    // ---- LayerNorm -----------------------------------------------------------
    /// <summary>
    /// Row-wise: out = (x - mean)/sqrt(var+eps) * w + b.
    /// x,w,b,out: [rows,cols],[cols],[cols],[rows,cols]; mean,rstd:[rows] receive cached stats.
    /// </summary>
    void LayerNormForward(Tensor x, Tensor w, Tensor b,
        Tensor output, Tensor mean, Tensor rstd, int rows, int cols, float eps);
    /// <summary>Accumulates dW and dB; dX is overwritten.</summary>
    void LayerNormBackward(Tensor dOut, Tensor x, Tensor w,
        Tensor mean, Tensor rstd,
        Tensor dX, Tensor dW, Tensor dB, int rows, int cols);

    // ---- Softmax --------------------------------------------------------------
    /// <summary>Row-wise softmax, in place. x:[rows,cols]</summary>
    void SoftmaxForward(Tensor x, int rows, int cols);
    /// <summary>dX = s * (dOut - sum(dOut*s)) row-wise; s = softmax output. All [rows,cols].</summary>
    void SoftmaxBackward(Tensor dOut, Tensor softmaxOut, Tensor dX, int rows, int cols);

    // ---- GELU ------------------------------------------------------------------
    /// <summary>Tanh-approximation GELU, elementwise.</summary>
    void GeluForward(Tensor x, Tensor output);
    void GeluBackward(Tensor dOut, Tensor x, Tensor dX);

    // ---- Embedding ---------------------------------------------------------------
    /// <summary>out[t,:] = table[idx[t],:]. table:[V,D], out:[T,D]; indices is a host-side int[T].</summary>
    void EmbeddingForward(Tensor table, int[] indices, Tensor output, int D);
    /// <summary>dTable[idx[t],:] += dOut[t,:]. Accumulates.</summary>
    void EmbeddingBackward(Tensor dOut, int[] indices, Tensor dTable, int D);

    // ---- Attention helpers ---------------------------------------------------------
    /// <summary>Sets scores[i,j] = -inf for j &gt; i (causal mask), per T×T block. scores is one or more packed [T,T] blocks.</summary>
    void CausalMask(Tensor scores, int T);

    // ---- Cross-entropy --------------------------------------------------------------
    /// <summary>
    /// Mean cross-entropy over T positions. logits:[T,V]; writes softmax probs into probs:[T,V].
    /// targets is a host-side int[T]; positions with target == ignoreIndex are excluded from the mean.
    /// probs may alias logits (the kernel is row read-then-write and captures the target
    /// logit before writing), enabling in-place softmax for large vocabularies.
    /// </summary>
    float CrossEntropyForward(Tensor logits, int[] targets, Tensor probs, int T, int V, int ignoreIndex);
    /// <summary>dLogits = (probs - onehot(target)) / count. probs from CrossEntropyForward.
    /// dLogits may alias probs (elementwise kernel).</summary>
    void CrossEntropyBackward(Tensor probs, int[] targets, Tensor dLogits, int T, int V, int ignoreIndex);

    // ---- Host/device synchronization ------------------------------------------
    // These two hooks are how non-kernel code interoperates with device-resident
    // backends. The CPU backend implements both as no-ops; on the GPU backend
    // Tensor.Data is NOT guaranteed current after kernel execution (device buffers
    // are authoritative), and device caches are NOT aware of direct writes to
    // Tensor.Data. The contract:
    //  - Any code that writes t.Data outside of kernel calls (optimizers, gradient
    //    clipping, checkpoint loading, init fills) MUST call InvalidateDeviceCache(t)
    //    afterwards, so the next kernel re-uploads instead of using a stale copy.
    //  - Any code that reads t.Data after kernels may have written t on device
    //    (checkpoint saving, host-side copies, sampling) MUST call
    //    EnsureHostCurrent(t) first, so the device contents are downloaded.
    // Reading t.Data without EnsureHostCurrent is only valid for tensors that no
    // kernel has written since the last host write; writing t.Data without
    // InvalidateDeviceCache leaves subsequent kernels operating on stale data.

    /// <summary>
    /// Marks any device-side copy of <paramref name="t"/> stale: the caller just
    /// wrote <see cref="Tensor.Data"/> directly and host memory is authoritative.
    /// </summary>
    void InvalidateDeviceCache(Tensor t);

    /// <summary>
    /// Brings <see cref="Tensor.Data"/> up to date with the device copy if a kernel
    /// wrote <paramref name="t"/> on device since the last synchronization. No-op
    /// when the host copy is already current (or the tensor is host-only).
    /// </summary>
    void EnsureHostCurrent(Tensor t);

    /// <summary>
    /// Zeroes <paramref name="t"/>. The default clears the host data and drops any
    /// device copy (re-uploaded on next kernel use); device backends can override
    /// to zero in place on device and skip the re-upload.
    /// </summary>
    void Zero(Tensor t)
    {
        t.Zero();
        InvalidateDeviceCache(t);
    }

    /// <summary>Prints backend profiling counters since the last call (profiling builds/runs only); no-op by default.</summary>
    void DumpStats(string tag) { }

    /// <summary>
    /// Sum of squares of all elements of <paramref name="t"/> (for gradient-norm
    /// clipping). The default downloads (if needed) and sums on the host in double
    /// precision; device backends can override to reduce on device.
    /// </summary>
    double SumSquares(Tensor t)
    {
        EnsureHostCurrent(t);
        double sum = 0;
        foreach (float x in t.Data) sum += (double)x * x;
        return sum;
    }

    /// <summary>
    /// One AdamW update over a parameter: m/v moment EMAs with bias correction
    /// (<paramref name="step"/> is 1-based) and decoupled weight decay for rank &gt; 1
    /// tensors when <paramref name="weightDecay"/> != 0. The default runs the update on
    /// the host and marks the updated tensors' device caches stale; device backends can
    /// override to run it fully on device.
    /// </summary>
    void AdamWStep(Tensor w, Tensor g, Tensor m, Tensor v,
        float lr, float beta1, float beta2, float eps, float weightDecay, int step)
    {
        EnsureHostCurrent(w);
        EnsureHostCurrent(g);
        EnsureHostCurrent(m);
        EnsureHostCurrent(v);
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        bool decay = weightDecay != 0f && w.Rank > 1;
        float[] wd = w.Data, gd = g.Data, md = m.Data, vd = v.Data;
        for (int i = 0; i < wd.Length; i++)
        {
            float gi = gd[i];
            md[i] = beta1 * md[i] + (1f - beta1) * gi;
            vd[i] = beta2 * vd[i] + (1f - beta2) * gi * gi;
            if (decay) wd[i] -= lr * weightDecay * wd[i];
            float mHat = md[i] / bc1;
            float vHat = vd[i] / bc2;
            wd[i] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
        }
        InvalidateDeviceCache(w);
        InvalidateDeviceCache(m);
        InvalidateDeviceCache(v);
    }
}
