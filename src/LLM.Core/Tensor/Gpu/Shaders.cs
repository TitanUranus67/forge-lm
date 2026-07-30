using ComputeSharp;

namespace LLM.Core.Tensor.Gpu;

// ComputeSharp compute shaders for GpuBackend — one shader per kernel (some kernels
// use two passes). Conventions mirror CpuBackend: row-major [rows, cols] matrices,
// "accumulate" means += into the destination.
//
// Every float buffer parameter is paired with an int offset: tensors are sub-allocated
// chunks of shared arena buffers (ComputeSharp caps live buffer descriptors at 2048,
// far below the live-tensor count of a full training step), so all indexing is
// buffer[offset + i]. Dedicated buffers (index arrays, scratch) have no offset.
//
// "Flat" shaders are indexed by a single flat element index i and are dispatched
// either 1-D (stride = 0, i = ThreadIds.X) or 2-D (i = ThreadIds.Y * stride + ThreadIds.X)
// when the element count exceeds the 1-D dispatch limit; all of them bounds-check i.

// ---- Elementwise (flat) -------------------------------------------------------

/// <summary>dst[i] += src[i]</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct AddInPlaceShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> dst;
    public readonly int dstOff;
    public readonly ReadWriteBuffer<float> src;
    public readonly int srcOff;
    public readonly int length;
    public readonly int stride;

    public AddInPlaceShader(ReadWriteBuffer<float> dst, int dstOff, ReadWriteBuffer<float> src, int srcOff, int length, int stride)
    { this.dst = dst; this.dstOff = dstOff; this.src = src; this.srcOff = srcOff; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i < length) dst[dstOff + i] += src[srcOff + i];
    }
}

/// <summary>dst[i] = src[i]</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct CopyShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> src;
    public readonly int srcOff;
    public readonly ReadWriteBuffer<float> dst;
    public readonly int dstOff;
    public readonly int length;
    public readonly int stride;

    public CopyShader(ReadWriteBuffer<float> src, int srcOff, ReadWriteBuffer<float> dst, int dstOff, int length, int stride)
    { this.src = src; this.srcOff = srcOff; this.dst = dst; this.dstOff = dstOff; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i < length) dst[dstOff + i] = src[srcOff + i];
    }
}

/// <summary>Strided 2-D block copy: dst[dstRow+r, dstCol+c] = src[srcRow+r, srcCol+c]. Flat over rows*cols.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct CopyBlockShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> src;
    public readonly int srcOff;
    public readonly ReadWriteBuffer<float> dst;
    public readonly int dstOff;
    public readonly int srcRow;
    public readonly int srcCol;
    public readonly int dstRow;
    public readonly int dstCol;
    public readonly int srcCols;
    public readonly int dstCols;
    public readonly int cols;
    public readonly int length;
    public readonly int stride;

    public CopyBlockShader(ReadWriteBuffer<float> src, int srcOff, ReadWriteBuffer<float> dst, int dstOff,
        int srcRow, int srcCol, int dstRow, int dstCol, int srcCols, int dstCols, int cols, int length, int stride)
    { this.src = src; this.srcOff = srcOff; this.dst = dst; this.dstOff = dstOff; this.srcRow = srcRow; this.srcCol = srcCol; this.dstRow = dstRow; this.dstCol = dstCol; this.srcCols = srcCols; this.dstCols = dstCols; this.cols = cols; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i >= length) return;
        int r = i / cols, c = i % cols;
        dst[dstOff + (dstRow + r) * dstCols + dstCol + c] = src[srcOff + (srcRow + r) * srcCols + srcCol + c];
    }
}

/// <summary>x[i] *= factor</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct ScaleShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> x;
    public readonly int xOff;
    public readonly float factor;
    public readonly int length;
    public readonly int stride;

    public ScaleShader(ReadWriteBuffer<float> x, int xOff, float factor, int length, int stride)
    { this.x = x; this.xOff = xOff; this.factor = factor; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i < length) x[xOff + i] *= factor;
    }
}

/// <summary>y[r,c] += bias[c]. y is [length] flat with row width cols.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct AddBiasShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> y;
    public readonly int yOff;
    public readonly ReadWriteBuffer<float> bias;
    public readonly int biasOff;
    public readonly int cols;
    public readonly int length;
    public readonly int stride;

    public AddBiasShader(ReadWriteBuffer<float> y, int yOff, ReadWriteBuffer<float> bias, int biasOff, int cols, int length, int stride)
    { this.y = y; this.yOff = yOff; this.bias = bias; this.biasOff = biasOff; this.cols = cols; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i < length) y[yOff + i] += bias[biasOff + i % cols];
    }
}

/// <summary>out[c,r] = x[r,c] for x [rows,cols].</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct TransposeShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> x;
    public readonly int xOff;
    public readonly ReadWriteBuffer<float> output;
    public readonly int outputOff;
    public readonly int rows;
    public readonly int cols;
    public readonly int length;
    public readonly int stride;

    public TransposeShader(ReadWriteBuffer<float> x, int xOff, ReadWriteBuffer<float> output, int outputOff, int rows, int cols, int length, int stride)
    { this.x = x; this.xOff = xOff; this.output = output; this.outputOff = outputOff; this.rows = rows; this.cols = cols; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i < length) output[outputOff + (i % cols) * rows + i / cols] = x[xOff + i];
    }
}

/// <summary>Tanh-approximation GELU, elementwise.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct GeluForwardShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> x;
    public readonly int xOff;
    public readonly ReadWriteBuffer<float> output;
    public readonly int outputOff;
    public readonly int length;
    public readonly int stride;

    public GeluForwardShader(ReadWriteBuffer<float> x, int xOff, ReadWriteBuffer<float> output, int outputOff, int length, int stride)
    { this.x = x; this.xOff = xOff; this.output = output; this.outputOff = outputOff; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i >= length) return;
        float v = x[xOff + i];
        float u = 0.7978845608028654f * (v + 0.044715f * v * v * v); // sqrt(2/pi)
        output[outputOff + i] = 0.5f * v * (1f + Hlsl.Tanh(u));
    }
}

/// <summary>dX = dOut * gelu'(x), tanh-approximation GELU.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct GeluBackwardShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> dOut;
    public readonly int dOutOff;
    public readonly ReadWriteBuffer<float> x;
    public readonly int xOff;
    public readonly ReadWriteBuffer<float> dX;
    public readonly int dXOff;
    public readonly int length;
    public readonly int stride;

    public GeluBackwardShader(ReadWriteBuffer<float> dOut, int dOutOff, ReadWriteBuffer<float> x, int xOff, ReadWriteBuffer<float> dX, int dXOff, int length, int stride)
    { this.dOut = dOut; this.dOutOff = dOutOff; this.x = x; this.xOff = xOff; this.dX = dX; this.dXOff = dXOff; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i >= length) return;
        float v = x[xOff + i];
        float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        float t = Hlsl.Tanh(u);
        float du = 0.7978845608028654f * (1f + 3f * 0.044715f * v * v);
        dX[dXOff + i] = dOut[dOutOff + i] * 0.5f * (1f + t + v * (1f - t * t) * du);
    }
}

/// <summary>scores[i,j] = -inf for j &gt; i (causal mask). scores is [T,T] flat.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct CausalMaskShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> scores;
    public readonly int scoresOff;
    public readonly int t;
    public readonly int length;
    public readonly int stride;

    public CausalMaskShader(ReadWriteBuffer<float> scores, int scoresOff, int t, int length, int stride)
    { this.scores = scores; this.scoresOff = scoresOff; this.t = t; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        // scores is one or more packed [t,t] blocks: local row/col within the block
        if (i < length && i % t > (i / t) % t) scores[scoresOff + i] = float.NegativeInfinity;
    }
}

/// <summary>dst[i] = asint(src[i]) — bit-cast float bits into an int buffer (for CAS atomics).</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct FloatBitsToIntShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> src;
    public readonly int srcOff;
    public readonly ReadWriteBuffer<int> dst;
    public readonly int length;
    public readonly int stride;

    public FloatBitsToIntShader(ReadWriteBuffer<float> src, int srcOff, ReadWriteBuffer<int> dst, int length, int stride)
    { this.src = src; this.srcOff = srcOff; this.dst = dst; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i < length) dst[i] = Hlsl.AsInt(src[srcOff + i]);
    }
}

/// <summary>dst[i] = asfloat(src[i]) — bit-cast int bits back into a float buffer.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct IntBitsToFloatShader : IComputeShader
{
    public readonly ReadWriteBuffer<int> src;
    public readonly ReadWriteBuffer<float> dst;
    public readonly int dstOff;
    public readonly int length;
    public readonly int stride;

    public IntBitsToFloatShader(ReadWriteBuffer<int> src, ReadWriteBuffer<float> dst, int dstOff, int length, int stride)
    { this.src = src; this.dst = dst; this.dstOff = dstOff; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i < length) dst[dstOff + i] = Hlsl.AsFloat(src[i]);
    }
}

/// <summary>
/// dTable[idx[t],d] += dOut[t,d] via compare-exchange float atomics on a bit-cast
/// int copy of the table (HLSL has no float InterlockedAdd on typed buffers).
/// Dispatched over (D, T).
/// </summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct EmbeddingBackwardShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> dOut;
    public readonly int dOutOff;
    public readonly ReadWriteBuffer<int> indices;
    public readonly ReadWriteBuffer<int> dTableBits;
    public readonly int d;

    public EmbeddingBackwardShader(ReadWriteBuffer<float> dOut, int dOutOff, ReadWriteBuffer<int> indices, ReadWriteBuffer<int> dTableBits, int d)
    { this.dOut = dOut; this.dOutOff = dOutOff; this.indices = indices; this.dTableBits = dTableBits; this.d = d; }

    public void Execute()
    {
        int t = ThreadIds.Y, col = ThreadIds.X;
        if (col >= d) return;
        int i = indices[t] * d + col;
        float add = dOut[dOutOff + t * d + col];
        int assumed, oldBits = dTableBits[i];
        do
        {
            assumed = oldBits;
            int newBits = Hlsl.AsInt(Hlsl.AsFloat(assumed) + add);
            Hlsl.InterlockedCompareExchange(ref dTableBits[i], assumed, newBits, out oldBits);
        } while (oldBits != assumed);
    }
}

// ---- Row-wise (one thread per row) ---------------------------------------------

/// <summary>Row-wise softmax, in place. One 256-thread group per row; block max/sum reductions in groupshared memory.</summary>
[ThreadGroupSize(256, 1, 1)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct SoftmaxForwardShader : IComputeShader
{
    [GroupShared(256)]
    private static float[] red = null!;

    public readonly ReadWriteBuffer<float> x;
    public readonly int xOff;
    public readonly int cols;

    public SoftmaxForwardShader(ReadWriteBuffer<float> x, int xOff, int cols) { this.x = x; this.xOff = xOff; this.cols = cols; }

    public void Execute()
    {
        int o = xOff + GridIds.X * cols;
        int lane = GroupIds.X; // one group per row: GridIds.X = row
        float m = float.NegativeInfinity;
        for (int c = lane; c < cols; c += 256) m = Hlsl.Max(m, x[o + c]);
        red[lane] = m;
        Hlsl.GroupMemoryBarrierWithGroupSync();
        for (int s = 128; s > 0; s >>= 1)
        {
            if (lane < s) red[lane] = Hlsl.Max(red[lane], red[lane + s]);
            Hlsl.GroupMemoryBarrierWithGroupSync();
        }
        float max = red[0];
        Hlsl.GroupMemoryBarrierWithGroupSync(); // red is reused for the sum below
        float sum = 0f;
        for (int c = lane; c < cols; c += 256) { float e = Hlsl.Exp(x[o + c] - max); x[o + c] = e; sum += e; }
        red[lane] = sum;
        Hlsl.GroupMemoryBarrierWithGroupSync();
        for (int s = 128; s > 0; s >>= 1)
        {
            if (lane < s) red[lane] += red[lane + s];
            Hlsl.GroupMemoryBarrierWithGroupSync();
        }
        float inv = 1f / red[0];
        for (int c = lane; c < cols; c += 256) x[o + c] *= inv;
    }
}

/// <summary>dX = s * (dOut - sum(dOut*s)) row-wise; s = softmax output. One 256-thread group per row.</summary>
[ThreadGroupSize(256, 1, 1)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct SoftmaxBackwardShader : IComputeShader
{
    [GroupShared(256)]
    private static float[] red = null!;

    public readonly ReadWriteBuffer<float> dOut;
    public readonly int dOutOff;
    public readonly ReadWriteBuffer<float> softmaxOut;
    public readonly int softmaxOutOff;
    public readonly ReadWriteBuffer<float> dX;
    public readonly int dXOff;
    public readonly int cols;

    public SoftmaxBackwardShader(ReadWriteBuffer<float> dOut, int dOutOff, ReadWriteBuffer<float> softmaxOut, int softmaxOutOff, ReadWriteBuffer<float> dX, int dXOff, int cols)
    { this.dOut = dOut; this.dOutOff = dOutOff; this.softmaxOut = softmaxOut; this.softmaxOutOff = softmaxOutOff; this.dX = dX; this.dXOff = dXOff; this.cols = cols; }

    public void Execute()
    {
        int r = GridIds.X, lane = GroupIds.X;
        int oD = dOutOff + r * cols, oS = softmaxOutOff + r * cols, oX = dXOff + r * cols;
        float dot = 0f;
        for (int c = lane; c < cols; c += 256) dot += dOut[oD + c] * softmaxOut[oS + c];
        red[lane] = dot;
        Hlsl.GroupMemoryBarrierWithGroupSync();
        for (int s = 128; s > 0; s >>= 1)
        {
            if (lane < s) red[lane] += red[lane + s];
            Hlsl.GroupMemoryBarrierWithGroupSync();
        }
        float d = red[0];
        for (int c = lane; c < cols; c += 256) dX[oX + c] = softmaxOut[oS + c] * (dOut[oD + c] - d);
    }
}

/// <summary>Row-wise layer normalization; also writes mean/rstd caches. One thread per row.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct LayerNormForwardShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> x;
    public readonly int xOff;
    public readonly ReadWriteBuffer<float> w;
    public readonly int wOff;
    public readonly ReadWriteBuffer<float> b;
    public readonly int bOff;
    public readonly ReadWriteBuffer<float> output;
    public readonly int outputOff;
    public readonly ReadWriteBuffer<float> mean;
    public readonly int meanOff;
    public readonly ReadWriteBuffer<float> rstd;
    public readonly int rstdOff;
    public readonly int cols;
    public readonly float eps;

    public LayerNormForwardShader(ReadWriteBuffer<float> x, int xOff, ReadWriteBuffer<float> w, int wOff, ReadWriteBuffer<float> b, int bOff,
        ReadWriteBuffer<float> output, int outputOff, ReadWriteBuffer<float> mean, int meanOff, ReadWriteBuffer<float> rstd, int rstdOff,
        int cols, float eps)
    { this.x = x; this.xOff = xOff; this.w = w; this.wOff = wOff; this.b = b; this.bOff = bOff; this.output = output; this.outputOff = outputOff; this.mean = mean; this.meanOff = meanOff; this.rstd = rstd; this.rstdOff = rstdOff; this.cols = cols; this.eps = eps; }

    public void Execute()
    {
        int r = ThreadIds.X, oX = xOff + r * cols, oO = outputOff + r * cols;
        float sum = 0f;
        for (int c = 0; c < cols; c++) sum += x[oX + c];
        float mu = sum / cols;
        float varSum = 0f;
        for (int c = 0; c < cols; c++) { float d0 = x[oX + c] - mu; varSum += d0 * d0; }
        float rs = 1f / Hlsl.Sqrt(varSum / cols + eps); // biased variance
        mean[meanOff + r] = mu;
        rstd[rstdOff + r] = rs;
        for (int c = 0; c < cols; c++)
            output[oO + c] = (x[oX + c] - mu) * rs * w[wOff + c] + b[bOff + c];
    }
}

/// <summary>dX pass of layer-normalization backward. One thread per row.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct LayerNormBackwardDxShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> dOut;
    public readonly int dOutOff;
    public readonly ReadWriteBuffer<float> x;
    public readonly int xOff;
    public readonly ReadWriteBuffer<float> w;
    public readonly int wOff;
    public readonly ReadWriteBuffer<float> mean;
    public readonly int meanOff;
    public readonly ReadWriteBuffer<float> rstd;
    public readonly int rstdOff;
    public readonly ReadWriteBuffer<float> dX;
    public readonly int dXOff;
    public readonly int cols;

    public LayerNormBackwardDxShader(ReadWriteBuffer<float> dOut, int dOutOff, ReadWriteBuffer<float> x, int xOff, ReadWriteBuffer<float> w, int wOff,
        ReadWriteBuffer<float> mean, int meanOff, ReadWriteBuffer<float> rstd, int rstdOff, ReadWriteBuffer<float> dX, int dXOff, int cols)
    { this.dOut = dOut; this.dOutOff = dOutOff; this.x = x; this.xOff = xOff; this.w = w; this.wOff = wOff; this.mean = mean; this.meanOff = meanOff; this.rstd = rstd; this.rstdOff = rstdOff; this.dX = dX; this.dXOff = dXOff; this.cols = cols; }

    public void Execute()
    {
        int r = ThreadIds.X;
        float mu = mean[meanOff + r], rs = rstd[rstdOff + r];
        int oD = dOutOff + r * cols, oX = xOff + r * cols, oDx = dXOff + r * cols;
        float sumDxhat = 0f, sumDxhatXhat = 0f;
        for (int c = 0; c < cols; c++)
        {
            float xhat = (x[oX + c] - mu) * rs;
            float dxhat = dOut[oD + c] * w[wOff + c];
            dX[oDx + c] = dxhat; // stash dxhat; converted to dX in the second pass
            sumDxhat += dxhat;
            sumDxhatXhat += dxhat * xhat;
        }
        float m1 = sumDxhat / cols, m2 = sumDxhatXhat / cols;
        for (int c = 0; c < cols; c++)
        {
            float xhat = (x[oX + c] - mu) * rs;
            dX[oDx + c] = rs * (dX[oDx + c] - m1 - xhat * m2);
        }
    }
}

/// <summary>dW/dB accumulation pass of layer-normalization backward. One thread per column.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct LayerNormBackwardDwDbShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> dOut;
    public readonly int dOutOff;
    public readonly ReadWriteBuffer<float> x;
    public readonly int xOff;
    public readonly ReadWriteBuffer<float> mean;
    public readonly int meanOff;
    public readonly ReadWriteBuffer<float> rstd;
    public readonly int rstdOff;
    public readonly ReadWriteBuffer<float> dW;
    public readonly int dWOff;
    public readonly ReadWriteBuffer<float> dB;
    public readonly int dBOff;
    public readonly int rows;
    public readonly int cols;

    public LayerNormBackwardDwDbShader(ReadWriteBuffer<float> dOut, int dOutOff, ReadWriteBuffer<float> x, int xOff,
        ReadWriteBuffer<float> mean, int meanOff, ReadWriteBuffer<float> rstd, int rstdOff,
        ReadWriteBuffer<float> dW, int dWOff, ReadWriteBuffer<float> dB, int dBOff, int rows, int cols)
    { this.dOut = dOut; this.dOutOff = dOutOff; this.x = x; this.xOff = xOff; this.mean = mean; this.meanOff = meanOff; this.rstd = rstd; this.rstdOff = rstdOff; this.dW = dW; this.dWOff = dWOff; this.dB = dB; this.dBOff = dBOff; this.rows = rows; this.cols = cols; }

    public void Execute()
    {
        int c = ThreadIds.X;
        if (c >= cols) return;
        float accW = 0f, accB = 0f;
        for (int r = 0; r < rows; r++)
        {
            float xhat = (x[xOff + r * cols + c] - mean[meanOff + r]) * rstd[rstdOff + r];
            accW += dOut[dOutOff + r * cols + c] * xhat;
            accB += dOut[dOutOff + r * cols + c];
        }
        dW[dWOff + c] += accW;
        dB[dBOff + c] += accB;
    }
}

/// <summary>dBias[c] = sum_r dY[r,c]. Accumulates. One thread per column.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct SumRowsShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> dY;
    public readonly int dYOff;
    public readonly ReadWriteBuffer<float> dBias;
    public readonly int dBiasOff;
    public readonly int rows;
    public readonly int cols;

    public SumRowsShader(ReadWriteBuffer<float> dY, int dYOff, ReadWriteBuffer<float> dBias, int dBiasOff, int rows, int cols)
    { this.dY = dY; this.dYOff = dYOff; this.dBias = dBias; this.dBiasOff = dBiasOff; this.rows = rows; this.cols = cols; }

    public void Execute()
    {
        int c = ThreadIds.X;
        if (c >= cols) return;
        float acc = 0f;
        for (int r = 0; r < rows; r++) acc += dY[dYOff + r * cols + c];
        dBias[dBiasOff + c] += acc;
    }
}

// ---- Cross-entropy --------------------------------------------------------------

/// <summary>
/// Softmax over each row of logits into probs, plus per-row NLL into nll
/// (0 for ignored positions). The mean loss is computed on the host. One thread per row.
/// </summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct CrossEntropyForwardShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> logits;
    public readonly int logitsOff;
    public readonly ReadWriteBuffer<int> targets;
    public readonly ReadWriteBuffer<float> probs;
    public readonly int probsOff;
    public readonly ReadWriteBuffer<float> nll;
    public readonly int v;
    public readonly int ignoreIndex;

    public CrossEntropyForwardShader(ReadWriteBuffer<float> logits, int logitsOff, ReadWriteBuffer<int> targets,
        ReadWriteBuffer<float> probs, int probsOff, ReadWriteBuffer<float> nll, int v, int ignoreIndex)
    { this.logits = logits; this.logitsOff = logitsOff; this.targets = targets; this.probs = probs; this.probsOff = probsOff; this.nll = nll; this.v = v; this.ignoreIndex = ignoreIndex; }

    public void Execute()
    {
        int t = ThreadIds.X, oL = logitsOff + t * v, oP = probsOff + t * v;
        // probs may alias logits (in-place softmax): capture the target logit before any writes.
        int target = targets[t];
        float targetLogit = target == ignoreIndex ? 0f : logits[oL + target];
        float max = logits[oL];
        for (int c = 1; c < v; c++) max = Hlsl.Max(max, logits[oL + c]);
        float sum = 0f;
        for (int c = 0; c < v; c++) { float e = Hlsl.Exp(logits[oL + c] - max); probs[oP + c] = e; sum += e; }
        float inv = 1f / sum;
        for (int c = 0; c < v; c++) probs[oP + c] *= inv;
        nll[t] = target == ignoreIndex ? 0f : Hlsl.Log(sum) + max - targetLogit; // -log softmax, stable
    }
}

/// <summary>dLogits = (probs - onehot(target)) / count. Flat over T*V.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct CrossEntropyBackwardShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> probs;
    public readonly int probsOff;
    public readonly ReadWriteBuffer<int> targets;
    public readonly ReadWriteBuffer<float> dLogits;
    public readonly int dLogitsOff;
    public readonly int v;
    public readonly int ignoreIndex;
    public readonly float scale;
    public readonly int length;
    public readonly int stride;

    public CrossEntropyBackwardShader(ReadWriteBuffer<float> probs, int probsOff, ReadWriteBuffer<int> targets,
        ReadWriteBuffer<float> dLogits, int dLogitsOff, int v, int ignoreIndex, float scale, int length, int stride)
    { this.probs = probs; this.probsOff = probsOff; this.targets = targets; this.dLogits = dLogits; this.dLogitsOff = dLogitsOff; this.v = v; this.ignoreIndex = ignoreIndex; this.scale = scale; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i >= length) return;
        int target = targets[i / v];
        dLogits[dLogitsOff + i] = target == ignoreIndex ? 0f : probs[probsOff + i] * scale - (i % v == target ? scale : 0f);
    }
}

// ---- Embedding -------------------------------------------------------------------

/// <summary>out[t,:] = table[idx[t],:]. Dispatched over (D, T).</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct EmbeddingForwardShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> table;
    public readonly int tableOff;
    public readonly ReadWriteBuffer<int> indices;
    public readonly ReadWriteBuffer<float> output;
    public readonly int outputOff;
    public readonly int d;

    public EmbeddingForwardShader(ReadWriteBuffer<float> table, int tableOff, ReadWriteBuffer<int> indices, ReadWriteBuffer<float> output, int outputOff, int d)
    { this.table = table; this.tableOff = tableOff; this.indices = indices; this.output = output; this.outputOff = outputOff; this.d = d; }

    public void Execute()
    {
        int t = ThreadIds.Y, col = ThreadIds.X;
        if (col < d) output[outputOff + t * d + col] = table[tableOff + indices[t] * d + col];
    }
}

// ---- Matmul (16x16 tiled with groupshared memory, dispatched over (N, M)) -----------
//
// Each 16x16 thread block computes one 16x16 output tile: tiles of A and B are staged
// through groupshared memory in K-steps of 16, giving 16x reuse of every global load.
// Edge tiles are zero-padded, so any (M, K, N) works.

/// <summary>y[m,n] = sum_k a[m,k]*b[k,n]. a:[M,K], b:[K,N], y:[M,N]</summary>
[ThreadGroupSize(16, 16, 1)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct MatMulNnShader : IComputeShader
{
    [GroupShared(256)]
    private static float[] tileA = null!;
    [GroupShared(256)]
    private static float[] tileB = null!;

    public readonly ReadWriteBuffer<float> a;
    public readonly int aOff;
    public readonly ReadWriteBuffer<float> b;
    public readonly int bOff;
    public readonly ReadWriteBuffer<float> y;
    public readonly int yOff;
    public readonly int m;
    public readonly int k;
    public readonly int n;
    public readonly int accumulate;

    public MatMulNnShader(ReadWriteBuffer<float> a, int aOff, ReadWriteBuffer<float> b, int bOff, ReadWriteBuffer<float> y, int yOff,
        int m, int k, int n, int accumulate)
    { this.a = a; this.aOff = aOff; this.b = b; this.bOff = bOff; this.y = y; this.yOff = yOff; this.m = m; this.k = k; this.n = n; this.accumulate = accumulate; }

    public void Execute()
    {
        int tx = GroupIds.X, ty = GroupIds.Y; // thread id within the 16x16 block
        int row = ThreadIds.Y, col = ThreadIds.X; // global output coordinates
        float sum = 0f;
        for (int k0 = 0; k0 < k; k0 += 16)
        {
            tileA[ty * 16 + tx] = row < m && k0 + tx < k ? a[aOff + row * k + k0 + tx] : 0f;
            tileB[ty * 16 + tx] = k0 + ty < k && col < n ? b[bOff + (k0 + ty) * n + col] : 0f;
            Hlsl.GroupMemoryBarrierWithGroupSync();
            for (int kk = 0; kk < 16; kk++)
                sum += tileA[ty * 16 + kk] * tileB[kk * 16 + tx];
            Hlsl.GroupMemoryBarrierWithGroupSync();
        }
        if (row >= m || col >= n) return;
        if (accumulate != 0) y[yOff + row * n + col] += sum;
        else y[yOff + row * n + col] = sum;
    }
}

/// <summary>y[m,n] = sum_k a[m,k]*b[n,k]. a:[M,K], b:[N,K], y:[M,N]</summary>
[ThreadGroupSize(16, 16, 1)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct MatMulNtShader : IComputeShader
{
    [GroupShared(256)]
    private static float[] tileA = null!;
    [GroupShared(256)]
    private static float[] tileB = null!;

    public readonly ReadWriteBuffer<float> a;
    public readonly int aOff;
    public readonly ReadWriteBuffer<float> b;
    public readonly int bOff;
    public readonly ReadWriteBuffer<float> y;
    public readonly int yOff;
    public readonly int m;
    public readonly int k;
    public readonly int n;
    public readonly int accumulate;

    public MatMulNtShader(ReadWriteBuffer<float> a, int aOff, ReadWriteBuffer<float> b, int bOff, ReadWriteBuffer<float> y, int yOff,
        int m, int k, int n, int accumulate)
    { this.a = a; this.aOff = aOff; this.b = b; this.bOff = bOff; this.y = y; this.yOff = yOff; this.m = m; this.k = k; this.n = n; this.accumulate = accumulate; }

    public void Execute()
    {
        int tx = GroupIds.X, ty = GroupIds.Y; // thread id within the 16x16 block
        int row = ThreadIds.Y, col = ThreadIds.X; // global output coordinates
        float sum = 0f;
        for (int k0 = 0; k0 < k; k0 += 16)
        {
            tileA[ty * 16 + tx] = row < m && k0 + tx < k ? a[aOff + row * k + k0 + tx] : 0f;
            tileB[ty * 16 + tx] = k0 + ty < k && col < n ? b[bOff + col * k + k0 + ty] : 0f;
            Hlsl.GroupMemoryBarrierWithGroupSync();
            for (int kk = 0; kk < 16; kk++)
                sum += tileA[ty * 16 + kk] * tileB[kk * 16 + tx];
            Hlsl.GroupMemoryBarrierWithGroupSync();
        }
        if (row >= m || col >= n) return;
        if (accumulate != 0) y[yOff + row * n + col] += sum;
        else y[yOff + row * n + col] = sum;
    }
}

/// <summary>y[m,n] = sum_k a[k,m]*b[k,n]. a:[K,M], b:[K,N], y:[M,N]</summary>
[ThreadGroupSize(16, 16, 1)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct MatMulTnShader : IComputeShader
{
    [GroupShared(256)]
    private static float[] tileA = null!;
    [GroupShared(256)]
    private static float[] tileB = null!;

    public readonly ReadWriteBuffer<float> a;
    public readonly int aOff;
    public readonly ReadWriteBuffer<float> b;
    public readonly int bOff;
    public readonly ReadWriteBuffer<float> y;
    public readonly int yOff;
    public readonly int m;
    public readonly int k;
    public readonly int n;
    public readonly int accumulate;

    public MatMulTnShader(ReadWriteBuffer<float> a, int aOff, ReadWriteBuffer<float> b, int bOff, ReadWriteBuffer<float> y, int yOff,
        int m, int k, int n, int accumulate)
    { this.a = a; this.aOff = aOff; this.b = b; this.bOff = bOff; this.y = y; this.yOff = yOff; this.m = m; this.k = k; this.n = n; this.accumulate = accumulate; }

    public void Execute()
    {
        int tx = GroupIds.X, ty = GroupIds.Y; // thread id within the 16x16 block
        int row = ThreadIds.Y, col = ThreadIds.X; // global output coordinates
        int m0 = row - ty; // block's first output row
        float sum = 0f;
        for (int k0 = 0; k0 < k; k0 += 16)
        {
            tileA[ty * 16 + tx] = k0 + ty < k && m0 + tx < m ? a[aOff + (k0 + ty) * m + m0 + tx] : 0f;
            tileB[ty * 16 + tx] = k0 + ty < k && col < n ? b[bOff + (k0 + ty) * n + col] : 0f;
            Hlsl.GroupMemoryBarrierWithGroupSync();
            for (int kk = 0; kk < 16; kk++)
                sum += tileA[kk * 16 + ty] * tileB[kk * 16 + tx];
            Hlsl.GroupMemoryBarrierWithGroupSync();
        }
        if (row >= m || col >= n) return;
        if (accumulate != 0) y[yOff + row * n + col] += sum;
        else y[yOff + row * n + col] = sum;
    }
}

// ---- Batched matmul (packed slots; 3-D dispatch (N, M, slots), 16x16 tiles) ----------
//
// One dispatch covers all slots: the Z axis is the slot index, so every thread block
// belongs to exactly one slot and can stage its A/B tiles in groupshared memory like
// the single matmuls. Dispatches are padded to multiples of 16 on X/Y (see GpuBackend).

/// <summary>Per slot s: y[s,m,n] = sum_k a[s,m,k]*b[s,k,n]. a:[S*M,K], b:[S*K,N], y:[S*M,N]</summary>
[ThreadGroupSize(16, 16, 1)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct BatchedMatMulNnShader : IComputeShader
{
    [GroupShared(256)]
    private static float[] tileA = null!;
    [GroupShared(256)]
    private static float[] tileB = null!;

    public readonly ReadWriteBuffer<float> a;
    public readonly int aOff;
    public readonly ReadWriteBuffer<float> b;
    public readonly int bOff;
    public readonly ReadWriteBuffer<float> y;
    public readonly int yOff;
    public readonly int m;
    public readonly int k;
    public readonly int n;
    public readonly int accumulate;

    public BatchedMatMulNnShader(ReadWriteBuffer<float> a, int aOff, ReadWriteBuffer<float> b, int bOff, ReadWriteBuffer<float> y, int yOff,
        int m, int k, int n, int accumulate)
    { this.a = a; this.aOff = aOff; this.b = b; this.bOff = bOff; this.y = y; this.yOff = yOff; this.m = m; this.k = k; this.n = n; this.accumulate = accumulate; }

    public void Execute()
    {
        int tx = GroupIds.X, ty = GroupIds.Y;
        int s = ThreadIds.Z, row = ThreadIds.Y, col = ThreadIds.X;
        int aRow = s * m + row; // row within packed a/y
        float sum = 0f;
        for (int k0 = 0; k0 < k; k0 += 16)
        {
            tileA[ty * 16 + tx] = row < m && k0 + tx < k ? a[aOff + aRow * k + k0 + tx] : 0f;
            tileB[ty * 16 + tx] = k0 + ty < k && col < n ? b[bOff + (s * k + k0 + ty) * n + col] : 0f;
            Hlsl.GroupMemoryBarrierWithGroupSync();
            for (int kk = 0; kk < 16; kk++)
                sum += tileA[ty * 16 + kk] * tileB[kk * 16 + tx];
            Hlsl.GroupMemoryBarrierWithGroupSync();
        }
        if (row >= m || col >= n) return;
        if (accumulate != 0) y[yOff + aRow * n + col] += sum;
        else y[yOff + aRow * n + col] = sum;
    }
}

/// <summary>Per slot s: y[s,m,n] = sum_k a[s,m,k]*b[s,n,k]. a:[S*M,K], b:[S*N,K], y:[S*M,N]</summary>
[ThreadGroupSize(16, 16, 1)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct BatchedMatMulNtShader : IComputeShader
{
    [GroupShared(256)]
    private static float[] tileA = null!;
    [GroupShared(256)]
    private static float[] tileB = null!;

    public readonly ReadWriteBuffer<float> a;
    public readonly int aOff;
    public readonly ReadWriteBuffer<float> b;
    public readonly int bOff;
    public readonly ReadWriteBuffer<float> y;
    public readonly int yOff;
    public readonly int m;
    public readonly int k;
    public readonly int n;
    public readonly int accumulate;

    public BatchedMatMulNtShader(ReadWriteBuffer<float> a, int aOff, ReadWriteBuffer<float> b, int bOff, ReadWriteBuffer<float> y, int yOff,
        int m, int k, int n, int accumulate)
    { this.a = a; this.aOff = aOff; this.b = b; this.bOff = bOff; this.y = y; this.yOff = yOff; this.m = m; this.k = k; this.n = n; this.accumulate = accumulate; }

    public void Execute()
    {
        int tx = GroupIds.X, ty = GroupIds.Y;
        int s = ThreadIds.Z, row = ThreadIds.Y, col = ThreadIds.X;
        int aRow = s * m + row;
        float sum = 0f;
        for (int k0 = 0; k0 < k; k0 += 16)
        {
            tileA[ty * 16 + tx] = row < m && k0 + tx < k ? a[aOff + aRow * k + k0 + tx] : 0f;
            tileB[ty * 16 + tx] = k0 + ty < k && col < n ? b[bOff + (s * n + col) * k + k0 + ty] : 0f;
            Hlsl.GroupMemoryBarrierWithGroupSync();
            for (int kk = 0; kk < 16; kk++)
                sum += tileA[ty * 16 + kk] * tileB[kk * 16 + tx];
            Hlsl.GroupMemoryBarrierWithGroupSync();
        }
        if (row >= m || col >= n) return;
        if (accumulate != 0) y[yOff + aRow * n + col] += sum;
        else y[yOff + aRow * n + col] = sum;
    }
}

/// <summary>Per slot s: y[s,m,n] = sum_k a[s,k,m]*b[s,k,n]. a:[S*K,M], b:[S*K,N], y:[S*M,N]</summary>
[ThreadGroupSize(16, 16, 1)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct BatchedMatMulTnShader : IComputeShader
{
    [GroupShared(256)]
    private static float[] tileA = null!;
    [GroupShared(256)]
    private static float[] tileB = null!;

    public readonly ReadWriteBuffer<float> a;
    public readonly int aOff;
    public readonly ReadWriteBuffer<float> b;
    public readonly int bOff;
    public readonly ReadWriteBuffer<float> y;
    public readonly int yOff;
    public readonly int m;
    public readonly int k;
    public readonly int n;
    public readonly int accumulate;

    public BatchedMatMulTnShader(ReadWriteBuffer<float> a, int aOff, ReadWriteBuffer<float> b, int bOff, ReadWriteBuffer<float> y, int yOff,
        int m, int k, int n, int accumulate)
    { this.a = a; this.aOff = aOff; this.b = b; this.bOff = bOff; this.y = y; this.yOff = yOff; this.m = m; this.k = k; this.n = n; this.accumulate = accumulate; }

    public void Execute()
    {
        int tx = GroupIds.X, ty = GroupIds.Y;
        int s = ThreadIds.Z, row = ThreadIds.Y, col = ThreadIds.X;
        int aRow = s * m + row;
        int m0 = row - ty; // block's first output row within the slot
        float sum = 0f;
        for (int k0 = 0; k0 < k; k0 += 16)
        {
            tileA[ty * 16 + tx] = k0 + ty < k && m0 + tx < m ? a[aOff + (s * k + k0 + ty) * m + m0 + tx] : 0f;
            tileB[ty * 16 + tx] = k0 + ty < k && col < n ? b[bOff + (s * k + k0 + ty) * n + col] : 0f;
            Hlsl.GroupMemoryBarrierWithGroupSync();
            for (int kk = 0; kk < 16; kk++)
                sum += tileA[kk * 16 + ty] * tileB[kk * 16 + tx];
            Hlsl.GroupMemoryBarrierWithGroupSync();
        }
        if (row >= m || col >= n) return;
        if (accumulate != 0) y[yOff + aRow * n + col] += sum;
        else y[yOff + aRow * n + col] = sum;
    }
}

// ---- Attention head packing ----------------------------------------------------------

/// <summary>dst[s*T + t, d] = src[seq*T + t, colBase + h*headDim + d], s = seq*nHeads + h. Flat over slots*T*headDim.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct PackHeadsShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> src;
    public readonly int srcOff;
    public readonly ReadWriteBuffer<float> dst;
    public readonly int dstOff;
    public readonly int srcCols;
    public readonly int colBase;
    public readonly int t;
    public readonly int headDim;
    public readonly int nHeads;
    public readonly int length;
    public readonly int stride;

    public PackHeadsShader(ReadWriteBuffer<float> src, int srcOff, ReadWriteBuffer<float> dst, int dstOff,
        int srcCols, int colBase, int t, int headDim, int nHeads, int length, int stride)
    { this.src = src; this.srcOff = srcOff; this.dst = dst; this.dstOff = dstOff; this.srcCols = srcCols; this.colBase = colBase; this.t = t; this.headDim = headDim; this.nHeads = nHeads; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i >= length) return;
        int slot = i / (t * headDim), r = i % (t * headDim), row = r / headDim, d = r % headDim;
        int seq = slot / nHeads, h = slot % nHeads;
        dst[dstOff + i] = src[srcOff + (seq * t + row) * srcCols + colBase + h * headDim + d];
    }
}

/// <summary>dst[seq*T + t, colBase + h*headDim + d] = src[s*T + t, d], s = seq*nHeads + h. Flat over slots*T*headDim.</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct UnpackHeadsShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> src;
    public readonly int srcOff;
    public readonly ReadWriteBuffer<float> dst;
    public readonly int dstOff;
    public readonly int dstCols;
    public readonly int colBase;
    public readonly int t;
    public readonly int headDim;
    public readonly int nHeads;
    public readonly int length;
    public readonly int stride;

    public UnpackHeadsShader(ReadWriteBuffer<float> src, int srcOff, ReadWriteBuffer<float> dst, int dstOff,
        int dstCols, int colBase, int t, int headDim, int nHeads, int length, int stride)
    { this.src = src; this.srcOff = srcOff; this.dst = dst; this.dstOff = dstOff; this.dstCols = dstCols; this.colBase = colBase; this.t = t; this.headDim = headDim; this.nHeads = nHeads; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i >= length) return;
        int slot = i / (t * headDim), r = i % (t * headDim), row = r / headDim, d = r % headDim;
        int seq = slot / nHeads, h = slot % nHeads;
        dst[dstOff + (seq * t + row) * dstCols + colBase + h * headDim + d] = src[srcOff + i];
    }
}

/// <summary>x[i] = value (flat fill; used for device-side zeroing).</summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct FillShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> x;
    public readonly int xOff;
    public readonly float value;
    public readonly int length;
    public readonly int stride;

    public FillShader(ReadWriteBuffer<float> x, int xOff, float value, int length, int stride)
    { this.x = x; this.xOff = xOff; this.value = value; this.length = length; this.stride = stride; }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i < length) x[xOff + i] = value;
    }
}

/// <summary>result[0] = sum_i x[i]^2. Single 256-thread group, grid-stride loop + tree reduction.</summary>
[ThreadGroupSize(256, 1, 1)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct SumSquaresShader : IComputeShader
{
    [GroupShared(256)]
    private static float[] red = null!;

    public readonly ReadWriteBuffer<float> x;
    public readonly int xOff;
    public readonly int length;
    public readonly ReadWriteBuffer<float> result;

    public SumSquaresShader(ReadWriteBuffer<float> x, int xOff, int length, ReadWriteBuffer<float> result)
    { this.x = x; this.xOff = xOff; this.length = length; this.result = result; }

    public void Execute()
    {
        int lane = GroupIds.X;
        float s = 0f;
        for (int i = lane; i < length; i += 256) { float v = x[xOff + i]; s += v * v; }
        red[lane] = s;
        Hlsl.GroupMemoryBarrierWithGroupSync();
        for (int t = 128; t > 0; t >>= 1)
        {
            if (lane < t) red[lane] += red[lane + t];
            Hlsl.GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0) result[0] = red[0];
    }
}

/// <summary>
/// One AdamW update step, elementwise: m/v EMAs with bias correction, decoupled
/// weight decay when decay != 0. Mirrors the host loop in ITensorBackend.AdamWStep.
/// </summary>
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct AdamWShader : IComputeShader
{
    public readonly ReadWriteBuffer<float> w;
    public readonly int wOff;
    public readonly ReadWriteBuffer<float> g;
    public readonly int gOff;
    public readonly ReadWriteBuffer<float> m;
    public readonly int mOff;
    public readonly ReadWriteBuffer<float> v;
    public readonly int vOff;
    public readonly float lr;
    public readonly float lrWd;
    public readonly float beta1;
    public readonly float oneBeta1;
    public readonly float beta2;
    public readonly float oneBeta2;
    public readonly float bc1;
    public readonly float bc2;
    public readonly float eps;
    public readonly int decay;
    public readonly int length;
    public readonly int stride;

    public AdamWShader(ReadWriteBuffer<float> w, int wOff, ReadWriteBuffer<float> g, int gOff,
        ReadWriteBuffer<float> m, int mOff, ReadWriteBuffer<float> v, int vOff,
        float lr, float lrWd, float beta1, float oneBeta1, float beta2, float oneBeta2,
        float bc1, float bc2, float eps, int decay, int length, int stride)
    {
        this.w = w; this.wOff = wOff; this.g = g; this.gOff = gOff;
        this.m = m; this.mOff = mOff; this.v = v; this.vOff = vOff;
        this.lr = lr; this.lrWd = lrWd;
        this.beta1 = beta1; this.oneBeta1 = oneBeta1; this.beta2 = beta2; this.oneBeta2 = oneBeta2;
        this.bc1 = bc1; this.bc2 = bc2; this.eps = eps; this.decay = decay;
        this.length = length; this.stride = stride;
    }

    public void Execute()
    {
        int i = ThreadIds.Y * stride + ThreadIds.X;
        if (i >= length) return;
        float gi = g[gOff + i];
        float mi = beta1 * m[mOff + i] + oneBeta1 * gi;
        float vi = beta2 * v[vOff + i] + oneBeta2 * gi * gi;
        m[mOff + i] = mi;
        v[vOff + i] = vi;
        float wi = w[wOff + i];
        if (decay != 0) wi -= lrWd * wi;
        w[wOff + i] = wi - lr * (mi / bc1) / (Hlsl.Sqrt(vi / bc2) + eps);
    }
}
