using LLM.Core.Tensor;

namespace LLM.Core.Tests;

/// <summary>
/// Forward-correctness tests against naive references plus finite-difference
/// gradient checks for every backward kernel in <see cref="CpuBackend"/>.
/// </summary>
public static class CpuBackendTests
{
    private static readonly CpuBackend B = new();

    // ---------- helpers -------------------------------------------------------

    private static float[] Rand(int n, Random rng, float std = 1f)
    {
        // Box-Muller N(0,1) samples, mirroring Tensor.FillNormal.
        var data = new float[n];
        for (int i = 0; i < n; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            data[i] = (float)(r * Math.Cos(2 * Math.PI * u2) * std);
            if (i + 1 < n)
                data[i + 1] = (float)(r * Math.Sin(2 * Math.PI * u2) * std);
        }
        return data;
    }

    /// <summary>Relative-or-absolute closeness: |a-e| &lt;= tol * max(1, |e|).</summary>
    private static void NearRel(float actual, float expected, float tol, string msg)
    {
        float err = Math.Abs(actual - expected);
        if (err > tol * Math.Max(1f, Math.Abs(expected)))
            Check.Fail($"expected {expected} (rel/abs tol {tol}), got {actual}: {msg}");
    }

    private static void SpanNearRel(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected, float tol, string msg)
    {
        for (int i = 0; i < actual.Length; i++)
            NearRel(actual[i], expected[i], tol, $"{msg} [{i}]");
    }

    /// <summary>Central-difference gradient of a scalar loss over buffer[i].</summary>
    private static float NumericGrad(float[] buffer, int i, Func<float> loss, float eps = 1e-3f)
    {
        float orig = buffer[i];
        buffer[i] = orig + eps;
        float hi = loss();
        buffer[i] = orig - eps;
        float lo = loss();
        buffer[i] = orig;
        return (hi - lo) / (2f * eps);
    }

    private static void CheckGrad(float[] analytic, float[] buffer, Func<float> loss, string msg, float tol = 1e-2f)
    {
        for (int i = 0; i < buffer.Length; i++)
            NearRel(analytic[i], NumericGrad(buffer, i, loss), tol, $"{msg} [{i}]");
    }

    // ---------- forward: matmul ------------------------------------------------

    private static float[] NaiveMatMul(float[] a, float[] b, int M, int K, int N, bool transA, bool transB)
    {
        // transA: a is [K,M] instead of [M,K]; transB: b is [N,K] instead of [K,N].
        var y = new float[M * N];
        for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            float s = 0f;
            for (int k = 0; k < K; k++)
            {
                float av = transA ? a[k * M + m] : a[m * K + k];
                float bv = transB ? b[n * K + k] : b[k * N + n];
                s += av * bv;
            }
            y[m * N + n] = s;
        }
        return y;
    }

    [Test]
    public static void MatMulNN_Forward()
    {
        var rng = new Random(1);
        const int M = 3, K = 5, N = 4;
        float[] a = Rand(M * K, rng), b = Rand(K * N, rng);
        var y = new float[M * N];
        B.MatMulNN(a, b, y, M, K, N);
        Check.SpanNear(y, NaiveMatMul(a, b, M, K, N, false, false), 1e-4f, "MatMulNN basic");

        var prior = Rand(M * N, rng);
        var expected = NaiveMatMul(a, b, M, K, N, false, false);
        for (int i = 0; i < expected.Length; i++) expected[i] += prior[i];
        prior.CopyTo(y, 0);
        B.MatMulNN(a, b, y, M, K, N, accumulate: true);
        Check.SpanNear(y, expected, 1e-4f, "MatMulNN accumulate");
    }

    [Test]
    public static void MatMulNT_Forward()
    {
        var rng = new Random(2);
        const int M = 3, K = 5, N = 4;
        float[] a = Rand(M * K, rng), b = Rand(N * K, rng);
        var y = new float[M * N];
        B.MatMulNT(a, b, y, M, K, N);
        Check.SpanNear(y, NaiveMatMul(a, b, M, K, N, false, true), 1e-4f, "MatMulNT basic");

        var expected = NaiveMatMul(a, b, M, K, N, false, true);
        for (int i = 0; i < y.Length; i++) y[i] = 2f;
        for (int i = 0; i < expected.Length; i++) expected[i] += 2f;
        B.MatMulNT(a, b, y, M, K, N, accumulate: true);
        Check.SpanNear(y, expected, 1e-4f, "MatMulNT accumulate");
    }

    [Test]
    public static void MatMulTN_Forward()
    {
        var rng = new Random(3);
        const int M = 3, K = 5, N = 4;
        float[] a = Rand(K * M, rng), b = Rand(K * N, rng);
        var y = new float[M * N];
        B.MatMulTN(a, b, y, M, K, N);
        Check.SpanNear(y, NaiveMatMul(a, b, M, K, N, true, false), 1e-4f, "MatMulTN basic");

        var expected = NaiveMatMul(a, b, M, K, N, true, false);
        for (int i = 0; i < y.Length; i++) y[i] = -1.5f;
        for (int i = 0; i < expected.Length; i++) expected[i] += -1.5f;
        B.MatMulTN(a, b, y, M, K, N, accumulate: true);
        Check.SpanNear(y, expected, 1e-4f, "MatMulTN accumulate");
    }

    // ---------- forward: elementwise / rows -------------------------------------

    [Test]
    public static void Transpose_MatchesNaive()
    {
        var rng = new Random(4);
        const int rows = 3, cols = 5;
        float[] x = Rand(rows * cols, rng);
        var output = new float[rows * cols];
        B.Transpose(x, output, rows, cols);
        for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            Check.Near(output[c * rows + r], x[r * cols + c], 0f, $"transpose [{r},{c}]");
    }

    [Test]
    public static void AddBiasSumRowsScaleAddInPlace()
    {
        var rng = new Random(5);
        const int rows = 3, cols = 4;
        float[] y = Rand(rows * cols, rng), bias = Rand(cols, rng);
        var expected = (float[])y.Clone();
        B.AddBias(y, bias, rows, cols);
        for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            Check.Near(y[r * cols + c], expected[r * cols + c] + bias[c], 1e-6f, "AddBias");

        // SumRows accumulates: seed with prior content.
        var dBias = Rand(cols, rng);
        var prior = (float[])dBias.Clone();
        B.SumRows(y, dBias, rows, cols);
        for (int c = 0; c < cols; c++)
        {
            float s = prior[c];
            for (int r = 0; r < rows; r++) s += y[r * cols + c];
            Check.Near(dBias[c], s, 1e-5f, "SumRows");
        }

        var z = Rand(rows * cols, rng);
        var zCopy = (float[])z.Clone();
        B.Scale(z, 2.5f);
        for (int i = 0; i < z.Length; i++) Check.Near(z[i], zCopy[i] * 2.5f, 1e-5f, "Scale");
        B.AddInPlace(z, zCopy);
        for (int i = 0; i < z.Length; i++) Check.Near(z[i], zCopy[i] * 3.5f, 1e-5f, "AddInPlace");
    }

    // ---------- forward: softmax / gelu -------------------------------------------

    [Test]
    public static void Softmax_RowsSumToOne()
    {
        const int rows = 3, cols = 6;
        var x = new float[rows * cols];
        var rng = new Random(6);
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 20 - 10);
        var raw = (float[])x.Clone();
        x[0] = 1000f; // extreme value: must not overflow
        B.SoftmaxForward(x, rows, cols);
        for (int r = 0; r < rows; r++)
        {
            float sum = 0f;
            for (int c = 0; c < cols; c++) sum += x[r * cols + c];
            Check.Near(sum, 1f, 1e-5f, $"softmax row {r} sums to 1");
        }
        // naive reference for the last row
        var expected = new float[cols];
        float eSum = 0f;
        for (int c = 0; c < cols; c++) { expected[c] = MathF.Exp(raw[2 * cols + c]); eSum += expected[c]; }
        for (int c = 0; c < cols; c++)
            Check.Near(x[2 * cols + c], expected[c] / eSum, 1e-5f, $"softmax value [2,{c}]");
        Check.True(x[0] > 0.99f, "softmax picks the 1000-logit");
    }

    [Test]
    public static void Gelu_KnownValues()
    {
        float[] x = { 0f, 1f, -1f, 2f, -2f };
        float[] expected = { 0f, 0.841192f, -0.158808f, 1.954598f, -0.045402f };
        var y = new float[x.Length];
        B.GeluForward(x, y);
        Check.SpanNear(y, expected, 1e-4f, "GELU known values");
    }

    // ---------- forward: embedding / causal mask / layernorm -----------------------

    [Test]
    public static void Embedding_Lookup()
    {
        const int V = 5, D = 3;
        float[] table = new float[V * D];
        for (int i = 0; i < table.Length; i++) table[i] = i;
        int[] indices = { 0, 2, 2, 4 };
        var output = new float[indices.Length * D];
        B.EmbeddingForward(table, indices, output, D);
        for (int t = 0; t < indices.Length; t++)
        for (int d = 0; d < D; d++)
            Check.Near(output[t * D + d], indices[t] * D + d, 0f, $"embedding [{t},{d}]");
    }

    [Test]
    public static void CausalMask_SetsUpperTriangle()
    {
        const int T = 4;
        var scores = new float[T * T];
        Array.Fill(scores, 1f);
        B.CausalMask(scores, T);
        for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++)
        {
            if (j > i) Check.True(float.IsNegativeInfinity(scores[i * T + j]), $"mask [{i},{j}] is -inf");
            else Check.Near(scores[i * T + j], 1f, 0f, $"mask [{i},{j}] untouched");
        }
    }

    [Test]
    public static void LayerNorm_NormalizesRows()
    {
        var rng = new Random(7);
        const int rows = 3, cols = 8;
        float[] x = Rand(rows * cols, rng, 3f);
        float[] w = Rand(cols, rng), b = Rand(cols, rng);
        var output = new float[rows * cols];
        var mean = new float[rows];
        var rstd = new float[rows];
        B.LayerNormForward(x, w, b, output, mean, rstd, rows, cols, 1e-5f);

        for (int r = 0; r < rows; r++)
        {
            // stats before affine: reconstruct xhat = (out - b)/w where w != 0,
            // and check cached mean/rstd against naive computation.
            float mu = 0f;
            for (int c = 0; c < cols; c++) mu += x[r * cols + c];
            mu /= cols;
            float var = 0f;
            for (int c = 0; c < cols; c++) { float d = x[r * cols + c] - mu; var += d * d; }
            var /= cols;
            Check.Near(mean[r], mu, 1e-4f, "cached mean");
            Check.Near(rstd[r], 1f / MathF.Sqrt(var + 1e-5f), 1e-4f, "cached rstd");
            for (int c = 0; c < cols; c++)
                Check.Near(output[r * cols + c], (x[r * cols + c] - mu) * rstd[r] * w[c] + b[c], 1e-3f, "layernorm value");
        }

        // with w=1, b=0 rows must be normalized
        var w1 = new float[cols]; Array.Fill(w1, 1f);
        var b0 = new float[cols];
        B.LayerNormForward(x, w1, b0, output, mean, rstd, rows, cols, 1e-5f);
        for (int r = 0; r < rows; r++)
        {
            float mu = 0f, var = 0f;
            for (int c = 0; c < cols; c++) mu += output[r * cols + c];
            mu /= cols;
            for (int c = 0; c < cols; c++) { float d = output[r * cols + c] - mu; var += d * d; }
            var /= cols;
            Check.Near(mu, 0f, 1e-4f, "normalized mean ~0");
            Check.Near(var, 1f, 1e-3f, "normalized var ~1");
        }
    }

    [Test]
    public static void CrossEntropy_MatchesLogSoftmax()
    {
        // hand-computed on tiny logits, including an ignored position
        const int T = 3, V = 4;
        float[] logits =
        {
            1f, 2f, 0.5f, -1f,
            0f, 0f, 0f, 0f,
            -2f, 3f, 1f, 0f,
        };
        int[] targets = { 1, 99, 0 };
        var probs = new float[T * V];
        float loss = B.CrossEntropyForward(logits, targets, probs, T, V, ignoreIndex: 99);

        double expected = 0.0;
        for (int t = 0; t < T; t++)
        {
            if (targets[t] == 99) continue;
            double max = double.NegativeInfinity;
            for (int v = 0; v < V; v++) max = Math.Max(max, logits[t * V + v]);
            double sum = 0;
            for (int v = 0; v < V; v++) sum += Math.Exp(logits[t * V + v] - max);
            expected += Math.Log(sum) + max - logits[t * V + targets[t]];
        }
        expected /= 2; // two non-ignored positions
        Check.Near(loss, (float)expected, 1e-4f, "mean NLL with ignoreIndex");

        for (int t = 0; t < T; t++)
        {
            float s = 0f;
            for (int v = 0; v < V; v++) s += probs[t * V + v];
            Check.Near(s, 1f, 1e-5f, $"probs row {t} sums to 1");
        }
        // uniform logits -> uniform probs
        for (int v = 0; v < V; v++) Check.Near(probs[1 * V + v], 0.25f, 1e-6f, "uniform probs");
    }

    // ---------- gradient checks -------------------------------------------------

    [Test]
    public static void MatMulBackward_GradCheck()
    {
        var rng = new Random(42);
        const int M = 3, K = 4, N = 2;
        float[] a = Rand(M * K, rng), b = Rand(K * N, rng), dY = Rand(M * N, rng);
        var y = new float[M * N];
        float Loss()
        {
            B.MatMulNN(a, b, y, M, K, N);
            float l = 0f;
            for (int i = 0; i < y.Length; i++) l += y[i] * dY[i];
            return l;
        }

        // analytic: dA = dY @ B^T (NT), dB = A^T @ dY (TN)
        var dA = new float[M * K];
        var dB = new float[K * N];
        B.MatMulNT(dY, b, dA, M, N, K);
        B.MatMulTN(a, dY, dB, K, M, N);

        CheckGrad(dA, a, Loss, "matmul dA");
        CheckGrad(dB, b, Loss, "matmul dB");
    }

    [Test]
    public static void LayerNormBackward_GradCheck()
    {
        var rng = new Random(42);
        const int rows = 3, cols = 4;
        const float eps = 1e-5f;
        float[] x = Rand(rows * cols, rng), w = Rand(cols, rng), b = Rand(cols, rng), dOut = Rand(rows * cols, rng);
        var output = new float[rows * cols];
        var mean = new float[rows];
        var rstd = new float[rows];
        float Loss()
        {
            B.LayerNormForward(x, w, b, output, mean, rstd, rows, cols, eps);
            float l = 0f;
            for (int i = 0; i < output.Length; i++) l += output[i] * dOut[i];
            return l;
        }

        var dX = new float[rows * cols];
        var dW = new float[cols];
        var dB = new float[cols];
        B.LayerNormForward(x, w, b, output, mean, rstd, rows, cols, eps); // populate stats
        B.LayerNormBackward(dOut, x, w, mean, rstd, dX, dW, dB, rows, cols);

        CheckGrad(dX, x, Loss, "layernorm dX");
        CheckGrad(dW, w, Loss, "layernorm dW");
        CheckGrad(dB, b, Loss, "layernorm dB");
    }

    [Test]
    public static void SoftmaxBackward_GradCheck()
    {
        var rng = new Random(42);
        const int rows = 3, cols = 4;
        float[] x = Rand(rows * cols, rng), dOut = Rand(rows * cols, rng);
        var sm = new float[rows * cols];
        float Loss()
        {
            x.CopyTo(sm, 0);
            B.SoftmaxForward(sm, rows, cols);
            float l = 0f;
            for (int i = 0; i < sm.Length; i++) l += sm[i] * dOut[i];
            return l;
        }

        x.CopyTo(sm, 0);
        B.SoftmaxForward(sm, rows, cols);
        var dX = new float[rows * cols];
        B.SoftmaxBackward(dOut, sm, dX, rows, cols);

        CheckGrad(dX, x, Loss, "softmax dX");
    }

    [Test]
    public static void GeluBackward_GradCheck()
    {
        var rng = new Random(42);
        const int n = 12;
        float[] x = Rand(n, rng), dOut = Rand(n, rng);
        var y = new float[n];
        float Loss()
        {
            B.GeluForward(x, y);
            float l = 0f;
            for (int i = 0; i < n; i++) l += y[i] * dOut[i];
            return l;
        }

        var dX = new float[n];
        B.GeluBackward(dOut, x, dX);
        CheckGrad(dX, x, Loss, "gelu dX");
    }

    [Test]
    public static void EmbeddingBackward_GradCheck()
    {
        var rng = new Random(42);
        const int V = 5, D = 3;
        float[] table = Rand(V * D, rng);
        int[] indices = { 0, 2, 2, 4 };
        float[] dOut = Rand(indices.Length * D, rng);
        var output = new float[indices.Length * D];
        float Loss()
        {
            B.EmbeddingForward(table, indices, output, D);
            float l = 0f;
            for (int i = 0; i < output.Length; i++) l += output[i] * dOut[i];
            return l;
        }

        var dTable = new float[V * D]; // accumulates into zeroed buffer
        B.EmbeddingBackward(dOut, indices, dTable, D);

        CheckGrad(dTable, table, Loss, "embedding dTable");
        // repeated index 2 must accumulate contributions from both positions
        for (int d = 0; d < D; d++)
            Check.Near(dTable[2 * D + d], dOut[1 * D + d] + dOut[2 * D + d], 1e-6f, "embedding scatter-add");
        // untouched rows stay zero
        Check.Near(dTable[1 * D], 0f, 0f, "untouched embedding row stays zero");
    }

    [Test]
    public static void CrossEntropyBackward_GradCheck()
    {
        var rng = new Random(42);
        const int T = 3, V = 4;
        float[] logits = Rand(T * V, rng);
        int[] targets = { 1, 99, 0 }; // one ignored position
        var probs = new float[T * V];
        float Loss() => B.CrossEntropyForward(logits, targets, probs, T, V, ignoreIndex: 99);

        B.CrossEntropyForward(logits, targets, probs, T, V, ignoreIndex: 99);
        var dLogits = new float[T * V];
        B.CrossEntropyBackward(probs, targets, dLogits, T, V, ignoreIndex: 99);

        CheckGrad(dLogits, logits, Loss, "cross-entropy dLogits");
        // ignored row gets exactly zero gradient
        for (int v = 0; v < V; v++)
            Check.Near(dLogits[1 * V + v], 0f, 0f, "ignored row zero gradient");
    }
}
