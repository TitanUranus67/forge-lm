namespace LLM.Core.Tests
{
    using LLM.Core.Model;
    using LLM.Core.Tensor;
    using LLM.Core.Tensor.Gpu;
    using Tensor = LLM.Core.Tensor.Tensor;

    /// <summary>
    /// Numerical validation of <see cref="GpuBackend"/> against <see cref="CpuBackend"/>:
    /// every kernel's forward and backward outputs are compared element-wise (GPU float
    /// reduction order differs, so tolerances are relative-or-absolute, never bitwise),
    /// plus residency/sync-model tests, a full-model batched finite-difference gradient
    /// check, and a batched overfit smoke test — all running end-to-end on the GPU.
    /// Every test skips cleanly (and passes) when no D3D12 device is available.
    /// </summary>
    public static class GpuBackendTests
    {
        private static readonly CpuBackend Cpu = new();
        private static GpuBackend? _gpu;
        private static bool _probed;

        private const float FwdTol = 1e-4f;  // forward kernels
        private const float BwdTol = 1e-2f;  // backward kernels (reduction order differs)

        private static GpuBackend? Gpu
        {
            get
            {
                if (!_probed)
                {
                    _probed = true;
                    try
                    {
                        if (GpuBackend.IsAvailable) _gpu = new GpuBackend();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"    GPU probe failed: {ex.GetType().Name}: {ex.Message}");
                    }
                }
                return _gpu;
            }
        }

        /// <summary>Reports SKIP and returns true when there is no D3D12 device.</summary>
        private static bool Skip()
        {
            if (Gpu is not null) return false;
            Console.WriteLine("    SKIP: no D3D12 device available");
            return true;
        }

        // ---------- helpers -------------------------------------------------------

        private static float[] Rand(int n, Random rng, float std = 1f)
        {
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
            if (float.IsNaN(err) || err > tol * Math.Max(1f, Math.Abs(expected)))
                Check.Fail($"expected {expected} (rel/abs tol {tol}), got {actual}: {msg}");
        }

        private static void SpanNearRel(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected, float tol, string msg)
        {
            Check.True(actual.Length == expected.Length, $"{msg} (length {actual.Length} vs {expected.Length})");
            if (actual.Length != expected.Length) return;
            float worst = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                float err = Math.Abs(actual[i] - expected[i]);
                worst = Math.Max(worst, err / Math.Max(1f, Math.Abs(expected[i])));
            }
            if (worst > tol) Check.Fail($"max rel/abs diff {worst:G4} > {tol}: {msg}");
        }

        /// <summary>Fresh tensor holding a copy of <paramref name="data"/>.</summary>
        private static Tensor Dup(float[] data, params int[] shape) => new((float[])data.Clone(), shape);

        // ---------- matmul ---------------------------------------------------------

        [Test]
        public static void MatMul_MatchesCpu()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(101);

            foreach ((int M, int K, int N) in new[] { (37, 53, 29), (128, 256, 192) })
            {
                float[] a = Rand(M * K, rng), bN = Rand(K * N, rng), bT = Rand(N * K, rng), prior = Rand(M * N, rng);

                foreach (bool acc in new[] { false, true })
                {
                    // NN: a [M,K] @ b [K,N]
                    var yCpu = Dup(prior, M, N); var yGpu = Dup(prior, M, N);
                    Cpu.MatMulNN(Dup(a, M, K), Dup(bN, K, N), yCpu, M, K, N, acc);
                    gpu.MatMulNN(Dup(a, M, K), Dup(bN, K, N), yGpu, M, K, N, acc);
                    gpu.EnsureHostCurrent(yGpu);
                    SpanNearRel(yGpu.Data, yCpu.Data, FwdTol, $"MatMulNN {M}x{K}x{N} acc={acc}");

                    // NT: a [M,K] @ b [N,K]^T
                    Cpu.MatMulNT(Dup(a, M, K), Dup(bT, N, K), yCpu, M, K, N, acc);
                    gpu.MatMulNT(Dup(a, M, K), Dup(bT, N, K), yGpu, M, K, N, acc);
                    gpu.EnsureHostCurrent(yGpu);
                    SpanNearRel(yGpu.Data, yCpu.Data, FwdTol, $"MatMulNT {M}x{K}x{N} acc={acc}");

                    // TN: a [K,M]^T @ b [K,N]
                    Cpu.MatMulTN(Dup(a, K, M), Dup(bN, K, N), yCpu, M, K, N, acc);
                    gpu.MatMulTN(Dup(a, K, M), Dup(bN, K, N), yGpu, M, K, N, acc);
                    gpu.EnsureHostCurrent(yGpu);
                    SpanNearRel(yGpu.Data, yCpu.Data, FwdTol, $"MatMulTN {M}x{K}x{N} acc={acc}");
                }
            }
        }

        // ---------- elementwise / rows ------------------------------------------------

        [Test]
        public static void Elementwise_MatchesCpu()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(102);
            const int rows = 33, cols = 65;

            float[] y = Rand(rows * cols, rng), bias = Rand(cols, rng);
            var yCpu = Dup(y, rows, cols); var yGpu = Dup(y, rows, cols);
            Cpu.AddBias(yCpu, Dup(bias, cols), rows, cols);
            gpu.AddBias(yGpu, Dup(bias, cols), rows, cols);
            gpu.EnsureHostCurrent(yGpu);
            SpanNearRel(yGpu.Data, yCpu.Data, FwdTol, "AddBias");

            float[] db0 = Rand(cols, rng);
            var dbCpu = Dup(db0, cols); var dbGpu = Dup(db0, cols);
            Cpu.SumRows(Dup(y, rows, cols), dbCpu, rows, cols);
            gpu.SumRows(Dup(y, rows, cols), dbGpu, rows, cols);
            gpu.EnsureHostCurrent(dbGpu);
            SpanNearRel(dbGpu.Data, dbCpu.Data, BwdTol, "SumRows accumulate");

            float[] src = Rand(rows * cols, rng);
            yCpu = Dup(y, rows, cols); yGpu = Dup(y, rows, cols);
            Cpu.AddInPlace(yCpu, Dup(src, rows, cols));
            gpu.AddInPlace(yGpu, Dup(src, rows, cols));
            gpu.EnsureHostCurrent(yGpu);
            SpanNearRel(yGpu.Data, yCpu.Data, FwdTol, "AddInPlace");

            yCpu = Dup(y, rows, cols); yGpu = Dup(y, rows, cols);
            Cpu.Scale(yCpu, -1.75f);
            gpu.Scale(yGpu, -1.75f);
            gpu.EnsureHostCurrent(yGpu);
            SpanNearRel(yGpu.Data, yCpu.Data, FwdTol, "Scale");

            var trCpu = new Tensor(cols, rows); var trGpu = new Tensor(cols, rows);
            Cpu.Transpose(Dup(y, rows, cols), trCpu, rows, cols);
            gpu.Transpose(Dup(y, rows, cols), trGpu, rows, cols);
            gpu.EnsureHostCurrent(trGpu);
            SpanNearRel(trGpu.Data, trCpu.Data, 0f, "Transpose");

            var cpCpu = new Tensor(rows, cols); var cpGpu = new Tensor(rows, cols);
            Cpu.Copy(Dup(y, rows, cols), cpCpu);
            gpu.Copy(Dup(y, rows, cols), cpGpu);
            gpu.EnsureHostCurrent(cpGpu);
            SpanNearRel(cpGpu.Data, cpCpu.Data, 0f, "Copy");

            // block copy: extract a 17x23 block from [rows,cols] into the middle of a zeroed [12, 40]
            var blkCpu = new Tensor(12, 40); var blkGpu = new Tensor(12, 40);
            Cpu.CopyBlock(Dup(y, rows, cols), blkCpu, 5, 11, 3, 7, 7, 23);
            gpu.CopyBlock(Dup(y, rows, cols), blkGpu, 5, 11, 3, 7, 7, 23);
            gpu.EnsureHostCurrent(blkGpu);
            SpanNearRel(blkGpu.Data, blkCpu.Data, 0f, "CopyBlock extract");
            // ...and scatter one back over part of another tensor (partial write must preserve the rest)
            var intoCpu = Dup(y, rows, cols); var intoGpu = Dup(y, rows, cols);
            Cpu.CopyBlock(Dup(blkCpu.Data, 12, 40), intoCpu, 3, 7, 9, 30, 7, 23);
            gpu.CopyBlock(Dup(blkCpu.Data, 12, 40), intoGpu, 3, 7, 9, 30, 7, 23);
            gpu.EnsureHostCurrent(intoGpu);
            SpanNearRel(intoGpu.Data, intoCpu.Data, 0f, "CopyBlock scatter preserves surroundings");
        }

        [Test]
        public static void FlatDispatch_BeyondOneDimensionalLimit()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            // 64 threads/group * 65535 groups = 4,194,240 max 1-D dispatch; exceed it.
            const int n = 5_000_123;
            var rng = new Random(103);
            float[] a = Rand(n, rng), b = Rand(n, rng);
            var aCpu = Dup(a, n); var aGpu = Dup(a, n);
            Cpu.AddInPlace(aCpu, Dup(b, n));
            gpu.AddInPlace(aGpu, Dup(b, n));
            gpu.EnsureHostCurrent(aGpu);
            SpanNearRel(aGpu.Data, aCpu.Data, FwdTol, "AddInPlace with 2-D split dispatch");
        }

        // ---------- layernorm / softmax / gelu -----------------------------------------

        [Test]
        public static void LayerNorm_MatchesCpu()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(104);
            const int rows = 33, cols = 64;
            const float eps = 1e-5f;

            float[] x = Rand(rows * cols, rng), w = Rand(cols, rng), b = Rand(cols, rng);
            var outCpu = new Tensor(rows, cols); var outGpu = new Tensor(rows, cols);
            var meanCpu = new Tensor(rows); var meanGpu = new Tensor(rows);
            var rstdCpu = new Tensor(rows); var rstdGpu = new Tensor(rows);
            Cpu.LayerNormForward(Dup(x, rows, cols), Dup(w, cols), Dup(b, cols), outCpu, meanCpu, rstdCpu, rows, cols, eps);
            gpu.LayerNormForward(Dup(x, rows, cols), Dup(w, cols), Dup(b, cols), outGpu, meanGpu, rstdGpu, rows, cols, eps);
            gpu.EnsureHostCurrent(outGpu); gpu.EnsureHostCurrent(meanGpu); gpu.EnsureHostCurrent(rstdGpu);
            SpanNearRel(outGpu.Data, outCpu.Data, FwdTol, "LayerNormForward out");
            SpanNearRel(meanGpu.Data, meanCpu.Data, FwdTol, "LayerNormForward mean");
            SpanNearRel(rstdGpu.Data, rstdCpu.Data, FwdTol, "LayerNormForward rstd");

            float[] dOut = Rand(rows * cols, rng), dw0 = Rand(cols, rng), db0 = Rand(cols, rng);
            var dxCpu = new Tensor(rows, cols); var dxGpu = new Tensor(rows, cols);
            var dwCpu = Dup(dw0, cols); var dwGpu = Dup(dw0, cols);
            var dbCpu = Dup(db0, cols); var dbGpu = Dup(db0, cols);
            Cpu.LayerNormBackward(Dup(dOut, rows, cols), Dup(x, rows, cols), Dup(w, cols),
                Dup(meanCpu.Data, rows), Dup(rstdCpu.Data, rows), dxCpu, dwCpu, dbCpu, rows, cols);
            gpu.LayerNormBackward(Dup(dOut, rows, cols), Dup(x, rows, cols), Dup(w, cols),
                Dup(meanCpu.Data, rows), Dup(rstdCpu.Data, rows), dxGpu, dwGpu, dbGpu, rows, cols);
            gpu.EnsureHostCurrent(dxGpu); gpu.EnsureHostCurrent(dwGpu); gpu.EnsureHostCurrent(dbGpu);
            SpanNearRel(dxGpu.Data, dxCpu.Data, BwdTol, "LayerNormBackward dX");
            SpanNearRel(dwGpu.Data, dwCpu.Data, BwdTol, "LayerNormBackward dW accumulate");
            SpanNearRel(dbGpu.Data, dbCpu.Data, BwdTol, "LayerNormBackward dB accumulate");
        }

        [Test]
        public static void Softmax_MatchesCpu()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(105);
            const int rows = 29, cols = 257; // odd width, larger than one warp

            float[] x = Rand(rows * cols, rng, 3f); // wide range stresses the max-shift
            var xCpu = Dup(x, rows, cols); var xGpu = Dup(x, rows, cols);
            Cpu.SoftmaxForward(xCpu, rows, cols);
            gpu.SoftmaxForward(xGpu, rows, cols);
            gpu.EnsureHostCurrent(xGpu);
            SpanNearRel(xGpu.Data, xCpu.Data, FwdTol, "SoftmaxForward");

            float[] dOut = Rand(rows * cols, rng);
            var dxCpu = new Tensor(rows, cols); var dxGpu = new Tensor(rows, cols);
            Cpu.SoftmaxBackward(Dup(dOut, rows, cols), Dup(xCpu.Data, rows, cols), dxCpu, rows, cols);
            gpu.SoftmaxBackward(Dup(dOut, rows, cols), Dup(xCpu.Data, rows, cols), dxGpu, rows, cols);
            gpu.EnsureHostCurrent(dxGpu);
            SpanNearRel(dxGpu.Data, dxCpu.Data, BwdTol, "SoftmaxBackward");
        }

        [Test]
        public static void Gelu_MatchesCpu()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(106);
            const int n = 10_003;
            float[] x = Rand(n, rng, 2f);

            var outCpu = new Tensor(n); var outGpu = new Tensor(n);
            Cpu.GeluForward(Dup(x, n), outCpu);
            gpu.GeluForward(Dup(x, n), outGpu);
            gpu.EnsureHostCurrent(outGpu);
            SpanNearRel(outGpu.Data, outCpu.Data, FwdTol, "GeluForward");

            float[] dOut = Rand(n, rng);
            var dxCpu = new Tensor(n); var dxGpu = new Tensor(n);
            Cpu.GeluBackward(Dup(dOut, n), Dup(x, n), dxCpu);
            gpu.GeluBackward(Dup(dOut, n), Dup(x, n), dxGpu);
            gpu.EnsureHostCurrent(dxGpu);
            SpanNearRel(dxGpu.Data, dxCpu.Data, FwdTol, "GeluBackward");
        }

        // ---------- embedding / causal mask --------------------------------------------

        [Test]
        public static void Embedding_MatchesCpu()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(107);
            const int V = 51, D = 37, Tn = 96;

            float[] table = Rand(V * D, rng);
            int[] idx = new int[Tn];
            for (int i = 0; i < Tn; i++) idx[i] = rng.Next(V / 3); // heavy duplication exercises CAS atomics

            var outCpu = new Tensor(Tn, D); var outGpu = new Tensor(Tn, D);
            Cpu.EmbeddingForward(Dup(table, V, D), idx, outCpu, D);
            gpu.EmbeddingForward(Dup(table, V, D), idx, outGpu, D);
            gpu.EnsureHostCurrent(outGpu);
            SpanNearRel(outGpu.Data, outCpu.Data, 0f, "EmbeddingForward");

            float[] dOut = Rand(Tn * D, rng), dt0 = Rand(V * D, rng);
            var dtCpu = Dup(dt0, V, D); var dtGpu = Dup(dt0, V, D);
            Cpu.EmbeddingBackward(Dup(dOut, Tn, D), idx, dtCpu, D);
            gpu.EmbeddingBackward(Dup(dOut, Tn, D), idx, dtGpu, D);
            gpu.EnsureHostCurrent(dtGpu);
            SpanNearRel(dtGpu.Data, dtCpu.Data, BwdTol, "EmbeddingBackward scatter-add accumulate");
        }

        [Test]
        public static void CausalMask_MatchesCpu()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(108);
            const int T = 47;
            float[] s = Rand(T * T * 3, rng); // three packed [T,T] blocks (batched attention)
            var sCpu = Dup(s, 3 * T, T); var sGpu = Dup(s, 3 * T, T);
            Cpu.CausalMask(sCpu, T);
            gpu.CausalMask(sGpu, T);
            gpu.EnsureHostCurrent(sGpu);
            for (int blk = 0; blk < 3; blk++)
                for (int i = 0; i < T; i++)
                    for (int j = 0; j < T; j++)
                    {
                        int f = blk * T * T + i * T + j;
                        if (j > i) Check.True(float.IsNegativeInfinity(sGpu.Data[f]), $"CausalMask block {blk} [{i},{j}] = -inf");
                        else NearRel(sGpu.Data[f], sCpu.Data[f], 0f, $"CausalMask block {blk} keeps [{i},{j}]");
                    }
        }

        // ---------- cross-entropy -------------------------------------------------------

        [Test]
        public static void CrossEntropy_MatchesCpu()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(109);
            const int T = 61, V = 389, ignore = -1;

            float[] logits = Rand(T * V, rng, 2f);
            int[] targets = new int[T];
            for (int i = 0; i < T; i++) targets[i] = rng.Next(V);
            targets[7] = ignore; targets[58] = ignore; // exercise ignored positions

            var pCpu = new Tensor(T, V); var pGpu = new Tensor(T, V);
            float lossCpu = Cpu.CrossEntropyForward(Dup(logits, T, V), targets, pCpu, T, V, ignore);
            float lossGpu = gpu.CrossEntropyForward(Dup(logits, T, V), targets, pGpu, T, V, ignore);
            gpu.EnsureHostCurrent(pGpu);
            SpanNearRel(pGpu.Data, pCpu.Data, FwdTol, "CrossEntropyForward probs");
            NearRel(lossGpu, lossCpu, FwdTol, "CrossEntropyForward mean loss");

            var dlCpu = new Tensor(T, V); var dlGpu = new Tensor(T, V);
            Cpu.CrossEntropyBackward(Dup(pCpu.Data, T, V), targets, dlCpu, T, V, ignore);
            gpu.CrossEntropyBackward(Dup(pCpu.Data, T, V), targets, dlGpu, T, V, ignore);
            gpu.EnsureHostCurrent(dlGpu);
            SpanNearRel(dlGpu.Data, dlCpu.Data, BwdTol, "CrossEntropyBackward");

            // all-ignored edge case: zero gradient, zero loss
            int[] allIgnored = Enumerable.Repeat(ignore, T).ToArray();
            var pGpu2 = new Tensor(T, V);
            float loss2 = gpu.CrossEntropyForward(Dup(logits, T, V), allIgnored, pGpu2, T, V, ignore);
            var dlGpu2 = new Tensor(T, V);
            gpu.CrossEntropyBackward(Dup(logits, T, V), allIgnored, dlGpu2, T, V, ignore);
            gpu.EnsureHostCurrent(dlGpu2);
            NearRel(loss2, 0f, 0f, "CrossEntropyForward all-ignored loss");
            foreach (float g in dlGpu2.Data) NearRel(g, 0f, 0f, "CrossEntropyBackward all-ignored zeroes");
        }

        [Test]
        public static void CrossEntropy_AliasedInPlaceMatchesCpu()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(110);
            const int T = 61, V = 389, ignore = -1;

            float[] logits = Rand(T * V, rng, 2f);
            int[] targets = new int[T];
            for (int i = 0; i < T; i++) targets[i] = rng.Next(V);
            targets[7] = ignore; targets[58] = ignore;

            var pCpu = new Tensor(T, V);
            float lossCpu = Cpu.CrossEntropyForward(Dup(logits, T, V), targets, pCpu, T, V, ignore);
            var dlCpu = new Tensor(T, V);
            Cpu.CrossEntropyBackward(Dup(pCpu.Data, T, V), targets, dlCpu, T, V, ignore);

            // probs aliased onto the logits buffer, then dLogits aliased onto that
            var buf = Dup(logits, T, V);
            float lossGpu = gpu.CrossEntropyForward(buf, targets, buf, T, V, ignore);
            gpu.EnsureHostCurrent(buf);
            NearRel(lossGpu, lossCpu, FwdTol, "aliased CE forward loss");
            SpanNearRel(buf.Data, pCpu.Data, FwdTol, "aliased CE forward probs");

            gpu.CrossEntropyBackward(buf, targets, buf, T, V, ignore);
            gpu.EnsureHostCurrent(buf);
            SpanNearRel(buf.Data, dlCpu.Data, BwdTol, "aliased CE backward dLogits");
        }

        [Test]
        public static void BatchedAttentionKernels_MatchCpu()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(113);
            const int batch = 3, T = 7, H = 2, hd = 5, D = H * hd;

            // head packing round trip
            float[] src = Rand(batch * T * 3 * D, rng);
            var pkCpu = new Tensor(batch * H * T, hd); var pkGpu = new Tensor(batch * H * T, hd);
            Cpu.PackHeads(Dup(src, batch * T, 3 * D), pkCpu, batch, T, H, hd, D);
            gpu.PackHeads(Dup(src, batch * T, 3 * D), pkGpu, batch, T, H, hd, D);
            gpu.EnsureHostCurrent(pkGpu);
            SpanNearRel(pkGpu.Data, pkCpu.Data, 0f, "PackHeads");

            var unCpu = new Tensor(batch * T, 3 * D); var unGpu = new Tensor(batch * T, 3 * D);
            Cpu.UnpackHeads(Dup(pkCpu.Data, batch * H * T, hd), unCpu, batch, T, H, hd, 2 * D);
            gpu.UnpackHeads(Dup(pkCpu.Data, batch * H * T, hd), unGpu, batch, T, H, hd, 2 * D);
            gpu.EnsureHostCurrent(unGpu);
            for (int i = 0; i < unCpu.Length; i++)
            {
                int d = i % (3 * D);
                bool written = d >= 2 * D;
                NearRel(unGpu.Data[i], written ? unCpu.Data[i] : 0f, 0f, $"UnpackHeads [{i}]");
            }

            // batched matmuls, odd sizes + accumulate
            int slots = batch * H, M = T, K = hd, N = T;
            float[] a = Rand(slots * M * K, rng), bN = Rand(slots * K * N, rng), bT = Rand(slots * N * K, rng), prior = Rand(slots * M * N, rng);
            foreach (bool acc in new[] { false, true })
            {
                var yCpu = Dup(prior, slots * M, N); var yGpu = Dup(prior, slots * M, N);
                Cpu.BatchedMatMulNN(Dup(a, slots * M, K), Dup(bN, slots * K, N), yCpu, slots, M, K, N, acc);
                gpu.BatchedMatMulNN(Dup(a, slots * M, K), Dup(bN, slots * K, N), yGpu, slots, M, K, N, acc);
                gpu.EnsureHostCurrent(yGpu);
                SpanNearRel(yGpu.Data, yCpu.Data, FwdTol, $"BatchedMatMulNN acc={acc}");

                Cpu.BatchedMatMulNT(Dup(a, slots * M, K), Dup(bT, slots * N, K), yCpu, slots, M, K, N, acc);
                gpu.BatchedMatMulNT(Dup(a, slots * M, K), Dup(bT, slots * N, K), yGpu, slots, M, K, N, acc);
                gpu.EnsureHostCurrent(yGpu);
                SpanNearRel(yGpu.Data, yCpu.Data, FwdTol, $"BatchedMatMulNT acc={acc}");

                Cpu.BatchedMatMulTN(Dup(a, slots * K, M), Dup(bN, slots * K, N), yCpu, slots, M, K, N, acc);
                gpu.BatchedMatMulTN(Dup(a, slots * K, M), Dup(bN, slots * K, N), yGpu, slots, M, K, N, acc);
                gpu.EnsureHostCurrent(yGpu);
                SpanNearRel(yGpu.Data, yCpu.Data, FwdTol, $"BatchedMatMulTN acc={acc}");
            }
        }

        // ---------- residency / sync model -----------------------------------------------

        [Test]
        public static void Residency_DirectHostWriteWithoutInvalidateIsStale()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(110);
            const int n = 128;
            float[] a = Rand(n, rng), b = Rand(n, rng);

            var t = Dup(a, n);
            gpu.AddInPlace(t, Dup(b, n)); // device now holds a+b; t.Data still holds a
            t.Data[0] = 12345f;         // direct host write, NO InvalidateDeviceCache
            gpu.Scale(t, 2f);           // reads the device copy: the host write is invisible
            gpu.EnsureHostCurrent(t);
            NearRel(t.Data[0], (a[0] + b[0]) * 2f, 1e-3f, "host write without invalidate is lost (documented stale behavior)");
            NearRel(t.Data[1], (a[1] + b[1]) * 2f, 1e-3f, "untouched element scaled on device");
        }

        [Test]
        public static void Residency_InvalidateRefreshesDeviceCopy()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(111);
            const int n = 128;
            float[] a = Rand(n, rng), b = Rand(n, rng);

            var t = Dup(a, n);
            gpu.AddInPlace(t, Dup(b, n)); // device holds a+b
            t.Data[0] = 12345f;
            gpu.InvalidateDeviceCache(t); // host is now authoritative
            gpu.Scale(t, 2f);
            gpu.EnsureHostCurrent(t);
            // device re-uploaded from host: element 0 sees the raw host value 12345 (a+b lost),
            // all other elements see the STALE host values a[i] (documented: invalidate is all-or-nothing)
            NearRel(t.Data[0], 12345f * 2f, 1e-1f, "host write visible after invalidate");
            NearRel(t.Data[1], a[1] * 2f, 1e-3f, "whole buffer re-uploaded from host after invalidate");
        }

        [Test]
        public static void EnsureHostCurrent_RoundTripsDeviceResults()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var rng = new Random(112);
            const int n = 256;
            float[] a = Rand(n, rng), b = Rand(n, rng);

            var t = Dup(a, n);
            gpu.AddInPlace(t, Dup(b, n));
            gpu.EnsureHostCurrent(t);
            for (int i = 0; i < n; i++) NearRel(t.Data[i], a[i] + b[i], 1e-5f, $"downloaded value [{i}]");
            float[] snapshot = (float[])t.Data.Clone();
            gpu.EnsureHostCurrent(t); // second call is a no-op, must not corrupt anything
            Check.SpanNear(t.Data, snapshot, 0f, "second EnsureHostCurrent is a no-op");

            var hostOnly = Dup(a, n);
            gpu.EnsureHostCurrent(hostOnly); // never touched by a kernel: no-op, no crash
            Check.SpanNear(hostOnly.Data, a, 0f, "EnsureHostCurrent on host-only tensor is a no-op");
            gpu.InvalidateDeviceCache(hostOnly); // also a no-op
            Check.SpanNear(hostOnly.Data, a, 0f, "InvalidateDeviceCache on host-only tensor is a no-op");
        }

        // ---------- full-model validation -------------------------------------------------

        private static ModelConfig Tiny => new(VocabSize: 12, ContextLength: 6, DModel: 8, NLayers: 1, NHeads: 2);

        [Test]
        public static void Model_ForwardMatchesCpu()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var config = Tiny;
            int[] tokens = { 3, 1, 4, 1, 5, 2 };
            Tensor cpuLogits = new GptModel(config, Cpu, new Random(42)).Forward(tokens);
            Tensor gpuLogits = new GptModel(config, gpu, new Random(42)).Forward(tokens);
            SpanNearRel(gpuLogits.Data, cpuLogits.Data, 1e-3f, "full-model forward logits GPU vs CPU");
        }

        [Test]
        public static void Model_GradientCheck_Batched()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var config = Tiny;
            var model = new GptModel(config, gpu, new Random(123));
            const int batch = 2;
            int[] inputs = { 1, 2, 3, 4, 0, 5, 5, 4, 3, 2, 1, 0 };
            int[] targets = { 2, 3, 4, 5, 1, 0, 4, 3, 2, 1, 0, 5 };
            const float eps = 1e-3f, tol = 2e-2f;

            model.Params.ZeroGrads();
            float loss = model.ForwardBackward(inputs, targets, batch);
            Check.True(loss > 0f && float.IsFinite(loss), "loss is positive and finite");

            // Snapshot the analytic grads once: LossFn below calls ForwardBackward, which
            // would keep accumulating into the same gradient tensors.
            var analytic = new Dictionary<string, float[]>();
            foreach (string name in model.Params.Names)
            {
                Tensor g = model.Params.Grad(name);
                gpu.EnsureHostCurrent(g); // grads were accumulated on device
                analytic[name] = (float[])g.Data.Clone();
            }

            float LossFn()
            {
                // Batched loss: ForwardBackward returns the mean cross-entropy over all B*T
                // positions (its grad accumulation is irrelevant — see the snapshot above).
                return model.ForwardBackward(inputs, targets, batch);
            }

            float maxErr = 0f;
            string worst = "";
            foreach (string name in model.Params.Names)
            {
                Tensor w = model.Params.Weight(name);
                float[] g = analytic[name];
                for (int i = 0; i < w.Length; i++)
                {
                    float orig = w.Data[i];
                    w.Data[i] = orig + eps;
                    gpu.InvalidateDeviceCache(w); // direct host write
                    float hi = LossFn();
                    w.Data[i] = orig - eps;
                    gpu.InvalidateDeviceCache(w);
                    float lo = LossFn();
                    w.Data[i] = orig;
                    gpu.InvalidateDeviceCache(w);
                    float numeric = (hi - lo) / (2f * eps);
                    float err = Math.Abs(numeric - g[i]) / Math.Max(1f, Math.Abs(numeric));
                    if (err > maxErr) { maxErr = err; worst = $"{name}[{i}] numeric={numeric:G6} analytic={g[i]:G6}"; }
                }
            }
            Check.True(maxErr < tol, $"max grad error {maxErr:G4} < {tol} (worst: {worst})");
            Console.WriteLine($"    GPU batched grad-check max rel/abs err {maxErr:G4} (worst: {worst})");
        }

        [Test]
        public static void Overfit_SmallBatchLossDrops()
        {
            if (Skip()) return;
            GpuBackend gpu = Gpu!;
            var config = Tiny;
            var model = new GptModel(config, gpu, new Random(5));
            const int batch = 2;
            int[] inputs = { 0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0 };
            int[] targets = { 1, 2, 3, 4, 5, 0, 4, 3, 2, 1, 0, 5 };
            const float lr = 0.05f;

            model.Params.ZeroGrads();
            float initial = model.ForwardBackward(inputs, targets, batch);
            for (int step = 0; step < 200; step++)
            {
                model.Params.ZeroGrads();
                model.ForwardBackward(inputs, targets, batch);
                foreach (string name in model.Params.Names)
                {
                    Tensor w = model.Params.Weight(name);
                    Tensor g = model.Params.Grad(name);
                    gpu.EnsureHostCurrent(g); // device-accumulated grads
                    for (int i = 0; i < w.Length; i++)
                        w.Data[i] -= lr * g.Data[i];
                    gpu.InvalidateDeviceCache(w); // direct host write
                }
            }
            model.Params.ZeroGrads();
            float final = model.ForwardBackward(inputs, targets, batch);
            Check.True(final < 0.2f * initial,
                $"GPU overfit: loss {initial:F3} -> {final:F3}, expected final < {0.2f * initial:F3}");
            Console.WriteLine($"    GPU overfit loss {initial:F4} -> {final:F4} after 200 SGD steps");
        }
    }
}
