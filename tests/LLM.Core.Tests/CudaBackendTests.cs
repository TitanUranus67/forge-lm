namespace LLM.Core.Tests;

using LLM.Core.Tensor;
using LLM.Core.Tensor.Cuda;
using Tensor = LLM.Core.Tensor.Tensor;

/// <summary>CUDA-specific validation plus the complete device-backend numerical contract.</summary>
public static class CudaBackendTests
{
    private static CudaBackend? _cuda;
    private static bool _probed;

    private static CudaBackend? Cuda
    {
        get
        {
            if (_probed) return _cuda;
            _probed = true;
            try { _cuda = new CudaBackend(); }
            catch (Exception ex) { Console.WriteLine($"    CUDA probe failed: {ex.GetType().Name}: {ex.Message}"); }
            return _cuda;
        }
    }

    private static bool Skip()
    {
        if (Cuda is not null) return false;
        Console.WriteLine("    SKIP: no CUDA device available");
        return true;
    }

    [Test]
    public static void AllKernelsAndFullModel_MatchCpu()
    {
        if (Skip()) return;
        GpuBackendTests.RunWithBackend(Cuda!, () =>
        {
            GpuBackendTests.MatMul_MatchesCpu();
            GpuBackendTests.Elementwise_MatchesCpu();
            GpuBackendTests.FlatDispatch_BeyondOneDimensionalLimit();
            GpuBackendTests.LayerNorm_MatchesCpu();
            GpuBackendTests.Softmax_MatchesCpu();
            GpuBackendTests.Gelu_MatchesCpu();
            GpuBackendTests.Embedding_MatchesCpu();
            GpuBackendTests.CausalMask_MatchesCpu();
            GpuBackendTests.CrossEntropy_MatchesCpu();
            GpuBackendTests.CrossEntropy_AliasedInPlaceMatchesCpu();
            GpuBackendTests.BatchedAttentionKernels_MatchCpu();
            GpuBackendTests.Residency_DirectHostWriteWithoutInvalidateIsStale();
            GpuBackendTests.Residency_InvalidateRefreshesDeviceCopy();
            GpuBackendTests.EnsureHostCurrent_RoundTripsDeviceResults();
            GpuBackendTests.Model_ForwardMatchesCpu();
            GpuBackendTests.Model_GradientCheck_Batched();
            GpuBackendTests.Overfit_SmallBatchLossDrops();
        });
    }

    [Test]
    public static void AdamWAndSumSquares_MatchCpu()
    {
        if (Skip()) return;
        ITensorBackend cpu = new CpuBackend();
        CudaBackend cuda = Cuda!;
        var rng = new Random(912);
        const int rows = 17, cols = 19;
        float[] RandomValues()
        {
            var values = new float[rows * cols];
            for (int i = 0; i < values.Length; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
            return values;
        }

        Tensor CpuTensor(float[] x) => new((float[])x.Clone(), rows, cols);
        Tensor CudaTensor(float[] x) => new((float[])x.Clone(), rows, cols);
        float[] w = RandomValues(), g = RandomValues(), m = RandomValues(), v = RandomValues().Select(MathF.Abs).ToArray();
        Tensor wc = CpuTensor(w), gc = CpuTensor(g), mc = CpuTensor(m), vc = CpuTensor(v);
        Tensor wg = CudaTensor(w), gg = CudaTensor(g), mg = CudaTensor(m), vg = CudaTensor(v);

        cpu.AdamWStep(wc, gc, mc, vc, 3e-4f, 0.9f, 0.95f, 1e-8f, 0.1f, 37);
        cuda.AdamWStep(wg, gg, mg, vg, 3e-4f, 0.9f, 0.95f, 1e-8f, 0.1f, 37);
        cuda.EnsureHostCurrent(wg); cuda.EnsureHostCurrent(mg); cuda.EnsureHostCurrent(vg);
        Check.SpanNear(wg.Data, wc.Data, 2e-5f, "CUDA AdamW weights");
        Check.SpanNear(mg.Data, mc.Data, 2e-5f, "CUDA AdamW first moment");
        Check.SpanNear(vg.Data, vc.Data, 2e-5f, "CUDA AdamW second moment");

        double expected = cpu.SumSquares(gc), actual = cuda.SumSquares(gg);
        Check.Near((float)actual, (float)expected, 2e-4f, "CUDA SumSquares");
    }

    [Test]
    public static void BucketOf_IdempotentAlignedAndBounded()
    {
        Check.True(CudaBackend.BucketOf(1) == 16, "min CUDA bucket is 16");
        var rng = new Random(913);
        for (int i = 1; i <= 100_000; i++)
        {
            int request = i <= 5000 ? i : rng.Next(1, 1 << 28);
            int bucket = CudaBackend.BucketOf(request);
            Check.True(bucket >= request, $"bucket {bucket} >= request {request}");
            Check.True((bucket & 15) == 0, $"bucket {bucket} is aligned");
            Check.True(CudaBackend.BucketOf(bucket) == bucket, $"bucket {bucket} is a fixed point");
        }
    }

    [Test]
    public static void Allocator_SteadyStatePlateaus()
    {
        if (Skip()) return;
        using var cuda = new CudaBackend();
        int[] sizes =
        {
            393216, 393216, 393216, 393216,
            196608, 196608, 196608, 196608,
            98304, 98304, 98304, 98304,
            786432, 786432, 2048, 2048, 768, 768,
        };
        const int warmup = 20, iterations = 80;
        var persistent = new List<Tensor>();
        long committedWarm = 0, hitsWarm = 0, carvesWarm = 0;
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            for (int i = 0; i < sizes.Length; i++)
            {
                int length = sizes[i] + sizes[i] * (((iteration + i) % 10) - 5) / 40;
                cuda.Zero(new Tensor(length));
            }
            if (iteration < warmup && iteration % 5 == 0)
            {
                var keep = new Tensor(50_000);
                cuda.Zero(keep);
                persistent.Add(keep);
            }
            if (iteration == warmup - 1)
            {
                committedWarm = cuda.CommittedBytes;
                (hitsWarm, carvesWarm) = cuda.AllocStats;
            }
        }

        long committedEnd = cuda.CommittedBytes;
        (long hitsEnd, long carvesEnd) = cuda.AllocStats;
        long hits = hitsEnd - hitsWarm, carves = carvesEnd - carvesWarm;
        double hitRate = (double)hits / (hits + carves);
        Console.WriteLine($"    CUDA allocator: {committedWarm / 1e6:F0}MB -> {committedEnd / 1e6:F0}MB, " +
            $"steady-state {hits} hits/{carves} carves ({hitRate:P1})");
        Check.True(committedWarm > 0, "CUDA warm-up committed memory is nonzero");
        Check.True(committedEnd == committedWarm, "CUDA committed memory plateaus after warm-up");
        Check.True(hitRate > 0.95, $"CUDA steady-state free-list hit rate {hitRate:P1} > 95%");
    }
}
