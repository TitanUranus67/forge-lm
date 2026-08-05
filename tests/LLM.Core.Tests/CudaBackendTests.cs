namespace LLM.Core.Tests;

using LLM.Core.Checkpoint;
using LLM.Core.Model;
using LLM.Core.Tensor;
using LLM.Core.Tensor.Cuda;
using LLM.Core.Training;
using Tensor = LLM.Core.Tensor.Tensor;

/// <summary>CUDA-specific validation plus the complete device-backend numerical contract.</summary>
public static class CudaBackendTests
{
    [Test]
    public static void TrainingGraph_MatchesOrdinaryCudaTrajectory()
    {
        if (!CudaBackend.IsAvailable) { Console.WriteLine("    skipped (CUDA unavailable)"); return; }
        var config = new ModelConfig(31, 8, 16, 1, 2);
        int[] ids = Enumerable.Range(0, 512).Select(i => (i * 7 + 3) % config.VocabSize).ToArray();
        var options = new TrainOptions
        {
            Steps = 3,
            MaxLr = 2e-4f,
            MinLr = 2e-4f,
            WarmupSteps = 0,
            WeightDecay = 0.1f,
            GradClip = 1f,
            ContextLength = config.ContextLength,
            BatchSize = 2,
            AccumulationSteps = 3,
            Seed = 71,
            LogEvery = 3,
        };

        Dictionary<string, float[]> expected;
        float expectedLoss;
        using (var ordinaryBackend = new CudaBackend(matMulMode: CudaMatMulMode.CuBlasFp32))
        using (var ordinaryData = new DataLoader(ids))
        {
            var ordinary = new GptModel(config, ordinaryBackend, new Random(19));
            TrainSummary result = Trainer.Train(ordinary, ordinaryData, null, options);
            expectedLoss = result.FinalTrainLoss;
            expected = ordinary.Params.Names.ToDictionary(name => name, name =>
            {
                Tensor weight = ordinary.Params.Weight(name);
                ordinaryBackend.EnsureHostCurrent(weight);
                return (float[])weight.Data.Clone();
            });
        }

        using var graphBackend = new CudaBackend(matMulMode: CudaMatMulMode.CuBlasFp32);
        using var graphData = new DataLoader(ids);
        var graphModel = new GptModel(config, graphBackend, new Random(19));
        TrainSummary graphResult = Trainer.Train(graphModel, graphData, null,
            options with { UseCudaGraphs = true });
        Check.Near(graphResult.FinalTrainLoss, expectedLoss, 2e-5f,
            "CUDA graph loss follows ordinary CUDA");
        foreach (string name in graphModel.Params.Names)
        {
            Tensor weight = graphModel.Params.Weight(name);
            graphBackend.EnsureHostCurrent(weight);
            Check.SpanNear(weight.Data, expected[name], 2e-5f,
                $"{name}: CUDA graph update follows ordinary CUDA");
        }
    }

    [Test]
    public static void StreamCapturedGraph_ReplaysKernelSequence()
    {
        if (!CudaBackend.IsAvailable) { Console.WriteLine("    skipped (CUDA unavailable)"); return; }
        using var cuda = new CudaBackend();
        var x = new Tensor(4); x.Data.AsSpan().Fill(2f);
        var y = new Tensor(4); y.Data.AsSpan().Fill(3f);

        // Upload before capture so capture contains only replayable device work.
        cuda.Scale(x, 1f);
        cuda.Scale(y, 1f);
        using CudaGraphExecutable graph = cuda.CaptureGraphForTest(() =>
        {
            cuda.Scale(x, 2f);
            cuda.AddInPlace(x, y);
        });

        graph.Launch();
        cuda.EnsureHostCurrent(x);
        Check.SpanNear(x.Data, new float[] { 7f, 7f, 7f, 7f }, 0f,
            "captured CUDA kernel sequence replays once");
        // A graph launch bypasses backend tensor bookkeeping. This no-op device write
        // makes the second replay's result eligible for another host download.
        cuda.Scale(x, 1f);
        graph.Launch();
        cuda.EnsureHostCurrent(x);
        Check.SpanNear(x.Data, new float[] { 17f, 17f, 17f, 17f }, 0f,
            "captured CUDA kernel sequence replays repeatedly");
    }

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
    public static void LossAccumulator_MatchesCpuWithOneReadback()
    {
        if (Skip()) return;
        var cpu = new CpuBackend();
        float[] firstValues = [2f, 0f, -1f, 0f, 1f, 2f];
        float[] secondValues = [-1f, 3f, 0f];
        int[] firstTargets = [0, 2];
        int[] secondTargets = [1];

        float first = cpu.CrossEntropyForward(new Tensor((float[])firstValues.Clone(), 2, 3), firstTargets,
            new Tensor(2, 3), 2, 3, -1);
        float second = cpu.CrossEntropyForward(new Tensor((float[])secondValues.Clone(), 1, 3), secondTargets,
            new Tensor(1, 3), 1, 3, -1);
        long readbacksBefore = Cuda!.ReductionReadbackCount;

        Cuda.BeginLossAccumulation();
        Cuda.CrossEntropyForward(new Tensor((float[])firstValues.Clone(), 2, 3), firstTargets,
            new Tensor(2, 3), 2, 3, -1);
        Cuda.CrossEntropyForward(new Tensor((float[])secondValues.Clone(), 1, 3), secondTargets,
            new Tensor(1, 3), 1, 3, -1);
        float actual = Cuda.EndLossAccumulation();

        Check.Near(actual, (first * 2f + second) / 3f, 1e-5f, "CUDA accumulated loss matches CPU");
        Check.True(Cuda.ReductionReadbackCount - readbacksBefore == 1,
            "multiple CUDA losses require one final host readback");
    }

    [Test]
    public static void CuBlasFp32_MatmulsAndModelMatchCpu()
    {
        if (Skip()) return;
        using var cuda = new CudaBackend(matMulMode: CudaMatMulMode.CuBlasFp32);
        Check.True(cuda.MatMulMode == CudaMatMulMode.CuBlasFp32, "cuBLAS FP32 mode is active");
        long batchedCallsBefore = cuda.StridedBatchedMatMulCallCount;
        long fusedAttentionCallsBefore = cuda.FusedAttentionCallCount;
        GpuBackendTests.RunWithBackend(cuda, () =>
        {
            GpuBackendTests.MatMul_MatchesCpu();
            GpuBackendTests.BatchedAttentionKernels_MatchCpu();
            GpuBackendTests.Model_ForwardMatchesCpu();
            GpuBackendTests.Model_GradientCheck_Batched();
            GpuBackendTests.Overfit_SmallBatchLossDrops();
        });
        Check.True(cuda.StridedBatchedMatMulCallCount > batchedCallsBefore,
            "cuBLAS attention uses strided-batched GEMM");
        Check.True(cuda.FusedAttentionCallCount > fusedAttentionCallsBefore,
            "CUDA attention uses fused packing and softmax kernels");
    }

    [Test]
    public static void Tf32_RequiresAmpereOrNewer()
    {
        if (Skip()) return;
        if (Cuda!.SupportsTf32)
        {
            using var tf32 = new CudaBackend(matMulMode: CudaMatMulMode.CuBlasTf32);
            Check.True(tf32.MatMulMode == CudaMatMulMode.CuBlasTf32, "TF32 mode is active");
            return;
        }

        try
        {
            using var unexpected = new CudaBackend(matMulMode: CudaMatMulMode.CuBlasTf32);
            Check.Fail("pre-Ampere CUDA device accepted TF32 mode");
        }
        catch (InvalidOperationException ex)
        {
            Check.True(ex.InnerException is NotSupportedException,
                "pre-Ampere TF32 rejection reports NotSupportedException");
        }
    }

    [Test]
    public static void CustomCheckpoint_ResumesWithCuBlasFp32()
    {
        if (Skip()) return;
        string dataPath = Path.GetTempFileName();
        string checkpointPath = Path.GetTempFileName();
        try
        {
            const int vocab = 32, context = 8;
            using (var writer = new BinaryWriter(File.Create(dataPath)))
                for (int i = 0; i < 2048; i++) writer.Write((ushort)(i % vocab));

            var options = new TrainOptions
            {
                Steps = 2,
                MaxLr = 1e-3f,
                MinLr = 1e-4f,
                WarmupSteps = 1,
                WeightDecay = 0.1f,
                GradClip = 1f,
                ContextLength = context,
                BatchSize = 2,
                AccumulationSteps = 2,
                Seed = 99,
                LogEvery = 1,
                ValEvery = 0,
            };

            using (var custom = new CudaBackend())
            {
                var model = new GptModel(new ModelConfig(vocab, context, 16, 2, 2), custom, new Random(42));
                TrainingState state = TrainingState.CreateNew(custom, options);
                using var data = new DataLoader(dataPath);
                Trainer.Train(model, data, val: null, options,
                    controlHook: step => step == 1 ? TrainCommand.SaveAndQuit : TrainCommand.Continue,
                    state: state);
                Checkpoint.SaveTraining(model, state, checkpointPath);
            }

            using var cuBlas = new CudaBackend(matMulMode: CudaMatMulMode.CuBlasFp32);
            Checkpoint.LoadedTrainingCheckpoint loaded = Checkpoint.LoadTraining(checkpointPath, cuBlas);
            Check.True(loaded.TrainingState is not null, "custom checkpoint retains training state");
            Check.True(loaded.TrainingState!.GlobalStep == 1, "custom checkpoint global step loads on cuBLAS");
            Check.True(loaded.TrainingState.Optimizer.StepCount == 1, "custom checkpoint Adam state loads on cuBLAS");
            using (var data = new DataLoader(dataPath))
                Trainer.Train(loaded.Model, data, val: null, options, state: loaded.TrainingState);
            Check.True(loaded.TrainingState.GlobalStep == 2, "cuBLAS resume reaches the stored target");
            Check.True(loaded.TrainingState.Optimizer.StepCount == 2, "cuBLAS resume advances Adam state");
        }
        finally
        {
            File.Delete(dataPath);
            File.Delete(checkpointPath);
        }
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
    public static void GlobalSumSquares_MatchesCpuWithOneReadback()
    {
        if (Skip()) return;
        CudaBackend cuda = Cuda!;
        var rng = new Random(914);
        int[] lengths = { 1, 257, 4097, 1_000_003 };
        var tensors = new List<Tensor>();
        double expected = 0;
        foreach (int length in lengths)
        {
            var values = new float[length];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (float)((rng.NextDouble() * 2 - 1) * 0.1);
                expected += (double)values[i] * values[i];
            }
            tensors.Add(new Tensor(values, length));
        }

        long before = cuda.ReductionReadbackCount;
        double actual = cuda.GlobalSumSquares(tensors);
        long after = cuda.ReductionReadbackCount;
        double relativeError = Math.Abs(actual - expected) / Math.Max(1.0, expected);
        Check.True(relativeError < 2e-5,
            $"CUDA global sum-squares relative error {relativeError:G4} < 2e-5");
        Check.True(after - before == 1, "CUDA global sum-squares performs one scalar readback");

        before = cuda.ReductionReadbackCount;
        actual = cuda.GlobalSumSquares(Array.Empty<Tensor>());
        after = cuda.ReductionReadbackCount;
        Check.True(actual == 0, "empty CUDA global sum-squares is zero");
        Check.True(after == before, "empty CUDA global sum-squares performs no readback");
    }

    [Test]
    public static void ClipGradNorm_ClipsAndSkipsWithOneReadback()
    {
        if (Skip()) return;
        CudaBackend cuda = Cuda!;
        var clipped = new Parameters(cuda);
        clipped.Add("a", 1);
        clipped.Add("b", 1);
        clipped.Grad("a").Data[0] = 3f;
        clipped.Grad("b").Data[0] = 4f;

        long before = cuda.ReductionReadbackCount;
        Trainer.ClipGradNorm(clipped, cuda, 1f);
        long after = cuda.ReductionReadbackCount;
        cuda.EnsureHostCurrent(clipped.Grad("a"));
        cuda.EnsureHostCurrent(clipped.Grad("b"));
        Check.Near(clipped.Grad("a").Data[0], 0.6f, 1e-6f, "CUDA clipped gradient a");
        Check.Near(clipped.Grad("b").Data[0], 0.8f, 1e-6f, "CUDA clipped gradient b");
        Check.True(after - before == 1, "CUDA clipped norm performs one scalar readback");

        var unchanged = new Parameters(cuda);
        unchanged.Add("g", 2);
        unchanged.Grad("g").Data[0] = 0.1f;
        unchanged.Grad("g").Data[1] = -0.2f;
        before = cuda.ReductionReadbackCount;
        Trainer.ClipGradNorm(unchanged, cuda, 1f);
        after = cuda.ReductionReadbackCount;
        Check.Near(unchanged.Grad("g").Data[0], 0.1f, 0f, "CUDA unclipped gradient 0");
        Check.Near(unchanged.Grad("g").Data[1], -0.2f, 0f, "CUDA unclipped gradient 1");
        Check.True(after - before == 1, "CUDA unclipped norm performs one scalar readback");
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
