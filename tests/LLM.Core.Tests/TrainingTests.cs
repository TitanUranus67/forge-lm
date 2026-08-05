
namespace LLM.Core.Tests
{
    using LLM.Core.Model;
    using LLM.Core.Tensor;
    using LLM.Core.Training;
    using Tensor = LLM.Core.Tensor.Tensor;

    /// <summary>
    /// Tests for the training stack: LR schedule shape, AdamW mechanics
    /// (first step size, convergence on a quadratic, decoupled decay only on
    /// 2-D tensors), DataLoader sampling, and an end-to-end Trainer run that
    /// must cut the loss to under a quarter of its initial value.
    /// </summary>
    public static class TrainingTests
    {
        private static readonly CpuBackend B = new();

        private sealed class OverflowingNormBackend : CpuBackend
        {
            public double GlobalSumSquares(IReadOnlyList<Tensor> tensors) => double.PositiveInfinity;
        }

        private static ModelConfig Small => new(VocabSize: 32, ContextLength: 8, DModel: 16, NLayers: 2, NHeads: 2);

        [Test]
        public static void LrSchedule_WarmupPeakDecay()
        {
            // totalSteps=100, maxLr=1, minLr=0.1, warmup=10
            Check.Near(LrSchedule.GetLr(0, 100, 1f, 0.1f, 10), 0.1f, 1e-6f, "warmup step 0 = maxLr/warmup");
            Check.Near(LrSchedule.GetLr(4, 100, 1f, 0.1f, 10), 0.5f, 1e-6f, "warmup midpoint is linear");
            Check.Near(LrSchedule.GetLr(9, 100, 1f, 0.1f, 10), 1.0f, 1e-6f, "peak at end of warmup");
            Check.Near(LrSchedule.GetLr(55, 100, 1f, 0.1f, 10), 0.55f, 1e-5f, "cosine midpoint t=0.5 -> (max+min)/2");
            Check.Near(LrSchedule.GetLr(99, 100, 1f, 0.1f, 10), 0.1f, 2e-3f, "near minLr just before totalSteps");
            Check.Near(LrSchedule.GetLr(100, 100, 1f, 0.1f, 10), 0.1f, 0f, "exactly minLr at totalSteps");
            Check.Near(LrSchedule.GetLr(150, 100, 1f, 0.1f, 10), 0.1f, 0f, "clamped to minLr past totalSteps");
        }

        [Test]
        public static void AdamW_FirstStepMovesByAboutLr()
        {
            var p = new Parameters();
            Tensor w = p.Add("w", 1, 1);
            w.Data[0] = 1f;
            p.Grad("w").Data[0] = 1f;

            var adam = new AdamW();
            adam.Step(p, lr: 0.1f, weightDecay: 0f);
            // mHat = vHat = 1 after bias correction, so the step is ~lr in the -grad direction.
            Check.Near(w.Data[0], 0.9f, 1e-4f, "first AdamW step moves param by ~lr against the gradient");
        }

        [Test]
        public static void AdamW_QuadraticConvergesTowardZero()
        {
            var p = new Parameters();
            Tensor w = p.Add("w", 1, 1);
            Tensor g = p.Grad("w");
            w.Data[0] = 2f;

            var adam = new AdamW();
            float prev = Math.Abs(w.Data[0]);
            for (int step = 0; step < 200; step++)
            {
                g.Data[0] = w.Data[0]; // gradient of f(w) = w^2/2
                adam.Step(p, lr: 0.05f, weightDecay: 0f);
                float cur = Math.Abs(w.Data[0]);
                // Far from the optimum the steps point straight at 0; near it,
                // momentum legitimately overshoots and |w| can tick up briefly.
                if (step < 30)
                    Check.True(cur <= prev + 1e-6f, $"|param| decreases while far from 0 (step {step}: {prev} -> {cur})");
                prev = cur;
            }
            Check.True(prev < 0.005f, $"after 200 steps |param| {prev} should be near 0");
        }

        [Test]
        public static void AdamW_WeightDecaySkips1D()
        {
            var p = new Parameters();
            Tensor w2d = p.Add("w", 2, 2);
            Tensor b1d = p.Add("b", 3);
            w2d.Fill(1f);
            b1d.Fill(1f);
            // grads stay zero: only the decoupled decay term can move the weights.

            var adam = new AdamW();
            adam.Step(p, lr: 1f, weightDecay: 0.5f);
            foreach (float x in w2d.Data) Check.Near(x, 0.5f, 1e-6f, "2-D param shrinks by lr*wd");
            foreach (float x in b1d.Data) Check.Near(x, 1.0f, 0f, "1-D param is not weight-decayed");
        }

        [Test]
        public static void ClipGradNorm_NonFiniteFailsLoudly()
        {
            var p = new Parameters();
            p.Add("w", 2, 2);
            p.Grad("w").Data[0] = float.NaN;

            bool threw = false;
            try
            {
                Trainer.ClipGradNorm(p, B, 1f);
            }
            catch (InvalidOperationException ex)
            {
                threw = ex.Message.Contains("'w'[0]", StringComparison.Ordinal) &&
                    ex.Message.Contains("NaN", StringComparison.Ordinal);
            }
            Check.True(threw, "non-finite gradient reports the exact tensor and element");
        }

        [Test]
        public static void RobustGradientNorm_HandlesValuesWhoseSquaresOverflowFloat()
        {
            var p = new Parameters();
            p.Add("huge", 2);
            p.Grad("huge").Data[0] = float.MaxValue;
            p.Grad("huge").Data[1] = -float.MaxValue;

            double norm = Trainer.RobustGradientNorm(p, B);
            double expected = Math.Sqrt(2d) * float.MaxValue;
            Check.True(double.IsFinite(norm), "robust gradient norm remains finite");
            Check.True(Math.Abs(norm - expected) / expected < 1e-12,
                $"robust norm {norm:G6} matches scaled reference {expected:G6}");
        }

        [Test]
        public static void ClipGradNorm_FallsBackWhenFastReductionOverflows()
        {
            var backend = new OverflowingNormBackend();
            var p = new Parameters(backend);
            p.Add("a", 1);
            p.Add("b", 1);
            p.Grad("a").Data[0] = 3f;
            p.Grad("b").Data[0] = 4f;

            Trainer.ClipGradNorm(p, backend, 1f);

            Check.Near(p.Grad("a").Data[0], 0.6f, 1e-6f,
                "overflowing fast norm falls back and clips gradient a");
            Check.Near(p.Grad("b").Data[0], 0.8f, 1e-6f,
                "overflowing fast norm falls back and clips gradient b");
        }

        [Test]
        public static void DataLoader_SampleIsContiguousShiftedWindow()
        {
            string path = Path.GetTempFileName();
            try
            {
                using (var bw = new BinaryWriter(File.Create(path)))
                    for (int i = 0; i < 1000; i++) bw.Write((ushort)i);

                var loader = new DataLoader(path);
                Check.True(loader.Length == 1000, $"Length {loader.Length} should be 1000");

                var sampler = new TrainingSampler(7);
                const int ctx = 8;
                int[] inputs = new int[ctx], targets = new int[ctx];
                for (int trial = 0; trial < 200; trial++)
                {
                    loader.Sample(sampler, ctx, inputs, targets);
                    Check.True(inputs[0] >= 0 && inputs[0] <= 1000 - ctx - 1,
                        $"offset {inputs[0]} within bounds");
                    for (int i = 0; i < ctx; i++)
                    {
                        Check.True(targets[i] == inputs[i] + 1, $"targets[{i}] = inputs[{i}] + 1");
                        if (i + 1 < ctx)
                            Check.True(inputs[i + 1] == inputs[i] + 1, $"inputs contiguous at {i}");
                    }
                }
            }
            finally { File.Delete(path); }
        }

        [Test]
        public static void DataLoader_MemoryMappedPathSamplesCorrectly()
        {
            string path = Path.GetTempFileName();
            try
            {
                using (var bw = new BinaryWriter(File.Create(path)))
                    for (int i = 0; i < 1000; i++) bw.Write((ushort)(i * 3 % 65536));

                // inMemoryLimit: 0 forces the memory-mapped path even for a tiny file
                using var loader = new DataLoader(path, inMemoryLimit: 0);
                Check.True(loader.Length == 1000, $"Length {loader.Length} should be 1000");

                var sampler = new TrainingSampler(7);
                const int ctx = 8;
                int[] inputs = new int[ctx], targets = new int[ctx];
                for (int trial = 0; trial < 200; trial++)
                {
                    loader.Sample(sampler, ctx, inputs, targets);
                    int o = inputs[0] / 3; // ids[i] = i*3, all unique multiples of 3
                    Check.True(inputs[0] % 3 == 0, $"offset id is a multiple of 3");
                    for (int i = 0; i < ctx; i++)
                    {
                        Check.True(inputs[i] == (o + i) * 3 % 65536, $"inputs[{i}] follows ids[{o + i}]");
                        Check.True(targets[i] == (o + i + 1) * 3 % 65536, $"targets[{i}] follows ids[{o + i + 1}]");
                    }
                }
            }
            finally { File.Delete(path); }
        }

        [Test]
        public static void DataLoader_VisitsEveryWindowBeforeRepeating()
        {
            const int ctx = 8;
            int[] ids = Enumerable.Range(0, 81).ToArray(); // exactly ten full shifted windows
            using var loader = new DataLoader(ids);
            var sampler = new TrainingSampler(17);
            int[] inputs = new int[ctx], targets = new int[ctx];
            var firstEpoch = new HashSet<int>();

            for (int i = 0; i < 10; i++)
            {
                loader.Sample(sampler, ctx, inputs, targets);
                Check.True(inputs[0] % ctx == 0, $"window {inputs[0]} is context aligned");
                Check.True(firstEpoch.Add(inputs[0]), $"window {inputs[0]} is not repeated in the epoch");
            }

            Check.True(firstEpoch.Count == 10, "all complete windows are visited once");
            loader.Sample(sampler, ctx, inputs, targets);
            Check.True(sampler.Epoch == 1, "the next sample starts a new shuffled epoch");
        }

        [Test]
        public static void TrainingRandom_SupportsBoundsAboveInt32()
        {
            const long bound = 10_000_000_000L;
            var first = new TrainingRandom(123);
            var second = new TrainingRandom(123);
            bool reachedPastInt = false;

            for (int i = 0; i < 1000; i++)
            {
                long a = first.NextInt64(bound);
                long b = second.NextInt64(bound);
                Check.True(a == b, $"large-bound sample {i} is deterministic");
                Check.True(a >= 0 && a < bound, $"large-bound sample {a} is in range");
                reachedPastInt |= a > int.MaxValue;
            }

            Check.True(reachedPastInt, "large-bound generator can reach offsets beyond Int32.MaxValue");
        }

        [Test]
        public static void Trainer_TrainLossDrops()
        {
            // Repetitive data: next token is fully predictable (i -> (i+1) mod 16).
            string path = Path.GetTempFileName();
            try
            {
                using (var bw = new BinaryWriter(File.Create(path)))
                    for (int i = 0; i < 4096; i++) bw.Write((ushort)(i % 16));

                var train = new DataLoader(path);
                var val = new DataLoader(path);
                var model = new GptModel(Small, B, new Random(3));
                var opts = new TrainOptions
                {
                    Steps = 300,
                    MaxLr = 3e-3f,
                    MinLr = 3e-4f,
                    WarmupSteps = 10,
                    WeightDecay = 0.1f,
                    GradClip = 1.0f,
                    Seed = 123,
                    LogEvery = 50,
                    ValEvery = 100,
                    ValBatches = 2,
                };

                var logs = new List<TrainLog>();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                TrainSummary summary = Trainer.Train(model, train, val, opts, logs.Add);
                sw.Stop();

                float initial = logs[0].TrainLoss;
                Check.True(summary.Steps == 300, "summary reports step count");
                Check.True(summary.FinalValLoss is not null, "val loss was evaluated");
                Check.True(summary.FinalTrainLoss < 0.25f * initial,
                    $"train loss {initial:F3} -> {summary.FinalTrainLoss:F3}, expected < {0.25f * initial:F3}");
                Console.WriteLine($"    trainer loss {initial:F4} -> {summary.FinalTrainLoss:F4} " +
                                  $"(val {summary.FinalValLoss:F4}) in {sw.Elapsed.TotalSeconds:F1}s, {logs.Count} logs");
                Check.True(sw.Elapsed.TotalSeconds < 15, $"trainer run should take <15s, took {sw.Elapsed.TotalSeconds:F1}s");
            }
            finally { File.Delete(path); }
        }

        [Test]
        public static void Trainer_InvalidOptionsFailBeforeTraining()
        {
            string path = WriteRepetitiveTokens(256);
            try
            {
                using var data = new DataLoader(path);
                var model = new GptModel(Small, B, new Random(1));
                var valid = new TrainOptions
                {
                    Steps = 2,
                    WarmupSteps = 1,
                    ContextLength = Small.ContextLength,
                    ValEvery = 0,
                };

                foreach (TrainOptions invalid in new[]
                {
                    valid with { Steps = 0 },
                    valid with { MaxLr = float.NaN },
                    valid with { MinLr = valid.MaxLr + 1f },
                    valid with { WarmupSteps = valid.Steps },
                    valid with { WeightDecay = -0.1f },
                    valid with { LogEvery = 0 },
                    valid with { ContextLength = Small.ContextLength + 1 },
                })
                {
                    bool threw = false;
                    try { Trainer.Train(model, data, val: null, invalid); }
                    catch (ArgumentException) { threw = true; }
                    Check.True(threw, $"invalid training options fail before the first step: {invalid}");
                }
            }
            finally { File.Delete(path); }
        }

        [Test]
        public static void Validation_DoesNotPerturbTrainingTrajectory()
        {
            string path = WriteRepetitiveTokens(2048);
            try
            {
                var withoutValidation = new GptModel(Small, B, new Random(17));
                var withValidation = new GptModel(Small, B, new Random(17));
                var baseOptions = new TrainOptions
                {
                    Steps = 4,
                    MaxLr = 1e-3f,
                    MinLr = 1e-4f,
                    WarmupSteps = 1,
                    ContextLength = Small.ContextLength,
                    BatchSize = 2,
                    Seed = 81,
                    LogEvery = 1,
                    ValEvery = 0,
                    ValBatches = 3,
                    ValSeed = 1234,
                };

                using (var train = new DataLoader(path))
                    Trainer.Train(withoutValidation, train, val: null, baseOptions);
                using (var train = new DataLoader(path))
                using (var val = new DataLoader(path))
                    Trainer.Train(withValidation, train, val, baseOptions with { ValEvery = 1 });

                foreach (string name in withoutValidation.Params.Names)
                {
                    float[] expected = withoutValidation.Params.Weight(name).Data;
                    float[] actual = withValidation.Params.Weight(name).Data;
                    for (int i = 0; i < expected.Length; i++)
                        Check.True(BitConverter.SingleToInt32Bits(actual[i]) == BitConverter.SingleToInt32Bits(expected[i]),
                            $"{name}[{i}] unchanged by validation sampling");
                }
            }
            finally { File.Delete(path); }
        }

        [Test]
        public static void Validation_CadenceIsIndependentOfLogging()
        {
            string path = WriteRepetitiveTokens(2048);
            try
            {
                var model = new GptModel(Small, B, new Random(41));
                var options = new TrainOptions
                {
                    Steps = 7,
                    MaxLr = 1e-3f,
                    MinLr = 1e-4f,
                    WarmupSteps = 1,
                    ContextLength = Small.ContextLength,
                    BatchSize = 2,
                    Seed = 7,
                    LogEvery = 5,
                    ValEvery = 3,
                    ValBatches = 1,
                };
                var logs = new List<TrainLog>();
                using var train = new DataLoader(path);
                using var val = new DataLoader(path);

                Trainer.Train(model, train, val, options, logs.Add);

                int[] validationSteps = logs.Where(l => l.ValLoss.HasValue).Select(l => l.Step).ToArray();
                Check.True(validationSteps.SequenceEqual(new[] { 1, 3, 6, 7 }),
                    $"validation steps [{string.Join(",", validationSteps)}] follow ValEvery independently");
                Check.True(logs.Any(l => l.Step == 5 && !l.ValLoss.HasValue),
                    "regular logging still occurs between validation events");
            }
            finally { File.Delete(path); }
        }

        [Test]
        public static void GradientAccumulation_MatchesEquivalentLargeBatch()
        {
            string path = WriteRepetitiveTokens(2048);
            try
            {
                var accumulated = new GptModel(Small, B, new Random(23));
                var largeBatch = new GptModel(Small, B, new Random(23));
                var accumulatedOptions = new TrainOptions
                {
                    Steps = 2,
                    MaxLr = 1e-3f,
                    MinLr = 1e-4f,
                    WarmupSteps = 0,
                    WeightDecay = 0f,
                    GradClip = 0f,
                    ContextLength = Small.ContextLength,
                    BatchSize = 2,
                    AccumulationSteps = 2,
                    Seed = 222,
                    LogEvery = 1,
                    ValEvery = 0,
                };
                TrainCommand StopAfterOne(int step) =>
                    step == 1 ? TrainCommand.SaveAndQuit : TrainCommand.Continue;

                TrainingState accumulatedState = TrainingState.CreateNew(B, accumulatedOptions);
                using (var data = new DataLoader(path))
                    Trainer.Train(accumulated, data, val: null, accumulatedOptions,
                        controlHook: StopAfterOne, state: accumulatedState);

                TrainOptions largeBatchOptions = accumulatedOptions with
                {
                    BatchSize = 4,
                    AccumulationSteps = 1,
                };
                TrainingState largeBatchState = TrainingState.CreateNew(B, largeBatchOptions);
                using (var data = new DataLoader(path))
                    Trainer.Train(largeBatch, data, val: null, largeBatchOptions,
                        controlHook: StopAfterOne, state: largeBatchState);

                foreach (string name in accumulated.Params.Names)
                    Check.SpanNear(accumulated.Params.Weight(name).Data, largeBatch.Params.Weight(name).Data,
                        2e-5f, $"{name}: accumulation equals one equivalent large batch");
                Check.True(accumulatedState.GlobalStep == 1, "one accumulated optimizer update");
                Check.True(accumulatedState.Optimizer.StepCount == 1, "Adam advances once per accumulated update");
            }
            finally { File.Delete(path); }
        }

        private static string WriteRepetitiveTokens(int count)
        {
            string path = Path.GetTempFileName();
            using (var bw = new BinaryWriter(File.Create(path)))
                for (int i = 0; i < count; i++) bw.Write((ushort)(i % 16));
            return path;
        }

        [Test]
        public static void Trainer_ControlHookInvokedOncePerStep()
        {
            string path = WriteRepetitiveTokens(1024);
            try
            {
                var train = new DataLoader(path);
                var model = new GptModel(Small, B, new Random(3));
                var opts = new TrainOptions { Steps = 10, WarmupSteps = 2, Seed = 1, LogEvery = 5 };

                var seen = new List<int>();
                TrainSummary summary = Trainer.Train(model, train, val: null, opts,
                    controlHook: s => { seen.Add(s); return TrainCommand.Continue; });

                Check.True(seen.Count == 10, $"hook called {seen.Count} times, expected 10");
                for (int i = 0; i < seen.Count; i++)
                    Check.True(seen[i] == i + 1, $"hook call {i} got step {seen[i]}, expected {i + 1}");
                Check.True(summary.Steps == 10, $"summary.Steps {summary.Steps}, expected 10");
            }
            finally { File.Delete(path); }
        }

        [Test]
        public static void Trainer_SaveAndQuitStopsEarly()
        {
            string path = WriteRepetitiveTokens(1024);
            try
            {
                var train = new DataLoader(path);
                var model = new GptModel(Small, B, new Random(3));
                var opts = new TrainOptions { Steps = 20, WarmupSteps = 2, Seed = 1, LogEvery = 5 };

                int calls = 0;
                TrainSummary summary = Trainer.Train(model, train, val: null, opts,
                    controlHook: s => { calls++; return s == 5 ? TrainCommand.SaveAndQuit : TrainCommand.Continue; });

                Check.True(calls == 5, $"hook called {calls} times, expected 5");
                Check.True(summary.Steps == 5, $"summary.Steps {summary.Steps}, expected 5");
            }
            finally { File.Delete(path); }
        }
    }
}
