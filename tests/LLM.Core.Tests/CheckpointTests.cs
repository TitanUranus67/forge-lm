
namespace LLM.Core.Tests
{
    using LLM.Core.Checkpoint;
    using LLM.Core.Model;
    using LLM.Core.Tensor;
    using LLM.Core.Training;
    using Tensor = LLM.Core.Tensor.Tensor;

    /// <summary>
    /// Checkpoint round-trip tests: bitwise-equal weights and identical logits
    /// after save/load, plus fail-loud behavior on a corrupted file.
    /// </summary>
    public static class CheckpointTests
    {
        private static readonly CpuBackend B = new();

        private static ModelConfig Small => new(VocabSize: 32, ContextLength: 8, DModel: 16, NLayers: 2, NHeads: 2);

        private static TrainingState FreshState() => TrainingState.CreateNew(B, new TrainOptions
        {
            Steps = 2,
            MaxLr = 1e-3f,
            MinLr = 1e-4f,
            WarmupSteps = 1,
            ContextLength = Small.ContextLength,
            BatchSize = 1,
            ValEvery = 0,
        }, dataIdentity: "data", tokenizerIdentity: "tokenizer");

        [Test]
        public static void RoundTrip_BitwiseEqualWeightsAndLogits()
        {
            string path = Path.GetTempFileName();
            try
            {
                var model = new GptModel(Small, B, new Random(42));
                int[] tokens = { 5, 6, 7, 8 };
                Tensor before = model.ForwardLast(tokens);

                Checkpoint.SaveTraining(model, FreshState(), path);
                GptModel loaded = Checkpoint.Load(path, B);

                Check.True(loaded.Config == Small, "config round-trips");
                foreach (string name in model.Params.Names)
                {
                    float[] a = model.Params.Weight(name).Data;
                    float[] b = loaded.Params.Weight(name).Data;
                    Check.True(a.Length == b.Length, $"{name}: same length");
                    for (int i = 0; i < a.Length; i++)
                        Check.True(BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i]),
                            $"{name}[{i}] bitwise equal");
                }

                Tensor after = loaded.ForwardLast(tokens);
                Check.SpanNear(after.Data, before.Data, 0f, "logits identical after round-trip");
            }
            finally { File.Delete(path); }
        }

        [Test]
        public static void TrainingCheckpoint_ResumeMatchesUninterruptedRun()
        {
            string dataPath = Path.GetTempFileName();
            string checkpointPath = Path.GetTempFileName();
            try
            {
                using (var bw = new BinaryWriter(File.Create(dataPath)))
                    for (int i = 0; i < 2048; i++) bw.Write((ushort)(i % Small.VocabSize));

                var opts = new TrainOptions
                {
                    Steps = 6,
                    MaxLr = 1e-3f,
                    MinLr = 1e-4f,
                    WarmupSteps = 2,
                    WeightDecay = 0.1f,
                    GradClip = 1f,
                    ContextLength = Small.ContextLength,
                    BatchSize = 2,
                    Seed = 99,
                    LogEvery = 1,
                    ValEvery = 0,
                };

                var uninterrupted = new GptModel(Small, B, new Random(42));
                using (var data = new DataLoader(dataPath))
                    Trainer.Train(uninterrupted, data, val: null, opts);

                var interrupted = new GptModel(Small, B, new Random(42));
                TrainingState state = TrainingState.CreateNew(B, opts, dataIdentity: "data-A", tokenizerIdentity: "tok-A");
                using (var data = new DataLoader(dataPath))
                    Trainer.Train(interrupted, data, val: null, opts,
                        controlHook: step => step == 3 ? TrainCommand.SaveAndQuit : TrainCommand.Continue,
                        state: state);
                Checkpoint.SaveTraining(interrupted, state, checkpointPath);

                Checkpoint.LoadedTrainingCheckpoint loaded = Checkpoint.LoadTraining(checkpointPath, B);
                Check.True(loaded.TrainingState is not null, "checkpoint restores training state");
                Check.True(loaded.TrainingState!.GlobalStep == 3, "global step round-trips");
                Check.True(loaded.TrainingState.Optimizer.StepCount == 3, "Adam age round-trips");
                Check.True(loaded.TrainingState.DataIdentity == "data-A", "training data identity round-trips");
                Check.True(loaded.TokenizerIdentity == "tok-A", "tokenizer identity is available as metadata");
                using (var data = new DataLoader(dataPath))
                    Trainer.Train(loaded.Model, data, val: null, opts, state: loaded.TrainingState);

                foreach (string name in uninterrupted.Params.Names)
                {
                    float[] expected = uninterrupted.Params.Weight(name).Data;
                    float[] actual = loaded.Model.Params.Weight(name).Data;
                    for (int i = 0; i < expected.Length; i++)
                        Check.True(BitConverter.SingleToInt32Bits(actual[i]) == BitConverter.SingleToInt32Bits(expected[i]),
                            $"{name}[{i}] exact after interrupted resume");
                }
                Check.True(loaded.TrainingState.GlobalStep == 6, "resumed global step reaches target");
                Check.True(loaded.TrainingState.Optimizer.StepCount == 6, "resumed Adam age reaches target");

                GptModel inferenceOnly = Checkpoint.Load(checkpointPath, B);
                Check.True(inferenceOnly.Config == Small, "inference loader accepts training checkpoint");
            }
            finally
            {
                File.Delete(dataPath);
                File.Delete(checkpointPath);
            }
        }

        [Test]
        public static void Load_BadMagicThrows()
        {
            string path = Path.GetTempFileName();
            try
            {
                File.WriteAllBytes(path, "NOTACHECKPOINT!"u8.ToArray());
                bool threw = false;
                try { Checkpoint.Load(path, B); }
                catch (InvalidDataException) { threw = true; }
                Check.True(threw, "loading a file with bad magic throws InvalidDataException");
            }
            finally { File.Delete(path); }
        }

        [Test]
        public static void Load_TruncatedPayloadThrows()
        {
            string path = Path.GetTempFileName();
            try
            {
                var model = new GptModel(Small, B, new Random(1));
                Checkpoint.SaveTraining(model, FreshState(), path);
                // chop 4 bytes: float count no longer matches the registry
                byte[] bytes = File.ReadAllBytes(path);
                File.WriteAllBytes(path, bytes.AsSpan(0, bytes.Length - 4).ToArray());

                bool threw = false;
                try { Checkpoint.Load(path, B); }
                catch (InvalidDataException) { threw = true; }
                catch (EndOfStreamException) { threw = true; }
                Check.True(threw, "loading a truncated checkpoint throws");
            }
            finally { File.Delete(path); }
        }

        [Test]
        public static void OpenReader_AllowsAtomicCheckpointReplacement()
        {
            string path = Path.GetTempFileName();
            string replacement = path + ".replacement";
            string backup = path + ".backup";
            try
            {
                File.WriteAllBytes(path, new byte[] { 1, 2, 3 });
                File.WriteAllBytes(replacement, new byte[] { 4, 5, 6 });

                using (FileStream existingReader = Checkpoint.OpenCheckpointRead(path))
                {
                    Checkpoint.PublishAtomically(replacement, path, backup);

                    Check.True(existingReader.ReadByte() == 1,
                        "reader opened before replacement retains the old checkpoint generation");
                    Check.True(File.ReadAllBytes(path).SequenceEqual(new byte[] { 4, 5, 6 }),
                        "new readers see the replacement checkpoint generation");
                    Check.True(File.ReadAllBytes(backup).SequenceEqual(new byte[] { 1, 2, 3 }),
                        "atomic publication retains the previous checkpoint as a backup");
                }

                File.WriteAllBytes(replacement, new byte[] { 7, 8, 9 });
                Checkpoint.PublishAtomically(replacement, path, backup);
                Check.True(File.ReadAllBytes(path).SequenceEqual(new byte[] { 7, 8, 9 }),
                    "a later publication replaces the destination again");
                Check.True(File.ReadAllBytes(backup).SequenceEqual(new byte[] { 4, 5, 6 }),
                    "an existing backup is rotated to the immediately previous generation");
            }
            finally
            {
                File.Delete(path);
                File.Delete(replacement);
                File.Delete(backup);
            }
        }

        [Test]
        public static void TrainingCheckpoint_SameSizeCorruptionFailsChecksum()
        {
            string path = Path.GetTempFileName();
            try
            {
                var model = new GptModel(Small, B, new Random(9));
                var options = new TrainOptions
                {
                    Steps = 2,
                    MaxLr = 1e-3f,
                    MinLr = 1e-4f,
                    WarmupSteps = 1,
                    ContextLength = Small.ContextLength,
                    BatchSize = 1,
                    ValEvery = 0,
                };
                TrainingState state = TrainingState.CreateNew(B, options,
                    dataIdentity: "data", tokenizerIdentity: "tokenizer");
                Checkpoint.SaveTraining(model, state, path);

                using (var fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
                {
                    fs.Position = fs.Length - 33; // final payload byte, immediately before the SHA-256 trailer
                    int original = fs.ReadByte();
                    fs.Position--;
                    fs.WriteByte((byte)(original ^ 0x01));
                }

                bool threw = false;
                try { Checkpoint.LoadTraining(path, B); }
                catch (InvalidDataException ex) when (ex.Message.Contains("checksum", StringComparison.OrdinalIgnoreCase))
                {
                    threw = true;
                }
                Check.True(threw, "same-size checkpoint corruption is detected by SHA-256");
            }
            finally { File.Delete(path); }
        }
    }
}
