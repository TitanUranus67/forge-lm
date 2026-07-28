
namespace LLM.Core.Tests
{
    using LLM.Core.Checkpoint;
    using LLM.Core.Model;
    using LLM.Core.Tensor;
    using Tensor = LLM.Core.Tensor.Tensor;

    /// <summary>
    /// Checkpoint round-trip tests: bitwise-equal weights and identical logits
    /// after save/load, plus fail-loud behavior on a corrupted file.
    /// </summary>
    public static class CheckpointTests
    {
        private static readonly CpuBackend B = new();

        private static ModelConfig Small => new(VocabSize: 32, ContextLength: 8, DModel: 16, NLayers: 2, NHeads: 2);

        [Test]
        public static void RoundTrip_BitwiseEqualWeightsAndLogits()
        {
            string path = Path.GetTempFileName();
            try
            {
                var model = new GptModel(Small, B, new Random(42));
                int[] tokens = { 5, 6, 7, 8 };
                Tensor before = model.ForwardLast(tokens);

                Checkpoint.Save(model, path);
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
                Checkpoint.Save(model, path);
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
    }
}
