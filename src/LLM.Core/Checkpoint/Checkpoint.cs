namespace LLM.Core.Checkpoint
{
    using System.Text.Json;
    using LLM.Core.Model;
    using LLM.Core.Tensor;

    /// <summary>
    /// Custom binary checkpoint format, deterministic and fail-loud on any mismatch:
    ///   [magic "LLMSCRATCH1" (11 ascii bytes)]
    ///   [int32 jsonLength, little-endian]
    ///   [UTF-8 JSON header: ModelConfig fields + parameter names in registry order]
    ///   [raw little-endian float32 weights, concatenated in registry order]
    /// Loading validates the magic, that the header names exactly match the freshly
    /// constructed model's registry (in order), and that the remaining byte count
    /// matches the total parameter count — anything else throws.
    /// </summary>
    public static class Checkpoint
    {
        private static readonly byte[] Magic = "LLMSCRATCH1"u8.ToArray();

        private sealed class Header
        {
            public int VocabSize { get; set; }
            public int ContextLength { get; set; }
            public int DModel { get; set; }
            public int NLayers { get; set; }
            public int NHeads { get; set; }
            public string[] Names { get; set; } = [];
        }

        /// <summary>Writes the model's config and weights to <paramref name="path"/>.</summary>
        public static void Save(GptModel model, string path)
        {
            var header = new Header
            {
                VocabSize = model.Config.VocabSize,
                ContextLength = model.Config.ContextLength,
                DModel = model.Config.DModel,
                NLayers = model.Config.NLayers,
                NHeads = model.Config.NHeads,
                Names = model.Params.Names.ToArray(),
            };
            byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);

            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);
            bw.Write(Magic);
            bw.Write(json.Length);
            bw.Write(json);
            foreach (string name in model.Params.Names)
                foreach (float x in model.Params.Weight(name).Data)
                    bw.Write(x);
        }

        /// <summary>
        /// Loads a checkpoint: constructs a model from the stored config on
        /// <paramref name="backend"/>, then overwrites every weight tensor with the
        /// stored values. Throws <see cref="InvalidDataException"/> on any
        /// corruption or mismatch.
        /// </summary>
        public static GptModel Load(string path, ITensorBackend backend)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs);

            byte[] magic = br.ReadBytes(Magic.Length);
            if (magic.Length != Magic.Length || !magic.SequenceEqual(Magic))
                throw new InvalidDataException($"'{path}' is not an LLMSCRATCH1 checkpoint (bad magic).");

            int jsonLength = br.ReadInt32();
            if (jsonLength <= 0 || jsonLength > 1 << 20)
                throw new InvalidDataException($"Implausible header length {jsonLength}.");
            Header header = JsonSerializer.Deserialize<Header>(br.ReadBytes(jsonLength))
                ?? throw new InvalidDataException("Checkpoint header failed to parse.");

            var config = new ModelConfig(header.VocabSize, header.ContextLength, header.DModel, header.NLayers, header.NHeads);
            var model = new GptModel(config, backend, new Random(0)); // init values are overwritten below

            string[] actual = model.Params.Names.ToArray();
            if (!header.Names.SequenceEqual(actual))
                throw new InvalidDataException("Checkpoint parameter names do not match the model registry.");

            long expectedBytes = model.Params.Count * sizeof(float);
            if (fs.Length - fs.Position != expectedBytes)
                throw new InvalidDataException(
                    $"Checkpoint weight payload is {fs.Length - fs.Position} bytes, expected {expectedBytes} ({model.Params.Count} floats).");

            foreach (string name in actual)
            {
                Tensor w = model.Params.Weight(name);
                for (int i = 0; i < w.Length; i++)
                    w.Data[i] = br.ReadSingle();
            }
            return model;
        }
    }
}
