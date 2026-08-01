namespace LLM.Core.Checkpoint
{
    using System.Text.Json;
    using LLM.Core.Model;
    using LLM.Core.Tensor;
    using LLM.Core.Training;

    /// <summary>
    /// Versioned, deterministic binary checkpoints. V1 contains model configuration
    /// and weights. V2 additionally contains the complete resumable training state:
    /// cumulative global step, LR schedule/configuration, sampler RNG state, Adam age,
    /// and first/second moment tensors. Both versions remain loadable for inference.
    /// </summary>
    public static class Checkpoint
    {
        private static readonly byte[] MagicV1 = "LLMSCRATCH1"u8.ToArray();
        private static readonly byte[] MagicV2 = "LLMSCRATCH2"u8.ToArray();

        private sealed class Header
        {
            public int VocabSize { get; set; }
            public int ContextLength { get; set; }
            public int DModel { get; set; }
            public int NLayers { get; set; }
            public int NHeads { get; set; }
            public string[] Names { get; set; } = [];
            public TrainingHeader? Training { get; set; }
        }

        private sealed class TrainingHeader
        {
            public int GlobalStep { get; set; }
            public int AdamStep { get; set; }
            public ulong DataRngState { get; set; }
            public ulong DataRngIncrement { get; set; }
            public bool HasOptimizerState { get; set; }
            public int TotalSteps { get; set; }
            public float MaxLr { get; set; }
            public float MinLr { get; set; }
            public int WarmupSteps { get; set; }
            public float WeightDecay { get; set; }
            public float GradClip { get; set; }
            public int BatchSize { get; set; }
            public int TrainingContextLength { get; set; }
        }

        public sealed record LoadedTrainingCheckpoint(GptModel Model, TrainingState? TrainingState)
        {
            public bool IsWeightsOnly => TrainingState is null;
        }

        /// <summary>Writes a legacy-compatible model-only checkpoint.</summary>
        public static void Save(GptModel model, string path)
        {
            Header header = CreateModelHeader(model);
            WriteHeaderAndWeights(model, path, MagicV1, header, writeExtraPayload: null);
        }

        /// <summary>Writes model weights plus all state required for exact training continuation.</summary>
        public static void SaveTraining(GptModel model, TrainingState state, string path)
        {
            if (state.Optimizer.StepCount > 0 &&
                state.Optimizer.StateEntries.Select(e => e.Name).SequenceEqual(model.Params.Names) is false)
                throw new InvalidOperationException("Adam state does not match the model parameter registry.");

            TrainingConfiguration c = state.Configuration;
            bool hasOptimizerState = state.Optimizer.StepCount > 0;
            Header header = CreateModelHeader(model);
            header.Training = new TrainingHeader
            {
                GlobalStep = state.GlobalStep,
                AdamStep = state.Optimizer.StepCount,
                DataRngState = state.DataRandom.State,
                DataRngIncrement = state.DataRandom.Increment,
                HasOptimizerState = hasOptimizerState,
                TotalSteps = c.TotalSteps,
                MaxLr = c.MaxLr,
                MinLr = c.MinLr,
                WarmupSteps = c.WarmupSteps,
                WeightDecay = c.WeightDecay,
                GradClip = c.GradClip,
                BatchSize = c.BatchSize,
                TrainingContextLength = c.ContextLength,
            };

            Dictionary<string, (Tensor M, Tensor V)> moments = state.Optimizer.StateEntries
                .ToDictionary(e => e.Name, e => (e.M, e.V));
            WriteHeaderAndWeights(model, path, MagicV2, header, bw =>
            {
                if (!hasOptimizerState) return;
                foreach (string name in model.Params.Names)
                {
                    var (m, v) = moments[name];
                    model.Backend.EnsureHostCurrent(m);
                    model.Backend.EnsureHostCurrent(v);
                    WriteTensor(bw, m);
                    WriteTensor(bw, v);
                }
            });
        }

        /// <summary>Loads either checkpoint version for inference, ignoring V2 optimizer payload.</summary>
        public static GptModel Load(string path, ITensorBackend backend) =>
            LoadCore(path, backend, restoreTrainingState: false).Model;

        /// <summary>
        /// Loads model and training state. V1 files return a null state because their
        /// optimizer, scheduler, global step, and RNG were never serialized.
        /// </summary>
        public static LoadedTrainingCheckpoint LoadTraining(string path, ITensorBackend backend) =>
            LoadCore(path, backend, restoreTrainingState: true);

        private static Header CreateModelHeader(GptModel model) => new()
        {
            VocabSize = model.Config.VocabSize,
            ContextLength = model.Config.ContextLength,
            DModel = model.Config.DModel,
            NLayers = model.Config.NLayers,
            NHeads = model.Config.NHeads,
            Names = model.Params.Names.ToArray(),
        };

        private static void WriteHeaderAndWeights(GptModel model, string path, byte[] magic,
            Header header, Action<BinaryWriter>? writeExtraPayload)
        {
            byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);
            bw.Write(magic);
            bw.Write(json.Length);
            bw.Write(json);
            foreach (string name in model.Params.Names)
            {
                Tensor weight = model.Params.Weight(name);
                model.Backend.EnsureHostCurrent(weight);
                WriteTensor(bw, weight);
            }
            writeExtraPayload?.Invoke(bw);
        }

        private static void WriteTensor(BinaryWriter writer, Tensor tensor)
        {
            foreach (float value in tensor.Data)
                writer.Write(value);
        }

        private static LoadedTrainingCheckpoint LoadCore(string path, ITensorBackend backend,
            bool restoreTrainingState)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs);

            byte[] magic = br.ReadBytes(MagicV1.Length);
            bool isV1 = magic.SequenceEqual(MagicV1);
            bool isV2 = magic.SequenceEqual(MagicV2);
            if (!isV1 && !isV2)
                throw new InvalidDataException($"'{path}' is not an LLMSCRATCH checkpoint (bad magic).");

            int jsonLength = br.ReadInt32();
            if (jsonLength <= 0 || jsonLength > 1 << 20)
                throw new InvalidDataException($"Implausible header length {jsonLength}.");
            byte[] json = br.ReadBytes(jsonLength);
            if (json.Length != jsonLength)
                throw new InvalidDataException("Checkpoint ended inside its JSON header.");
            Header header = JsonSerializer.Deserialize<Header>(json)
                ?? throw new InvalidDataException("Checkpoint header failed to parse.");
            if (isV2 && header.Training is null)
                throw new InvalidDataException("V2 checkpoint is missing its training-state header.");

            var config = new ModelConfig(header.VocabSize, header.ContextLength,
                header.DModel, header.NLayers, header.NHeads);
            var model = new GptModel(config, backend, new Random(0));
            string[] actualNames = model.Params.Names.ToArray();
            if (!header.Names.SequenceEqual(actualNames))
                throw new InvalidDataException("Checkpoint parameter names do not match the model registry.");

            long weightBytes = checked(model.Params.Count * sizeof(float));
            bool hasMoments = header.Training?.HasOptimizerState == true;
            long expectedPayloadBytes = checked(weightBytes * (hasMoments ? 3L : 1L));
            if (fs.Length - fs.Position != expectedPayloadBytes)
                throw new InvalidDataException(
                    $"Checkpoint payload is {fs.Length - fs.Position} bytes, expected {expectedPayloadBytes}.");

            foreach (string name in actualNames)
            {
                Tensor weight = model.Params.Weight(name);
                ReadTensor(br, weight);
                backend.InvalidateDeviceCache(weight);
            }

            if (!isV2 || !restoreTrainingState)
                return new LoadedTrainingCheckpoint(model, null);

            TrainingHeader t = header.Training!;
            var optimizer = new AdamW(backend);
            var entries = new List<(string Name, Tensor M, Tensor V)>();
            if (hasMoments)
            {
                foreach (string name in actualNames)
                {
                    Tensor weight = model.Params.Weight(name);
                    var m = new Tensor(weight.Shape);
                    var v = new Tensor(weight.Shape);
                    ReadTensor(br, m);
                    ReadTensor(br, v);
                    backend.InvalidateDeviceCache(m);
                    backend.InvalidateDeviceCache(v);
                    entries.Add((name, m, v));
                }
            }
            optimizer.RestoreState(model.Params, t.AdamStep, entries);

            var trainingConfig = new TrainingConfiguration(t.TotalSteps, t.MaxLr, t.MinLr,
                t.WarmupSteps, t.WeightDecay, t.GradClip, t.BatchSize, t.TrainingContextLength);
            var rng = new TrainingRandom(t.DataRngState, t.DataRngIncrement);
            var state = new TrainingState(t.GlobalStep, optimizer, rng, trainingConfig);
            return new LoadedTrainingCheckpoint(model, state);
        }

        private static void ReadTensor(BinaryReader reader, Tensor tensor)
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor.Data[i] = reader.ReadSingle();
        }
    }
}
