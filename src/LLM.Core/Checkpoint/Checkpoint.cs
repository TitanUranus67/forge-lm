namespace LLM.Core.Checkpoint
{
    using System.Text.Json;
    using System.Security.Cryptography;
    using System.Text;
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
        private static readonly byte[] MagicV3 = "LLMSCRATCH3"u8.ToArray();
        private const int ChecksumBytes = 32;

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
            public int AccumulationSteps { get; set; }
            public int TrainingContextLength { get; set; }
            public string? DataIdentity { get; set; }
            public string? TokenizerIdentity { get; set; }
        }

        public sealed record LoadedTrainingCheckpoint(GptModel Model, TrainingState? TrainingState,
            string? TokenizerIdentity = null)
        {
            public bool IsWeightsOnly => TrainingState is null;
        }

        /// <summary>Writes a legacy-compatible model-only checkpoint.</summary>
        public static void Save(GptModel model, string path)
        {
            Header header = CreateModelHeader(model);
            WriteHeaderAndWeights(model, path, MagicV1, header, writeExtraPayload: null, checksum: false);
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
                AccumulationSteps = c.AccumulationSteps,
                TrainingContextLength = c.ContextLength,
                DataIdentity = state.DataIdentity,
                TokenizerIdentity = state.TokenizerIdentity,
            };

            Dictionary<string, (Tensor M, Tensor V)> moments = state.Optimizer.StateEntries
                .ToDictionary(e => e.Name, e => (e.M, e.V));
            WriteHeaderAndWeights(model, path, MagicV3, header, bw =>
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
            }, checksum: true);
        }

        /// <summary>Loads either checkpoint version for inference, ignoring V2 optimizer payload.</summary>
        public static GptModel Load(string path, ITensorBackend backend) =>
            LoadCore(path, backend, restoreTrainingState: false).Model;

        /// <summary>Loads a model for inference together with checkpoint metadata such as tokenizer identity.</summary>
        public static LoadedTrainingCheckpoint LoadWithMetadata(string path, ITensorBackend backend) =>
            LoadCore(path, backend, restoreTrainingState: false);

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
            Header header, Action<BinaryWriter>? writeExtraPayload, bool checksum)
        {
            byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20,
                FileOptions.SequentialScan);
            using var hasher = checksum ? IncrementalHash.CreateHash(HashAlgorithmName.SHA256) : null;
            using var hashingStream = checksum ? new HashingWriteStream(fs, hasher!) : null;
            Stream payload = (Stream?)hashingStream ?? fs;
            using var bw = new BinaryWriter(payload, Encoding.UTF8, leaveOpen: true);
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
            bw.Flush();
            if (hasher is not null)
                fs.Write(hasher.GetHashAndReset());
            fs.Flush(flushToDisk: true);
        }

        private static void WriteTensor(BinaryWriter writer, Tensor tensor)
        {
            foreach (float value in tensor.Data)
                writer.Write(value);
        }

        private static LoadedTrainingCheckpoint LoadCore(string path, ITensorBackend backend,
            bool restoreTrainingState)
        {
            using var fs = OpenCheckpointRead(path);
            byte[] magic = new byte[MagicV1.Length];
            fs.ReadExactly(magic);
            bool isV1 = magic.SequenceEqual(MagicV1);
            bool isV2 = magic.SequenceEqual(MagicV2);
            bool isV3 = magic.SequenceEqual(MagicV3);
            if (!isV1 && !isV2 && !isV3)
                throw new InvalidDataException($"'{path}' is not an LLMSCRATCH checkpoint (bad magic).");

            long contentLength = isV3 ? fs.Length - ChecksumBytes : fs.Length;
            if (contentLength < fs.Position)
                throw new InvalidDataException("Checkpoint is too short to contain its checksum.");
            using var hasher = isV3 ? IncrementalHash.CreateHash(HashAlgorithmName.SHA256) : null;
            hasher?.AppendData(magic);
            using var hashingStream = isV3 ? new HashingReadStream(fs, hasher!, contentLength - fs.Position) : null;
            Stream payload = (Stream?)hashingStream ?? fs;
            using var br = new BinaryReader(payload, Encoding.UTF8, leaveOpen: true);

            int jsonLength = br.ReadInt32();
            if (jsonLength <= 0 || jsonLength > 1 << 20)
                throw new InvalidDataException($"Implausible header length {jsonLength}.");
            byte[] json = br.ReadBytes(jsonLength);
            if (json.Length != jsonLength)
                throw new InvalidDataException("Checkpoint ended inside its JSON header.");
            Header header = JsonSerializer.Deserialize<Header>(json)
                ?? throw new InvalidDataException("Checkpoint header failed to parse.");
            if ((isV2 || isV3) && header.Training is null)
                throw new InvalidDataException("Training checkpoint is missing its training-state header.");

            var config = new ModelConfig(header.VocabSize, header.ContextLength,
                header.DModel, header.NLayers, header.NHeads);
            var model = new GptModel(config, backend, new Random(0));
            string[] actualNames = model.Params.Names.ToArray();
            if (!header.Names.SequenceEqual(actualNames))
                throw new InvalidDataException("Checkpoint parameter names do not match the model registry.");

            long weightBytes = checked(model.Params.Count * sizeof(float));
            bool hasMoments = header.Training?.HasOptimizerState == true;
            long expectedPayloadBytes = checked(weightBytes * (hasMoments ? 3L : 1L));
            if (contentLength - fs.Position != expectedPayloadBytes)
                throw new InvalidDataException(
                    $"Checkpoint payload is {contentLength - fs.Position} bytes, expected {expectedPayloadBytes}.");

            foreach (string name in actualNames)
            {
                Tensor weight = model.Params.Weight(name);
                ReadTensor(br, weight);
                backend.InvalidateDeviceCache(weight);
            }

            if (isV1 || !restoreTrainingState)
            {
                if (isV3) DrainAndVerifyChecksum(fs, hashingStream!, hasher!);
                return new LoadedTrainingCheckpoint(model, null, header.Training?.TokenizerIdentity);
            }

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
                t.WarmupSteps, t.WeightDecay, t.GradClip, t.BatchSize,
                Math.Max(1, t.AccumulationSteps), t.TrainingContextLength);
            var rng = new TrainingRandom(t.DataRngState, t.DataRngIncrement);
            var state = new TrainingState(t.GlobalStep, optimizer, rng, trainingConfig,
                t.DataIdentity, t.TokenizerIdentity);
            if (isV3) DrainAndVerifyChecksum(fs, hashingStream!, hasher!);
            return new LoadedTrainingCheckpoint(model, state, t.TokenizerIdentity);
        }

        private static void DrainAndVerifyChecksum(FileStream fs, HashingReadStream hashingStream,
            IncrementalHash hasher)
        {
            Span<byte> buffer = stackalloc byte[8192];
            while (hashingStream.Read(buffer) != 0) { }
            byte[] actual = hasher.GetHashAndReset();
            byte[] expected = new byte[ChecksumBytes];
            fs.ReadExactly(expected);
            if (!CryptographicOperations.FixedTimeEquals(actual, expected))
                throw new InvalidDataException("Checkpoint SHA-256 checksum does not match; the file is corrupted.");
            if (fs.Position != fs.Length)
                throw new InvalidDataException("Checkpoint contains trailing data after its checksum.");
        }

        /// <summary>
        /// Opens a stable handle to the current checkpoint generation while allowing
        /// the trainer to atomically replace the path with a newer generation.
        /// Existing readers continue consuming the old file; new readers see the new one.
        /// </summary>
        internal static FileStream OpenCheckpointRead(string path) => new(path, new FileStreamOptions
        {
            Mode = FileMode.Open,
            Access = FileAccess.Read,
            Share = FileShare.ReadWrite | FileShare.Delete,
            BufferSize = 1 << 20,
            Options = FileOptions.SequentialScan,
        });

        /// <summary>Publishes a fully written temporary checkpoint without disrupting existing shared readers.</summary>
        public static void PublishAtomically(string temporaryPath, string destinationPath, string? backupPath = null)
        {
            if (File.Exists(destinationPath))
                File.Replace(temporaryPath, destinationPath, backupPath, ignoreMetadataErrors: true);
            else
                File.Move(temporaryPath, destinationPath);
        }

        private static void ReadTensor(BinaryReader reader, Tensor tensor)
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor.Data[i] = reader.ReadSingle();
        }

        private sealed class HashingWriteStream(Stream inner, IncrementalHash hasher) : Stream
        {
            public override bool CanRead => false;
            public override bool CanSeek => false;
            public override bool CanWrite => true;
            public override long Length => inner.Length;
            public override long Position { get => inner.Position; set => throw new NotSupportedException(); }
            public override void Flush() => inner.Flush();
            public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => inner.SetLength(value);

            public override void Write(byte[] buffer, int offset, int count)
            {
                hasher.AppendData(buffer, offset, count);
                inner.Write(buffer, offset, count);
            }

            public override void Write(ReadOnlySpan<byte> buffer)
            {
                hasher.AppendData(buffer);
                inner.Write(buffer);
            }

            protected override void Dispose(bool disposing) { }
        }

        private sealed class HashingReadStream : Stream
        {
            private readonly Stream _inner;
            private readonly IncrementalHash _hasher;
            private readonly long _length;
            private long _remaining;

            public HashingReadStream(Stream inner, IncrementalHash hasher, long remaining)
            {
                _inner = inner;
                _hasher = hasher;
                _length = remaining;
                _remaining = remaining;
            }

            public override bool CanRead => true;
            public override bool CanSeek => false;
            public override bool CanWrite => false;
            public override long Length => _length;
            public override long Position { get => _length - _remaining; set => throw new NotSupportedException(); }
            public override void Flush() { }
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

            public override int Read(byte[] buffer, int offset, int count)
            {
                int read = _inner.Read(buffer, offset, (int)Math.Min(count, _remaining));
                if (read > 0)
                {
                    _hasher.AppendData(buffer, offset, read);
                    _remaining -= read;
                }
                return read;
            }

            public override int Read(Span<byte> buffer)
            {
                int requested = (int)Math.Min(buffer.Length, _remaining);
                int read = _inner.Read(buffer[..requested]);
                if (read > 0)
                {
                    _hasher.AppendData(buffer[..read]);
                    _remaining -= read;
                }
                return read;
            }

            protected override void Dispose(bool disposing) { }
        }
    }
}
