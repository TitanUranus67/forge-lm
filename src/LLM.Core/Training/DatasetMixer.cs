using System.Buffers.Binary;

namespace LLM.Core.Training;

public sealed record DatasetMixSource(string Name, string TokenPath, double Weight);

public sealed record DatasetMixResult(long TokensWritten, IReadOnlyDictionary<string, long> SourceTokens);

/// <summary>
/// Deterministically interleaves complete EOS-terminated documents from compatible
/// token files. Selection minimizes emittedTokens / weight, keeping the running
/// token mixture close to the requested proportions without splitting documents.
/// </summary>
public static class DatasetMixer
{
    public static DatasetMixResult Mix(IReadOnlyList<DatasetMixSource> sources, string outputPath,
        long targetTokens, ushort eosToken)
    {
        if (sources.Count < 2) throw new ArgumentException("A mixture requires at least two sources.");
        if (targetTokens < 1) throw new ArgumentOutOfRangeException(nameof(targetTokens));
        if (sources.Any(source => string.IsNullOrWhiteSpace(source.Name) ||
                                  !double.IsFinite(source.Weight) || source.Weight <= 0))
            throw new ArgumentException("Every mixture source needs a name and a finite positive weight.");
        if (sources.Select(source => source.Name).Distinct(StringComparer.OrdinalIgnoreCase).Count() != sources.Count)
            throw new ArgumentException("Mixture source names must be unique.");

        double totalWeight = sources.Sum(source => source.Weight);
        for (int i = 0; i < sources.Count; i++)
        {
            var file = new FileInfo(sources[i].TokenPath);
            if (!file.Exists) throw new FileNotFoundException("Mixture token file not found.", file.FullName);
            if ((file.Length & 1) != 0) throw new InvalidDataException($"Token file '{file.FullName}' has an odd byte length.");
            long proportionalMinimum = (long)Math.Floor(targetTokens * (sources[i].Weight / totalWeight));
            if (file.Length / 2 < proportionalMinimum)
                throw new InvalidDataException(
                    $"Source '{sources[i].Name}' has {file.Length / 2:N0} tokens but its requested share " +
                    $"requires at least {proportionalMinimum:N0}.");
        }

        var readers = sources.Select(source => new EosDocumentReader(source.TokenPath, eosToken)).ToArray();
        var emitted = new long[sources.Count];
        long total = 0;
        try
        {
            string? parent = Path.GetDirectoryName(Path.GetFullPath(outputPath));
            if (parent is not null) Directory.CreateDirectory(parent);
            using var output = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.None,
                1 << 20, FileOptions.SequentialScan);
            while (total < targetTokens)
            {
                int selected = 0;
                double bestScore = emitted[0] / sources[0].Weight;
                for (int i = 1; i < sources.Count; i++)
                {
                    double score = emitted[i] / sources[i].Weight;
                    if (score < bestScore)
                    {
                        selected = i;
                        bestScore = score;
                    }
                }

                long documentTokens = readers[selected].CopyNextDocument(output);
                if (documentTokens == 0)
                    throw new InvalidDataException(
                        $"Source '{sources[selected].Name}' ended before the {targetTokens:N0}-token mixture was complete.");
                emitted[selected] = checked(emitted[selected] + documentTokens);
                total = checked(total + documentTokens);
            }
        }
        finally
        {
            foreach (EosDocumentReader reader in readers) reader.Dispose();
        }

        return new DatasetMixResult(total,
            sources.Select((source, i) => new KeyValuePair<string, long>(source.Name, emitted[i]))
                .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.OrdinalIgnoreCase));
    }

    private sealed class EosDocumentReader : IDisposable
    {
        private readonly FileStream _stream;
        private readonly ushort _eos;
        private readonly byte[] _buffer = new byte[1 << 20];
        private int _position;
        private int _length;

        public EosDocumentReader(string path, ushort eos)
        {
            _stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
                _buffer.Length, FileOptions.SequentialScan);
            _eos = eos;
        }

        public long CopyNextDocument(Stream output)
        {
            long tokens = 0;
            while (true)
            {
                if (_position == _length)
                {
                    _length = _stream.Read(_buffer);
                    _position = 0;
                    if (_length == 0)
                    {
                        if (tokens != 0)
                            throw new InvalidDataException($"Token file '{_stream.Name}' ends without EOS.");
                        return 0;
                    }
                    if ((_length & 1) != 0)
                    {
                        int finalByte = _stream.ReadByte();
                        if (finalByte < 0)
                            throw new InvalidDataException($"Token file '{_stream.Name}' is not aligned to uint16 tokens.");
                        _buffer[_length++] = (byte)finalByte;
                    }
                }

                int start = _position;
                while (_position < _length)
                {
                    ushort token = BinaryPrimitives.ReadUInt16LittleEndian(_buffer.AsSpan(_position, 2));
                    _position += 2;
                    tokens++;
                    if (token == _eos)
                    {
                        output.Write(_buffer, start, _position - start);
                        return tokens;
                    }
                }
                output.Write(_buffer, start, _position - start);
            }
        }

        public void Dispose() => _stream.Dispose();
    }
}
