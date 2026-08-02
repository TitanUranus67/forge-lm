namespace LLM.Core.Training;

using System.IO.MemoryMappedFiles;

/// <summary>
/// Random-access sampler over a prepared token file: raw little-endian uint16
/// token ids, no header. The model trains on single sequences, so one call to
/// <see cref="Sample"/> yields one (inputs, targets) pair offset by one token.
/// Splitting a corpus into train/val is the caller's job — see
/// <see cref="Split"/>.
/// Files larger than <see cref="DefaultInMemoryLimit"/> are memory-mapped and
/// paged on demand instead of being loaded fully (train.bin can exceed 2GB).
/// </summary>
public sealed class DataLoader : IDisposable
{
    /// <summary>Files up to this many bytes are loaded into memory (faster sampling).</summary>
    public const long DefaultInMemoryLimit = 1L << 30; // 1 GiB

    private readonly int[]? _ids;
    private readonly MemoryMappedFile? _mmf;
    private readonly MemoryMappedViewAccessor? _view;
    private readonly long _length;

    /// <summary>Opens a raw little-endian uint16 token file (in-memory if small, memory-mapped if large).</summary>
    public DataLoader(string path) : this(path, DefaultInMemoryLimit) { }

    /// <summary>Like <see cref="DataLoader(string)"/>, with an explicit in-memory threshold in bytes.</summary>
    public DataLoader(string path, long inMemoryLimit)
    {
        long bytes = new FileInfo(path).Length;
        if (bytes % 2 != 0)
            throw new InvalidDataException($"Token file '{path}' has odd byte count {bytes}; expected raw uint16 data.");
        _length = bytes / 2;
        if (_length < 2) throw new ArgumentException("Need at least 2 tokens.");
        if (bytes <= inMemoryLimit)
        {
            _ids = ReadIds(path);
        }
        else
        {
            _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
            _view = _mmf.CreateViewAccessor(0, bytes, MemoryMappedFileAccess.Read);
        }
    }

    /// <summary>Wraps an in-memory token array (e.g. one half of a <see cref="Split"/>).</summary>
    public DataLoader(int[] ids)
    {
        if (ids.Length < 2) throw new ArgumentException("Need at least 2 tokens.");
        _ids = ids;
        _length = ids.Length;
    }

    /// <summary>Number of tokens in this split.</summary>
    public long Length => _length;

    /// <summary>
    /// Picks a uniform random offset and fills <paramref name="inputs"/> and
    /// <paramref name="targets"/> (both of length <paramref name="contextLength"/>)
    /// with inputs[i] = ids[o+i], targets[i] = ids[o+i+1].
    /// </summary>
    public void Sample(Random rng, int contextLength, int[] inputs, int[] targets)
    {
        ValidateSampleArguments(contextLength, inputs, targets);
        long bound = _length - contextLength;
        long offset = bound <= int.MaxValue ? rng.Next((int)bound) : rng.NextInt64(bound);
        FillSample(offset, contextLength, inputs, targets);
    }

    /// <summary>
    /// Checkpointable equivalent of <see cref="Sample(Random,int,int[],int[])"/> used by training.
    /// </summary>
    public void Sample(TrainingRandom rng, int contextLength, int[] inputs, int[] targets)
    {
        ValidateSampleArguments(contextLength, inputs, targets);
        long bound = _length - contextLength;
        long offset = bound <= int.MaxValue ? rng.Next((int)bound) : rng.NextInt64(bound);
        FillSample(offset, contextLength, inputs, targets);
    }

    private void ValidateSampleArguments(int contextLength, int[] inputs, int[] targets)
    {
        if (inputs.Length != contextLength || targets.Length != contextLength)
            throw new ArgumentException("inputs and targets must both have length contextLength.");
        if (_length < contextLength + 1)
            throw new ArgumentException($"Not enough tokens ({_length}) for context length {contextLength}.");
    }

    private void FillSample(long o, int contextLength, int[] inputs, int[] targets)
    {
        if (_ids is not null)
        {
            int offset = checked((int)o); // in-memory arrays cannot exceed Int32 length
            for (int i = 0; i < contextLength; i++)
            {
                inputs[i] = _ids[offset + i];
                targets[i] = _ids[offset + i + 1];
            }
        }
        else
        {
            long basePos = checked(o * 2);
            for (int i = 0; i < contextLength; i++)
            {
                inputs[i] = _view!.ReadUInt16(basePos + (long)i * 2);
                targets[i] = _view!.ReadUInt16(basePos + (long)(i + 1) * 2);
            }
        }
    }

    /// <summary>
    /// Splits an in-memory token array into train/val loaders; the first
    /// (1 - valFraction) of the tokens become the train split.
    /// </summary>
    public static (DataLoader Train, DataLoader Val) Split(int[] ids, float valFraction = 0.1f)
    {
        if (valFraction <= 0f || valFraction >= 1f)
            throw new ArgumentException("valFraction must be in (0,1).");
        int cut = (int)(ids.Length * (1f - valFraction));
        if (cut < 2 || ids.Length - cut < 2)
            throw new ArgumentException("Split leaves a split with fewer than 2 tokens.");
        return (new DataLoader(ids.AsSpan(0, cut).ToArray()), new DataLoader(ids.AsSpan(cut).ToArray()));
    }

    /// <summary>Releases the memory-mapped view, if any.</summary>
    public void Dispose()
    {
        _view?.Dispose();
        _mmf?.Dispose();
    }

    private static int[] ReadIds(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var ids = new int[bytes.Length / 2];
        for (int i = 0; i < ids.Length; i++)
            ids[i] = BitConverter.ToUInt16(bytes, i * 2);
        return ids;
    }
}
