namespace LLM.Core.Training;

/// <summary>
/// Random-access sampler over a prepared token file: raw little-endian uint16
/// token ids, no header. The model trains on single sequences, so one call to
/// <see cref="Sample"/> yields one (inputs, targets) pair offset by one token.
/// Splitting a corpus into train/val is the caller's job — see
/// <see cref="Split"/>.
/// </summary>
public sealed class DataLoader
{
    private readonly int[] _ids;

    /// <summary>Loads a raw little-endian uint16 token file fully into memory.</summary>
    public DataLoader(string path) : this(ReadIds(path)) { }

    /// <summary>Wraps an in-memory token array (e.g. one half of a <see cref="Split"/>).</summary>
    public DataLoader(int[] ids)
    {
        if (ids.Length < 2) throw new ArgumentException("Need at least 2 tokens.");
        _ids = ids;
    }

    /// <summary>Number of tokens in this split.</summary>
    public int Length => _ids.Length;

    /// <summary>
    /// Picks a uniform random offset and fills <paramref name="inputs"/> and
    /// <paramref name="targets"/> (both of length <paramref name="contextLength"/>)
    /// with inputs[i] = ids[o+i], targets[i] = ids[o+i+1].
    /// </summary>
    public void Sample(Random rng, int contextLength, int[] inputs, int[] targets)
    {
        if (inputs.Length != contextLength || targets.Length != contextLength)
            throw new ArgumentException("inputs and targets must both have length contextLength.");
        if (_ids.Length < contextLength + 1)
            throw new ArgumentException($"Not enough tokens ({_ids.Length}) for context length {contextLength}.");
        int o = rng.Next(_ids.Length - contextLength);
        for (int i = 0; i < contextLength; i++)
        {
            inputs[i] = _ids[o + i];
            targets[i] = _ids[o + i + 1];
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

    private static int[] ReadIds(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length % 2 != 0)
            throw new InvalidDataException($"Token file '{path}' has odd byte count {bytes.Length}; expected raw uint16 data.");
        var ids = new int[bytes.Length / 2];
        for (int i = 0; i < ids.Length; i++)
            ids[i] = BitConverter.ToUInt16(bytes, i * 2);
        return ids;
    }
}
