namespace LLM.Core.Training;

/// <summary>
/// Checkpointable, no-replacement sampler over non-overlapping context windows.
/// Each epoch visits every full window exactly once using an affine permutation;
/// only the final short tail is skipped. The permutation costs O(1) memory even
/// for multi-billion-token corpora.
/// </summary>
public sealed class TrainingSampler
{
    private readonly TrainingRandom _random;

    public TrainingSampler(int seed) => _random = new TrainingRandom(seed);

    internal TrainingSampler(ulong randomState, ulong randomIncrement, long epoch,
        long cursor, long start, long stride, long chunkCount, int contextLength)
    {
        _random = new TrainingRandom(randomState, randomIncrement);
        Epoch = epoch;
        Cursor = cursor;
        Start = start;
        Stride = stride;
        ChunkCount = chunkCount;
        ContextLength = contextLength;
    }

    internal ulong RandomState => _random.State;
    internal ulong RandomIncrement => _random.Increment;
    internal long Epoch { get; private set; }
    internal long Cursor { get; private set; }
    internal long Start { get; private set; }
    internal long Stride { get; private set; }
    internal long ChunkCount { get; private set; }
    internal int ContextLength { get; private set; }

    internal long NextOffset(long tokenCount, int contextLength)
    {
        long chunks = (tokenCount - 1) / contextLength;
        if (chunks < 1)
            throw new ArgumentException($"Not enough tokens ({tokenCount}) for context length {contextLength}.");

        if (ChunkCount == 0)
        {
            ChunkCount = chunks;
            ContextLength = contextLength;
            BeginEpoch(initial: true);
        }
        else if (ChunkCount != chunks || ContextLength != contextLength)
        {
            throw new InvalidOperationException(
                $"Sampler was initialized for {ChunkCount:N0} chunks of {ContextLength} tokens, " +
                $"not {chunks:N0} chunks of {contextLength} tokens.");
        }

        if (Cursor == ChunkCount)
            BeginEpoch(initial: false);

        ulong index = (ulong)(((UInt128)(ulong)Cursor * (ulong)Stride + (ulong)Start) % (ulong)ChunkCount);
        Cursor++;
        return checked((long)index * contextLength);
    }

    private void BeginEpoch(bool initial)
    {
        if (!initial) Epoch++;
        Cursor = 0;
        if (ChunkCount == 1)
        {
            Start = 0;
            Stride = 0;
            return;
        }

        Start = _random.NextInt64(ChunkCount);
        do
        {
            Stride = _random.NextInt64(ChunkCount - 1) + 1;
        } while (GreatestCommonDivisor(Stride, ChunkCount) != 1);
    }

    private static long GreatestCommonDivisor(long a, long b)
    {
        while (b != 0)
            (a, b) = (b, a % b);
        return a;
    }
}
