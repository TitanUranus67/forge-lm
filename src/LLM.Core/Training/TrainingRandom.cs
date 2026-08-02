namespace LLM.Core.Training;

/// <summary>
/// Small deterministic PCG32 generator whose complete state can be checkpointed.
/// <see cref="System.Random"/> does not expose a portable state snapshot, so it
/// cannot provide exact data-sampling continuation after a process restart.
/// </summary>
public sealed class TrainingRandom
{
    private ulong _state;
    private readonly ulong _increment;

    public TrainingRandom(int seed)
    {
        _increment = 0xDA3E39CB94B95BDBUL;
        _state = 0;
        NextUInt32();
        _state += unchecked((uint)seed);
        NextUInt32();
    }

    internal TrainingRandom(ulong state, ulong increment)
    {
        if ((increment & 1UL) == 0)
            throw new InvalidDataException("Training RNG increment must be odd.");
        _state = state;
        _increment = increment;
    }

    internal ulong State => _state;
    internal ulong Increment => _increment;

    /// <summary>Returns an unbiased integer in [0, <paramref name="maxExclusive"/>).</summary>
    public int Next(int maxExclusive)
    {
        if (maxExclusive <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxExclusive), "Upper bound must be positive.");

        uint bound = (uint)maxExclusive;
        uint threshold = unchecked(0u - bound) % bound;
        while (true)
        {
            uint value = NextUInt32();
            if (value >= threshold)
                return (int)(value % bound);
        }
    }

    /// <summary>Returns an unbiased integer in [0, <paramref name="maxExclusive"/>) for bounds above Int32.</summary>
    public long NextInt64(long maxExclusive)
    {
        if (maxExclusive <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxExclusive), "Upper bound must be positive.");

        ulong bound = (ulong)maxExclusive;
        ulong threshold = unchecked(0UL - bound) % bound;
        while (true)
        {
            ulong value = ((ulong)NextUInt32() << 32) | NextUInt32();
            if (value >= threshold)
                return (long)(value % bound);
        }
    }

    private uint NextUInt32()
    {
        ulong oldState = _state;
        _state = unchecked(oldState * 6364136223846793005UL + _increment);
        uint xorShifted = (uint)(((oldState >> 18) ^ oldState) >> 27);
        int rotation = (int)(oldState >> 59);
        return (xorShifted >> rotation) | (xorShifted << ((-rotation) & 31));
    }
}
