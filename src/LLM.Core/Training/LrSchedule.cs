namespace LLM.Core.Training;

/// <summary>
/// Linear warmup followed by cosine decay to a floor (the nanoGPT schedule).
/// </summary>
public static class LrSchedule
{
    /// <summary>
    /// Learning rate at a 0-based <paramref name="step"/>:
    /// ramps linearly from ~0 to <paramref name="maxLr"/> over the first
    /// <paramref name="warmupSteps"/> steps, then cosine-decays toward
    /// <paramref name="minLr"/>, reaching it at <paramref name="totalSteps"/>
    /// and staying there afterwards.
    /// </summary>
    public static float GetLr(int step, int totalSteps, float maxLr, float minLr, int warmupSteps)
    {
        if (totalSteps <= 0) throw new ArgumentException("totalSteps must be positive.");
        if (warmupSteps < 0) throw new ArgumentException("warmupSteps must be non-negative.");
        if (warmupSteps >= totalSteps) throw new ArgumentException("warmupSteps must be < totalSteps.");

        if (step < warmupSteps) return maxLr * (step + 1) / warmupSteps;
        if (step >= totalSteps) return minLr;
        float t = (float)(step - warmupSteps) / (totalSteps - warmupSteps); // in [0,1)
        return minLr + 0.5f * (maxLr - minLr) * (1f + MathF.Cos(MathF.PI * t));
    }
}
