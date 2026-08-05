namespace LLM.Core.Model;

/// <summary>
/// A reviewed model shape plus conservative starting geometry for benchmarking and training.
/// Token budgets remain explicit CLI choices: selecting a preset never starts or enlarges a run.
/// </summary>
public sealed record ModelPreset(
    string Key,
    string ModelName,
    int ContextLength,
    int DModel,
    int Layers,
    int Heads,
    int DefaultBatch,
    int DefaultAccumulation,
    string DefaultCheckpoint,
    long PlanningTokenBudget)
{
    public ModelConfig CreateConfig(int vocabSize) =>
        new(vocabSize, ContextLength, DModel, Layers, Heads);

    /// <summary>Exact tied-embedding GPT parameter count without allocating a model.</summary>
    public long ParameterCount(int vocabSize)
    {
        checked
        {
            long d = DModel;
            long embeddings = (long)vocabSize * d + (long)ContextLength * d;
            long block = 12L * d * d + 13L * d;
            return embeddings + (long)Layers * block + 2L * d;
        }
    }
}

/// <summary>Supported Forge architectures used by both train and benchmark.</summary>
public static class ModelPresets
{
    private static readonly ModelPreset[] Presets =
    [
        new("forge-98m", "Forge-98M", 512, 768, 12, 12, 4, 16,
            Path.Combine("out", "forge-98m.bin"), 1_024_000_000),
        new("forge-220m", "Forge-220M", 1024, 1024, 16, 16, 1, 32,
            Path.Combine("out", "forge-220m.bin"), 4_400_000_000),
        new("forge-320m", "Forge-320M", 1024, 1024, 24, 16, 1, 32,
            Path.Combine("out", "forge-320m.bin"), 6_400_000_000),
    ];

    public static IReadOnlyList<ModelPreset> All => Presets;

    public static ModelPreset Get(string key) =>
        Presets.FirstOrDefault(p => p.Key.Equals(key, StringComparison.OrdinalIgnoreCase))
        ?? throw new ArgumentException(
            $"Unknown model preset '{key}'. Expected {string.Join(", ", Presets.Select(p => p.Key))}, or custom.");
}
