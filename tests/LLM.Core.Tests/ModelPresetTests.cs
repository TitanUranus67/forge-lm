using LLM.Core.Model;
using LLM.Core.Tensor;

namespace LLM.Core.Tests;

public static class ModelPresetTests
{
    [Test]
    public static void Registry_HasReviewedUniqueShapes()
    {
        Check.True(ModelPresets.All.Select(p => p.Key).Distinct(StringComparer.OrdinalIgnoreCase).Count() ==
                   ModelPresets.All.Count, "preset keys are unique");

        foreach (ModelPreset preset in ModelPresets.All)
        {
            ModelConfig config = preset.CreateConfig(16257);
            Check.True(config.HeadDim == 64, $"{preset.Key} uses 64-wide attention heads");
            Check.True(preset.DefaultBatch >= 1 && preset.DefaultAccumulation >= 1,
                $"{preset.Key} has valid batch geometry");
            Check.True(preset.PlanningTokenBudget > 0, $"{preset.Key} has a planning budget");
        }
    }

    [Test]
    public static void ParameterCount_MatchesAllocatedModel()
    {
        var small = new ModelPreset("test", "Test", 8, 16, 2, 2, 1, 1, "test.bin", 1);
        var model = new GptModel(small.CreateConfig(32), new CpuBackend(), new Random(1));
        Check.True(small.ParameterCount(32) == model.Params.Count,
            $"calculated {small.ParameterCount(32)} params should equal allocated {model.Params.Count}");
    }

    [Test]
    public static void CandidateCounts_AreStable()
    {
        Check.True(ModelPresets.Get("forge-98m").ParameterCount(16257) == 97_934_592,
            "Forge-98M parameter count");
        Check.True(ModelPresets.Get("forge-220m").ParameterCount(16257) == 219_237_376,
            "Forge-220M parameter count");
        Check.True(ModelPresets.Get("forge-320m").ParameterCount(16257) == 320_007_168,
            "Forge-320M parameter count");
    }

    [Test]
    public static void UnknownPreset_FailsClearly()
    {
        Check.Throws<ArgumentException>(() => ModelPresets.Get("forge-nope"), "Unknown model preset");
    }
}
