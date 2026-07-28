namespace LLM.Core.Model;

/// <summary>
/// Hyperparameters of a GPT model. The MLP hidden size is fixed at 4 * DModel.
/// Construction throws on inconsistent values (non-positive sizes, DModel not
/// divisible by NHeads).
/// </summary>
public sealed record ModelConfig(int VocabSize, int ContextLength, int DModel, int NLayers, int NHeads)
{
    /// <summary>Per-head attention dimension (DModel / NHeads). Runs validation.</summary>
    public int HeadDim { get; } = Validate(VocabSize, ContextLength, DModel, NLayers, NHeads);

    /// <summary>MLP hidden size, fixed at 4 * DModel.</summary>
    public int MlpHidden { get; } = 4 * DModel;

    private static int Validate(int vocabSize, int contextLength, int dModel, int nLayers, int nHeads)
    {
        if (vocabSize <= 0) throw new ArgumentException("VocabSize must be positive.");
        if (contextLength <= 0) throw new ArgumentException("ContextLength must be positive.");
        if (dModel <= 0) throw new ArgumentException("DModel must be positive.");
        if (nLayers <= 0) throw new ArgumentException("NLayers must be positive.");
        if (nHeads <= 0) throw new ArgumentException("NHeads must be positive.");
        if (dModel % nHeads != 0) throw new ArgumentException($"DModel ({dModel}) must be divisible by NHeads ({nHeads}).");
        return dModel / nHeads;
    }
}
