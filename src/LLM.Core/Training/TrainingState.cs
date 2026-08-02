namespace LLM.Core.Training
{
    using LLM.Core.Tensor;

    /// <summary>The trajectory-defining settings stored with a training checkpoint.</summary>
    public sealed record TrainingConfiguration(
        int TotalSteps,
        float MaxLr,
        float MinLr,
        int WarmupSteps,
        float WeightDecay,
        float GradClip,
        int BatchSize,
        int AccumulationSteps,
        int ContextLength);

    /// <summary>
    /// Mutable state required to continue training rather than merely reload weights:
    /// cumulative scheduler position, Adam moments/age, and data-sampler RNG state.
    /// </summary>
    public sealed class TrainingState
    {
        internal TrainingState(int globalStep, AdamW optimizer, TrainingRandom dataRandom,
            TrainingConfiguration configuration, string? dataIdentity = null, string? tokenizerIdentity = null)
        {
            if (globalStep < 0) throw new ArgumentOutOfRangeException(nameof(globalStep));
            GlobalStep = globalStep;
            Optimizer = optimizer;
            DataRandom = dataRandom;
            Configuration = configuration;
            DataIdentity = dataIdentity;
            TokenizerIdentity = tokenizerIdentity;
        }

        public int GlobalStep { get; internal set; }
        public AdamW Optimizer { get; }
        public TrainingRandom DataRandom { get; }
        public TrainingConfiguration Configuration { get; }
        public string? DataIdentity { get; private set; }
        public string? TokenizerIdentity { get; private set; }

        public static TrainingState CreateNew(ITensorBackend backend, TrainOptions options, int globalStep = 0,
            string? dataIdentity = null, string? tokenizerIdentity = null) =>
            new(globalStep, new AdamW(backend), new TrainingRandom(options.Seed), FromOptions(options),
                dataIdentity, tokenizerIdentity);

        /// <summary>Binds a legacy checkpoint once, or verifies that a bound checkpoint uses the same inputs.</summary>
        public void RequireDataIdentity(string dataIdentity, string tokenizerIdentity)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(dataIdentity);
            ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerIdentity);
            if (DataIdentity is not null && DataIdentity != dataIdentity)
                throw new InvalidDataException("Training data does not match the dataset recorded in the checkpoint.");
            if (TokenizerIdentity is not null && TokenizerIdentity != tokenizerIdentity)
                throw new InvalidDataException("Tokenizer does not match the tokenizer recorded in the checkpoint.");
            DataIdentity ??= dataIdentity;
            TokenizerIdentity ??= tokenizerIdentity;
        }

        internal static TrainingConfiguration FromOptions(TrainOptions options) => new(
            options.Steps,
            options.MaxLr,
            options.MinLr,
            options.WarmupSteps,
            options.WeightDecay,
            options.GradClip,
            options.BatchSize,
            options.AccumulationSteps,
            options.ContextLength);

        internal void RequireCompatible(TrainOptions options)
        {
            TrainingConfiguration requested = FromOptions(options);
            if (Configuration != requested)
                throw new InvalidOperationException(
                    $"Training options do not match the checkpoint. Stored: {Describe(Configuration)}. " +
                    $"Requested: {Describe(requested)}.");
            if (GlobalStep > options.Steps)
                throw new InvalidOperationException(
                    $"Checkpoint is at global step {GlobalStep:N0}, beyond target --steps {options.Steps:N0}.");
        }

        private static string Describe(TrainingConfiguration c) =>
            $"steps={c.TotalSteps}, lr={c.MaxLr:G9}, minlr={c.MinLr:G9}, warmup={c.WarmupSteps}, " +
            $"wd={c.WeightDecay:G9}, gradclip={c.GradClip:G9}, batch={c.BatchSize}, " +
            $"accum={c.AccumulationSteps}, ctx={c.ContextLength}";
    }
}
