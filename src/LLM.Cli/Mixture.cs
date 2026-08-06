using System.Text.Json;
using LLM.Core.Tokenizer;
using LLM.Core.Training;

internal static partial class Cli
{
    private const int MixtureManifestVersion = 1;

    private sealed record MixtureSourceSpec(string Name, string Path, double Weight);
    private sealed record MixtureSpec(int Version, long TrainTokens, long ValidationTokens,
        MixtureSourceSpec[] Sources);
    private sealed record ResolvedMixtureSource(string Name, string Path, double Weight,
        long TrainBytes, long ValidationBytes, string TokenizerSha256);
    private sealed record PublishedMixture(int Version, string SpecificationId, string TokenizerSha256,
        ResolvedMixtureSource[] Sources, long TrainTokens, long ValidationTokens,
        IReadOnlyDictionary<string, long> TrainSourceTokens,
        IReadOnlyDictionary<string, long> ValidationSourceTokens,
        long TrainBytes, long ValidationBytes);

    internal static int PrepareMixture(string[] args)
    {
        var p = new Args(args);
        if (p.Help)
        {
            Console.WriteLine("""
                forge prepare-mixture --manifest <mixture.json> --out <dir> [--rebuild true]

                  Builds one deterministic train/validation dataset from two or more
                  prepared source directories. All sources must use byte-identical
                  tokenizer.json files and contain EOS-terminated train.bin/val.bin.
                  Weighted scheduling interleaves complete documents, never token
                  fragments. Requested token counts are soft ceilings because the
                  final document is kept whole. Existing outputs are reused only when
                  their manifest, sources, tokenizer, and lengths still match.
                """);
            return 0;
        }

        string specificationPath = Path.GetFullPath(p.Require("manifest"));
        string outputDirectory = Path.GetFullPath(p.Require("out"));
        bool rebuild = p.GetBool("rebuild", false);
        p.Done();

        MixtureSpec spec;
        try
        {
            spec = JsonSerializer.Deserialize<MixtureSpec>(File.ReadAllText(specificationPath),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
                ?? throw new InvalidDataException("Mixture specification is empty.");
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"Mixture specification '{specificationPath}' is malformed.", ex);
        }
        if (spec.Version != MixtureManifestVersion)
            throw new InvalidDataException($"Mixture specification version must be {MixtureManifestVersion}.");
        if (spec.TrainTokens < 1 || spec.ValidationTokens < 1)
            throw new InvalidDataException("Mixture trainTokens and validationTokens must be positive.");
        if (spec.Sources is null || spec.Sources.Length < 2)
            throw new InvalidDataException("Mixture specification requires at least two sources.");

        string specDirectory = Path.GetDirectoryName(specificationPath)!;
        var resolved = new List<ResolvedMixtureSource>(spec.Sources.Length);
        foreach (MixtureSourceSpec source in spec.Sources)
        {
            if (string.IsNullOrWhiteSpace(source.Name) || string.IsNullOrWhiteSpace(source.Path) ||
                !double.IsFinite(source.Weight) || source.Weight <= 0)
                throw new InvalidDataException("Every mixture source needs a name, path, and finite positive weight.");
            string directory = Path.GetFullPath(source.Path, specDirectory);
            if (directory.Equals(outputDirectory, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    $"Mixture output directory cannot overwrite source '{source.Name}': {directory}");
            string tokenizerPath = Path.Combine(directory, "tokenizer.json");
            string trainPath = Path.Combine(directory, "train.bin");
            string validationPath = Path.Combine(directory, "val.bin");
            if (!File.Exists(tokenizerPath) || !File.Exists(trainPath) || !File.Exists(validationPath))
                throw new FileNotFoundException(
                    $"Mixture source '{source.Name}' must contain tokenizer.json, train.bin, and val.bin: {directory}");
            resolved.Add(new ResolvedMixtureSource(source.Name, directory, source.Weight,
                new FileInfo(trainPath).Length, new FileInfo(validationPath).Length, FileSha256(tokenizerPath)));
        }
        if (resolved.Select(source => source.Name).Distinct(StringComparer.OrdinalIgnoreCase).Count() != resolved.Count)
            throw new InvalidDataException("Mixture source names must be unique.");
        if (resolved.Select(source => source.Path).Distinct(StringComparer.OrdinalIgnoreCase).Count() != resolved.Count)
            throw new InvalidDataException("Mixture sources must use distinct directories.");
        string tokenizerSha = resolved[0].TokenizerSha256;
        if (resolved.Any(source => source.TokenizerSha256 != tokenizerSha))
            throw new InvalidDataException("All mixture sources must use byte-identical tokenizer.json files.");

        string normalizedSpecId = ManifestId(new
        {
            spec.Version,
            spec.TrainTokens,
            spec.ValidationTokens,
            Sources = resolved,
        });
        string tokenizerOutput = Path.Combine(outputDirectory, "tokenizer.json");
        string trainOutput = Path.Combine(outputDirectory, "train.bin");
        string validationOutput = Path.Combine(outputDirectory, "val.bin");
        string publishedPath = Path.Combine(outputDirectory, ".forge-mixture.json");
        PublishedMixture? published = ReadManifest<PublishedMixture>(publishedPath);
        bool verified = published is not null && published.Version == MixtureManifestVersion &&
            published.SpecificationId == normalizedSpecId && published.TokenizerSha256 == tokenizerSha &&
            File.Exists(tokenizerOutput) && FileSha256(tokenizerOutput) == tokenizerSha &&
            File.Exists(trainOutput) && new FileInfo(trainOutput).Length == published.TrainBytes &&
            File.Exists(validationOutput) && new FileInfo(validationOutput).Length == published.ValidationBytes;
        if (verified && !rebuild)
        {
            Console.WriteLine($"mixture: reusing verified {trainOutput} ({published!.TrainTokens:N0} tokens) and " +
                              $"{validationOutput} ({published.ValidationTokens:N0} tokens)");
            return 0;
        }
        if (!rebuild && (File.Exists(tokenizerOutput) || File.Exists(trainOutput) ||
                         File.Exists(validationOutput) || File.Exists(publishedPath)))
            throw new InvalidDataException(
                $"Existing mixture in '{outputDirectory}' is incomplete or stale; use --rebuild true to replace it.");

        Directory.CreateDirectory(outputDirectory);
        string tokenizerTemp = tokenizerOutput + ".tmp";
        string trainTemp = trainOutput + ".tmp";
        string validationTemp = validationOutput + ".tmp";
        foreach (string temp in new[] { tokenizerTemp, trainTemp, validationTemp })
            if (File.Exists(temp)) File.Delete(temp);

        try
        {
            File.Copy(Path.Combine(resolved[0].Path, "tokenizer.json"), tokenizerTemp, overwrite: true);
            var tokenizer = BpeTokenizer.Load(tokenizerTemp);
            ushort eos = checked((ushort)tokenizer.EosTokenId);
            DatasetMixSource[] trainSources = resolved.Select(source =>
                new DatasetMixSource(source.Name, Path.Combine(source.Path, "train.bin"), source.Weight)).ToArray();
            DatasetMixSource[] validationSources = resolved.Select(source =>
                new DatasetMixSource(source.Name, Path.Combine(source.Path, "val.bin"), source.Weight)).ToArray();

            Console.WriteLine($"mixture: writing approximately {spec.TrainTokens:N0} train tokens...");
            DatasetMixResult train = DatasetMixer.Mix(trainSources, trainTemp, spec.TrainTokens, eos);
            Console.WriteLine($"mixture: writing approximately {spec.ValidationTokens:N0} validation tokens...");
            DatasetMixResult validation = DatasetMixer.Mix(validationSources, validationTemp,
                spec.ValidationTokens, eos);

            File.Move(tokenizerTemp, tokenizerOutput, overwrite: true);
            File.Move(trainTemp, trainOutput, overwrite: true);
            File.Move(validationTemp, validationOutput, overwrite: true);
            published = new PublishedMixture(MixtureManifestVersion, normalizedSpecId, tokenizerSha,
                resolved.ToArray(), train.TokensWritten, validation.TokensWritten,
                train.SourceTokens, validation.SourceTokens,
                new FileInfo(trainOutput).Length, new FileInfo(validationOutput).Length);
            WriteManifest(publishedPath, published);
            Console.WriteLine($"mixture: wrote {train.TokensWritten:N0} train and " +
                              $"{validation.TokensWritten:N0} validation tokens to {outputDirectory}");
            foreach (ResolvedMixtureSource source in resolved)
                Console.WriteLine($"  {source.Name}: train {train.SourceTokens[source.Name]:N0}, " +
                                  $"validation {validation.SourceTokens[source.Name]:N0}");
            return 0;
        }
        finally
        {
            foreach (string temp in new[] { tokenizerTemp, trainTemp, validationTemp })
                if (File.Exists(temp)) File.Delete(temp);
        }
    }
}
