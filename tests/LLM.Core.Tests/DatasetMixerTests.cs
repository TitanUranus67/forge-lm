using LLM.Core.Training;

namespace LLM.Core.Tests;

public static class DatasetMixerTests
{
    [Test]
    public static void Mix_IsDeterministicWeightedAndDocumentAligned()
    {
        string dir = Path.Combine(Path.GetTempPath(), "forge-mix-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string a = Path.Combine(dir, "a.bin");
            string b = Path.Combine(dir, "b.bin");
            WriteTokens(a, Enumerable.Range(0, 20).SelectMany(i => new ushort[] { 10, 11, 99 }));
            WriteTokens(b, Enumerable.Range(0, 20).SelectMany(i => new ushort[] { 20, 21, 99 }));
            DatasetMixSource[] sources = [new("a", a, 3), new("b", b, 1)];

            string first = Path.Combine(dir, "first.bin");
            string second = Path.Combine(dir, "second.bin");
            DatasetMixResult result = DatasetMixer.Mix(sources, first, 24, 99);
            DatasetMixer.Mix(sources, second, 24, 99);

            Check.True(File.ReadAllBytes(first).SequenceEqual(File.ReadAllBytes(second)), "mixture is deterministic");
            Check.True(result.TokensWritten == 24, "whole equal-size documents meet target exactly");
            Check.True(result.SourceTokens["a"] == 18 && result.SourceTokens["b"] == 6,
                "3:1 weights yield a 3:1 token mixture");
            ushort[] tokens = ReadTokens(first);
            for (int i = 2; i < tokens.Length; i += 3)
                Check.True(tokens[i] == 99, $"document {i / 3} ends in EOS");
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Test]
    public static void Mix_RejectsIncompatibleOrInsufficientInputs()
    {
        string dir = Path.Combine(Path.GetTempPath(), "forge-mix-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string a = Path.Combine(dir, "a.bin");
            string b = Path.Combine(dir, "b.bin");
            WriteTokens(a, [1, 99]);
            WriteTokens(b, [2, 99]);
            Check.Throws<InvalidDataException>(() => DatasetMixer.Mix(
                [new("a", a, 1), new("b", b, 1)], Path.Combine(dir, "out.bin"), 10, 99),
                "requires at least");

            File.WriteAllBytes(b, [1]);
            Check.Throws<InvalidDataException>(() => DatasetMixer.Mix(
                [new("a", a, 1), new("b", b, 1)], Path.Combine(dir, "out.bin"), 2, 99),
                "odd byte length");
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static void WriteTokens(string path, IEnumerable<ushort> tokens)
    {
        using var writer = new BinaryWriter(File.Create(path));
        foreach (ushort token in tokens) writer.Write(token);
    }

    private static ushort[] ReadTokens(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var result = new ushort[bytes.Length / 2];
        Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
        return result;
    }
}
