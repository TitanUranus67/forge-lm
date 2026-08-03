using System.Text;
using LLM.Core.Tokenizer;

namespace LLM.Core.Tests;

public static class IndexedDocumentEncoderTests
{
    [Test]
    public static void ParallelEncoding_IsByteIdenticalToDocumentOrderReference()
    {
        byte[][] documents = Enumerable.Range(0, 257)
            .Select(i => Encoding.UTF8.GetBytes(i == 3
                ? new string('x', 100_000) + "\n"
                : $"document {i}: héllo world {new string((char)('a' + i % 26), i % 97)}\n"))
            .ToArray();
        bool[] validation = Enumerable.Range(0, documents.Length).Select(i => i % 7 == 0).ToArray();
        var tokenizer = BpeTokenizer.TrainDocuments(documents.Take(64).ToArray(), 80);
        string dir = Path.Combine(Path.GetTempPath(), $"forge-encoder-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        string corpusPath = Path.Combine(dir, "corpus.txt");
        string indexPath = Path.Combine(dir, "corpus.idx");
        string trainPath = Path.Combine(dir, "train.bin");
        string valPath = Path.Combine(dir, "val.bin");

        try
        {
            using (var corpus = File.Create(corpusPath))
            using (var index = new BinaryWriter(File.Create(indexPath)))
            {
                for (int i = 0; i < documents.Length; i++)
                {
                    corpus.Write(documents[i]);
                    index.Write((long)documents[i].Length);
                    index.Write(validation[i]);
                    index.Write((ulong)i);
                }
            }

            IndexedDocumentEncoder.Progress? lastProgress = null;
            IndexedDocumentEncoder.Summary summary = IndexedDocumentEncoder.Encode(tokenizer,
                corpusPath, indexPath, documents.Length, trainPath, valPath, workers: 4,
                progressIntervalBytes: 1, onProgress: progress => lastProgress = progress);

            byte[] expectedTrain = ReferenceBytes(tokenizer, documents, validation, selectValidation: false);
            byte[] expectedValidation = ReferenceBytes(tokenizer, documents, validation, selectValidation: true);
            Check.True(File.ReadAllBytes(trainPath).SequenceEqual(expectedTrain),
                "parallel train output is byte-identical to ordered reference");
            Check.True(File.ReadAllBytes(valPath).SequenceEqual(expectedValidation),
                "parallel validation output is byte-identical to ordered reference");
            Check.True(summary.TrainTokens * 2 == expectedTrain.Length, "train token count matches output");
            Check.True(summary.ValidationTokens * 2 == expectedValidation.Length,
                "validation token count matches output");
            Check.True(summary.Documents == documents.Length, "every indexed document is written once");
            Check.True(summary.CorpusBytes == documents.Sum(d => (long)d.Length), "all corpus bytes are read");
            Check.True(lastProgress?.DocumentsWritten == documents.Length, "progress reports ordered completion");
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Test]
    public static void InvalidWorkerCount_FailsBeforeCreatingOutputs()
    {
        var tokenizer = BpeTokenizer.Train(Array.Empty<byte>(), 0);
        bool threw = false;
        try
        {
            _ = IndexedDocumentEncoder.Encode(tokenizer, "missing-corpus", "missing-index", 0,
                "unused-train", "unused-val", workers: 0);
        }
        catch (ArgumentOutOfRangeException)
        {
            threw = true;
        }
        Check.True(threw, "zero encoding workers are rejected before files are opened");
    }

    private static byte[] ReferenceBytes(BpeTokenizer tokenizer, IReadOnlyList<byte[]> documents,
        IReadOnlyList<bool> validation, bool selectValidation)
    {
        using var output = new MemoryStream();
        using var writer = new BinaryWriter(output);
        for (int i = 0; i < documents.Count; i++)
        {
            if (validation[i] != selectValidation) continue;
            foreach (int id in tokenizer.EncodeDocument(documents[i])) writer.Write((ushort)id);
        }
        return output.ToArray();
    }
}
