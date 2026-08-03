using System.Buffers;
using System.Diagnostics;

namespace LLM.Core.Tokenizer;

/// <summary>
/// Encodes a byte corpus described by fixed-size document-index records. Documents
/// are tokenized concurrently, then written in original order to deterministic
/// train/validation uint16 streams.
/// </summary>
public static class IndexedDocumentEncoder
{
    private const int DocumentsPerWorkerBatch = 64;

    public sealed record Progress(long BytesRead, long TotalBytes, long TokensWritten,
        long DocumentsWritten, TimeSpan Elapsed);

    public sealed record Summary(long TrainTokens, long ValidationTokens, long Documents,
        long CorpusBytes, TimeSpan Elapsed);

    private sealed class DocumentWork
    {
        public required byte[] Buffer { get; init; }
        public required int Length { get; init; }
        public required bool Validation { get; init; }
        public int[]? Tokens { get; set; }
    }

    /// <summary>
    /// Encodes every indexed document followed by EOS. Each index record is a
    /// little-endian Int64 byte length, a Boolean validation flag, and a UInt64
    /// sampling key. At most <c>workers * 64</c> documents are buffered at once.
    /// </summary>
    public static Summary Encode(BpeTokenizer tokenizer, string corpusPath, string documentIndexPath,
        long expectedDocuments, string trainPath, string validationPath, int workers,
        long progressIntervalBytes = 50L << 20, Action<Progress>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentException.ThrowIfNullOrWhiteSpace(corpusPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(documentIndexPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(trainPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(validationPath);
        if (expectedDocuments < 0) throw new ArgumentOutOfRangeException(nameof(expectedDocuments));
        if (workers < 1) throw new ArgumentOutOfRangeException(nameof(workers));
        if (progressIntervalBytes < 1) throw new ArgumentOutOfRangeException(nameof(progressIntervalBytes));

        int batchCapacity = checked(Math.Min(4096, workers * DocumentsPerWorkerBatch));
        int eos = tokenizer.EosTokenId;
        long trainTokens = 0, validationTokens = 0, documents = 0;
        long nextReport = progressIntervalBytes;
        var stopwatch = Stopwatch.StartNew();
        using var input = new FileStream(corpusPath, FileMode.Open, FileAccess.Read, FileShare.Read,
            1 << 20, FileOptions.SequentialScan);
        using var index = new BinaryReader(new FileStream(documentIndexPath, FileMode.Open, FileAccess.Read,
            FileShare.Read, 1 << 20, FileOptions.SequentialScan));
        using var trainOutput = new BufferedTokenWriter(trainPath);
        using var validationOutput = new BufferedTokenWriter(validationPath);
        long totalBytes = input.Length;
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = workers,
            CancellationToken = cancellationToken,
        };
        var batch = new List<DocumentWork>(batchCapacity);

        while (index.BaseStream.Position < index.BaseStream.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();
            batch.Clear();
            while (batch.Count < batchCapacity && index.BaseStream.Position < index.BaseStream.Length)
            {
                long documentBytes = index.ReadInt64();
                bool validation = index.ReadBoolean();
                _ = index.ReadUInt64(); // tokenizer-sampling key
                if (documentBytes <= 0 || documentBytes > int.MaxValue)
                    throw new InvalidDataException($"invalid document length {documentBytes:N0}");

                byte[] rented = ArrayPool<byte>.Shared.Rent((int)documentBytes);
                try
                {
                    input.ReadExactly(rented.AsSpan(0, (int)documentBytes));
                    batch.Add(new DocumentWork
                    {
                        Buffer = rented,
                        Length = (int)documentBytes,
                        Validation = validation,
                    });
                }
                catch
                {
                    ArrayPool<byte>.Shared.Return(rented);
                    throw;
                }
            }

            try
            {
                Parallel.For(0, batch.Count, parallelOptions, i =>
                {
                    DocumentWork work = batch[i];
                    work.Tokens = tokenizer.Encode(work.Buffer.AsSpan(0, work.Length));
                });

                foreach (DocumentWork work in batch)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int[] ids = work.Tokens
                        ?? throw new InvalidOperationException("Document encoding did not produce tokens.");
                    BufferedTokenWriter output = work.Validation ? validationOutput : trainOutput;
                    output.Write(ids);
                    output.Write(eos);
                    long count = ids.Length + 1L;
                    if (work.Validation) validationTokens = checked(validationTokens + count);
                    else trainTokens = checked(trainTokens + count);
                    documents++;
                }
            }
            finally
            {
                foreach (DocumentWork work in batch)
                {
                    work.Tokens = null;
                    ArrayPool<byte>.Shared.Return(work.Buffer);
                }
            }

            if (input.Position >= nextReport)
            {
                while (nextReport <= input.Position) nextReport = checked(nextReport + progressIntervalBytes);
                onProgress?.Invoke(new Progress(input.Position, totalBytes,
                    trainTokens + validationTokens, documents, stopwatch.Elapsed));
            }
        }

        if (input.Position != input.Length)
            throw new InvalidDataException(
                $"Document index covers {input.Position:N0} of {input.Length:N0} corpus bytes.");
        if (documents != expectedDocuments)
            throw new InvalidDataException(
                $"Document index contains {documents:N0} entries, expected {expectedDocuments:N0}.");

        return new Summary(trainTokens, validationTokens, documents, totalBytes, stopwatch.Elapsed);
    }

    private sealed class BufferedTokenWriter : IDisposable
    {
        private readonly FileStream _output;
        private readonly byte[] _buffer = new byte[1 << 20];
        private int _fill;

        public BufferedTokenWriter(string path) =>
            _output = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None,
                1 << 20, FileOptions.SequentialScan);

        public void Write(ReadOnlySpan<int> ids)
        {
            foreach (int id in ids) Write(id);
        }

        public void Write(int id)
        {
            if ((uint)id > ushort.MaxValue)
                throw new InvalidDataException($"token id {id} exceeds the uint16 data format");
            if (_fill == _buffer.Length) FlushBuffer();
            _buffer[_fill++] = (byte)id;
            _buffer[_fill++] = (byte)(id >> 8);
        }

        public void Dispose()
        {
            FlushBuffer();
            _output.Dispose();
        }

        private void FlushBuffer()
        {
            if (_fill == 0) return;
            _output.Write(_buffer, 0, _fill);
            _fill = 0;
        }
    }
}
