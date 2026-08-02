using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using LLM.Core.Tokenizer;
using Parquet;
using Parquet.Schema;

// `prepare-fineweb`: download FineWeb (sample-10BT) parquet shards from
// Hugging Face, extract the `text` column into a corpus, then run the same
// tokenizer + uint16-bin flow as `prepare`, but streaming so a multi-GB
// corpus never has to fit in memory.

internal static partial class Cli
{
    private const string FineWebListUrl =
        "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT";
    private const string FineWebResolveUrl =
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/";

    private const int EncodeChunkSize = 50 << 20; // 50 MB encode chunks, split at newlines
    private const int FineWebManifestVersion = 1;

    private sealed record ShardIdentity(string Path, long Size);
    private sealed record CorpusManifest(int Version, ShardIdentity[] Shards, long CorpusBytes, long Documents);
    private sealed record TokenizerManifest(int Version, string CorpusId, int Merges, int TrainingMegabytes,
        int VocabularySize, string TokenizerSha256);
    private sealed record DataManifest(int Version, string CorpusId, string TokenizerSha256,
        long Tokens, long TrainBytes, long ValBytes);

    internal static int PrepareFineWeb(string[] args)
    {
        var p = new Args(args);
        if (p.Help)
        {
            Console.WriteLine("""
                llm prepare-fineweb --out <dir> [--shards 10] [--merges 16000] [--toktrainmb 200] [--rebuild true]

                  Downloads the first --shards parquet shards of FineWeb sample-10BT
                  from Hugging Face into <out>/shards (existing files are skipped,
                  so reruns resume), extracts the `text` column into <out>/corpus.txt,
                  trains a byte-level BPE tokenizer on the first --toktrainmb MB only,
                  then stream-encodes the full corpus in 50 MB chunks and writes the
                  usual tokenizer.json + train.bin/val.bin (90/10, LE uint16).
                  Derived artifacts are written transactionally and accompanied by
                  manifests. A stale or unverifiable corpus is never silently reused;
                  use --rebuild true to regenerate it while keeping downloaded shards.
                """);
            return 0;
        }

        string outDir = p.Require("out");
        int shards = p.GetInt("shards", 10);
        int merges = p.GetInt("merges", 16000);
        int tokTrainMb = p.GetInt("toktrainmb", 200);
        bool rebuild = p.GetBool("rebuild", false);
        p.Done();

        const int maxMerges = 65536 - 256; // uint16 token files cap the vocab at 65536
        if (merges < 0 || merges > maxMerges)
            throw new ArgumentException($"--merges must be in [0, {maxMerges}] (uint16 token ids cap vocab at 65536).");
        if (shards < 1) throw new ArgumentException("--shards must be >= 1");
        if (tokTrainMb < 1) throw new ArgumentException("--toktrainmb must be >= 1");

        string shardDir = Path.Combine(outDir, "shards");
        string corpusPath = Path.Combine(outDir, "corpus.txt");
        string corpusManifestPath = Path.Combine(outDir, ".fineweb-corpus.json");
        string tokenizerManifestPath = Path.Combine(outDir, ".fineweb-tokenizer.json");
        string dataManifestPath = Path.Combine(outDir, ".fineweb-data.json");
        Directory.CreateDirectory(shardDir);

        if (rebuild)
        {
            foreach (string path in new[]
            {
                corpusPath, corpusPath + ".tmp", corpusManifestPath,
                Path.Combine(outDir, "tokenizer.json"), Path.Combine(outDir, "tokenizer.json.tmp"), tokenizerManifestPath,
                Path.Combine(outDir, "tokens.tmp"), Path.Combine(outDir, "train.bin"), Path.Combine(outDir, "train.bin.tmp"),
                Path.Combine(outDir, "val.bin"), Path.Combine(outDir, "val.bin.tmp"), dataManifestPath,
            })
                if (File.Exists(path)) File.Delete(path);
        }

        using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        var swTotal = Stopwatch.StartNew();

        // 1. list + download shards (skip complete ones)
        List<(string Path, long Size)> all = ListFineWebShards(http);
        var wanted = all.Take(shards).ToList();
        Console.WriteLine($"fineweb: {all.Count} shards available, using first {wanted.Count}");
        var localShards = new List<string>(wanted.Count);
        foreach ((string path, long size) in wanted)
        {
            string local = Path.Combine(shardDir, Path.GetFileName(path));
            if (File.Exists(local) && new FileInfo(local).Length == size)
            {
                Console.WriteLine($"fineweb: reusing {local} ({size / 1e9:F2} GB)");
            }
            else
            {
                Download(http, FineWebResolveUrl + path, local, size);
            }
            localShards.Add(local);
        }

        ShardIdentity[] expectedShards = wanted.Select(s => new ShardIdentity(s.Path, s.Size)).ToArray();

        // 2. extract the text column -> corpus.txt (streamed, one \n between docs)
        long docs;
        CorpusManifest? corpusManifest = ReadManifest<CorpusManifest>(corpusManifestPath);
        bool corpusVerified = File.Exists(corpusPath) && corpusManifest is not null &&
            corpusManifest.Version == FineWebManifestVersion &&
            corpusManifest.Shards.SequenceEqual(expectedShards) &&
            new FileInfo(corpusPath).Length == corpusManifest.CorpusBytes;
        if (File.Exists(corpusPath) || File.Exists(corpusManifestPath))
        {
            if (!corpusVerified)
                throw new InvalidDataException(
                    $"Existing FineWeb corpus in '{outDir}' is incomplete, stale, or predates manifests. " +
                    "Refusing to reuse it; rerun with --rebuild true to regenerate derived artifacts.");
            docs = corpusManifest!.Documents;
            Console.WriteLine($"corpus: reusing {corpusPath} ({new FileInfo(corpusPath).Length / 1e9:F2} GB)");
        }
        else
        {
            string corpusTmp = corpusPath + ".tmp";
            if (File.Exists(corpusTmp)) File.Delete(corpusTmp);
            var sw = Stopwatch.StartNew();
            docs = 0;
            foreach (string shard in localShards)
                docs += ExtractText(shard, corpusTmp).GetAwaiter().GetResult();
            File.Move(corpusTmp, corpusPath, overwrite: true);
            corpusManifest = new CorpusManifest(FineWebManifestVersion, expectedShards,
                new FileInfo(corpusPath).Length, docs);
            WriteManifest(corpusManifestPath, corpusManifest);
            Console.WriteLine($"corpus: {docs:N0} docs, {new FileInfo(corpusPath).Length:N0} bytes " +
                              $"in {sw.Elapsed.TotalSeconds:F1}s");
        }
        long corpusBytes = new FileInfo(corpusPath).Length;
        string corpusId = ManifestId(corpusManifest!);

        // 3. tokenizer: reuse only when its manifest matches this corpus and configuration
        string tokOut = Path.Combine(outDir, "tokenizer.json");
        string tokTmp = tokOut + ".tmp";
        BpeTokenizer tok;
        TokenizerManifest? tokenizerManifest = ReadManifest<TokenizerManifest>(tokenizerManifestPath);
        bool tokenizerVerified = File.Exists(tokOut) && tokenizerManifest is not null &&
            tokenizerManifest.Version == FineWebManifestVersion &&
            tokenizerManifest.CorpusId == corpusId && tokenizerManifest.Merges == merges &&
            tokenizerManifest.TrainingMegabytes == tokTrainMb &&
            tokenizerManifest.TokenizerSha256 == FileSha256(tokOut);
        if (tokenizerVerified)
        {
            tok = BpeTokenizer.Load(tokOut);
            if (tok.VocabSize != tokenizerManifest!.VocabularySize)
                throw new InvalidDataException("FineWeb tokenizer manifest vocabulary size does not match tokenizer.json.");
            Console.WriteLine($"tokenizer: reusing {tokOut} (vocab {tok.VocabSize})");
        }
        else
        {
            if (File.Exists(tokTmp)) File.Delete(tokTmp);
            byte[] trainBytes = ReadHead(corpusPath, tokTrainMb << 20);
            Console.WriteLine($"tokenizer: training {merges} merges on first {trainBytes.Length / (1 << 20)} MB...");
            var sw = Stopwatch.StartNew();
            tok = BpeTokenizer.Train(trainBytes, merges, (done, total) =>
            {
                if (done % 100 == 0 || done == total)
                    Console.WriteLine($"tokenizer: {done}/{total} merges ({sw.Elapsed.TotalSeconds:F1}s)");
            });
            tok.Save(tokTmp);
            File.Move(tokTmp, tokOut, overwrite: true);
            tokenizerManifest = new TokenizerManifest(FineWebManifestVersion, corpusId, merges,
                tokTrainMb, tok.VocabSize, FileSha256(tokOut));
            WriteManifest(tokenizerManifestPath, tokenizerManifest);
            Console.WriteLine($"tokenizer: saved {tokOut} (vocab {tok.VocabSize})");
        }

        // 4. reuse verified bins, otherwise stream-encode to temp files and publish atomically
        string tmpBin = Path.Combine(outDir, "tokens.tmp");
        string trainPath = Path.Combine(outDir, "train.bin");
        string valPath = Path.Combine(outDir, "val.bin");
        string trainTmp = trainPath + ".tmp";
        string valTmp = valPath + ".tmp";
        string tokenizerSha = tokenizerManifest!.TokenizerSha256;
        DataManifest? dataManifest = ReadManifest<DataManifest>(dataManifestPath);
        bool dataVerified = File.Exists(trainPath) && File.Exists(valPath) && dataManifest is not null &&
            dataManifest.Version == FineWebManifestVersion && dataManifest.CorpusId == corpusId &&
            dataManifest.TokenizerSha256 == tokenizerSha &&
            new FileInfo(trainPath).Length == dataManifest.TrainBytes &&
            new FileInfo(valPath).Length == dataManifest.ValBytes;
        long tokens;
        if (dataVerified)
        {
            tokens = dataManifest!.Tokens;
            Console.WriteLine($"data: reusing verified train.bin/val.bin ({tokens:N0} tokens)");
        }
        else
        {
            foreach (string path in new[] { tmpBin, trainTmp, valTmp })
                if (File.Exists(path)) File.Delete(path);
            Console.WriteLine($"encoding {corpusBytes / 1e9:F2} GB corpus in {EncodeChunkSize >> 20} MB chunks...");
            var sw = Stopwatch.StartNew();
            tokens = StreamEncode(tok, corpusPath, tmpBin, sw);
            Console.WriteLine($"encoded {tokens:N0} tokens in {sw.Elapsed.TotalMinutes:F1} min " +
                              $"({tokens / Math.Max(sw.Elapsed.TotalSeconds, 1e-9):N0} tok/s)");
            long cutTokens = (long)(tokens * 0.9);
            SplitBin(tmpBin, cutTokens * 2, trainTmp, valTmp);
            File.Move(trainTmp, trainPath, overwrite: true);
            File.Move(valTmp, valPath, overwrite: true);
            File.Delete(tmpBin);
            dataManifest = new DataManifest(FineWebManifestVersion, corpusId, tokenizerSha, tokens,
                new FileInfo(trainPath).Length, new FileInfo(valPath).Length);
            WriteManifest(dataManifestPath, dataManifest);
        }

        long cut = new FileInfo(trainPath).Length / 2;
        Console.WriteLine($"train.bin: {cut:N0} tokens, val.bin: {tokens - cut:N0} tokens");
        Console.WriteLine($"stats: {corpusBytes:N0} bytes -> {tokens:N0} tokens, vocab {tok.VocabSize}, " +
                          $"compression {corpusBytes / (double)tokens:F2}x, " +
                          $"~{tokens / wanted.Count:N0} tokens/shard ({docs:N0} docs)");
        Console.WriteLine($"total elapsed: {swTotal.Elapsed:h\\:mm\\:ss}");
        return 0;
    }

    private static T? ReadManifest<T>(string path) where T : class
    {
        if (!File.Exists(path)) return null;
        try
        {
            return JsonSerializer.Deserialize<T>(File.ReadAllText(path));
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"Manifest '{path}' is malformed.", ex);
        }
    }

    private static void WriteManifest<T>(string path, T manifest)
    {
        string tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));
        File.Move(tmp, path, overwrite: true);
    }

    private static string ManifestId<T>(T manifest) =>
        Convert.ToHexString(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(manifest)));

    private static string FileSha256(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    /// <summary>Lists parquet shards of the sample-10BT config via the HF API, sorted by path.</summary>
    private static List<(string Path, long Size)> ListFineWebShards(HttpClient http)
    {
        string json = http.GetStringAsync(FineWebListUrl).GetAwaiter().GetResult();
        using var doc = JsonDocument.Parse(json);
        var shards = new List<(string, long)>();
        foreach (JsonElement e in doc.RootElement.EnumerateArray())
        {
            string? path = e.GetProperty("path").GetString();
            if (e.GetProperty("type").GetString() == "file" &&
                path is not null && path.EndsWith(".parquet", StringComparison.Ordinal))
                shards.Add((path, e.GetProperty("size").GetInt64()));
        }
        shards.Sort((a, b) => string.CompareOrdinal(a.Item1, b.Item1));
        if (shards.Count == 0)
            throw new InvalidDataException($"no parquet shards listed at {FineWebListUrl}");
        return shards;
    }

    /// <summary>Downloads one shard with progress; partial files go to *.part first.</summary>
    private static void Download(HttpClient http, string url, string local, long size)
    {
        string part = local + ".part";
        Console.WriteLine($"fineweb: downloading {url} ({size / 1e9:F2} GB)");
        var sw = Stopwatch.StartNew();
        using (var response = http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult())
        {
            response.EnsureSuccessStatusCode();
            using var src = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult();
            using var dst = new FileStream(part, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
            var buf = new byte[8 << 20];
            long got = 0, nextReport = 512 << 20;
            int r;
            while ((r = src.Read(buf, 0, buf.Length)) > 0)
            {
                dst.Write(buf, 0, r);
                got += r;
                if (got >= nextReport)
                {
                    Console.WriteLine($"fineweb: {got / 1e9:F2}/{size / 1e9:F2} GB " +
                                      $"({got / 1e6 / Math.Max(sw.Elapsed.TotalSeconds, 1e-9):F0} MB/s)");
                    nextReport += 512 << 20;
                }
            }
        }
        File.Move(part, local, overwrite: true);
        Console.WriteLine($"fineweb: saved {local} in {sw.Elapsed.TotalMinutes:F1} min");
    }

    /// <summary>Appends the `text` column of one parquet shard to the corpus; returns doc count.</summary>
    private static async Task<long> ExtractText(string shardPath, string corpusPath)
    {
        Console.WriteLine($"extract: {Path.GetFileName(shardPath)}");
        using var input = File.OpenRead(shardPath);
        await using var reader = await ParquetReader.CreateAsync(input);
        DataField textField = reader.Schema.GetDataFields().First(f => f.Name == "text");
        using var writer = new StreamWriter(corpusPath, append: true,
            new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false), 1 << 20);
        long docs = 0;
        for (int rg = 0; rg < reader.RowGroupCount; rg++)
        {
            using var rowGroup = reader.OpenRowGroupReader(rg);
            var rows = new string?[rowGroup.RowCount];
            await rowGroup.ReadAsync(textField, rows);
            foreach (string? text in rows)
            {
                if (!string.IsNullOrEmpty(text))
                {
                    writer.Write(text);
                    writer.Write('\n');
                    docs++;
                }
            }
        }
        return docs;
    }

    /// <summary>Reads the first up-to-<paramref name="limit"/> bytes, backing off to a newline.</summary>
    private static byte[] ReadHead(string path, int limit)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
        int n = (int)Math.Min(limit, fs.Length);
        var buf = new byte[n];
        int fill = 0;
        while (fill < n)
        {
            int r = fs.Read(buf, fill, n - fill);
            if (r == 0) break;
            fill += r;
        }
        if (fs.Position < fs.Length)
        {
            int nl = Array.LastIndexOf(buf, (byte)'\n', fill - 1);
            if (nl > 0) fill = nl + 1;
        }
        return buf.AsSpan(0, fill).ToArray();
    }

    /// <summary>Encodes the corpus in newline-aligned chunks, appending LE uint16 ids; returns token count.</summary>
    private static long StreamEncode(BpeTokenizer tok, string corpusPath, string tmpBin, Stopwatch sw)
    {
        var buf = new byte[EncodeChunkSize];
        var outBuf = new byte[EncodeChunkSize * 2]; // worst case: one token per byte
        long tokens = 0, done = 0;
        using var input = new FileStream(corpusPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        using var output = new FileStream(tmpBin, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, FileOptions.SequentialScan);
        long total = input.Length;
        while (true)
        {
            int fill = 0;
            while (fill < buf.Length)
            {
                int r = input.Read(buf, fill, buf.Length - fill);
                if (r == 0) break;
                fill += r;
            }
            if (fill == 0) break;

            // more data follows: back off to the last newline so docs stay whole
            int n = fill;
            if (input.Position < total)
            {
                int nl = Array.LastIndexOf(buf, (byte)'\n', fill - 1);
                if (nl > 0) n = nl + 1;
            }

            int[] ids = tok.Encode(buf.AsSpan(0, n));
            for (int i = 0; i < ids.Length; i++)
            {
                outBuf[2 * i] = (byte)ids[i];
                outBuf[2 * i + 1] = (byte)(ids[i] >> 8);
            }
            output.Write(outBuf, 0, ids.Length * 2);
            tokens += ids.Length;
            done += n;
            if (n < fill) input.Seek(n - fill, SeekOrigin.Current);
            Console.WriteLine($"encode: {done / 1e6:F0}/{total / 1e6:F0} MB " +
                              $"({tokens:N0} tokens, {done / 1e6 / Math.Max(sw.Elapsed.TotalSeconds, 1e-9):F1} MB/s)");
        }
        return tokens;
    }

    /// <summary>Splits a uint16 bin at a byte offset into train/val files.</summary>
    private static void SplitBin(string src, long cutBytes, string trainPath, string valPath)
    {
        using var input = new FileStream(src, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        var buf = new byte[8 << 20];
        void CopyTo(string path, long bytes)
        {
            using var dst = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, FileOptions.SequentialScan);
            long left = bytes;
            while (left > 0)
            {
                int r = input.Read(buf, 0, (int)Math.Min(buf.Length, left));
                if (r == 0) break;
                dst.Write(buf, 0, r);
                left -= r;
            }
        }
        CopyTo(trainPath, cutBytes);
        CopyTo(valPath, input.Length - cutBytes);
    }
}
