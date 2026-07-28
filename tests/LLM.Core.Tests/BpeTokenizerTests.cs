using System.Text;
using LLM.Core.Tokenizer;

namespace LLM.Core.Tests;

/// <summary>
/// Round-trip, merge-learning, compression and serialization tests for
/// <see cref="BpeTokenizer"/>.
/// </summary>
public static class BpeTokenizerTests
{
    // A few KB of varied English with punctuation, newlines and unicode.
    private static readonly string Corpus = """
        The quick brown fox jumps over the lazy dog. The dog, unimpressed,
        does not move. "Why so quick?" asks the fox. "Why so lazy?" asks the dog.
        It is a tale as old as time: the fox runs, the dog sleeps, and the world
        keeps turning around them both. Meanwhile, in the city, people hurry
        through the rain-slicked streets, umbrellas blooming like black flowers.
        Somewhere a kettle whistles; somewhere a door closes; somewhere the héllo
        of a friend rings out across a crowded room. Numbers drift by: 3, 1, 4, 1,
        5, 9, 2, 6. Symbols too: !@#$%^&*()_+-=[]{}|;':",./<>? And newlines,
        plenty of newlines, because real text is never one long line.
        The fox, who is quick, is also brown. The dog, who is lazy, is also a dog.
        Repetition repetition repetition: the same words again and again teach
        the tokenizer what matters. Time after time the fox jumps, and time after
        time the dog declines to care. Such is life. Such is the corpus. The end.
        """;

    [Test]
    public static void RoundTrip_TrainingCorpus()
    {
        var tok = BpeTokenizer.Train(Encoding.UTF8.GetBytes(Corpus), 200);
        string roundTripped = tok.Decode(tok.Encode(Corpus));
        Check.True(roundTripped == Corpus, "training corpus round-trips exactly");
    }

    [Test]
    public static void RoundTrip_UnseenAndUnicodeText()
    {
        var tok = BpeTokenizer.Train(Encoding.UTF8.GetBytes(Corpus), 200);
        string[] samples =
        {
            "This sentence never appeared in the training data at all.",
            "héllo 🌍 — unicode, accents, and an emoji",
            "tab\tseparated\nnewlines\r\nand CRLF",
            "",           // empty string
            "a",          // single char
            "🦊🐕",        // emoji only
        };
        foreach (string s in samples)
        {
            string roundTripped = tok.Decode(tok.Encode(s));
            Check.True(roundTripped == s, $"round-trip of {s.Substring(0, Math.Min(30, s.Length))!}");
        }
    }

    [Test]
    public static void Train_LearnsObviousMerge()
    {
        // "abababab...": the pair ('a','b') dominates and must be learned first.
        var corpus = Encoding.ASCII.GetBytes(string.Concat(Enumerable.Repeat("ab", 100)));
        var tok = BpeTokenizer.Train(corpus, 1);
        Check.True(tok.VocabSize == 257, "VocabSize == 256 + numMerges");
        int[] ids = tok.Encode("ab");
        Check.True(ids.Length == 1 && ids[0] == 256, "one merge collapses 'ab' to id 256");

        var tok5 = BpeTokenizer.Train(corpus, 5);
        Check.True(tok5.VocabSize == 261, "VocabSize grows with numMerges");
    }

    [Test]
    public static void Encode_CompressesRepetitiveText()
    {
        var tok = BpeTokenizer.Train(Encoding.UTF8.GetBytes(Corpus), 300);
        string repetitive = string.Concat(Enumerable.Repeat("the fox jumps over the lazy dog. ", 20));
        int byteLen = Encoding.UTF8.GetByteCount(repetitive);
        int[] ids = tok.Encode(repetitive);
        Check.True(ids.Length < byteLen, $"encoded {ids.Length} tokens < {byteLen} bytes");
        Check.True(tok.Decode(ids) == repetitive, "repetitive text still round-trips");
    }

    [Test]
    public static void SaveLoad_RoundTrip()
    {
        var tok = BpeTokenizer.Train(Encoding.UTF8.GetBytes(Corpus), 150);
        string path = Path.GetTempFileName();
        try
        {
            tok.Save(path);
            var loaded = BpeTokenizer.Load(path);
            Check.True(loaded.VocabSize == tok.VocabSize, "VocabSize survives save/load");

            string[] samples = { Corpus, "unseen text with unicode héllo 🌍", "abc123!@#" };
            foreach (string s in samples)
            {
                int[] a = tok.Encode(s);
                int[] b = loaded.Encode(s);
                Check.True(a.SequenceEqual(b), $"encodings identical after load ({s.Substring(0, Math.Min(20, s.Length))!})");
                Check.True(loaded.Decode(b) == s, "loaded tokenizer round-trips");
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public static void Untrained_RoundTripsArbitraryBytes()
    {
        var tok = BpeTokenizer.Train(Array.Empty<byte>(), 0);
        Check.True(tok.VocabSize == 256, "untrained vocab is exactly the 256 bytes");

        // Any byte sequence encodes: with no merges the ids ARE the bytes.
        var rng = new Random(7);
        var bytes = new byte[1000];
        rng.NextBytes(bytes);
        int[] ids = tok.Encode(bytes);
        Check.True(ids.Length == bytes.Length, "untrained encoding is identity over bytes");
        bool identity = true;
        for (int i = 0; i < bytes.Length; i++) identity &= ids[i] == bytes[i];
        Check.True(identity, "untrained token id equals byte value");

        // And valid UTF-8 text still decodes back to the original string.
        const string s = "héllo 🌍 untrained";
        Check.True(tok.Decode(tok.Encode(s)) == s, "unicode string round-trips untrained");
    }

    [Test]
    public static void Train_Deterministic_SeededRandomCorpus()
    {
        byte[] corpus = SeededText(1234, 5000);
        const string sample = "the seed determines everything: same corpus, same merges, same encoding";

        var tok1 = BpeTokenizer.Train(corpus, 50);
        var tok2 = BpeTokenizer.Train(corpus, 50);
        int[] a = tok1.Encode(sample);
        int[] b = tok2.Encode(sample);
        Check.True(a.SequenceEqual(b), "two training runs on the same corpus encode identically");
        Check.True(tok1.VocabSize == tok2.VocabSize, "two training runs reach the same vocab size");
        Check.True(tok1.Decode(a) == sample, "sample round-trips after 50 merges");
    }

    [Test]
    public static void Train_MatchesNaiveReference()
    {
        // Corpus with count-unique merges: 20 segments "u_k v_k" repeated
        // f_k = 500-10k times over disjoint byte ranges, separated by unique
        // bytes. The best pair counts are f_k and f_k-1 per segment — all
        // distinct, and all above any later chain count — so the tie-break
        // rule cannot make the two trainers diverge (the naive trainer also
        // asserts max-count uniqueness at every step).
        byte[] corpus = CountUniqueCorpus();
        const int merges = 30;

        var naiveMerges = NaiveTrain(corpus, merges, requireUniqueMax: true);
        Check.True(naiveMerges.Count == merges, $"naive trainer learned all {merges} merges");

        var fast = BpeTokenizer.Train(corpus, merges);
        Check.True(fast.VocabSize == 256 + naiveMerges.Count, "fast trainer learned the same number of merges");

        // Held-out sample from the same generator (different seed).
        byte[] sample = SeededText(7, 3000);
        int[] expected = NaiveEncode(sample, naiveMerges);
        int[] actual = fast.Encode(sample);
        Check.True(actual.SequenceEqual(expected), "fast trainer + encoder match the naive reference on held-out text");
        Check.True(fast.Decode(actual).AsSpan().SequenceEqual(System.Text.Encoding.UTF8.GetString(sample)),
            "held-out sample round-trips through the fast tokenizer");
    }

    [Test]
    public static void Train_PerfGuard()
    {
        // 1MB of repetitive-ish text; 300 merges. The old 3-pass trainer needed
        // ~64s here; the incremental one should finish in a couple of seconds.
        // The 60s bound is only a regression tripwire, not a target.
        byte[] paragraph = SeededText(5, 8192);
        var corpus = new byte[1024 * 1024];
        for (int off = 0; off < corpus.Length; off += paragraph.Length)
            Array.Copy(paragraph, 0, corpus, off, Math.Min(paragraph.Length, corpus.Length - off));

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var tok = BpeTokenizer.Train(corpus, 300);
        sw.Stop();
        Check.True(tok.VocabSize > 256, "perf-guard tokenizer learned merges");
        Check.True(sw.Elapsed < TimeSpan.FromSeconds(60), $"300 merges on 1MB took {sw.Elapsed.TotalSeconds:F1}s (< 60s)");
    }

    // Pseudo-random but seeded English-ish text: random words over a small
    // alphabet, so pair statistics are rich but reproducible.
    private static byte[] SeededText(int seed, int approxBytes)
    {
        const string alphabet = "etaoinshrdlucmfwypvbgkqjxz";
        var rng = new Random(seed);
        var sb = new StringBuilder(approxBytes + 16);
        while (sb.Length < approxBytes)
        {
            int wordLen = 1 + rng.Next(8);
            for (int i = 0; i < wordLen; i++)
                sb.Append(alphabet[rng.Next(alphabet.Length)]);
            sb.Append(rng.Next(6) == 0 ? '\n' : ' ');
        }
        return Encoding.ASCII.GetBytes(sb.ToString());
    }

    // 20 segments over disjoint byte ranges: segment k is (u_k v_k) repeated
    // f_k = 500-10k times, followed by a unique separator byte. Segment k's
    // first two merges have counts f_k and f_k-1; deeper chain counts stay
    // below f_k/2, so the first ~30 best-pair counts are all distinct.
    private static byte[] CountUniqueCorpus()
    {
        var bytes = new List<byte>();
        for (int k = 0; k < 20; k++)
        {
            byte u = (byte)(100 + k), v = (byte)(140 + k);
            int f = 500 - 10 * k;
            for (int i = 0; i < f; i++) { bytes.Add(u); bytes.Add(v); }
            bytes.Add((byte)(200 + k)); // separator, occurs once
        }
        return bytes.ToArray();
    }

    // Reference trainer: the original 3-pass algorithm (full recount, earliest
    // first occurrence wins ties, in-place replacement), kept deliberately
    // naive. When requireUniqueMax is set, every merge's max count must be
    // unique, which makes the result tie-break-independent.
    private static List<(int Left, int Right, int NewId)> NaiveTrain(byte[] corpus, int numMerges, bool requireUniqueMax)
    {
        var ids = new List<int>(corpus.Length);
        foreach (byte b in corpus) ids.Add(b);
        var merges = new List<(int Left, int Right, int NewId)>();
        var counts = new Dictionary<long, int>();
        for (int m = 0; m < numMerges && ids.Count >= 2; m++)
        {
            counts.Clear();
            for (int i = 0; i + 1 < ids.Count; i++)
            {
                long key = Key(ids[i], ids[i + 1]);
                counts.TryGetValue(key, out int c);
                counts[key] = c + 1;
            }

            long bestKey = -1;
            int bestCount = 1;
            for (int i = 0; i + 1 < ids.Count; i++)
            {
                long key = Key(ids[i], ids[i + 1]);
                if (counts[key] > bestCount) { bestCount = counts[key]; bestKey = key; }
            }
            if (bestKey < 0) break;

            if (requireUniqueMax)
            {
                int atMax = 0;
                foreach (var kv in counts) if (kv.Value == bestCount) atMax++;
                Check.True(atMax == 1, $"naive merge {m}: max count {bestCount} is unique ({atMax} pairs share it)");
            }

            int left = (int)(bestKey >> 32), right = (int)(bestKey & 0xFFFFFFFFL);
            int newId = 256 + merges.Count;
            merges.Add((left, right, newId));
            ReplaceAll(ids, left, right, newId);
        }
        return merges;
    }

    // Reference encoder: iterative rank-greedy over an explicit merge list.
    private static int[] NaiveEncode(byte[] data, List<(int Left, int Right, int NewId)> merges)
    {
        var rank = new Dictionary<long, int>();
        for (int i = 0; i < merges.Count; i++) rank[Key(merges[i].Left, merges[i].Right)] = i;
        var ids = new List<int>(data.Length);
        foreach (byte b in data) ids.Add(b);

        while (ids.Count >= 2)
        {
            long bestKey = -1;
            int bestRank = int.MaxValue;
            for (int i = 0; i + 1 < ids.Count; i++)
            {
                long key = Key(ids[i], ids[i + 1]);
                if (rank.TryGetValue(key, out int r) && r < bestRank) { bestRank = r; bestKey = key; }
            }
            if (bestKey < 0) break;
            ReplaceAll(ids, (int)(bestKey >> 32), (int)(bestKey & 0xFFFFFFFFL), merges[bestRank].NewId);
        }
        return ids.ToArray();
    }

    private static void ReplaceAll(List<int> ids, int left, int right, int newId)
    {
        int w = 0;
        for (int i = 0; i < ids.Count; i++)
        {
            if (i + 1 < ids.Count && ids[i] == left && ids[i + 1] == right) { ids[w++] = newId; i++; }
            else ids[w++] = ids[i];
        }
        ids.RemoveRange(w, ids.Count - w);
    }

    private static long Key(int left, int right) => ((long)left << 32) | (uint)right;
}
