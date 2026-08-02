using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

namespace LLM.Core.Tokenizer;

/// <summary>
/// Byte-level BPE tokenizer (simplified GPT-2 style). Token ids 0..255 are the
/// raw bytes; learned merges produce ids 256 and up. Because every byte is a
/// base token, ANY byte sequence can be encoded and decoded losslessly.
/// </summary>
/// <remarks>
/// Encoding strategy: rank-greedy. Starting from byte ids, at each step we
/// scan the current sequence for the adjacent pair with the lowest merge rank
/// (i.e. the earliest-learned merge present) and replace all its
/// non-overlapping occurrences, repeating until no learned pair remains. This
/// matches standard BPE inference: the result is deterministic and identical
/// to what training would have produced on the same input.
/// </remarks>
public sealed class BpeTokenizer
{
    private const int CurrentVersion = 1;

    // Merge rules in learned (rank) order; merge k produces id 256 + k.
    private readonly List<(int Left, int Right, int NewId)> _merges = new();
    // (left, right) -> rank (index into _merges); packed into a long key.
    private readonly Dictionary<long, int> _rank = new();
    // Token id -> the byte sequence it stands for (used by Decode).
    private readonly List<byte[]> _vocab = new();

    private BpeTokenizer()
    {
        for (int i = 0; i < 256; i++)
            _vocab.Add(new[] { (byte)i });
    }

    /// <summary>Total vocabulary size: 256 base byte tokens plus one id per learned merge.</summary>
    public int VocabSize => 256 + _merges.Count;

    /// <summary>
    /// Stateful UTF-8 decoder for autoregressive output. BPE token boundaries are
    /// byte boundaries, not necessarily Unicode character boundaries, so decoding
    /// one token at a time with <see cref="Decode"/> can emit replacement characters.
    /// This decoder retains incomplete UTF-8 sequences until later tokens arrive.
    /// </summary>
    public sealed class Utf8StreamDecoder
    {
        private readonly BpeTokenizer _tokenizer;
        private readonly Decoder _decoder = Encoding.UTF8.GetDecoder();

        internal Utf8StreamDecoder(BpeTokenizer tokenizer) => _tokenizer = tokenizer;

        /// <summary>Decodes one token, returning only characters completed by the bytes seen so far.</summary>
        public string DecodeToken(int tokenId)
        {
            byte[] bytes = _tokenizer.TokenBytes(tokenId);
            var chars = new char[Encoding.UTF8.GetMaxCharCount(bytes.Length)];
            int written = _decoder.GetChars(bytes, 0, bytes.Length, chars, 0, flush: false);
            return new string(chars, 0, written);
        }

        /// <summary>Flushes any incomplete trailing sequence using the standard UTF-8 replacement fallback.</summary>
        public string Flush()
        {
            var chars = new char[2];
            int written = _decoder.GetChars(Array.Empty<byte>(), 0, 0, chars, 0, flush: true);
            return new string(chars, 0, written);
        }
    }

    /// <summary>Creates a stateful decoder suitable for printing generated tokens as they arrive.</summary>
    public Utf8StreamDecoder CreateUtf8StreamDecoder() => new(this);

    /// <summary>
    /// Trains a tokenizer on <paramref name="corpus"/> by learning
    /// <paramref name="numMerges"/> merge rules. Adjacent-pair counts are built
    /// once and then maintained incrementally around each merge site, and the
    /// best pair is drawn from a lazy max-heap (stale entries are skipped on
    /// pop), so each merge costs a single linear replacement scan instead of
    /// three full counting passes. Ties go to the pair with the lowest packed
    /// pair key (deterministic). Stops early if no pair occurs at least twice
    /// or the corpus has fewer than 2 tokens left.
    /// </summary>
    /// <param name="onProgress">Optional callback invoked after each learned merge
    /// with (merges learned so far, requested total) — for progress reporting.</param>
    public static BpeTokenizer Train(byte[] corpus, int numMerges, Action<int, int>? onProgress = null)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentOutOfRangeException.ThrowIfNegative(numMerges);

        var tok = new BpeTokenizer();
        int len = corpus.Length;
        var seq = new int[len];
        for (int i = 0; i < len; i++) seq[i] = corpus[i];

        // Adjacent-pair counts over the current token sequence, built once.
        var counts = new Dictionary<long, int>();
        for (int i = 0; i + 1 < len; i++)
        {
            ref int c = ref CollectionsMarshal.GetValueRefOrAddDefault(counts, PairKey(seq[i], seq[i + 1]), out _);
            c++;
        }

        // Lazy max-heap ordered by count (desc) then pair key (asc). Fresh
        // entries are pushed whenever a pair's count changes; on pop an entry
        // is only valid if its count still matches the dictionary.
        var heap = new PriorityQueue<long, (int Count, long Key)>(PairPriorityComparer.Instance);
        foreach (var kv in counts)
            heap.Enqueue(kv.Key, (kv.Value, kv.Key));

        var sites = new List<int>();       // compacted positions where the merged id was written
        var changed = new HashSet<long>(); // pair keys whose count changed during the current merge

        void ChangeCount(long key, int delta)
        {
            ref int c = ref CollectionsMarshal.GetValueRefOrAddDefault(counts, key, out _);
            c += delta;
            changed.Add(key);
        }

        for (int m = 0; m < numMerges && len >= 2; m++)
        {
            // Pop until a fresh entry: its count must still match the dictionary.
            long bestKey = -1;
            while (heap.TryDequeue(out long key, out (int Count, long Key) prio))
            {
                if (counts.TryGetValue(key, out int c) && c == prio.Count)
                {
                    if (c >= 2) bestKey = key; // a merge must occur at least twice to be useful
                    break; // fresh maximum below 2: no repeated pair remains
                }
            }
            if (bestKey < 0) break;

            int left = (int)(bestKey >> 32);
            int right = (int)(bestKey & 0xFFFFFFFFL);
            int newId = tok._vocab.Count;

            byte[] merged = new byte[tok._vocab[left].Length + tok._vocab[right].Length];
            tok._vocab[left].CopyTo(merged, 0);
            tok._vocab[right].CopyTo(merged, tok._vocab[left].Length);
            tok._vocab.Add(merged);
            tok._merges.Add((left, right, newId));
            tok._rank[bestKey] = tok._merges.Count - 1;

            // One pass: replace all non-overlapping occurrences and decrement
            // the counts of the boundary pairs each occurrence destroys. The
            // (left, right) pair itself is removed wholesale below.
            changed.Clear();
            sites.Clear();
            int w = 0;
            int prevEnd = -2; // old index of the right half of the previous merge site
            for (int i = 0; i < len; i++)
            {
                if (i + 1 < len && seq[i] == left && seq[i + 1] == right)
                {
                    // Left boundary pair, unless it is the middle pair shared
                    // with the previous site (already counted as its right pair).
                    if (i > 0 && prevEnd != i - 1) ChangeCount(PairKey(seq[i - 1], left), -1);
                    if (i + 2 < len) ChangeCount(PairKey(right, seq[i + 2]), -1);
                    sites.Add(w);
                    seq[w++] = newId;
                    prevEnd = i + 1;
                    i++;
                }
                else
                {
                    seq[w++] = seq[i];
                }
            }
            len = w;

            // The merged pair is destroyed entirely, and the fresh id can never
            // recreate it, so its count stays zero from now on.
            counts.Remove(bestKey);

            // Increment the counts of the new boundary pairs, read from the
            // compacted sequence so adjacent sites yield (newId, newId). The
            // pair between two adjacent sites is attributed to the left site.
            int prevPos = -2;
            foreach (int pos in sites)
            {
                if (pos > 0 && pos != prevPos + 1) ChangeCount(PairKey(seq[pos - 1], newId), +1);
                if (pos + 1 < len) ChangeCount(PairKey(newId, seq[pos + 1]), +1);
                prevPos = pos;
            }

            // Publish fresh counts for every touched pair. Pairs that dropped
            // below 2 need no entry: any stale higher-count entry is skipped on
            // pop, and a later increment past 2 pushes a fresh entry then.
            foreach (long key in changed)
                if (counts.TryGetValue(key, out int c) && c >= 2)
                    heap.Enqueue(key, (c, key));

            // Rebuild the heap if stale entries bloat it too far.
            if (heap.Count > 4 * counts.Count + 16)
            {
                heap.Clear();
                foreach (var kv in counts)
                    if (kv.Value >= 2)
                        heap.Enqueue(kv.Key, (kv.Value, kv.Key));
            }

            onProgress?.Invoke(m + 1, numMerges);
        }
        return tok;
    }

    /// <summary>Encodes UTF-8 <paramref name="text"/> into token ids.</summary>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Encode(Encoding.UTF8.GetBytes(text));
    }

    /// <summary>
    /// Encodes raw bytes into token ids: byte ids first, then rank-greedy
    /// merging (see class remarks). Implemented as a doubly-linked list with
    /// one FIFO position queue per merge rank: pairs are always merged in
    /// ascending rank order, leftmost occurrence first. Because a merge only
    /// creates pairs involving its fresh id, newly created pairs always have a
    /// strictly higher rank than the merge being applied, so draining the
    /// queues in rank order produces exactly the iterative rank-greedy result
    /// in O(n + applied merges) instead of O(n * merge rounds).
    /// </summary>
    public int[] Encode(ReadOnlySpan<byte> data)
    {
        int n = data.Length;
        if (n == 0) return Array.Empty<int>();
        var seq = new int[n];
        var next = new int[n];
        var prev = new int[n];
        for (int i = 0; i < n; i++)
        {
            seq[i] = data[i];
            next[i] = i + 1 < n ? i + 1 : -1;
            prev[i] = i - 1;
        }

        int mergeCount = _merges.Count;
        var queues = new Queue<int>[mergeCount];
        for (int i = 0; i + 1 < n; i++)
            if (_rank.TryGetValue(PairKey(seq[i], seq[i + 1]), out int r))
                (queues[r] ??= new Queue<int>()).Enqueue(i);

        for (int r = 0; r < mergeCount; r++)
        {
            var q = queues[r];
            if (q is null) continue;
            int newId = _merges[r].NewId;
            while (q.Count > 0)
            {
                int i = q.Dequeue();
                int j = next[i];
                if (j < 0) continue; // i was absorbed into its left neighbor or is now last
                if (!_rank.TryGetValue(PairKey(seq[i], seq[j]), out int cr) || cr != r)
                    continue; // stale: the pair at i changed since it was enqueued

                // Merge j into i.
                seq[i] = newId;
                int nj = next[j];
                next[i] = nj;
                next[j] = -1;
                prev[j] = -1;
                if (nj >= 0)
                {
                    prev[nj] = i;
                    if (_rank.TryGetValue(PairKey(newId, seq[nj]), out int r2))
                        (queues[r2] ??= new Queue<int>()).Enqueue(i);
                }
                int p = prev[i];
                if (p >= 0 && _rank.TryGetValue(PairKey(seq[p], newId), out int r3))
                    (queues[r3] ??= new Queue<int>()).Enqueue(p);
            }
        }

        // Compact the linked list into the output array.
        int count = 0;
        for (int i = 0; i >= 0; i = next[i]) count++;
        var result = new int[count];
        int w = 0;
        for (int i = 0; i >= 0; i = next[i]) result[w++] = seq[i];
        return result;
    }

    /// <summary>
    /// Decodes token ids back to a string by concatenating the bytes each id
    /// stands for and interpreting the result as UTF-8. Unknown ids throw.
    /// </summary>
    public string Decode(IReadOnlyList<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        int total = 0;
        foreach (int id in tokens)
        {
            if (id < 0 || id >= _vocab.Count)
                throw new ArgumentOutOfRangeException(nameof(tokens), id, "token id is outside the vocabulary");
            total += _vocab[id].Length;
        }
        var bytes = new byte[total];
        int pos = 0;
        foreach (int id in tokens)
        {
            _vocab[id].CopyTo(bytes, pos);
            pos += _vocab[id].Length;
        }
        return Encoding.UTF8.GetString(bytes);
    }

    private byte[] TokenBytes(int tokenId)
    {
        if ((uint)tokenId >= (uint)_vocab.Count)
            throw new ArgumentOutOfRangeException(nameof(tokenId), tokenId, "token id is outside the vocabulary");
        return _vocab[tokenId];
    }

    /// <summary>Serializes the tokenizer (merges only; the vocab is derived) to JSON.</summary>
    public void Save(string path)
    {
        var dto = new TokenizerFile
        {
            Version = CurrentVersion,
            Merges = _merges.Select(m => new[] { m.Left, m.Right, m.NewId }).ToList(),
        };
        var json = JsonSerializer.Serialize(dto, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    /// <summary>Loads a tokenizer written by <see cref="Save"/>.</summary>
    public static BpeTokenizer Load(string path)
    {
        var dto = JsonSerializer.Deserialize<TokenizerFile>(File.ReadAllText(path))
            ?? throw new InvalidDataException("tokenizer file is empty or malformed");
        if (dto.Version != CurrentVersion)
            throw new InvalidDataException($"unsupported tokenizer version {dto.Version}");
        dto.Merges ??= new List<int[]>();

        var tok = new BpeTokenizer();
        for (int i = 0; i < dto.Merges.Count; i++)
        {
            int[] m = dto.Merges[i];
            if (m.Length != 3)
                throw new InvalidDataException($"merge {i} must have exactly 3 entries");
            int left = m[0], right = m[1], newId = m[2];
            if (left < 0 || left >= tok._vocab.Count || right < 0 || right >= tok._vocab.Count)
                throw new InvalidDataException($"merge {i} references unknown token ids");
            if (newId != tok._vocab.Count)
                throw new InvalidDataException($"merge {i} has id {newId}, expected {tok._vocab.Count} (ids must be 256 + rank)");

            byte[] merged = new byte[tok._vocab[left].Length + tok._vocab[right].Length];
            tok._vocab[left].CopyTo(merged, 0);
            tok._vocab[right].CopyTo(merged, tok._vocab[left].Length);
            tok._vocab.Add(merged);
            tok._merges.Add((left, right, newId));
            tok._rank[PairKey(left, right)] = i;
        }
        return tok;
    }

    private static long PairKey(int left, int right) => ((long)left << 32) | (uint)right;

    /// <summary>Orders training-heap entries by count descending, then pair key ascending.</summary>
    private sealed class PairPriorityComparer : IComparer<(int Count, long Key)>
    {
        public static readonly PairPriorityComparer Instance = new();

        public int Compare((int Count, long Key) x, (int Count, long Key) y)
        {
            int c = y.Count.CompareTo(x.Count); // higher count first
            return c != 0 ? c : x.Key.CompareTo(y.Key); // ties: lowest pair key first
        }
    }

    private sealed class TokenizerFile
    {
        public int Version { get; set; }
        public List<int[]>? Merges { get; set; }
    }
}
