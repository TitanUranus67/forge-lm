namespace LLM.Core.Tensor.Gpu
{
    using ComputeSharp;
    using ComputeSharp.Descriptors;
    using Tensor = LLM.Core.Tensor.Tensor;

    /// <summary>
    /// D3D12 compute-shader implementation of <see cref="ITensorBackend"/> built on
    /// ComputeSharp. Every tensor gets a device-side allocation cached in
    /// <see cref="Tensor.DeviceResource"/>, uploaded lazily on first kernel use and
    /// reused across operations — activations and parameters stay resident on the GPU
    /// for the whole forward/backward pass.
    ///
    /// Synchronization model (see <see cref="ITensorBackend"/> for the contract):
    ///  - Kernels read from and write to device memory; <see cref="Tensor.Data"/> is
    ///    NOT updated after kernel writes (a per-entry HostStale flag records this).
    ///  - <see cref="EnsureHostCurrent"/> downloads only when a kernel wrote the tensor
    ///    since the last sync; <see cref="InvalidateDeviceCache"/> marks the device copy
    ///    stale after host-side writes so the next kernel re-uploads.
    ///
    /// All device access is serialized on a private gate: model code dispatches kernels
    /// from <see cref="System.Threading.Tasks.Parallel.For"/> (batched attention slots),
    /// and ComputeSharp devices are not thread-safe.
    /// </summary>
    public sealed class GpuBackend : ITensorBackend, IDisposable
    {
        /// <summary>64 threads/group * 65535 max groups on the 1-D dispatch X axis.</summary>
        private const int MaxFlatX = 64 * 65535;

        /// <summary>Arena size in floats (64 MB): tensors are sub-allocated chunks of arenas.</summary>
        private const int ArenaFloats = 16 * 1024 * 1024;

        /// <summary>Chunks at least this large (32 MB) get a dedicated buffer instead of arena space.</summary>
        private const int DedicatedFloats = ArenaFloats / 2;

        private readonly GraphicsDevice _device;
        private readonly object _gate = new();

        // Allocator. ComputeSharp caps the number of LIVE buffer objects at 2048 (UAV
        // descriptor heap; verified empirically: allocation #2049 throws
        // InvalidOperationException), while a GPT-1-scale training step keeps ~5000
        // activation tensors alive at once. Therefore tensors are chunks (buffer +
        // offset) sub-allocated from shared 64 MB arenas; shaders take explicit
        // offsets. Two more constraints shape the design:
        //  - ComputeSharp releases native resources from finalizers, and a
        //    finalizer-thread release racing an in-flight dispatch corrupts the
        //    device (also observed) — so all device buffers get GC.SuppressFinalize
        //    and are released only via deterministic Dispose on the calling thread.
        //  - Tensors are managed-tiny, so the GC does not recycle them fast enough
        //    on its own and weak-ref reclaim starves (each step would carve a fresh
        //    ~6 GB working set and page). Every allocation miss therefore forces a
        //    collection (gen 0 first — per-step activations die young; a full GC is
        //    only the backstop) and reclaims chunks of collected tensors eagerly.
        //
        // Fragmentation control: requests are rounded UP to size buckets (see
        // BucketOf) and free lists are keyed by bucket, so a freed chunk is
        // reusable for any request landing in the same bucket. Training repeats
        // the same activation sizes every step, so after a short warm-up Rent is
        // a free-list hit ~100% of the time, carves stop, and committed device
        // memory stays flat instead of creeping. Chunks are never split: a miss
        // may take the smallest larger free bucket within 4x whole instead.
        private readonly Dictionary<int, Stack<Chunk>> _freeChunks = new();
        private readonly List<(WeakReference<Tensor> Owner, Entry Entry)> _entries = new();
        private readonly List<ReadWriteBuffer<float>> _deviceBuffers = new();
        private ReadWriteBuffer<float>? _arena;
        private int _arenaUsed;
        private bool _disposed;

        // cumulative allocator diagnostics (free-list hit vs fresh device carve)
        private long _allocHits, _allocCarves;

        // device scratch for the embedding-backward CAS pass (bit-cast of a float tensor)
        private ReadWriteBuffer<int>? _scatterBits;
        private int _scatterBitsLength;

        /// <summary>A device allocation: <see cref="Length"/> floats at <see cref="Off"/> in <see cref="Buf"/>.</summary>
        private readonly struct Chunk
        {
            public readonly ReadWriteBuffer<float> Buf;
            public readonly int Off;
            public readonly int Length;

            public Chunk(ReadWriteBuffer<float> buf, int off, int length)
            {
                Buf = buf;
                Off = off;
                Length = length;
            }
        }

        private sealed class Entry
        {
            public Chunk Storage;
            /// <summary>Device contents match <see cref="Tensor.Data"/>.</summary>
            public bool DeviceCurrent;
            /// <summary>A kernel wrote the device copy after the last host sync.</summary>
            public bool HostStale;

            public Entry(Chunk storage) => Storage = storage;
        }

        /// <summary>Creates a backend on the default D3D12 device. Throws when none is available.</summary>
        public GpuBackend()
        {
            _device = GraphicsDevice.GetDefault();
        }

        /// <summary>True when a default D3D12 device can be created on this machine.</summary>
        public static bool IsAvailable
        {
            get
            {
                try { _ = GraphicsDevice.GetDefault(); return true; }
                catch { return false; }
            }
        }

        /// <summary>Adapter name, e.g. "NVIDIA GeForce RTX 2080".</summary>
        public string DeviceName => _device.Name;

        /// <summary>Dedicated video memory in bytes.</summary>
        public long DeviceMemoryBytes => (long)_device.DedicatedMemorySize;

        /// <summary>Total bytes of live device buffers (arenas + dedicated). Diagnostic.</summary>
        public long CommittedBytes
        {
            get
            {
                lock (_gate)
                {
                    long bytes = 0;
                    foreach (ReadWriteBuffer<float> b in _deviceBuffers) bytes += b.Length * 4L;
                    return bytes;
                }
            }
        }

        /// <summary>Cumulative free-list hits vs fresh device carves since creation. Diagnostic.</summary>
        public (long Hits, long Carves) AllocStats
        {
            get { lock (_gate) return (_allocHits, _allocCarves); }
        }

        // ---- profiling (enabled with LLM_GPU_STATS=1) --------------------------------
        private readonly bool _prof = Environment.GetEnvironmentVariable("LLM_GPU_STATS") == "1";
        private readonly Dictionary<string, (long Count, double Ms)> _profOps = new();
        private double _profUpMs, _profDownMs, _profGcMs, _profRentMs;
        private long _profUpBytes, _profDownBytes;
        private int _profGcCount, _profCarves, _profReclaims;
        private long _profHits;

        private void RecOp(string op, long t0)
        {
            if (!_prof) return;
            double ms = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            _profOps.TryGetValue(op, out (long Count, double Ms) cur);
            _profOps[op] = (cur.Count + 1, cur.Ms + ms);
        }

        /// <summary>Prints per-op dispatch/upload/download/GC totals since the last call, then resets them.</summary>
        public void DumpStats(string tag)
        {
            lock (_gate)
            {
                if (!_prof) return;
                long devBytes = 0;
                foreach (ReadWriteBuffer<float> b in _deviceBuffers) devBytes += b.Length * 4L;
                Console.Error.WriteLine($"[gpu:{tag}] up {_profUpBytes / 1e6:F0}MB/{_profUpMs:F0}ms  down {_profDownBytes / 1e6:F0}MB/{_profDownMs:F0}ms  " +
                    $"GCs {_profGcCount} ({_profGcMs:F0}ms)  rent {_profRentMs:F0}ms  hits {_profHits}  carves {_profCarves}  reclaims {_profReclaims}  " +
                    $"heap {GC.GetTotalMemory(false) / 1e6:F0}MB  dev {devBytes / 1e6:F0}MB/{_deviceBuffers.Count}buf");
                foreach (KeyValuePair<string, (long Count, double Ms)> kv in _profOps.OrderByDescending(kv => kv.Value.Ms))
                    Console.Error.WriteLine($"    {kv.Key,-24} x{kv.Value.Count,-5} {kv.Value.Ms,10:F1}ms");
                _profOps.Clear();
                _profUpMs = _profDownMs = _profGcMs = _profRentMs = 0;
                _profUpBytes = _profDownBytes = 0;
                _profGcCount = _profCarves = _profReclaims = 0;
                _profHits = 0;
            }
        }

        // ---- allocator ---------------------------------------------------------------

        /// <summary>Rents device storage for <paramref name="length"/> floats. Caller holds the gate.</summary>
        private Chunk Rent(int length)
        {
            long t0 = _prof ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            int bucket = BucketOf(length);
            if (TryFree(bucket, out Chunk c)) return Hit(c);
            ReclaimDeadEntries();
            if (TryFree(bucket, out c)) return Hit(c);
            // Tensors are managed-tiny, so natural GC does not collect dead ones in time
            // and weak-ref reclaim starves (each step would carve a fresh ~6 GB working
            // set and page). Per-step activations die young, so a gen-0 collection
            // usually uncovers them; the full GC is only the backstop. In steady state
            // this fires about once per step (the first miss drains the dead entries of
            // the previous step), never per allocation.
            var swg = _prof ? System.Diagnostics.Stopwatch.StartNew() : null;
            GC.Collect(0, GCCollectionMode.Forced, blocking: true);
            ReclaimDeadEntries();
            if (TryFree(bucket, out c))
            {
                if (swg is not null) { _profGcMs += swg.Elapsed.TotalMilliseconds; _profGcCount++; }
                return Hit(c);
            }
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
            if (swg is not null) { _profGcMs += swg.Elapsed.TotalMilliseconds; _profGcCount++; }
            ReclaimDeadEntries();
            if (TryFree(bucket, out c)) return Hit(c);
            Chunk carved = Carve(bucket);
            _allocCarves++;
            if (_prof)
            {
                _profCarves++;
                _profRentMs += (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            }
            return carved;
        }

        private Chunk Hit(Chunk c)
        {
            _allocHits++;
            if (_prof) _profHits++;
            return c;
        }

        /// <summary>
        /// Allocation granularity: requests round UP to a bucket so freed chunks are
        /// reusable across different tensor sizes. At or below 1024 floats the bucket
        /// is the plain 16-float alignment (waste &lt;= 15 floats); above, buckets grow
        /// geometrically at 1.25x (16-aligned), capping waste at ~25% while ~60
        /// buckets cover every request up to 2^31 floats. All buckets are multiples
        /// of 16, so arena offsets stay 16-aligned without extra padding.
        /// </summary>
        internal static int BucketOf(int length)
        {
            if (length <= 1024) return (length + 15) & ~15;
            // Geometric 1.25x series with 16-alignment FOLDED INTO the rungs, so every
            // bucket size is itself a series member and therefore a fixed point of this
            // function (free lists are keyed by bucket; non-fixed-point sizes would be
            // re-bucketed upward and leak past the space they claim — see Carve).
            long bucket = 1024;
            while (bucket < length)
                bucket = (bucket + (bucket >> 2) + 15) & ~15L;
            if (bucket > int.MaxValue) return (length + 15) & ~15; // huge request: exact, aligned
            return (int)bucket;
        }

        private bool TryFree(int bucket, out Chunk chunk)
        {
            // exact bucket match first
            if (_freeChunks.TryGetValue(bucket, out Stack<Chunk>? stack) && stack.Count > 0)
            {
                chunk = stack.Pop();
                return true;
            }
            // no splitting (arenas stay whole): take the smallest larger free bucket
            // within 4x whole, if one exists
            int bestLen = -1;
            foreach (KeyValuePair<int, Stack<Chunk>> kv in _freeChunks)
                if (kv.Value.Count > 0 && kv.Key >= bucket && kv.Key <= bucket * 4L && (bestLen < 0 || kv.Key < bestLen))
                    bestLen = kv.Key;
            if (bestLen < 0)
            {
                chunk = default;
                return false;
            }
            chunk = _freeChunks[bestLen].Pop();
            return true;
        }

        private void PushFree(Chunk chunk)
        {
            if (!_freeChunks.TryGetValue(chunk.Length, out Stack<Chunk>? stack))
                _freeChunks[chunk.Length] = stack = new Stack<Chunk>();
            stack.Push(chunk);
        }

        /// <summary>Old-arena tails smaller than this (256 KB) are stranded, not recycled.</summary>
        private const int MinSplitFloats = 64 * 1024;

        /// <summary>Allocates fresh device storage: dedicated buffer when large, arena carve otherwise.</summary>
        private Chunk Carve(int bucket)
        {
            if (bucket >= DedicatedFloats)
                return new Chunk(AllocDeviceBuffer(bucket), 0, bucket);
            if (_arena is null || _arenaUsed + bucket > _arena.Length)
            {
                // recycle the old arena's tail as a free chunk instead of stranding it.
                // The tail must be floored to a true bucket size (a fixed point of
                // BucketOf) — BucketOf rounds UP, which would let the chunk claim
                // space past the end of the arena.
                if (_arena is not null)
                {
                    int tail = (_arena.Length - _arenaUsed) & ~15;
                    while (tail > 0 && BucketOf(tail) != tail) tail -= 16;
                    if (tail >= MinSplitFloats)
                        PushFree(new Chunk(_arena, _arenaUsed, tail));
                }
                _arena = AllocDeviceBuffer(Math.Max(ArenaFloats, bucket));
                _arenaUsed = 0;
            }
            var chunk = new Chunk(_arena, _arenaUsed, bucket);
            _arenaUsed += bucket;
            return chunk;
        }

        private ReadWriteBuffer<float> AllocDeviceBuffer(int length)
        {
            ReadWriteBuffer<float> buffer = _device.AllocateReadWriteBuffer<float>(length, AllocationMode.Default);
            GC.SuppressFinalize(buffer); // deterministic releases only; see the allocator comment above
            _deviceBuffers.Add(buffer);
            return buffer;
        }

        /// <summary>Returns chunks of collected tensors to the free lists. Caller holds the gate.</summary>
        private void ReclaimDeadEntries()
        {
            if (_prof) _profReclaims++;
            for (int i = _entries.Count - 1; i >= 0; i--)
            {
                if (_entries[i].Owner.TryGetTarget(out _)) continue;
                Entry e = _entries[i].Entry;
                if (e.Storage.Buf is not null)
                {
                    PushFree(e.Storage);
                    e.Storage = default;
                }
                _entries.RemoveAt(i);
            }
        }

        // ---- buffer cache / sync ---------------------------------------------------

        private Entry GetEntry(Tensor t)
        {
            if (t.DeviceResource is Entry existing && existing.Storage.Buf is not null) return existing;
            var e = new Entry(Rent(t.Length));
            t.DeviceResource = e;
            _entries.Add((new WeakReference<Tensor>(t), e));
            return e;
        }

        /// <summary>Device chunk for a kernel input; uploads host data when the device copy is stale.</summary>
        private Chunk Read(Tensor t)
        {
            Entry e = GetEntry(t);
            if (!e.DeviceCurrent)
            {
                Upload(e, t);
                e.DeviceCurrent = true;
                e.HostStale = false;
            }
            return e.Storage;
        }

        private void Upload(Entry e, Tensor t)
        {
            if (!_prof) { e.Storage.Buf.CopyFrom(t.Data, 0, e.Storage.Off, t.Length); return; }
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            e.Storage.Buf.CopyFrom(t.Data, 0, e.Storage.Off, t.Length);
            _profUpBytes += t.Length * 4L;
            _profUpMs += (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        }

        /// <summary>
        /// Device chunk for a kernel output; marks the host copy stale. Pass
        /// <paramref name="readModifyWrite"/> when the kernel reads the old contents
        /// (accumulate targets, in-place and partial writes) so they are uploaded first.
        /// </summary>
        private Chunk Write(Tensor t, bool readModifyWrite = false)
        {
            Entry e = GetEntry(t);
            if (readModifyWrite && !e.DeviceCurrent)
            {
                Upload(e, t);
            }
            e.DeviceCurrent = true;
            e.HostStale = true;
            return e.Storage;
        }

        /// <inheritdoc/>
        public void InvalidateDeviceCache(Tensor t)
        {
            lock (_gate)
            {
                if (t.DeviceResource is Entry e)
                {
                    e.DeviceCurrent = false;
                    e.HostStale = false;
                }
            }
        }

        /// <inheritdoc/>
        public void EnsureHostCurrent(Tensor t)
        {
            lock (_gate)
            {
                if (t.DeviceResource is Entry e && e.HostStale && e.Storage.Buf is not null)
                {
                    if (!_prof) { e.Storage.Buf.CopyTo(t.Data, e.Storage.Off, 0, t.Length); }
                    else
                    {
                        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                        e.Storage.Buf.CopyTo(t.Data, e.Storage.Off, 0, t.Length);
                        _profDownBytes += t.Length * 4L;
                        _profDownMs += (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                    }
                    e.HostStale = false;
                }
            }
        }

        /// <inheritdoc/>
        public void Zero(Tensor t)
        {
            lock (_gate)
            {
                t.Zero(); // keep the host copy authoritative-zero as well
                Flat(t.Length, s =>
                {
                    Chunk cx = Write(t);
                    return new FillShader(cx.Buf, cx.Off, 0f, t.Length, s);
                });
                // host and device both hold zeros now: neither side is stale
                if (t.DeviceResource is Entry e) e.HostStale = false;
            }
        }

        // one-float scratch for device-side scalar reductions (SumSquares readback)
        private ReadWriteBuffer<float>? _scalar;
        private readonly float[] _scalarHost = new float[1];

        /// <inheritdoc/>
        public double SumSquares(Tensor t)
        {
            lock (_gate)
            {
                Chunk cx = Read(t);
                _scalar ??= AllocDeviceBuffer(1);
                TimedFor(256, new SumSquaresShader(cx.Buf, cx.Off, t.Length, _scalar), nameof(SumSquares));
                _scalar.CopyTo(_scalarHost);
                return _scalarHost[0];
            }
        }

        /// <inheritdoc/>
        public void AdamWStep(Tensor w, Tensor g, Tensor m, Tensor v,
            float lr, float beta1, float beta2, float eps, float weightDecay, int step)
        {
            lock (_gate)
            {
                Chunk cw = Write(w, readModifyWrite: true), cg = Read(g);
                Chunk cm = Write(m, readModifyWrite: true), cv = Write(v, readModifyWrite: true);
                float bc1 = 1f - MathF.Pow(beta1, step);
                float bc2 = 1f - MathF.Pow(beta2, step);
                int decay = weightDecay != 0f && w.Rank > 1 ? 1 : 0;
                Flat(w.Length, s => new AdamWShader(cw.Buf, cw.Off, cg.Buf, cg.Off, cm.Buf, cm.Off, cv.Buf, cv.Off,
                    lr, lr * weightDecay, beta1, 1f - beta1, beta2, 1f - beta2, bc1, bc2, eps, decay, w.Length, s));
            }
        }

        /// <summary>Releases all device buffers and scratch.</summary>
        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _scatterBits?.Dispose();
                _scalar?.Dispose();
                foreach (ReadWriteBuffer<float> buffer in _deviceBuffers)
                    buffer.Dispose();
                _deviceBuffers.Clear();
                _freeChunks.Clear();
                _arena = null;
            }
        }

        /// <summary>Bit-cast int scratch of at least <paramref name="length"/> elements (embedding backward).</summary>
        private ReadWriteBuffer<int> ScatterBits(int length)
        {
            if (_scatterBits is null || _scatterBitsLength < length)
            {
                _scatterBits?.Dispose();
                _scatterBits = _device.AllocateReadWriteBuffer<int>(length, AllocationMode.Default);
                GC.SuppressFinalize(_scatterBits);
                _scatterBitsLength = length;
            }
            return _scatterBits;
        }

        /// <summary>
        /// Dispatches a flat (single-index) shader, splitting into a 2-D dispatch when
        /// the element count exceeds the 1-D limit. <paramref name="make"/> builds the
        /// shader for a given stride (0 for the 1-D case).
        /// </summary>
        private void Flat<TShader>(int length, Func<int, TShader> make,
            [System.Runtime.CompilerServices.CallerMemberName] string op = "")
            where TShader : struct, IComputeShader, IComputeShaderDescriptor<TShader>
        {
            if (length <= MaxFlatX) TimedFor(length, make(0), op);
            else TimedFor(MaxFlatX, (length + MaxFlatX - 1) / MaxFlatX, make(MaxFlatX), op);
        }

        private void TimedFor<TShader>(int x, in TShader shader, string op)
            where TShader : struct, IComputeShader, IComputeShaderDescriptor<TShader>
        {
            if (!_prof) { _device.For(x, shader); return; }
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            _device.For(x, shader);
            RecOp(op, t0);
        }

        private void TimedFor<TShader>(int x, int y, in TShader shader, string op)
            where TShader : struct, IComputeShader, IComputeShaderDescriptor<TShader>
        {
            if (!_prof) { _device.For(x, y, shader); return; }
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            _device.For(x, y, shader);
            RecOp(op, t0);
        }

        private void TimedFor<TShader>(int x, int y, int z, in TShader shader, string op)
            where TShader : struct, IComputeShader, IComputeShaderDescriptor<TShader>
        {
            if (!_prof) { _device.For(x, y, z, shader); return; }
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            _device.For(x, y, z, shader);
            RecOp(op, t0);
        }

        /// <summary>
        /// Rounds up to a multiple of the 16x16 matmul tile. ComputeSharp rejects threads
        /// outside the dispatch bounds BEFORE the shader body runs; in an unpadded edge
        /// block those threads would skip the groupshared barriers and corrupt the tile.
        /// </summary>
        private static int Padded16(int x) => (x + 15) & ~15;

        // ---- Matmul ------------------------------------------------------------

        /// <inheritdoc/>
        public void MatMulNN(Tensor a, Tensor b, Tensor y, int M, int K, int N, bool accumulate = false)
        {
            lock (_gate)
            {
                Chunk ca = Read(a), cb = Read(b), cy = Write(y, accumulate);
                TimedFor(Padded16(N), Padded16(M), new MatMulNnShader(ca.Buf, ca.Off, cb.Buf, cb.Off, cy.Buf, cy.Off, M, K, N, accumulate ? 1 : 0), nameof(MatMulNN));
            }
        }

        /// <inheritdoc/>
        public void MatMulNT(Tensor a, Tensor b, Tensor y, int M, int K, int N, bool accumulate = false)
        {
            lock (_gate)
            {
                Chunk ca = Read(a), cb = Read(b), cy = Write(y, accumulate);
                TimedFor(Padded16(N), Padded16(M), new MatMulNtShader(ca.Buf, ca.Off, cb.Buf, cb.Off, cy.Buf, cy.Off, M, K, N, accumulate ? 1 : 0), nameof(MatMulNT));
            }
        }

        /// <inheritdoc/>
        public void MatMulTN(Tensor a, Tensor b, Tensor y, int M, int K, int N, bool accumulate = false)
        {
            lock (_gate)
            {
                Chunk ca = Read(a), cb = Read(b), cy = Write(y, accumulate);
                TimedFor(Padded16(N), Padded16(M), new MatMulTnShader(ca.Buf, ca.Off, cb.Buf, cb.Off, cy.Buf, cy.Off, M, K, N, accumulate ? 1 : 0), nameof(MatMulTN));
            }
        }

        // ---- Batched matmul (packed independent slots) --------------------------------

        /// <inheritdoc/>
        public void BatchedMatMulNN(Tensor a, Tensor b, Tensor y, int slots, int M, int K, int N, bool accumulate = false)
        {
            lock (_gate)
            {
                Chunk ca = Read(a), cb = Read(b), cy = Write(y, accumulate);
                TimedFor(Padded16(N), Padded16(M), slots,
                    new BatchedMatMulNnShader(ca.Buf, ca.Off, cb.Buf, cb.Off, cy.Buf, cy.Off, M, K, N, accumulate ? 1 : 0), nameof(BatchedMatMulNN));
            }
        }

        /// <inheritdoc/>
        public void BatchedMatMulNT(Tensor a, Tensor b, Tensor y, int slots, int M, int K, int N, bool accumulate = false)
        {
            lock (_gate)
            {
                Chunk ca = Read(a), cb = Read(b), cy = Write(y, accumulate);
                TimedFor(Padded16(N), Padded16(M), slots,
                    new BatchedMatMulNtShader(ca.Buf, ca.Off, cb.Buf, cb.Off, cy.Buf, cy.Off, M, K, N, accumulate ? 1 : 0), nameof(BatchedMatMulNT));
            }
        }

        /// <inheritdoc/>
        public void BatchedMatMulTN(Tensor a, Tensor b, Tensor y, int slots, int M, int K, int N, bool accumulate = false)
        {
            lock (_gate)
            {
                Chunk ca = Read(a), cb = Read(b), cy = Write(y, accumulate);
                TimedFor(Padded16(N), Padded16(M), slots,
                    new BatchedMatMulTnShader(ca.Buf, ca.Off, cb.Buf, cb.Off, cy.Buf, cy.Off, M, K, N, accumulate ? 1 : 0), nameof(BatchedMatMulTN));
            }
        }

        // ---- Attention head packing -----------------------------------------------------

        /// <inheritdoc/>
        public void PackHeads(Tensor src, Tensor dst, int batch, int T, int nHeads, int headDim, int colBase)
        {
            lock (_gate)
                Flat(dst.Length, s =>
                {
                    Chunk cs = Read(src), cd = Write(dst);
                    return new PackHeadsShader(cs.Buf, cs.Off, cd.Buf, cd.Off,
                        src.Cols, colBase, T, headDim, nHeads, dst.Length, s);
                });
        }

        /// <inheritdoc/>
        public void UnpackHeads(Tensor src, Tensor dst, int batch, int T, int nHeads, int headDim, int colBase)
        {
            lock (_gate)
                Flat(src.Length, s =>
                {
                    // full coverage means the old dst contents are irrelevant (no upload)
                    bool full = colBase == 0 && nHeads * headDim == dst.Cols && batch * T == dst.Rows;
                    Chunk cs = Read(src), cd = Write(dst, readModifyWrite: !full);
                    return new UnpackHeadsShader(cs.Buf, cs.Off, cd.Buf, cd.Off,
                        dst.Cols, colBase, T, headDim, nHeads, src.Length, s);
                });
        }

        // ---- Elementwise / rows -------------------------------------------------

        /// <inheritdoc/>
        public void AddBias(Tensor y, Tensor bias, int rows, int cols)
        {
            lock (_gate)
                Flat(rows * cols, s =>
                {
                    Chunk cy = Write(y, readModifyWrite: true), cb = Read(bias);
                    return new AddBiasShader(cy.Buf, cy.Off, cb.Buf, cb.Off, cols, rows * cols, s);
                });
        }

        /// <inheritdoc/>
        public void SumRows(Tensor dY, Tensor dBias, int rows, int cols)
        {
            lock (_gate)
            {
                Chunk cdY = Read(dY), cdB = Write(dBias, readModifyWrite: true);
                TimedFor(cols, new SumRowsShader(cdY.Buf, cdY.Off, cdB.Buf, cdB.Off, rows, cols), nameof(SumRows));
            }
        }

        /// <inheritdoc/>
        public void AddInPlace(Tensor dst, Tensor src)
        {
            lock (_gate)
                Flat(dst.Length, s =>
                {
                    Chunk cd = Write(dst, readModifyWrite: true), cs = Read(src);
                    return new AddInPlaceShader(cd.Buf, cd.Off, cs.Buf, cs.Off, dst.Length, s);
                });
        }

        /// <inheritdoc/>
        public void Copy(Tensor src, Tensor dst)
        {
            lock (_gate)
                Flat(src.Length, s =>
                {
                    Chunk cs = Read(src), cd = Write(dst);
                    return new CopyShader(cs.Buf, cs.Off, cd.Buf, cd.Off, src.Length, s);
                });
        }

        /// <inheritdoc/>
        public void CopyBlock(Tensor src, Tensor dst, int srcRow, int srcCol, int dstRow, int dstCol, int rows, int cols)
        {
            lock (_gate)
                Flat(rows * cols, s =>
                {
                    // full coverage means the old dst contents are irrelevant (no upload)
                    bool full = dstRow == 0 && dstCol == 0 && rows == dst.Rows && cols == dst.Cols;
                    Chunk cs = Read(src), cd = Write(dst, readModifyWrite: !full);
                    return new CopyBlockShader(cs.Buf, cs.Off, cd.Buf, cd.Off,
                        srcRow, srcCol, dstRow, dstCol, src.Cols, dst.Cols, cols, rows * cols, s);
                });
        }

        /// <inheritdoc/>
        public void Scale(Tensor x, float factor)
        {
            lock (_gate)
                Flat(x.Length, s =>
                {
                    Chunk cx = Write(x, readModifyWrite: true);
                    return new ScaleShader(cx.Buf, cx.Off, factor, x.Length, s);
                });
        }

        /// <inheritdoc/>
        public void Transpose(Tensor x, Tensor output, int rows, int cols)
        {
            lock (_gate)
                Flat(rows * cols, s =>
                {
                    Chunk cx = Read(x), co = Write(output);
                    return new TransposeShader(cx.Buf, cx.Off, co.Buf, co.Off, rows, cols, rows * cols, s);
                });
        }

        // ---- LayerNorm -----------------------------------------------------------

        /// <inheritdoc/>
        public void LayerNormForward(Tensor x, Tensor w, Tensor b,
            Tensor output, Tensor mean, Tensor rstd, int rows, int cols, float eps)
        {
            lock (_gate)
            {
                Chunk cx = Read(x), cw = Read(w), cb = Read(b);
                Chunk co = Write(output), cm = Write(mean), cr = Write(rstd);
                TimedFor(rows, new LayerNormForwardShader(cx.Buf, cx.Off, cw.Buf, cw.Off, cb.Buf, cb.Off,
                    co.Buf, co.Off, cm.Buf, cm.Off, cr.Buf, cr.Off, cols, eps), nameof(LayerNormForward));
            }
        }

        /// <inheritdoc/>
        public void LayerNormBackward(Tensor dOut, Tensor x, Tensor w,
            Tensor mean, Tensor rstd,
            Tensor dX, Tensor dW, Tensor dB, int rows, int cols)
        {
            lock (_gate)
            {
                Chunk cdO = Read(dOut), cx = Read(x), cw = Read(w), cm = Read(mean), cr = Read(rstd);
                Chunk cdX = Write(dX), cdW = Write(dW, readModifyWrite: true), cdB = Write(dB, readModifyWrite: true);
                TimedFor(rows, new LayerNormBackwardDxShader(cdO.Buf, cdO.Off, cx.Buf, cx.Off, cw.Buf, cw.Off,
                    cm.Buf, cm.Off, cr.Buf, cr.Off, cdX.Buf, cdX.Off, cols), "LayerNormBackward.Dx");
                TimedFor(cols, new LayerNormBackwardDwDbShader(cdO.Buf, cdO.Off, cx.Buf, cx.Off,
                    cm.Buf, cm.Off, cr.Buf, cr.Off, cdW.Buf, cdW.Off, cdB.Buf, cdB.Off, rows, cols), "LayerNormBackward.DwDb");
            }
        }

        // ---- Softmax --------------------------------------------------------------

        /// <inheritdoc/>
        public void SoftmaxForward(Tensor x, int rows, int cols)
        {
            lock (_gate)
            {
                Chunk cx = Write(x, readModifyWrite: true);
                // one 256-thread group per row (the shader reduces within its group)
                TimedFor(rows * 256, new SoftmaxForwardShader(cx.Buf, cx.Off, cols), nameof(SoftmaxForward));
            }
        }

        /// <inheritdoc/>
        public void SoftmaxBackward(Tensor dOut, Tensor softmaxOut, Tensor dX, int rows, int cols)
        {
            lock (_gate)
            {
                Chunk cdO = Read(dOut), cs = Read(softmaxOut), cdX = Write(dX);
                TimedFor(rows * 256, new SoftmaxBackwardShader(cdO.Buf, cdO.Off, cs.Buf, cs.Off, cdX.Buf, cdX.Off, cols), nameof(SoftmaxBackward));
            }
        }

        // ---- GELU ------------------------------------------------------------------

        /// <inheritdoc/>
        public void GeluForward(Tensor x, Tensor output)
        {
            lock (_gate)
                Flat(x.Length, s =>
                {
                    Chunk cx = Read(x), co = Write(output);
                    return new GeluForwardShader(cx.Buf, cx.Off, co.Buf, co.Off, x.Length, s);
                });
        }

        /// <inheritdoc/>
        public void GeluBackward(Tensor dOut, Tensor x, Tensor dX)
        {
            lock (_gate)
                Flat(x.Length, s =>
                {
                    Chunk cdO = Read(dOut), cx = Read(x), cdX = Write(dX);
                    return new GeluBackwardShader(cdO.Buf, cdO.Off, cx.Buf, cx.Off, cdX.Buf, cdX.Off, x.Length, s);
                });
        }

        // ---- Embedding ---------------------------------------------------------------

        /// <inheritdoc/>
        public void EmbeddingForward(Tensor table, int[] indices, Tensor output, int D)
        {
            lock (_gate)
            {
                Chunk ct = Read(table), co = Write(output);
                using ReadWriteBuffer<int> idx = _device.AllocateReadWriteBuffer(indices);
                TimedFor(D, indices.Length, new EmbeddingForwardShader(ct.Buf, ct.Off, idx, co.Buf, co.Off, D), nameof(EmbeddingForward));
            }
        }

        /// <inheritdoc/>
        public void EmbeddingBackward(Tensor dOut, int[] indices, Tensor dTable, int D)
        {
            lock (_gate)
            {
                // HLSL typed buffers have no float InterlockedAdd: bit-cast the table into
                // an int scratch, scatter-add with compare-exchange, bit-cast back.
                Chunk ct = Write(dTable, readModifyWrite: true);
                Chunk cdO = Read(dOut);
                ReadWriteBuffer<int> bits = ScatterBits(dTable.Length);
                using ReadWriteBuffer<int> idx = _device.AllocateReadWriteBuffer(indices);
                Flat(dTable.Length, s => new FloatBitsToIntShader(ct.Buf, ct.Off, bits, dTable.Length, s), "EmbeddingBackward.ToBits");
                TimedFor(D, indices.Length, new EmbeddingBackwardShader(cdO.Buf, cdO.Off, idx, bits, D), "EmbeddingBackward.Scatter");
                Flat(dTable.Length, s => new IntBitsToFloatShader(bits, ct.Buf, ct.Off, dTable.Length, s), "EmbeddingBackward.ToFloat");
            }
        }

        // ---- Attention helpers ---------------------------------------------------------

        /// <inheritdoc/>
        public void CausalMask(Tensor scores, int T)
        {
            lock (_gate)
                Flat(scores.Length, s =>
                {
                    Chunk cs = Write(scores, readModifyWrite: true);
                    return new CausalMaskShader(cs.Buf, cs.Off, T, scores.Length, s);
                });
        }

        // ---- Cross-entropy --------------------------------------------------------------

        /// <inheritdoc/>
        public float CrossEntropyForward(Tensor logits, int[] targets, Tensor probs, int T, int V, int ignoreIndex)
        {
            lock (_gate)
            {
                Chunk cl = Read(logits), cp = Write(probs);
                using ReadWriteBuffer<int> tgt = _device.AllocateReadWriteBuffer(targets);
                using ReadWriteBuffer<float> nll = _device.AllocateReadWriteBuffer<float>(T, AllocationMode.Default);
                TimedFor(T, new CrossEntropyForwardShader(cl.Buf, cl.Off, tgt, cp.Buf, cp.Off, nll, V, ignoreIndex), nameof(CrossEntropyForward));
                float[] perRow = new float[T];
                nll.CopyTo(perRow);
                double total = 0;
                int count = 0;
                for (int t = 0; t < T; t++)
                    if (targets[t] != ignoreIndex) { total += perRow[t]; count++; }
                return count > 0 ? (float)(total / count) : 0f;
            }
        }

        /// <inheritdoc/>
        public void CrossEntropyBackward(Tensor probs, int[] targets, Tensor dLogits, int T, int V, int ignoreIndex)
        {
            int count = 0;
            for (int t = 0; t < T; t++)
                if (targets[t] != ignoreIndex) count++;
            lock (_gate)
            {
                if (count == 0)
                {
                    // zero the whole gradient tensor on device
                    Flat(T * V, s =>
                    {
                        Chunk cd = Write(dLogits);
                        return new ScaleShader(cd.Buf, cd.Off, 0f, T * V, s);
                    });
                    return;
                }
                Chunk cp = Read(probs), cdL = Write(dLogits);
                using ReadWriteBuffer<int> tgt = _device.AllocateReadWriteBuffer(targets);
                float scale = 1f / count;
                Flat(T * V, s => new CrossEntropyBackwardShader(cp.Buf, cp.Off, tgt, cdL.Buf, cdL.Off, V, ignoreIndex, scale, T * V, s));
            }
        }
    }
}
