using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using TensorValue = LLM.Core.Tensor.Tensor;

namespace LLM.Core.Tensor.Cuda;

/// <summary>
/// CUDA implementation of <see cref="ITensorBackend"/> built on ILGPU. Tensor data is
/// uploaded lazily and remains resident in CUDA memory across forward, backward, and
/// optimizer operations. The host/device synchronization contract is identical to the
/// D3D12 backend, while the runtime path is supported on Linux and Windows NVIDIA hosts.
/// </summary>
public sealed class CudaBackend : ITensorBackend, IDisposable
{
    // cublasMath_t value from CUDA: CUBLAS_TF32_TENSOR_OP_MATH. ILGPU 1.5.3
    // supports cuBLAS v12 but its public enum predates this named member.
    private const CuBlasMathMode Tf32TensorOpMath = (CuBlasMathMode)3;

    private const int ArenaFloats = 16 * 1024 * 1024;
    private const int DedicatedFloats = ArenaFloats / 2;
    private const int MinSplitFloats = 64 * 1024;

    // ILGPU 1.5 installs a process-wide CUDA native-library resolver. Recreating a
    // CUDA Context on Linux attempts to install that resolver twice and fails, so all
    // backend instances intentionally share one process-lifetime compiler context.
    private static readonly Lazy<Context> SharedContext = new(
        () => Context.Create(builder => builder.Cuda().EnableAlgorithms().Optimize(OptimizationLevel.Release)),
        LazyThreadSafetyMode.ExecutionAndPublication);

    private readonly CudaAccelerator _accelerator;
    private readonly CudaMatMulMode _matMulMode;
    private CuBlas? _cuBlas;
    private readonly object _gate = new();
    private readonly Dictionary<int, Stack<Chunk>> _freeChunks = new();
    private readonly List<(WeakReference<TensorValue> Owner, Entry Entry)> _entries = new();
    private readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _deviceBuffers = new();
    private MemoryBuffer1D<float, Stride1D.Dense>? _arena;
    private int _arenaUsed;
    private bool _disposed;
    private long _allocHits, _allocCarves;

    private MemoryBuffer1D<int, Stride1D.Dense>? _indices;
    private int _indicesLength;
    private MemoryBuffer1D<float, Stride1D.Dense>? _nll;
    private int _nllLength;
    private MemoryBuffer1D<float, Stride1D.Dense>? _scalar;
    private readonly float[] _scalarHost = new float[1];

    private readonly Action<Index1D, CudaKernels.BinaryArgs> _addInPlace, _copy, _geluForward;
    private readonly Action<Index1D, CudaKernels.CopyBlockArgs> _copyBlock;
    private readonly Action<Index1D, CudaKernels.ScaleArgs> _scale, _fill;
    private readonly Action<Index1D, CudaKernels.AddBiasArgs> _addBias;
    private readonly Action<Index1D, CudaKernels.TransposeArgs> _transpose;
    private readonly Action<Index1D, CudaKernels.GeluBackwardArgs> _geluBackward;
    private readonly Action<Index1D, CudaKernels.CausalMaskArgs> _causalMask;
    private readonly Action<Index1D, CudaKernels.HeadPackArgs> _packHeads, _unpackHeads;
    private readonly Action<Index1D, CudaKernels.SumRowsArgs> _sumRows;
    private readonly Action<Index1D, CudaKernels.LayerNormForwardArgs> _layerNormForward;
    private readonly Action<Index1D, CudaKernels.LayerNormBackwardDxArgs> _layerNormBackwardDx;
    private readonly Action<Index1D, CudaKernels.LayerNormBackwardParamsArgs> _layerNormBackwardParams;
    private readonly Action<KernelConfig, CudaKernels.UnaryArgs> _softmaxForward;
    private readonly Action<KernelConfig, CudaKernels.SoftmaxBackwardArgs> _softmaxBackward;
    private readonly Action<Index1D, CudaKernels.EmbeddingArgs> _embeddingForward, _embeddingBackward;
    private readonly Action<Index1D, CudaKernels.CrossEntropyForwardArgs> _crossEntropyForward;
    private readonly Action<Index1D, CudaKernels.CrossEntropyBackwardArgs> _crossEntropyBackward;
    private readonly Action<KernelConfig, CudaKernels.MatMulArgs> _matMul;
    private readonly Action<KernelConfig, CudaKernels.SumSquaresArgs> _sumSquares;
    private readonly Action<Index1D, CudaKernels.AdamWArgs> _adamW;

    private readonly struct Chunk
    {
        public readonly MemoryBuffer1D<float, Stride1D.Dense>? Buffer;
        public readonly int Offset, Length;
        public Chunk(MemoryBuffer1D<float, Stride1D.Dense> buffer, int offset, int length)
        { Buffer = buffer; Offset = offset; Length = length; }
        public ArrayView<float> View => Buffer!.View.SubView(Offset, Length);
    }

    private sealed class Entry
    {
        public Chunk Storage;
        public bool DeviceCurrent;
        public bool HostStale;
        public Entry(Chunk storage) => Storage = storage;
    }

    /// <summary>Creates a backend on a CUDA device and compiles the kernel set.</summary>
    public CudaBackend(int deviceIndex = 0, CudaMatMulMode matMulMode = CudaMatMulMode.Custom)
    {
        CudaAccelerator? accelerator = null;
        CuBlas? cuBlas = null;
        string stage = "CUDA context";
        try
        {
            Context context = SharedContext.Value;
            accelerator = context.CreateCudaAccelerator(deviceIndex);
            _accelerator = accelerator;
            _matMulMode = matMulMode;

            stage = nameof(CudaKernels.AddInPlace);
            _addInPlace = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.BinaryArgs>(CudaKernels.AddInPlace);
            stage = nameof(CudaKernels.Copy);
            _copy = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.BinaryArgs>(CudaKernels.Copy);
            stage = nameof(CudaKernels.GeluForward);
            _geluForward = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.BinaryArgs>(CudaKernels.GeluForward);
            stage = nameof(CudaKernels.CopyBlock);
            _copyBlock = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.CopyBlockArgs>(CudaKernels.CopyBlock);
            stage = nameof(CudaKernels.Scale);
            _scale = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.ScaleArgs>(CudaKernels.Scale);
            stage = nameof(CudaKernels.Fill);
            _fill = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.ScaleArgs>(CudaKernels.Fill);
            stage = nameof(CudaKernels.AddBias);
            _addBias = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.AddBiasArgs>(CudaKernels.AddBias);
            stage = nameof(CudaKernels.Transpose);
            _transpose = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.TransposeArgs>(CudaKernels.Transpose);
            stage = nameof(CudaKernels.GeluBackward);
            _geluBackward = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.GeluBackwardArgs>(CudaKernels.GeluBackward);
            stage = nameof(CudaKernels.CausalMask);
            _causalMask = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.CausalMaskArgs>(CudaKernels.CausalMask);
            stage = nameof(CudaKernels.PackHeads);
            _packHeads = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.HeadPackArgs>(CudaKernels.PackHeads);
            stage = nameof(CudaKernels.UnpackHeads);
            _unpackHeads = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.HeadPackArgs>(CudaKernels.UnpackHeads);
            stage = nameof(CudaKernels.SumRows);
            _sumRows = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.SumRowsArgs>(CudaKernels.SumRows);
            stage = nameof(CudaKernels.LayerNormForward);
            _layerNormForward = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.LayerNormForwardArgs>(CudaKernels.LayerNormForward);
            stage = nameof(CudaKernels.LayerNormBackwardDx);
            _layerNormBackwardDx = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.LayerNormBackwardDxArgs>(CudaKernels.LayerNormBackwardDx);
            stage = nameof(CudaKernels.LayerNormBackwardParams);
            _layerNormBackwardParams = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.LayerNormBackwardParamsArgs>(CudaKernels.LayerNormBackwardParams);
            stage = nameof(CudaKernels.SoftmaxForward);
            _softmaxForward = accelerator.LoadStreamKernel<CudaKernels.UnaryArgs>(CudaKernels.SoftmaxForward);
            stage = nameof(CudaKernels.SoftmaxBackward);
            _softmaxBackward = accelerator.LoadStreamKernel<CudaKernels.SoftmaxBackwardArgs>(CudaKernels.SoftmaxBackward);
            stage = nameof(CudaKernels.EmbeddingForward);
            _embeddingForward = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.EmbeddingArgs>(CudaKernels.EmbeddingForward);
            stage = nameof(CudaKernels.EmbeddingBackward);
            _embeddingBackward = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.EmbeddingArgs>(CudaKernels.EmbeddingBackward);
            stage = nameof(CudaKernels.CrossEntropyForward);
            _crossEntropyForward = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.CrossEntropyForwardArgs>(CudaKernels.CrossEntropyForward);
            stage = nameof(CudaKernels.CrossEntropyBackward);
            _crossEntropyBackward = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.CrossEntropyBackwardArgs>(CudaKernels.CrossEntropyBackward);
            stage = nameof(CudaKernels.MatMul);
            _matMul = accelerator.LoadStreamKernel<CudaKernels.MatMulArgs>(CudaKernels.MatMul);
            stage = nameof(CudaKernels.SumSquares);
            _sumSquares = accelerator.LoadStreamKernel<CudaKernels.SumSquaresArgs>(CudaKernels.SumSquares);
            stage = nameof(CudaKernels.AdamW);
            _adamW = accelerator.LoadAutoGroupedStreamKernel<Index1D, CudaKernels.AdamWArgs>(CudaKernels.AdamW);

            if (matMulMode is not CudaMatMulMode.Custom)
            {
                if (matMulMode is CudaMatMulMode.CuBlasTf32 && accelerator.Architecture.Major < 8)
                    throw new NotSupportedException(
                        $"TF32 cuBLAS matmuls require CUDA compute capability 8.0 or newer; " +
                        $"{accelerator.Name} is {accelerator.Architecture}.");

                stage = "cuBLAS initialization";
                cuBlas = new CuBlas(accelerator)
                {
                    Stream = (CudaStream)accelerator.DefaultStream,
                    MathMode = matMulMode is CudaMatMulMode.CuBlasTf32
                        ? Tf32TensorOpMath
                        : CuBlasMathMode.DefaultMath,
                };
                _cuBlas = cuBlas;
            }
        }
        catch (Exception ex)
        {
            cuBlas?.Dispose();
            accelerator?.Dispose();
            throw new InvalidOperationException($"CUDA backend initialization failed during {stage}.", ex);
        }
    }

    /// <summary>True when CUDA device zero can be opened by ILGPU.</summary>
    public static bool IsAvailable
    {
        get
        {
            try
            {
                using CudaAccelerator accelerator = SharedContext.Value.CreateCudaAccelerator(0);
                return true;
            }
            catch { return false; }
        }
    }

    public string DeviceName => _accelerator.Name;
    public long DeviceMemoryBytes => _accelerator.MemorySize;
    public bool SupportsTf32 => _accelerator.Architecture.Major >= 8;
    public CudaMatMulMode MatMulMode => _matMulMode;
    public string MatMulDescription => _matMulMode switch
    {
        CudaMatMulMode.Custom => "custom FP32",
        CudaMatMulMode.CuBlasFp32 => "cuBLAS FP32",
        CudaMatMulMode.CuBlasTf32 => "cuBLAS TF32",
        _ => throw new InvalidOperationException($"Unknown CUDA matmul mode {_matMulMode}."),
    };

    public long CommittedBytes
    {
        get { lock (_gate) return _deviceBuffers.Sum(b => b.Length * 4L); }
    }

    public (long Hits, long Carves) AllocStats
    {
        get { lock (_gate) return (_allocHits, _allocCarves); }
    }

    internal static int BucketOf(int length)
    {
        if (length <= 1024) return (length + 15) & ~15;
        long bucket = 1024;
        while (bucket < length) bucket = (bucket + (bucket >> 2) + 15) & ~15L;
        if (bucket > int.MaxValue) return (length + 15) & ~15;
        return (int)bucket;
    }

    private Chunk Rent(int length)
    {
        int bucket = BucketOf(length);
        if (TryFree(bucket, out Chunk chunk)) { _allocHits++; return chunk; }
        ReclaimDeadEntries();
        if (TryFree(bucket, out chunk)) { _allocHits++; return chunk; }
        GC.Collect(0, GCCollectionMode.Forced, blocking: true);
        ReclaimDeadEntries();
        if (TryFree(bucket, out chunk)) { _allocHits++; return chunk; }
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        ReclaimDeadEntries();
        if (TryFree(bucket, out chunk)) { _allocHits++; return chunk; }
        _allocCarves++;
        return Carve(bucket);
    }

    private bool TryFree(int bucket, out Chunk chunk)
    {
        if (_freeChunks.TryGetValue(bucket, out Stack<Chunk>? exact) && exact.Count > 0)
        { chunk = exact.Pop(); return true; }
        int best = -1;
        foreach ((int size, Stack<Chunk> stack) in _freeChunks)
            if (stack.Count > 0 && size >= bucket && size <= bucket * 4L && (best < 0 || size < best)) best = size;
        if (best < 0) { chunk = default; return false; }
        chunk = _freeChunks[best].Pop();
        return true;
    }

    private void PushFree(Chunk chunk)
    {
        if (!_freeChunks.TryGetValue(chunk.Length, out Stack<Chunk>? stack))
            _freeChunks[chunk.Length] = stack = new Stack<Chunk>();
        stack.Push(chunk);
    }

    private Chunk Carve(int bucket)
    {
        if (bucket >= DedicatedFloats) return new Chunk(AllocateFloatBuffer(bucket), 0, bucket);
        if (_arena is null || _arenaUsed + bucket > _arena.Length)
        {
            if (_arena is not null)
            {
                int tail = ((int)_arena.Length - _arenaUsed) & ~15;
                while (tail > 0 && BucketOf(tail) != tail) tail -= 16;
                if (tail >= MinSplitFloats) PushFree(new Chunk(_arena, _arenaUsed, tail));
            }
            _arena = AllocateFloatBuffer(Math.Max(ArenaFloats, bucket));
            _arenaUsed = 0;
        }
        var chunk = new Chunk(_arena, _arenaUsed, bucket);
        _arenaUsed += bucket;
        return chunk;
    }

    private MemoryBuffer1D<float, Stride1D.Dense> AllocateFloatBuffer(int length)
    {
        MemoryBuffer1D<float, Stride1D.Dense> buffer = _accelerator.Allocate1D<float>(length);
        _deviceBuffers.Add(buffer);
        return buffer;
    }

    private void ReclaimDeadEntries()
    {
        for (int i = _entries.Count - 1; i >= 0; i--)
        {
            if (_entries[i].Owner.TryGetTarget(out _)) continue;
            Entry entry = _entries[i].Entry;
            if (entry.Storage.Buffer is not null)
            { PushFree(entry.Storage); entry.Storage = default; }
            _entries.RemoveAt(i);
        }
    }

    private Entry GetEntry(TensorValue tensor)
    {
        if (tensor.DeviceResource is Entry existing && existing.Storage.Buffer is not null) return existing;
        var entry = new Entry(Rent(tensor.Length));
        tensor.DeviceResource = entry;
        _entries.Add((new WeakReference<TensorValue>(tensor), entry));
        return entry;
    }

    private static ArrayView<float> View(Chunk chunk, int length) => chunk.View.SubView(0, length);

    private Chunk Read(TensorValue tensor)
    {
        Entry entry = GetEntry(tensor);
        if (!entry.DeviceCurrent)
        {
            View(entry.Storage, tensor.Length).CopyFromCPU(tensor.Data);
            entry.DeviceCurrent = true;
            entry.HostStale = false;
        }
        return entry.Storage;
    }

    private Chunk Write(TensorValue tensor, bool readModifyWrite = false)
    {
        Entry entry = GetEntry(tensor);
        if (readModifyWrite && !entry.DeviceCurrent)
            View(entry.Storage, tensor.Length).CopyFromCPU(tensor.Data);
        entry.DeviceCurrent = true;
        entry.HostStale = true;
        return entry.Storage;
    }

    private ArrayView<int> CopyIndices(int[] indices)
    {
        if (_indices is null || _indicesLength < indices.Length)
        {
            _accelerator.Synchronize();
            _indices?.Dispose();
            _indices = _accelerator.Allocate1D<int>(indices.Length);
            _indicesLength = indices.Length;
        }
        ArrayView<int> view = _indices.View.SubView(0, indices.Length);
        view.CopyFromCPU(indices);
        return view;
    }

    private ArrayView<float> NllView(int length)
    {
        if (_nll is null || _nllLength < length)
        {
            _accelerator.Synchronize();
            _nll?.Dispose();
            _nll = _accelerator.Allocate1D<float>(length);
            _nllLength = length;
        }
        return _nll.View.SubView(0, length);
    }

    public void InvalidateDeviceCache(TensorValue t)
    {
        lock (_gate)
            if (t.DeviceResource is Entry e) { e.DeviceCurrent = false; e.HostStale = false; }
    }

    public void EnsureHostCurrent(TensorValue t)
    {
        lock (_gate)
        {
            if (t.DeviceResource is not Entry e || !e.HostStale || e.Storage.Buffer is null) return;
            View(e.Storage, t.Length).CopyToCPU(t.Data);
            e.HostStale = false;
        }
    }

    public void Zero(TensorValue t)
    {
        lock (_gate)
        {
            t.Zero();
            Chunk x = Write(t);
            _fill(t.Length, new CudaKernels.ScaleArgs(View(x, t.Length), 0f, t.Length));
            if (t.DeviceResource is Entry e) e.HostStale = false;
        }
    }

    public double SumSquares(TensorValue t)
    {
        lock (_gate)
        {
            Chunk x = Read(t);
            _scalar ??= _accelerator.Allocate1D<float>(1);
            _sumSquares((1, 256), new CudaKernels.SumSquaresArgs(View(x, t.Length), _scalar.View, t.Length));
            _scalar.View.CopyToCPU(_scalarHost);
            return _scalarHost[0];
        }
    }

    public void AdamWStep(TensorValue w, TensorValue g, TensorValue m, TensorValue v,
        float lr, float beta1, float beta2, float eps, float weightDecay, int step)
    {
        lock (_gate)
        {
            Chunk cw = Write(w, true), cg = Read(g), cm = Write(m, true), cv = Write(v, true);
            float bc1 = 1f - MathF.Pow(beta1, step), bc2 = 1f - MathF.Pow(beta2, step);
            _adamW(w.Length, new CudaKernels.AdamWArgs(
                View(cw, w.Length), View(cg, g.Length), View(cm, m.Length), View(cv, v.Length),
                lr, lr * weightDecay, beta1, 1f - beta1, beta2, 1f - beta2,
                bc1, bc2, eps, weightDecay != 0f && w.Rank > 1, w.Length));
        }
    }

    private void MatMulCore(TensorValue a, TensorValue b, TensorValue y,
        int slots, int m, int k, int n, int mode, bool accumulate)
    {
        Chunk ca = Read(a), cb = Read(b), cy = Write(y, accumulate);
        if (_cuBlas is not null)
        {
            CuBlasMatMul(ca, cb, cy, slots, m, k, n, mode, accumulate);
            return;
        }

        var grid = new Index3D((n + 15) / 16, (m + 15) / 16, slots);
        var group = new Index3D(16, 16, 1);
        _matMul((grid, group), new CudaKernels.MatMulArgs(
            View(ca, a.Length), View(cb, b.Length), View(cy, y.Length), m, k, n, slots, mode, accumulate));
    }

    /// <summary>
    /// Maps the project's row-major matrices to cuBLAS's column-major SGEMM by
    /// computing C^T = op(B)^T * op(A)^T. Batched attention matrices are contiguous,
    /// so each slot is queued on the same stream without any host staging.
    /// </summary>
    private void CuBlasMatMul(Chunk a, Chunk b, Chunk y,
        int slots, int m, int k, int n, int mode, bool accumulate)
    {
        CuBlasOperation opB;
        CuBlasOperation opA;
        int ldb;
        int lda;
        switch (mode)
        {
            case 0: // A[M,K] * B[K,N]
                opB = CuBlasOperation.NonTranspose; ldb = n;
                opA = CuBlasOperation.NonTranspose; lda = k;
                break;
            case 1: // A[M,K] * B[N,K]^T
                opB = CuBlasOperation.Transpose; ldb = k;
                opA = CuBlasOperation.NonTranspose; lda = k;
                break;
            case 2: // A[K,M]^T * B[K,N]
                opB = CuBlasOperation.NonTranspose; ldb = n;
                opA = CuBlasOperation.Transpose; lda = m;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unknown matmul transpose mode.");
        }

        ArrayView<float> aView = a.View;
        ArrayView<float> bView = b.View;
        ArrayView<float> yView = y.View;
        int aStride = m * k;
        int bStride = k * n;
        int yStride = m * n;
        float beta = accumulate ? 1f : 0f;

        for (int slot = 0; slot < slots; slot++)
        {
            // Arguments are deliberately reversed: row-major C is column-major C^T.
            _cuBlas!.Gemm(opB, opA, n, m, k, 1f,
                bView.SubView(slot * bStride, bStride), ldb,
                aView.SubView(slot * aStride, aStride), lda,
                beta, yView.SubView(slot * yStride, yStride), n);
        }
    }

    public void MatMulNN(TensorValue a, TensorValue b, TensorValue y, int M, int K, int N, bool accumulate = false)
    { lock (_gate) MatMulCore(a, b, y, 1, M, K, N, 0, accumulate); }
    public void MatMulNT(TensorValue a, TensorValue b, TensorValue y, int M, int K, int N, bool accumulate = false)
    { lock (_gate) MatMulCore(a, b, y, 1, M, K, N, 1, accumulate); }
    public void MatMulTN(TensorValue a, TensorValue b, TensorValue y, int M, int K, int N, bool accumulate = false)
    { lock (_gate) MatMulCore(a, b, y, 1, M, K, N, 2, accumulate); }
    public void BatchedMatMulNN(TensorValue a, TensorValue b, TensorValue y, int slots, int M, int K, int N, bool accumulate = false)
    { lock (_gate) MatMulCore(a, b, y, slots, M, K, N, 0, accumulate); }
    public void BatchedMatMulNT(TensorValue a, TensorValue b, TensorValue y, int slots, int M, int K, int N, bool accumulate = false)
    { lock (_gate) MatMulCore(a, b, y, slots, M, K, N, 1, accumulate); }
    public void BatchedMatMulTN(TensorValue a, TensorValue b, TensorValue y, int slots, int M, int K, int N, bool accumulate = false)
    { lock (_gate) MatMulCore(a, b, y, slots, M, K, N, 2, accumulate); }

    public void PackHeads(TensorValue src, TensorValue dst, int batch, int T, int nHeads, int headDim, int colBase)
    {
        lock (_gate)
        {
            Chunk s = Read(src), d = Write(dst);
            _packHeads(dst.Length, new CudaKernels.HeadPackArgs(View(s, src.Length), View(d, dst.Length),
                src.Cols, colBase, T, headDim, nHeads, dst.Length));
        }
    }

    public void UnpackHeads(TensorValue src, TensorValue dst, int batch, int T, int nHeads, int headDim, int colBase)
    {
        lock (_gate)
        {
            bool full = colBase == 0 && nHeads * headDim == dst.Cols && batch * T == dst.Rows;
            Chunk s = Read(src), d = Write(dst, !full);
            _unpackHeads(src.Length, new CudaKernels.HeadPackArgs(View(s, src.Length), View(d, dst.Length),
                dst.Cols, colBase, T, headDim, nHeads, src.Length));
        }
    }

    public void AddBias(TensorValue y, TensorValue bias, int rows, int cols)
    {
        lock (_gate)
        {
            Chunk cy = Write(y, true), cb = Read(bias);
            _addBias(rows * cols, new CudaKernels.AddBiasArgs(View(cy, y.Length), View(cb, bias.Length), cols, rows * cols));
        }
    }

    public void SumRows(TensorValue dY, TensorValue dBias, int rows, int cols)
    {
        lock (_gate)
        {
            Chunk y = Read(dY), b = Write(dBias, true);
            _sumRows(cols, new CudaKernels.SumRowsArgs(View(y, dY.Length), View(b, dBias.Length), rows, cols));
        }
    }

    public void AddInPlace(TensorValue dst, TensorValue src)
    {
        lock (_gate)
        {
            Chunk d = Write(dst, true), s = Read(src);
            _addInPlace(dst.Length, new CudaKernels.BinaryArgs(View(s, src.Length), View(d, dst.Length), dst.Length));
        }
    }

    public void Copy(TensorValue src, TensorValue dst)
    {
        lock (_gate)
        {
            Chunk s = Read(src), d = Write(dst);
            _copy(src.Length, new CudaKernels.BinaryArgs(View(s, src.Length), View(d, dst.Length), src.Length));
        }
    }

    public void CopyBlock(TensorValue src, TensorValue dst, int srcRow, int srcCol, int dstRow, int dstCol, int rows, int cols)
    {
        lock (_gate)
        {
            bool full = dstRow == 0 && dstCol == 0 && rows == dst.Rows && cols == dst.Cols;
            Chunk s = Read(src), d = Write(dst, !full);
            _copyBlock(rows * cols, new CudaKernels.CopyBlockArgs(View(s, src.Length), View(d, dst.Length),
                srcRow, srcCol, dstRow, dstCol, src.Cols, dst.Cols, cols, rows * cols));
        }
    }

    public void Scale(TensorValue x, float factor)
    {
        lock (_gate)
        {
            Chunk cx = Write(x, true);
            _scale(x.Length, new CudaKernels.ScaleArgs(View(cx, x.Length), factor, x.Length));
        }
    }

    public void Transpose(TensorValue x, TensorValue output, int rows, int cols)
    {
        lock (_gate)
        {
            Chunk cx = Read(x), co = Write(output);
            _transpose(rows * cols, new CudaKernels.TransposeArgs(View(cx, x.Length), View(co, output.Length), rows, cols));
        }
    }

    public void LayerNormForward(TensorValue x, TensorValue w, TensorValue b,
        TensorValue output, TensorValue mean, TensorValue rstd, int rows, int cols, float eps)
    {
        lock (_gate)
        {
            Chunk cx = Read(x), cw = Read(w), cb = Read(b), co = Write(output), cm = Write(mean), cr = Write(rstd);
            _layerNormForward(rows, new CudaKernels.LayerNormForwardArgs(
                View(cx, x.Length), View(cw, w.Length), View(cb, b.Length), View(co, output.Length),
                View(cm, mean.Length), View(cr, rstd.Length), rows, cols, eps));
        }
    }

    public void LayerNormBackward(TensorValue dOut, TensorValue x, TensorValue w,
        TensorValue mean, TensorValue rstd, TensorValue dX, TensorValue dW, TensorValue dB, int rows, int cols)
    {
        lock (_gate)
        {
            Chunk cdo = Read(dOut), cx = Read(x), cw = Read(w), cm = Read(mean), cr = Read(rstd);
            Chunk cdx = Write(dX), cdw = Write(dW, true), cdb = Write(dB, true);
            _layerNormBackwardDx(rows, new CudaKernels.LayerNormBackwardDxArgs(
                View(cdo, dOut.Length), View(cx, x.Length), View(cw, w.Length), View(cm, mean.Length),
                View(cr, rstd.Length), View(cdx, dX.Length), rows, cols));
            _layerNormBackwardParams(cols, new CudaKernels.LayerNormBackwardParamsArgs(
                View(cdo, dOut.Length), View(cx, x.Length), View(cm, mean.Length), View(cr, rstd.Length),
                View(cdw, dW.Length), View(cdb, dB.Length), rows, cols));
        }
    }

    public void SoftmaxForward(TensorValue x, int rows, int cols)
    {
        lock (_gate)
        {
            Chunk cx = Write(x, true);
            _softmaxForward((rows, 256), new CudaKernels.UnaryArgs(View(cx, x.Length), cols));
        }
    }

    public void SoftmaxBackward(TensorValue dOut, TensorValue softmaxOut, TensorValue dX, int rows, int cols)
    {
        lock (_gate)
        {
            Chunk cdo = Read(dOut), cs = Read(softmaxOut), cdx = Write(dX);
            _softmaxBackward((rows, 256), new CudaKernels.SoftmaxBackwardArgs(
                View(cdo, dOut.Length), View(cs, softmaxOut.Length), View(cdx, dX.Length), rows, cols));
        }
    }

    public void GeluForward(TensorValue x, TensorValue output)
    {
        lock (_gate)
        {
            Chunk cx = Read(x), co = Write(output);
            _geluForward(x.Length, new CudaKernels.BinaryArgs(View(cx, x.Length), View(co, output.Length), x.Length));
        }
    }

    public void GeluBackward(TensorValue dOut, TensorValue x, TensorValue dX)
    {
        lock (_gate)
        {
            Chunk cdo = Read(dOut), cx = Read(x), cdx = Write(dX);
            _geluBackward(x.Length, new CudaKernels.GeluBackwardArgs(
                View(cdo, dOut.Length), View(cx, x.Length), View(cdx, dX.Length), x.Length));
        }
    }

    public void EmbeddingForward(TensorValue table, int[] indices, TensorValue output, int D)
    {
        lock (_gate)
        {
            Chunk ct = Read(table), co = Write(output);
            ArrayView<int> idx = CopyIndices(indices);
            _embeddingForward(indices.Length * D, new CudaKernels.EmbeddingArgs(
                View(ct, table.Length), idx, View(co, output.Length), D, indices.Length * D));
        }
    }

    public void EmbeddingBackward(TensorValue dOut, int[] indices, TensorValue dTable, int D)
    {
        lock (_gate)
        {
            Chunk cdo = Read(dOut), cdt = Write(dTable, true);
            ArrayView<int> idx = CopyIndices(indices);
            _embeddingBackward(indices.Length * D, new CudaKernels.EmbeddingArgs(
                View(cdo, dOut.Length), idx, View(cdt, dTable.Length), D, indices.Length * D));
        }
    }

    public void CausalMask(TensorValue scores, int T)
    {
        lock (_gate)
        {
            Chunk cs = Write(scores, true);
            _causalMask(scores.Length, new CudaKernels.CausalMaskArgs(View(cs, scores.Length), T, scores.Length));
        }
    }

    public float CrossEntropyForward(TensorValue logits, int[] targets, TensorValue probs, int T, int V, int ignoreIndex)
    {
        lock (_gate)
        {
            Chunk cl = Read(logits), cp = Write(probs);
            ArrayView<int> targetView = CopyIndices(targets);
            ArrayView<float> nll = NllView(T);
            _crossEntropyForward(T, new CudaKernels.CrossEntropyForwardArgs(
                View(cl, logits.Length), targetView, View(cp, probs.Length), nll, T, V, ignoreIndex));
            float[] perRow = new float[T];
            nll.CopyToCPU(perRow);
            double total = 0; int count = 0;
            for (int t = 0; t < T; t++)
                if (targets[t] != ignoreIndex) { total += perRow[t]; count++; }
            return count == 0 ? 0f : (float)(total / count);
        }
    }

    public void CrossEntropyBackward(TensorValue probs, int[] targets, TensorValue dLogits, int T, int V, int ignoreIndex)
    {
        int count = 0;
        for (int t = 0; t < T; t++) if (targets[t] != ignoreIndex) count++;
        lock (_gate)
        {
            if (count == 0)
            {
                Chunk zero = Write(dLogits);
                _fill(T * V, new CudaKernels.ScaleArgs(View(zero, dLogits.Length), 0f, T * V));
                return;
            }
            Chunk cp = Read(probs), cdl = Write(dLogits);
            ArrayView<int> targetView = CopyIndices(targets);
            _crossEntropyBackward(T * V, new CudaKernels.CrossEntropyBackwardArgs(
                View(cp, probs.Length), targetView, View(cdl, dLogits.Length), V, ignoreIndex, 1f / count, T * V));
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            _accelerator.Synchronize();
            _cuBlas?.Dispose(); _cuBlas = null;
            _indices?.Dispose(); _nll?.Dispose(); _scalar?.Dispose();
            foreach ((WeakReference<TensorValue> owner, Entry entry) in _entries)
            {
                if (owner.TryGetTarget(out TensorValue? tensor) && ReferenceEquals(tensor.DeviceResource, entry))
                {
                    if (entry.HostStale && entry.Storage.Buffer is not null)
                        View(entry.Storage, tensor.Length).CopyToCPU(tensor.Data);
                    tensor.DeviceResource = null;
                }
                entry.Storage = default;
            }
            foreach (MemoryBuffer1D<float, Stride1D.Dense> buffer in _deviceBuffers) buffer.Dispose();
            _deviceBuffers.Clear(); _freeChunks.Clear(); _entries.Clear(); _arena = null;
            _accelerator.Dispose();
        }
    }
}
