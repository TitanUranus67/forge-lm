using System.Runtime.InteropServices;
using ILGPU.Runtime.Cuda;

namespace LLM.Core.Tensor.Cuda;

/// <summary>Narrow CUDA Driver API binding for stream-captured graph replay.</summary>
internal static class CudaGraphNative
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int StreamBeginCaptureDelegate(nint stream, int mode);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int StreamEndCaptureDelegate(nint stream, out nint graph);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GraphInstantiateDelegate(out nint graphExec, nint graph, ulong flags);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GraphLaunchDelegate(nint graphExec, nint stream);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GraphDestroyDelegate(nint graph);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GraphExecDestroyDelegate(nint graphExec);

    private static readonly nint Library = LoadLibrary();
    private static readonly StreamBeginCaptureDelegate StreamBeginCapture =
        Load<StreamBeginCaptureDelegate>("cuStreamBeginCapture");
    private static readonly StreamEndCaptureDelegate StreamEndCapture =
        Load<StreamEndCaptureDelegate>("cuStreamEndCapture");
    private static readonly GraphInstantiateDelegate GraphInstantiate =
        Load<GraphInstantiateDelegate>("cuGraphInstantiateWithFlags");
    private static readonly GraphLaunchDelegate GraphLaunch =
        Load<GraphLaunchDelegate>("cuGraphLaunch");
    private static readonly GraphDestroyDelegate GraphDestroy =
        Load<GraphDestroyDelegate>("cuGraphDestroy");
    private static readonly GraphExecDestroyDelegate GraphExecDestroy =
        Load<GraphExecDestroyDelegate>("cuGraphExecDestroy");

    internal static void BeginCapture(CudaStream stream) =>
        Check(StreamBeginCapture(stream.StreamPtr, mode: 0), "cuStreamBeginCapture");

    internal static nint EndCaptureAndInstantiate(CudaStream stream)
    {
        Check(StreamEndCapture(stream.StreamPtr, out nint graph), "cuStreamEndCapture");
        if (graph == 0) throw new InvalidOperationException("CUDA stream capture returned a null graph.");
        try
        {
            Check(GraphInstantiate(out nint graphExec, graph, flags: 0),
                "cuGraphInstantiateWithFlags");
            if (graphExec == 0) throw new InvalidOperationException("CUDA graph instantiation returned a null executable.");
            return graphExec;
        }
        finally
        {
            Check(GraphDestroy(graph), "cuGraphDestroy");
        }
    }

    internal static void AbortCapture(CudaStream stream)
    {
        int result = StreamEndCapture(stream.StreamPtr, out nint graph);
        if (graph != 0) GraphDestroy(graph);
        // An invalidated capture commonly returns CUDA_ERROR_STREAM_CAPTURE_INVALIDATED.
        // Preserve the original managed exception instead of replacing it here.
        _ = result;
    }

    internal static void Launch(nint graphExec, CudaStream stream) =>
        Check(GraphLaunch(graphExec, stream.StreamPtr), "cuGraphLaunch");

    internal static void DestroyExecutable(nint graphExec)
    {
        if (graphExec != 0) Check(GraphExecDestroy(graphExec), "cuGraphExecDestroy");
    }

    private static T Load<T>(string export) where T : Delegate
    {
        if (!NativeLibrary.TryGetExport(Library, export, out nint function))
            throw new EntryPointNotFoundException($"CUDA driver does not export {export}.");
        return Marshal.GetDelegateForFunctionPointer<T>(function);
    }

    private static nint LoadLibrary()
    {
        string[] candidates = OperatingSystem.IsWindows()
            ? ["nvcuda.dll"]
            : ["libcuda.so.1", "libcuda.so"];
        foreach (string candidate in candidates)
            if (NativeLibrary.TryLoad(candidate, out nint library)) return library;
        throw new DllNotFoundException($"Could not load the CUDA driver from: {string.Join(", ", candidates)}");
    }

    private static void Check(int result, string operation)
    {
        if (result != 0)
            throw new InvalidOperationException($"{operation} failed with CUDA driver result {result}.");
    }
}

/// <summary>A stream-captured CUDA graph executable bound to its originating stream.</summary>
internal sealed class CudaGraphExecutable : IDisposable
{
    private readonly CudaStream _stream;
    private nint _handle;

    internal CudaGraphExecutable(CudaStream stream, nint handle)
    {
        _stream = stream;
        _handle = handle;
    }

    internal void Launch()
    {
        if (_handle == 0) throw new ObjectDisposedException(nameof(CudaGraphExecutable));
        CudaGraphNative.Launch(_handle, _stream);
    }

    public void Dispose()
    {
        nint handle = Interlocked.Exchange(ref _handle, 0);
        CudaGraphNative.DestroyExecutable(handle);
    }
}

/// <summary>A captured forward/backward pass with mutable token and target staging.</summary>
internal sealed class CudaTrainingGraph : IDisposable
{
    private readonly CudaBackend _backend;
    private readonly CudaGraphExecutable _executable;
    private readonly int _tokenCount;
    private bool _disposed;

    internal CudaTrainingGraph(CudaBackend backend, CudaGraphExecutable executable, int tokenCount)
    {
        _backend = backend;
        _executable = executable;
        _tokenCount = tokenCount;
    }

    internal void Replay(int[] inputs, int[] targets)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaTrainingGraph));
        _backend.ReplayTrainingGraph(_executable, inputs, targets, _tokenCount);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _backend.ReleaseTrainingGraph(_executable);
    }
}
