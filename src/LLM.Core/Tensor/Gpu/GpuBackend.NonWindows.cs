namespace LLM.Core.Tensor.Gpu;

/// <summary>
/// Compile-time placeholder for the Windows-only D3D12 backend. Keeping the public
/// surface available gives callers a clear runtime error while allowing the same
/// project to build and publish natively on Linux without loading ComputeSharp's
/// Windows-only source generator.
/// </summary>
public sealed class GpuBackend : CpuBackend, IDisposable
{
    public GpuBackend() => throw new PlatformNotSupportedException(
        "The D3D12 backend is available only on Windows; use --backend cuda or --backend cpu on this host.");

    public static bool IsAvailable => false;
    public string DeviceName => throw new PlatformNotSupportedException();
    public long DeviceMemoryBytes => 0;
    public long CommittedBytes => 0;
    public (long Hits, long Carves) AllocStats => (0, 0);
    public void Zero(Tensor t) => t.Zero();

    internal static int BucketOf(int length)
    {
        if (length <= 1024) return (length + 15) & ~15;
        long bucket = 1024;
        while (bucket < length) bucket = (bucket + (bucket >> 2) + 15) & ~15L;
        if (bucket > int.MaxValue) return (length + 15) & ~15;
        return (int)bucket;
    }

    public void Dispose() { }
}
