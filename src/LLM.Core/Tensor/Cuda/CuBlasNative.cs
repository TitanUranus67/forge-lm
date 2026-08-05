using System.Runtime.InteropServices;
using ILGPU.Runtime.Cuda;

namespace LLM.Core.Tensor.Cuda;

/// <summary>
/// Narrow binding for the one cuBLAS operation ILGPU 1.5 does not expose. The
/// handle, stream, pointer mode, and math mode remain owned by ILGPU's
/// <see cref="CuBlas"/> instance; this class only submits one strided batch.
/// </summary>
internal static class CuBlasNative
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate CuBlasStatus SgemmStridedBatchedDelegate(
        nint handle,
        CuBlasOperation transA,
        CuBlasOperation transB,
        int m,
        int n,
        int k,
        float* alpha,
        nint a,
        int lda,
        long strideA,
        nint b,
        int ldb,
        long strideB,
        float* beta,
        nint c,
        int ldc,
        long strideC,
        int batchCount);

    private static readonly Lazy<SgemmStridedBatchedDelegate> SgemmStridedBatched =
        new(LoadSgemmStridedBatched, LazyThreadSafetyMode.ExecutionAndPublication);

    internal static unsafe CuBlasStatus SgemmStridedBatchedCall(
        nint handle,
        CuBlasOperation transA,
        CuBlasOperation transB,
        int m,
        int n,
        int k,
        float alpha,
        nint a,
        int lda,
        long strideA,
        nint b,
        int ldb,
        long strideB,
        float beta,
        nint c,
        int ldc,
        long strideC,
        int batchCount) =>
        SgemmStridedBatched.Value(handle, transA, transB, m, n, k,
            &alpha, a, lda, strideA, b, ldb, strideB,
            &beta, c, ldc, strideC, batchCount);

    private static SgemmStridedBatchedDelegate LoadSgemmStridedBatched()
    {
        string[] candidates = OperatingSystem.IsWindows()
            ? ["cublas64_12.dll", "cublas64_11.dll", "cublas64_10.dll"]
            : ["libcublas.so.12", "libcublas.so.11", "libcublas.so.10", "libcublas.so"];

        foreach (string candidate in candidates)
        {
            if (!NativeLibrary.TryLoad(candidate, out nint library)) continue;
            if (NativeLibrary.TryGetExport(library, "cublasSgemmStridedBatched", out nint function))
                return Marshal.GetDelegateForFunctionPointer<SgemmStridedBatchedDelegate>(function);
            NativeLibrary.Free(library);
        }

        throw new DllNotFoundException(
            $"Could not load cublasSgemmStridedBatched from: {string.Join(", ", candidates)}");
    }
}
