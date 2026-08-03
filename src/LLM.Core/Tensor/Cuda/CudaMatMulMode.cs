namespace LLM.Core.Tensor.Cuda;

/// <summary>Selects the CUDA matrix-multiplication implementation and precision.</summary>
public enum CudaMatMulMode
{
    /// <summary>The project-owned FP32 ILGPU tiled kernel.</summary>
    Custom,

    /// <summary>cuBLAS SGEMM using strict FP32 math.</summary>
    CuBlasFp32,

    /// <summary>cuBLAS SGEMM with NVIDIA TF32 tensor-operation math enabled.</summary>
    CuBlasTf32,
}
