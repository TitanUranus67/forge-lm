namespace LLM.Core.Tensor;

/// <summary>
/// Dense row-major float32 tensor. Storage is always contiguous; Shape.Length is the rank.
/// This is a plain data container — all math lives in <see cref="ITensorBackend"/>.
/// </summary>
public sealed class Tensor
{
    public float[] Data { get; }
    public int[] Shape { get; }
    public int Length => Data.Length;

    public Tensor(params int[] shape)
    {
        if (shape.Length == 0) throw new ArgumentException("Tensor needs at least 1 dimension.");
        int len = 1;
        foreach (int d in shape)
        {
            if (d <= 0) throw new ArgumentException($"Dimension must be positive, got {d}.");
            len = checked(len * d);
        }
        Shape = shape;
        Data = new float[len];
    }

    public Tensor(float[] data, params int[] shape)
    {
        int len = 1;
        foreach (int d in shape) len = checked(len * d);
        if (data.Length != len) throw new ArgumentException($"Data length {data.Length} does not match shape [{string.Join(",", shape)}].");
        Data = data;
        Shape = shape;
    }

    public int Rank => Shape.Length;
    public int Rows => Shape[^2];
    public int Cols => Shape[^1];

    public Span<float> AsSpan() => Data;
    public ReadOnlySpan<float> AsReadOnlySpan() => Data;

    public float this[params int[] idx]
    {
        get => Data[FlatIndex(idx)];
        set => Data[FlatIndex(idx)] = value;
    }

    private int FlatIndex(int[] idx)
    {
        if (idx.Length != Shape.Length) throw new ArgumentException("Index rank mismatch.");
        int flat = 0;
        foreach (var (i, d) in idx.Zip(Shape))
        {
            if ((uint)i >= (uint)d) throw new IndexOutOfRangeException();
            flat = flat * d + i;
        }
        return flat;
    }

    /// <summary>Same storage, new shape. Total length must match.</summary>
    public Tensor Reshaped(params int[] shape)
    {
        int len = 1;
        foreach (int d in shape) len = checked(len * d);
        if (len != Data.Length) throw new ArgumentException("Reshape length mismatch.");
        return new Tensor(Data, shape);
    }

    public Tensor Clone() => new Tensor((float[])Data.Clone(), (int[])Shape.Clone());

    public void Fill(float value) => Array.Fill(Data, value);
    public void Zero() => Array.Clear(Data);

    /// <summary>Fills with independent N(0,1) * std samples.</summary>
    public void FillNormal(Random rng, float std)
    {
        for (int i = 0; i < Data.Length; i += 2)
        {
            // Box-Muller
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            Data[i] = (float)(r * Math.Cos(2 * Math.PI * u2) * std);
            if (i + 1 < Data.Length)
                Data[i + 1] = (float)(r * Math.Sin(2 * Math.PI * u2) * std);
        }
    }

    public override string ToString() => $"Tensor([{string.Join(",", Shape)}])";
}
