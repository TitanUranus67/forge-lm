namespace LLM.Core.Tests;

[AttributeUsage(AttributeTargets.Method)]
public sealed class TestAttribute : Attribute { }

/// <summary>Minimal assertion helpers; failures are collected, not thrown.</summary>
public static class Check
{
    public static int Failures;

    public static void True(bool cond, string msg)
    {
        if (!cond) Fail($"expected true: {msg}");
    }

    public static void Near(float actual, float expected, float tol, string msg)
    {
        if (Math.Abs(actual - expected) > tol)
            Fail($"expected {expected} +/- {tol}, got {actual}: {msg}");
    }

    public static void SpanNear(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected, float tol, string msg)
    {
        True(actual.Length == expected.Length, $"{msg} (length {actual.Length} vs {expected.Length})");
        if (actual.Length != expected.Length) return;
        float worst = 0;
        for (int i = 0; i < actual.Length; i++)
            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
        if (worst > tol) Fail($"max abs diff {worst} > {tol}: {msg}");
    }

    public static void Fail(string msg)
    {
        Failures++;
        Console.WriteLine($"    FAIL: {msg}");
    }
}
