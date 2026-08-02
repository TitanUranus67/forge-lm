namespace LLM.Core.Tensor;

/// <summary>Capability-driven backend selection policy, separated for deterministic testing.</summary>
public static class BackendSelection
{
    public readonly record struct Choice<T>(string Name, T Value);

    /// <summary>
    /// Creates an explicit backend, or probes the auto candidates in platform order.
    /// Explicit choices are fail-loud. Auto treats optional accelerator failures as
    /// unavailable and invokes <paramref name="onUnavailable"/> before continuing.
    /// </summary>
    public static Choice<T> Create<T>(string requested, bool isWindows,
        Func<string, T> factory, Action<string, Exception>? onUnavailable = null)
    {
        string name = requested.ToLowerInvariant();
        if (name is not ("auto" or "cpu" or "gpu" or "cuda"))
            throw new ArgumentException($"unknown backend '{requested}' (expected auto, cpu, gpu, or cuda)");

        if (name != "auto") return new Choice<T>(name, factory(name));

        string[] candidates = isWindows ? ["cuda", "gpu", "cpu"] : ["cuda", "cpu"];
        foreach (string candidate in candidates)
        {
            try { return new Choice<T>(candidate, factory(candidate)); }
            catch (Exception ex) when (candidate != "cpu")
            {
                onUnavailable?.Invoke(candidate, ex);
            }
        }

        throw new InvalidOperationException("CPU backend creation unexpectedly failed.");
    }
}
