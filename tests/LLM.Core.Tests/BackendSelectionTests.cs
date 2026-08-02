namespace LLM.Core.Tests;

using LLM.Core.Tensor;

public static class BackendSelectionTests
{
    [Test]
    public static void AutoLinux_PrefersCudaThenCpu()
    {
        var attempts = new List<string>();
        BackendSelection.Choice<string> choice = BackendSelection.Create("auto", isWindows: false, candidate =>
        {
            attempts.Add(candidate);
            if (candidate == "cuda") throw new InvalidOperationException("unavailable");
            return candidate;
        });
        Check.True(choice.Name == "cpu", "Linux auto falls back to CPU");
        Check.True(attempts.SequenceEqual(["cuda", "cpu"]), "Linux auto probe order");
    }

    [Test]
    public static void AutoWindows_PrefersCudaThenD3d12ThenCpu()
    {
        var attempts = new List<string>();
        BackendSelection.Choice<string> choice = BackendSelection.Create("auto", isWindows: true, candidate =>
        {
            attempts.Add(candidate);
            if (candidate == "cuda") throw new InvalidOperationException("unavailable");
            return candidate;
        });
        Check.True(choice.Name == "gpu", "Windows auto falls back to D3D12");
        Check.True(attempts.SequenceEqual(["cuda", "gpu"]), "Windows auto probe order");
    }

    [Test]
    public static void AutoCpuOnly_StillSucceeds()
    {
        BackendSelection.Choice<string> choice = BackendSelection.Create("auto", isWindows: true,
            candidate => candidate == "cpu" ? candidate : throw new InvalidOperationException("unavailable"));
        Check.True(choice.Name == "cpu", "CPU-only auto selection succeeds");
    }

    [Test]
    public static void ExplicitAccelerator_DoesNotFallback()
    {
        int attempts = 0;
        bool threw = false;
        try
        {
            BackendSelection.Create<string>("cuda", isWindows: false, candidate =>
            { attempts++; throw new InvalidOperationException("CUDA failed"); });
        }
        catch (InvalidOperationException) { threw = true; }
        Check.True(threw, "explicit CUDA propagates initialization failure");
        Check.True(attempts == 1, "explicit CUDA is attempted exactly once");
    }
}
