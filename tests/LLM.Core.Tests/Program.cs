using System.Reflection;

namespace LLM.Core.Tests;

/// <summary>
/// Hand-rolled test runner (no external test frameworks allowed).
/// Discovers public static parameterless methods marked [Test] and runs them all.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        string? filter = args.Length > 0 ? args[0] : null;
        var tests = Assembly.GetExecutingAssembly().GetTypes()
            .Where(t => t.IsClass && t.IsAbstract && t.IsSealed) // static classes
            .SelectMany(t => t.GetMethods(BindingFlags.Public | BindingFlags.Static))
            .Where(m => m.GetCustomAttribute<TestAttribute>() != null && m.GetParameters().Length == 0)
            .OrderBy(m => m.DeclaringType!.Name).ThenBy(m => m.Name)
            .ToList();

        if (filter != null)
            tests = tests.Where(m => ($"{m.DeclaringType!.Name}.{m.Name}").Contains(filter, StringComparison.OrdinalIgnoreCase)).ToList();

        Console.WriteLine($"Discovered {tests.Count} test(s).");
        int failed = 0;
        foreach (var test in tests)
        {
            string name = $"{test.DeclaringType!.Name}.{test.Name}";
            Check.Failures = 0;
            try
            {
                test.Invoke(null, null);
                if (Check.Failures > 0) { failed++; Console.WriteLine($"[FAIL] {name}"); }
                else Console.WriteLine($"[ ok ] {name}");
            }
            catch (Exception ex)
            {
                failed++;
                Console.WriteLine($"[FAIL] {name} threw {ex.InnerException ?? ex}");
            }
        }

        Console.WriteLine(failed == 0 ? "ALL TESTS PASSED" : $"{failed} TEST(S) FAILED");
        return failed == 0 ? 0 : 1;
    }
}
