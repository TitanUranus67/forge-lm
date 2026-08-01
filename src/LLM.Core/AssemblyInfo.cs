using System.Runtime.CompilerServices;

// Lets the hand-rolled test runner (LLM.Core.Tests) exercise internal helpers
// such as GpuBackend.BucketOf without widening the public API.
[assembly: InternalsVisibleTo("LLM.Core.Tests")]
