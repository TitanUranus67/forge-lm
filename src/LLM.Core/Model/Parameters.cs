
namespace LLM.Core.Model
{
    using LLM.Core.Tensor;

    /// <summary>
    /// Named registry of all trainable parameters of a model. Each entry holds the
    /// weight tensor and a same-shaped gradient tensor. Names are unique and iterate
    /// in registration order, so checkpoint serialization is deterministic.
    /// </summary>
    public sealed class Parameters
    {
        private readonly Dictionary<string, (Tensor Weight, Tensor Grad)> _map = new();
        private readonly List<string> _names = new();

        /// <summary>
        /// Registers a parameter: creates the weight tensor (zero-initialized; callers
        /// fill it with the init scheme) and a zeroed gradient tensor. Returns the weight.
        /// </summary>
        public Tensor Add(string name, params int[] shape)
        {
            if (_map.ContainsKey(name)) throw new ArgumentException($"Duplicate parameter name '{name}'.");
            var weight = new Tensor(shape);
            var grad = new Tensor(shape);
            _map.Add(name, (weight, grad));
            _names.Add(name);
            return weight;
        }

        /// <summary>Returns the weight tensor registered under <paramref name="name"/>.</summary>
        public Tensor Weight(string name) =>
            _map.TryGetValue(name, out var e) ? e.Weight : throw new KeyNotFoundException($"No parameter '{name}'.");

        /// <summary>Returns the gradient tensor registered under <paramref name="name"/>.</summary>
        public Tensor Grad(string name) =>
            _map.TryGetValue(name, out var e) ? e.Grad : throw new KeyNotFoundException($"No parameter '{name}'.");

        /// <summary>All parameter names in registration order.</summary>
        public IEnumerable<string> Names => _names;

        /// <summary>Total number of scalar parameters across all tensors.</summary>
        public long Count => _map.Values.Sum(e => (long)e.Weight.Length);

        /// <summary>Zeroes every gradient tensor.</summary>
        public void ZeroGrads()
        {
            foreach (var (_, grad) in _map.Values) grad.Zero();
        }
    }
}
