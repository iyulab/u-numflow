// Run by scripts/nuget_package_smoke.sh against the freshly packed package:
// the native library loads, answers, and refuses with the documented surface.
using UNumflow;

Console.WriteLine($"native version: {Numflow.Version}");

var z = Distribution.Normal(0, 1).Quantile(0.975);
if (Math.Abs(z - 1.959963984540054) > 1e-9)
    throw new Exception($"Normal(0, 1).Quantile(0.975) = {z}");

var draws = Distribution.Weibull(2, 100).Sample(5, seed: 42);
if (draws.Length != 5 || !draws.All(x => x > 0))
    throw new Exception($"Weibull sample: {string.Join(", ", draws)}");

try
{
    Distribution.Normal(0, sigma: -1);
    throw new Exception("a negative sigma was accepted");
}
catch (NumflowException e) when (e.ErrorCode == "parameter_out_of_range" && e.Parameter == "distribution.sigma")
{
}

Console.WriteLine("nuget package smoke test passed");
