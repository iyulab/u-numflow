using System.Text.Json.Nodes;
using UNumflow.Interop;

namespace UNumflow;

/// <summary>
/// One of u-numflow's continuous distributions. Create it with a factory such as
/// <see cref="Weibull"/>; the parameters are checked there, so a distribution that
/// exists is a valid one.
/// </summary>
/// <example>
/// <code>
/// var chi = Distribution.ChiSquared(4);
/// double critical = chi.Quantile(0.95);          // 9.4877…
/// double[] lives = Distribution.Weibull(2, 100).Sample(1000, seed: 42);
/// </code>
/// </example>
public sealed class Distribution
{
    private readonly JsonObject _spec;

    /// <summary>The mean.</summary>
    public double Mean { get; }

    /// <summary>The variance.</summary>
    public double Variance { get; }

    /// <summary>The engine's name for the distribution (<c>"weibull"</c>, <c>"chi_squared"</c>, …).</summary>
    public string Kind { get; }

    private Distribution(string kind, JsonObject parameters)
    {
        Kind = kind;
        _spec = new JsonObject { ["kind"] = kind };
        foreach (var (name, value) in parameters)
            _spec[name] = value?.DeepClone();
        var moments = Engine.Call(NativeInterop.unumflow_distribution_moments, Request());
        Mean = moments.GetProperty("mean").GetDouble();
        Variance = moments.GetProperty("variance").GetDouble();
    }

    private JsonObject Request() => new() { ["distribution"] = _spec.DeepClone() };

    /// <summary>The parameters as JSON, each checked finite (JSON cannot carry NaN or ±∞).</summary>
    private static JsonObject Params(params (string Name, double Value)[] values)
    {
        var o = new JsonObject();
        foreach (var (name, value) in values)
            o[name] = Engine.Finite($"distribution.{name}", value);
        return o;
    }

    /// <summary>Continuous uniform on [<paramref name="min"/>, <paramref name="max"/>].</summary>
    public static Distribution Uniform(double min, double max) =>
        new("uniform", Params(("min", min), ("max", max)));

    /// <summary>Triangular on [<paramref name="min"/>, <paramref name="max"/>] peaking at <paramref name="mode"/>.</summary>
    public static Distribution Triangular(double min, double mode, double max) =>
        new("triangular", Params(("min", min), ("mode", mode), ("max", max)));

    /// <summary>PERT (a scaled Beta); <paramref name="lambda"/> weights the mode (4 when omitted).</summary>
    public static Distribution Pert(double min, double mode, double max, double? lambda = null)
    {
        var p = Params(("min", min), ("mode", mode), ("max", max));
        if (lambda is { } l)
            p["lambda"] = Engine.Finite("distribution.lambda", l);
        return new("pert", p);
    }

    /// <summary>Normal with mean <paramref name="mu"/> and standard deviation <paramref name="sigma"/>.</summary>
    public static Distribution Normal(double mu, double sigma) =>
        new("normal", Params(("mu", mu), ("sigma", sigma)));

    /// <summary>Log-normal: ln X ~ Normal(<paramref name="mu"/>, <paramref name="sigma"/>).</summary>
    public static Distribution LogNormal(double mu, double sigma) =>
        new("lognormal", Params(("mu", mu), ("sigma", sigma)));

    /// <summary>Weibull with <paramref name="shape"/> (β) and <paramref name="scale"/> (η).</summary>
    public static Distribution Weibull(double shape, double scale) =>
        new("weibull", Params(("shape", shape), ("scale", scale)));

    /// <summary>Exponential with <paramref name="rate"/> (λ).</summary>
    public static Distribution Exponential(double rate) =>
        new("exponential", Params(("rate", rate)));

    /// <summary>Gamma with <paramref name="shape"/> (α) and <paramref name="rate"/> (β).</summary>
    public static Distribution Gamma(double shape, double rate) =>
        new("gamma", Params(("shape", shape), ("rate", rate)));

    /// <summary>Beta on [0, 1] with shapes <paramref name="alpha"/> and <paramref name="beta"/>.</summary>
    public static Distribution Beta(double alpha, double beta) =>
        new("beta", Params(("alpha", alpha), ("beta", beta)));

    /// <summary>Chi-squared with <paramref name="k"/> degrees of freedom.</summary>
    public static Distribution ChiSquared(double k) =>
        new("chi_squared", Params(("k", k)));

    /// <summary>P(X ≤ <paramref name="x"/>).</summary>
    public double Cdf(double x)
    {
        if (double.IsNegativeInfinity(x))
            return 0.0;
        if (double.IsPositiveInfinity(x))
            return 1.0;
        var r = Request();
        Engine.Finite("x", x);
        r["x"] = x;
        return Engine.Call(NativeInterop.unumflow_distribution_cdf, r).GetProperty("value").GetDouble();
    }

    /// <summary>The x with P(X ≤ x) = <paramref name="p"/>, for p strictly inside (0, 1).</summary>
    public double Quantile(double p)
    {
        Engine.Finite("p", p);
        var r = Request();
        r["p"] = p;
        return Engine.Call(NativeInterop.unumflow_distribution_quantile, r).GetProperty("value").GetDouble();
    }

    /// <summary>
    /// <paramref name="n"/> random variates. The same <paramref name="seed"/> always gives the
    /// same values; the uniforms behind them come from the open interval (0, 1), so no variate
    /// is infinite. <paramref name="seed"/> must be at most 2⁵³.
    /// </summary>
    public double[] Sample(int n, long seed)
    {
        if (n < 0)
            throw new ArgumentOutOfRangeException(nameof(n), n, "n must be at least 0.");
        var r = Request();
        r["n"] = n;
        r["seed"] = seed;
        var values = Engine.Call(NativeInterop.unumflow_distribution_sample, r).GetProperty("values");
        var result = new double[values.GetArrayLength()];
        var i = 0;
        foreach (var v in values.EnumerateArray())
            result[i++] = v.GetDouble();
        return result;
    }
}
