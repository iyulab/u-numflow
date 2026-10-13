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

    /// <summary>Which family this is.</summary>
    public DistributionKind Kind { get; }

    private Distribution(DistributionKind kind, JsonObject parameters)
    {
        Kind = kind;
        _spec = new JsonObject { ["kind"] = WireName(kind) };
        foreach (var (name, value) in parameters)
            _spec[name] = value?.DeepClone();
        var moments = Engine.Call(NativeInterop.unumflow_distribution_moments, Request());
        Mean = moments.GetProperty("mean").GetDouble();
        Variance = moments.GetProperty("variance").GetDouble();
    }

    private JsonObject Request() => new() { ["distribution"] = _spec.DeepClone() };

    /// <summary>The engine's name for <paramref name="kind"/>.</summary>
    internal static string WireName(DistributionKind kind) => kind switch
    {
        DistributionKind.Uniform => "uniform",
        DistributionKind.Triangular => "triangular",
        DistributionKind.Pert => "pert",
        DistributionKind.Normal => "normal",
        DistributionKind.LogNormal => "lognormal",
        DistributionKind.Weibull => "weibull",
        DistributionKind.Exponential => "exponential",
        DistributionKind.Gamma => "gamma",
        DistributionKind.Beta => "beta",
        DistributionKind.ChiSquared => "chi_squared",
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "not a distribution kind"),
    };

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
        new(DistributionKind.Uniform, Params(("min", min), ("max", max)));

    /// <summary>Triangular on [<paramref name="min"/>, <paramref name="max"/>] peaking at <paramref name="mode"/>.</summary>
    public static Distribution Triangular(double min, double mode, double max) =>
        new(DistributionKind.Triangular, Params(("min", min), ("mode", mode), ("max", max)));

    /// <summary>PERT (a scaled Beta); <paramref name="lambda"/> weights the mode (4 when omitted).</summary>
    public static Distribution Pert(double min, double mode, double max, double? lambda = null)
    {
        var p = Params(("min", min), ("mode", mode), ("max", max));
        if (lambda is { } l)
            p["lambda"] = Engine.Finite("distribution.lambda", l);
        return new(DistributionKind.Pert, p);
    }

    /// <summary>Normal with mean <paramref name="mu"/> and standard deviation <paramref name="sigma"/>.</summary>
    public static Distribution Normal(double mu, double sigma) =>
        new(DistributionKind.Normal, Params(("mu", mu), ("sigma", sigma)));

    /// <summary>Log-normal: ln X ~ Normal(<paramref name="mu"/>, <paramref name="sigma"/>).</summary>
    public static Distribution LogNormal(double mu, double sigma) =>
        new(DistributionKind.LogNormal, Params(("mu", mu), ("sigma", sigma)));

    /// <summary>Weibull with <paramref name="shape"/> (β) and <paramref name="scale"/> (η).</summary>
    public static Distribution Weibull(double shape, double scale) =>
        new(DistributionKind.Weibull, Params(("shape", shape), ("scale", scale)));

    /// <summary>Exponential with <paramref name="rate"/> (λ).</summary>
    public static Distribution Exponential(double rate) =>
        new(DistributionKind.Exponential, Params(("rate", rate)));

    /// <summary>Gamma with <paramref name="shape"/> (α) and <paramref name="rate"/> (β).</summary>
    public static Distribution Gamma(double shape, double rate) =>
        new(DistributionKind.Gamma, Params(("shape", shape), ("rate", rate)));

    /// <summary>Beta on [0, 1] with shapes <paramref name="alpha"/> and <paramref name="beta"/>.</summary>
    public static Distribution Beta(double alpha, double beta) =>
        new(DistributionKind.Beta, Params(("alpha", alpha), ("beta", beta)));

    /// <summary>Chi-squared with <paramref name="k"/> degrees of freedom.</summary>
    public static Distribution ChiSquared(double k) =>
        new(DistributionKind.ChiSquared, Params(("k", k)));

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
    /// is infinite. <paramref name="seed"/> must be at most 2⁵³. A negative <paramref name="n"/>
    /// is refused by the engine like any other setting: a <see cref="NumflowException"/> with
    /// <c>parameter_out_of_range</c> on <c>n</c>.
    /// </summary>
    public double[] Sample(int n, long seed)
    {
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

/// <summary>The distribution families u-numflow has; <see cref="Distribution.Kind"/> says which one a distribution is.</summary>
public enum DistributionKind
{
    /// <summary>Continuous uniform.</summary>
    Uniform,
    /// <summary>Triangular.</summary>
    Triangular,
    /// <summary>PERT (a scaled Beta).</summary>
    Pert,
    /// <summary>Normal.</summary>
    Normal,
    /// <summary>Log-normal.</summary>
    LogNormal,
    /// <summary>Two-parameter Weibull.</summary>
    Weibull,
    /// <summary>Exponential.</summary>
    Exponential,
    /// <summary>Gamma.</summary>
    Gamma,
    /// <summary>Beta.</summary>
    Beta,
    /// <summary>χ² (chi-squared).</summary>
    ChiSquared,
}
