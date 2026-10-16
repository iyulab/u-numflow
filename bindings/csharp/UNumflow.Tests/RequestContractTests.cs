using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using UNumflow.Interop;
using Xunit;

namespace UNumflow.Tests;

/// <summary>
/// The requests the client writes and the ones the engine reads are the same, both ways, against
/// the real native library with reflection-based serialization disabled: every key and name the
/// client can write is one the entry point reads (<c>request_keys.json</c>, which the engine's
/// tests measure from the entry points), every one the entry point reads can be written, and a
/// request with every setting given runs.
/// </summary>
public class RequestContractTests
{
    /// <summary>Places of a request the client does not write, each with why.</summary>
    private static readonly Dictionary<string, string> NotWritten = [];

    /// <summary>Every distribution, each parameter given.</summary>
    private static readonly Dictionary<string, Func<Distribution>> Distributions = new()
    {
        ["uniform"] = () => Distribution.Uniform(0, 10),
        ["triangular"] = () => Distribution.Triangular(0, 3, 10),
        ["pert"] = () => Distribution.Pert(0, 3, 10, lambda: 5),
        ["normal"] = () => Distribution.Normal(0, 1),
        ["lognormal"] = () => Distribution.LogNormal(0, 0.5),
        ["weibull"] = () => Distribution.Weibull(2, 100),
        ["exponential"] = () => Distribution.Exponential(0.5),
        ["gamma"] = () => Distribution.Gamma(3, 2),
        ["beta"] = () => Distribution.Beta(2, 5),
        ["chi_squared"] = () => Distribution.ChiSquared(4),
    };

    private static IntervalSet Set => IntervalSet.From([(0, 2), (1, 4), (6, 7)]);
    private static IntervalSet Other => IntervalSet.From([(3, 5)]);

    /// <summary>A definition with every setting given.</summary>
    private static Units Defined => new([new UnitDefinition("degF", "K", 5.0 / 9, 459.67 * 5 / 9)]);

    /// <summary>Every call with every setting it has given.</summary>
    private static readonly Dictionary<string, Action> Full = Distributions
        .Select(d => KeyValuePair.Create<string, Action>($"Distribution({d.Key})", () =>
        {
            var distribution = d.Value();
            Assert.Equal(d.Key, Distribution.WireName(distribution.Kind));
            distribution.Cdf(distribution.Mean);
            distribution.Quantile(0.9);
            distribution.Sample(3, seed: 7);
        }))
        .Concat(new Dictionary<string, Action>
        {
            ["IntervalSet.From"] = () => IntervalSet.From([(0, 2), (1, 4), (6, 7)]),
            ["IntervalSet.Measure"] = () => _ = Set.Measure,
            ["IntervalSet.Contains"] = () => Set.Contains(1.5),
            ["IntervalSet.Union"] = () => Set.Union(Other),
            ["IntervalSet.Intersection"] = () => Set.Intersection(Other),
            ["IntervalSet.Difference"] = () => Set.Difference(Other),
            ["Units.Convert"] = () => Defined.Convert(1, "degF", "Cel"),
            ["Units.ConvertDifference"] = () => Defined.ConvertDifference(1, "degF", "K"),
            ["Units.Info"] = () => Defined.Info("degF"),
        })
        .ToDictionary();

    /// <summary>Each call's requests: (call, entry point, body). A refused one is still written.</summary>
    private static readonly Lazy<List<(string Call, string Entry, string Body)>> Written = new(() =>
    {
        var written = new List<(string, string, string)>();
        try
        {
            foreach (var (call, run) in Full)
            {
                Engine.RequestObserver = (entry, body) => written.Add((call, entry, body));
                try
                {
                    run();
                }
                catch (NumflowException)
                {
                }
            }
        }
        finally
        {
            Engine.RequestObserver = null;
        }
        return written;
    });

    public static TheoryData<string> Calls() => [.. Full.Keys];

    [Theory]
    [MemberData(nameof(Calls))]
    public void A_request_with_every_setting_is_accepted_and_runs(string call) => Full[call]();

    [Fact]
    public void Every_key_and_name_the_client_writes_is_one_the_engine_reads()
    {
        var problems = new List<string>();
        foreach (var (call, entry, body) in Written.Value)
        {
            var found = new List<string>();
            RequestKeys.Read(RequestKeys.Schema(entry), JsonNode.Parse(body), "$", [], found);
            problems.AddRange(found.Select(p => $"{call} → {entry} {p}"));
        }
        Assert.True(problems.Count == 0, string.Join("\n", problems.Distinct()));
    }

    [Fact]
    public void Every_key_and_name_the_engine_reads_is_one_the_client_writes()
    {
        var written = new HashSet<string>();
        foreach (var (_, entry, body) in Written.Value)
        {
            var places = new HashSet<string>();
            RequestKeys.Read(RequestKeys.Schema(entry), JsonNode.Parse(body), "$", places, []);
            written.UnionWith(places.Select(p => $"{entry} {p}"));
        }
        var engine = RequestKeys.EntryPoints
            .SelectMany(e => RequestKeys.Places(e.Key).Select(p => $"{e.Key} {p}"))
            .ToHashSet();

        var unwritable = engine.Except(written).Except(NotWritten.Keys).Order(StringComparer.Ordinal).ToList();
        Assert.True(unwritable.Count == 0, $"the engine reads what the client cannot write:\n{string.Join("\n", unwritable)}");

        var stale = NotWritten.Keys.Where(p => written.Contains(p) || !engine.Contains(p)).ToList();
        Assert.True(stale.Count == 0, $"stale reasons: {string.Join(", ", stale)}");
    }

    [Fact]
    public void Every_entry_point_the_engine_measures_is_bound()
    {
        var bound = typeof(NativeInterop)
            .GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly)
            .Select(m => m.Name)
            .Where(n => n is not ("unumflow_version" or "unumflow_codes" or "unumflow_free_string"))
            .Order(StringComparer.Ordinal);
        Assert.Equal(RequestKeys.EntryPoints.Select(e => e.Key).Order(StringComparer.Ordinal), bound);
    }

    [Fact]
    public void Each_distribution_is_a_kind_the_engine_has()
        => Assert.Equal(
            RequestKeys.Places("unumflow_distribution_moments")
                .Where(p => p.StartsWith("$.distribution(kind=\"", StringComparison.Ordinal) && p.EndsWith("\")", StringComparison.Ordinal))
                .Select(p => p["$.distribution(kind=\"".Length..^2])
                .Order(StringComparer.Ordinal),
            Distributions.Keys.Order(StringComparer.Ordinal));

    [Fact]
    public void The_suite_runs_as_a_trimmed_host_does()
    {
        Assert.False(JsonSerializer.IsReflectionEnabledByDefault);
        Assert.Throws<InvalidOperationException>(() => JsonSerializer.Serialize(new { x = 1.0 }, new JsonSerializerOptions()));
    }
}
