using System.Text.Json.Nodes;
using UNumflow.Interop;

namespace UNumflow;

/// <summary>
/// Statistics of a sample — the values you have, as opposed to a <see cref="Distribution"/>'s.
/// Quantiles use R-7 linear interpolation (Hyndman &amp; Fan 1996, type 7 — the default of R,
/// NumPy and Excel): for the sorted sample x₀ … xₙ₋₁ and p in [0, 1], h = (n − 1)·p and the
/// quantile is x⌊h⌋ + (h − ⌊h⌋)·(x⌊h⌋₊₁ − x⌊h⌋).
/// </summary>
/// <example>
/// <code>
/// double[] minutes = [12, 7, 95, 9, 14, 11];
/// Stats.Median(minutes);                       // 11.5 — the 95 does not pull it
/// Stats.Quantiles(minutes, [0.25, 0.5, 0.9]);  // [9.5, 11.5, 54.5], one sort for all three
/// </code>
/// </example>
public static class Stats
{
    /// <summary>The median of <paramref name="data"/> — for an even count, the mean of the
    /// middle two.</summary>
    /// <exception cref="NumflowException">
    /// <c>empty_input</c> for no values; <c>value_not_finite</c> for a NaN or infinite value
    /// (with its <c>index</c>).
    /// </exception>
    public static double Median(ReadOnlySpan<double> data) =>
        Engine.Call(NativeInterop.unumflow_median, new JsonObject { ["data"] = Values("data", data) })
            .GetProperty("value").GetDouble();

    /// <summary>The <paramref name="p"/>-quantile of <paramref name="data"/> (R-7), p in [0, 1].</summary>
    /// <exception cref="NumflowException">
    /// As <see cref="Quantiles"/>, with <c>parameter</c> <c>probabilities[0]</c> for
    /// <paramref name="p"/>.
    /// </exception>
    public static double Quantile(ReadOnlySpan<double> data, double p) => Quantiles(data, [p])[0];

    /// <summary>The quantiles of <paramref name="data"/> at each of
    /// <paramref name="probabilities"/> (R-7), in the order asked — the sample is sorted once.</summary>
    /// <exception cref="NumflowException">
    /// <c>empty_input</c> for no values; <c>value_not_finite</c> for a NaN or infinite value or
    /// probability (<c>parameter</c> <c>data</c> or <c>probabilities</c>, with its <c>index</c>);
    /// <c>parameter_out_of_range</c> for a probability outside [0, 1] (<c>parameter</c>
    /// <c>probabilities[i]</c>, <c>min</c> 0, <c>max</c> 1).
    /// </exception>
    public static double[] Quantiles(ReadOnlySpan<double> data, ReadOnlySpan<double> probabilities)
    {
        var values = Engine.Call(NativeInterop.unumflow_quantiles, new JsonObject
        {
            ["data"] = Values("data", data),
            ["probabilities"] = Values("probabilities", probabilities),
        }).GetProperty("values");
        var result = new double[values.GetArrayLength()];
        var i = 0;
        foreach (var v in values.EnumerateArray())
            result[i++] = v.GetDouble();
        return result;
    }

    private static JsonArray Values(string parameter, ReadOnlySpan<double> values)
    {
        var array = new JsonArray();
        for (var i = 0; i < values.Length; i++)
            array.Add((JsonNode)JsonValue.Create(Engine.Finite(parameter, values[i], i)));
        return array;
    }
}
