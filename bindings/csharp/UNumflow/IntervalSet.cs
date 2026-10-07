using System.Text.Json.Nodes;
using UNumflow.Interop;

namespace UNumflow;

/// <summary>
/// A finite union of half-open intervals [start, end), kept in normal form: sorted, disjoint,
/// touching pieces merged, empty ones dropped. Because the pieces never overlap,
/// <see cref="Measure"/> counts every point once, however much the input overlapped.
/// Immutable — every operation returns a new set.
/// </summary>
/// <example>
/// <code>
/// var planned = IntervalSet.From([(0, 24)]);
/// var unplanned = IntervalSet.From([(10, 30)]);
/// var up = IntervalSet.From([(0, 168)]).Difference(planned.Union(unplanned));
/// // up.Measure == 138: the 14 h the stops share is down-time once, not twice
/// </code>
/// </example>
public sealed class IntervalSet
{
    /// <summary>The empty set.</summary>
    public static IntervalSet Empty { get; } = new([]);

    /// <summary>The normal-form pieces, in increasing order.</summary>
    public IReadOnlyList<(double Start, double End)> Pieces { get; }

    private IntervalSet(IReadOnlyList<(double Start, double End)> pieces) => Pieces = pieces;

    /// <summary>
    /// The union of <paramref name="intervals"/>, each read as [Start, End).
    /// </summary>
    /// <exception cref="NumflowException">
    /// <c>reversed_interval</c> for an interval with Start &gt; End (with its <c>index</c>) —
    /// refused rather than swapped; <c>value_not_finite</c> for a NaN or infinite bound
    /// (<c>parameter</c> is the interval's path, e.g. <c>intervals[2]</c>, <c>index</c> 0 or 1).
    /// </exception>
    public static IntervalSet From(IEnumerable<(double Start, double End)> intervals)
    {
        var body = Engine.Call(
            NativeInterop.unumflow_interval_normalize,
            new JsonObject { ["intervals"] = Engine.Rows("intervals", intervals) });
        return new(Engine.Pieces(body));
    }

    /// <summary>Total length — each point counted once.</summary>
    public double Measure =>
        Engine.Call(NativeInterop.unumflow_interval_measure, One())
            .GetProperty("value").GetDouble();

    /// <summary>Whether <paramref name="t"/> lies in the set (Start ≤ t &lt; End for some piece).</summary>
    public bool Contains(double t)
    {
        if (double.IsNaN(t))
            return false;
        if (double.IsInfinity(t))
            return false; // every piece is bounded
        var r = One();
        r["t"] = t;
        return Engine.Call(NativeInterop.unumflow_interval_contains, r).GetProperty("value").GetBoolean();
    }

    /// <summary>Points in this set or <paramref name="other"/>.</summary>
    public IntervalSet Union(IntervalSet other) => Binary(NativeInterop.unumflow_interval_union, other);

    /// <summary>Points in both sets.</summary>
    public IntervalSet Intersection(IntervalSet other) => Binary(NativeInterop.unumflow_interval_intersection, other);

    /// <summary>Points in this set but not in <paramref name="other"/>.</summary>
    public IntervalSet Difference(IntervalSet other) => Binary(NativeInterop.unumflow_interval_difference, other);

    /// <summary>The part of the set inside [<paramref name="start"/>, <paramref name="end"/>).</summary>
    public IntervalSet Clip(double start, double end) => Intersection(From([(start, end)]));

    private JsonObject One() => new() { ["intervals"] = Engine.Rows("intervals", Pieces) };

    private IntervalSet Binary(Engine.Export export, IntervalSet other)
    {
        var body = Engine.Call(export, new JsonObject
        {
            ["a"] = Engine.Rows("a", Pieces),
            ["b"] = Engine.Rows("b", other.Pieces),
        });
        return new(Engine.Pieces(body));
    }
}
