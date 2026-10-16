using System.Text.Json;
using System.Text.Json.Nodes;
using UNumflow.Interop;

namespace UNumflow;

/// <summary>
/// A unit the caller adds to the SI: <paramref name="Symbol"/> = <paramref name="Factor"/> ×
/// <paramref name="Unit"/>, plus <paramref name="Offset"/> SI units at its zero for a scale such
/// as a Fahrenheit degree.
/// </summary>
/// <param name="Symbol">The new symbol, e.g. <c>cal_IT</c>.</param>
/// <param name="Unit">What it is a multiple of, as a unit expression, e.g. <c>J</c>. It may use
/// the SI and the definitions listed before it.</param>
/// <param name="Factor">How many <paramref name="Unit"/> one <paramref name="Symbol"/> is (&gt; 0).</param>
/// <param name="Offset">SI value at the symbol's zero (0 except for a scale).</param>
public sealed record UnitDefinition(string Symbol, string Unit, double Factor = 1, double Offset = 0);

/// <summary>What a unit expression stands for: <c>value_in_si = value · Scale + Offset</c>.</summary>
/// <param name="Dimension">The dimension in SI base units, e.g. <c>kg.m2.s-2</c> (<c>1</c> for a
/// pure number). Two units convert into each other exactly when their dimensions are equal.</param>
/// <param name="Exponents">Exponents of m, kg, s, A, K, mol, cd.</param>
/// <param name="Scale">Size of one unit in coherent SI units.</param>
/// <param name="Offset">SI value of the unit's zero (0 except for scales such as <c>Cel</c>).</param>
public sealed record UnitInfo(string Dimension, IReadOnlyList<int> Exponents, double Scale, double Offset);

/// <summary>
/// Conversion between units of the same dimension. Units are UCUM expressions — <c>kW.h</c>,
/// <c>m/s2</c>, <c>J/(kg.K)</c>, <c>Gcal_IT</c> — over the SI (base units, the 22 named derived
/// units, <c>min</c> <c>h</c> <c>d</c> <c>L</c> <c>t</c>, decimal prefixes) and the
/// <see cref="Definitions"/> given. The engine embeds no other unit table, so a unit whose size
/// varies by convention (the calorie, the tonne of oil equivalent) is the one you define.
/// </summary>
/// <example>
/// <code>
/// var units = new Units([
///     new UnitDefinition("cal_IT", "J", 4.1868),
///     new UnitDefinition("toe", "kcal_IT", 1e7),
/// ]);
/// units.Convert(1, "Gcal_IT", "GJ");          // 4.1868
/// units.Convert(20, "Cel", "K");               // 293.15 — a reading
/// units.ConvertDifference(20, "Cel", "K");     // 20 — a change
/// units.Info("toe").Dimension;                 // "kg.m2.s-2"
/// </code>
/// </example>
public sealed class Units
{
    /// <summary>The SI alone.</summary>
    public static Units Si { get; } = new([]);

    /// <summary>The units added to the SI, in order.</summary>
    public IReadOnlyList<UnitDefinition> Definitions { get; }

    /// <summary>The SI plus <paramref name="definitions"/>.</summary>
    /// <remarks>The definitions are checked on first use, where their refusal names the
    /// definition (<c>definitions[2].factor</c>).</remarks>
    public Units(IEnumerable<UnitDefinition> definitions) => Definitions = [.. definitions];

    /// <summary>A reading <paramref name="value"/> in <paramref name="from"/>, in
    /// <paramref name="to"/>: 20 <c>Cel</c> is 293.15 <c>K</c>.</summary>
    /// <exception cref="NumflowException">
    /// <c>dimension_mismatch</c> (<c>parameter</c> <c>to</c>; <c>Details</c> carries
    /// <c>expected</c> and <c>got</c>, the two dimensions); <c>unknown_unit</c> (<c>got</c> = the
    /// symbol); <c>malformed_input</c> for an expression that does not parse;
    /// <c>invalid_option</c> for a scale such as <c>Cel</c> given a prefix, exponent or product;
    /// for a definition, <c>duplicate_id</c> and <c>parameter_out_of_range</c> at
    /// <c>definitions[i].…</c>; <c>value_not_finite</c> for NaN or ±∞.
    /// </exception>
    public double Convert(double value, string from, string to) =>
        Engine.Call(NativeInterop.unumflow_unit_convert, Conversion(value, from, to))
            .GetProperty("value").GetDouble();

    /// <summary>An amount of change <paramref name="value"/> in <paramref name="from"/>, in
    /// <paramref name="to"/>: the offset of a scale cancels, so a rise of 20 <c>Cel</c> is a rise
    /// of 20 <c>K</c>.</summary>
    /// <exception cref="NumflowException">As <see cref="Convert"/>.</exception>
    public double ConvertDifference(double value, string from, string to) =>
        Engine.Call(NativeInterop.unumflow_unit_convert_difference, Conversion(value, from, to))
            .GetProperty("value").GetDouble();

    /// <summary>What <paramref name="unit"/> stands for.</summary>
    /// <exception cref="NumflowException">As <see cref="Convert"/>, at <c>unit</c>.</exception>
    public UnitInfo Info(string unit)
    {
        var request = WithDefinitions(new JsonObject { ["unit"] = unit });
        var body = Engine.Call(NativeInterop.unumflow_unit_info, request);
        return body.Deserialize(NumflowJson.Default.UnitInfo)
            ?? throw new InvalidOperationException("The native engine returned no unit info.");
    }

    private JsonObject Conversion(double value, string from, string to) => WithDefinitions(new JsonObject
    {
        ["value"] = Engine.Finite("value", value),
        ["from"] = from,
        ["to"] = to,
    });

    private JsonObject WithDefinitions(JsonObject request)
    {
        if (Definitions.Count == 0)
            return request;
        var list = new JsonArray();
        for (var i = 0; i < Definitions.Count; i++)
        {
            var d = Definitions[i];
            list.Add((JsonNode)new JsonObject
            {
                ["symbol"] = d.Symbol,
                ["unit"] = d.Unit,
                ["factor"] = Engine.Finite($"definitions[{i}].factor", d.Factor),
                ["offset"] = Engine.Finite($"definitions[{i}].offset", d.Offset),
            });
        }
        request["definitions"] = list;
        return request;
    }
}
