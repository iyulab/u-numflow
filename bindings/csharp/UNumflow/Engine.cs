using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using UNumflow.Interop;

namespace UNumflow;

/// <summary>The native library's version.</summary>
public static class Numflow
{
    /// <summary>Version of the bundled u-numflow library.</summary>
    public static string Version
    {
        get
        {
            var ptr = NativeInterop.unumflow_version();
            try
            {
                return Marshal.PtrToStringUTF8(ptr) ?? "unknown";
            }
            finally
            {
                NativeInterop.unumflow_free_string(ptr);
            }
        }
    }
}

/// <summary>One JSON request to the native library, one JSON body back.</summary>
internal static class Engine
{
    internal delegate int Export(string requestJson, out IntPtr resultPtr);

    internal static JsonElement Call(Export export, JsonObject request)
    {
        var status = export(request.ToJsonString(), out var resultPtr);
        try
        {
            if (resultPtr == IntPtr.Zero)
                throw new NumflowException(status, "The engine returned no result.");
            var body = Marshal.PtrToStringUTF8(resultPtr) ?? "";
            if (status != 0)
                throw NumflowException.FromErrorBody(status, body);
            using var doc = JsonDocument.Parse(body);
            return doc.RootElement.Clone();
        }
        finally
        {
            if (resultPtr != IntPtr.Zero)
                NativeInterop.unumflow_free_string(resultPtr);
        }
    }

    /// <summary>
    /// <paramref name="x"/> if it is finite. JSON has no NaN or infinity, so such a value
    /// cannot reach the engine; it is refused here with the engine's own reason.
    /// </summary>
    internal static double Finite(string parameter, double x, int? index = null)
    {
        if (double.IsFinite(x))
            return x;
        var body = new JsonObject
        {
            ["error"] = $"{parameter}: expected a finite number, got {x.ToString(CultureInfo.InvariantCulture)}",
            ["code"] = "value_not_finite",
            ["parameter"] = parameter,
        };
        if (index is { } i)
            body["index"] = i;
        throw NumflowException.FromErrorBody(-3, body.ToJsonString());
    }

    internal static JsonArray Rows(string parameter, IEnumerable<(double Start, double End)> intervals)
    {
        var rows = new JsonArray();
        var i = 0;
        foreach (var (start, end) in intervals)
        {
            rows.Add(new JsonArray(Finite($"{parameter}[{i}]", start, 0), Finite($"{parameter}[{i}]", end, 1)));
            i++;
        }
        return rows;
    }

    internal static IReadOnlyList<(double Start, double End)> Pieces(JsonElement body)
    {
        var list = new List<(double, double)>();
        foreach (var row in body.GetProperty("intervals").EnumerateArray())
            list.Add((row[0].GetDouble(), row[1].GetDouble()));
        return list;
    }
}

/// <summary>
/// A request the engine refused. <see cref="Exception.Message"/> is readable text;
/// <see cref="Reason"/> and <see cref="Details"/> are for programs.
/// </summary>
public class NumflowException : Exception
{
    /// <summary>Native status: -1 null pointer, -2 malformed request, -3 refused request, -4 internal panic.</summary>
    public int Code { get; }

    /// <summary>
    /// Stable, machine-readable reason, e.g. <c>parameter_out_of_range</c>, <c>unknown_option</c>,
    /// <c>invalid_option</c>, <c>value_not_finite</c>, <c>reversed_interval</c>,
    /// <c>malformed_input</c>, <c>internal</c>. <c>null</c> when the engine returned no readable body.
    /// </summary>
    public string? Reason { get; }

    /// <summary>
    /// The whole error body: <c>error</c>, <c>code</c> and the values behind the reason
    /// (<c>parameter</c>, <c>index</c>, <c>min</c>, <c>max</c>, <c>got</c>, <c>expected</c>).
    /// <c>null</c> when there is no body.
    /// </summary>
    public JsonElement? Details { get; }

    public NumflowException(int code, string message) : base(message)
    {
        Code = code;
    }

    public NumflowException(int code, string message, string? reason, JsonElement? details) : base(message)
    {
        Code = code;
        Reason = reason;
        Details = details;
    }

    /// <summary>Reads the engine's <c>{"error", "code", ...}</c> error body.</summary>
    internal static NumflowException FromErrorBody(int code, string body)
    {
        try
        {
            using var doc = JsonDocument.Parse(body);
            var root = doc.RootElement;
            var message = root.TryGetProperty("error", out var e) && e.ValueKind == JsonValueKind.String
                ? e.GetString()!
                : body;
            string? reason = root.TryGetProperty("code", out var c) && c.ValueKind == JsonValueKind.String
                ? c.GetString()
                : null;
            return new NumflowException(code, message, reason, root.Clone());
        }
        catch (JsonException)
        {
            return new NumflowException(code, body);
        }
    }
}
