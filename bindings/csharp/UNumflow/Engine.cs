using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
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

    /// <summary>
    /// Every refusal <see cref="NumflowException.ErrorCode"/> the engine and this client
    /// raise, with the fields beside each — built from the table the refusals are made from,
    /// so a caller that translates codes can check its list against the engine's.
    /// </summary>
    public static Codes Codes =>
        JsonSerializer.Deserialize(CodesJson(), NumflowJson.Default.Codes)
        ?? throw new InvalidOperationException("The native engine returned no code catalog.");

    /// <summary>The code catalog as the native engine returns it.</summary>
    internal static string CodesJson()
    {
        var ptr = NativeInterop.unumflow_codes();
        try
        {
            return Marshal.PtrToStringUTF8(ptr)
                ?? throw new InvalidOperationException("The native engine returned no code catalog.");
        }
        finally
        {
            NativeInterop.unumflow_free_string(ptr);
        }
    }
}

/// <summary>A refusal reason: the <c>code</c> an exception's <c>ErrorCode</c> carries and the
/// error body fields that can stand beside it.</summary>
/// <param name="Code">The stable reason, e.g. <c>parameter_out_of_range</c>.</param>
/// <param name="Fields">The error body's fields beside it (<c>parameter</c>, <c>min</c>, …).</param>
public sealed record RefusalCode(string Code, IReadOnlyList<string> Fields);

/// <summary>A violation a checker reports: its <c>code</c>, the fields beside it, and the
/// engine operations that report it.</summary>
/// <param name="Code">The stable reason.</param>
/// <param name="Fields">The violation's fields beside <c>code</c>, in wire names.</param>
/// <param name="Checkers">The engine operations reporting it, by their engine names.</param>
public sealed record ViolationCode(string Code, IReadOnlyList<string> Fields, IReadOnlyList<string> Checkers);

/// <summary>Every code the engine can answer with besides a result — the same catalog the
/// WebAssembly package's <c>codes()</c> returns, so it also names reasons only functions
/// outside this package's surface raise.</summary>
/// <param name="Refusals">Each refusal reason.</param>
/// <param name="Violations">Each checker violation (none in this engine).</param>
public sealed record Codes(IReadOnlyList<RefusalCode> Refusals, IReadOnlyList<ViolationCode> Violations);

/// <summary>Source-generated reading of the code catalog — no reflection.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    RespectNullableAnnotations = true,
    RespectRequiredConstructorParameters = true)]
[JsonSerializable(typeof(Codes))]
internal sealed partial class NumflowJson : JsonSerializerContext;

/// <summary>One JSON request to the native library, one JSON body back.</summary>
internal static class Engine
{
    internal delegate int Export(string requestJson, out IntPtr resultPtr);

    /// <summary>
    /// Sees each request body as it is sent, with the entry point it goes to — the contract
    /// tests check every key and name in it against the ones the engine reads.
    /// </summary>
    internal static Action<string, string>? RequestObserver { get; set; }

    internal static JsonElement Call(Export export, JsonObject request)
    {
        var json = request.ToJsonString();
        RequestObserver?.Invoke(export.Method.Name, json);
        var status = export(json, out var resultPtr);
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
            rows.Add((JsonNode)new JsonArray(Finite($"{parameter}[{i}]", start, 0), Finite($"{parameter}[{i}]", end, 1)));
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
/// A request the engine refused. <see cref="Exception.Message"/> is readable text; <see
/// cref="ErrorCode"/>,
/// <see cref="Parameter"/>, <see cref="Index"/> and <see cref="Details"/> are for programs —
/// the same members, with the same meaning, on every sibling package's exception.
/// </summary>
public class NumflowException : Exception
{
    /// <summary>Native status: -1 null pointer, -2 malformed request, -3 refused request,
    /// -4 internal panic.</summary>
    public int Status { get; }

    /// <summary>
    /// Stable, machine-readable reason: the <c>code</c> of the engine's error body, the same
    /// name the WebAssembly binding puts on its <c>Error</c> —
    /// <c>parameter_out_of_range</c>, <c>unknown_option</c>, <c>value_not_finite</c>,
    /// <c>reversed_interval</c>, <c>malformed_input</c>, <c>internal</c>, ….
    /// <c>null</c> when the engine returned no readable body.
    /// </summary>
    public string? ErrorCode { get; }

    /// <summary>
    /// The argument or option the refusal is about, as a path into the request
    /// (<c>p</c>, <c>intervals[2]</c>, …) — the body's <c>parameter</c>. <c>null</c> when
    /// the refusal is not
    /// about one value.
    /// </summary>
    public string? Parameter { get; }

    /// <summary>
    /// Zero-based position of the offending element in the array <see cref="Parameter"/>
    /// names — the body's <c>index</c>. <c>null</c> when the value is not an array element.
    /// </summary>
    public int? Index { get; }

    /// <summary>
    /// The whole error body: <c>error</c>, <c>code</c> and the values behind the reason
    /// (<c>parameter</c>, <c>index</c>, <c>min</c>, <c>max</c>, <c>exclusive_min</c>,
    /// <c>exclusive_max</c>, <c>got</c>, <c>expected</c>). For
    /// <c>parameter_out_of_range</c>, <c>min</c> / <c>max</c> are the closed
    /// (included) bounds — <c>null</c> on a side that is open or unbounded — and
    /// <c>exclusive_min</c> / <c>exclusive_max</c> the open (excluded) ones, present only on
    /// an open side. <c>null</c> when there is no body.
    /// </summary>
    public JsonElement? Details { get; }

    /// <summary>A refusal with no readable body.</summary>
    /// <param name="status">Native status, see <see cref="Status"/>.</param>
    /// <param name="message">What went wrong, in words.</param>
    public NumflowException(int status, string message) : base(message)
    {
        Status = status;
    }

    /// <summary>
    /// A refusal with the engine's reason and the values behind it; <see cref="Parameter"/>
    /// and <see cref="Index"/> are read from <paramref name="details"/>.
    /// </summary>
    /// <param name="status">Native status, see <see cref="Status"/>.</param>
    /// <param name="message">What went wrong, in words.</param>
    /// <param name="errorCode">The body's <c>code</c>.</param>
    /// <param name="details">The error body.</param>
    public NumflowException(int status, string message, string? errorCode, JsonElement? details) : base(message)
    {
        Status = status;
        ErrorCode = errorCode;
        Details = details;
        Parameter = ParameterIn(details);
        Index = IndexIn(details);
    }

    /// <summary>Reads the engine's <c>{"error", "code", ...}</c> error body.</summary>
    internal static NumflowException FromErrorBody(int status, string body)
    {
        try
        {
            using var doc = JsonDocument.Parse(body);
            var root = doc.RootElement;
            var message = root.TryGetProperty("error", out var e) && e.ValueKind == JsonValueKind.String
                ? e.GetString()!
                : body;
            string? errorCode = root.TryGetProperty("code", out var c) && c.ValueKind == JsonValueKind.String
                ? c.GetString()
                : null;
            return new NumflowException(status, message, errorCode, root.Clone());
        }
        catch (JsonException)
        {
            return new NumflowException(status, body);
        }
    }

    /// <summary>The body's <c>parameter</c>, when it is a string.</summary>
    private static string? ParameterIn(JsonElement? details) =>
        details is { ValueKind: JsonValueKind.Object } d
        && d.TryGetProperty("parameter", out var p) && p.ValueKind == JsonValueKind.String
            ? p.GetString()
            : null;

    /// <summary>The body's <c>index</c>, when it is a whole number.</summary>
    private static int? IndexIn(JsonElement? details) =>
        details is { ValueKind: JsonValueKind.Object } d
        && d.TryGetProperty("index", out var i) && i.ValueKind == JsonValueKind.Number
        && i.TryGetInt32(out var n)
            ? n
            : null;
}
