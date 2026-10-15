using System.Text.Json;
using System.Text.Json.Nodes;
using Xunit;

namespace UNumflow.Tests;

/// <summary>
/// A refusal reaches the caller the way the README says it does: as a
/// <see cref="NumflowException"/> carrying the engine's reason, not as an
/// exception the binding raised in front of the engine.
/// </summary>
public class RefusalTests
{
    [Fact]
    public void A_negative_sample_size_is_the_engines_range_refusal()
    {
        var e = Assert.Throws<NumflowException>(() => Distribution.Normal(0, 1).Sample(-3, seed: 1));
        Assert.Equal("parameter_out_of_range", e.ErrorCode);
        Assert.Equal(-3, e.Status);
        Assert.Equal("n", e.Parameter);
        Assert.Null(e.Index);
        Assert.Equal("n", e.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(-3, e.Details!.Value.GetProperty("got").GetDouble());
    }

    [Fact]
    public void The_code_catalog_carries_exactly_what_the_engine_sent()
    {
        var raw = Numflow.CodesJson();
        var written = JsonSerializer.SerializeToNode(Numflow.Codes, NumflowJson.Default.Codes);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(raw), written), $"engine: {raw}\nrecord: {written}");

        // The client refuses a NaN itself; the catalog names that reason and the one above.
        var codes = Numflow.Codes;
        Assert.Contains(codes.Refusals, r => r.Code == "value_not_finite");
        Assert.Contains(codes.Refusals, r => r.Code == "parameter_out_of_range" && r.Fields.Contains("got"));
        Assert.Empty(codes.Violations);
    }
}
