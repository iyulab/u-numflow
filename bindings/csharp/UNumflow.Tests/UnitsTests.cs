using System.Text.Json;
using System.Text.Json.Nodes;
using UNumflow.Interop;
using Xunit;

namespace UNumflow.Tests;

/// <summary>Unit conversion through the native engine, as the README describes it.</summary>
public class UnitsTests
{
    private static readonly Units Energy = new([
        new UnitDefinition("bar", "Pa", 1e5),
        new UnitDefinition("cal_IT", "J", 4.1868),
        new UnitDefinition("cal_th", "J", 4.184),
        new UnitDefinition("toe", "kcal_IT", 1e7),
        new UnitDefinition("Nm3", "m3"),
    ]);

    [Fact]
    public void Converts_readings_and_differences()
    {
        Assert.Equal(100, Energy.Convert(1, "bar", "kPa"), 9);
        Assert.Equal(293.15, Units.Si.Convert(20, "Cel", "K"), 9);
        Assert.Equal(20, Units.Si.ConvertDifference(20, "Cel", "K"), 12);
        Assert.Equal(3.6e6, Units.Si.Convert(1, "kW.h", "J"), 6);
        Assert.Equal(4186.8, Energy.Convert(1, "kcal_IT", "J"), 9);
        Assert.Equal(4184, Energy.Convert(1, "kcal_th", "J"), 9);
        Assert.Equal(41.868, Energy.Convert(1, "toe", "GJ"), 9);
    }

    [Fact]
    public void Volume_is_not_mass()
    {
        var e = Assert.Throws<NumflowException>(() => Energy.Convert(1, "m3", "kg"));
        Assert.Equal("dimension_mismatch", e.ErrorCode);
        Assert.Equal("to", e.Parameter);
        Assert.Equal("m3", e.Details!.Value.GetProperty("expected").GetString());
        Assert.Equal("kg", e.Details!.Value.GetProperty("got").GetString());
    }

    [Fact]
    public void A_definition_is_refused_at_its_place()
    {
        var units = new Units([new UnitDefinition("x", "m", Factor: 0)]);
        var e = Assert.Throws<NumflowException>(() => units.Convert(1, "x", "m"));
        Assert.Equal("parameter_out_of_range", e.ErrorCode);
        Assert.Equal("definitions[0].factor", e.Parameter);

        var nan = new Units([new UnitDefinition("x", "m", Factor: double.NaN)]);
        var n = Assert.Throws<NumflowException>(() => nan.Convert(1, "x", "m"));
        Assert.Equal("value_not_finite", n.ErrorCode);
        Assert.Equal("definitions[0].factor", n.Parameter);
    }

    [Fact]
    public void An_unknown_symbol_is_named()
    {
        var e = Assert.Throws<NumflowException>(() => Units.Si.Convert(1, "kW.hr", "J"));
        Assert.Equal("unknown_unit", e.ErrorCode);
        Assert.Equal("from", e.Parameter);
        Assert.Equal("hr", e.Details!.Value.GetProperty("got").GetString());
    }

    [Fact]
    public void The_unit_info_record_carries_exactly_what_the_engine_sent()
    {
        var request = new JsonObject { ["unit"] = "J/(kg.K)" }.ToJsonString();
        Assert.Equal(0, NativeInterop.unumflow_unit_info(request, out var ptr));
        string raw;
        try
        {
            raw = System.Runtime.InteropServices.Marshal.PtrToStringUTF8(ptr)!;
        }
        finally
        {
            NativeInterop.unumflow_free_string(ptr);
        }
        var info = Units.Si.Info("J/(kg.K)");
        var written = JsonSerializer.SerializeToNode(info, NumflowJson.Default.UnitInfo);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(raw), written), $"engine: {raw}\nrecord: {written}");
        Assert.Equal("m2.s-2.K-1", info.Dimension);
        Assert.Equal([2, 0, -2, 0, -1, 0, 0], info.Exponents);
    }
}
