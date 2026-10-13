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
        Assert.Equal("parameter_out_of_range", e.Reason);
        Assert.Equal("n", e.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(-3, e.Details!.Value.GetProperty("got").GetDouble());
    }
}
