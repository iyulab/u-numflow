using Xunit;

namespace UNumflow.Tests;

public class StatsTests
{
    [Fact]
    public void Median_of_odd_and_even_counts()
    {
        Assert.Equal(5.0, Stats.Median([9, 1, 5]));
        Assert.Equal(11.5, Stats.Median([12, 7, 95, 9, 14, 11]));
    }

    [Fact]
    public void Quantiles_are_R7()
    {
        // NumPy: np.quantile(np.arange(1, 11), [0, .1, .25, .5, .9, 1]) = [1, 1.9, 3.25, 5.5, 9.1, 10]
        double[] data = [10, 1, 9, 2, 8, 3, 7, 4, 6, 5];
        var got = Stats.Quantiles(data, [0, 0.1, 0.25, 0.5, 0.9, 1]);
        double[] want = [1, 1.9, 3.25, 5.5, 9.1, 10];
        for (var i = 0; i < want.Length; i++)
            Assert.Equal(want[i], got[i], 12);
        Assert.Equal(Stats.Median(data), Stats.Quantile(data, 0.5), 12);
    }

    [Fact]
    public void Empty_sample_is_refused()
    {
        var e = Assert.Throws<NumflowException>(() => Stats.Median([]));
        Assert.Equal(("empty_input", "data"), (e.ErrorCode, e.Parameter));
    }

    [Fact]
    public void A_value_that_is_not_finite_is_refused_with_its_index()
    {
        var e = Assert.Throws<NumflowException>(() => Stats.Quantiles([1, double.NaN], [0.5]));
        Assert.Equal(("value_not_finite", "data", 1), (e.ErrorCode, e.Parameter, e.Index));
    }

    [Fact]
    public void A_probability_outside_0_1_is_refused_at_its_place()
    {
        var e = Assert.Throws<NumflowException>(() => Stats.Quantiles([1, 2], [0.5, 1.5]));
        Assert.Equal(("parameter_out_of_range", "probabilities[1]"), (e.ErrorCode, e.Parameter));
    }
}
