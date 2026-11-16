using UNumflow;
using Xunit;

namespace UNumflow.Tests;

public class DistributionTests
{
    /// <summary>Past 8σ the CDF is 1 to a double: the upper tail has to be formed directly.</summary>
    [Fact]
    public void Survival_keeps_its_digits_where_one_minus_the_cdf_is_zero()
    {
        var normal = Distribution.Normal(0, 1);
        Assert.Equal(0.0, 1.0 - normal.Cdf(10));
        Assert.Equal(7.619853024160527e-24, normal.Survival(10), 1e-35);

        // k = 2 is an exponential: S(x) = exp(-x/2).
        Assert.Equal(Math.Exp(-100), Distribution.ChiSquared(2).Survival(200), 1e-45);
    }

    [Fact]
    public void Survival_is_one_and_zero_at_the_infinities()
    {
        var d = Distribution.Gamma(2, 1);
        Assert.Equal(1.0, d.Survival(double.NegativeInfinity));
        Assert.Equal(0.0, d.Survival(double.PositiveInfinity));
        Assert.Throws<NumflowException>(() => d.Survival(double.NaN));
    }
}
