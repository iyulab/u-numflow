//! Random variates from the continuous distributions.
//!
//! # Methods
//!
//! - **Inverse transform** — `x = F⁻¹(u)` — wherever the quantile is closed
//!   form or a direct formula: [`Uniform`], [`Triangular`], [`Normal`],
//!   [`LogNormal`], [`Weibull`], [`Exponential`]. A larger `u` always gives a
//!   larger `x`, so two runs on one seed are coupled point by point (common
//!   random numbers).
//! - **Marsaglia–Tsang** squeeze for the gamma family, whose quantile is a
//!   Newton iteration costing dozens of incomplete-gamma evaluations per
//!   draw: [`GammaDistribution`], [`ChiSquared`] (Gamma(k/2, rate ½)),
//!   [`BetaDistribution`] (X/(X+Y) with X ~ Gamma(α), Y ~ Gamma(β)) and
//!   [`Pert`] (a scaled Beta).
//!
//! Every uniform is drawn from the **open** interval (0, 1) — 52 random bits
//! offset by half a step — so `F⁻¹(0)` and `ln 0` cannot occur and no variate
//! is infinite.
//!
//! # References
//!
//! - Devroye (1986), *Non-Uniform Random Variate Generation*, §II.2
//!   (inversion), §IX.4 (beta from gammas).
//! - Marsaglia & Tsang (2000), "A simple method for generating gamma
//!   variables", *ACM TOMS* 26(3), 363–372 — including the `U^(1/a)` boost
//!   for shape `a < 1`.

use rand::Rng;

use super::{
    BetaDistribution, ChiSquared, Exponential, GammaDistribution, LogNormal, Normal, Pert,
    Triangular, Uniform, Weibull,
};
use crate::random::create_rng;

/// A distribution that can draw random variates.
///
/// # Examples
/// ```
/// use u_numflow::distributions::{Sample, Weibull};
///
/// let w = Weibull::new(2.0, 100.0).unwrap();
/// let a = w.sample_n(1000, 42);
/// assert_eq!(a, w.sample_n(1000, 42)); // same seed, same draws
/// assert!(a.iter().all(|&t| t > 0.0 && t.is_finite()));
/// ```
pub trait Sample {
    /// One variate, drawn with `rng`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64;

    /// `n` variates from a generator seeded with `seed` ([`create_rng`]): the
    /// same `(n, seed)` always gives the same values.
    fn sample_n(&self, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = create_rng(seed);
        (0..n).map(|_| self.sample(&mut rng)).collect()
    }
}

/// A uniform variate on the open interval (0, 1).
///
/// The top 52 bits of a `u64` give `k ∈ [0, 2⁵²)`; `(k + ½)·2⁻⁵²` lies in
/// `[2⁻⁵³, 1 − 2⁻⁵³]` and is exact in `f64`, so it is never 0 or 1. (With 53
/// bits the top value would be `1 − 2⁻⁵⁴`, which `f64` rounds to 1.)
pub(crate) fn open_unit<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    const STEP: f64 = 1.0 / (1u64 << 52) as f64;
    ((rng.next_u64() >> 12) as f64 + 0.5) * STEP
}

/// Standard normal by inversion of an open uniform.
fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    crate::special::inverse_normal_cdf(open_unit(rng))
}

/// Gamma(shape, rate 1) by Marsaglia & Tsang (2000).
fn standard_gamma<R: Rng + ?Sized>(shape: f64, rng: &mut R) -> f64 {
    if shape < 1.0 {
        // Gamma(a) = Gamma(a + 1) · U^(1/a)  (Marsaglia & Tsang §4).
        return standard_gamma(shape + 1.0, rng) * open_unit(rng).powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = open_unit(rng);
        let x2 = x * x;
        // Squeeze, then the exact log test.
        if u < 1.0 - 0.0331 * x2 * x2 || u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

macro_rules! by_inversion {
    ($($t:ty),*) => {$(
        impl Sample for $t {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
                self.quantile(open_unit(rng))
                    .expect("the quantile is defined on the open unit interval")
            }
        }
    )*};
}

by_inversion!(Uniform, Triangular, Normal, LogNormal, Weibull, Exponential);

impl Sample for GammaDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        standard_gamma(self.shape(), rng) / self.rate()
    }
}

impl Sample for ChiSquared {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        2.0 * standard_gamma(self.k / 2.0, rng)
    }
}

/// `X / (X + Y)` with `X ~ Gamma(α)`, `Y ~ Gamma(β)`.
fn beta_variate<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> f64 {
    let x = standard_gamma(alpha, rng);
    let y = standard_gamma(beta, rng);
    x / (x + y)
}

impl Sample for BetaDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        beta_variate(self.alpha, self.beta, rng)
    }
}

impl Sample for Pert {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.min() + (self.max() - self.min()) * beta_variate(self.alpha(), self.beta_param(), rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kolmogorov–Smirnov distance between `xs` and `cdf`.
    fn ks(mut xs: Vec<f64>, cdf: impl Fn(f64) -> f64) -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).expect("finite variates"));
        let n = xs.len() as f64;
        xs.iter()
            .enumerate()
            .map(|(i, &x)| {
                let f = cdf(x);
                (f - i as f64 / n).abs().max(((i + 1) as f64 / n - f).abs())
            })
            .fold(0.0, f64::max)
    }

    /// 20 000 draws; the 0.1 % critical value of D is 1.95/√n ≈ 0.0138.
    const N: usize = 20_000;
    const D_CRIT: f64 = 0.0138;

    fn check<D: Sample>(d: &D, cdf: impl Fn(f64) -> f64, name: &str) {
        let xs = d.sample_n(N, 7);
        assert!(
            xs.iter().all(|x| x.is_finite()),
            "{name}: non-finite variate"
        );
        let dist = ks(xs, cdf);
        assert!(dist < D_CRIT, "{name}: KS distance {dist} >= {D_CRIT}");
    }

    #[test]
    fn every_distribution_matches_its_cdf() {
        let u = Uniform::new(-2.0, 5.0).unwrap();
        check(&u, |x| u.cdf(x), "uniform");
        let t = Triangular::new(0.0, 1.0, 4.0).unwrap();
        check(&t, |x| t.cdf(x), "triangular");
        let n = Normal::new(3.0, 2.0).unwrap();
        check(&n, |x| n.cdf(x), "normal");
        let ln = LogNormal::new(0.5, 0.8).unwrap();
        check(&ln, |x| ln.cdf(x), "lognormal");
        let w = Weibull::new(1.7, 40.0).unwrap();
        check(&w, |x| w.cdf(x), "weibull");
        let e = Exponential::new(0.25).unwrap();
        check(&e, |x| e.cdf(x), "exponential");
        for shape in [0.3, 1.0, 2.5, 30.0] {
            let g = GammaDistribution::new(shape, 2.0).unwrap();
            check(&g, |x| g.cdf(x), &format!("gamma({shape})"));
        }
        for k in [1.0, 3.0, 17.0] {
            let c = ChiSquared::new(k).unwrap();
            check(&c, |x| c.cdf(x), &format!("chi2({k})"));
        }
        for (a, b) in [(0.5, 0.5), (2.0, 5.0), (8.0, 1.5)] {
            let bd = BetaDistribution::new(a, b).unwrap();
            check(&bd, |x| bd.cdf(x), &format!("beta({a},{b})"));
        }
        let p = Pert::new(2.0, 3.0, 10.0).unwrap();
        check(&p, |x| p.cdf(x), "pert");
    }

    #[test]
    fn same_seed_same_draws_different_seed_different_draws() {
        let g = GammaDistribution::new(2.0, 1.0).unwrap();
        assert_eq!(g.sample_n(50, 1), g.sample_n(50, 1));
        assert_ne!(g.sample_n(50, 1), g.sample_n(50, 2));
    }

    #[test]
    fn inversion_is_monotone_in_the_uniform() {
        // Common random numbers: a larger scale never lowers any draw.
        let small = Weibull::new(2.0, 10.0).unwrap().sample_n(200, 5);
        let large = Weibull::new(2.0, 20.0).unwrap().sample_n(200, 5);
        assert!(small.iter().zip(&large).all(|(a, b)| a < b));
    }

    #[test]
    fn the_open_uniform_never_reaches_its_ends() {
        struct Fixed(u64);
        impl rand::TryRng for Fixed {
            type Error = std::convert::Infallible;
            fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
                Ok(self.0 as u32)
            }
            fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
                Ok(self.0)
            }
            fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
                dst.fill(0);
                Ok(())
            }
        }
        let lo = open_unit(&mut Fixed(0));
        let hi = open_unit(&mut Fixed(u64::MAX));
        assert!(lo > 0.0 && hi < 1.0);
        assert!(Exponential::new(1.0)
            .unwrap()
            .quantile(hi)
            .unwrap()
            .is_finite());
    }
}
