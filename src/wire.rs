//! The shapes the language bindings exchange: a distribution named by `kind`
//! with its parameters, built into the crate's own distribution types.
//!
//! Kept apart from any one binding so every transport reads the same contract.

#![cfg(feature = "wasm")]

use serde::Deserialize;

use crate::distributions::{
    BetaDistribution, ChiSquared, DistributionError, Exponential, GammaDistribution, LogNormal,
    Normal, Pert, Sample, Triangular, Uniform, Weibull,
};

/// The `kind` names a [`DistributionSpec`] accepts, in declaration order.
pub(crate) const KINDS: &[&str] = &[
    "uniform",
    "triangular",
    "pert",
    "normal",
    "lognormal",
    "weibull",
    "exponential",
    "gamma",
    "beta",
    "chi_squared",
];

/// A distribution named by `kind`, with its parameters.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum DistributionSpec {
    /// Continuous uniform on `[min, max]`.
    Uniform { min: f64, max: f64 },
    /// Triangular on `[min, max]` peaking at `mode`.
    Triangular { min: f64, mode: f64, max: f64 },
    /// PERT (scaled Beta) on `[min, max]`; `lambda` weights the mode (default 4).
    Pert {
        min: f64,
        mode: f64,
        max: f64,
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional, type = "number"))]
        lambda: Option<f64>,
    },
    /// Normal with mean `mu` and standard deviation `sigma`.
    Normal { mu: f64, sigma: f64 },
    /// Log-normal: `ln X ~ Normal(mu, sigma)`.
    Lognormal { mu: f64, sigma: f64 },
    /// Weibull with `shape` (β) and `scale` (η).
    Weibull { shape: f64, scale: f64 },
    /// Exponential with `rate` (λ).
    Exponential { rate: f64 },
    /// Gamma with `shape` (α) and `rate` (β).
    Gamma { shape: f64, rate: f64 },
    /// Beta on `[0, 1]` with shapes `alpha` and `beta`.
    Beta { alpha: f64, beta: f64 },
    /// Chi-squared with `k` degrees of freedom.
    ChiSquared { k: f64 },
}

/// A built distribution — one of the crate's types.
#[derive(Debug, Clone)]
pub(crate) enum Distribution {
    Uniform(Uniform),
    Triangular(Triangular),
    Pert(Pert),
    Normal(Normal),
    LogNormal(LogNormal),
    Weibull(Weibull),
    Exponential(Exponential),
    Gamma(GammaDistribution),
    Beta(BetaDistribution),
    ChiSquared(ChiSquared),
}

impl DistributionSpec {
    /// The distribution, or the parameter its constructor refused.
    pub(crate) fn build(&self) -> Result<Distribution, DistributionError> {
        use DistributionSpec as S;
        Ok(match *self {
            S::Uniform { min, max } => Distribution::Uniform(Uniform::new(min, max)?),
            S::Triangular { min, mode, max } => {
                Distribution::Triangular(Triangular::new(min, mode, max)?)
            }
            S::Pert {
                min,
                mode,
                max,
                lambda,
            } => Distribution::Pert(Pert::with_shape(min, mode, max, lambda.unwrap_or(4.0))?),
            S::Normal { mu, sigma } => Distribution::Normal(Normal::new(mu, sigma)?),
            S::Lognormal { mu, sigma } => Distribution::LogNormal(LogNormal::new(mu, sigma)?),
            S::Weibull { shape, scale } => Distribution::Weibull(Weibull::new(shape, scale)?),
            S::Exponential { rate } => Distribution::Exponential(Exponential::new(rate)?),
            S::Gamma { shape, rate } => Distribution::Gamma(GammaDistribution::new(shape, rate)?),
            S::Beta { alpha, beta } => Distribution::Beta(BetaDistribution::new(alpha, beta)?),
            S::ChiSquared { k } => Distribution::ChiSquared(ChiSquared::new(k)?),
        })
    }
}

macro_rules! each {
    ($self:ident, $d:ident => $e:expr) => {
        match $self {
            Distribution::Uniform($d) => $e,
            Distribution::Triangular($d) => $e,
            Distribution::Pert($d) => $e,
            Distribution::Normal($d) => $e,
            Distribution::LogNormal($d) => $e,
            Distribution::Weibull($d) => $e,
            Distribution::Exponential($d) => $e,
            Distribution::Gamma($d) => $e,
            Distribution::Beta($d) => $e,
            Distribution::ChiSquared($d) => $e,
        }
    };
}

impl Distribution {
    /// `P(X ≤ x)`.
    pub(crate) fn cdf(&self, x: f64) -> f64 {
        each!(self, d => d.cdf(x))
    }

    /// The `x` with `P(X ≤ x) = p`, for `p` in `(0, 1)`.
    pub(crate) fn quantile(&self, p: f64) -> f64 {
        each!(self, d => d.quantile(p))
            .expect("every distribution's quantile is defined on the open unit interval")
    }

    /// `n` variates from `seed` (see [`Sample::sample_n`]).
    pub(crate) fn sample_n(&self, n: usize, seed: u64) -> Vec<f64> {
        each!(self, d => d.sample_n(n, seed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_lists_every_variant_by_its_serde_name() {
        for kind in KINDS {
            let json = match *kind {
                "exponential" => r#"{"rate": 1}"#.to_string(),
                "chi_squared" => r#"{"k": 1}"#.to_string(),
                _ => String::from("{}"),
            };
            let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
            value["kind"] = serde_json::Value::String(kind.to_string());
            // A known kind fails, if at all, on a missing field — never on the tag.
            if let Err(e) = serde_json::from_value::<DistributionSpec>(value) {
                assert!(e.to_string().contains("missing field"), "{kind}: {e}");
            }
        }
    }

    #[test]
    fn unknown_fields_are_refused() {
        let v = serde_json::json!({ "kind": "normal", "mu": 0, "sigma": 1, "sd": 2 });
        assert!(serde_json::from_value::<DistributionSpec>(v).is_err());
    }

    #[test]
    fn a_spec_builds_its_distribution() {
        let s: DistributionSpec =
            serde_json::from_value(serde_json::json!({ "kind": "chi_squared", "k": 4 })).unwrap();
        let d = s.build().unwrap();
        assert!((d.quantile(0.95) - 9.487_729).abs() < 1e-5);
        assert!((d.cdf(9.487_729) - 0.95).abs() < 1e-6);
        let p: DistributionSpec = serde_json::from_value(
            serde_json::json!({ "kind": "pert", "min": 0, "mode": 1, "max": 4 }),
        )
        .unwrap();
        assert_eq!(p.build().unwrap().sample_n(3, 9).len(), 3);
        let bad = DistributionSpec::Weibull {
            shape: 0.0,
            scale: 1.0,
        };
        assert_eq!(
            bad.build().unwrap_err(),
            DistributionError::NotPositive {
                parameter: "shape",
                got: 0.0
            }
        );
    }
}
