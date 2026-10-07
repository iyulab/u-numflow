//! The shapes the language bindings exchange: a distribution named by `kind`
//! with its parameters, built into the crate's own distribution types.
//!
//! Kept apart from any one binding so every transport reads the same contract.

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

// ── Refusals ──────────────────────────────────────────────────────────────

/// A value carried on a refusal.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Field {
    Num(f64),
    Str(&'static str),
    /// A path built at run time, such as `a[3]`.
    Path(String),
    Null,
}

/// A refusal on its way to a caller: readable text, a stable `code`, and the
/// values behind it. WebAssembly throws it as an `Error` with those properties;
/// the C ABI writes it as `{"error": text, "code": …, …fields}`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Refusal {
    pub(crate) code: &'static str,
    pub(crate) message: String,
    pub(crate) fields: Vec<(&'static str, Field)>,
}

impl Refusal {
    /// An argument outside the range the function accepts; `max` is `None`
    /// when the range is open above.
    pub(crate) fn out_of_range(
        parameter: &'static str,
        min: f64,
        max: Option<f64>,
        got: f64,
        message: String,
    ) -> Self {
        Refusal {
            code: "parameter_out_of_range",
            message,
            fields: vec![
                ("parameter", Field::Str(parameter)),
                ("min", Field::Num(min)),
                ("max", max.map_or(Field::Null, Field::Num)),
                ("got", Field::Num(got)),
            ],
        }
    }
}

/// `p` strictly inside `(0, 1)`: `parameter_out_of_range` with `min` 0 and
/// `max` 1 -- both excluded, as the message says.
pub(crate) fn check_probability(p: f64) -> Result<(), Refusal> {
    if p.is_finite() && p > 0.0 && p < 1.0 {
        Ok(())
    } else {
        Err(Refusal::out_of_range(
            "p",
            0.0,
            Some(1.0),
            p,
            format!("p must be strictly between 0 and 1, got {p}"),
        ))
    }
}

/// A number that is not NaN (±∞ is a valid argument to a CDF).
pub(crate) fn check_finite(name: &'static str, x: f64) -> Result<(), Refusal> {
    if x.is_nan() {
        Err(Refusal {
            code: "value_not_finite",
            message: format!("{name} must be a number, got NaN"),
            fields: vec![("parameter", Field::Str(name))],
        })
    } else {
        Ok(())
    }
}

/// A whole number in `[0, max]`, refused as `parameter_out_of_range` otherwise.
pub(crate) fn whole(parameter: &'static str, x: f64, max: f64) -> Result<f64, Refusal> {
    if x.is_finite() && x >= 0.0 && x <= max && x.fract() == 0.0 {
        Ok(x)
    } else {
        Err(Refusal::out_of_range(
            parameter,
            0.0,
            Some(max),
            x,
            format!("{parameter} must be a whole number in [0, {max}], got {x}"),
        ))
    }
}

/// The interval set given by `rows` (each `[start, end]`, already read as
/// finite numbers): a row that is not a pair is `malformed_input` at
/// `parameter[i]`, a reversed pair `reversed_interval` with `parameter` and
/// `index`.
pub(crate) fn interval_set_from(
    parameter: &'static str,
    rows: Vec<Vec<f64>>,
) -> Result<crate::collections::IntervalSet<f64>, Refusal> {
    let mut pairs = Vec::with_capacity(rows.len());
    for (i, row) in rows.into_iter().enumerate() {
        match row[..] {
            [start, end] => pairs.push((start, end)),
            _ => {
                return Err(Refusal {
                    code: "malformed_input",
                    message: format!(
                        "{parameter}[{i}]: expected [start, end], got {} numbers",
                        row.len()
                    ),
                    fields: vec![("parameter", Field::Path(format!("{parameter}[{i}]")))],
                })
            }
        }
    }
    crate::collections::IntervalSet::from_intervals(pairs).map_err(|e| match e {
        crate::collections::IntervalError::Reversed { index } => Refusal {
            code: "reversed_interval",
            message: format!("{parameter}[{index}]: start is after end"),
            fields: vec![
                ("parameter", Field::Str(parameter)),
                ("index", Field::Num(index as f64)),
            ],
        },
        // Unreachable: every number was checked finite while reading.
        crate::collections::IntervalError::NotFinite { index } => Refusal {
            code: "value_not_finite",
            message: format!("{parameter}[{index}]: a bound is not finite"),
            fields: vec![
                ("parameter", Field::Str(parameter)),
                ("index", Field::Num(index as f64)),
            ],
        },
    })
}

/// `kind` names a distribution this crate has, or `unknown_option` listing them.
pub(crate) fn check_kind(kind: &str) -> Result<(), Refusal> {
    if KINDS.contains(&kind) {
        return Ok(());
    }
    Err(Refusal {
        code: "unknown_option",
        message: format!(
            "distribution.kind: unknown distribution \"{kind}\"; expected one of {}",
            KINDS.join(", ")
        ),
        fields: vec![
            ("parameter", Field::Str("distribution.kind")),
            ("got", Field::Path(kind.to_string())),
            ("expected", Field::Path(KINDS.join(", "))),
        ],
    })
}

/// A distribution constructor's refusal, with the parameter's path
/// (`distribution.sigma`).
pub(crate) fn distribution_refusal(e: DistributionError) -> Refusal {
    const P: &str = "distribution";
    let message = format!("{P}: {e}");
    match e {
        DistributionError::NotFinite { parameter } => Refusal {
            code: "value_not_finite",
            message,
            fields: vec![("parameter", Field::Path(format!("{P}.{parameter}")))],
        },
        DistributionError::NotPositive { parameter, got } => Refusal {
            code: "parameter_out_of_range",
            message,
            fields: vec![
                ("parameter", Field::Path(format!("{P}.{parameter}"))),
                ("min", Field::Num(0.0)),
                ("max", Field::Null),
                ("got", Field::Num(got)),
            ],
        },
        DistributionError::Unordered { parameter, .. } => Refusal {
            code: "invalid_option",
            message,
            fields: vec![("parameter", Field::Path(format!("{P}.{parameter}")))],
        },
    }
}

// ── Distributions ─────────────────────────────────────────────────────────

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
    /// The mean.
    pub(crate) fn mean(&self) -> f64 {
        each!(self, d => d.mean())
    }

    /// The variance.
    pub(crate) fn variance(&self) -> f64 {
        each!(self, d => d.variance())
    }

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
