//! The shapes the language bindings exchange: a distribution named by `kind`
//! with its parameters, built into the crate's own distribution types.
//!
//! Kept apart from any one binding so every transport reads the same contract.

use serde::Deserialize;

use crate::distributions::{
    BetaDistribution, ChiSquared, DistributionError, Exponential, GammaDistribution, LogNormal,
    Normal, Pert, Sample, Triangular, Uniform, Weibull,
};
use crate::units::{Unit, UnitError, Units};

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
    /// A closed set of names, such as the ones an option takes.
    List(&'static [&'static str]),
    Null,
}

/// Every reason a function refuses with: the stable `code` a refusal carries,
/// and the fields that can stand beside it. A [`Refusal`] holds one of these,
/// so the published list ([`Code::ALL`]) is the list the crate refuses from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Code {
    MalformedInput,
    ValueNotFinite,
    ParameterOutOfRange,
    UnknownOption,
    ReversedInterval,
    EmptyInput,
    InsufficientData,
    NonPositiveData,
    InvalidTransform,
    InvalidInverse,
    UnknownUnit,
    DimensionMismatch,
    InvalidOption,
    DuplicateId,
    Internal,
}

/// A refusal reason: the `code` a refusal carries (the WebAssembly `Error`'s,
/// the C error body's) and the fields that can stand beside it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[serde(deny_unknown_fields)]
pub(crate) struct RefusalCode {
    pub(crate) code: String,
    pub(crate) fields: Vec<String>,
}

/// A violation a checker reports. This package has no checker; the type is
/// here so every engine's [`Codes`] has one shape.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[serde(deny_unknown_fields)]
pub(crate) struct ViolationCode {
    pub(crate) code: String,
    pub(crate) fields: Vec<String>,
    pub(crate) checkers: Vec<String>,
}

/// What this package can answer with besides a result: every refusal
/// `code`, and no violations -- it has no checker.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[serde(deny_unknown_fields)]
pub(crate) struct Codes {
    pub(crate) refusals: Vec<RefusalCode>,
    pub(crate) violations: Vec<ViolationCode>,
}

pub(crate) fn code_catalog() -> Codes {
    Codes {
        refusals: Code::ALL
            .into_iter()
            .map(|c| RefusalCode {
                code: c.name().to_string(),
                fields: c.fields().iter().map(|f| f.to_string()).collect(),
            })
            .collect(),
        violations: vec![],
    }
}

impl Code {
    pub(crate) const ALL: [Code; 15] = [
        Code::MalformedInput,
        Code::ValueNotFinite,
        Code::ParameterOutOfRange,
        Code::UnknownOption,
        Code::ReversedInterval,
        Code::EmptyInput,
        Code::InsufficientData,
        Code::NonPositiveData,
        Code::InvalidTransform,
        Code::InvalidInverse,
        Code::UnknownUnit,
        Code::DimensionMismatch,
        Code::InvalidOption,
        Code::DuplicateId,
        Code::Internal,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Code::MalformedInput => "malformed_input",
            Code::ValueNotFinite => "value_not_finite",
            Code::ParameterOutOfRange => "parameter_out_of_range",
            Code::UnknownOption => "unknown_option",
            Code::ReversedInterval => "reversed_interval",
            Code::EmptyInput => "empty_input",
            Code::InsufficientData => "insufficient_data",
            Code::NonPositiveData => "non_positive_data",
            Code::InvalidTransform => "invalid_transform",
            Code::InvalidInverse => "invalid_inverse",
            Code::UnknownUnit => "unknown_unit",
            Code::DimensionMismatch => "dimension_mismatch",
            Code::InvalidOption => "invalid_option",
            Code::DuplicateId => "duplicate_id",
            Code::Internal => "internal",
        }
    }

    /// The fields a refusal with this code can carry besides `code`.
    pub(crate) fn fields(self) -> &'static [&'static str] {
        match self {
            Code::MalformedInput => &["parameter", "index"],
            Code::ValueNotFinite => &["parameter", "index"],
            Code::ParameterOutOfRange => &[
                "parameter",
                "min",
                "max",
                "exclusive_min",
                "exclusive_max",
                "got",
            ],
            Code::UnknownOption => &["parameter", "got", "expected"],
            Code::ReversedInterval => &["parameter", "index"],
            Code::EmptyInput => &["parameter"],
            Code::InsufficientData => &["parameter", "min", "got"],
            Code::NonPositiveData => &["parameter", "index", "got"],
            Code::InvalidTransform => &[],
            Code::InvalidInverse => &[],
            Code::UnknownUnit => &["parameter", "got"],
            Code::DimensionMismatch => &["parameter", "expected", "got"],
            Code::InvalidOption => &["parameter"],
            Code::DuplicateId => &["parameter", "id", "index"],
            Code::Internal => &[],
        }
    }
}

impl PartialEq<&str> for Code {
    fn eq(&self, other: &&str) -> bool {
        self.name() == *other
    }
}

/// A refusal on its way to a caller: readable text, a stable `code`, and the
/// values behind it. WebAssembly throws it as an `Error` with those properties;
/// the C ABI writes it as `{"error": text, "code": …, …fields}`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Refusal {
    pub(crate) code: Code,
    pub(crate) message: String,
    pub(crate) fields: Vec<(&'static str, Field)>,
}

impl Refusal {
    /// Whether every field is one its code declares -- checked where a refusal
    /// leaves the crate, so a field missing from [`Code::fields`] fails a test.
    pub(crate) fn fields_declared(&self) -> bool {
        self.fields
            .iter()
            .all(|(k, _)| self.code.fields().contains(k))
    }
}

/// One end of the range an argument must fall in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Bound {
    /// The end value is accepted: reported as `min` / `max`.
    Closed(f64),
    /// The end value itself is refused: reported as `exclusive_min` /
    /// `exclusive_max`, with `min` / `max` `null`.
    Open(f64),
    /// No limit on this side: `min` / `max` `null`.
    Unbounded,
}

impl From<std::ops::Bound<f64>> for Bound {
    fn from(end: std::ops::Bound<f64>) -> Self {
        match end {
            std::ops::Bound::Included(v) => Bound::Closed(v),
            std::ops::Bound::Excluded(v) => Bound::Open(v),
            std::ops::Bound::Unbounded => Bound::Unbounded,
        }
    }
}

impl Bound {
    /// The `min`/`max` value and, for an open end, the `exclusive_*` value.
    fn split(self) -> (Field, Option<f64>) {
        match self {
            Bound::Closed(v) => (Field::Num(v), None),
            Bound::Open(v) => (Field::Null, Some(v)),
            Bound::Unbounded => (Field::Null, None),
        }
    }
}

impl Refusal {
    /// An argument outside the range the function accepts. `min` and `max`
    /// are always present (`null` unless that end is closed); an open end
    /// adds `exclusive_min` / `exclusive_max`.
    pub(crate) fn out_of_range(
        parameter: Field,
        low: Bound,
        high: Bound,
        got: f64,
        message: String,
    ) -> Self {
        let (min, exclusive_min) = low.split();
        let (max, exclusive_max) = high.split();
        let mut fields = vec![("parameter", parameter), ("min", min)];
        if let Some(v) = exclusive_min {
            fields.push(("exclusive_min", Field::Num(v)));
        }
        fields.push(("max", max));
        if let Some(v) = exclusive_max {
            fields.push(("exclusive_max", Field::Num(v)));
        }
        fields.push(("got", Field::Num(got)));
        Refusal {
            code: Code::ParameterOutOfRange,
            message,
            fields,
        }
    }
}

/// `p` strictly inside `(0, 1)`: `parameter_out_of_range` with
/// `exclusive_min` 0 and `exclusive_max` 1, as the message says.
pub(crate) fn check_probability(p: f64) -> Result<(), Refusal> {
    if p.is_finite() && p > 0.0 && p < 1.0 {
        Ok(())
    } else {
        Err(Refusal::out_of_range(
            Field::Str("p"),
            Bound::Open(0.0),
            Bound::Open(1.0),
            p,
            format!("p must be strictly between 0 and 1, got {p}"),
        ))
    }
}

/// A number that is not NaN (±∞ is a valid argument to a CDF).
pub(crate) fn check_finite(name: &'static str, x: f64) -> Result<(), Refusal> {
    if x.is_nan() {
        Err(Refusal {
            code: Code::ValueNotFinite,
            message: format!("{name} must be a number, got NaN"),
            fields: vec![("parameter", Field::Str(name))],
        })
    } else {
        Ok(())
    }
}

// ── Sample statistics ─────────────────────────────────────────────────────

/// `data` sorted, refusing an empty sample (`empty_input`) and a value that is
/// not finite (`value_not_finite` with its `index`) at `parameter`.
fn sorted_sample(parameter: &'static str, mut data: Vec<f64>) -> Result<Vec<f64>, Refusal> {
    if data.is_empty() {
        return Err(Refusal {
            code: Code::EmptyInput,
            message: format!("{parameter}: a sample needs at least one value"),
            fields: vec![("parameter", Field::Str(parameter))],
        });
    }
    if let Some(i) = data.iter().position(|x| !x.is_finite()) {
        return Err(Refusal {
            code: Code::ValueNotFinite,
            message: format!(
                "{parameter}[{i}]: expected a finite number, got {}",
                data[i]
            ),
            fields: vec![
                ("parameter", Field::Str(parameter)),
                ("index", Field::Num(i as f64)),
            ],
        });
    }
    data.sort_unstable_by(f64::total_cmp);
    Ok(data)
}

/// The median of `data` (even length: the mean of the middle two).
pub(crate) fn sample_median(data: Vec<f64>) -> Result<f64, Refusal> {
    let sorted = sorted_sample("data", data)?;
    Ok(crate::stats::quantile_sorted(&sorted, 0.5).expect("a non-empty sample has a median"))
}

/// The R-7 quantiles of `data` at each of `probabilities` (each in `[0, 1]`),
/// sorting the sample once. A probability that is not finite is
/// `value_not_finite` at `probabilities` with its `index`; one outside
/// `[0, 1]` is `parameter_out_of_range` at `probabilities[i]`.
pub(crate) fn sample_quantiles(data: Vec<f64>, probabilities: &[f64]) -> Result<Vec<f64>, Refusal> {
    for (i, &p) in probabilities.iter().enumerate() {
        if !p.is_finite() {
            return Err(Refusal {
                code: Code::ValueNotFinite,
                message: format!("probabilities[{i}]: expected a finite number, got {p}"),
                fields: vec![
                    ("parameter", Field::Str("probabilities")),
                    ("index", Field::Num(i as f64)),
                ],
            });
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(Refusal::out_of_range(
                Field::Path(format!("probabilities[{i}]")),
                Bound::Closed(0.0),
                Bound::Closed(1.0),
                p,
                format!("probabilities[{i}] must be in [0, 1], got {p}"),
            ));
        }
    }
    let sorted = sorted_sample("data", data)?;
    Ok(probabilities
        .iter()
        .map(|&p| crate::stats::quantile_sorted(&sorted, p).expect("p checked to lie in [0, 1]"))
        .collect())
}

/// A whole number in `[0, max]`.
///
/// Whether the value is a whole number is a matter of its type, whether it is
/// in range a matter of its value: NaN or ±∞ is `value_not_finite`, a fraction
/// `malformed_input`, and a whole number outside `[0, max]` (a negative one
/// included) `parameter_out_of_range` with `min` 0 and `max`.
pub(crate) fn whole(parameter: &'static str, x: f64, max: f64) -> Result<f64, Refusal> {
    if !x.is_finite() {
        return Err(Refusal {
            code: Code::ValueNotFinite,
            message: format!("{parameter} must be a finite number, got {x}"),
            fields: vec![("parameter", Field::Str(parameter))],
        });
    }
    if x.fract() != 0.0 {
        return Err(Refusal {
            code: Code::MalformedInput,
            message: format!("{parameter} must be a whole number, got {x}"),
            fields: vec![("parameter", Field::Str(parameter))],
        });
    }
    if (0.0..=max).contains(&x) {
        Ok(x)
    } else {
        Err(Refusal::out_of_range(
            Field::Str(parameter),
            Bound::Closed(0.0),
            Bound::Closed(max),
            x,
            format!("{parameter} must be in [0, {max}], got {x}"),
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
                    code: Code::MalformedInput,
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
            code: Code::ReversedInterval,
            message: format!("{parameter}[{index}]: start is after end"),
            fields: vec![
                ("parameter", Field::Str(parameter)),
                ("index", Field::Num(index as f64)),
            ],
        },
        // Unreachable: every number was checked finite while reading.
        crate::collections::IntervalError::NotFinite { index } => Refusal {
            code: Code::ValueNotFinite,
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
        code: Code::UnknownOption,
        message: format!(
            "distribution.kind: unknown distribution \"{kind}\"; expected one of {}",
            KINDS.join(", ")
        ),
        fields: vec![
            ("parameter", Field::Str("distribution.kind")),
            ("got", Field::Path(kind.to_string())),
            ("expected", Field::List(KINDS)),
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
            code: Code::ValueNotFinite,
            message,
            fields: vec![("parameter", Field::Path(format!("{P}.{parameter}")))],
        },
        DistributionError::NotPositive { parameter, got } => Refusal::out_of_range(
            Field::Path(format!("{P}.{parameter}")),
            Bound::Open(0.0),
            Bound::Unbounded,
            got,
            message,
        ),
        // `max` above `min`, `mode` within `[min, max]`: a range whose ends
        // are the other parameters' values.
        DistributionError::Unordered {
            parameter,
            low,
            high,
            got,
            ..
        } => Refusal::out_of_range(
            Field::Path(format!("{P}.{parameter}")),
            Bound::from(low),
            Bound::from(high),
            got,
            message,
        ),
    }
}

// ── Units ─────────────────────────────────────────────────────────────────

fn one() -> f64 {
    1.0
}

/// A unit the caller defines: `symbol` = `factor` × `unit` (+ `offset` SI
/// units at its zero, for a scale such as a Fahrenheit degree).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(test, derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct UnitDefinitionSpec {
    /// The new symbol, e.g. `cal_IT`.
    pub(crate) symbol: String,
    /// What it is a multiple of, as a unit expression, e.g. `J`.
    pub(crate) unit: String,
    /// How many `unit` one `symbol` is (default 1, `> 0`).
    #[serde(default = "one")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    pub(crate) factor: f64,
    /// SI value at the symbol's zero (default 0).
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    pub(crate) offset: f64,
}

/// What a unit expression stands for.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(test, derive(schemars::JsonSchema))]
pub(crate) struct UnitInfo {
    /// The dimension in SI base units, e.g. `kg.m2.s-2` (`1` for a pure number).
    pub(crate) dimension: String,
    /// Exponents of m, kg, s, A, K, mol, cd.
    pub(crate) exponents: [i32; 7],
    /// Size of one unit in coherent SI units.
    pub(crate) scale: f64,
    /// SI value of the unit's zero (0 except for scales such as `Cel`).
    pub(crate) offset: f64,
}

impl From<Unit> for UnitInfo {
    fn from(u: Unit) -> Self {
        UnitInfo {
            dimension: u.dimension.to_string(),
            exponents: u.dimension.0,
            scale: u.scale,
            offset: u.offset,
        }
    }
}

/// The SI units plus `definitions`, each refused at its own path
/// (`definitions[2].unit`).
pub(crate) fn units_from(definitions: &[UnitDefinitionSpec]) -> Result<Units, Refusal> {
    let mut units = Units::si();
    for (i, d) in definitions.iter().enumerate() {
        units
            .define(&d.symbol, d.factor, &d.unit, d.offset)
            .map_err(|e| {
                let at = |key: &str| Field::Path(format!("definitions[{i}].{key}"));
                let place = match &e {
                    UnitError::NotFinite { parameter } => at(parameter),
                    UnitError::NonPositiveFactor { .. } => at("factor"),
                    UnitError::DuplicateSymbol { .. } | UnitError::InvalidSymbol { .. } => {
                        at("symbol")
                    }
                    _ => at("unit"),
                };
                let refusal = unit_refusal(e, place);
                match refusal.code {
                    Code::DuplicateId => Refusal {
                        fields: refusal
                            .fields
                            .into_iter()
                            .chain([("index", Field::Num(i as f64))])
                            .collect(),
                        ..refusal
                    },
                    _ => refusal,
                }
            })?;
    }
    Ok(units)
}

/// `expression` read against `units`, refused at `parameter`.
pub(crate) fn unit_at(
    units: &Units,
    expression: &str,
    parameter: &'static str,
) -> Result<Unit, Refusal> {
    units
        .parse(expression)
        .map_err(|e| unit_refusal(e, Field::Str(parameter)))
}

/// A unit error as a refusal at `parameter`.
pub(crate) fn unit_refusal(e: UnitError, parameter: Field) -> Refusal {
    let message = match &parameter {
        Field::Str(p) => format!("{p}: {e}"),
        Field::Path(p) => format!("{p}: {e}"),
        _ => e.to_string(),
    };
    let (code, fields) = match e {
        UnitError::Malformed { .. } | UnitError::InvalidSymbol { .. } => {
            (Code::MalformedInput, vec![("parameter", parameter)])
        }
        UnitError::UnknownSymbol { symbol, .. } => (
            Code::UnknownUnit,
            vec![("parameter", parameter), ("got", Field::Path(symbol))],
        ),
        UnitError::OffsetCombined { .. } => (Code::InvalidOption, vec![("parameter", parameter)]),
        UnitError::DimensionMismatch { from, to } => (
            Code::DimensionMismatch,
            vec![
                ("parameter", parameter),
                ("expected", Field::Path(from.to_string())),
                ("got", Field::Path(to.to_string())),
            ],
        ),
        UnitError::DuplicateSymbol { symbol } => (
            Code::DuplicateId,
            vec![("parameter", parameter), ("id", Field::Path(symbol))],
        ),
        UnitError::NotFinite { .. } => (Code::ValueNotFinite, vec![("parameter", parameter)]),
        UnitError::NonPositiveFactor { got } => {
            return Refusal::out_of_range(
                parameter,
                Bound::Open(0.0),
                Bound::Unbounded,
                got,
                message,
            )
        }
    };
    Refusal {
        code,
        message,
        fields,
    }
}

/// A conversion of `value` from `from` to `to`, a reading or (`difference`)
/// an amount of change, with each failure at its own parameter.
pub(crate) fn convert_units(
    value: f64,
    from: &str,
    to: &str,
    definitions: &[UnitDefinitionSpec],
    difference: bool,
) -> Result<f64, Refusal> {
    if !value.is_finite() {
        return Err(Refusal {
            code: Code::ValueNotFinite,
            message: format!("value must be a finite number, got {value}"),
            fields: vec![("parameter", Field::Str("value"))],
        });
    }
    let units = units_from(definitions)?;
    let (from, to) = (unit_at(&units, from, "from")?, unit_at(&units, to, "to")?);
    let converted = if difference {
        from.difference_to(value, &to)
    } else {
        from.convert_to(value, &to)
    };
    converted.map_err(|e| unit_refusal(e, Field::Str("to")))
}

// ── Distributions ─────────────────────────────────────────────────────────

/// A distribution named by `kind`, with its parameters.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(test, derive(schemars::JsonSchema))]
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

    #[test]
    fn a_parameter_that_must_be_positive_reports_an_open_lower_bound() {
        let r = distribution_refusal(DistributionError::NotPositive {
            parameter: "sigma",
            got: -1.0,
        });
        assert_eq!(r.code, "parameter_out_of_range");
        assert_eq!(
            r.fields,
            vec![
                ("parameter", Field::Path("distribution.sigma".into())),
                ("min", Field::Null),
                ("exclusive_min", Field::Num(0.0)),
                ("max", Field::Null),
                ("got", Field::Num(-1.0)),
            ]
        );
    }

    /// A parameter that must lie inside the range the others leave it is a
    /// range refusal whose ends are those parameters' values, read off the
    /// distribution's own check.
    #[test]
    fn a_parameter_out_of_order_reports_the_range_the_others_leave_it() {
        let uniform = DistributionSpec::Uniform { min: 3.0, max: 1.0 };
        let r = distribution_refusal(uniform.build().unwrap_err());
        assert_eq!(r.code, "parameter_out_of_range");
        assert_eq!(
            r.fields,
            vec![
                ("parameter", Field::Path("distribution.max".into())),
                ("min", Field::Null),
                ("exclusive_min", Field::Num(3.0)),
                ("max", Field::Null),
                ("got", Field::Num(1.0)),
            ]
        );
        // Equal ends leave no room either: `max` must be above `min`.
        let equal = DistributionSpec::Uniform { min: 2.0, max: 2.0 };
        let r = distribution_refusal(equal.build().unwrap_err());
        assert_eq!(r.fields[2], ("exclusive_min", Field::Num(2.0)));

        let triangular = DistributionSpec::Triangular {
            min: 0.0,
            mode: 5.0,
            max: 4.0,
        };
        let r = distribution_refusal(triangular.build().unwrap_err());
        assert_eq!(r.code, "parameter_out_of_range");
        assert_eq!(
            r.fields,
            vec![
                ("parameter", Field::Path("distribution.mode".into())),
                ("min", Field::Num(0.0)),
                ("max", Field::Num(4.0)),
                ("got", Field::Num(5.0)),
            ]
        );
        assert_eq!(
            r.message,
            "distribution: mode breaks the required order min ≤ mode ≤ max, got 5"
        );
    }
}
