//! WASM bindings for u-numflow.
//!
//! Exposes a subset of the library's public API to JavaScript via `wasm-bindgen`.
//! Only enabled when the `wasm` feature is active.
//!
//! # Exported functions
//! - `mean(data)` → `f64` (throws `empty_input` when `data` is empty)
//! - `std_dev(data)` → `f64` (throws `insufficient_data` below 2 elements)
//! - `variance(data)` → `f64` (throws `insufficient_data` below 2 elements)
//! - `normal_cdf(x)` → `f64` — standard normal CDF Φ(x), i.e. N(0,1)
//! - `normal_sf(x)` → `f64` — standard normal upper tail P(Z > x), tail-precise
//! - `box_cox(data, lambda)` → `Result<Vec<f64>, JsValue>`
//! - `estimate_lambda(data, lambda_min, lambda_max)` → `{ lambda, at_bound }` (or throws)
//! - `rfft(data)` → `Vec<f64>` — DFT of a real sequence, interleaved `[re0, im0, re1, im1, …]`
//! - `inverse_normal_cdf(p)` → `f64` — standard normal quantile (throws outside `(0, 1)`)
//! - `t_distribution_cdf(t, df)` / `t_distribution_quantile(p, df)` → `f64` (throw on an invalid argument)
//! - `f_distribution_cdf(x, df1, df2)` / `f_distribution_quantile(p, df1, df2)` → `f64` (same)
//! - `chi_squared_cdf(x, k)` / `chi_squared_quantile(p, k)` → `f64` (same)
//! - `distribution_moments(distribution)` → `{ mean, variance }`
//! - `distribution_cdf(distribution, x)` · `distribution_quantile(distribution, p)` →
//!   `f64`; `distribution_sample(distribution, n, seed)` → `Float64Array`
//! - `interval_normalize(intervals)` · `interval_union(a, b)` ·
//!   `interval_intersection(a, b)` · `interval_difference(a, b)` →
//!   `[number, number][]` in normal form; `interval_measure(intervals)` → `f64`
//!
//! Every `data` argument is a `number[]` or a `Float64Array`, read as sent: an
//! element that is not a number (`null`, a string) is refused as
//! `malformed_input` and a NaN or ±Infinity as `value_not_finite`, each with
//! the element's `index`. (A `&[f64]` parameter would let wasm-bindgen copy the
//! array into a typed array first, turning `null` into 0 and a string into NaN.)
//!
//! Every refusal throws an `Error` whose `message` is readable text and which
//! carries `code` -- a stable reason -- and the values behind it (`parameter`,
//! `min`, `max`, `got`). See the README's *Errors*.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use crate::transforms::TransformError;
use crate::wire::{
    check_finite, check_kind, check_probability, distribution_refusal, interval_set_from, whole,
    Field, Refusal,
};

/// One element of a JS number array, as found.
#[derive(Debug, Clone, PartialEq)]
enum Element {
    Number(f64),
    /// Anything else, by its JS type name (`"null"`, `"string"`, …).
    Other(String),
}

/// The values of a number array, refusing the first element that is not a
/// finite number and naming where it sits.
///
/// Kept apart from the `JsValue` walk so it runs off `wasm32` in tests.
fn numbers_from(
    parameter: &'static str,
    elements: impl IntoIterator<Item = Element>,
) -> Result<Vec<f64>, Refusal> {
    let mut out = Vec::new();
    for (i, element) in elements.into_iter().enumerate() {
        match element {
            Element::Number(x) if x.is_finite() => out.push(x),
            Element::Number(x) => {
                return Err(Refusal {
                    code: "value_not_finite",
                    message: format!("{parameter}[{i}]: expected a finite number, got {x}"),
                    fields: vec![
                        ("parameter", Field::Str(parameter)),
                        ("index", Field::Num(i as f64)),
                    ],
                })
            }
            Element::Other(kind) => {
                return Err(Refusal {
                    code: "malformed_input",
                    message: format!("{parameter}[{i}]: expected a number, got {kind}"),
                    fields: vec![
                        ("parameter", Field::Str(parameter)),
                        ("index", Field::Num(i as f64)),
                    ],
                })
            }
        }
    }
    Ok(out)
}

/// Reads a `number[]` or `Float64Array` argument without converting it.
fn read_numbers(value: &JsValue, parameter: &'static str) -> Result<Vec<f64>, Refusal> {
    use wasm_bindgen::JsCast;
    if let Some(typed) = value.dyn_ref::<js_sys::Float64Array>() {
        return numbers_from(parameter, typed.to_vec().into_iter().map(Element::Number));
    }
    if !js_sys::Array::is_array(value) {
        return Err(Refusal {
            code: "malformed_input",
            message: format!("{parameter}: expected an array of numbers or a Float64Array"),
            fields: vec![("parameter", Field::Str(parameter))],
        });
    }
    let array: &js_sys::Array = value.unchecked_ref();
    numbers_from(
        parameter,
        array.iter().map(|item| match item.as_f64() {
            Some(x) => Element::Number(x),
            None if item.is_null() => Element::Other("null".to_string()),
            None => Element::Other(item.js_typeof().as_string().unwrap_or_default()),
        }),
    )
}

/// `data` with at least `min` values, or `insufficient_data` saying so.
fn at_least(data: Vec<f64>, min: usize) -> Result<Vec<f64>, Refusal> {
    if data.len() >= min {
        Ok(data)
    } else {
        Err(Refusal {
            code: "insufficient_data",
            message: format!("data: at least {min} values are needed, got {}", data.len()),
            fields: vec![
                ("parameter", Field::Str("data")),
                ("min", Field::Num(min as f64)),
                ("got", Field::Num(data.len() as f64)),
            ],
        })
    }
}

impl From<TransformError> for Refusal {
    fn from(e: TransformError) -> Self {
        let data = ("parameter", Field::Str("data"));
        let (code, fields) = match e {
            TransformError::NonPositiveData { index, value } => (
                "non_positive_data",
                vec![
                    data,
                    ("index", Field::Num(index as f64)),
                    ("got", Field::Num(value)),
                ],
            ),
            TransformError::NonFiniteData { index } => (
                "value_not_finite",
                vec![data, ("index", Field::Num(index as f64))],
            ),
            TransformError::InsufficientData { min, got } => (
                "insufficient_data",
                vec![
                    data,
                    ("min", Field::Num(min as f64)),
                    ("got", Field::Num(got as f64)),
                ],
            ),
            TransformError::InvalidLambdaRange { min, max } => (
                "invalid_lambda_range",
                vec![
                    ("parameter", Field::Str("lambda_min")),
                    ("min", Field::Num(min)),
                    ("max", Field::Num(max)),
                ],
            ),
            TransformError::InvalidTransform => ("invalid_transform", Vec::new()),
            TransformError::InvalidInverse => ("invalid_inverse", Vec::new()),
        };
        Refusal {
            code,
            message: e.to_string(),
            fields,
        }
    }
}

/// Every refusal crosses into JavaScript as an `Error` whose `message` is the
/// readable text, with `code` and the fields set on it as properties.
impl From<Refusal> for JsValue {
    fn from(refusal: Refusal) -> JsValue {
        let err = js_sys::Error::new(&refusal.message);
        // `Reflect::set` on a freshly created ordinary object cannot fail.
        let _ = js_sys::Reflect::set(&err, &"code".into(), &refusal.code.into());
        for (key, value) in refusal.fields {
            let value = match value {
                Field::Num(n) => JsValue::from_f64(n),
                Field::Str(s) => s.into(),
                Field::Path(s) => s.into(),
                Field::Null => JsValue::NULL,
            };
            let _ = js_sys::Reflect::set(&err, &key.into(), &value);
        }
        err.into()
    }
}

/// Arithmetic mean of `data` using Kahan compensated summation.
///
/// # Errors
/// Throws `empty_input` when `data` is empty, and as every `data` argument
/// does for an element that is not a finite number.
#[wasm_bindgen]
pub fn mean(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
) -> Result<f64, JsValue> {
    let data = read_numbers(&data, "data")?;
    if data.is_empty() {
        return Err(Refusal {
            code: "empty_input",
            message: "data: the mean of no values is undefined".to_string(),
            fields: vec![("parameter", Field::Str("data"))],
        }
        .into());
    }
    Ok(crate::stats::mean(&data).expect("non-empty finite data has a mean"))
}

/// Sample standard deviation of `data` (Bessel-corrected, denominator n−1).
///
/// # Errors
/// Throws `insufficient_data` below 2 values.
#[wasm_bindgen]
pub fn std_dev(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
) -> Result<f64, JsValue> {
    let data = at_least(read_numbers(&data, "data")?, 2)?;
    Ok(crate::stats::std_dev(&data).expect("two or more finite values have a std_dev"))
}

/// Sample variance of `data` (Bessel-corrected, denominator n−1).
///
/// # Errors
/// Throws `insufficient_data` below 2 values.
#[wasm_bindgen]
pub fn variance(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
) -> Result<f64, JsValue> {
    let data = at_least(read_numbers(&data, "data")?, 2)?;
    Ok(crate::stats::variance(&data).expect("two or more finite values have a variance"))
}

/// Standard normal CDF Φ(x) = P(Z ≤ x) for Z ~ N(0, 1).
///
/// To evaluate a general normal N(μ, σ), pass `(x - μ) / σ`.
///
/// Computed as `erfc(−x/√2)/2` with a direct `erfc`, so the lower tail keeps
/// relative precision. For the upper tail use [`normal_sf`].
#[wasm_bindgen]
pub fn normal_cdf(x: f64) -> f64 {
    crate::special::standard_normal_cdf(x)
}

/// Standard normal survival function P(Z > x) = 1 − Φ(x) for Z ~ N(0, 1).
///
/// Computed directly (not as `1 − normal_cdf(x)`), so upper-tail
/// probabilities such as PPM defect rates keep their significant digits:
/// `normal_sf(6)` is `9.865876450376948e-10` to about 15 digits.
#[wasm_bindgen]
pub fn normal_sf(x: f64) -> f64 {
    crate::special::standard_normal_sf(x)
}

/// Apply the Box-Cox power transformation to positive data.
///
/// All values in `data` must be strictly positive.
///
/// # Errors
/// Throws `non_positive_data`, `value_not_finite` or `insufficient_data` when
/// data contains a value ≤ 0, a NaN or infinity, or fewer than 2 elements, and
/// `invalid_transform` when a result is not finite.
#[wasm_bindgen]
pub fn box_cox(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
    lambda: f64,
) -> Result<Vec<f64>, JsValue> {
    let data = read_numbers(&data, "data")?;
    crate::transforms::box_cox(&data, lambda).map_err(|e| Refusal::from(e).into())
}

/// Output of [`estimate_lambda`]: `{ lambda, at_bound }`.
#[derive(serde::Serialize, tsify::Tsify)]
struct LambdaEstimateDto {
    lambda: f64,
    at_bound: bool,
}

/// Estimate the optimal Box-Cox lambda via profile maximum likelihood.
///
/// Searches in the range `[lambda_min, lambda_max]` using golden-section search
/// and returns `{ lambda: number, at_bound: boolean }`. `at_bound` is `true`
/// when the maximum lies on an end of the range — the likelihood was still
/// rising there, so `lambda` is that range limit (reported exactly), not an
/// interior estimate; widen the range to find the unconstrained optimum.
///
/// # Errors
/// Throws as [`box_cox`] does for the data, and `invalid_lambda_range` when the
/// range is not finite with `lambda_min < lambda_max`.
#[wasm_bindgen(unchecked_return_type = "LambdaEstimateDto")]
pub fn estimate_lambda(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
    lambda_min: f64,
    lambda_max: f64,
) -> Result<JsValue, JsValue> {
    let data = read_numbers(&data, "data")?;
    let est =
        crate::transforms::estimate_lambda(&data, lambda_min, lambda_max).map_err(Refusal::from)?;
    serde_wasm_bindgen::to_value(&LambdaEstimateDto {
        lambda: est.lambda,
        at_bound: est.at_bound,
    })
    .map_err(|e| {
        Refusal {
            code: "malformed_input",
            message: e.to_string(),
            fields: vec![("parameter", Field::Str("result"))],
        }
        .into()
    })
}

/// Forward DFT of a real sequence of any length.
///
/// Returns the `n` complex bins interleaved as `[re0, im0, re1, im1, …]`
/// (length `2n`). Bins `k` and `n − k` are conjugates, so the spectrum of a
/// real signal is fully described by bins `0..=n/2`.
///
/// # Errors
/// Throws as every `data` argument does for an element that is not a finite
/// number.
#[wasm_bindgen]
pub fn rfft(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
) -> Result<Vec<f64>, JsValue> {
    let data = read_numbers(&data, "data")?;
    Ok(crate::fourier::rfft(&data)
        .into_iter()
        .flat_map(|z| [z.re, z.im])
        .collect())
}

// ── Distribution CDFs and quantiles (critical values) ─────────────────────
//
// The crate functions return NaN for an argument outside their domain. Across
// the JS boundary a NaN critical value draws nothing and raises nothing, so
// these wrappers refuse such arguments instead and name the one that failed.

/// Degrees of freedom: a finite number above 0 (`min` 0, excluded).
fn check_df(name: &'static str, df: f64) -> Result<(), Refusal> {
    if df.is_finite() && df > 0.0 {
        Ok(())
    } else {
        Err(Refusal::out_of_range(
            name,
            0.0,
            None,
            df,
            format!("{name} must be a finite number > 0, got {df}"),
        ))
    }
}

/// Standard normal quantile Φ⁻¹(p): the `z` with `P(Z ≤ z) = p`.
///
/// # Errors
/// Throws if `p` is not strictly between 0 and 1.
#[wasm_bindgen]
pub fn inverse_normal_cdf(p: f64) -> Result<f64, JsValue> {
    check_probability(p)?;
    Ok(crate::special::inverse_normal_cdf(p))
}

/// Student's t CDF: `P(T ≤ t)` for `T ~ t(df)`. `df` may be fractional.
///
/// # Errors
/// Throws if `t` is NaN or `df` is not a finite number > 0.
#[wasm_bindgen]
pub fn t_distribution_cdf(t: f64, df: f64) -> Result<f64, JsValue> {
    check_finite("t", t)?;
    check_df("df", df)?;
    Ok(crate::special::t_distribution_cdf(t, df))
}

/// Student's t quantile: the `t` with `P(T ≤ t) = p`. A two-sided critical
/// value at level α is `t_distribution_quantile(1 − α/2, df)`.
///
/// # Errors
/// Throws if `p` is not strictly between 0 and 1, or `df` is not a finite
/// number > 0.
#[wasm_bindgen]
pub fn t_distribution_quantile(p: f64, df: f64) -> Result<f64, JsValue> {
    check_probability(p)?;
    check_df("df", df)?;
    Ok(crate::special::t_distribution_quantile(p, df))
}

/// F CDF: `P(X ≤ x)` for `X ~ F(df1, df2)` (`0` for `x ≤ 0`).
///
/// # Errors
/// Throws if `x` is NaN, or `df1`/`df2` is not a finite number > 0.
#[wasm_bindgen]
pub fn f_distribution_cdf(x: f64, df1: f64, df2: f64) -> Result<f64, JsValue> {
    check_finite("x", x)?;
    check_df("df1", df1)?;
    check_df("df2", df2)?;
    Ok(crate::special::f_distribution_cdf(x, df1, df2))
}

/// F quantile: the `x` with `P(X ≤ x) = p` for `X ~ F(df1, df2)`.
///
/// # Errors
/// Throws if `p` is not strictly between 0 and 1, or `df1`/`df2` is not a
/// finite number > 0.
#[wasm_bindgen]
pub fn f_distribution_quantile(p: f64, df1: f64, df2: f64) -> Result<f64, JsValue> {
    check_probability(p)?;
    check_df("df1", df1)?;
    check_df("df2", df2)?;
    Ok(crate::special::f_distribution_quantile(p, df1, df2))
}

/// Chi-squared CDF: `P(X ≤ x)` for `X ~ χ²(k)` (`0` for `x ≤ 0`).
///
/// # Errors
/// Throws if `x` is NaN or `k` is not a finite number > 0.
#[wasm_bindgen]
pub fn chi_squared_cdf(x: f64, k: f64) -> Result<f64, JsValue> {
    check_finite("x", x)?;
    check_df("k", k)?;
    Ok(crate::special::chi_squared_cdf(x, k))
}

/// Chi-squared quantile: the `x` with `P(X ≤ x) = p` for `X ~ χ²(k)`.
///
/// # Errors
/// Throws if `p` is not strictly between 0 and 1, or `k` is not a finite
/// number > 0.
#[wasm_bindgen]
pub fn chi_squared_quantile(p: f64, k: f64) -> Result<f64, JsValue> {
    check_probability(p)?;
    check_df("k", k)?;
    Ok(crate::special::chi_squared_quantile(p, k))
}

// ── Distributions by specification ────────────────────────────────────────
//
// `distribution` is `{ kind, ...parameters }` (see `wire::DistributionSpec`).

/// A NaN or ±Infinity anywhere in a JS value, with the path to it.
fn find_non_finite(value: &JsValue, parameter: &str) -> Option<(String, f64)> {
    if let Some(n) = value.as_f64() {
        return (!n.is_finite()).then(|| (parameter.to_string(), n));
    }
    if !value.is_object() {
        return None;
    }
    let object: &js_sys::Object = wasm_bindgen::JsCast::unchecked_ref(value);
    for entry in js_sys::Object::entries(object).iter() {
        let pair: js_sys::Array = wasm_bindgen::JsCast::unchecked_into(entry);
        let key = pair.get(0).as_string().unwrap_or_default();
        let inner = find_non_finite(&pair.get(1), &format!("{parameter}.{key}"));
        if inner.is_some() {
            return inner;
        }
    }
    None
}

/// A `distribution` argument, read as sent and built.
fn read_distribution(value: &JsValue) -> Result<crate::wire::Distribution, Refusal> {
    const P: &str = "distribution";
    let malformed = |message: String| Refusal {
        code: "malformed_input",
        message,
        fields: vec![("parameter", Field::Str(P))],
    };
    if !value.is_object() {
        return Err(malformed(format!(
            "{P}: expected an object such as {{ kind: \"normal\", mu: 0, sigma: 1 }}"
        )));
    }
    let kind = js_sys::Reflect::get(value, &"kind".into()).unwrap_or(JsValue::UNDEFINED);
    let Some(kind) = kind.as_string() else {
        return Err(malformed(format!(
            "{P}.kind: expected a string naming the distribution"
        )));
    };
    check_kind(&kind)?;
    if let Some((path, n)) = find_non_finite(value, P) {
        return Err(Refusal {
            code: "value_not_finite",
            message: format!("{path}: expected a finite number, got {n}"),
            fields: vec![("parameter", Field::Path(path))],
        });
    }
    let spec: crate::wire::DistributionSpec = serde_wasm_bindgen::from_value(value.clone())
        .map_err(|e| {
            // serde-wasm-bindgen's text carries a leading "Error: ".
            let text = e.to_string();
            malformed(format!("{P}: {}", text.trim_start_matches("Error: ")))
        })?;
    spec.build().map_err(distribution_refusal)
}

/// Mean and variance of a distribution.
#[derive(serde::Serialize, tsify::Tsify)]
struct MomentsDto {
    mean: f64,
    variance: f64,
}

/// Mean and variance of `distribution` — also the way to check a
/// specification before using it.
///
/// # Errors
/// As [`distribution_cdf`] for `distribution`.
#[wasm_bindgen(unchecked_return_type = "MomentsDto")]
pub fn distribution_moments(
    #[wasm_bindgen(unchecked_param_type = "DistributionSpec")] distribution: JsValue,
) -> Result<JsValue, JsValue> {
    let d = read_distribution(&distribution)?;
    let dto = MomentsDto {
        mean: d.mean(),
        variance: d.variance(),
    };
    Ok(serde_wasm_bindgen::to_value(&dto).map_err(|e| Refusal {
        code: "internal",
        message: e.to_string(),
        fields: Vec::new(),
    })?)
}

/// `P(X ≤ x)` for the distribution `{ kind, ...parameters }`.
///
/// # Errors
/// As every `distribution` argument: `malformed_input` for a value that is
/// not such an object, `unknown_option` for an unknown `kind` (with
/// `expected`), `value_not_finite` for a NaN or infinite parameter,
/// `parameter_out_of_range` for a parameter that must be `> 0`, and
/// `invalid_option` for parameters out of order (`min < max`,
/// `min ≤ mode ≤ max`). `x` NaN throws `value_not_finite`.
#[wasm_bindgen]
pub fn distribution_cdf(
    #[wasm_bindgen(unchecked_param_type = "DistributionSpec")] distribution: JsValue,
    x: f64,
) -> Result<f64, JsValue> {
    let d = read_distribution(&distribution)?;
    check_finite("x", x)?;
    Ok(d.cdf(x))
}

/// The `x` with `P(X ≤ x) = p`.
///
/// # Errors
/// As [`distribution_cdf`] for `distribution`; `p` outside `(0, 1)` throws
/// `parameter_out_of_range`.
#[wasm_bindgen]
pub fn distribution_quantile(
    #[wasm_bindgen(unchecked_param_type = "DistributionSpec")] distribution: JsValue,
    p: f64,
) -> Result<f64, JsValue> {
    let d = read_distribution(&distribution)?;
    check_probability(p)?;
    Ok(d.quantile(p))
}

/// `n` random variates, reproducible from `seed`: the same `(distribution, n,
/// seed)` always returns the same values. Uniforms are drawn from the open
/// interval (0, 1), so no variate is infinite.
///
/// # Errors
/// As [`distribution_cdf`] for `distribution`; `n` not a whole number in
/// `[0, 2³² − 1]` or `seed` not a whole number in `[0, 2⁵³]` throws
/// `parameter_out_of_range`.
#[wasm_bindgen]
pub fn distribution_sample(
    #[wasm_bindgen(unchecked_param_type = "DistributionSpec")] distribution: JsValue,
    n: f64,
    seed: f64,
) -> Result<Vec<f64>, JsValue> {
    let d = read_distribution(&distribution)?;
    let n = whole("n", n, u32::MAX as f64)? as usize;
    let seed = whole("seed", seed, 9_007_199_254_740_992.0)? as u64;
    Ok(d.sample_n(n, seed))
}

// ── Interval sets ─────────────────────────────────────────────────────────

/// Reads a `[number, number][]` argument as sent.
fn read_intervals(
    value: &JsValue,
    parameter: &'static str,
) -> Result<crate::collections::IntervalSet<f64>, Refusal> {
    use wasm_bindgen::JsCast;
    if !js_sys::Array::is_array(value) {
        return Err(Refusal {
            code: "malformed_input",
            message: format!("{parameter}: expected an array of [start, end] pairs"),
            fields: vec![("parameter", Field::Str(parameter))],
        });
    }
    let array: &js_sys::Array = value.unchecked_ref();
    let mut rows = Vec::with_capacity(array.length() as usize);
    for (i, item) in array.iter().enumerate() {
        let row = read_numbers(&item, parameter).map_err(|mut r| {
            let path = format!("{parameter}[{i}]");
            r.message = r.message.replacen(parameter, &path, 1);
            for field in &mut r.fields {
                if field.0 == "parameter" {
                    field.1 = Field::Path(path.clone());
                }
            }
            r
        })?;
        rows.push(row);
    }
    interval_set_from(parameter, rows)
}

/// The pieces of a set as a JS `[start, end][]`.
fn intervals_to_js(set: &crate::collections::IntervalSet<f64>) -> JsValue {
    set.iter()
        .map(|(s, e)| js_sys::Array::of2(&JsValue::from_f64(s), &JsValue::from_f64(e)))
        .collect::<js_sys::Array>()
        .into()
}

/// The union of `intervals` (each `[start, end]`, half-open) in normal form:
/// sorted, disjoint, touching pieces merged, empty ones dropped.
///
/// # Errors
/// Throws `malformed_input` for a row that is not two numbers (`parameter` is
/// the row's path, e.g. `intervals[2]`), `value_not_finite` for a NaN or
/// ±Infinity bound, and `reversed_interval` (`parameter`, `index`) for a row
/// with start > end — reversed rows are refused, not swapped.
#[wasm_bindgen(unchecked_return_type = "[number, number][]")]
pub fn interval_normalize(
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] intervals: JsValue,
) -> Result<JsValue, JsValue> {
    Ok(intervals_to_js(&read_intervals(&intervals, "intervals")?))
}

/// Total length of the union of `intervals`: each point counts once, however
/// many rows cover it.
///
/// # Errors
/// As [`interval_normalize`].
#[wasm_bindgen]
pub fn interval_measure(
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] intervals: JsValue,
) -> Result<f64, JsValue> {
    Ok(read_intervals(&intervals, "intervals")?.measure())
}

/// Points in `a` or `b`, in normal form.
///
/// # Errors
/// As [`interval_normalize`], naming `a` or `b`.
#[wasm_bindgen(unchecked_return_type = "[number, number][]")]
pub fn interval_union(
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] a: JsValue,
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] b: JsValue,
) -> Result<JsValue, JsValue> {
    let (a, b) = (read_intervals(&a, "a")?, read_intervals(&b, "b")?);
    Ok(intervals_to_js(&a.union(&b)))
}

/// Points in both `a` and `b`, in normal form. Clipping to a window is
/// `interval_intersection(a, [[from, to]])`.
///
/// # Errors
/// As [`interval_normalize`], naming `a` or `b`.
#[wasm_bindgen(unchecked_return_type = "[number, number][]")]
pub fn interval_intersection(
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] a: JsValue,
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] b: JsValue,
) -> Result<JsValue, JsValue> {
    let (a, b) = (read_intervals(&a, "a")?, read_intervals(&b, "b")?);
    Ok(intervals_to_js(&a.intersection(&b)))
}

/// Points in `a` but not in `b`, in normal form — e.g. a reporting window
/// minus down-time is up-time.
///
/// # Errors
/// As [`interval_normalize`], naming `a` or `b`.
#[wasm_bindgen(unchecked_return_type = "[number, number][]")]
pub fn interval_difference(
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] a: JsValue,
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] b: JsValue,
) -> Result<JsValue, JsValue> {
    let (a, b) = (read_intervals(&a, "a")?, read_intervals(&b, "b")?);
    Ok(intervals_to_js(&a.difference(&b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_interval_row_that_is_not_a_pair_names_its_path() {
        let r = interval_set_from("a", vec![vec![0.0, 1.0], vec![2.0]]).expect_err("short row");
        assert_eq!(r.code, "malformed_input");
        assert_eq!(r.fields, vec![("parameter", Field::Path("a[1]".into()))]);
        assert_eq!(r.message, "a[1]: expected [start, end], got 1 numbers");
    }

    #[test]
    fn a_reversed_interval_is_refused_at_its_index() {
        let r = interval_set_from("b", vec![vec![0.0, 1.0], vec![5.0, 3.0]]).expect_err("reversed");
        assert_eq!(r.code, "reversed_interval");
        assert_eq!(
            r.fields,
            vec![("parameter", Field::Str("b")), ("index", Field::Num(1.0))]
        );
    }

    #[test]
    fn interval_rows_build_a_normal_form_set() {
        let s = interval_set_from("a", vec![vec![3.0, 5.0], vec![0.0, 4.0]]).expect("valid");
        assert_eq!(s.as_slice(), &[(0.0, 5.0)]);
    }

    #[test]
    fn a_probability_outside_0_1_names_the_range() {
        let r = check_probability(1.5).expect_err("outside");
        assert_eq!(r.code, "parameter_out_of_range");
        assert_eq!(
            r.fields,
            vec![
                ("parameter", Field::Str("p")),
                ("min", Field::Num(0.0)),
                ("max", Field::Num(1.0)),
                ("got", Field::Num(1.5)),
            ]
        );
        assert!(check_probability(0.5).is_ok());
    }

    #[test]
    fn degrees_of_freedom_are_open_above() {
        let r = check_df("df2", 0.0).expect_err("zero");
        assert_eq!(r.fields[0], ("parameter", Field::Str("df2")));
        assert_eq!(r.fields[2], ("max", Field::Null));
    }

    #[test]
    fn nan_is_refused_as_not_finite() {
        assert_eq!(
            check_finite("x", f64::NAN).expect_err("NaN").code,
            "value_not_finite"
        );
        assert!(check_finite("x", f64::INFINITY).is_ok());
    }

    #[test]
    fn a_number_array_is_read_as_sent() {
        let ok = numbers_from("data", [1.0, -2.5].map(Element::Number)).expect("numbers");
        assert_eq!(ok, vec![1.0, -2.5]);
        assert!(numbers_from("data", []).expect("empty").is_empty());
    }

    #[test]
    fn null_is_refused_where_it_sits_not_read_as_zero() {
        let r = numbers_from(
            "data",
            [
                Element::Number(1.0),
                Element::Other("null".into()),
                Element::Number(3.0),
            ],
        )
        .expect_err("null");
        assert_eq!(r.code, "malformed_input");
        assert_eq!(
            r.fields,
            vec![
                ("parameter", Field::Str("data")),
                ("index", Field::Num(1.0))
            ]
        );
        assert!(r.message.contains("got null"), "{}", r.message);
    }

    #[test]
    fn a_non_finite_element_is_not_finite_with_its_index() {
        let r = numbers_from("data", [0.0, f64::INFINITY].map(Element::Number)).expect_err("inf");
        assert_eq!(r.code, "value_not_finite");
        assert_eq!(r.fields[1], ("index", Field::Num(1.0)));
    }

    #[test]
    fn too_few_values_say_how_many_are_needed() {
        let r = at_least(vec![1.0], 2).expect_err("one");
        assert_eq!(r.code, "insufficient_data");
        assert_eq!(r.fields[1], ("min", Field::Num(2.0)));
        assert_eq!(r.fields[2], ("got", Field::Num(1.0)));
        assert!(at_least(vec![1.0, 2.0], 2).is_ok());
    }

    #[test]
    fn transform_errors_keep_their_reason() {
        let r = Refusal::from(TransformError::NonPositiveData {
            index: 2,
            value: -0.5,
        });
        assert_eq!(r.code, "non_positive_data");
        assert_eq!(
            r.message,
            TransformError::NonPositiveData {
                index: 2,
                value: -0.5
            }
            .to_string()
        );
    }
}
