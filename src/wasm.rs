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

// ── Refusals ──────────────────────────────────────────────────────────────

/// A value carried on a refusal.
#[derive(Debug, Clone, PartialEq)]
enum Field {
    Num(f64),
    Str(&'static str),
    /// A path built at run time, such as `a[3]`.
    Path(String),
    Null,
}

/// A refusal on its way to JavaScript: the text for `Error.message`, a stable
/// `code`, and the values behind it.
#[derive(Debug, Clone, PartialEq)]
struct Refusal {
    code: &'static str,
    message: String,
    fields: Vec<(&'static str, Field)>,
}

impl Refusal {
    /// An argument outside the range the function accepts; `max` is `None`
    /// when the range is open above.
    fn out_of_range(
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
        let code = match e {
            TransformError::NonPositiveData => "non_positive_data",
            TransformError::NonFiniteData => "value_not_finite",
            TransformError::InsufficientData => "insufficient_data",
            TransformError::InvalidTransform => "invalid_transform",
            TransformError::InvalidInverse => "invalid_inverse",
            TransformError::InvalidLambdaRange => "invalid_lambda_range",
        };
        Refusal {
            code,
            message: e.to_string(),
            fields: Vec::new(),
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

/// `p` strictly inside `(0, 1)`: `parameter_out_of_range` with `min` 0 and
/// `max` 1 -- both excluded, as the message says.
fn check_probability(p: f64) -> Result<(), Refusal> {
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

fn check_finite(name: &'static str, x: f64) -> Result<(), Refusal> {
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

// ── Interval sets ─────────────────────────────────────────────────────────

/// The interval set given by `rows` (each `[start, end]`, already read as
/// finite numbers): a row that is not a pair is `malformed_input` at
/// `parameter[i]`, a reversed pair `reversed_interval` with `parameter` and
/// `index`.
///
/// Kept apart from the `JsValue` walk so it runs off `wasm32` in tests.
fn interval_set_from(
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
        let r = Refusal::from(TransformError::NonPositiveData);
        assert_eq!(r.code, "non_positive_data");
        assert_eq!(r.message, TransformError::NonPositiveData.to_string());
    }
}
