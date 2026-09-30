//! WASM bindings for u-numflow.
//!
//! Exposes a subset of the library's public API to JavaScript via `wasm-bindgen`.
//! Only enabled when the `wasm` feature is active.
//!
//! # Exported functions
//! - `mean(data)` → `f64` (NaN if empty/invalid)
//! - `std_dev(data)` → `f64` (NaN if < 2 elements or invalid)
//! - `variance(data)` → `f64` (NaN if < 2 elements or invalid)
//! - `normal_cdf(x)` → `f64` — standard normal CDF Φ(x), i.e. N(0,1)
//! - `normal_sf(x)` → `f64` — standard normal upper tail P(Z > x), tail-precise
//! - `box_cox(data, lambda)` → `Result<Vec<f64>, JsValue>`
//! - `estimate_lambda(data, lambda_min, lambda_max)` → `{ lambda, at_bound }` (or throws)
//! - `rfft(data)` → `Vec<f64>` — DFT of a real sequence, interleaved `[re0, im0, re1, im1, …]`
//! - `inverse_normal_cdf(p)` → `f64` — standard normal quantile (throws outside `(0, 1)`)
//! - `t_distribution_cdf(t, df)` / `t_distribution_quantile(p, df)` → `f64` (throw on an invalid argument)
//! - `f_distribution_cdf(x, df1, df2)` / `f_distribution_quantile(p, df1, df2)` → `f64` (same)
//! - `chi_squared_cdf(x, k)` / `chi_squared_quantile(p, k)` → `f64` (same)

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
                Field::Null => JsValue::NULL,
            };
            let _ = js_sys::Reflect::set(&err, &key.into(), &value);
        }
        err.into()
    }
}

/// Arithmetic mean of `data` using Kahan compensated summation.
///
/// Returns `NaN` if `data` is empty or contains non-finite values.
#[wasm_bindgen]
pub fn mean(data: &[f64]) -> f64 {
    crate::stats::mean(data).unwrap_or(f64::NAN)
}

/// Sample standard deviation of `data` (Bessel-corrected, denominator n−1).
///
/// Returns `NaN` if `data` has fewer than 2 elements or contains non-finite values.
#[wasm_bindgen]
pub fn std_dev(data: &[f64]) -> f64 {
    crate::stats::std_dev(data).unwrap_or(f64::NAN)
}

/// Sample variance of `data` (Bessel-corrected, denominator n−1).
///
/// Returns `NaN` if `data` has fewer than 2 elements or contains non-finite values.
#[wasm_bindgen]
pub fn variance(data: &[f64]) -> f64 {
    crate::stats::variance(data).unwrap_or(f64::NAN)
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
pub fn box_cox(data: &[f64], lambda: f64) -> Result<Vec<f64>, JsValue> {
    crate::transforms::box_cox(data, lambda).map_err(|e| Refusal::from(e).into())
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
pub fn estimate_lambda(data: &[f64], lambda_min: f64, lambda_max: f64) -> Result<JsValue, JsValue> {
    let est =
        crate::transforms::estimate_lambda(data, lambda_min, lambda_max).map_err(Refusal::from)?;
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
#[wasm_bindgen]
pub fn rfft(data: &[f64]) -> Vec<f64> {
    crate::fourier::rfft(data)
        .into_iter()
        .flat_map(|z| [z.re, z.im])
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn transform_errors_keep_their_reason() {
        let r = Refusal::from(TransformError::NonPositiveData);
        assert_eq!(r.code, "non_positive_data");
        assert_eq!(r.message, TransformError::NonPositiveData.to_string());
    }
}
