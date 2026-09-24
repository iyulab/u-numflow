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

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

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
/// Returns a `JsValue` error string if data contains non-positive values or
/// has fewer than 2 elements.
#[wasm_bindgen]
pub fn box_cox(data: &[f64], lambda: f64) -> Result<Vec<f64>, JsValue> {
    crate::transforms::box_cox(data, lambda).map_err(|e| JsValue::from_str(&e.to_string()))
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
/// Returns a `JsValue` error string if data contains non-positive values,
/// has fewer than 2 elements, or the range is not finite with
/// `lambda_min < lambda_max`.
#[wasm_bindgen(unchecked_return_type = "LambdaEstimateDto")]
pub fn estimate_lambda(data: &[f64], lambda_min: f64, lambda_max: f64) -> Result<JsValue, JsValue> {
    let est = crate::transforms::estimate_lambda(data, lambda_min, lambda_max)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&LambdaEstimateDto {
        lambda: est.lambda,
        at_bound: est.at_bound,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
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

fn check_probability(p: f64) -> Result<(), JsValue> {
    if p.is_finite() && p > 0.0 && p < 1.0 {
        Ok(())
    } else {
        Err(JsValue::from_str(&format!(
            "p must be strictly between 0 and 1, got {p}"
        )))
    }
}

fn check_df(name: &str, df: f64) -> Result<(), JsValue> {
    if df.is_finite() && df > 0.0 {
        Ok(())
    } else {
        Err(JsValue::from_str(&format!(
            "{name} must be a finite number > 0, got {df}"
        )))
    }
}

fn check_finite(name: &str, x: f64) -> Result<(), JsValue> {
    if x.is_nan() {
        Err(JsValue::from_str(&format!(
            "{name} must be a number, got NaN"
        )))
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
