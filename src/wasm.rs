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
