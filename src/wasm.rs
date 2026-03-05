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
//! - `box_cox(data, lambda)` → `Result<Vec<f64>, JsValue>`
//! - `estimate_lambda(data, lambda_min, lambda_max)` → `Result<f64, JsValue>`

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
/// Uses Abramowitz & Stegun formula 26.2.17 (max abs error < 7.5 × 10⁻⁸).
#[wasm_bindgen]
pub fn normal_cdf(x: f64) -> f64 {
    crate::special::standard_normal_cdf(x)
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

/// Estimate the optimal Box-Cox lambda via profile maximum likelihood.
///
/// Searches in the range `[lambda_min, lambda_max]` using golden-section search.
///
/// # Errors
/// Returns a `JsValue` error string if data contains non-positive values,
/// has fewer than 2 elements, or `lambda_min >= lambda_max`.
#[wasm_bindgen]
pub fn estimate_lambda(data: &[f64], lambda_min: f64, lambda_max: f64) -> Result<f64, JsValue> {
    crate::transforms::estimate_lambda(data, lambda_min, lambda_max)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
