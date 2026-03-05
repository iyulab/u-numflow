//! Data transformations for statistical analysis.
//!
//! Currently provides the Box-Cox power transformation, which maps non-normal
//! positive data to approximate normality. The optimal transformation parameter
//! λ is estimated via maximum likelihood.
//!
//! # References
//! Box, G. E. P. & Cox, D. R. (1964). "An analysis of transformations."
//! *Journal of the Royal Statistical Society, Series B*, 26(2), 211–252.

use std::fmt;

use crate::stats::population_variance;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can arise from Box-Cox transformations.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformError {
    /// Box-Cox requires all y > 0.
    NonPositiveData,
    /// Need at least 2 data points.
    InsufficientData,
    /// Inverse transformation produced non-finite values.
    InvalidInverse,
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransformError::NonPositiveData => {
                write!(f, "Box-Cox requires all y > 0")
            }
            TransformError::InsufficientData => {
                write!(f, "need at least 2 data points")
            }
            TransformError::InvalidInverse => {
                write!(f, "inverse transformation produced non-finite values")
            }
        }
    }
}

impl std::error::Error for TransformError {}

// ── Validation helpers ────────────────────────────────────────────────────────

fn validate_positive_slice(y: &[f64]) -> Result<(), TransformError> {
    if y.len() < 2 {
        return Err(TransformError::InsufficientData);
    }
    if y.iter().any(|&v| v <= 0.0) {
        return Err(TransformError::NonPositiveData);
    }
    Ok(())
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Apply the Box-Cox power transformation to positive data.
///
/// For each element `yᵢ > 0`:
///
/// ```text
/// y(λ) = ln(y)           when |λ| < 1e-10
///         (y^λ - 1) / λ   otherwise
/// ```
///
/// # Errors
/// - [`TransformError::InsufficientData`] — fewer than 2 data points
/// - [`TransformError::NonPositiveData`]  — any element ≤ 0
///
/// # Examples
/// ```
/// use u_numflow::transforms::box_cox;
///
/// let y = vec![1.0, std::f64::consts::E];
/// let y_t = box_cox(&y, 0.0).unwrap();
/// assert!((y_t[0] - 0.0).abs() < 1e-10);
/// assert!((y_t[1] - 1.0).abs() < 1e-9);
/// ```
pub fn box_cox(y: &[f64], lambda: f64) -> Result<Vec<f64>, TransformError> {
    validate_positive_slice(y)?;
    let result = if lambda.abs() < 1e-10 {
        y.iter().map(|&v| v.ln()).collect()
    } else {
        y.iter().map(|&v| (v.powf(lambda) - 1.0) / lambda).collect()
    };
    Ok(result)
}

/// Invert a Box-Cox transformation.
///
/// For each transformed element `y_tᵢ`:
///
/// ```text
/// y = exp(y_t)                  when |λ| < 1e-10
///     (y_t · λ + 1)^(1/λ)      otherwise
/// ```
///
/// Accepts slices of any length, including empty. Unlike [`box_cox`], no
/// minimum length is required. An empty slice returns an empty `Vec` silently.
///
/// # Errors
/// - [`TransformError::InvalidInverse`] — any result is non-finite
///
/// # Examples
/// ```
/// use u_numflow::transforms::{box_cox, inverse_box_cox};
///
/// let y = vec![2.0, 5.0, 10.0];
/// let y_t = box_cox(&y, 1.0).unwrap();
/// let y_rec = inverse_box_cox(&y_t, 1.0).unwrap();
/// for (a, b) in y.iter().zip(y_rec.iter()) {
///     assert!((a - b).abs() < 1e-9);
/// }
/// ```
pub fn inverse_box_cox(y_t: &[f64], lambda: f64) -> Result<Vec<f64>, TransformError> {
    let result: Vec<f64> = if lambda.abs() < 1e-10 {
        y_t.iter().map(|&v| v.exp()).collect()
    } else {
        y_t.iter()
            .map(|&v| (v * lambda + 1.0).powf(1.0 / lambda))
            .collect()
    };
    if result.iter().any(|v| !v.is_finite()) {
        return Err(TransformError::InvalidInverse);
    }
    Ok(result)
}

/// Estimate the optimal Box-Cox λ via maximum likelihood.
///
/// Maximises the profile log-likelihood:
///
/// ```text
/// ℓ(λ) = -(n/2)·ln(Var_population(y(λ))) + (λ-1)·Σ ln(yᵢ)
/// ```
///
/// using a **golden-section search** over `[lambda_min, lambda_max]` with up
/// to 100 iterations (terminating early when the bracket width < 1e-6).
///
/// # Errors
/// - [`TransformError::InsufficientData`] — fewer than 2 data points
/// - [`TransformError::NonPositiveData`]  — any element ≤ 0
///
/// # Examples
/// ```
/// use u_numflow::transforms::estimate_lambda;
///
/// // Exponential data is well-linearised by log (λ ≈ 0).
/// let y: Vec<f64> = (1..=30).map(|i| (i as f64 * 0.2_f64).exp()).collect();
/// let lambda = estimate_lambda(&y, -2.0, 2.0).unwrap();
/// assert!(lambda.abs() < 0.3, "expected lambda near 0, got {lambda}");
/// ```
pub fn estimate_lambda(y: &[f64], lambda_min: f64, lambda_max: f64) -> Result<f64, TransformError> {
    if lambda_min >= lambda_max {
        return Err(TransformError::InsufficientData);
    }
    validate_positive_slice(y)?;

    let n = y.len() as f64;
    let log_sum: f64 = y.iter().map(|&v| v.ln()).sum::<f64>();

    // Profile log-likelihood (higher is better).
    let profile_ll = |lambda: f64| -> f64 {
        let y_t: Vec<f64> = if lambda.abs() < 1e-10 {
            y.iter().map(|&v| v.ln()).collect()
        } else {
            y.iter().map(|&v| (v.powf(lambda) - 1.0) / lambda).collect()
        };
        let var = population_variance(&y_t).expect("slice has >= 2 elements — variance is defined");
        if var <= 0.0 {
            return f64::NEG_INFINITY;
        }
        -(n / 2.0) * var.ln() + (lambda - 1.0) * log_sum
    };

    // Golden-section search (maximisation).
    const PHI: f64 = 0.618_033_988_749_895; // (√5 - 1) / 2
    let mut a = lambda_min;
    let mut b = lambda_max;

    let mut x1 = b - PHI * (b - a);
    let mut x2 = a + PHI * (b - a);
    let mut f1 = profile_ll(x1);
    let mut f2 = profile_ll(x2);

    for _ in 0..100 {
        if (b - a).abs() < 1e-6 {
            break;
        }
        if f1 < f2 {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + PHI * (b - a);
            f2 = profile_ll(x2);
        } else {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - PHI * (b - a);
            f1 = profile_ll(x1);
        }
    }

    Ok((a + b) / 2.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_cox_log_transform() {
        // lambda=0: y(0) = ln(y)
        let y = vec![1.0, std::f64::consts::E, std::f64::consts::E.powi(2)];
        let y_t = box_cox(&y, 0.0).unwrap();
        assert!((y_t[0] - 0.0).abs() < 1e-10);
        assert!((y_t[1] - 1.0).abs() < 1e-9);
        assert!((y_t[2] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn box_cox_identity_lambda_1() {
        // lambda=1: y(1) = (y^1 - 1)/1 = y - 1
        let y = vec![2.0, 5.0, 10.0];
        let y_t = box_cox(&y, 1.0).unwrap();
        assert!((y_t[0] - 1.0).abs() < 1e-10);
        assert!((y_t[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn box_cox_sqrt_lambda_half() {
        // lambda=0.5: y(0.5) = (sqrt(y)-1)/0.5 = 2*(sqrt(y)-1)
        let y = vec![4.0, 9.0];
        let y_t = box_cox(&y, 0.5).unwrap();
        assert!((y_t[0] - 2.0).abs() < 1e-10); // 2*(2-1)=2
        assert!((y_t[1] - 4.0).abs() < 1e-10); // 2*(3-1)=4
    }

    #[test]
    fn inverse_roundtrip_multiple_lambdas() {
        let y = vec![1.5, 2.3, 4.7, 8.1, 15.2];
        for &lambda in &[-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let y_t = box_cox(&y, lambda).unwrap();
            let y_rec = inverse_box_cox(&y_t, lambda).unwrap();
            for (orig, rec) in y.iter().zip(y_rec.iter()) {
                assert!(
                    (orig - rec).abs() < 1e-9,
                    "lambda={lambda} orig={orig} rec={rec}"
                );
            }
        }
    }

    #[test]
    fn estimate_lambda_near_zero_for_exponential() {
        // Exponential-like data → log transform (lambda ≈ 0) is optimal
        let y: Vec<f64> = (1..=30).map(|i| (i as f64 * 0.2).exp()).collect();
        let lambda = estimate_lambda(&y, -2.0, 2.0).unwrap();
        assert!(lambda.abs() < 0.3, "Expected lambda near 0, got {lambda}");
    }

    #[test]
    fn estimate_lambda_near_half_for_quadratic() {
        // y = i^2 data → sqrt transform (lambda ≈ 0.5)
        let y: Vec<f64> = (1..=20).map(|i| (i as f64).powi(2)).collect();
        let lambda = estimate_lambda(&y, -2.0, 2.0).unwrap();
        assert!(
            lambda > 0.2 && lambda < 0.8,
            "Expected lambda ~0.5, got {lambda}"
        );
    }

    #[test]
    fn non_positive_returns_error() {
        assert!(box_cox(&[1.0, -1.0, 2.0], 0.5).is_err());
        assert!(box_cox(&[0.0, 1.0, 2.0], 0.5).is_err());
    }

    #[test]
    fn insufficient_data_returns_error() {
        assert!(box_cox(&[1.0], 0.5).is_err());
        assert!(estimate_lambda(&[1.0], -2.0, 2.0).is_err());
    }

    #[test]
    fn inverse_invalid_returns_error() {
        // (y_t * lambda + 1) must be positive for real result.
        // For lambda=2: base = y_t*2+1; choose y_t=-1.0 → base=-1 → (-1)^0.5 = NaN
        let y_t = vec![-1.0, -0.8];
        assert!(inverse_box_cox(&y_t, 2.0).is_err());
    }

    #[test]
    fn estimate_lambda_invalid_range() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        assert!(estimate_lambda(&y, 1.0, 0.0).is_err()); // reversed range
        assert!(estimate_lambda(&y, 0.5, 0.5).is_err()); // equal bounds
    }

    #[test]
    fn box_cox_negative_lambda() {
        // lambda=-1: y(-1) = (y^-1 - 1) / -1 = (1 - 1/y)
        let y = vec![2.0, 4.0];
        let y_t = box_cox(&y, -1.0).unwrap();
        // (2^(-1) - 1) / (-1) = (0.5 - 1) / (-1) = 0.5
        assert!((y_t[0] - 0.5).abs() < 1e-10);
        // (4^(-1) - 1) / (-1) = (0.25 - 1) / (-1) = 0.75
        assert!((y_t[1] - 0.75).abs() < 1e-10);
    }
}
