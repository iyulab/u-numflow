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
    /// Data contains NaN or an infinity.
    NonFiniteData,
    /// Need at least 2 data points.
    InsufficientData,
    /// Inverse transformation produced non-finite values.
    InvalidInverse,
    /// The λ search range is empty or not finite (`lambda_min >= lambda_max`).
    InvalidLambdaRange,
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransformError::NonPositiveData => {
                write!(f, "Box-Cox requires all y > 0")
            }
            TransformError::NonFiniteData => {
                write!(f, "data must not contain NaN or infinite values")
            }
            TransformError::InsufficientData => {
                write!(f, "need at least 2 data points")
            }
            TransformError::InvalidInverse => {
                write!(f, "inverse transformation produced non-finite values")
            }
            TransformError::InvalidLambdaRange => {
                write!(
                    f,
                    "lambda search range must be finite with lambda_min < lambda_max"
                )
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
    if y.iter().any(|v| !v.is_finite()) {
        return Err(TransformError::NonFiniteData);
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
/// - [`TransformError::NonFiniteData`]    — any element is NaN or infinite
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

/// A maximum-likelihood Box-Cox λ, with where it sits in its search range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LambdaEstimate {
    /// The λ that maximises the profile log-likelihood within the range.
    pub lambda: f64,
    /// `true` when the maximum lies on an end of the search range (within
    /// the search tolerance): the likelihood was still rising there, so the
    /// unconstrained optimum is at or beyond that end and `lambda` is the
    /// range limit, not an interior estimate.
    pub at_bound: bool,
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
/// The result says whether the maximum was found on an end of the range
/// ([`LambdaEstimate::at_bound`]): a λ of `lambda_max` because the likelihood
/// peaks there is indistinguishable by value from one where the likelihood
/// keeps rising past it, and only the latter means the range cut the search
/// short.
///
/// # Errors
/// - [`TransformError::InvalidLambdaRange`] — `lambda_min >= lambda_max`, or
///   either bound is not finite
/// - [`TransformError::InsufficientData`] — fewer than 2 data points
/// - [`TransformError::NonPositiveData`]  — any element ≤ 0
/// - [`TransformError::NonFiniteData`]    — any element is NaN or infinite
///
/// # Examples
/// ```
/// use u_numflow::transforms::estimate_lambda;
///
/// // Exponential data is well-linearised by log (λ ≈ 0).
/// let y: Vec<f64> = (1..=30).map(|i| (i as f64 * 0.2_f64).exp()).collect();
/// let est = estimate_lambda(&y, -2.0, 2.0).unwrap();
/// assert!(est.lambda.abs() < 0.3, "expected lambda near 0, got {}", est.lambda);
/// assert!(!est.at_bound);
/// ```
pub fn estimate_lambda(
    y: &[f64],
    lambda_min: f64,
    lambda_max: f64,
) -> Result<LambdaEstimate, TransformError> {
    if !lambda_min.is_finite() || !lambda_max.is_finite() || lambda_min >= lambda_max {
        return Err(TransformError::InvalidLambdaRange);
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
        // A λ whose transform overflows (`y^λ` beyond f64 for large |λ| or
        // large y) has no finite variance: it is not a candidate, not a panic.
        let var = match population_variance(&y_t) {
            Some(v) if v > 0.0 && v.is_finite() => v,
            _ => return f64::NEG_INFINITY,
        };
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

    // Golden-section only ever moves the end of the bracket on the side away
    // from the maximum. An end that never moved means every comparison pointed
    // toward it: the maximum is on that end of the range.
    // On a bound the constrained maximiser *is* that bound, so report it exactly
    // rather than the bracket midpoint half a tolerance inside it.
    let (lambda, at_bound) = if a == lambda_min {
        (lambda_min, true)
    } else if b == lambda_max {
        (lambda_max, true)
    } else {
        ((a + b) / 2.0, false)
    };
    Ok(LambdaEstimate { lambda, at_bound })
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
        let lambda = estimate_lambda(&y, -2.0, 2.0).unwrap().lambda;
        assert!(lambda.abs() < 0.3, "Expected lambda near 0, got {lambda}");
    }

    /// Data that is exactly normal after a Box-Cox transform with `lambda0`:
    /// normal quantiles pushed through the inverse transform.
    fn normal_after_boxcox(lambda0: f64, n: usize) -> Vec<f64> {
        (1..=n)
            .map(|i| {
                let p = (i as f64 - 0.5) / n as f64;
                let z = 10.0 + 2.0 * crate::special::inverse_normal_cdf(p);
                (lambda0 * z + 1.0).powf(1.0 / lambda0)
            })
            .collect()
    }

    #[test]
    fn estimate_lambda_flags_a_range_that_cuts_the_search_short() {
        let y = normal_after_boxcox(4.0, 100);
        // Wide enough: an interior maximum near the generating lambda.
        let wide = estimate_lambda(&y, -5.0, 5.0).unwrap();
        assert!(!wide.at_bound, "interior maximum: {wide:?}");
        assert!((wide.lambda - 4.0).abs() < 0.5, "{wide:?}");
        // Too narrow: the likelihood is still rising at 2, which must be said —
        // and the reported lambda is the bound itself, not a midpoint near it.
        let narrow = estimate_lambda(&y, -2.0, 2.0).unwrap();
        assert!(narrow.at_bound, "{narrow:?}");
        assert_eq!(narrow.lambda, 2.0);
        // The lower end, from the reciprocals: 1/y transformed at −λ is the
        // negated transform of y at λ, so the likelihood mirrors to λ ≈ −4.
        let recip: Vec<f64> = y.iter().map(|v| 1.0 / v).collect();
        let low = estimate_lambda(&recip, -2.0, 2.0).unwrap();
        assert!(low.at_bound, "{low:?}");
        assert_eq!(low.lambda, -2.0);
        let wide = estimate_lambda(&recip, -5.0, 5.0).unwrap();
        assert!(
            !wide.at_bound && (wide.lambda + 4.0).abs() < 0.5,
            "{wide:?}"
        );
    }

    #[test]
    fn estimate_lambda_survives_a_transform_that_overflows() {
        // y^λ overflows f64 for λ near the top of the range: such λ are not
        // candidates, and the search must not panic on them.
        let y: Vec<f64> = (1..=20).map(|i| 10f64.powi(i * 3)).collect();
        let est = estimate_lambda(&y, -20.0, 20.0).expect("finite data, valid range");
        assert!(est.lambda.is_finite() && est.lambda < 20.0, "{est:?}");
    }

    #[test]
    fn non_finite_data_is_reported_not_passed_through() {
        for bad in [f64::NAN, f64::INFINITY] {
            let y = [1.0, bad, 3.0];
            assert_eq!(box_cox(&y, 0.5), Err(TransformError::NonFiniteData));
            assert_eq!(
                estimate_lambda(&y, -2.0, 2.0),
                Err(TransformError::NonFiniteData)
            );
        }
    }

    #[test]
    fn estimate_lambda_near_half_for_quadratic() {
        // y = i^2 data → sqrt transform (lambda ≈ 0.5)
        let y: Vec<f64> = (1..=20).map(|i| (i as f64).powi(2)).collect();
        let est = estimate_lambda(&y, -2.0, 2.0).unwrap();
        assert!(!est.at_bound);
        let lambda = est.lambda;
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
        assert_eq!(
            estimate_lambda(&[1.0], -2.0, 2.0),
            Err(TransformError::InsufficientData)
        );
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
        for (lo, hi) in [
            (1.0, 0.0),
            (0.5, 0.5),
            (f64::NAN, 1.0),
            (-1.0, f64::INFINITY),
        ] {
            assert_eq!(
                estimate_lambda(&y, lo, hi),
                Err(TransformError::InvalidLambdaRange),
                "range [{lo}, {hi}]"
            );
        }
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
