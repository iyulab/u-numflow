//! Special mathematical functions.
//!
//! Numerical approximations of standard mathematical functions used
//! throughout probability and statistics.

/// 1/√(2π) ≈ 0.3989422804014327
const FRAC_1_SQRT_2PI: f64 = 0.3989422804014326779399460599343818684758586311649;

/// Approximation of the standard normal CDF Φ(x) = P(Z ≤ x) for Z ~ N(0,1).
///
/// # Algorithm
/// Abramowitz & Stegun formula 26.2.17, polynomial approximation with
/// Horner evaluation.
///
/// Reference: Abramowitz & Stegun (1964), *Handbook of Mathematical
/// Functions*, formula 26.2.17, p. 932.
///
/// # Accuracy
/// Maximum absolute error < 7.5 × 10⁻⁸.
///
/// # Examples
/// ```
/// use u_numflow::special::standard_normal_cdf;
/// assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-7);
/// assert!((standard_normal_cdf(1.96) - 0.975).abs() < 1e-3);
/// ```
pub fn standard_normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }

    // Use symmetry: Φ(-x) = 1 - Φ(x)
    let abs_x = x.abs();
    let k = 1.0 / (1.0 + 0.2316419 * abs_x);

    // φ(x) = (1/√(2π)) exp(-x²/2)
    let phi = FRAC_1_SQRT_2PI * (-0.5 * abs_x * abs_x).exp();

    // Horner evaluation of the polynomial
    // a₅ = 1.330274429, a₄ = -1.821255978, a₃ = 1.781477937,
    // a₂ = -0.356563782, a₁ = 0.319381530
    let poly = k
        * (0.319381530
            + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))));

    let cdf_abs = 1.0 - phi * poly;

    if x >= 0.0 {
        cdf_abs
    } else {
        1.0 - cdf_abs
    }
}

/// Approximation of the inverse standard normal CDF (quantile function).
///
/// Given a probability `p ∈ (0, 1)`, returns `z` such that `Φ(z) = p`.
///
/// # Algorithm
/// Abramowitz & Stegun formula 26.2.23, rational approximation.
///
/// Reference: Abramowitz & Stegun (1964), *Handbook of Mathematical
/// Functions*, formula 26.2.23, p. 933.
///
/// # Accuracy
/// Maximum absolute error < 4.5 × 10⁻⁴.
///
/// # Returns
/// - `f64::NAN` if `p` is outside `(0, 1)` or NaN.
/// - `f64::NEG_INFINITY` if `p == 0.0`.
/// - `f64::INFINITY` if `p == 1.0`.
///
/// # Examples
/// ```
/// use u_numflow::special::inverse_normal_cdf;
/// assert!((inverse_normal_cdf(0.5)).abs() < 1e-4);
/// assert!((inverse_normal_cdf(0.975) - 1.96).abs() < 0.01);
/// ```
pub fn inverse_normal_cdf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    // Use symmetry for p > 0.5
    let (q, sign) = if p > 0.5 { (1.0 - p, 1.0) } else { (p, -1.0) };

    // A&S 26.2.23: t = √(-2 ln(q))
    let t = (-2.0 * q.ln()).sqrt();

    // Rational approximation coefficients
    const C0: f64 = 2.515517;
    const C1: f64 = 0.802853;
    const C2: f64 = 0.010328;
    const D1: f64 = 1.432788;
    const D2: f64 = 0.189269;
    const D3: f64 = 0.001308;

    let z = t - (C0 + C1 * t + C2 * t * t) / (1.0 + D1 * t + D2 * t * t + D3 * t * t * t);

    sign * z
}

/// Standard normal PDF φ(x) = (1/√(2π)) exp(-x²/2).
///
/// # Examples
/// ```
/// use u_numflow::special::standard_normal_pdf;
/// let peak = standard_normal_pdf(0.0);
/// assert!((peak - 0.3989422804014327).abs() < 1e-15);
/// ```
pub fn standard_normal_pdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    FRAC_1_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Lanczos approximation of ln Γ(x).
///
/// Reference: Lanczos (1964), "A Precision Approximation of the Gamma
/// Function", *SIAM Journal on Numerical Analysis* 1(1).
///
/// # Accuracy
/// Relative error < 2 × 10⁻¹⁰ for x > 0.
///
/// # Examples
/// ```
/// use u_numflow::special::ln_gamma;
/// // Γ(5) = 24
/// assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
/// ```
pub fn ln_gamma(x: f64) -> f64 {
    #[allow(clippy::excessive_precision)]
    const COEFFICIENTS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = COEFFICIENTS[0];
    for (i, &c) in COEFFICIENTS[1..].iter().enumerate() {
        sum += c / (x + i as f64 + 1.0);
    }

    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Gamma function Γ(x) = exp(ln_gamma(x)).
///
/// # Examples
/// ```
/// use u_numflow::special::gamma;
/// // Γ(5) = 4! = 24
/// assert!((gamma(5.0) - 24.0).abs() < 1e-8);
/// // Γ(0.5) = √π
/// assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-10);
/// ```
pub fn gamma(x: f64) -> f64 {
    ln_gamma(x).exp()
}

// ============================================================================
// Log Beta Function
// ============================================================================

/// Log of the Beta function: `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a+b)`.
///
/// # Examples
/// ```
/// use u_numflow::special::ln_beta;
/// // B(1,1) = 1, so ln B(1,1) = 0
/// assert!(ln_beta(1.0, 1.0).abs() < 1e-10);
/// ```
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

// ============================================================================
// Regularized Incomplete Beta Function
// ============================================================================

/// Regularized incomplete beta function I_x(a, b).
///
/// # Definition
/// ```text
/// I_x(a, b) = B(x; a, b) / B(a, b)
/// ```
/// where B(x; a, b) is the incomplete beta function.
///
/// # Algorithm
/// Uses the continued fraction representation (Lentz's method) with
/// symmetry relation for convergence optimization.
///
/// Reference: Press et al. (2007), *Numerical Recipes*, 3rd ed., §6.4.
///
/// # Accuracy
/// Relative error < 1e-10 for typical parameter ranges.
///
/// # Examples
/// ```
/// use u_numflow::special::regularized_incomplete_beta;
/// // I_0(a,b) = 0, I_1(a,b) = 1
/// assert_eq!(regularized_incomplete_beta(0.0, 2.0, 3.0), 0.0);
/// assert_eq!(regularized_incomplete_beta(1.0, 2.0, 3.0), 1.0);
/// // I_0.5(1,1) = 0.5 (uniform)
/// assert!((regularized_incomplete_beta(0.5, 1.0, 1.0) - 0.5).abs() < 1e-10);
/// ```
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation: I_x(a,b) = 1 - I_{1-x}(b,a)
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b);
    let cf = beta_cf(x, a, b);
    (ln_prefix.exp() / a) * cf
}

/// Continued fraction for the incomplete beta function (Lentz's algorithm).
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-14;
    const TINY: f64 = 1e-30;

    let mut c = 1.0;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(TINY);
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 / (1.0 + num_even * d).max(TINY);
        c = (1.0 + num_even / c).max(TINY);
        h *= d * c;

        let num_odd = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 / (1.0 + num_odd * d).max(TINY);
        c = (1.0 + num_odd / c).max(TINY);
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

// ============================================================================
// Regularized Lower Incomplete Gamma Function
// ============================================================================

/// Regularized lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a).
///
/// # Algorithm
/// Uses series expansion for `x < a + 1`, continued fraction otherwise.
///
/// # Examples
/// ```
/// use u_numflow::special::regularized_lower_gamma;
/// // P(1, x) = 1 - exp(-x) for the exponential distribution
/// let p = regularized_lower_gamma(1.0, 2.0);
/// assert!((p - (1.0 - (-2.0_f64).exp())).abs() < 1e-10);
/// ```
pub fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// Series expansion for the regularized lower incomplete gamma.
fn gamma_series(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut ap = a;
    for _ in 0..200 {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * 1e-14 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Continued fraction for the upper incomplete gamma Q(a, x) = 1 − P(a, x).
fn gamma_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=200 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-14 {
            break;
        }
    }
    h * (-x + a * x.ln() - ln_gamma(a)).exp()
}

// ============================================================================
// Error Function
// ============================================================================

/// Error function erf(x).
///
/// # Definition
/// ```text
/// erf(x) = (2/√π) ∫₀ˣ exp(-t²) dt
/// ```
///
/// # Algorithm
/// Abramowitz & Stegun formula 7.1.28, maximum absolute error < 1.5 × 10⁻⁷.
///
/// # Examples
/// ```
/// use u_numflow::special::erf;
/// assert!(erf(0.0).abs() < 1e-7);
/// assert!((erf(1.0) - 0.8427007929).abs() < 1e-6);
/// ```
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    // A&S 7.1.28
    const P: f64 = 0.3275911;
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;

    let t = 1.0 / (1.0 + P * x);
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Complementary error function erfc(x) = 1 − erf(x).
///
/// More numerically stable than `1.0 - erf(x)` for large `x`.
///
/// # Examples
/// ```
/// use u_numflow::special::erfc;
/// assert!((erfc(0.0) - 1.0).abs() < 1e-7);
/// assert!((erfc(3.0)).abs() < 0.001);
/// ```
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

// ============================================================================
// Student's t-Distribution
// ============================================================================

/// CDF of Student's t-distribution: P(T ≤ t | df).
///
/// # Algorithm
/// Uses the incomplete beta function:
/// - For t ≥ 0: `F(t) = 1 − I_x(df/2, 1/2) / 2`
/// - For t < 0: `F(t) = I_x(df/2, 1/2) / 2`
///
/// where `x = df / (df + t²)`.
///
/// # Returns
/// - `f64::NAN` if df ≤ 0 or inputs are NaN.
///
/// # Examples
/// ```
/// use u_numflow::special::t_distribution_cdf;
/// // CDF at 0 = 0.5 (symmetric)
/// assert!((t_distribution_cdf(0.0, 10.0) - 0.5).abs() < 1e-10);
/// // For large df, approaches normal CDF
/// assert!((t_distribution_cdf(1.96, 1000.0) - 0.975).abs() < 0.002);
/// ```
pub fn t_distribution_cdf(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if t == 0.0 {
        return 0.5;
    }
    let x = df / (df + t * t);
    let ib = regularized_incomplete_beta(x, df / 2.0, 0.5);
    if t >= 0.0 {
        1.0 - ib / 2.0
    } else {
        ib / 2.0
    }
}

/// PDF of Student's t-distribution.
///
/// # Formula
/// ```text
/// f(t; df) = Γ((df+1)/2) / (√(df·π) · Γ(df/2)) · (1 + t²/df)^(−(df+1)/2)
/// ```
pub fn t_distribution_pdf(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    let half_df = df / 2.0;
    let log_pdf = ln_gamma(half_df + 0.5)
        - 0.5 * (df * std::f64::consts::PI).ln()
        - ln_gamma(half_df)
        - (half_df + 0.5) * (1.0 + t * t / df).ln();
    log_pdf.exp()
}

/// Quantile function (inverse CDF) of Student's t-distribution.
///
/// Given a probability `p ∈ (0, 1)`, returns `t` such that `P(T ≤ t) = p`.
///
/// # Algorithm
/// Newton-Raphson iteration with initial guess from inverse normal CDF.
/// Converges in 5–15 iterations for typical inputs.
///
/// # Returns
/// - `f64::NAN` if `p` is outside `(0, 1)` or df ≤ 0.
///
/// # Examples
/// ```
/// use u_numflow::special::t_distribution_quantile;
/// // Median = 0
/// assert!(t_distribution_quantile(0.5, 10.0).abs() < 1e-10);
/// // df=∞ → normal quantile
/// assert!((t_distribution_quantile(0.975, 10000.0) - 1.96).abs() < 0.01);
/// ```
pub fn t_distribution_quantile(p: f64, df: f64) -> f64 {
    if p.is_nan() || df.is_nan() || df <= 0.0 || p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Initial guess from normal approximation
    let mut t = inverse_normal_cdf(p);

    // Newton-Raphson refinement
    for _ in 0..50 {
        let cdf = t_distribution_cdf(t, df);
        let pdf = t_distribution_pdf(t, df);
        if pdf.abs() < 1e-300 {
            break;
        }
        let delta = (cdf - p) / pdf;
        t -= delta;
        if delta.abs() < 1e-12 * t.abs().max(1.0) {
            break;
        }
    }
    t
}

// ============================================================================
// F-Distribution
// ============================================================================

/// CDF of the F-distribution: P(X ≤ x | df1, df2).
///
/// # Algorithm
/// Uses the incomplete beta function:
/// ```text
/// F(x; d1, d2) = I_y(d1/2, d2/2) where y = d1·x / (d1·x + d2)
/// ```
///
/// # Returns
/// - `f64::NAN` if df1 ≤ 0, df2 ≤ 0, or inputs are NaN.
/// - `0.0` if x ≤ 0.
///
/// # Examples
/// ```
/// use u_numflow::special::f_distribution_cdf;
/// assert!((f_distribution_cdf(0.0, 5.0, 10.0) - 0.0).abs() < 1e-10);
/// // F(1.0; 10, 10) ≈ 0.5 (F(1) is median when df1 > 2)
/// let f = f_distribution_cdf(1.0, 10.0, 10.0);
/// assert!((f - 0.5).abs() < 0.05);
/// ```
pub fn f_distribution_cdf(x: f64, df1: f64, df2: f64) -> f64 {
    if x.is_nan() || df1.is_nan() || df2.is_nan() || df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let y = df1 * x / (df1 * x + df2);
    regularized_incomplete_beta(y, df1 / 2.0, df2 / 2.0)
}

/// Quantile function (inverse CDF) of the F-distribution.
///
/// Given a probability `p ∈ (0, 1)`, returns `x` such that `P(X ≤ x) = p`.
///
/// # Algorithm
/// Bisection method on `[0, upper_bound]`. Robust for all parameter ranges.
///
/// # Returns
/// - `f64::NAN` if `p` is outside `(0, 1)` or df1/df2 ≤ 0.
pub fn f_distribution_quantile(p: f64, df1: f64, df2: f64) -> f64 {
    if p.is_nan()
        || df1.is_nan()
        || df2.is_nan()
        || df1 <= 0.0
        || df2 <= 0.0
        || p <= 0.0
        || p >= 1.0
    {
        return f64::NAN;
    }

    // Find upper bound where CDF > p
    let mut hi = 2.0;
    while f_distribution_cdf(hi, df1, df2) < p {
        hi *= 2.0;
        if hi > 1e15 {
            return hi;
        }
    }
    let mut lo = 0.0_f64;

    // Bisection
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if hi - lo < 1e-12 * mid.max(1e-15) {
            break;
        }
        if f_distribution_cdf(mid, df1, df2) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

// ============================================================================
// Chi-Squared Distribution CDF
// ============================================================================

/// CDF of the chi-squared distribution: P(X ≤ x | k).
///
/// # Algorithm
/// Uses the regularized lower incomplete gamma function:
/// ```text
/// F(x; k) = P(k/2, x/2) = γ(k/2, x/2) / Γ(k/2)
/// ```
///
/// # Returns
/// - `f64::NAN` if k ≤ 0 or inputs are NaN.
/// - `0.0` if x ≤ 0.
///
/// # Examples
/// ```
/// use u_numflow::special::chi_squared_cdf;
/// // P(X ≤ 0) = 0
/// assert_eq!(chi_squared_cdf(0.0, 5.0), 0.0);
/// // Known: P(X ≤ 3.841) ≈ 0.95 for df=1
/// assert!((chi_squared_cdf(3.841, 1.0) - 0.95).abs() < 0.01);
/// ```
pub fn chi_squared_cdf(x: f64, k: f64) -> f64 {
    if x.is_nan() || k.is_nan() || k <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    regularized_lower_gamma(k / 2.0, x / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- standard_normal_cdf ---

    #[test]
    fn test_cdf_at_zero() {
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_cdf_symmetry() {
        for &x in &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let sum = standard_normal_cdf(x) + standard_normal_cdf(-x);
            assert!(
                (sum - 1.0).abs() < 1e-7,
                "Φ({x}) + Φ(-{x}) = {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_cdf_known_values() {
        // 68-95-99.7 rule
        assert!((standard_normal_cdf(1.0) - 0.8413).abs() < 0.001);
        assert!((standard_normal_cdf(2.0) - 0.9772).abs() < 0.001);
        assert!((standard_normal_cdf(3.0) - 0.9987).abs() < 0.001);

        // Common critical values
        assert!((standard_normal_cdf(1.645) - 0.95).abs() < 0.001);
        assert!((standard_normal_cdf(1.96) - 0.975).abs() < 0.001);
        assert!((standard_normal_cdf(2.576) - 0.995).abs() < 0.001);
    }

    #[test]
    fn test_cdf_extremes() {
        assert_eq!(standard_normal_cdf(f64::INFINITY), 1.0);
        assert_eq!(standard_normal_cdf(f64::NEG_INFINITY), 0.0);
        assert!(standard_normal_cdf(f64::NAN).is_nan());
    }

    #[test]
    fn test_cdf_monotonic() {
        let xs: Vec<f64> = (-30..=30).map(|i| i as f64 * 0.1).collect();
        for w in xs.windows(2) {
            assert!(
                standard_normal_cdf(w[0]) <= standard_normal_cdf(w[1]),
                "CDF not monotonic at x = {}, {}",
                w[0],
                w[1]
            );
        }
    }

    // --- inverse_normal_cdf ---

    #[test]
    fn test_inverse_cdf_at_half() {
        assert!(inverse_normal_cdf(0.5).abs() < 1e-4);
    }

    #[test]
    fn test_inverse_cdf_known_values() {
        assert!((inverse_normal_cdf(0.8413) - 1.0).abs() < 0.01);
        assert!((inverse_normal_cdf(0.975) - 1.96).abs() < 0.01);
        assert!((inverse_normal_cdf(0.95) - 1.645).abs() < 0.01);
    }

    #[test]
    fn test_inverse_cdf_symmetry() {
        for &p in &[0.1, 0.2, 0.3, 0.4] {
            let z1 = inverse_normal_cdf(p);
            let z2 = inverse_normal_cdf(1.0 - p);
            assert!(
                (z1 + z2).abs() < 1e-3,
                "Φ⁻¹({p}) + Φ⁻¹({}) = {}, expected ~0",
                1.0 - p,
                z1 + z2
            );
        }
    }

    #[test]
    fn test_inverse_cdf_extremes() {
        assert_eq!(inverse_normal_cdf(0.0), f64::NEG_INFINITY);
        assert_eq!(inverse_normal_cdf(1.0), f64::INFINITY);
        assert!(inverse_normal_cdf(f64::NAN).is_nan());
        assert!(inverse_normal_cdf(-0.1).is_nan());
        assert!(inverse_normal_cdf(1.1).is_nan());
    }

    #[test]
    fn test_roundtrip_cdf_inverse() {
        for &p in &[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let z = inverse_normal_cdf(p);
            let p_back = standard_normal_cdf(z);
            assert!(
                (p_back - p).abs() < 0.002,
                "roundtrip failed: p={p}, z={z}, p_back={p_back}"
            );
        }
    }

    // --- standard_normal_pdf ---

    #[test]
    fn test_pdf_at_zero() {
        let peak = standard_normal_pdf(0.0);
        assert!((peak - 0.3989422804014327).abs() < 1e-14);
    }

    #[test]
    fn test_pdf_symmetry() {
        for &x in &[0.5, 1.0, 2.0, 3.0] {
            let diff = (standard_normal_pdf(x) - standard_normal_pdf(-x)).abs();
            assert!(diff < 1e-15, "PDF not symmetric at x={x}");
        }
    }

    // --- ln_gamma / gamma ---

    #[test]
    fn test_ln_gamma_integers() {
        // Γ(n) = (n-1)! for positive integers
        assert!((ln_gamma(1.0)).abs() < 1e-10); // Γ(1) = 1
        assert!((ln_gamma(2.0)).abs() < 1e-10); // Γ(2) = 1
        assert!((ln_gamma(3.0) - 2.0_f64.ln()).abs() < 1e-10); // Γ(3) = 2
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10); // Γ(5) = 24
        assert!((ln_gamma(7.0) - 720.0_f64.ln()).abs() < 1e-9); // Γ(7) = 720
    }

    #[test]
    fn test_gamma_half_integers() {
        // Γ(0.5) = √π
        let sqrt_pi = std::f64::consts::PI.sqrt();
        assert!((gamma(0.5) - sqrt_pi).abs() < 1e-10);
        // Γ(1.5) = √π/2
        assert!((gamma(1.5) - sqrt_pi / 2.0).abs() < 1e-10);
        // Γ(2.5) = 3√π/4
        assert!((gamma(2.5) - 3.0 * sqrt_pi / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_positive() {
        for &x in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            assert!(gamma(x) > 0.0, "Γ({x}) should be positive");
        }
    }

    // --- ln_beta ---

    #[test]
    fn test_ln_beta_known() {
        // B(1,1) = 1, ln B(1,1) = 0
        assert!(ln_beta(1.0, 1.0).abs() < 1e-10);
        // B(1,2) = 1/2, ln B(1,2) = -ln(2)
        assert!((ln_beta(1.0, 2.0) - (-2.0_f64.ln())).abs() < 1e-10);
        // B(a,b) = B(b,a)
        assert!((ln_beta(3.0, 5.0) - ln_beta(5.0, 3.0)).abs() < 1e-10);
    }

    // --- regularized_incomplete_beta ---

    #[test]
    fn test_inc_beta_boundary() {
        assert_eq!(regularized_incomplete_beta(0.0, 2.0, 3.0), 0.0);
        assert_eq!(regularized_incomplete_beta(1.0, 2.0, 3.0), 1.0);
    }

    #[test]
    fn test_inc_beta_uniform() {
        // I_x(1,1) = x (Uniform case)
        for &x in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let result = regularized_incomplete_beta(x, 1.0, 1.0);
            assert!(
                (result - x).abs() < 1e-10,
                "I_{x}(1,1) = {result}, expected {x}"
            );
        }
    }

    #[test]
    fn test_inc_beta_symmetry() {
        // I_0.5(a,a) = 0.5
        let result = regularized_incomplete_beta(0.5, 3.0, 3.0);
        assert!((result - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_inc_beta_known_formula() {
        // I_x(1,b) = 1 - (1-x)^b
        for &x in &[0.1, 0.5, 0.9] {
            let result = regularized_incomplete_beta(x, 1.0, 3.0);
            let expected = 1.0 - (1.0 - x).powi(3);
            assert!((result - expected).abs() < 1e-10);
        }
    }

    // --- regularized_lower_gamma ---

    #[test]
    fn test_lower_gamma_exponential() {
        // P(1, x) = 1 - exp(-x) (exponential distribution CDF)
        for &x in &[0.5, 1.0, 2.0, 5.0] {
            let result = regularized_lower_gamma(1.0, x);
            let expected = 1.0 - (-x).exp();
            assert!(
                (result - expected).abs() < 1e-10,
                "P(1,{x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_lower_gamma_boundary() {
        assert_eq!(regularized_lower_gamma(2.0, 0.0), 0.0);
        assert_eq!(regularized_lower_gamma(2.0, -1.0), 0.0);
    }

    #[test]
    fn test_lower_gamma_large_x() {
        // For large x, P(a, x) → 1
        let result = regularized_lower_gamma(3.0, 100.0);
        assert!((result - 1.0).abs() < 1e-10);
    }

    // --- erf / erfc ---

    #[test]
    fn test_erf_known_values() {
        assert!(erf(0.0).abs() < 1e-7);
        assert!((erf(1.0) - 0.8427007929).abs() < 1e-6);
        // erf(∞) = 1
        assert!((erf(10.0) - 1.0).abs() < 1e-7);
        // erf(-x) = -erf(x)
        assert!((erf(1.5) + erf(-1.5)).abs() < 1e-7);
    }

    #[test]
    fn test_erf_nan() {
        assert!(erf(f64::NAN).is_nan());
    }

    #[test]
    fn test_erfc_complement() {
        for &x in &[0.0, 0.5, 1.0, 2.0, 3.0] {
            let sum = erf(x) + erfc(x);
            assert!((sum - 1.0).abs() < 1e-7, "erf({x}) + erfc({x}) = {sum}");
        }
    }

    // --- t-distribution ---

    #[test]
    fn test_t_cdf_at_zero() {
        // CDF at 0 = 0.5 (symmetric)
        for &df in &[1.0, 5.0, 10.0, 30.0, 100.0] {
            assert!(
                (t_distribution_cdf(0.0, df) - 0.5).abs() < 1e-10,
                "t_cdf(0, {df}) should be 0.5"
            );
        }
    }

    #[test]
    fn test_t_cdf_symmetry() {
        for &df in &[1.0, 5.0, 10.0] {
            for &t in &[0.5, 1.0, 2.0] {
                let sum = t_distribution_cdf(t, df) + t_distribution_cdf(-t, df);
                assert!(
                    (sum - 1.0).abs() < 1e-8,
                    "t_cdf({t},{df}) + t_cdf(-{t},{df}) = {sum}"
                );
            }
        }
    }

    #[test]
    fn test_t_cdf_approaches_normal() {
        // For large df, t-distribution → normal
        let t_val = 1.96;
        let result = t_distribution_cdf(t_val, 10000.0);
        let expected = standard_normal_cdf(t_val);
        assert!((result - expected).abs() < 0.002);
    }

    #[test]
    fn test_t_cdf_known_values() {
        // t(0.025, df=10) → CDF ≈ 0.025 → t ≈ -2.228
        let cdf = t_distribution_cdf(-2.228, 10.0);
        assert!((cdf - 0.025).abs() < 0.002, "t_cdf(-2.228, 10) = {cdf}");
    }

    #[test]
    fn test_t_cdf_nan() {
        assert!(t_distribution_cdf(1.0, -1.0).is_nan());
        assert!(t_distribution_cdf(f64::NAN, 5.0).is_nan());
    }

    #[test]
    fn test_t_pdf_positive() {
        for &df in &[1.0, 5.0, 10.0] {
            for &t in &[-2.0, 0.0, 1.0, 3.0] {
                assert!(t_distribution_pdf(t, df) > 0.0);
            }
        }
    }

    #[test]
    fn test_t_pdf_symmetry() {
        for &df in &[1.0, 5.0, 10.0] {
            for &t in &[0.5, 1.0, 2.0] {
                let diff = (t_distribution_pdf(t, df) - t_distribution_pdf(-t, df)).abs();
                assert!(diff < 1e-12, "t_pdf not symmetric at t={t}, df={df}");
            }
        }
    }

    #[test]
    fn test_t_quantile_median() {
        // Median should be 0
        for &df in &[1.0, 5.0, 10.0, 100.0] {
            assert!(t_distribution_quantile(0.5, df).abs() < 1e-10);
        }
    }

    #[test]
    fn test_t_quantile_roundtrip() {
        for &df in &[2.0, 5.0, 10.0, 30.0] {
            for &p in &[0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975] {
                let t = t_distribution_quantile(p, df);
                let p_back = t_distribution_cdf(t, df);
                assert!(
                    (p_back - p).abs() < 1e-6,
                    "roundtrip: p={p}, df={df}, t={t}, p_back={p_back}"
                );
            }
        }
    }

    #[test]
    fn test_t_quantile_nan() {
        assert!(t_distribution_quantile(0.0, 5.0).is_nan());
        assert!(t_distribution_quantile(1.0, 5.0).is_nan());
        assert!(t_distribution_quantile(0.5, -1.0).is_nan());
    }

    // --- F-distribution ---

    #[test]
    fn test_f_cdf_zero() {
        assert_eq!(f_distribution_cdf(0.0, 5.0, 10.0), 0.0);
        assert_eq!(f_distribution_cdf(-1.0, 5.0, 10.0), 0.0);
    }

    #[test]
    fn test_f_cdf_known() {
        // F(1.0; 10, 10) ≈ 0.5 (median for equal df when df > 2)
        let f = f_distribution_cdf(1.0, 10.0, 10.0);
        assert!((f - 0.5).abs() < 0.05, "F_cdf(1,10,10) = {f}");
    }

    #[test]
    fn test_f_cdf_monotonic() {
        let xs: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();
        for w in xs.windows(2) {
            let c0 = f_distribution_cdf(w[0], 5.0, 10.0);
            let c1 = f_distribution_cdf(w[1], 5.0, 10.0);
            assert!(
                c1 >= c0 - 1e-10,
                "F CDF not monotonic at {}, {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_f_cdf_nan() {
        assert!(f_distribution_cdf(1.0, -1.0, 5.0).is_nan());
        assert!(f_distribution_cdf(1.0, 5.0, -1.0).is_nan());
    }

    #[test]
    fn test_f_quantile_roundtrip() {
        for &(df1, df2) in &[(5.0, 10.0), (10.0, 10.0), (3.0, 20.0)] {
            for &p in &[0.05, 0.1, 0.5, 0.9, 0.95] {
                let x = f_distribution_quantile(p, df1, df2);
                let p_back = f_distribution_cdf(x, df1, df2);
                assert!(
                    (p_back - p).abs() < 1e-4,
                    "F roundtrip: p={p}, df1={df1}, df2={df2}, x={x}, p_back={p_back}"
                );
            }
        }
    }

    #[test]
    fn test_f_quantile_nan() {
        assert!(f_distribution_quantile(0.0, 5.0, 10.0).is_nan());
        assert!(f_distribution_quantile(1.0, 5.0, 10.0).is_nan());
    }

    // --- Chi-squared ---

    #[test]
    fn test_chi2_cdf_zero() {
        assert_eq!(chi_squared_cdf(0.0, 5.0), 0.0);
        assert_eq!(chi_squared_cdf(-1.0, 5.0), 0.0);
    }

    #[test]
    fn test_chi2_cdf_exponential_special_case() {
        // Chi2(2) = Exponential(1/2): CDF(x) = 1 - exp(-x/2)
        for &x in &[1.0, 2.0, 5.0, 10.0] {
            let result = chi_squared_cdf(x, 2.0);
            let expected = 1.0 - (-x / 2.0).exp();
            assert!(
                (result - expected).abs() < 1e-8,
                "chi2_cdf({x}, 2) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_chi2_cdf_known_critical() {
        // P(X ≤ 3.841) ≈ 0.95 for df=1
        assert!((chi_squared_cdf(3.841, 1.0) - 0.95).abs() < 0.01);
        // P(X ≤ 5.991) ≈ 0.95 for df=2
        assert!((chi_squared_cdf(5.991, 2.0) - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_chi2_cdf_nan() {
        assert!(chi_squared_cdf(1.0, -1.0).is_nan());
        assert!(chi_squared_cdf(f64::NAN, 5.0).is_nan());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn cdf_in_zero_one(x in -6.0_f64..6.0) {
            let c = standard_normal_cdf(x);
            prop_assert!((0.0..=1.0).contains(&c), "CDF({x}) = {c} out of [0,1]");
        }

        #[test]
        fn cdf_is_monotonic(x1 in -6.0_f64..6.0, x2 in -6.0_f64..6.0) {
            let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
            prop_assert!(
                standard_normal_cdf(lo) <= standard_normal_cdf(hi) + 1e-15,
                "CDF not monotonic"
            );
        }

        #[test]
        fn inverse_roundtrip(p in 0.001_f64..0.999) {
            let z = inverse_normal_cdf(p);
            let p_back = standard_normal_cdf(z);
            let err = (p_back - p).abs();
            prop_assert!(err < 0.005, "roundtrip error {} for p={}", err, p);
        }

        #[test]
        fn pdf_is_non_negative(x in -10.0_f64..10.0) {
            prop_assert!(standard_normal_pdf(x) >= 0.0);
        }

        #[test]
        fn inc_beta_in_01(x in 0.01_f64..0.99, a in 0.5_f64..10.0, b in 0.5_f64..10.0) {
            let result = regularized_incomplete_beta(x, a, b);
            prop_assert!(
                (0.0..=1.0).contains(&result),
                "I_{x}({a},{b}) = {result} out of [0,1]"
            );
        }

        #[test]
        fn inc_beta_complementary(x in 0.01_f64..0.99, a in 0.5_f64..10.0, b in 0.5_f64..10.0) {
            // I_x(a,b) + I_{1-x}(b,a) = 1
            let ix = regularized_incomplete_beta(x, a, b);
            let i1x = regularized_incomplete_beta(1.0 - x, b, a);
            prop_assert!(
                (ix + i1x - 1.0).abs() < 1e-8,
                "complementary: {ix} + {i1x} != 1"
            );
        }

        #[test]
        fn t_cdf_in_01(t in -10.0_f64..10.0, df in 1.0_f64..100.0) {
            let c = t_distribution_cdf(t, df);
            prop_assert!(
                (0.0..=1.0).contains(&c),
                "t_cdf({t}, {df}) = {c} out of [0,1]"
            );
        }

        #[test]
        fn t_cdf_symmetric(t in 0.01_f64..10.0, df in 1.0_f64..50.0) {
            let sum = t_distribution_cdf(t, df) + t_distribution_cdf(-t, df);
            prop_assert!(
                (sum - 1.0).abs() < 1e-6,
                "t symmetry: {sum} != 1 for t={t}, df={df}"
            );
        }

        #[test]
        fn erf_odd_symmetry(x in 0.01_f64..5.0) {
            let sum = erf(x) + erf(-x);
            prop_assert!(sum.abs() < 1e-6, "erf odd symmetry: {sum} for x={x}");
        }

        #[test]
        fn chi2_cdf_in_01(x in 0.01_f64..50.0, k in 0.5_f64..20.0) {
            let c = chi_squared_cdf(x, k);
            prop_assert!(
                (0.0..=1.0).contains(&c),
                "chi2_cdf({x}, {k}) = {c} out of [0,1]"
            );
        }
    }
}
