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
    }
}
