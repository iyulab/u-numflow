//! Special mathematical functions.
//!
//! Numerical approximations of standard mathematical functions used
//! throughout probability and statistics.

use std::f64::consts::FRAC_1_SQRT_2;

/// 1/√(2π) ≈ 0.3989422804014327
const FRAC_1_SQRT_2PI: f64 = 0.3989422804014326779399460599343818684758586311649;

/// Standard normal CDF Φ(x) = P(Z ≤ x) for Z ~ N(0,1).
///
/// # Algorithm
/// `Φ(x) = erfc(−x/√2) / 2`, with [`erfc`] computed directly (never as
/// `1 − erf`), so the lower tail keeps its relative precision.
///
/// # Accuracy
/// Relative error of a few 10⁻¹⁶ wherever the result is representable,
/// including the lower tail (`x ≪ 0`, down to `x ≈ −38`): the tail is formed
/// from `x` itself, not from `x/√2` rounded to a double, whose error is
/// multiplied by `x²` in the result. For the upper tail use
/// [`standard_normal_sf`]: `1 − Φ(x)` cannot resolve values below ~10⁻¹⁶.
///
/// # Examples
/// ```
/// use u_numflow::special::standard_normal_cdf;
/// assert_eq!(standard_normal_cdf(0.0), 0.5);
/// assert!((standard_normal_cdf(1.96) - 0.9750021048517795).abs() < 1e-15);
/// // Lower tail keeps relative precision: Φ(−6) = 9.865876450376981e−10
/// assert!((standard_normal_cdf(-6.0) / 9.865876450376981e-10 - 1.0).abs() < 1e-15);
/// assert_eq!(standard_normal_cdf(f64::INFINITY), 1.0);
/// assert_eq!(standard_normal_cdf(f64::NEG_INFINITY), 0.0);
/// ```
pub fn standard_normal_cdf(x: f64) -> f64 {
    if x <= -TAIL_FROM {
        return standard_normal_upper_tail(-x);
    }
    0.5 * erfc(-x * FRAC_1_SQRT_2)
}

/// Standard normal survival function `1 − Φ(x) = P(Z > x)` for Z ~ N(0,1).
///
/// Computed directly as `erfc(x/√2) / 2`, so the upper tail keeps its
/// relative precision — the quantity behind PPM defect rates, one-sided
/// p-values and tail probabilities, where `1.0 - standard_normal_cdf(x)`
/// has no significant digits left.
///
/// # Accuracy
/// Relative error of a few 10⁻¹⁶ wherever the result is representable
/// (up to `x ≈ 38`, beyond which it underflows to 0): the tail is formed from
/// `x` itself, not from `x/√2` rounded to a double, whose error is multiplied
/// by `x²` in the result (10⁻¹³ at `x = 35`).
///
/// # Examples
/// ```
/// use u_numflow::special::standard_normal_sf;
/// assert_eq!(standard_normal_sf(0.0), 0.5);
/// // P(Z > 6) = 9.865876450376981e−10
/// assert!((standard_normal_sf(6.0) / 9.865876450376981e-10 - 1.0).abs() < 1e-15);
/// assert_eq!(standard_normal_sf(f64::INFINITY), 0.0);
/// ```
pub fn standard_normal_sf(x: f64) -> f64 {
    if x >= TAIL_FROM {
        return standard_normal_upper_tail(x);
    }
    0.5 * erfc(x * FRAC_1_SQRT_2)
}

/// `1.25·√2`: from here `x/√2` is in the range where [`erfc`] is an exponential.
const TAIL_FROM: f64 = 1.767_766_952_966_368_9;

/// `exp(−x²/2)` with the square taken exactly: `x²/2` has the rounding error of
/// a product that large, and the exponential multiplies it by `x²/2` again
/// (10⁻¹³ at `x = 35`). `x = z + (x − z)` with `z` the 32 high bits of `x`,
/// whose square is exactly representable, leaves the rounded part to a factor
/// within 10⁻³ of 1 (fdlibm `s_erf.c`, note 1).
fn exp_neg_half_square(x: f64) -> f64 {
    let ax = x.abs();
    if ax.is_infinite() {
        return 0.0;
    }
    let z = f64::from_bits(ax.to_bits() & 0xFFFF_FFFF_0000_0000);
    (-0.5 * z * z).exp() * (-0.5 * (ax - z) * (ax + z)).exp()
}

/// `1 − Φ(x) = ½·erfc(x/√2)` for `x ≥ 1.25·√2`, formed without subtraction and
/// without rounding `x/√2`: the exponent `−x²/2 − 0.5625 + R/S` of fdlibm's
/// `erfc` is built from `x` (`z = x` truncated to 32 bits, so `z²/2 + 0.5625`
/// is exact), so the result keeps a few ulp of relative accuracy to the
/// underflow at `x ≈ 38.5`.
fn standard_normal_upper_tail(x: f64) -> f64 {
    let t = x * FRAC_1_SQRT_2;
    if t >= 28.0 {
        return 0.0;
    }
    let s = 2.0 / (x * x); // 1/t²
    let (r, big_s) = if t < 1.0 / 0.35 {
        (horner(s, &RA), 1.0 + s * horner(s, &SA))
    } else {
        (horner(s, &RB), 1.0 + s * horner(s, &SB))
    };
    let z = f64::from_bits(x.to_bits() & 0xFFFF_FFFF_0000_0000);
    0.5 * (-0.5 * z * z - 0.5625).exp() * (-0.5 * (x - z) * (x + z) + r / big_s).exp() / t
}

/// Inverse of the standard normal CDF (quantile function).
///
/// Given a probability `p ∈ (0, 1)`, returns `z` such that `Φ(z) = p`.
///
/// # Algorithm
/// Wichura's algorithm AS 241 (PPND16): three piecewise rational
/// approximations in `q = p − ½` (centre) and `r = √(−ln min(p, 1−p))`
/// (tails).
///
/// Reference: Wichura, M. J. (1988), "Algorithm AS 241: The Percentage
/// Points of the Normal Distribution", *Applied Statistics* 37(3), 477–484.
///
/// # Accuracy
/// About 1 part in 10¹⁶ in `z`. For `p` near 1 the argument `p` itself
/// cannot carry an upper tail below ~10⁻¹⁶; use [`inverse_normal_sf`] with
/// the tail probability instead.
///
/// # Returns
/// - `f64::NAN` if `p` is outside `[0, 1]` or NaN.
/// - `f64::NEG_INFINITY` if `p == 0.0`.
/// - `f64::INFINITY` if `p == 1.0`.
///
/// # Examples
/// ```
/// use u_numflow::special::inverse_normal_cdf;
/// assert_eq!(inverse_normal_cdf(0.5), 0.0);
/// assert!((inverse_normal_cdf(0.975) - 1.959963984540054).abs() < 1e-14);
/// assert!((inverse_normal_cdf(0.025) + 1.959963984540054).abs() < 1e-14);
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

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q
            * (((((((r * 2509.0809287301227 + 33430.57558358813) * r + 67265.7709270087) * r
                + 45921.95393154987)
                * r
                + 13731.69376550946)
                * r
                + 1971.5909503065513)
                * r
                + 133.14166789178438)
                * r
                + 3.3871328727963665)
            / (((((((r * 5226.495278852854 + 28729.085735721943) * r + 39307.89580009271) * r
                + 21213.794301586597)
                * r
                + 5394.196021424751)
                * r
                + 687.1870074920579)
                * r
                + 42.31333070160091)
                * r
                + 1.0);
    }

    // Tail: r = √(−ln(tail probability)), tail = min(p, 1 − p).
    let tail = if q < 0.0 { p } else { 1.0 - p };
    let z = as241_tail(tail);
    if q < 0.0 {
        -z
    } else {
        z
    }
}

/// Inverse of the standard normal survival function.
///
/// Given an upper-tail probability `q ∈ (0, 1)`, returns `z` such that
/// `P(Z > z) = q`. Unlike `inverse_normal_cdf(1.0 - q)`, the tail
/// probability is used as given, so `q` down to the smallest positive `f64`
/// maps to its exact deviate — the inverse of [`standard_normal_sf`].
///
/// # Returns
/// - `f64::NAN` if `q` is outside `[0, 1]` or NaN.
/// - `f64::INFINITY` if `q == 0.0`.
/// - `f64::NEG_INFINITY` if `q == 1.0`.
///
/// # Examples
/// ```
/// use u_numflow::special::{inverse_normal_sf, standard_normal_sf};
/// let z = inverse_normal_sf(1e-12);
/// assert!((standard_normal_sf(z) / 1e-12 - 1.0).abs() < 1e-12);
/// ```
pub fn inverse_normal_sf(q: f64) -> f64 {
    // Φ⁻¹ is odd about ½: P(Z > z) = q  ⇔  Φ(−z) = q.
    -inverse_normal_cdf(q)
}

/// AS 241 tail branch: the deviate for a tail probability `tail ∈ (0, 0.075)`.
fn as241_tail(tail: f64) -> f64 {
    let r = (-tail.ln()).sqrt();
    if r <= 5.0 {
        let r = r - 1.6;
        (((((((r * 0.0007745450142783414 + 0.022723844989269184) * r + 0.2417807251774506) * r
            + 1.2704582524523684)
            * r
            + 3.6478483247632045)
            * r
            + 5.769497221460691)
            * r
            + 4.630337846156546)
            * r
            + 1.4234371107496835)
            / (((((((r * 1.0507500716444169e-09 + 0.0005475938084995345) * r
                + 0.015198666563616457)
                * r
                + 0.14810397642748008)
                * r
                + 0.6897673349851)
                * r
                + 1.6763848301838038)
                * r
                + 2.053191626637759)
                * r
                + 1.0)
    } else {
        // Wichura's minimax fit covers 5 < r ≤ 27; the smallest positive
        // subnormal reaches r ≈ 27.3, a negligible extrapolation.
        let r = r - 5.0;
        (((((((r * 2.0103343992922881e-07 + 2.7115555687434876e-05) * r
            + 0.0012426609473880784)
            * r
            + 0.026532189526576124)
            * r
            + 0.29656057182850487)
            * r
            + 1.7848265399172913)
            * r
            + 5.463784911164114)
            * r
            + 6.657904643501103)
            / (((((((r * 2.0442631033899397e-15 + 1.421511758316446e-07) * r
                + 1.8463183175100548e-05)
                * r
                + 0.0007868691311456133)
                * r
                + 0.014875361290850615)
                * r
                + 0.1369298809227358)
                * r
                + 0.599832206555888)
                * r
                + 1.0)
    }
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
    FRAC_1_SQRT_2PI * exp_neg_half_square(x)
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
/// assert!((ln_gamma(5.0) - 24.0f64.ln()).abs() < 1e-10);
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
    let (p, q) = if a < b { (a, b) } else { (b, a) };
    if p > 0.0 && q >= STIRLING_FROM && q.is_finite() {
        return ln_beta_large(p, q);
    }
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// From this argument on `ln Γ` is its Stirling series to a double's last digit.
const STIRLING_FROM: f64 = 10.0;

/// `ln Γ(z) − [(z − ½) ln z − z + ½ ln 2π]` for `z >= 10`: the tail of the
/// Stirling series, `1/(12z) − 1/(360z³) + 1/(1260z⁵) − …`, each term smaller
/// than the last by `1/z²`, the first one left out under 1e-16 of the sum.
fn stirling_correction(z: f64) -> f64 {
    let w = 1.0 / (z * z);
    (1.0 / 12.0
        + w * (-1.0 / 360.0
            + w * (1.0 / 1260.0
                + w * (-1.0 / 1680.0
                    + w * (1.0 / 1188.0 + w * (-691.0 / 360_360.0 + w * (1.0 / 156.0)))))))
        / z
}

/// `ln Γ(h + ½) − ln Γ(h)`: `½ ln h − 1/(8h) + …`, a number of order `ln h` that
/// the difference of two `ln Γ` of order `h ln h` cannot give once `h` is large
/// (a `ln Γ` of 1e10 has a last digit of 3e-5). For `h >= 10` the Stirling series
/// of both are subtracted first: `h ln(1 + 1/(2h)) + ½ ln h − ½ + c(h+½) − c(h)`.
pub(crate) fn ln_gamma_half_step(h: f64) -> f64 {
    if h >= STIRLING_FROM && h.is_finite() {
        return h * (0.5 / h).ln_1p() + 0.5 * h.ln() - 0.5 + stirling_correction(h + 0.5)
            - stirling_correction(h);
    }
    ln_gamma(h + 0.5) - ln_gamma(h)
}

/// The deviance term `a ln(a/z) + z − a` (Loader 2000, "Fast and accurate
/// computation of binomial probabilities"), which is `>= 0` and small when `z` is
/// near `a`: there the direct form subtracts numbers of order `a` to leave one of
/// order `(a − z)²/a`, so a series in `(a − z)/(a + z)` is summed instead.
fn deviance_term(a: f64, z: f64) -> f64 {
    deviance_near_mean(a, a - z, z).unwrap_or_else(|| a * (a / z).ln() + z - a)
}

/// The series of [`deviance_term`] for a `z` within a tenth of `a`, from the gap
/// `a − z` given on its own (a caller may have it to more digits than the
/// difference of the two rounded numbers); `None` farther out, where the direct
/// form loses nothing it needs.
fn deviance_near_mean(a: f64, gap: f64, z: f64) -> Option<f64> {
    if gap.abs() >= 0.1 * (a + z) {
        return None;
    }
    let v = gap / (a + z);
    let mut sum = gap * v;
    if sum.abs() < f64::MIN_POSITIVE {
        return Some(sum);
    }
    let mut term = 2.0 * a * v;
    let v2 = v * v;
    for j in 1..1000 {
        term *= v2;
        let next = sum + term / (2 * j + 1) as f64;
        if next == sum {
            return Some(next);
        }
        sum = next;
    }
    Some(sum)
}

/// `ln(z^a e^−z / Γ(a))` -- the kernel of the gamma density, whose terms
/// `a ln z`, `z` and `ln Γ(a)` are of order `a ln a` and cancel to the order of
/// `ln a` near the mode. For `a >= 10` it is `−deviance + ½ ln a − ½ ln 2π −
/// c(a)` with `c` the Stirling tail, which keeps every digit at any shape.
pub(crate) fn ln_gamma_kernel(a: f64, z: f64) -> f64 {
    if a >= STIRLING_FROM && a.is_finite() && z > 0.0 && z.is_finite() {
        return -deviance_term(a, z) + 0.5 * a.ln()
            - 0.5 * (2.0 * std::f64::consts::PI).ln()
            - stirling_correction(a);
    }
    a * z.ln() - z - ln_gamma(a)
}

/// `ln B(p, q)` for `0 < p <= q` and `q >= 10`, without the cancellation of
/// `ln Γ(p) + ln Γ(q) − ln Γ(p + q)`: the three terms are of order `q ln q`, so
/// when `q` is large next to `p` their rounding (a `q` of 1e10 has a `ln Γ` whose
/// last digit is 3e-5) leaves the sum, which is of order `p ln q`, with no digit
/// at all. Here the terms that cancel are subtracted before they are added up:
/// the Stirling series of the large ones differ by `ln_1p` of a ratio, which
/// keeps its digits at any size.
fn ln_beta_large(p: f64, q: f64) -> f64 {
    let sum = p + q;
    let half_ln_two_pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    if p >= STIRLING_FROM {
        // All three by Stirling: (p-½)ln p + (q-½)ln q − (p+q-½)ln(p+q)
        //   = −(p-½) ln(1 + q/p) − q ln(1 + p/q) − ½ ln q.
        return half_ln_two_pi - (p - 0.5) * (q / p).ln_1p() - q * (p / q).ln_1p() - 0.5 * q.ln()
            + stirling_correction(p)
            + stirling_correction(q)
            - stirling_correction(sum);
    }
    // ln Γ(p) exactly, and ln Γ(q) − ln Γ(p+q) by Stirling:
    //   = −(q-½) ln(1 + p/q) − p ln(p+q) + p + c(q) − c(p+q).
    ln_gamma(p) - (q - 0.5) * (p / q).ln_1p() - p * sum.ln() + p + stirling_correction(q)
        - stirling_correction(sum)
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
/// Relative error of the value in the smaller tail, measured against 60-digit
/// references over 541 points (shapes 0.05 to 9·10⁶, probabilities 10⁻¹² to
/// 1 − 10⁻⁸), and over 84 points at shapes 10⁷ to 10¹⁴:
///
/// - below 6·10⁻¹⁴ where one shape is small (`min(a, b) ≤ 3`), at every shape
///   tested up to 5·10⁹ (the Student t with 10¹⁰ degrees of freedom);
/// - below 2·10⁻¹³ with both shapes up to 10⁵, below 2·10⁻¹² to 10⁷, and below
///   10⁻¹⁰ to 10¹¹ (the long continued fraction rounds `~3·10⁻²¹ · shape`).
///
/// A large `a` against a small `b` (`a ≥ 15`, `a ≥ 8b`, `b³ ≤ a²`) with `x` at
/// or past the middle is evaluated from `1 − x` alone by an asymptotic expansion
/// (Didonato & Morris, ACM TOMS 708): a continued fraction handed an `x` that is
/// 1 minus a small number, rounded, loses `≈ 3·10⁻¹⁶ · a` of the result (6·10⁻⁷
/// at 5·10⁹). With both shapes ≥ 10 the prefix `x^a (1 − x)^b / B(a, b)` is
/// taken as a deviation from the mean `a / (a + b)` (Stirling's series for the
/// three gammas and the deviance of Loader 2000) rather than as three terms of
/// order `a ln a` that cancel to the order of `ln a`. The normal approximation
/// (with its skewness term, off by `1.2·10⁻⁹` of a far tail at 10¹²) takes over
/// at a shape of 3·10¹¹ in both parameters, where it overtakes the fraction.
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

    beta_cdf_pair(x, 1.0 - x, a, b)
}

/// `I_x(a, b)` for `x` and its complement `y = 1 - x` given separately (the
/// caller has them exactly, which `1 - x` of a rounded `x` is not).
fn beta_cdf_pair(x: f64, y: f64, a: f64, b: f64) -> f64 {
    // Shapes this large: the prefix below is a difference of terms of order
    // `a ln a`, which rounding leaves with no digit; the distribution is a
    // normal one with a small skew.
    if a.min(b) >= LARGE_SHAPE_BETA && a.max(b).is_finite() {
        return beta_normal_approximation(x, a, b);
    }

    incomplete_beta_with_complement(x, y, a, b)
}

/// `I_x(a, b)` given `x` and its complement `y = 1 - x` separately.
///
/// The symmetry relation `I_x(a, b) = 1 - I_y(b, a)` swaps the two, and `1 - x`
/// is rounded: for an `x` near 1 the `y` computed from it has lost `x`'s
/// digits (an `x` of 1 - 2.5e-14 differs from the nearest double to it by up to
/// 0.4 % of `y`), which a shape of 1e14 turns into an error in the answer. The
/// `x` and `y` handed down from the caller are exact, so the side that has to be
/// subtracted from 1 is never computed that way.
fn incomplete_beta_with_complement(x: f64, y: f64, a: f64, b: f64) -> f64 {
    // Use symmetry relation: I_x(a,b) = 1 - I_{1-x}(b,a)
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - incomplete_beta_with_complement(y, x, b, a);
    }

    // A large `a` against a small `b` with `x` near 1: the continued fraction
    // takes `x` itself, and an `x` that is 1 minus a small `y` has `y`'s digits
    // only as far as it was rounded, which a large `a` multiplies. The
    // expansion below is built from `y` alone.
    if x >= 0.5 && bgrat_applies(a, b) {
        if let Some(value) = beta_large_a_expansion(y, a, b) {
            return value;
        }
    }

    beta_by_continued_fraction(x, y, a, b)
}

/// `I_x(a, b)` as the prefix `x^a y^b / (a B(a, b))` times the continued
/// fraction, for an `x` at or below the mean (where it converges quickly).
fn beta_by_continued_fraction(x: f64, y: f64, a: f64, b: f64) -> f64 {
    let ln_prefix = if a.min(b) >= STIRLING_FROM && a.max(b) < 1e15 {
        ln_beta_density_over_a(x, y, a, b)
    } else {
        // The logarithm of whichever of the pair is near 1 from the other one:
        // ln(1 - y) = ln_1p(-y) keeps digits that ln(x) of a rounded x does not.
        let ln_x = if x < 0.5 { x.ln() } else { (-y).ln_1p() };
        let ln_y = if y < 0.5 { y.ln() } else { (-x).ln_1p() };
        a * ln_x + b * ln_y - ln_beta(a, b) - a.ln()
    };
    let cf = beta_cf(x, y, a, b);
    ln_prefix.exp() * cf
}

/// `ln(x^a y^b / (a B(a, b)))` for `a, b >= 10` and `y = 1 − x`, as the distance of
/// `x` from the mean `a / (a + b)` rather than as three large terms that cancel to
/// the order of `ln a`. With `s = a + b` and Stirling's series for the three
/// gammas,
///
/// ```text
/// x^a y^b / B(a, b) = (s x / a)^a (s y / b)^b · sqrt(a b / s) / sqrt(2π) · e^{-c(a) - c(b) + c(s)},
/// ln [(s x / a)^a (s y / b)^b] = −dev(a, s x) − dev(b, s y),
/// ```
///
/// the deviance terms of [`deviance_term`], which are small near the mean. The
/// gaps `a − s x = −g` and `b − s y = g` are one number, `g = s x − a`, which is
/// formed once from whichever of the pair is exact and with one rounding.
fn ln_beta_density_over_a(x: f64, y: f64, a: f64, b: f64) -> f64 {
    let s = a + b;
    // s = a + b is rounded: the error of that sum (TwoSum), kept, so that
    // `s x − a` near the mean is not off by `x` times it.
    let s_back = s - a;
    let s_tail = (a - (s - s_back)) + (b - s_back);
    // `g = s x − a = b − s y`: from the smaller of the pair, which is exact, and
    // with one rounding.
    let g = if x <= y {
        x.mul_add(s, -a) + s_tail * x
    } else {
        (-y).mul_add(s, b) - s_tail * y
    };
    // Far from the mean the direct form `a ln(a / (s x)) + s x − a` is used with
    // the logarithms of the pair taken from the exact one (`ln_1p`), so that a
    // `y` rounded from `1 − x` is never an input of the large `ln` terms.
    let ln_x = if x < 0.5 { x.ln() } else { (-y).ln_1p() };
    let ln_y = if y < 0.5 { y.ln() } else { (-x).ln_1p() };
    let dev_a =
        deviance_near_mean(a, -g, s * x).unwrap_or_else(|| a * (-(b / a).ln_1p() - ln_x) + g);
    let dev_b =
        deviance_near_mean(b, g, s * y).unwrap_or_else(|| b * (-(a / b).ln_1p() - ln_y) - g);
    -dev_a - dev_b + 0.5 * (b.ln() - s.ln() - a.ln())
        - 0.5 * (2.0 * std::f64::consts::PI).ln()
        - stirling_correction(a)
        - stirling_correction(b)
        + stirling_correction(s)
}

/// The most terms a series or continued fraction is given for a shape: near the
/// centre of the distribution they need a number proportional to `sqrt(shape)`
/// to converge, which a fixed 200 gave out at a shape of about 1e4.
fn iteration_limit(shape: f64) -> usize {
    // The cap keeps a shape beyond a double's integers from asking for
    // more terms than a loop can count.
    200 + (12.0 * shape.sqrt()).min(1_000_000.0) as usize
}

/// From this shape on (both parameters) the incomplete beta function is evaluated
/// by its normal limit. Measured against 70-digit references the continued
/// fraction holds `~1e-10` of the smaller tail to a shape of 10¹¹
/// (`I_{1/2}(a, a) = 1/2` at 10¹¹ to 1.6·10⁻¹¹), where the limit with its skewness
/// term is off by `1.2·10⁻⁵` at 10⁷, `1.2·10⁻⁹` at 10¹¹ and `1.2·10⁻¹⁰` at 10¹²:
/// the rounding of the long fraction (`~3·10⁻²¹ · shape`) overtakes the limit's
/// `O(1/shape)` at about 3·10¹¹.
const LARGE_SHAPE_BETA: f64 = 3e11;

/// From this shape on the incomplete gamma functions are evaluated by the
/// Wilson–Hilferty normal limit. The series and the continued fraction hold
/// `~1.5e-10` of the smaller tail to 10¹⁰ (70-digit references) where the limit
/// is off by `1.5·10⁻⁸` at 10⁸ and `8·10⁻¹⁰` at 10¹⁰: the `~1.5·10⁻²⁰ · shape` of
/// the long sums overtakes the limit's `O(1/shape)` at about 2·10¹⁰.
const LARGE_SHAPE_GAMMA: f64 = 1e10;

/// `I_x(a, b)` for large `a` and `b`: the normal approximation to the beta
/// distribution with its first skewness term (Edgeworth),
/// `Φ(z) − φ(z) · γ (z² − 1) / 6`, error `O(1/(a+b))`.
fn beta_normal_approximation(x: f64, a: f64, b: f64) -> f64 {
    // Work on the side where the mean is the small one, so `1 - mean` is not
    // rounded to 1: I_x(a, b) = 1 - I_{1-x}(b, a).
    if a > b {
        return 1.0 - beta_normal_approximation(1.0 - x, b, a);
    }
    let total = a + b;
    // z = (x - mean) / sd with mean = a/total, sd = sqrt(mean (1-mean) / (total+1)),
    // written as the relative distance from the mean so that neither the mean
    // nor the product `ab` underflows or overflows at shapes near 1e300.
    let ratio = x * (1.0 + b / a);
    let relative_sd = ((b / a) / (total + 1.0)).sqrt();
    let z = (ratio - 1.0) / relative_sd;
    if z.abs() > 40.0 {
        return if z > 0.0 { 1.0 } else { 0.0 };
    }
    let skew = 2.0 * (b - a) * (total + 1.0).sqrt() / ((total + 2.0) * a.sqrt() * b.sqrt());
    let density = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    (standard_normal_cdf(z) - density * skew * (z * z - 1.0) / 6.0).clamp(0.0, 1.0)
}

/// Continued fraction for the incomplete beta function (Lentz's algorithm);
/// `y` is `1 - x`, passed on exactly (see [`incomplete_beta_with_complement`]).
fn beta_cf(x: f64, y: f64, a: f64, b: f64) -> f64 {
    let max_iter = iteration_limit(a.max(b));
    const EPS: f64 = 1e-14;
    const TINY: f64 = 1e-30;

    let mut c = 1.0;
    // 1 - (a+b)x/(a+1) = ((a+b) y + 1 - b) / (a+1). Of the two the exact one is
    // used: `x` when it is the small one (then `y = 1 - x` was rounded), `y`
    // otherwise -- for an `x` near 1 the first form subtracts two numbers that
    // agree to the digits of `y`, which are all there are.
    let first = if x < 0.5 {
        1.0 - (a + b) * x / (a + 1.0)
    } else {
        ((a + b) * y + (1.0 - b)) / (a + 1.0)
    };
    let mut d = 1.0 / first.max(TINY);
    let mut h = d;

    for m in 1..=max_iter {
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

/// Where [`beta_large_a_expansion`] is used: `a` large, `b` small beside it.
///
/// The terms fall off by about `b³ / (24 a²)` each, so what it needs is `a`
/// large against `b`, not `a` large: `b³ <= a²` keeps the ratio under 1/24, and
/// `a >= 8 b` and `a >= 15` keep the gamma functions in it away from their small
/// arguments.
fn bgrat_applies(a: f64, b: f64) -> bool {
    a >= BGRAT_MIN_A && a >= BGRAT_RATIO * b && b * b * b <= a * a
}

const BGRAT_MIN_A: f64 = 15.0;
const BGRAT_RATIO: f64 = 8.0;
/// The most terms of the series used; the terms fall off by about `1 / (4 a²)`.
const BGRAT_TERMS: usize = 30;

/// `I_x(a, b)` for a large `a` against a small `b`, from `y = 1 − x` alone
/// (Didonato & Morris 1992, "Algorithm 708", the expansion for `a ≥ 15`;
/// derived here from the integral rather than from their code).
///
/// With `t = e^{-s}`, `I_x(a, b) = B(a, b)⁻¹ ∫_{s₀}^∞ e^{-a s} (1 − e^{-s})^{b−1} ds`
/// where `s₀ = −ln x`, and `(1 − e^{-s})^{b−1} = s^{b−1} e^{-(b−1)s/2} g(s)^{b−1}`
/// with `g(s) = sinh(s/2) / (s/2) = Σ σᵢ s^{2i}`, `σᵢ = 1 / (4ⁱ (2i+1)!)`. Writing
/// `g^{b−1} = Σ dₖ s^{2k}` (`dₖ` by the recurrence for a power of a series) and
/// `ν = a + (b−1)/2`, the integral of each term is an upper incomplete gamma:
///
/// ```text
/// I_x(a, b) = Γ(a+b) / (Γ(a) ν^b) · Σ dₖ (b)_{2k} ν^{-2k} · Q(b + 2k, z),   z = −ν ln x.
/// ```
///
/// `Q(b + 2k, z)` follows from `Q(b, z)` by `Q(c + 1, z) = Q(c, z) + z^c e^{-z} / Γ(c + 1)`,
/// every term positive. `z` comes from `ln_1p(−y)`, so there is no `1 − y`
/// anywhere for a large `a` to amplify. `None` where the series does not settle.
fn beta_large_a_expansion(y: f64, a: f64, b: f64) -> Option<f64> {
    if !(y > 0.0 && y < 1.0 && a.is_finite() && b > 0.0) {
        return None;
    }
    let nu = a + 0.5 * (b - 1.0);
    let z = -nu * (-y).ln_1p();
    if !(z > 0.0 && z.is_finite()) {
        return None;
    }
    // Γ(a+b) / (Γ(a) ν^b), the logarithm taken from Stirling's series so that
    // the `b ln a` of the two gammas and of `ν^b` cancel analytically:
    // (a + b − ½) ln(1 + b/a) − b − b ln(1 + (b−1)/(2a)) + c(a+b) − c(a).
    let ln_ratio = (a + b - 0.5) * (b / a).ln_1p() - b - b * ((b - 1.0) / (2.0 * a)).ln_1p()
        + stirling_correction(a + b)
        - stirling_correction(a);

    // σᵢ and dₖ.
    let power = b - 1.0;
    let mut sigma = [0.0_f64; BGRAT_TERMS + 1];
    let mut d = [0.0_f64; BGRAT_TERMS + 1];
    sigma[0] = 1.0;
    d[0] = 1.0;
    for i in 1..=BGRAT_TERMS {
        let two_i = 2.0 * i as f64;
        sigma[i] = sigma[i - 1] / (4.0 * two_i * (two_i + 1.0));
    }

    // Q(b, z) and the first of the terms z^c e^{-z} / Γ(c + 1) that carry it up.
    let mut q = regularized_upper_gamma(b, z);
    let mut e = (b * z.ln() - z - ln_gamma(b + 1.0)).exp();
    let mut scale = 1.0; // (b)_{2k} / ν^{2k}
    let nu2 = nu * nu;
    let mut sum = d[0] * q;
    for k in 1..=BGRAT_TERMS {
        let kf = k as f64;
        let two_k = 2.0 * kf;
        // Q(b + 2k, z) = Q(b + 2k − 2, z) + E(b + 2k − 2) + E(b + 2k − 1).
        let e_next = e * z / (b + two_k - 1.0);
        q += e + e_next;
        e = e_next * z / (b + two_k);
        scale *= (b + two_k - 2.0) * (b + two_k - 1.0) / nu2;

        let mut acc = 0.0;
        for i in 1..=k {
            acc += (power * i as f64 - (k - i) as f64) * sigma[i] * d[k - i];
        }
        d[k] = acc / kf;

        let term = d[k] * scale * q;
        sum += term;
        if term.abs() <= 1e-17 * sum.abs() {
            let value = ln_ratio.exp() * sum;
            return (value.is_finite() && value >= 0.0).then_some(value.min(1.0));
        }
    }
    None
}

// ============================================================================
// Regularized Lower Incomplete Gamma Function
// ============================================================================

/// Regularized lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a).
///
/// # Algorithm
/// Uses series expansion for `x < a + 1`, continued fraction otherwise.
///
/// # Accuracy
/// Relative error of the value in the smaller tail, against 50- to 70-digit
/// references: 1.4·10⁻¹⁴ at a shape of 50, 4·10⁻¹³ at 10⁵, 4.4·10⁻¹² at 10⁷ and
/// 1.5·10⁻¹⁰ at 10¹⁰ (about `1.5·10⁻²⁰ · shape`, the rounding of a long sum): from a shape
/// of 10 the prefix `x^a e^{−x} / Γ(a)` is formed without the cancellation of
/// three terms of order `a ln a`. From a shape of 10¹⁰ the Wilson–Hilferty
/// normal limit takes over (8·10⁻¹⁰ of the smaller tail there).
///
/// # Examples
/// ```
/// use u_numflow::special::regularized_lower_gamma;
/// // P(1, x) = 1 - exp(-x) for the exponential distribution
/// let p = regularized_lower_gamma(1.0, 2.0);
/// assert!((p - (1.0 - (-2.0f64).exp())).abs() < 1e-10);
/// ```
pub fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    if a >= LARGE_SHAPE_GAMMA {
        return standard_normal_cdf(gamma_normal_score(a, x));
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// Regularized upper incomplete gamma function Q(a, x) = Γ(a, x) / Γ(a) = 1 − P(a, x).
///
/// Computed from the continued fraction where `x ≥ a + 1`, so a small tail
/// keeps its relative precision instead of being the difference of two numbers
/// near 1.
///
/// # Examples
/// ```
/// use u_numflow::special::regularized_upper_gamma;
/// // Q(1, x) = exp(-x) for the exponential distribution
/// assert!((regularized_upper_gamma(1.0, 40.0) / (-40.0f64).exp() - 1.0).abs() < 1e-12);
/// ```
pub fn regularized_upper_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if a >= LARGE_SHAPE_GAMMA {
        return standard_normal_cdf(-gamma_normal_score(a, x));
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_cf(a, x)
    }
}

/// The standard normal score of `x` for a gamma variate of shape `a` (Wilson &
/// Hilferty 1931): `P(a, x) ≈ Φ(z)`, `z = 3 sqrt(a) ((x/a)^(1/3) − 1 + 1/(9a))`,
/// with error `O(1/a)` -- used where `a` is too large for the series.
fn gamma_normal_score(a: f64, x: f64) -> f64 {
    3.0 * a.sqrt() * ((x / a).cbrt() - 1.0 + 1.0 / (9.0 * a))
}

/// Series expansion for the regularized lower incomplete gamma.
fn gamma_series(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut ap = a;
    for _ in 0..iteration_limit(a) {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * 1e-14 {
            break;
        }
    }
    sum * ln_gamma_kernel(a, x).exp()
}

/// Continued fraction for the upper incomplete gamma Q(a, x) = 1 − P(a, x).
fn gamma_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=iteration_limit(a) {
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
    h * ln_gamma_kernel(a, x).exp()
}

// ============================================================================
// Error Function
// ============================================================================

// The erf/erfc approximations below are the FreeBSD msun `s_erf.c` rational
// fits, carried over with their coefficients unchanged:
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
// Intervals on |x|: [0, 0.84375) a Taylor-based fit to erf; [0.84375, 1.25)
// a fit around x = 1; [1.25, 1/0.35) and [1/0.35, 28) fits to
// ln(x·erfc(x)) + x² in z = 1/x², so erfc is formed from an exponential and
// never by subtraction from 1. Stated bounds of each fit are 2⁻⁵⁷·⁹ … 2⁻⁶²·⁵⁷.

const ERX: f64 = 0.8450629115104675;
const EFX8: f64 = 1.0270333367641007;
// erf on [0, 0.84375)
const PP: [f64; 5] = [
    0.12837916709551256,
    -0.3250421072470015,
    -0.02848174957559851,
    -0.005770270296489442,
    -2.3763016656650163e-05,
];
const QQ: [f64; 5] = [
    0.39791722395915535,
    0.0650222499887673,
    0.005081306281875766,
    0.00013249473800432164,
    -3.960228278775368e-06,
];
// erf on [0.84375, 1.25)
const PA: [f64; 7] = [
    -0.0023621185607526594,
    0.41485611868374833,
    -0.3722078760357013,
    0.31834661990116175,
    -0.11089469428239668,
    0.035478304325618236,
    -0.002166375594868791,
];
const QA: [f64; 6] = [
    0.10642088040084423,
    0.540397917702171,
    0.07182865441419627,
    0.12617121980876164,
    0.01363708391202905,
    0.011984499846799107,
];
// erfc on [1.25, 1/0.35)
const RA: [f64; 8] = [
    -0.009864944034847148,
    -0.6938585727071818,
    -10.558626225323291,
    -62.375332450326006,
    -162.39666946257347,
    -184.60509290671104,
    -81.2874355063066,
    -9.814329344169145,
];
const SA: [f64; 8] = [
    19.651271667439257,
    137.65775414351904,
    434.56587747522923,
    645.3872717332679,
    429.00814002756783,
    108.63500554177944,
    6.570249770319282,
    -0.0604244152148581,
];
// erfc on [1/0.35, 28)
const RB: [f64; 7] = [
    -0.0098649429247001,
    -0.799283237680523,
    -17.757954917754752,
    -160.63638485582192,
    -637.5664433683896,
    -1025.0951316110772,
    -483.5191916086514,
];
const SB: [f64; 7] = [
    30.33806074348246,
    325.7925129965739,
    1536.729586084437,
    3199.8582195085955,
    2553.0504064331644,
    474.52854120695537,
    -22.44095244658582,
];

/// Evaluates `c[0] + t·c[1] + … + t^(n−1)·c[n−1]` (Horner).
fn horner(t: f64, c: &[f64]) -> f64 {
    c.iter().rev().fold(0.0, |acc, &ci| acc * t + ci)
}

/// `x·R(x²)` where `erf(x) = x + x·R(x²)` on `|x| < 0.84375`.
fn erf_small_correction(x: f64) -> f64 {
    let z = x * x;
    x * horner(z, &PP) / (1.0 + z * horner(z, &QQ))
}

/// `erf(|x|) − ERX` on `0.84375 ≤ |x| < 1.25`.
fn erf_near_one(ax: f64) -> f64 {
    let s = ax - 1.0;
    horner(s, &PA) / (1.0 + s * horner(s, &QA))
}

/// `erfc(ax)` for `1.25 ≤ ax < 28`, formed without subtraction.
fn erfc_asymptotic(ax: f64) -> f64 {
    let s = 1.0 / (ax * ax);
    let (r, big_s) = if ax < 1.0 / 0.35 {
        (horner(s, &RA), 1.0 + s * horner(s, &SA))
    } else {
        (horner(s, &RB), 1.0 + s * horner(s, &SB))
    };
    // Split −x² as −z² + (z − x)(z + x) with z = x truncated to 32 high bits,
    // so the dominant exponential is evaluated on an exactly representable
    // square (fdlibm Note 1).
    let z = f64::from_bits(ax.to_bits() & 0xFFFF_FFFF_0000_0000);
    (-z * z - 0.5625).exp() * ((z - ax) * (z + ax) + r / big_s).exp() / ax
}

/// Error function erf(x).
///
/// # Definition
/// ```text
/// erf(x) = (2/√π) ∫₀ˣ exp(-t²) dt
/// ```
///
/// # Algorithm
/// Piecewise rational approximations from FreeBSD msun `s_erf.c`
/// (Sun Microsystems, 1993): a Taylor-based fit near 0, a fit around
/// `x = 1`, and `1 − erfc(x)` beyond, where [`erfc`] is formed from an
/// exponential. Error below one ulp for `|x| < 0.84375`, and bounded by
/// 2⁻⁵⁹ in the fitted function elsewhere.
///
/// # Examples
/// ```
/// use u_numflow::special::erf;
/// assert_eq!(erf(0.0), 0.0);
/// assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-15);
/// assert_eq!(erf(-1.0), -erf(1.0));
/// ```
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    let y = if ax < 0.84375 {
        if ax < 2f64.powi(-28) {
            return 0.125 * (8.0 * x + EFX8 * x);
        }
        return x + erf_small_correction(x);
    } else if ax < 1.25 {
        ERX + erf_near_one(ax)
    } else if ax < 6.0 {
        1.0 - erfc_asymptotic(ax)
    } else {
        1.0
    };
    if x < 0.0 {
        -y
    } else {
        y
    }
}

/// Complementary error function erfc(x) = 1 − erf(x), computed directly.
///
/// For `x > 0` the result is formed from an exponential rather than by
/// subtracting [`erf`] from 1, so it keeps its **relative** precision far into
/// the tail: `erfc(10) ≈ 2.09 × 10⁻⁴⁵` is resolved to about 15 significant
/// digits, where `1.0 - erf(10.0)` is exactly 0.
///
/// # Algorithm
/// FreeBSD msun `s_erf.c` (Sun Microsystems, 1993) — see [`erf`].
/// `erfc(−x) = 2 − erfc(x)`.
///
/// # Examples
/// ```
/// use u_numflow::special::erfc;
/// assert_eq!(erfc(0.0), 1.0);
/// assert!((erfc(3.0) / 2.209049699858544e-5 - 1.0).abs() < 1e-14);
/// assert!((erfc(10.0) / 2.088487583762545e-45 - 1.0).abs() < 1e-13);
/// assert_eq!(erfc(f64::INFINITY), 0.0);
/// assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
/// ```
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax < 0.84375 {
        if ax < 2f64.powi(-56) {
            return 1.0 - x;
        }
        let corr = erf_small_correction(x);
        return if x < 0.25 {
            1.0 - (x + corr)
        } else {
            0.5 - (x - 0.5 + corr)
        };
    }
    if ax < 1.25 {
        let d = erf_near_one(ax);
        return if x > 0.0 {
            1.0 - ERX - d
        } else {
            1.0 + ERX + d
        };
    }
    if ax < 28.0 {
        let tail = erfc_asymptotic(ax);
        return if x > 0.0 { tail } else { 2.0 - tail };
    }
    if x > 0.0 {
        0.0
    } else {
        2.0
    }
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
    let ib = t_two_tail(t, df);
    if t >= 0.0 {
        1.0 - ib / 2.0
    } else {
        ib / 2.0
    }
}

/// `I_x(df/2, 1/2)` with `x = df / (df + t²)`, the probability of `|T| > |t|`,
/// from the pair `x` and `1 − x = t² / (df + t²)` each formed directly: for a
/// large `df` a rounded `1 − x` is what a large shape would multiply.
fn t_two_tail(t: f64, df: f64) -> f64 {
    let t2 = t * t;
    let total = df + t2;
    let (x, y) = (df / total, t2 / total);
    if y.is_nan() {
        // t² overflowed: x is 0.
        return 0.0;
    }
    beta_cdf_from_pair(x, y, df / 2.0, 0.5)
}

/// The upper tail `1 − I_x(a, b) = I_{1−x}(b, a)` for an `x` given alone: the
/// complement of `x` is the one rounded, and the side of the pair that is exact
/// (the smaller of `x` and `1 − x`) is what the evaluation is built from, so a
/// large `b` does not multiply the rounding. The probability is not taken as
/// `1 − I_x(a, b)`, which has no digits left once the lower tail is near 1.
pub(crate) fn regularized_incomplete_beta_upper(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x >= 1.0 {
        return 0.0;
    }
    beta_cdf_pair(1.0 - x, x, b, a)
}

/// `I_x(a, b)` for a pair `x`, `y = 1 − x` the caller has formed directly.
pub(crate) fn beta_cdf_from_pair(x: f64, y: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if y <= 0.0 {
        return 1.0;
    }
    beta_cdf_pair(x, y, a, b)
}

/// Survival function of Student's t-distribution: P(T > t | df) = 1 − F(t).
///
/// Unlike `1.0 - t_distribution_cdf(..)`, the upper tail keeps its relative
/// precision: `t_distribution_sf(30.0, 10.0)` is 1.98e-11 to full precision,
/// where the complement of the CDF keeps about five digits and, further out,
/// none. A two-sided p-value is `2 · t_distribution_sf(|t|, df)`.
///
/// # Returns
/// - `f64::NAN` if df ≤ 0 or inputs are NaN.
///
/// # Examples
/// ```
/// use u_numflow::special::t_distribution_sf;
/// assert_eq!(t_distribution_sf(0.0, 10.0), 0.5);
/// // 1.98e-11, SciPy: scipy.stats.t.sf(30, 10)
/// assert!((t_distribution_sf(30.0, 10.0) / 1.980896171015662e-11 - 1.0).abs() < 1e-9);
/// ```
pub fn t_distribution_sf(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if t == 0.0 {
        return 0.5;
    }
    let tail = t_two_tail(t, df) / 2.0;
    if t > 0.0 {
        tail
    } else {
        1.0 - tail
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
    // `ln_1p`: for a `t²/df` under a double's epsilon the 1 + … rounds it away,
    // which a half-df of 1e14 multiplies back.
    let log_pdf = ln_gamma_half_step(half_df)
        - 0.5 * (df * std::f64::consts::PI).ln()
        - (half_df + 0.5) * (t * t / df).ln_1p();
    log_pdf.exp()
}

/// Quantile function (inverse CDF) of Student's t-distribution.
///
/// Given a probability `p ∈ (0, 1)`, returns `t` such that `P(T ≤ t) = p`.
///
/// # Algorithm
/// The two-sided tail `u = 2·min(p, 1 − p)` (both exact) gives
/// `I_x(df/2, ½) = u` for `x = df / (df + t²)`; [`inverse_regularized_incomplete_beta`]
/// solves it in the logarithm of whichever of `x` and `1 − x` is the smaller, and
/// `t² = df · (1 − x) / x`. The result keeps its relative precision for a
/// probability as small as 10⁻³⁰⁰ and a `p` within rounding of ½, at any `df`:
/// a Newton iteration on the CDF from the normal quantile does neither
/// (a Cauchy `t(1)` at `p = 10⁻³⁰` was off by a factor of 2).
///
/// # Returns
/// - `f64::NAN` if `p` is outside `(0, 1)`, `df ≤ 0` or infinite, or an input is NaN.
/// - `±∞` for a quantile beyond the largest double.
///
/// # Examples
/// ```
/// use u_numflow::special::t_distribution_quantile;
/// // Median = 0
/// assert!(t_distribution_quantile(0.5, 10.0).abs() < 1e-10);
/// // df=∞ → normal quantile
/// assert!((t_distribution_quantile(0.975, 10000.0) - 1.96).abs() < 0.01);
/// // A Cauchy has the closed form tan(π(p − ½)): t(1) at p = 10⁻³⁰ is −1/(π·p)
/// let t = t_distribution_quantile(1e-30, 1.0);
/// assert!((t / -3.183098861837907e29 - 1.0).abs() < 1e-13);
/// ```
pub fn t_distribution_quantile(p: f64, df: f64) -> f64 {
    if p.is_nan() || df.is_nan() || df <= 0.0 || df.is_infinite() || p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    if p == 0.5 {
        return 0.0;
    }
    // 1 − p is exact for p ≥ ½, so the smaller tail is never a rounding.
    let (tail, negative) = if p < 0.5 { (p, true) } else { (1.0 - p, false) };
    let two_sided = 2.0 * tail;
    let root = inverse_beta_tails(two_sided, 1.0 - two_sided, df / 2.0, 0.5);
    let magnitude = df.sqrt() * root.sqrt_y_over_x();
    if negative {
        -magnitude
    } else {
        magnitude
    }
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
    let scaled = df1 * x;
    let total = scaled + df2;
    if total.is_infinite() {
        return 1.0;
    }
    beta_cdf_from_pair(scaled / total, df2 / total, df1 / 2.0, df2 / 2.0)
}

/// Survival function of the F-distribution: P(X > x | df1, df2) = 1 − F(x).
///
/// ```text
/// S(x; d1, d2) = I_z(d2/2, d1/2)   where z = d2 / (d1·x + d2)
/// ```
///
/// `z` is formed directly, not as `1 − y`, so a small upper tail keeps its
/// relative precision — `1.0 - f_distribution_cdf(..)` has none left below
/// about 1e-16. This is the p-value of an F test.
///
/// # Returns
/// - `f64::NAN` if df1 ≤ 0, df2 ≤ 0, or inputs are NaN.
/// - `1.0` if x ≤ 0.
///
/// # Examples
/// ```
/// use u_numflow::special::f_distribution_sf;
/// assert_eq!(f_distribution_sf(0.0, 5.0, 10.0), 1.0);
/// // 2.56e-37, SciPy: scipy.stats.f.sf(1422.4107710431, 4, 34)
/// assert!((f_distribution_sf(1422.4107710431, 4.0, 34.0) / 2.5558624671479177e-37 - 1.0).abs() < 1e-9);
/// ```
pub fn f_distribution_sf(x: f64, df1: f64, df2: f64) -> f64 {
    if x.is_nan() || df1.is_nan() || df2.is_nan() || df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    let scaled = df1 * x;
    let total = scaled + df2;
    if total.is_infinite() {
        return 0.0;
    }
    beta_cdf_from_pair(df2 / total, scaled / total, df2 / 2.0, df1 / 2.0)
}

/// Quantile function (inverse CDF) of the F-distribution.
///
/// Given a probability `p ∈ (0, 1)`, returns `x` such that `P(X ≤ x) = p`.
///
/// # Algorithm
/// `X = (d2/d1) · B / (1 − B)` for `B ~ Beta(d1/2, d2/2)`:
/// [`inverse_regularized_incomplete_beta`] solves for `B` and `1 − B` together,
/// in the logarithm of the smaller, so the result keeps its relative precision
/// for a probability of 10⁻³⁰⁰ and for an upper tail of 10⁻¹⁴ given as
/// `p = 1 − 10⁻¹⁴` (bisection on the CDF was off by a factor of 400 at
/// `F(2, 5)`, `p = 10⁻¹⁰`).
///
/// # Returns
/// - `f64::NAN` if `p` is outside `(0, 1)` or df1/df2 ≤ 0 or infinite.
/// - `+∞` for a quantile beyond the largest double.
///
/// # Examples
/// ```
/// use u_numflow::special::f_distribution_quantile;
/// // F(2, d2) has the closed form (d2/2)·((1 − p)^(−2/d2) − 1); here d2 = 4: 2·((1 − p)^(−½) − 1)
/// let x = f_distribution_quantile(0.75, 2.0, 4.0);
/// assert!((x - 2.0).abs() < 1e-13);
/// ```
pub fn f_distribution_quantile(p: f64, df1: f64, df2: f64) -> f64 {
    if p.is_nan()
        || df1.is_nan()
        || df2.is_nan()
        || df1 <= 0.0
        || df2 <= 0.0
        || df1.is_infinite()
        || df2.is_infinite()
        || p <= 0.0
        || p >= 1.0
    {
        return f64::NAN;
    }
    let root = inverse_beta_tails(p, 1.0 - p, df1 / 2.0, df2 / 2.0);
    (df2 / df1) * root.x_over_y()
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

/// Survival function of the chi-squared distribution: P(X > x | k) = 1 − F(x).
///
/// `Q(k/2, x/2)` ([`regularized_upper_gamma`]): a small upper tail keeps its
/// relative precision, where `1.0 - chi_squared_cdf(..)` rounds to zero below
/// about 1e-16. This is the p-value of a chi-squared test.
///
/// # Returns
/// - `f64::NAN` if k ≤ 0 or inputs are NaN.
/// - `1.0` if x ≤ 0.
///
/// # Examples
/// ```
/// use u_numflow::special::chi_squared_sf;
/// // k = 2 is an exponential: S(x) = exp(-x/2)
/// assert!((chi_squared_sf(93.0, 2.0) / (-46.5f64).exp() - 1.0).abs() < 1e-12);
/// ```
pub fn chi_squared_sf(x: f64, k: f64) -> f64 {
    if x.is_nan() || k.is_nan() || k <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    regularized_upper_gamma(k / 2.0, x / 2.0)
}

/// Quantile function (inverse CDF) of the chi-squared distribution.
///
/// Given a probability `p ∈ (0, 1)`, returns `x` such that `P(X ≤ x) = p`
/// for `X ~ χ²(k)`.
///
/// # Algorithm
/// `2 · `[`inverse_regularized_lower_gamma`]`(k/2, p)`: Newton's method in
/// `ln x` on the logarithm of the incomplete gamma function, so a quantile far
/// below 1 (`p = 10⁻³⁰`, `k = 1`) keeps its relative precision. Exact to the
/// tolerance of the incomplete gamma function, unlike the Wilson–Hilferty
/// cube-root approximation, which is off by about 2–3 % for one or two
/// degrees of freedom in the tails.
///
/// # Returns
/// - `f64::NAN` if `p` is outside `(0, 1)`, `k ≤ 0` or infinite, or inputs are NaN.
/// - `+∞` for a quantile beyond the largest double.
///
/// # Examples
/// ```
/// use u_numflow::special::chi_squared_quantile;
/// // NIST/SEMATECH table: χ²(0.95; 1) = 3.841, χ²(0.975; 2) = 7.378
/// assert!((chi_squared_quantile(0.95, 1.0) - 3.8415).abs() < 1e-3);
/// assert!((chi_squared_quantile(0.975, 2.0) - 7.3778).abs() < 1e-3);
/// // k = 2 is an exponential with mean 2: x = −2·ln(1 − p), here p = 10⁻³⁰
/// assert!((chi_squared_quantile(1e-30, 2.0) / 2e-30 - 1.0).abs() < 1e-13);
/// ```
pub fn chi_squared_quantile(p: f64, k: f64) -> f64 {
    if p.is_nan() || k.is_nan() || k <= 0.0 || k.is_infinite() || p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    2.0 * inverse_gamma_tails(k / 2.0, p, 1.0 - p)
}

// ============================================================================
// Inverse of the regularized incomplete gamma and beta functions
// ============================================================================

/// The solution `y` of `h(y) = target` for an increasing `h` that returns its
/// own slope, by Newton steps kept inside a shrinking bracket `[lo, hi]` and
/// replaced by a bisection when a step leaves it or the slope is unusable.
///
/// The inverse solvers use `y = ln x` and `h = ln P` (or `−ln Q`): a tail
/// probability is a power of `x` for small `x`, so in the logarithm it is nearly
/// a straight line and Newton converges from any start — where a step in `x`
/// with an absolute floor (1e-15) or an absolute width (1e-14) cannot name a
/// quantile of 1e-50 at all.
pub(crate) fn solve_in_log(
    target: f64,
    start: f64,
    lo: f64,
    hi: f64,
    h: impl Fn(f64) -> (f64, f64),
) -> f64 {
    let (mut lo, mut hi) = (lo, hi);
    let mut y = start.clamp(lo, hi);
    for _ in 0..200 {
        let (value, slope) = h(y);
        let miss = value - target;
        if miss == 0.0 {
            return y;
        }
        if miss < 0.0 || miss.is_nan() {
            lo = y;
        } else {
            hi = y;
        }
        let mut next = y - miss / slope;
        if !(slope > 0.0 && slope.is_finite() && next > lo && next < hi) {
            next = 0.5 * (lo + hi);
        }
        if (next - y).abs() <= 2.0 * f64::EPSILON * next.abs().max(1.0)
            || hi - lo <= f64::EPSILON * next.abs().max(1.0)
        {
            return next;
        }
        y = next;
    }
    y
}

/// The smallest `ln x` an inverse solver looks for: `ln` of the least positive
/// double. A quantile below it is reported as that double.
pub(crate) const LN_LEAST_POSITIVE: f64 = -744.440_071_921_381_3;

/// `ln ½`.
const LN_HALF: f64 = -std::f64::consts::LN_2;

/// The `x` with `P(a, x) = p` (and `Q(a, x) = q`) for tails `p + q = 1` that
/// the caller holds exactly, `0 < p, q < 1`. The smaller tail is solved for, so
/// an upper tail of 1e-14 given as `q` is not a rounding of `1 − p`.
pub(crate) fn inverse_gamma_tails(a: f64, p: f64, q: f64) -> f64 {
    let lower = p <= q;
    let target = if lower { p.ln() } else { -q.ln() };

    // Wilson-Hilferty for a shape of 1 and up, where it names a positive value;
    // otherwise the leading term of the series (lower tail) or of the
    // asymptotic expansion of the upper tail.
    let v = 1.0 / (9.0 * a);
    let z = if lower {
        inverse_normal_cdf(p)
    } else {
        -inverse_normal_cdf(q)
    };
    let wilson = a * (1.0 - v + z * v.sqrt()).powi(3);
    let start = if a >= 1.0 && wilson.is_finite() && wilson > 0.0 {
        wilson.ln()
    } else if lower {
        (p.ln() + ln_gamma(a + 1.0)) / a
    } else {
        let l = -q.ln();
        (l + (a - 1.0) * l.max(1.0).ln()).max(0.5).ln()
    };

    let y = solve_in_log(target, start, LN_LEAST_POSITIVE, f64::MAX.ln(), |y| {
        let x = y.exp();
        // x·f(x) = exp(ln(x^a e^−x / Γ(a))) is the slope of P and −Q in ln x.
        let kernel = ln_gamma_kernel(a, x);
        if lower {
            let value = regularized_lower_gamma(a, x).ln();
            (value, (kernel - value).exp())
        } else {
            let value = -regularized_upper_gamma(a, x).ln();
            (value, (kernel + value).exp())
        }
    });
    y.exp()
}

/// Inverse of the regularized lower incomplete gamma function: the `x` with
/// `P(a, x) = p` ([`regularized_lower_gamma`]).
///
/// This is the quantile of a Gamma(shape `a`, scale 1) variable, and
/// `2 · inverse_regularized_lower_gamma(k/2, p)` is the chi-squared quantile.
///
/// # Algorithm
/// Newton's method on `ln P` in `ln x`, kept inside a bracket and replaced by a
/// bisection where a step leaves it. The relative error of `x` is the incomplete
/// gamma function's own, for `p` down to 10⁻³⁰⁰ (a root below the least positive
/// double, 5·10⁻³²⁴, is reported as that double, one above the largest double as
/// the largest double). For an upper tail much smaller than `p`'s own rounding
/// use [`inverse_regularized_upper_gamma`].
///
/// # Returns
/// - `0` for `p = 0`, `+∞` for `p = 1`.
/// - `f64::NAN` if `a ≤ 0` or infinite, `p ∉ [0, 1]` or an input is NaN.
///
/// # Examples
/// ```
/// use u_numflow::special::{inverse_regularized_lower_gamma, regularized_lower_gamma};
/// // a = 1 is the unit exponential: P(1, x) = 1 − e^−x, so x = −ln(1 − p)
/// let x = inverse_regularized_lower_gamma(1.0, 1e-30);
/// assert!((x / 1e-30 - 1.0).abs() < 1e-13);
/// // a round trip
/// let x = inverse_regularized_lower_gamma(3.5, 0.2);
/// assert!((regularized_lower_gamma(3.5, x) - 0.2).abs() < 1e-14);
/// ```
pub fn inverse_regularized_lower_gamma(a: f64, p: f64) -> f64 {
    if a.is_nan() || a <= 0.0 || a.is_infinite() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    inverse_gamma_tails(a, p, 1.0 - p)
}

/// Inverse of the regularized upper incomplete gamma function: the `x` with
/// `Q(a, x) = q` ([`regularized_upper_gamma`]).
///
/// The upper tail is taken as given, not as `1 − p`: a `q` of 10⁻¹⁴ or 10⁻²⁰⁰
/// (the p-value of a chi-squared test: `x = 2 · inverse_regularized_upper_gamma(k/2, p_value)`)
/// keeps every digit, where `inverse_regularized_lower_gamma(a, 1 − q)` has none
/// below `q ≈ 10⁻¹⁶`.
///
/// # Algorithm
/// As [`inverse_regularized_lower_gamma`], on `−ln Q`.
///
/// # Returns
/// - `+∞` for `q = 0`, `0` for `q = 1`.
/// - `f64::NAN` if `a ≤ 0` or infinite, `q ∉ [0, 1]` or an input is NaN.
///
/// # Examples
/// ```
/// use u_numflow::special::{inverse_regularized_upper_gamma, regularized_upper_gamma};
/// // a = 1: Q(1, x) = e^−x, so x = −ln q
/// let x = inverse_regularized_upper_gamma(1.0, 1e-200);
/// assert!((x / (200.0 * std::f64::consts::LN_10) - 1.0).abs() < 1e-14);
/// let x = inverse_regularized_upper_gamma(2.5, 1e-14);
/// assert!((regularized_upper_gamma(2.5, x) / 1e-14 - 1.0).abs() < 1e-12);
/// ```
pub fn inverse_regularized_upper_gamma(a: f64, q: f64) -> f64 {
    if a.is_nan() || a <= 0.0 || a.is_infinite() || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    if q == 0.0 {
        return f64::INFINITY;
    }
    if q == 1.0 {
        return 0.0;
    }
    inverse_gamma_tails(a, 1.0 - q, q)
}

/// The `ln x` of the `x ≤ ½` with `I_x(a, b) = p` (and `1 − I_x(a, b) = q`) for
/// tails `p + q = 1` held exactly, `0 < p, q < 1`. The caller has established
/// that the root is at most ½. The smaller tail is solved for, in `ln x`.
fn inverse_beta_small_side(p: f64, q: f64, a: f64, b: f64) -> f64 {
    let lower = p <= q;
    let ln_b = ln_beta(a, b);
    let target = if lower { p.ln() } else { -q.ln() };
    // I_x ≈ x^a / (a B(a, b)) for small x: the line `ln I = a·y − ln(a B)`.
    let line = (p.ln() + a.ln() + ln_b) / a;
    // The root is returned as a logarithm, so it may lie below the least positive
    // double: a shape of ½ and a probability of 1e-100 name an `x` of 1e-400. There
    // the leading term is the whole function (the next one is smaller by a factor
    // of `x`), so the line is the answer.
    if lower && line < LN_LEAST_POSITIVE {
        return line;
    }
    let start = if lower {
        line.min(LN_HALF)
    } else {
        LN_HALF - 0.5
    };
    solve_in_log(target, start, LN_LEAST_POSITIVE, LN_HALF, |y| {
        let x = y.exp();
        // x·f(x) is the slope of I and of −ln(1 − I) in ln x, up to the tail.
        let ln_x_f = a * y + (b - 1.0) * (-x).ln_1p() - ln_b;
        if lower {
            let value = beta_cdf_pair(x, 1.0 - x, a, b).ln();
            (value, (ln_x_f - value).exp())
        } else {
            // The upper tail is `I_y(b, a)`, and is handed `x` itself: a `y`
            // rounded from `1 − x` has lost the digits of a small `x`.
            let value = -beta_cdf_pair(1.0 - x, x, b, a).ln();
            (value, (ln_x_f + value).exp())
        }
    })
}

/// The root of `I_x(a, b) = p`, as `x` and `y = 1 − x` with the logarithm of each.
pub(crate) struct BetaRoot {
    x: f64,
    y: f64,
    ln_x: f64,
    ln_y: f64,
}

impl BetaRoot {
    /// Both parts are normal doubles: a quotient formed from them is exact to
    /// rounding. Below that a subnormal has lost digits and the quotient is
    /// formed from the logarithms.
    fn both_normal(&self) -> bool {
        self.x >= f64::MIN_POSITIVE && self.y >= f64::MIN_POSITIVE
    }

    /// `x / y`.
    pub(crate) fn x_over_y(&self) -> f64 {
        if self.both_normal() {
            self.x / self.y
        } else {
            (self.ln_x - self.ln_y).exp()
        }
    }

    /// `√(y / x)`.
    pub(crate) fn sqrt_y_over_x(&self) -> f64 {
        if self.both_normal() {
            (self.y / self.x).sqrt()
        } else {
            (0.5 * (self.ln_y - self.ln_x)).exp()
        }
    }
}

/// The root of `I_x(a, b) = p` for tails `p + q = 1` held exactly,
/// `0 < p, q < 1`: the smaller of `x` and `1 − x` (the one a quantile needs to
/// tell apart from 0) is the one solved for, so both come back with their
/// relative precision. Student's t and the F ratio are `(1 − x)/x` and `x/(1 − x)`.
pub(crate) fn inverse_beta_tails(p: f64, q: f64, a: f64, b: f64) -> BetaRoot {
    // The root is at most ½ exactly when P(½) ≥ p.
    if p <= regularized_incomplete_beta(0.5, a, b) {
        let ln_x = inverse_beta_small_side(p, q, a, b);
        let x = ln_x.exp().min(0.5);
        BetaRoot {
            x,
            y: 1.0 - x,
            ln_x,
            ln_y: (-x).ln_1p(),
        }
    } else {
        let ln_y = inverse_beta_small_side(q, p, b, a);
        let y = ln_y.exp().min(0.5);
        BetaRoot {
            x: 1.0 - y,
            y,
            ln_x: (-y).ln_1p(),
            ln_y,
        }
    }
}

/// Inverse of the regularized incomplete beta function: the `x ∈ [0, 1]` with
/// `I_x(a, b) = p` ([`regularized_incomplete_beta`]).
///
/// This is the quantile of a Beta(`a`, `b`) variable, and the building block of
/// the Student's t and F quantiles.
///
/// # Algorithm
/// Newton's method on `ln I` in `ln x`, kept inside a bracket and replaced by a
/// bisection where a step leaves it — in the logarithm of whichever of `x` and
/// `1 − x` is the smaller, against whichever of `p` and `1 − p` is the smaller.
/// The relative error of a root below ½ is the incomplete beta function's own,
/// from `x = 0.3` down to `x = 10⁻⁵⁰` and past it (a root below the least positive
/// double, 5·10⁻³²⁴, comes back as 0). A root within 10⁻¹⁶ of 1 comes back as a
/// double near 1 with that absolute error: a caller that needs `1 − x`
/// (the t and F quantiles do) uses the mirror identity
/// `inverse_regularized_incomplete_beta(1 − p, b, a) = 1 − x`, with `1 − p`
/// held exactly.
///
/// # Returns
/// - `0` for `p = 0`, `1` for `p = 1`.
/// - `f64::NAN` if `a` or `b` is `≤ 0` or infinite, `p ∉ [0, 1]` or an input is NaN.
///
/// # Examples
/// ```
/// use u_numflow::special::{inverse_regularized_incomplete_beta, regularized_incomplete_beta};
/// // b = 1: I_x(a, 1) = x^a, so x = p^(1/a)
/// let x = inverse_regularized_incomplete_beta(1e-30, 2.0, 1.0);
/// assert!((x / 1e-15 - 1.0).abs() < 1e-13);
/// let x = inverse_regularized_incomplete_beta(0.9, 3.0, 4.5);
/// assert!((regularized_incomplete_beta(x, 3.0, 4.5) - 0.9).abs() < 1e-14);
/// ```
pub fn inverse_regularized_incomplete_beta(p: f64, a: f64, b: f64) -> f64 {
    if a.is_nan()
        || a <= 0.0
        || a.is_infinite()
        || b.is_nan()
        || b <= 0.0
        || b.is_infinite()
        || !(0.0..=1.0).contains(&p)
    {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    inverse_beta_tails(p, 1.0 - p, a, b).x
}

// ============================================================================
// Noncentral t-Distribution
// ============================================================================

/// CDF of the noncentral t-distribution: `P(T ≤ t)` for `T ~ t(df, δ)`.
///
/// `T = (Z + δ) / √(V/df)` with `Z ~ N(0, 1)` and `V ~ χ²(df)` independent.
/// With `δ = 0` it is Student's t ([`t_distribution_cdf`]).
///
/// This is the distribution a t statistic follows when the null hypothesis
/// is false, so it is what the power of a t test (and of an effect test in a
/// two-level factorial) is computed from: a two-sided test at critical value
/// `c` has power `1 − F(c; df, δ) + F(−c; df, δ)`.
///
/// # Algorithm
/// Lenth (1989), *Algorithm AS 243: Cumulative distribution function of the
/// non-central t distribution*, Applied Statistics 38(1), 185–189 — the
/// Poisson-weighted series of incomplete beta functions, summed until its
/// error bound falls below `1e-12`.
///
/// # Accuracy
/// The series weights start at `exp(−δ²/2)`, which underflows for
/// `|δ| ≳ 37`; there the result is not reliable. Power calculations stay far
/// below that range.
///
/// # Returns
/// - `f64::NAN` if `df ≤ 0` or any input is not finite.
///
/// # Examples
/// ```
/// use u_numflow::special::{noncentral_t_cdf, t_distribution_cdf};
/// // δ = 0 is the central t
/// let c = noncentral_t_cdf(2.228, 10.0, 0.0);
/// assert!((c - t_distribution_cdf(2.228, 10.0)).abs() < 1e-10);
/// // t = 0: P(T ≤ 0) = Φ(−δ)
/// assert!((noncentral_t_cdf(0.0, 4.0, 1.0) - 0.158_655_253_9).abs() < 1e-9);
/// ```
pub fn noncentral_t_cdf(t: f64, df: f64, delta: f64) -> f64 {
    if !t.is_finite() || !df.is_finite() || !delta.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    const ERRMAX: f64 = 1e-12;
    const ITRMAX: usize = 1000;

    // F(t; df, δ) = 1 − F(−t; df, −δ): evaluate for t ≥ 0.
    let (tt, del, negdel) = if t < 0.0 {
        (-t, -delta, true)
    } else {
        (t, delta, false)
    };

    let mut tnc = 0.0;
    let x = tt * tt / (tt * tt + df);
    if x > 0.0 {
        let lambda = del * del;
        let mut p = 0.5 * (-0.5 * lambda).exp();
        let mut q = (2.0 / std::f64::consts::PI).sqrt() * p * del;
        let mut s = 0.5 - p;
        let mut a = 0.5;
        let b = 0.5 * df;
        let rxb = (1.0 - x).powf(b);
        let albeta = 0.5 * std::f64::consts::PI.ln() - ln_gamma_half_step(b);
        let mut xodd = regularized_incomplete_beta(x, a, b);
        let mut godd = 2.0 * rxb * (a * x.ln() - albeta).exp();
        let mut xeven = 1.0 - rxb;
        let mut geven = b * x * rxb;
        tnc = p * xodd + q * xeven;

        let mut en = 1.0;
        for _ in 0..ITRMAX {
            a += 1.0;
            xodd -= godd;
            xeven -= geven;
            godd *= x * (a + b - 1.0) / a;
            geven *= x * (a + b - 0.5) / (a + 0.5);
            p *= lambda / (2.0 * en);
            q *= lambda / (2.0 * en + 1.0);
            s -= p;
            en += 1.0;
            tnc += p * xodd + q * xeven;
            let errbd = 2.0 * s * (xodd - godd);
            if errbd.abs() <= ERRMAX {
                break;
            }
        }
    }
    tnc += standard_normal_sf(del);

    let out = if negdel { 1.0 - tnc } else { tnc };
    out.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(x, ½·erfc(x/√2), φ(x))` to 20 digits from mpmath at 40 digits: the
    /// tail beyond 6 sd was off by 10⁻¹³ (35 sd) when `x/√2` was rounded to a
    /// double before the exponential multiplied the error by `x²`.
    #[allow(clippy::excessive_precision)] // the oracle's digits, not a double's
    const NORMAL_TAIL_REFERENCE: [(f64, f64, f64); 8] = [
        (1.8, 0.035930319112925800454, 0.078950158300894149457),
        (3.0, 0.0013498980316300945267, 0.0044318484119380071756),
        (6.0, 9.865876450376981407e-10, 6.075882849823285487e-9),
        (10.0, 7.619853024160526066e-24, 7.6945986267064193463e-23),
        (15.5, 1.7344607917938700513e-54, 2.6995130245885872525e-53),
        (20.0, 2.7536241186062336951e-89, 5.5209483621597631896e-88),
        (27.3, 2.1207986243198491043e-164, 5.7975280291088499778e-163),
        (35.0, 1.124910706472406244e-268, 3.9403962771360243307e-267),
    ];

    /// Φ, 1 − Φ and φ keep a few ulp of relative accuracy out to 35 sd — the
    /// measured worst case is 4·10⁻¹⁶ over a grid to 38 (1.7·10⁻¹³ before).
    #[test]
    fn normal_tails_are_accurate_to_a_few_ulp() {
        for (x, tail, pdf) in NORMAL_TAIL_REFERENCE {
            for (name, got, want) in [
                ("sf", standard_normal_sf(x), tail),
                ("cdf(-x)", standard_normal_cdf(-x), tail),
                ("pdf", standard_normal_pdf(x), pdf),
                ("pdf(-x)", standard_normal_pdf(-x), pdf),
            ] {
                let rel = (got / want - 1.0).abs();
                assert!(rel < 1e-15, "{name} at {x}: {got:e} vs {want:e} ({rel:e})");
            }
        }
    }

    /// Tails of the t, chi-squared and F distributions over eleven decades of
    /// probability agree with 50-digit references (mpmath) to 1e-11 — the
    /// measured worst case over a grid of shapes up to 1000 is 4e-13, against a
    /// documented 1e-10. The values are the library's own `cdf`/`sf` of the
    /// *tail* side, the side `1 − p` cannot resolve.
    #[test]
    #[allow(clippy::excessive_precision)] // the oracle's digits, not a double's
    fn distribution_tails_match_high_precision_references() {
        let cases: [(&str, f64, f64); 11] = [
            (
                "t sf(50, df 5)",
                t_distribution_sf(50.0, 5.0),
                3.0238788133006126158e-8,
            ),
            (
                "t cdf(-1000, df 30)",
                t_distribution_cdf(-1000.0, 30.0),
                1.0360017415558665384e-69,
            ),
            (
                "t sf(40, df 100)",
                t_distribution_sf(40.0, 100.0),
                1.2310538010700354086e-63,
            ),
            (
                "t cdf(-0.0685, df 1000)",
                t_distribution_cdf(-0.0685, 1000.0),
                0.47270065703769550609,
            ),
            (
                "chi2 sf(150, k 10)",
                chi_squared_sf(150.0, 10.0),
                3.7274850550625096223e-27,
            ),
            (
                "chi2 cdf(162.6, k 300)",
                chi_squared_cdf(162.6, 300.0),
                6.0338105060092684794e-12,
            ),
            (
                "chi2 cdf(0.1, k 50)",
                chi_squared_cdf(0.1, 50.0),
                1.8311572110618871927e-58,
            ),
            (
                "F sf(43615, 5, 20)",
                f_distribution_sf(43615.0, 5.0, 20.0),
                1.1929248028039253673e-39,
            ),
            (
                "F cdf(0.0167, 400, 100)",
                f_distribution_cdf(0.0167, 400.0, 100.0),
                2.5142250142406613392e-190,
            ),
            (
                "F cdf(0.0084, 30, 30)",
                f_distribution_cdf(0.0084, 30.0, 30.0),
                4.4840323212262492187e-24,
            ),
            (
                "F sf(960750, 400, 100)",
                f_distribution_sf(960750.0, 400.0, 100.0),
                6.2974676960374199785e-277,
            ),
        ];
        for (name, got, want) in cases {
            let rel = (got / want - 1.0).abs();
            assert!(rel < 1e-11, "{name}: {got:e} vs {want:e} ({rel:e})");
        }
    }

    /// Where the tail takes over from `erfc` (x = 1.25·√2) the two agree, and
    /// the cut is monotone across it.
    #[test]
    fn normal_tail_is_continuous_across_the_handover() {
        let edge = TAIL_FROM;
        for k in -8..=8_i32 {
            let x = f64::from_bits((edge.to_bits() as i64 + i64::from(k)) as u64);
            let via_erfc = 0.5 * erfc(x * FRAC_1_SQRT_2);
            let rel = (standard_normal_sf(x) / via_erfc - 1.0).abs();
            assert!(rel < 2e-15, "x = {x}: {rel:e}");
        }
        let (below, above) = (
            standard_normal_sf(f64::from_bits(edge.to_bits() - 1)),
            standard_normal_sf(f64::from_bits(edge.to_bits() + 1)),
        );
        assert!(above <= below, "sf must not rise across the handover");
    }

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
        assert!((ln_gamma(3.0) - 2.0f64.ln()).abs() < 1e-10); // Γ(3) = 2
        assert!((ln_gamma(5.0) - 24.0f64.ln()).abs() < 1e-10); // Γ(5) = 24
        assert!((ln_gamma(7.0) - 720.0f64.ln()).abs() < 1e-9); // Γ(7) = 720
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
        assert!((ln_beta(1.0, 2.0) - (-2.0f64.ln())).abs() < 1e-10);
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
        for &x in &[-3.0, -1.0, -0.5, 0.0, 0.2, 0.5, 1.0, 2.0, 3.0] {
            let sum = erf(x) + erfc(x);
            assert!((sum - 1.0).abs() < 1e-15, "erf({x}) + erfc({x}) = {sum}");
        }
    }

    // --- tail precision ---
    //
    // Reference values are double-precision evaluations from independent
    // implementations (SciPy `norm.sf` / `erfc`, i.e. Cephes `ndtr`).

    fn rel_err(got: f64, want: f64) -> f64 {
        ((got - want) / want).abs()
    }

    #[test]
    fn test_erfc_reference_values_relative() {
        let cases = [
            (0.5, 0.4795001221869535),
            (1.0, 0.15729920705028513),
            (2.0, 0.004677734981047266),
            (3.0, 2.209049699858544e-5),
            (5.0, 1.537459794428035e-12),
            (10.0, 2.088487583762545e-45),
            (-1.0, 1.8427007929497148),
        ];
        for (x, want) in cases {
            let got = erfc(x);
            assert!(
                rel_err(got, want) < 1e-14,
                "erfc({x}) = {got:e}, want {want:e}"
            );
        }
    }

    #[test]
    fn test_sf_upper_tail_relative_precision() {
        // P(Z > z) — the PPM table of a process capability study, where an
        // absolute-error approximation leaves no significant digits (the
        // A&S 26.2.17 CDF was off by 3.6e-3 relative at z = 6).
        let cases = [
            (3.0, 0.001349898031630093),
            (4.0, 3.167124183311986e-05),
            (5.0, 2.866515718791933e-07),
            (6.0, 9.865876450376948e-10),
            (6.5, 4.016000583859089e-11),
            (7.0, 1.279812543885835e-12),
        ];
        for (z, want) in cases {
            let got = standard_normal_sf(z);
            assert!(
                rel_err(got, want) < 1e-13,
                "sf({z}) = {got:e}, want {want:e}"
            );
            // The lower tail is the mirror image, through the CDF.
            let lower = standard_normal_cdf(-z);
            assert!(rel_err(lower, want) < 1e-13, "cdf(-{z}) = {lower:e}");
        }
    }

    #[test]
    fn test_sf_vs_one_minus_cdf_cancellation() {
        // Negative control for the tail tests above: the subtraction form
        // really does lose the tail, so those tests are not vacuous.
        let z = 9.0;
        assert_eq!(1.0 - standard_normal_cdf(z), 0.0);
        assert!(standard_normal_sf(z) > 0.0);
    }

    #[test]
    fn test_cdf_and_sf_sum_to_one() {
        for i in -80..=80 {
            let x = i as f64 * 0.1;
            let sum = standard_normal_cdf(x) + standard_normal_sf(x);
            assert!((sum - 1.0).abs() < 1e-15, "cdf+sf at {x} = {sum}");
        }
    }

    #[test]
    fn test_inverse_reference_values() {
        let cases = [
            (0.975, 1.959963984540054),
            (0.8413447460685429, 1.0),
            (0.1, -1.2815515655446004),
            (1e-10, -6.361340902404056),
        ];
        for (p, want) in cases {
            let got = inverse_normal_cdf(p);
            assert!((got - want).abs() < 1e-13, "Φ⁻¹({p}) = {got}, want {want}");
        }
    }

    #[test]
    fn test_inverse_sf_roundtrip_deep_tail() {
        // Every tail probability down to 1e-300 maps back to itself.
        for k in 1..=300 {
            let q = 10f64.powi(-k);
            let z = inverse_normal_sf(q);
            assert!(z.is_finite() && z > 0.0, "isf(1e-{k}) = {z}");
            let back = standard_normal_sf(z);
            assert!(rel_err(back, q) < 1e-11, "sf(isf(1e-{k})) = {back:e}");
        }
    }

    #[test]
    fn test_inverse_sf_edges() {
        assert_eq!(inverse_normal_sf(0.5), 0.0);
        assert_eq!(inverse_normal_sf(0.0), f64::INFINITY);
        assert_eq!(inverse_normal_sf(1.0), f64::NEG_INFINITY);
        assert!(inverse_normal_sf(f64::NAN).is_nan());
        assert!(inverse_normal_sf(-0.1).is_nan());
    }

    #[test]
    fn test_erf_erfc_special_values() {
        assert!(erfc(f64::NAN).is_nan());
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(f64::NEG_INFINITY), -1.0);
        assert_eq!(erfc(f64::INFINITY), 0.0);
        assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
        assert_eq!(erfc(30.0), 0.0);
        assert_eq!(erfc(-30.0), 2.0);
        // Tiny arguments stay on the linear term.
        let tiny = 1e-300;
        assert!(rel_err(erf(tiny), tiny * 2.0 / std::f64::consts::PI.sqrt()) < 1e-15);
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
    // --- t quantile against published critical values ---

    #[test]
    fn test_t_quantile_nist_critical_values() {
        // NIST/SEMATECH e-Handbook §1.3.6.7.2 (upper critical values).
        let cases = [
            (0.975, 1.0, 12.7062),
            (0.975, 5.0, 2.5706),
            (0.975, 10.0, 2.2281),
            (0.975, 30.0, 2.0423),
            (0.95, 8.0, 1.8595),
            (0.995, 2.0, 9.9248),
        ];
        for (p, df, want) in cases {
            let got = t_distribution_quantile(p, df);
            assert!(
                (got - want).abs() < 5e-4,
                "t({p}; {df}) = {got}, want {want}"
            );
        }
    }

    // --- chi-squared quantile ---

    #[test]
    fn test_chi2_quantile_nist_critical_values() {
        // NIST/SEMATECH e-Handbook §1.3.6.7.4 (upper critical values).
        let cases = [
            (0.95, 1.0, 3.841),
            (0.975, 1.0, 5.024),
            (0.975, 2.0, 7.378),
            (0.999, 1.0, 10.828),
            (0.999, 2.0, 13.816),
            (0.999, 5.0, 20.515),
            (0.975, 10.0, 20.483),
            (0.05, 10.0, 3.940),
        ];
        for (p, k, want) in cases {
            let got = chi_squared_quantile(p, k);
            assert!(
                (got - want).abs() < 1e-3,
                "chi2({p}; {k}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_chi2_quantile_roundtrip() {
        for &k in &[0.5, 1.0, 2.0, 3.0, 7.5, 30.0, 200.0] {
            for &p in &[1e-6, 0.01, 0.3, 0.5, 0.9, 0.999, 1.0 - 1e-9] {
                let x = chi_squared_quantile(p, k);
                let back = chi_squared_cdf(x, k);
                assert!(
                    (back - p).abs() < 1e-10 * p.max(1e-3),
                    "k={k} p={p}: x={x}, cdf={back}"
                );
            }
        }
    }

    #[test]
    fn test_chi2_quantile_nan() {
        assert!(chi_squared_quantile(0.0, 2.0).is_nan());
        assert!(chi_squared_quantile(1.0, 2.0).is_nan());
        assert!(chi_squared_quantile(0.5, 0.0).is_nan());
        assert!(chi_squared_quantile(f64::NAN, 2.0).is_nan());
    }

    // --- noncentral t ---

    /// Oracle: composite Simpson integration of `Φ(t·√(u/df) − δ)` over the
    /// χ²(df) density (200 000 panels), computed independently of AS 243.
    #[test]
    fn test_noncentral_t_against_integration_oracle() {
        let cases = [
            (2.306, 8.0, 2.6667, 0.351_984_735_833_218),
            (-2.306, 8.0, 2.6667, 6.505_998_809_435e-6),
            (1.0, 10.0, 1.0, 0.490_240_051_395_437),
            (2.0, 5.0, 1.5, 0.631_449_247_255_644),
            (-1.0, 3.0, 0.5, 0.092_121_209_814_356),
            (3.0, 20.0, 2.0, 0.807_947_708_148_032),
            (0.0, 4.0, 1.0, 0.158_655_253_931_452),
        ];
        for (t, df, d, want) in cases {
            let got = noncentral_t_cdf(t, df, d);
            assert!(
                (got - want).abs() < 1e-9,
                "F({t}; {df}, {d}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_noncentral_t_zero_delta_is_central() {
        for &df in &[1.0, 2.5, 8.0, 40.0] {
            for &t in &[-4.0, -1.0, -0.2, 0.3, 1.7, 6.0] {
                let a = noncentral_t_cdf(t, df, 0.0);
                let b = t_distribution_cdf(t, df);
                assert!((a - b).abs() < 1e-10, "df={df} t={t}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn test_noncentral_t_two_sided_power_of_a_2x3_replicated_factorial() {
        // 2^3 with 2 replicates: N = 16, full model leaves df_E = 8.
        // Effect 2.0, σ = 1.5 ⇒ δ = effect / (2σ/√N) = 2.667. Power ≈ 0.648.
        let df = 8.0;
        let delta = 2.0 / (2.0 * 1.5 / 16f64.sqrt());
        let c = t_distribution_quantile(0.975, df);
        let power = 1.0 - noncentral_t_cdf(c, df, delta) + noncentral_t_cdf(-c, df, delta);
        assert!((power - 0.648_009_710_6).abs() < 1e-6, "power = {power}");
    }

    #[test]
    fn test_noncentral_t_monotone_in_t_and_delta() {
        let mut prev = 0.0;
        for i in -40..=40 {
            let t = i as f64 * 0.25;
            let c = noncentral_t_cdf(t, 6.0, 1.3);
            assert!(c >= prev - 1e-14, "not monotone in t at {t}");
            prev = c;
        }
        // Larger δ shifts mass right: CDF at fixed t decreases.
        assert!(noncentral_t_cdf(1.0, 6.0, 2.0) < noncentral_t_cdf(1.0, 6.0, 1.0));
    }

    #[test]
    fn test_noncentral_t_invalid() {
        assert!(noncentral_t_cdf(1.0, 0.0, 1.0).is_nan());
        assert!(noncentral_t_cdf(f64::NAN, 5.0, 1.0).is_nan());
        assert!(noncentral_t_cdf(1.0, 5.0, f64::INFINITY).is_nan());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn cdf_in_zero_one(x in -6.0f64..6.0) {
            let c = standard_normal_cdf(x);
            prop_assert!((0.0..=1.0).contains(&c), "CDF({x}) = {c} out of [0,1]");
        }

        #[test]
        fn cdf_is_monotonic(x1 in -6.0f64..6.0, x2 in -6.0f64..6.0) {
            let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
            prop_assert!(
                standard_normal_cdf(lo) <= standard_normal_cdf(hi) + 1e-15,
                "CDF not monotonic"
            );
        }

        #[test]
        fn inverse_roundtrip(p in 0.001f64..0.999) {
            let z = inverse_normal_cdf(p);
            let p_back = standard_normal_cdf(z);
            let err = (p_back - p).abs();
            prop_assert!(err < 1e-15, "roundtrip error {} for p={}", err, p);
        }

        #[test]
        fn sf_tail_roundtrip(z in 0.0f64..37.0) {
            let q = standard_normal_sf(z);
            let back = inverse_normal_sf(q);
            prop_assert!(
                (back - z).abs() < 1e-12 * z.max(1.0),
                "isf(sf({z})) = {back}"
            );
        }

        #[test]
        fn erfc_reflection(x in 0.0f64..27.0) {
            prop_assert!((erfc(-x) - (2.0 - erfc(x))).abs() < 1e-15);
        }

        #[test]
        fn pdf_is_non_negative(x in -10.0f64..10.0) {
            prop_assert!(standard_normal_pdf(x) >= 0.0);
        }

        #[test]
        fn inc_beta_in_01(x in 0.01f64..0.99, a in 0.5f64..10.0, b in 0.5f64..10.0) {
            let result = regularized_incomplete_beta(x, a, b);
            prop_assert!(
                (0.0..=1.0).contains(&result),
                "I_{x}({a},{b}) = {result} out of [0,1]"
            );
        }

        #[test]
        fn inc_beta_complementary(x in 0.01f64..0.99, a in 0.5f64..10.0, b in 0.5f64..10.0) {
            // I_x(a,b) + I_{1-x}(b,a) = 1
            let ix = regularized_incomplete_beta(x, a, b);
            let i1x = regularized_incomplete_beta(1.0 - x, b, a);
            prop_assert!(
                (ix + i1x - 1.0).abs() < 1e-8,
                "complementary: {ix} + {i1x} != 1"
            );
        }

        #[test]
        fn t_cdf_in_01(t in -10.0f64..10.0, df in 1.0f64..100.0) {
            let c = t_distribution_cdf(t, df);
            prop_assert!(
                (0.0..=1.0).contains(&c),
                "t_cdf({t}, {df}) = {c} out of [0,1]"
            );
        }

        #[test]
        fn t_cdf_symmetric(t in 0.01f64..10.0, df in 1.0f64..50.0) {
            let sum = t_distribution_cdf(t, df) + t_distribution_cdf(-t, df);
            prop_assert!(
                (sum - 1.0).abs() < 1e-6,
                "t symmetry: {sum} != 1 for t={t}, df={df}"
            );
        }

        #[test]
        fn erf_odd_symmetry(x in 0.01f64..5.0) {
            let sum = erf(x) + erf(-x);
            prop_assert!(sum.abs() < 1e-6, "erf odd symmetry: {sum} for x={x}");
        }

        #[test]
        fn chi2_cdf_in_01(x in 0.01f64..50.0, k in 0.5f64..20.0) {
            let c = chi_squared_cdf(x, k);
            prop_assert!(
                (0.0..=1.0).contains(&c),
                "chi2_cdf({x}, {k}) = {c} out of [0,1]"
            );
        }
    }
}

/// Against SciPy 1.17.1 (`scipy.stats` call named at each value).
#[cfg(test)]
mod scipy_reference {
    use super::*;

    /// `got` within `rel` of `want`, relative (absolute below 1).
    fn close(got: f64, want: f64, rel: f64, what: &str) {
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() <= rel * scale,
            "{what}: got {got}, reference {want} (relative tolerance {rel})"
        );
    }

    #[test]
    fn t_f_and_chi_squared_distributions_reproduce_scipy() {
        // t.cdf(1.7, 6), t.cdf(-2.3, 3.5), t.cdf(0.4, 40)
        close(
            t_distribution_cdf(1.7, 6.0),
            0.9299801726875513,
            TOL,
            "t cdf",
        );
        close(
            t_distribution_cdf(-2.3, 3.5),
            0.04619672248986906,
            TOL,
            "t cdf, fractional df",
        );
        close(
            t_distribution_cdf(0.4, 40.0),
            0.6543576534354631,
            TOL,
            "t cdf, df 40",
        );
        // f.cdf(2.5, 3, 12), f.cdf(0.7, 5, 7.5)
        close(
            f_distribution_cdf(2.5, 3.0, 12.0),
            0.8908452876049937,
            TOL,
            "F cdf",
        );
        close(
            f_distribution_cdf(0.7, 5.0, 7.5),
            0.3600807359231225,
            TOL,
            "F cdf, fractional df",
        );
        // f.ppf(0.95, 3, 12), f.ppf(0.01, 5, 7.5)
        close(
            f_distribution_quantile(0.95, 3.0, 12.0),
            3.490294819497605,
            TOL,
            "F quantile",
        );
        close(
            f_distribution_quantile(0.01, 5.0, 7.5),
            0.09645738537247153,
            TOL,
            "F quantile, low tail",
        );
        // chi2.cdf(7.3, 4), chi2.cdf(0.5, 2.5), chi2.cdf(30, 20)
        close(
            chi_squared_cdf(7.3, 4.0),
            0.8791412511787876,
            TOL,
            "chi-squared cdf",
        );
        close(
            chi_squared_cdf(0.5, 2.5),
            0.13611637154141112,
            TOL,
            "chi-squared cdf, fractional k",
        );
        close(
            chi_squared_cdf(30.0, 20.0),
            0.9301463393005901,
            TOL,
            "chi-squared cdf, k 20",
        );
    }

    const TOL: f64 = 1e-12;
}

/// Upper tails against SciPy 1.17.1 (`scipy.stats.chi2.sf`, `f.sf`, `t.sf`), at
/// sizes where the complement of the CDF has no digits left.
#[cfg(test)]
mod scipy_tail_reference {
    use super::*;

    fn close(got: f64, want: f64, what: &str) {
        assert!(
            (got - want).abs() <= 1e-9 * want.abs(),
            "{what}: got {got:e}, SciPy {want:e}"
        );
    }

    #[test]
    fn chi_squared_sf_reproduces_scipy() {
        for (x, k, want) in [
            (93.0, 2.0, 6.387142293058413e-21),
            (200.0, 5.0, 2.840622898641554e-41),
            (50.0, 1.0, 1.537459794428033e-12),
            (1000.0, 10.0, 1.870290720915977e-208),
            (3.841, 1.0, 0.050013683763956804),
            (0.5, 3.0, 0.9188914116546758),
            (7.5, 2.5, 0.038169088215206766),
            (30.0, 40.0, 0.875218784967475),
        ] {
            close(chi_squared_sf(x, k), want, &format!("chi2.sf({x}, {k})"));
        }
    }

    #[test]
    fn f_distribution_sf_reproduces_scipy() {
        for (x, d1, d2, want) in [
            (1422.4107710431, 4.0, 34.0, 2.5558624671479177e-37),
            (88.622245, 2.0, 34.0, 3.264299228911802e-14),
            (28.1253056582, 8.0, 30.0, 6.579056051547977e-12),
            (3.0, 5.0, 10.0, 0.0655575620938441),
            (0.2, 6.0, 12.0, 0.9703353508362454),
            (250.0, 1.0, 1000.0, 1.9705760832196082e-50),
            (11.0639154838, 4.0, 8.0, 0.00241021703771129),
        ] {
            close(
                f_distribution_sf(x, d1, d2),
                want,
                &format!("f.sf({x}, {d1}, {d2})"),
            );
        }
    }

    #[test]
    fn t_distribution_sf_reproduces_scipy() {
        for (t, df, want) in [
            (30.0, 10.0, 1.980896171015662e-11),
            (8.0, 100.0, 1.1364324038640345e-12),
            (-2.5, 7.0, 0.9795038907071235),
            (1.0, 1.5, 0.22556768363835517),
            (60.0, 5.0, 1.2168188794991892e-08),
            (-40.0, 12.0, 0.9999999999999808),
            (0.3, 2.0, 0.3962428304200888),
        ] {
            close(t_distribution_sf(t, df), want, &format!("t.sf({t}, {df})"));
        }
    }

    #[test]
    fn survival_and_cdf_add_to_one_where_both_have_digits() {
        for (x, k) in [(0.5, 3.0), (4.0, 2.0), (12.0, 7.0)] {
            assert!((chi_squared_cdf(x, k) + chi_squared_sf(x, k) - 1.0).abs() < 1e-13);
        }
        for (x, d1, d2) in [(0.7, 3.0, 9.0), (2.0, 5.0, 5.0), (6.0, 2.0, 20.0)] {
            assert!(
                (f_distribution_cdf(x, d1, d2) + f_distribution_sf(x, d1, d2) - 1.0).abs() < 1e-13
            );
        }
        for (t, df) in [(-1.2, 4.0), (0.8, 9.0), (3.0, 3.0)] {
            assert!((t_distribution_cdf(t, df) + t_distribution_sf(t, df) - 1.0).abs() < 1e-13);
        }
    }

    #[test]
    fn invalid_arguments_are_nan_and_the_far_side_is_one() {
        assert!(chi_squared_sf(1.0, 0.0).is_nan() && chi_squared_sf(f64::NAN, 2.0).is_nan());
        assert!(f_distribution_sf(1.0, 0.0, 3.0).is_nan());
        assert!(t_distribution_sf(1.0, -1.0).is_nan());
        assert_eq!(chi_squared_sf(0.0, 2.0), 1.0);
        assert_eq!(chi_squared_sf(f64::INFINITY, 2.0), 0.0);
        assert_eq!(f_distribution_sf(f64::INFINITY, 2.0, 3.0), 0.0);
        assert_eq!(t_distribution_sf(f64::INFINITY, 4.0), 0.0);
    }

    #[test]
    fn the_normal_limits_of_the_incomplete_beta_and_gamma_match_the_exact_functions_at_large_shapes(
    ) {
        // At shapes 1e6 and 1e7 the exact functions still hold their digits and
        // the limits are within their O(1/shape) error.
        for (a, b) in [(1e6_f64, 1e6_f64), (1e6, 3e6), (2e7, 1e7)] {
            let mean = a / (a + b);
            let sd = (mean * (1.0 - mean) / (a + b + 1.0)).sqrt();
            for k in [-3.0, -1.0, -0.2, 0.0, 0.7, 2.0] {
                let x = mean + k * sd;
                let exact = regularized_incomplete_beta(x, a, b);
                let limit = beta_normal_approximation(x, a, b);
                assert!(
                    (exact - limit).abs() < 2e-5,
                    "beta({a},{b}) at {k} sd: {exact} vs {limit}"
                );
            }
        }
        for a in [1e6_f64, 1e7] {
            for k in [-3.0, -1.0, 0.0, 1.0, 2.5] {
                let x = a + k * a.sqrt();
                let exact = regularized_lower_gamma(a, x);
                let limit = standard_normal_cdf(gamma_normal_score(a, x));
                assert!(
                    (exact - limit).abs() < 2e-4,
                    "gamma({a}) at {k} sd: {exact} vs {limit}"
                );
            }
        }
    }

    #[test]
    fn the_incomplete_beta_and_gamma_answer_at_shapes_near_1e300() {
        assert!((regularized_incomplete_beta(0.5, 1e300, 1e300) - 0.5).abs() < 1e-12);
        assert_eq!(regularized_incomplete_beta(0.4, 1e300, 1e300), 0.0);
        assert_eq!(regularized_incomplete_beta(0.6, 1e300, 1e300), 1.0);
        assert!((regularized_lower_gamma(1e300, 1e300) - 0.5).abs() < 1e-9);
        assert!((regularized_lower_gamma(1e300, 1.0) - 0.0).abs() < 1e-12);
        assert!((regularized_upper_gamma(1e300, 1e300) - 0.5).abs() < 1e-9);
        assert_eq!(regularized_lower_gamma(1e300, f64::INFINITY), 1.0);
    }

    /// With one shape equal to 1 the incomplete beta has a closed form --
    /// `I_x(a, 1) = x^a` and `I_x(1, b) = 1 - (1-x)^b` -- which holds at any other
    /// shape however large: the oracle for a shape that is large on one side only
    /// (the normal limit needs both). The arguments are doubles, and the oracle
    /// is computed from those exact values (`ln_1p(x - 1)` for the first, with
    /// `x - 1` exact, `exp_m1` for the second), so it carries no rounding of its own.
    #[test]
    fn the_incomplete_beta_with_one_shape_one_matches_its_closed_form() {
        let mut worst = (0.0_f64, String::new());
        for big in [1e2_f64, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e12, 1e14] {
            for c in [1e-3, 0.05, 0.3, 1.0, 2.5, 8.0] {
                // x = 1 - c/a: the point where x^a = exp(-c) -- the body of the law.
                let x = 1.0 - c / big;
                let want = (big * (x - 1.0).ln_1p()).exp();
                let got = regularized_incomplete_beta(x, big, 1.0);
                let rel = (got - want).abs() / want;
                if rel > worst.0 {
                    worst = (rel, format!("I_{x}({big}, 1) = {got}, want {want}"));
                }
                // x = c/b: the point where (1-x)^b = exp(-c).
                let x = c / big;
                let want = -(big * (-x).ln_1p()).exp_m1();
                let got = regularized_incomplete_beta(x, 1.0, big);
                let rel = (got - want).abs() / want;
                if rel > worst.0 {
                    worst = (rel, format!("I_{x}(1, {big}) = {got}, want {want}"));
                }
            }
        }
        assert!(
            worst.0 < 1e-9,
            "worst relative error {:e}: {}",
            worst.0,
            worst.1
        );
    }

    /// `ln B(a, 1) = -ln a` and `ln B(a, 2) = -ln(a (a + 1))` at any `a`: the
    /// log-beta of one large shape is small, and `ln Γ(a) − ln Γ(a + 1)` of two
    /// large numbers is not the way to get it.
    #[test]
    fn ln_beta_with_a_small_shape_holds_its_digits_at_a_large_one() {
        for a in [10.0_f64, 1e3, 1e6, 1e9, 1e12, 1e15, 1e100, 1e300] {
            let one = ln_beta(a, 1.0);
            assert!((one + a.ln()).abs() < 1e-13 * a.ln(), "B({a}, 1): {one}");
            assert_eq!(one, ln_beta(1.0, a), "symmetric at {a}");
            let two = ln_beta(2.0, a);
            let want = -(a.ln() + (a + 1.0).ln());
            assert!((two - want).abs() < 1e-13 * want.abs(), "B(2, {a}): {two}");
        }
        // Both large: the three Stirling terms agree with the direct sum where
        // the direct sum still has its digits.
        for (a, b) in [(10.0_f64, 10.0_f64), (30.0, 700.0), (1e3, 2e3)] {
            let direct = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
            assert!(
                (ln_beta(a, b) - direct).abs() < 1e-12 * direct.abs(),
                "B({a}, {b})"
            );
        }
    }

    /// `I_x(a, 1) = x^a` and `I_x(a, 2) = x^a (1 + a y)` in closed form: the large-`a`
    /// expansion built from `y` alone has to hold them at any `a`, including those
    /// where a continued fraction handed a rounded `x` cannot.
    #[test]
    fn the_large_a_expansion_matches_the_closed_forms_at_b_one_and_two() {
        for a in [15.0_f64, 100.0, 1e4, 1e7, 5e9] {
            for t in [0.05_f64, 0.5, 1.0, 2.0, 5.0] {
                // y chosen so that z = a t stays moderate: the lower tail of x^a.
                let y = (-t / a).exp_m1().abs();
                let ln_x = (-y).ln_1p();
                let one = beta_large_a_expansion(y, a, 1.0).expect("settles at b = 1");
                let want_one = (a * ln_x).exp();
                assert!(
                    (one - want_one).abs() <= 2e-14 * want_one,
                    "I(a={a}, 1, y={y:e}) = {one:e}, want {want_one:e}"
                );
                let two = beta_large_a_expansion(y, a, 2.0).expect("settles at b = 2");
                let want_two = want_one * (1.0 + a * y);
                assert!(
                    (two - want_two).abs() <= 2e-14 * want_two,
                    "I(a={a}, 2, y={y:e}) = {two:e}, want {want_two:e}"
                );
            }
        }
    }

    /// Where both the expansion and the continued fraction are good (`x` an exact
    /// dyadic fraction, so `1 - x` is exact) they are the same function.
    #[test]
    fn the_large_a_expansion_agrees_with_the_continued_fraction_where_both_hold() {
        for (a, b) in [(15.0_f64, 0.5_f64), (40.0, 3.0), (300.0, 7.0), (2e3, 0.1)] {
            let mean = (a + 1.0) / (a + b + 2.0);
            for k in 512..1024 {
                let x = k as f64 / 1024.0;
                if x >= mean {
                    break;
                }
                let y = 1.0 - x;
                let series = beta_large_a_expansion(y, a, b).expect("settles");
                let fraction = beta_by_continued_fraction(x, y, a, b);
                // Compare where the value is not lost below a double's smallest
                // normal numbers.
                if fraction > 1e-200 {
                    assert!(
                        (series - fraction).abs() <= 1e-12 * fraction,
                        "a={a}, b={b}, x={x}: expansion {series:e}, fraction {fraction:e}"
                    );
                }
            }
        }
    }

    /// The deviation form of `x^a y^b / (a B(a, b))` is the same number as the
    /// three-term form where that one still has its digits, and is symmetric in
    /// the pair (`x ≤ y` and `x > y` take different routes to the gap).
    #[test]
    fn the_density_prefix_in_deviation_form_agrees_with_the_direct_one() {
        for (a, b) in [
            (12.0_f64, 12.0_f64),
            (20.0, 45.0),
            (150.0, 60.0),
            (1e3, 2e3),
        ] {
            for k in [1, 3, 17, 100, 300, 511, 512, 513, 700, 900, 1020] {
                let x = k as f64 / 1024.0;
                let y = 1.0 - x;
                let direct = a * x.ln() + b * y.ln() - ln_beta(a, b) - a.ln();
                let deviation = ln_beta_density_over_a(x, y, a, b);
                let tol = 1e-12 + 2e-16 * a.max(b) * (a.max(b)).ln();
                assert!(
                    (deviation - direct).abs() <= tol,
                    "a={a}, b={b}, x={x}: deviation {deviation}, direct {direct}"
                );
                // Mirrored, the density is divided by `b` instead of `a`.
                let mirrored = ln_beta_density_over_a(y, x, b, a) + b.ln() - a.ln();
                assert!(
                    (deviation - mirrored).abs() <= 1e-13 + 4e-16 * deviation.abs(),
                    "a={a}, b={b}, x={x}: {deviation} against mirrored {mirrored}"
                );
            }
        }
    }

    #[test]
    fn the_large_a_expansion_refuses_what_it_cannot_do() {
        assert!(beta_large_a_expansion(0.0, 100.0, 1.0).is_none());
        assert!(beta_large_a_expansion(1.0, 100.0, 1.0).is_none());
        assert!(beta_large_a_expansion(0.1, 100.0, 0.0).is_none());
        assert!(beta_large_a_expansion(0.1, f64::INFINITY, 1.0).is_none());
        assert!(beta_large_a_expansion(f64::NAN, 100.0, 1.0).is_none());
    }
}
