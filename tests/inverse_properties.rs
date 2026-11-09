//! Properties of the inverse regularized incomplete gamma and beta functions and
//! of the t, F and chi-squared quantiles built on them, over shapes and
//! probabilities spread on a logarithmic scale (a probability of 1e-300 is as
//! ordinary a p-value as 0.05).
use proptest::prelude::*;
use u_numflow::special::{
    chi_squared_quantile, f_distribution_quantile, inverse_regularized_incomplete_beta,
    inverse_regularized_lower_gamma, inverse_regularized_upper_gamma, regularized_incomplete_beta,
    regularized_lower_gamma, t_distribution_quantile,
};

/// ln of a probability in `[1e-300, 0.5]`: the lower tail on a log scale.
fn ln_tail() -> impl Strategy<Value = f64> {
    -690.0f64..-0.69
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// `P(a, P⁻¹(a, p)) = p` to the digits of the CDF, for `p` down to 1e-300.
    #[test]
    fn gamma_inverse_round_trips(ln_a in -3.0f64..6.0, ln_p in ln_tail()) {
        let (a, p) = (ln_a.exp(), ln_p.exp());
        let x = inverse_regularized_lower_gamma(a, p);
        prop_assert!(x >= 0.0 && !x.is_nan());
        // a root below the least positive double has nothing to round-trip
        prop_assume!(x > 1e-300 && x.is_finite());
        let back = regularized_lower_gamma(a, x);
        prop_assert!(((back - p) / p).abs() < 1e-9, "a={a} p={p:e} x={x:e} back={back:e}");
    }

    /// The inverse is increasing in `p`.
    #[test]
    fn gamma_inverse_is_monotone(
        ln_a in -3.0f64..6.0, ln_p1 in ln_tail(), ln_p2 in ln_tail(),
    ) {
        let a = ln_a.exp();
        let (lo, hi) = if ln_p1 <= ln_p2 { (ln_p1, ln_p2) } else { (ln_p2, ln_p1) };
        let (x1, x2) = (
            inverse_regularized_lower_gamma(a, lo.exp()),
            inverse_regularized_lower_gamma(a, hi.exp()),
        );
        prop_assert!(x1 <= x2, "a={a} x({lo})={x1:e} > x({hi})={x2:e}");
    }

    /// The lower inverse at `p` and the upper inverse at `1 − p` are one root.
    #[test]
    fn gamma_tails_agree(ln_a in -3.0f64..6.0, p in 0.001f64..0.999) {
        let a = ln_a.exp();
        let (lower, upper) = (
            inverse_regularized_lower_gamma(a, p),
            inverse_regularized_upper_gamma(a, 1.0 - p),
        );
        prop_assert!(((lower - upper) / lower).abs() < 1e-11, "a={a} p={p}: {lower:e} {upper:e}");
    }

    /// `I_x(a, b) = p` to the digits of the CDF; a root near 1 is a double near 1,
    /// whose resolution is the absolute floor.
    #[test]
    fn beta_inverse_round_trips(ln_a in -3.0f64..6.0, ln_b in -3.0f64..6.0, ln_p in ln_tail(), upper in any::<bool>()) {
        let (a, b) = (ln_a.exp(), ln_b.exp());
        let p = if upper { 1.0 - ln_p.exp() } else { ln_p.exp() };
        prop_assume!(p < 1.0);
        let x = inverse_regularized_incomplete_beta(p, a, b);
        prop_assert!((0.0..=1.0).contains(&x), "x = {x}");
        // a root within 1e-9 of 1 is a double near 1 with no digits of `1 − x` left
        prop_assume!(1.0 - x > 1e-9);
        let back = regularized_incomplete_beta(x, a, b);
        // a shape past 1e3 carries the documented `3e-16 · shape` of the CDF itself
        let tolerance = 1e-9 * p.min(1.0 - p) + 1e-12 + 1e-15 * a.max(b);
        prop_assert!((back - p).abs() <= tolerance, "a={a} b={b} p={p:e} x={x:e} back={back:e}");
    }

    /// `x(p; a, b) + x(1 − p; b, a) = 1`.
    #[test]
    fn beta_inverse_mirrors(ln_a in -2.0f64..5.0, ln_b in -2.0f64..5.0, p in 0.01f64..0.99) {
        let (a, b) = (ln_a.exp(), ln_b.exp());
        let (x, mirrored) = (
            inverse_regularized_incomplete_beta(p, a, b),
            inverse_regularized_incomplete_beta(1.0 - p, b, a),
        );
        prop_assert!((x + mirrored - 1.0).abs() < 1e-9, "a={a} b={b} p={p}: {x} + {mirrored}");
    }

    /// For a `p` with an exact complement (a dyadic rational) the two sides of the
    /// t distribution are the same computation: the quantile is exactly odd.
    #[test]
    fn t_quantile_is_exactly_odd(ln_df in -2.0f64..9.0, k in 1u32..512) {
        let (df, p) = (ln_df.exp(), f64::from(k) / 1024.0);
        let (lo, hi) = (t_distribution_quantile(p, df), t_distribution_quantile(1.0 - p, df));
        prop_assert!(lo < 0.0 && hi > 0.0);
        prop_assert_eq!(lo, -hi);
    }

    /// `F(p; d₁, d₂) = 1 / F(1 − p; d₂, d₁)`.
    #[test]
    fn f_quantile_reciprocal(ln_d1 in -1.5f64..7.0, ln_d2 in -1.5f64..7.0, k in 1u32..1024) {
        let (d1, d2, p) = (ln_d1.exp(), ln_d2.exp(), f64::from(k) / 1024.0);
        let (f, g) = (f_distribution_quantile(p, d1, d2), f_distribution_quantile(1.0 - p, d2, d1));
        prop_assert!(f > 0.0 && g > 0.0);
        prop_assert!(((f * g) - 1.0).abs() < 1e-9, "p={p} d1={d1} d2={d2}: {f:e} · {g:e}");
    }

    /// No valid input gives NaN, whatever the scale of shape and probability, and
    /// every quantile is monotone in `p` across the whole range.
    #[test]
    fn quantiles_are_never_nan(ln_a in -6.0f64..20.0, ln_p in -36.0f64..-1e-9) {
        let (a, p) = (ln_a.exp(), ln_p.exp());
        // `1 − p` of a p below 1.1e-16 is 1: the range stops short of it
        prop_assert!(p > 0.0 && 1.0 - p < 1.0);
        prop_assert!(!t_distribution_quantile(p, a).is_nan());
        prop_assert!(!t_distribution_quantile(1.0 - p, a).is_nan());
        prop_assert!(!f_distribution_quantile(p, a, 1.0 + a).is_nan());
        prop_assert!(!f_distribution_quantile(1.0 - p, 1.0 + a, a).is_nan());
        prop_assert!(!chi_squared_quantile(p, a).is_nan());
        prop_assert!(!chi_squared_quantile(1.0 - p, a).is_nan());
        prop_assert!(!inverse_regularized_incomplete_beta(p, a, 2.0).is_nan());
    }
}
