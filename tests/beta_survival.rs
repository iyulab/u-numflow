//! `BetaDistribution::survival` is the upper tail formed directly.
//!
//! Beta(1, b) has the closed form `S(x) = (1 − x)^b = exp(b ln(1 − x))`; with a
//! large `b` the mass is near 0 and `1 − x` rounded from a small `x` would be
//! multiplied by `b`.

use u_numflow::distributions::BetaDistribution;

#[test]
fn the_survival_function_of_a_beta_with_a_large_second_shape_matches_its_closed_form() {
    for b in [1e3_f64, 1e6, 1e9, 1e11, 1e12] {
        let d = BetaDistribution::new(1.0, b).expect("valid shape");
        for c in [0.01_f64, 0.5, 1.0, 3.0, 10.0, 30.0] {
            let x = c / b;
            let want = (b * (-x).ln_1p()).exp();
            let got = d.survival(x);
            assert!(
                (got - want).abs() <= 1e-13 * want,
                "Beta(1, {b:e}).survival({x:e}) = {got:e}, want {want:e}"
            );
            // The CDF and the survival function sum to one where both have digits.
            if c <= 3.0 {
                assert!((d.cdf(x) + got - 1.0).abs() < 1e-13, "b = {b:e}, x = {x:e}");
            }
        }
    }
}

#[test]
fn the_survival_function_is_zero_and_one_at_the_ends() {
    let d = BetaDistribution::new(2.0, 3.0).expect("valid shape");
    assert_eq!(d.survival(0.0), 1.0);
    assert_eq!(d.survival(-1.0), 1.0);
    assert_eq!(d.survival(1.0), 0.0);
    assert_eq!(d.survival(2.0), 0.0);
}
