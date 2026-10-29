//! Local minimisation of a function of several variables without derivatives.
//!
//! # Algorithm
//!
//! The Nelder–Mead simplex method: a simplex of n + 1 points is reflected,
//! expanded, contracted or shrunk towards its best vertex until its values (and
//! its extent) fall below the tolerances. The coefficients adapt to the
//! dimension (Gao & Han 2012 — reflection 1, expansion 1 + 2/n, contraction
//! 0.75 − 1/(2n), shrink 1 − 1/n), which keeps the method from stalling in more
//! than a few dimensions; in one or two they are the classical 1, 2, ½, ½. The
//! initial simplex steps 5 % of each non-zero coordinate (0.00025 for a zero
//! one), as SciPy's `minimize(method="Nelder-Mead")` does.
//!
//! A value that is not finite (NaN, ±∞) is treated as +∞, so an objective can
//! refuse a point by returning NaN.
//!
//! # References
//!
//! Nelder, J.A. & Mead, R. (1965). "A simplex method for function
//! minimization", *The Computer Journal* 7(4), 308-313.
//! Lagarias, J.C., Reeds, J.A., Wright, M.H. & Wright, P.E. (1998).
//! "Convergence properties of the Nelder–Mead simplex method in low
//! dimensions", *SIAM Journal on Optimization* 9(1), 112-147.
//! Gao, F. & Han, L. (2012). "Implementing the Nelder-Mead simplex algorithm
//! with adaptive parameters", *Computational Optimization and Applications*
//! 51(1), 259-277.

/// Stopping rules of [`nelder_mead`]. The tolerances are relative to the
/// problem's own scale, so the stopping point does not depend on the units of
/// the variables or the values: a value is judged against the larger of the
/// best value and the size of the values at the start, a coordinate against the
/// larger of itself and its initial step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NelderMeadOptions {
    /// Stop when every vertex's value is within `f_tolerance` times the
    /// larger of `|best|` and the largest value at the start, of the best's
    /// (and the simplex within `x_tolerance`). Default 1e-10.
    pub f_tolerance: f64,
    /// Stop when every vertex is within `x_tolerance` times the larger of the
    /// best's coordinate and that coordinate's first step, of the best in
    /// every coordinate (and the values within `f_tolerance`). Default 1e-8.
    pub x_tolerance: f64,
    /// Most objective evaluations. Default 1000 per variable, at least 2000.
    pub max_evaluations: usize,
}

impl NelderMeadOptions {
    /// The defaults for `n` variables.
    pub fn for_dimension(n: usize) -> Self {
        Self {
            f_tolerance: 1e-10,
            x_tolerance: 1e-8,
            max_evaluations: (1000 * n).max(2000),
        }
    }
}

/// The best point [`nelder_mead`] found.
#[derive(Debug, Clone, PartialEq)]
pub struct Minimum {
    /// The point.
    pub x: Vec<f64>,
    /// The objective there (+∞ when every point tried was refused).
    pub value: f64,
    /// Objective evaluations spent.
    pub evaluations: usize,
    /// Whether the tolerances were met (rather than the evaluation budget
    /// spent).
    pub converged: bool,
}

/// Minimises `f` from `x0` by the Nelder–Mead simplex method.
///
/// The search is deterministic: the same `f` and `x0` give the same point.
/// It finds a local minimum; for a function with several, start it from more
/// than one point.
///
/// # Panics
///
/// Never; an empty `x0` returns it with `f(&[])`.
///
/// # Examples
///
/// ```
/// use u_numflow::optimize::{nelder_mead, NelderMeadOptions};
///
/// // Rosenbrock's valley: minimum 0 at (1, 1).
/// let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
/// let m = nelder_mead(f, &[-1.2, 1.0], NelderMeadOptions::for_dimension(2));
/// assert!(m.converged);
/// assert!((m.x[0] - 1.0).abs() < 1e-4 && (m.x[1] - 1.0).abs() < 1e-4);
/// ```
pub fn nelder_mead<F: FnMut(&[f64]) -> f64>(
    mut f: F,
    x0: &[f64],
    options: NelderMeadOptions,
) -> Minimum {
    let n = x0.len();
    let evaluations = std::cell::Cell::new(0usize);
    let mut eval = |x: &[f64]| {
        evaluations.set(evaluations.get() + 1);
        let v = f(x);
        if v.is_finite() {
            v
        } else {
            f64::INFINITY
        }
    };
    if n == 0 {
        let value = eval(x0);
        return Minimum {
            x: Vec::new(),
            value,
            evaluations: evaluations.get(),
            converged: true,
        };
    }

    let nf = n as f64;
    let (rho, chi, psi, sigma) = if n <= 2 {
        (1.0, 2.0, 0.5, 0.5)
    } else {
        (1.0, 1.0 + 2.0 / nf, 0.75 - 1.0 / (2.0 * nf), 1.0 - 1.0 / nf)
    };

    // A coordinate that starts at zero takes a step in proportion to the other
    // coordinates, so that the first simplex does not depend on their unit.
    let start_scale = x0
        .iter()
        .filter(|v| **v != 0.0)
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    let zero_step = 0.00025 * if start_scale > 0.0 { start_scale } else { 1.0 };
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] = if v[i] != 0.0 { v[i] * 1.05 } else { zero_step };
        simplex.push(v);
    }
    let mut values: Vec<f64> = simplex.iter().map(|v| eval(v)).collect();
    // The problem's own scales, from the start: no coordinate or value is
    // compared with a fixed number.
    let x_ref: Vec<f64> = (0..n)
        .map(|i| (simplex[i + 1][i] - x0[i]).abs().max(x0[i].abs()))
        .collect();
    let f_ref = values
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    let mut order: Vec<usize> = (0..=n).collect();
    let mut converged = false;

    loop {
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let best = order[0];
        let worst = order[n];
        let f_spread = order[1..]
            .iter()
            .map(|&i| (values[i] - values[best]).abs())
            .fold(0.0, f64::max);
        let x_close = order[1..].iter().all(|&i| {
            simplex[i]
                .iter()
                .zip(&simplex[best])
                .zip(&x_ref)
                .all(|((a, b), r)| (a - b).abs() <= options.x_tolerance * b.abs().max(*r))
        });
        // An infinite best leaves `f_spread` NaN: not converged.
        if f_spread <= options.f_tolerance * values[best].abs().max(f_ref) && x_close {
            converged = true;
            break;
        }
        if evaluations.get() >= options.max_evaluations {
            break;
        }

        // Centroid of every vertex but the worst.
        let mut centroid = vec![0.0; n];
        for &i in &order[..n] {
            for (c, v) in centroid.iter_mut().zip(&simplex[i]) {
                *c += v / nf;
            }
        }
        let toward = |t: f64| -> Vec<f64> {
            centroid
                .iter()
                .zip(&simplex[worst])
                .map(|(c, w)| c + t * (c - w))
                .collect()
        };

        let xr = toward(rho);
        let fr = eval(&xr);
        let second_worst = values[order[n - 1]];
        if fr < values[best] {
            let xe = toward(rho * chi);
            let fe = eval(&xe);
            if fe < fr {
                simplex[worst] = xe;
                values[worst] = fe;
            } else {
                simplex[worst] = xr;
                values[worst] = fr;
            }
            continue;
        }
        if fr < second_worst {
            simplex[worst] = xr;
            values[worst] = fr;
            continue;
        }
        // Contract: outside when the reflection beat the worst, else inside.
        let (xc, fc, accept) = if fr < values[worst] {
            let xc = toward(rho * psi);
            let fc = eval(&xc);
            let ok = fc <= fr;
            (xc, fc, ok)
        } else {
            let xc = toward(-psi);
            let fc = eval(&xc);
            let ok = fc < values[worst];
            (xc, fc, ok)
        };
        if accept {
            simplex[worst] = xc;
            values[worst] = fc;
            continue;
        }
        // Shrink towards the best vertex.
        let anchor = simplex[best].clone();
        for &i in &order[1..] {
            for (v, a) in simplex[i].iter_mut().zip(&anchor) {
                *v = a + sigma * (*v - a);
            }
            values[i] = eval(&simplex[i]);
        }
    }

    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    Minimum {
        x: simplex[order[0]].clone(),
        value: values[order[0]],
        evaluations: evaluations.get(),
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quadratic_bowl_in_five_dimensions() {
        let centre = [1.0, -2.0, 3.0, 0.5, -0.25];
        let f = |x: &[f64]| {
            x.iter()
                .zip(&centre)
                .enumerate()
                .map(|(i, (a, c))| (i + 1) as f64 * (a - c).powi(2))
                .sum::<f64>()
        };
        let m = nelder_mead(f, &[0.0; 5], NelderMeadOptions::for_dimension(5));
        assert!(m.converged, "{m:?}");
        for (a, c) in m.x.iter().zip(&centre) {
            assert!((a - c).abs() < 1e-4, "{m:?}");
        }
    }

    #[test]
    fn the_same_start_gives_the_same_point() {
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[0] * x[1] - 1.0).powi(2);
        let a = nelder_mead(f, &[0.5, 0.5], NelderMeadOptions::for_dimension(2));
        let b = nelder_mead(f, &[0.5, 0.5], NelderMeadOptions::for_dimension(2));
        assert_eq!(a, b);
    }

    #[test]
    fn a_refused_point_is_avoided() {
        // NaN outside the unit square: the search stays inside and reaches the
        // corner minimum (1, 1) of the bowl centred outside it.
        let f = |x: &[f64]| {
            if x.iter().any(|v| !(0.0..=1.0).contains(v)) {
                f64::NAN
            } else {
                (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2)
            }
        };
        let m = nelder_mead(f, &[0.5, 0.5], NelderMeadOptions::for_dimension(2));
        assert!(m.value.is_finite());
        assert!(m.x.iter().all(|v| (0.0..=1.0).contains(v)), "{m:?}");
        assert!(m.value < 2.0 + 1e-4, "{m:?}");
    }

    #[test]
    fn the_budget_is_kept() {
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let options = NelderMeadOptions {
            max_evaluations: 30,
            ..NelderMeadOptions::for_dimension(2)
        };
        let m = nelder_mead(f, &[-1.2, 1.0], options);
        assert!(!m.converged);
        // The last step may evaluate a few points past the check.
        assert!(m.evaluations <= 30 + 2, "{}", m.evaluations);
    }

    #[test]
    fn a_change_of_scale_moves_the_minimum_with_it() {
        let f = |k: f64| {
            move |x: &[f64]| 1e6 * k * (((x[0] - 2.0 * k) / k).powi(2) + ((x[1] + k) / k).powi(2))
        };
        for k in [1e-6, 1.0, 1e6] {
            let m = nelder_mead(f(k), &[k, k], NelderMeadOptions::for_dimension(2));
            assert!(m.converged, "{k}: {m:?}");
            assert!(
                (m.x[0] / k - 2.0).abs() < 1e-5 && (m.x[1] / k + 1.0).abs() < 1e-5,
                "{k}: {m:?}"
            );
        }
    }

    #[test]
    fn the_stopping_point_is_the_same_in_any_unit_of_the_variables_and_the_values() {
        // The variables in `k` and the objective in `s`, from picounits to
        // teraunits: the minimum, in the variables' own unit, is found as well
        // and the search says it converged, whatever the units.
        for k in [1e-12, 1e-9, 1e-6, 1.0, 1e6, 1e9, 1e12] {
            for s in [1e-14, 1e-6, 1.0, 1e6, 1e14] {
                let f = move |x: &[f64]| {
                    s * (((x[0] - 2.0 * k) / k).powi(2)
                        + ((x[1] + k) / k).powi(2)
                        + 0.5 * ((x[0] - 2.0 * k) / k) * ((x[1] + k) / k))
                };
                let m = nelder_mead(f, &[k, k], NelderMeadOptions::for_dimension(2));
                assert!(m.converged, "k {k:e}, s {s:e}: {m:?}");
                assert!(
                    (m.x[0] / k - 2.0).abs() < 1e-5 && (m.x[1] / k + 1.0).abs() < 1e-5,
                    "k {k:e}, s {s:e}: {m:?}"
                );
            }
        }
    }
}
