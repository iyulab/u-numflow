//! The same numbers written in another unit give the same answer, scaled the way
//! the unit scales it: statistics, matrix decompositions and the Fourier
//! transform, from 1e-12 to 1e12.

use u_numflow::fourier::{fft, rfft, Complex};
use u_numflow::matrix::Matrix;
use u_numflow::stats::{
    covariance, kahan_sum, kurtosis, kurtosis_moment, max, mean, median, min, population_std_dev,
    population_variance, quantile, skewness, skewness_moment, std_dev, variance,
};

const SCALES: [f64; 6] = [1e-12, 1e-6, 1e-3, 1e3, 1e6, 1e12];

/// A deterministic spread of `n` values around `centre`.
fn sample(seed: u64, n: usize, centre: f64, spread: f64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            centre + spread * (u - 0.5) * 2.0 + 0.05 * (i as f64).sin()
        })
        .collect()
}

fn scaled(v: &[f64], k: f64) -> Vec<f64> {
    v.iter().map(|x| x * k).collect()
}

fn close(got: f64, want: f64, what: &str, k: f64) {
    let tolerance = 1e-9 * want.abs().max(1e-300);
    assert!(
        (got - want).abs() <= tolerance,
        "{what} at scale {k:e}: {got:e} vs {want:e}"
    );
}

#[test]
fn statistics_scale_with_the_unit() {
    let x = sample(1, 40, 5.0, 3.0);
    let y = sample(2, 40, -2.0, 2.0);
    for k in SCALES {
        let (sx, sy) = (scaled(&x, k), scaled(&y, k));
        close(mean(&sx).unwrap(), k * mean(&x).unwrap(), "mean", k);
        close(
            variance(&sx).unwrap(),
            k * k * variance(&x).unwrap(),
            "variance",
            k,
        );
        close(
            population_variance(&sx).unwrap(),
            k * k * population_variance(&x).unwrap(),
            "population_variance",
            k,
        );
        close(
            std_dev(&sx).unwrap(),
            k * std_dev(&x).unwrap(),
            "std_dev",
            k,
        );
        close(
            population_std_dev(&sx).unwrap(),
            k * population_std_dev(&x).unwrap(),
            "population_std_dev",
            k,
        );
        close(median(&sx).unwrap(), k * median(&x).unwrap(), "median", k);
        close(min(&sx).unwrap(), k * min(&x).unwrap(), "min", k);
        close(max(&sx).unwrap(), k * max(&x).unwrap(), "max", k);
        for p in [0.05, 0.25, 0.9] {
            close(
                quantile(&sx, p).unwrap(),
                k * quantile(&x, p).unwrap(),
                "quantile",
                k,
            );
        }
        // Shape does not depend on the unit.
        close(skewness(&sx).unwrap(), skewness(&x).unwrap(), "skewness", k);
        close(kurtosis(&sx).unwrap(), kurtosis(&x).unwrap(), "kurtosis", k);
        close(
            skewness_moment(&sx).unwrap(),
            skewness_moment(&x).unwrap(),
            "skewness_moment",
            k,
        );
        close(
            kurtosis_moment(&sx).unwrap(),
            kurtosis_moment(&x).unwrap(),
            "kurtosis_moment",
            k,
        );
        close(
            covariance(&sx, &sy).unwrap(),
            k * k * covariance(&x, &y).unwrap(),
            "covariance",
            k,
        );
        close(kahan_sum(&sx), k * kahan_sum(&x), "kahan_sum", k);
    }
}

/// A well-conditioned symmetric positive definite matrix.
fn spd() -> Matrix {
    Matrix::from_rows(&[
        &[4.0, 1.0, 0.5, 0.2],
        &[1.0, 3.0, 0.3, 0.1],
        &[0.5, 0.3, 2.0, 0.4],
        &[0.2, 0.1, 0.4, 5.0],
    ])
}

#[test]
fn matrix_results_scale_with_the_unit() {
    let a = spd();
    let det = a.determinant().unwrap();
    let inv = a.inverse().unwrap();
    let chol = a.cholesky().unwrap();
    let (values, _) = a.eigen_symmetric().unwrap();
    let b = [1.0, 2.0, 3.0, 4.0];
    let solved = a.cholesky_solve(&b).unwrap();
    for k in SCALES {
        let m = a.scale(k);
        close(m.determinant().unwrap(), k.powi(4) * det, "determinant", k);
        let mi = m.inverse().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                close(mi.get(i, j) * k, inv.get(i, j), "inverse", k);
            }
        }
        let mc = m.cholesky().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let want = chol.get(i, j) * k.sqrt();
                let tol = 1e-9 * want.abs().max(1e-300);
                assert!((mc.get(i, j) - want).abs() <= tol, "cholesky at {k:e}");
            }
        }
        let (mv, _) = m.eigen_symmetric().unwrap();
        for (got, want) in mv.iter().zip(&values) {
            close(*got, k * want, "eigenvalue", k);
        }
        let ms = m.cholesky_solve(&b).unwrap();
        for (got, want) in ms.iter().zip(&solved) {
            close(*got * k, *want, "cholesky_solve", k);
        }
    }
}

#[test]
fn the_fourier_transform_is_linear_in_the_unit() {
    let x = sample(3, 48, 0.0, 4.0);
    let base = rfft(&x);
    let complex: Vec<Complex> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let full = fft(&complex);
    // A component that is zero in exact arithmetic comes out as rounding noise
    // of the size of the spectrum, so the tolerance is the spectrum's own scale.
    let peak = base.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
    let near = |got: f64, want: f64, what: &str, k: f64| {
        let tolerance = 1e-12 * peak * k;
        assert!(
            (got - want).abs() <= tolerance,
            "{what} at scale {k:e}: {got:e} vs {want:e}"
        );
    };
    for k in SCALES {
        let sx = scaled(&x, k);
        let got = rfft(&sx);
        for (g, w) in got.iter().zip(&base) {
            near(g.re, k * w.re, "rfft re", k);
            near(g.im, k * w.im, "rfft im", k);
        }
        let sc: Vec<Complex> = sx.iter().map(|&v| Complex::new(v, 0.0)).collect();
        for (g, w) in fft(&sc).iter().zip(&full) {
            near(g.re, k * w.re, "fft re", k);
            near(g.im, k * w.im, "fft im", k);
        }
    }
}

mod distributions {
    use super::{close, SCALES};
    use u_numflow::distributions::{
        Exponential, GammaDistribution, LogNormal, Normal, Pert, Triangular, Uniform, Weibull,
    };

    /// A distribution written in a unit: its quantile function and its distribution function.
    type Written = (Box<dyn Fn(f64) -> Option<f64>>, Box<dyn Fn(f64) -> f64>);

    const PS: [f64; 5] = [0.01, 0.1, 0.5, 0.9, 0.99];

    /// A distribution of a scale family: written in another unit, its quantiles
    /// scale with the unit and its distribution function does not.
    fn family(
        what: &str,
        base: &dyn Fn(f64) -> Option<f64>,
        cdf_base: &dyn Fn(f64) -> f64,
        scaled: &dyn Fn(f64) -> Written,
    ) {
        for k in SCALES {
            let (q, cdf) = scaled(k);
            for p in PS {
                let want = base(p).expect("a quantile");
                close(
                    q(p).expect("a quantile"),
                    k * want,
                    &format!("{what} quantile({p})"),
                    k,
                );
                close(
                    cdf(k * want),
                    cdf_base(want),
                    &format!("{what} cdf at the quantile({p})"),
                    k,
                );
            }
        }
    }

    #[test]
    fn the_scale_families_follow_the_unit() {
        let u = Uniform::new(1.0, 4.0).unwrap();
        family("uniform", &|p| u.quantile(p), &|x| u.cdf(x), &|k| {
            let d = Uniform::new(k, 4.0 * k).unwrap();
            {
                let d = std::rc::Rc::new(d);
                let e = d.clone();
                (
                    Box::new(move |p| d.quantile(p)),
                    Box::new(move |x| e.cdf(x)),
                )
            }
        });
        let t = Triangular::new(1.0, 2.0, 5.0).unwrap();
        family("triangular", &|p| t.quantile(p), &|x| t.cdf(x), &|k| {
            let d = Triangular::new(k, 2.0 * k, 5.0 * k).unwrap();
            {
                let d = std::rc::Rc::new(d);
                let e = d.clone();
                (
                    Box::new(move |p| d.quantile(p)),
                    Box::new(move |x| e.cdf(x)),
                )
            }
        });
        let n = Normal::new(2.0, 3.0).unwrap();
        family("normal", &|p| n.quantile(p), &|x| n.cdf(x), &|k| {
            let d = Normal::new(2.0 * k, 3.0 * k).unwrap();
            {
                let d = std::rc::Rc::new(d);
                let e = d.clone();
                (
                    Box::new(move |p| d.quantile(p)),
                    Box::new(move |x| e.cdf(x)),
                )
            }
        });
        let pert = Pert::new(1.0, 3.0, 8.0).unwrap();
        family("pert", &|p| pert.quantile(p), &|x| pert.cdf(x), &|k| {
            let d = Pert::new(k, 3.0 * k, 8.0 * k).unwrap();
            {
                let d = std::rc::Rc::new(d);
                let e = d.clone();
                (
                    Box::new(move |p| d.quantile(p)),
                    Box::new(move |x| e.cdf(x)),
                )
            }
        });
        let w = Weibull::new(1.7, 5.0).unwrap();
        family("weibull", &|p| w.quantile(p), &|x| w.cdf(x), &|k| {
            let d = Weibull::new(1.7, 5.0 * k).unwrap();
            {
                let d = std::rc::Rc::new(d);
                let e = d.clone();
                (
                    Box::new(move |p| d.quantile(p)),
                    Box::new(move |x| e.cdf(x)),
                )
            }
        });
        let e = Exponential::new(0.5).unwrap();
        family("exponential", &|p| e.quantile(p), &|x| e.cdf(x), &|k| {
            let d = Exponential::new(0.5 / k).unwrap();
            {
                let d = std::rc::Rc::new(d);
                let e = d.clone();
                (
                    Box::new(move |p| d.quantile(p)),
                    Box::new(move |x| e.cdf(x)),
                )
            }
        });
        let g = GammaDistribution::new(2.5, 0.5).unwrap();
        family("gamma", &|p| g.quantile(p), &|x| g.cdf(x), &|k| {
            let d = GammaDistribution::new(2.5, 0.5 / k).unwrap();
            {
                let d = std::rc::Rc::new(d);
                let e = d.clone();
                (
                    Box::new(move |p| d.quantile(p)),
                    Box::new(move |x| e.cdf(x)),
                )
            }
        });
        // The log-normal's scale is a shift of mu.
        let l = LogNormal::new(1.0, 0.6).unwrap();
        family("lognormal", &|p| l.quantile(p), &|x| l.cdf(x), &|k| {
            let d = LogNormal::new(1.0 + k.ln(), 0.6).unwrap();
            {
                let d = std::rc::Rc::new(d);
                let e = d.clone();
                (
                    Box::new(move |p| d.quantile(p)),
                    Box::new(move |x| e.cdf(x)),
                )
            }
        });
    }
}
