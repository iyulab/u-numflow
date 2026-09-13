//! Discrete Fourier transform.
//!
//! Provides the forward and inverse DFT of a complex sequence of **any
//! length**, and a real-input convenience. A power-of-two length is
//! transformed by the iterative radix-2 Cooley-Tukey algorithm; any other
//! length by Bluestein's chirp-z algorithm, which expresses the DFT as a
//! convolution and evaluates that with a power-of-two FFT of size ≥ 2n − 1.
//! Both are O(n log n); the naive DFT is O(n²).
//!
//! Conventions: the forward transform is `X_k = Σ_j x_j · exp(−2πi jk/n)`
//! (no scaling) and the inverse divides by `n`, so `ifft(fft(x)) == x`.
//!
//! # References
//! - Cooley, J. W. & Tukey, J. W. (1965). "An algorithm for the machine
//!   calculation of complex Fourier series." *Mathematics of Computation*,
//!   19(90), 297–301.
//! - Bluestein, L. I. (1970). "A linear filtering approach to the computation
//!   of discrete Fourier transform." *IEEE Transactions on Audio and
//!   Electroacoustics*, 18(4), 451–455.

use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};

/// A complex number with `f64` parts.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Creates `re + im·i`.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Creates a real number (imaginary part 0).
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// `exp(i·theta)`.
    pub fn from_polar_unit(theta: f64) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    /// Complex conjugate.
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Modulus `|z|`.
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Squared modulus `|z|²`.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Argument in radians, in `(−π, π]`.
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Multiplies by a real scalar.
    pub fn scale(self, k: f64) -> Self {
        Self {
            re: self.re * k,
            im: self.im * k,
        }
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// Forward DFT of a complex sequence of any length.
///
/// Returns `X_k = Σ_j x_j · exp(−2πi jk/n)` for `k = 0..n`. An empty input
/// returns an empty output.
///
/// # Complexity
/// O(n log n) — radix-2 for a power-of-two `n`, Bluestein otherwise.
///
/// # Examples
///
/// ```
/// use u_numflow::fourier::{fft, Complex};
///
/// // A constant sequence has all its energy in the zero frequency.
/// let x = vec![Complex::real(1.0); 4];
/// let spectrum = fft(&x);
/// assert!((spectrum[0].re - 4.0).abs() < 1e-12);
/// assert!(spectrum[1..].iter().all(|z| z.norm() < 1e-12));
/// ```
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    transform(input, false)
}

/// Inverse DFT: `x_j = (1/n) Σ_k X_k · exp(+2πi jk/n)`.
///
/// `ifft(fft(x))` returns `x` up to floating-point rounding.
///
/// # Examples
///
/// ```
/// use u_numflow::fourier::{fft, ifft, Complex};
///
/// let x: Vec<Complex> = (0..5).map(|j| Complex::new(j as f64, -(j as f64))).collect();
/// let back = ifft(&fft(&x));
/// for (a, b) in x.iter().zip(&back) {
///     assert!((a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12);
/// }
/// ```
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let scale = 1.0 / n as f64;
    transform(input, true)
        .into_iter()
        .map(|z| z.scale(scale))
        .collect()
}

/// Forward DFT of a real sequence.
///
/// Returns all `n` bins; for real input the upper half mirrors the lower
/// (`X_{n−k} = conj(X_k)`), so callers that only need the spectrum can stop
/// at `n / 2`.
///
/// # Examples
///
/// ```
/// use u_numflow::fourier::rfft;
///
/// // One cycle of a cosine over 8 samples: energy at bins 1 and 7.
/// let x: Vec<f64> = (0..8).map(|j| (2.0 * std::f64::consts::PI * j as f64 / 8.0).cos()).collect();
/// let spectrum = rfft(&x);
/// assert!((spectrum[1].re - 4.0).abs() < 1e-12);
/// assert!((spectrum[7].re - 4.0).abs() < 1e-12);
/// ```
pub fn rfft(input: &[f64]) -> Vec<Complex> {
    let complex: Vec<Complex> = input.iter().map(|&x| Complex::real(x)).collect();
    fft(&complex)
}

/// Dispatches on the length: radix-2 for powers of two, Bluestein otherwise.
fn transform(input: &[Complex], inverse: bool) -> Vec<Complex> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    let mut data = input.to_vec();
    if n.is_power_of_two() {
        radix2_in_place(&mut data, inverse);
        data
    } else {
        bluestein(&data, inverse)
    }
}

/// Iterative radix-2 Cooley-Tukey; `data.len()` must be a power of two.
fn radix2_in_place(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            data.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let step = Complex::from_polar_unit(sign * 2.0 * PI / len as f64);
        for start in (0..n).step_by(len) {
            let mut w = Complex::real(1.0);
            for k in 0..len / 2 {
                let a = data[start + k];
                let b = data[start + k + len / 2] * w;
                data[start + k] = a + b;
                data[start + k + len / 2] = a - b;
                w = w * step;
            }
        }
        len <<= 1;
    }
}

/// Bluestein's chirp-z algorithm for an arbitrary length `n`.
///
/// With the chirp `w_j = exp(∓πi j²/n)`, `X_k = w_k · Σ_j (x_j w_j) · conj(w_{k−j})`
/// is a linear convolution, evaluated as a circular one of power-of-two size
/// `m ≥ 2n − 1`.
fn bluestein(input: &[Complex], inverse: bool) -> Vec<Complex> {
    let n = input.len();
    let m = (2 * n - 1).next_power_of_two();
    let sign = if inverse { 1.0 } else { -1.0 };

    // Chirp: the angle is π·j²/n; j² is reduced modulo 2n first so the
    // argument stays exact for large j.
    let chirp: Vec<Complex> = (0..n)
        .map(|j| {
            let jj = (j * j) % (2 * n);
            Complex::from_polar_unit(sign * PI * jj as f64 / n as f64)
        })
        .collect();

    let mut a = vec![Complex::default(); m];
    for j in 0..n {
        a[j] = input[j] * chirp[j];
    }
    let mut b = vec![Complex::default(); m];
    b[0] = chirp[0].conj();
    for j in 1..n {
        let c = chirp[j].conj();
        b[j] = c;
        b[m - j] = c;
    }

    radix2_in_place(&mut a, false);
    radix2_in_place(&mut b, false);
    let mut c: Vec<Complex> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
    radix2_in_place(&mut c, true);
    let scale = 1.0 / m as f64;

    (0..n).map(|k| c[k].scale(scale) * chirp[k]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn naive_dft(x: &[Complex], inverse: bool) -> Vec<Complex> {
        let n = x.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| {
                let mut acc = Complex::default();
                for (j, &xj) in x.iter().enumerate() {
                    let angle = sign * 2.0 * PI * (j * k) as f64 / n as f64;
                    acc = acc + xj * Complex::from_polar_unit(angle);
                }
                if inverse {
                    acc.scale(1.0 / n as f64)
                } else {
                    acc
                }
            })
            .collect()
    }

    fn assert_close(a: &[Complex], b: &[Complex], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (*x - *y).norm() < tol,
                "bin {i}: {x:?} vs {y:?} (|Δ| = {})",
                (*x - *y).norm()
            );
        }
    }

    #[test]
    fn empty_and_singleton() {
        assert!(fft(&[]).is_empty());
        assert!(ifft(&[]).is_empty());
        let one = [Complex::new(2.0, -3.0)];
        assert_eq!(fft(&one), one.to_vec());
        assert_eq!(ifft(&one), one.to_vec());
    }

    #[test]
    fn impulse_is_flat() {
        let mut x = vec![Complex::default(); 7];
        x[0] = Complex::real(1.0);
        for z in fft(&x) {
            assert!((z.re - 1.0).abs() < 1e-12 && z.im.abs() < 1e-12);
        }
    }

    #[test]
    fn sine_lands_in_its_bin_for_every_length_kind() {
        for n in [8usize, 12, 30, 64, 97] {
            let x: Vec<f64> = (0..n)
                .map(|j| (2.0 * PI * 3.0 * j as f64 / n as f64).sin())
                .collect();
            let spectrum = rfft(&x);
            let power: Vec<f64> = spectrum.iter().map(|z| z.norm_sqr()).collect();
            let (argmax, _) = power[..n / 2 + 1]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            assert_eq!(argmax, 3, "n = {n}");
            assert!((spectrum[3].im + n as f64 / 2.0).abs() < 1e-9, "n = {n}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn matches_the_naive_dft_for_any_length(
            data in prop::collection::vec((-100.0f64..100.0, -100.0f64..100.0), 1..=64)
        ) {
            let x: Vec<Complex> = data.iter().map(|&(re, im)| Complex::new(re, im)).collect();
            let n = x.len() as f64;
            let tol = 1e-9 * n * 100.0;
            let forward = fft(&x);
            assert_close(&forward, &naive_dft(&x, false), tol);
            assert_close(&ifft(&forward), &x, tol);
            assert_close(&ifft(&x), &naive_dft(&x, true), tol);
        }

        /// Parseval: Σ|x_j|² = (1/n) Σ|X_k|².
        #[test]
        fn parseval_holds(
            data in prop::collection::vec(-100.0f64..100.0, 1..=100)
        ) {
            let n = data.len() as f64;
            let time: f64 = data.iter().map(|x| x * x).sum();
            let freq: f64 = rfft(&data).iter().map(|z| z.norm_sqr()).sum::<f64>() / n;
            prop_assert!((time - freq).abs() < 1e-7 * (1.0 + time), "{time} vs {freq}");
        }

        /// Real input: the upper half of the spectrum is the conjugate mirror.
        #[test]
        fn real_input_has_hermitian_spectrum(
            data in prop::collection::vec(-100.0f64..100.0, 2..=100)
        ) {
            let n = data.len();
            let spectrum = rfft(&data);
            for k in 1..n {
                let mirror = spectrum[n - k].conj();
                prop_assert!((spectrum[k] - mirror).norm() < 1e-7 * (1.0 + spectrum[k].norm()));
            }
            prop_assert!(spectrum[0].im.abs() < 1e-9);
        }
    }
}
