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

use std::f64::consts::FRAC_PI_4;
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

/// Forward DFT of a real sequence: the `n / 2 + 1` non-redundant bins
/// `0..=n/2` (NumPy `numpy.fft.rfft`, SciPy `scipy.fft.rfft`, FFTW r2c).
///
/// For real input the rest mirrors them (`X_{n−k} = conj(X_k)`), so nothing
/// is lost; [`fft`] of the same values, taken as complex, returns all `n`.
/// An empty input gives an empty spectrum.
///
/// # Examples
///
/// ```
/// use u_numflow::fourier::rfft;
///
/// // One cycle of a cosine over 8 samples: all its energy is in bin 1.
/// let x: Vec<f64> = (0..8).map(|j| (2.0 * std::f64::consts::PI * j as f64 / 8.0).cos()).collect();
/// let spectrum = rfft(&x);
/// assert_eq!(spectrum.len(), 5);
/// assert!((spectrum[1].re - 4.0).abs() < 1e-12);
/// ```
pub fn rfft(input: &[f64]) -> Vec<Complex> {
    let n = input.len();
    if n < 4 || n % 2 != 0 {
        let complex: Vec<Complex> = input.iter().map(|&x| Complex::real(x)).collect();
        let mut spectrum = fft(&complex);
        spectrum.truncate(n / 2 + 1);
        return spectrum;
    }
    // An even length: pack neighbours into one complex sequence of half the
    // length, `z_j = x_{2j} + i·x_{2j+1}`, transform that, and split the
    // even- and odd-indexed parts back out -- about half the work of
    // transforming the real sequence as complex.
    let h = n / 2;
    let packed: Vec<Complex> = (0..h)
        .map(|j| Complex::new(input[2 * j], input[2 * j + 1]))
        .collect();
    let z = fft(&packed);
    let mut spectrum = Vec::with_capacity(h + 1);
    for k in 0..=h {
        let zk = z[k % h];
        let zc = z[(h - k) % h].conj();
        // E_k = (Z_k + conj(Z_{h-k})) / 2, O_k = (Z_k - conj(Z_{h-k})) / (2i)
        let even = Complex::new(0.5 * (zk.re + zc.re), 0.5 * (zk.im + zc.im));
        let diff = Complex::new(zk.re - zc.re, zk.im - zc.im);
        let odd = Complex::new(0.5 * diff.im, -0.5 * diff.re);
        let twiddle = unit_root(k, n).conj();
        spectrum.push(even + twiddle * odd);
    }
    spectrum
}

/// Dispatches on the length: radix-2 for powers of two, Bluestein otherwise.
fn transform(input: &[Complex], inverse: bool) -> Vec<Complex> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    let mut data = input.to_vec();
    if n.is_power_of_two() {
        radix2_in_place(&mut data, &stage_roots(n), inverse);
        data
    } else {
        bluestein(&data, inverse)
    }
}

/// `exp(2πi·k/n)` for `0 ≤ k < n`, to an ulp of each part.
///
/// The angle is reduced to an octant by integer arithmetic (`8k` against `n`),
/// so what reaches `sin` and `cos` is at most π/4 and carries the rounding of
/// one quotient — where a running product of steps `w ← w·step`, or a
/// `2π·k/n` that reaches 2π, grows it with `k`.
fn unit_root(k: usize, n: usize) -> Complex {
    let (k8, n64) = (8 * k as u64, n as u64);
    let octant = k8 / n64;
    let r = k8 % n64; // the angle inside the octant is π/4 · r/n
    let (c, s) = if octant % 2 == 0 {
        let phi = FRAC_PI_4 * r as f64 / n as f64;
        (phi.cos(), phi.sin())
    } else {
        // θ = π/2 − ψ with ψ = π/4 · (n − r)/n
        let psi = FRAC_PI_4 * (n64 - r) as f64 / n as f64;
        (psi.sin(), psi.cos())
    };
    match octant / 2 {
        0 => Complex::new(c, s),
        1 => Complex::new(-s, c),
        2 => Complex::new(-c, -s),
        _ => Complex::new(s, -c),
    }
}

/// The twiddles of a radix-2 transform of size `n` (a power of two ≥ 2), one
/// block per stage: entry `half + k` is `exp(2πi·k/(2·half))`, `k < half`, so
/// a stage of length `len = 2·half` reads its block in order and the whole
/// table is `n` entries. Only the first octant of the last block is computed
/// with `sin`/`cos`; the rest of it are reflections and quarter turns of that
/// octant, exactly, and each earlier block is every other entry of the next.
fn stage_roots(n: usize) -> Vec<Complex> {
    let half = n / 2;
    let mut table = vec![Complex::default(); n];
    if n < 8 {
        for k in 0..half {
            table[half + k] = unit_root(k, n);
        }
    } else {
        let (quarter, eighth) = (n / 4, n / 8);
        for k in 0..=eighth {
            table[half + k] = unit_root(k, n);
        }
        for k in eighth + 1..quarter {
            let m = table[half + quarter - k];
            table[half + k] = Complex::new(m.im, m.re); // cos(π/2 − x) = sin x
        }
        for k in quarter..half {
            let m = table[half + k - quarter];
            table[half + k] = Complex::new(-m.im, m.re); // a quarter turn: ·i
        }
    }
    let mut block = half / 2;
    while block >= 1 {
        for k in 0..block {
            table[block + k] = table[2 * block + 2 * k];
        }
        block /= 2;
    }
    table
}

/// Iterative radix-2 Cooley-Tukey; `data.len()` must be a power of two and
/// `roots` its [`stage_roots`].
///
/// Every twiddle is read from the table, not accumulated as `w ← w·step`
/// along the stage: that product's rounding grows with the number of steps,
/// a relative error of 10⁻¹⁴ at 2¹⁰ points and 4·10⁻¹¹ at 2²² against the
/// 10⁻¹⁶ the butterflies themselves give.
fn radix2_in_place(data: &mut [Complex], roots: &[Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(n.is_power_of_two() && roots.len() == n);

    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            data.swap(i, j);
        }
    }

    // The table holds exp(+2πi·k/len); the forward transform uses its conjugate.
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let stage = &roots[half..len];
        for block in data.chunks_exact_mut(len) {
            let (lo, hi) = block.split_at_mut(half);
            for ((a, b), root) in lo.iter_mut().zip(hi.iter_mut()).zip(stage) {
                let w = Complex::new(root.re, sign * root.im);
                let t = *b * w;
                let u = *a;
                *a = u + t;
                *b = u - t;
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

    // Chirp: the angle is π·j²/n = 2π·(j² mod 2n)/(2n); j² is reduced modulo
    // 2n first so the argument stays exact for large j.
    let chirp: Vec<Complex> = (0..n)
        .map(|j| {
            let jj = ((j as u128 * j as u128) % (2 * n) as u128) as usize;
            let root = unit_root(jj, 2 * n);
            Complex::new(root.re, sign * root.im)
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

    let roots = stage_roots(m);
    radix2_in_place(&mut a, &roots, false);
    radix2_in_place(&mut b, &roots, false);
    let mut c: Vec<Complex> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
    radix2_in_place(&mut c, &roots, true);
    let scale = 1.0 / m as f64;

    (0..n).map(|k| c[k].scale(scale) * chirp[k]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::f64::consts::PI;

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

    /// The octant reduction names the right point for every `k`, on lengths of
    /// every kind (odd, 3·2ᵏ, a prime, a power of two).
    #[test]
    fn unit_roots_are_on_the_circle_at_the_right_angle() {
        for n in [1_usize, 2, 3, 5, 8, 12, 100, 1024, 4097] {
            for k in 0..n {
                let root = unit_root(k, n);
                let (sin, cos) = (2.0 * PI * k as f64 / n as f64).sin_cos();
                assert!(
                    (root.re - cos).abs() < 1e-12 && (root.im - sin).abs() < 1e-12,
                    "k {k} n {n}"
                );
                assert!((root.norm_sqr() - 1.0).abs() < 4e-16, "k {k} n {n}");
            }
        }
        // The axes and the half-turn are exact.
        assert_eq!(unit_root(0, 8), Complex::new(1.0, 0.0));
        assert_eq!(unit_root(2, 8).re.abs(), 0.0);
        assert_eq!(unit_root(2, 8).im, 1.0);
        assert_eq!(unit_root(4, 8), Complex::new(-1.0, 0.0));
    }

    /// A complex exponential `exp(2πi·m·j/n)` transforms to `n` at bin `m` and
    /// nothing elsewhere; whatever else shows is the transform's own error. It
    /// stays at a few 10⁻¹⁶ of the signal per bin at 2¹⁶ points and at a
    /// prime length through Bluestein (a running product of twiddles gave 10⁻¹³
    /// there, and grew with the length).
    #[test]
    fn a_pure_tone_stays_in_its_bin_at_large_lengths() {
        for (n, m) in [(1_usize << 16, 12345_usize), (50_021, 777)] {
            let tone: Vec<Complex> = (0..n)
                .map(|j| {
                    let r = ((m as u128 * j as u128) % n as u128) as f64;
                    let (sin, cos) = (2.0 * PI * r / n as f64).sin_cos();
                    Complex::new(cos, sin)
                })
                .collect();
            let spectrum = fft(&tone);
            let sumsq: f64 = spectrum
                .iter()
                .enumerate()
                .map(|(k, z)| {
                    let (re, im) = if k == m {
                        (z.re - n as f64, z.im)
                    } else {
                        (z.re, z.im)
                    };
                    re * re + im * im
                })
                .sum();
            let per_bin = (sumsq / n as f64).sqrt() / (n as f64).sqrt();
            assert!(per_bin < 2e-15, "n {n}: {per_bin:e}");
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
            // The full spectrum, rebuilt from the half rfft returns: every bin
            // between DC and Nyquist stands for itself and its mirror.
            let half = rfft(&data);
            let len = data.len();
            let freq: f64 = half
                .iter()
                .enumerate()
                .map(|(k, z)| {
                    let twice = k != 0 && 2 * k != len;
                    z.norm_sqr() * if twice { 2.0 } else { 1.0 }
                })
                .sum::<f64>()
                / n;
            prop_assert!((time - freq).abs() < 1e-7 * (1.0 + time), "{time} vs {freq}");
        }

        /// Real input: `rfft` is the first `n / 2 + 1` bins of `fft`, and the
        /// bins it leaves out are their conjugate mirror.
        #[test]
        fn real_input_has_hermitian_spectrum(
            data in prop::collection::vec(-100.0f64..100.0, 2..=100)
        ) {
            let n = data.len();
            let half = rfft(&data);
            prop_assert_eq!(half.len(), n / 2 + 1);
            let full = fft(&data.iter().map(|&x| Complex::real(x)).collect::<Vec<_>>());
            for k in 0..half.len() {
                prop_assert!((half[k] - full[k]).norm() < 1e-9 * (1.0 + full[k].norm()));
            }
            for k in 1..n {
                let mirror = full[n - k].conj();
                prop_assert!((full[k] - mirror).norm() < 1e-7 * (1.0 + full[k].norm()));
            }
            prop_assert!(half[0].im.abs() < 1e-9);
        }
    }
}

/// Against NumPy 2.4.6 (`numpy.fft.rfft`).
#[cfg(test)]
mod numpy_reference {
    use super::*;

    #[test]
    fn rfft_of_an_odd_length_reproduces_numpy() {
        // numpy.fft.rfft([0.5, 1.5, -0.25, 2.0, 0.0, -1.0, 3.5, 1.25, -0.75]) -- length 9,
        // not a power of two. Both return the bins 0..=n/2.
        let got = rfft(&[0.5, 1.5, -0.25, 2.0, 0.0, -1.0, 3.5, 1.25, -0.75]);
        let want = [
            (6.75, 0.0),
            (-0.5621258692079278, 0.9879574700745879),
            (-3.8255009306546883, -2.3590377252790837),
            (5.625, -4.113620667976083),
            (-2.362373200137384, 0.5501191216763025),
        ];
        assert_eq!(got.len(), 5);
        for (k, (re, im)) in want.into_iter().enumerate() {
            let g = got[k];
            assert!(
                (g.re - re).abs() <= 1e-12 && (g.im - im).abs() <= 1e-12,
                "bin {k}: got {} + {}i, NumPy {re} + {im}i",
                g.re,
                g.im
            );
        }
    }
}
