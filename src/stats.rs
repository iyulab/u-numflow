//! Descriptive statistics with numerical stability guarantees.
//!
//! All functions in this module handle edge cases explicitly and use
//! numerically stable algorithms to avoid catastrophic cancellation.
//!
//! # Algorithms
//!
//! - **Mean**: Kahan compensated summation for O(ε) error independent of n.
//! - **Variance/StdDev**: Welford's online algorithm.
//!   Reference: Welford (1962), "Note on a Method for Calculating
//!   Corrected Sums of Squares and Products", *Technometrics* 4(3).
//! - **Quantile**: R-7 linear interpolation (default in R, Python, Excel).
//!   Reference: Hyndman & Fan (1996), "Sample Quantiles in Statistical
//!   Packages", *The American Statistician* 50(4).

/// Computes the arithmetic mean using Kahan compensated summation.
///
/// # Algorithm
/// Kahan summation accumulates a compensation term to recover lost
/// low-order bits, achieving O(ε) total error independent of `n`.
///
/// # Complexity
/// Time: O(n), Space: O(1)
///
/// # Returns
/// - `None` if `data` is empty or contains any NaN/Inf.
///
/// # Examples
/// ```
/// use u_numflow::stats::mean;
/// let v = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert!((mean(&v).unwrap() - 3.0).abs() < 1e-15);
/// ```
pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    if !data.iter().all(|x| x.is_finite()) {
        return None;
    }
    Some(kahan_sum(data) / data.len() as f64)
}

/// Computes the sample variance using Welford's online algorithm.
///
/// Returns the **sample** (unbiased) variance with Bessel's correction
/// (denominator `n − 1`).
///
/// # Algorithm
/// Welford's method maintains a running mean and sum of squared deviations,
/// avoiding catastrophic cancellation inherent in the naive formula
/// `Var = E[X²] − (E[X])²`.
///
/// Reference: Welford (1962), *Technometrics* 4(3), pp. 419–420.
///
/// # Complexity
/// Time: O(n), Space: O(1)
///
/// # Returns
/// - `None` if `data.len() < 2` or contains NaN/Inf.
///
/// # Examples
/// ```
/// use u_numflow::stats::variance;
/// let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// assert!((variance(&v).unwrap() - 4.571428571428571).abs() < 1e-10);
/// ```
pub fn variance(data: &[f64]) -> Option<f64> {
    if data.len() < 2 {
        return None;
    }
    if !data.iter().all(|x| x.is_finite()) {
        return None;
    }
    let mut acc = WelfordAccumulator::new();
    for &x in data {
        acc.update(x);
    }
    acc.sample_variance()
}

/// Computes the population variance using Welford's online algorithm.
///
/// Returns the **population** variance (denominator `n`).
///
/// # Returns
/// - `None` if `data` is empty or contains NaN/Inf.
///
/// # Examples
/// ```
/// use u_numflow::stats::population_variance;
/// let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// assert!((population_variance(&v).unwrap() - 4.0).abs() < 1e-10);
/// ```
pub fn population_variance(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    if !data.iter().all(|x| x.is_finite()) {
        return None;
    }
    let mut acc = WelfordAccumulator::new();
    for &x in data {
        acc.update(x);
    }
    acc.population_variance()
}

/// Computes the sample standard deviation.
///
/// Equivalent to `sqrt(variance(data))`.
///
/// # Returns
/// - `None` if `data.len() < 2` or contains NaN/Inf.
///
/// # Examples
/// ```
/// use u_numflow::stats::std_dev;
/// let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let sd = std_dev(&v).unwrap();
/// assert!((sd - 2.138089935299395).abs() < 1e-10);
/// ```
pub fn std_dev(data: &[f64]) -> Option<f64> {
    variance(data).map(f64::sqrt)
}

/// Computes the population standard deviation.
///
/// Equivalent to `sqrt(population_variance(data))`.
///
/// # Returns
/// - `None` if `data` is empty or contains NaN/Inf.
pub fn population_std_dev(data: &[f64]) -> Option<f64> {
    population_variance(data).map(f64::sqrt)
}

/// Returns the minimum value in the slice.
///
/// # Returns
/// - `None` if `data` is empty or contains NaN.
///
/// # Examples
/// ```
/// use u_numflow::stats::min;
/// assert_eq!(min(&[3.0, 1.0, 4.0, 1.0, 5.0]), Some(1.0));
/// ```
pub fn min(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    data.iter().copied().try_fold(f64::INFINITY, |acc, x| {
        if x.is_nan() {
            None
        } else {
            Some(acc.min(x))
        }
    })
}

/// Returns the maximum value in the slice.
///
/// # Returns
/// - `None` if `data` is empty or contains NaN.
///
/// # Examples
/// ```
/// use u_numflow::stats::max;
/// assert_eq!(max(&[3.0, 1.0, 4.0, 1.0, 5.0]), Some(5.0));
/// ```
pub fn max(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    data.iter().copied().try_fold(f64::NEG_INFINITY, |acc, x| {
        if x.is_nan() {
            None
        } else {
            Some(acc.max(x))
        }
    })
}

/// Computes the median of `data` without mutating the input.
///
/// Internally clones and sorts the data, then returns the middle element
/// (or the average of the two middle elements for even-length data).
///
/// # Complexity
/// Time: O(n log n), Space: O(n)
///
/// # Returns
/// - `None` if `data` is empty or contains NaN.
///
/// # Examples
/// ```
/// use u_numflow::stats::median;
/// assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
/// assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
/// ```
pub fn median(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    if data.iter().any(|x| x.is_nan()) {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).expect("NaN filtered above"));
    let n = sorted.len();
    if n % 2 == 1 {
        Some(sorted[n / 2])
    } else {
        Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    }
}

/// Computes the `p`-th quantile using the R-7 linear interpolation method.
///
/// This is the default quantile method in R, NumPy, and Excel.
///
/// # Algorithm
/// For sorted data `x[0..n]` and quantile `p ∈ [0, 1]`:
/// 1. Compute `h = (n − 1) × p`
/// 2. Let `j = ⌊h⌋` and `g = h − j`
/// 3. Return `(1 − g) × x[j] + g × x[j+1]`
///
/// Reference: Hyndman & Fan (1996), *The American Statistician* 50(4), pp. 361–365.
///
/// # Complexity
/// Time: O(n log n) (dominated by sort), Space: O(n)
///
/// # Panics
/// Does not panic; returns `None` for invalid inputs.
///
/// # Returns
/// - `None` if `data` is empty, `p` is outside `[0, 1]`, or data contains NaN.
///
/// # Examples
/// ```
/// use u_numflow::stats::quantile;
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(quantile(&data, 0.0), Some(1.0));
/// assert_eq!(quantile(&data, 1.0), Some(5.0));
/// assert_eq!(quantile(&data, 0.5), Some(3.0));
/// ```
pub fn quantile(data: &[f64], p: f64) -> Option<f64> {
    if data.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    if data.iter().any(|x| x.is_nan()) {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).expect("NaN filtered above"));
    quantile_sorted(&sorted, p)
}

/// Computes the `p`-th quantile on **pre-sorted** data (R-7 method).
///
/// This avoids the O(n log n) sort when calling multiple quantiles on
/// the same dataset. The caller must guarantee that `sorted_data` is
/// sorted in non-decreasing order.
///
/// # Returns
/// - `None` if `sorted_data` is empty or `p` is outside `[0, 1]`.
pub fn quantile_sorted(sorted_data: &[f64], p: f64) -> Option<f64> {
    let n = sorted_data.len();
    if n == 0 || !(0.0..=1.0).contains(&p) {
        return None;
    }
    if n == 1 {
        return Some(sorted_data[0]);
    }

    let h = (n - 1) as f64 * p;
    let j = h.floor() as usize;
    let g = h - h.floor();

    if j + 1 >= n {
        Some(sorted_data[n - 1])
    } else {
        Some((1.0 - g) * sorted_data[j] + g * sorted_data[j + 1])
    }
}

/// Computes Fisher's adjusted sample skewness (G₁) with bias correction.
///
/// # Formula
/// ```text
/// G₁ = [√(n(n−1)) / (n−2)] × (m₃ / m₂^{3/2})
/// ```
/// where `m₂`, `m₃` are the biased second and third central moments.
///
/// This matches Excel `SKEW()` and `scipy.stats.skew(bias=False)`.
///
/// # Algorithm
/// Two-pass: first computes the mean (Kahan sum), then accumulates
/// central moments in a single sweep.
///
/// Reference: Joanes & Gill (1998), "Comparing measures of sample skewness
/// and kurtosis", *The Statistician* 47(1), pp. 183–189.
///
/// # Complexity
/// Time: O(n), Space: O(1)
///
/// # Returns
/// - `None` if `data.len() < 3`, data contains NaN/Inf, or variance is zero.
///
/// # Examples
/// ```
/// use u_numflow::stats::skewness;
/// // Symmetric data → skewness = 0
/// let sym = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert!(skewness(&sym).unwrap().abs() < 1e-14);
///
/// // Right-skewed data → positive skewness
/// let right = [1.0, 2.0, 3.0, 4.0, 50.0];
/// assert!(skewness(&right).unwrap() > 0.0);
/// ```
pub fn skewness(data: &[f64]) -> Option<f64> {
    let n = data.len();
    if n < 3 {
        return None;
    }
    if !data.iter().all(|x| x.is_finite()) {
        return None;
    }
    let nf = n as f64;
    let m = kahan_sum(data) / nf;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    for &x in data {
        let d = x - m;
        let d2 = d * d;
        sum2 += d2;
        sum3 += d2 * d;
    }
    let m2 = sum2 / nf;
    if m2 == 0.0 {
        return None;
    }
    let m3 = sum3 / nf;
    let g1 = m3 / m2.powf(1.5);
    let correction = (nf * (nf - 1.0)).sqrt() / (nf - 2.0);
    Some(correction * g1)
}

/// Computes Fisher's excess kurtosis (G₂) with bias correction.
///
/// # Formula
/// ```text
/// G₂ = [n(n+1) / ((n−1)(n−2)(n−3))] × Σ[(xᵢ−x̄)/s]⁴ − [3(n−1)² / ((n−2)(n−3))]
/// ```
/// where `s` is the sample standard deviation (n−1 denominator).
///
/// This matches Excel `KURT()` and `scipy.stats.kurtosis(bias=False)`.
/// Returns **0** for a normal distribution, positive for heavy tails
/// (leptokurtic), negative for light tails (platykurtic).
///
/// # Algorithm
/// Two-pass: first computes the mean (Kahan sum), then accumulates
/// central moments in a single sweep.
///
/// Reference: Joanes & Gill (1998), "Comparing measures of sample skewness
/// and kurtosis", *The Statistician* 47(1), pp. 183–189.
///
/// # Complexity
/// Time: O(n), Space: O(1)
///
/// # Returns
/// - `None` if `data.len() < 4`, data contains NaN/Inf, or variance is zero.
///
/// # Examples
/// ```
/// use u_numflow::stats::kurtosis;
/// // Uniform-ish data → negative excess kurtosis
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let k = kurtosis(&data).unwrap();
/// assert!(k < 0.0); // platykurtic
/// ```
pub fn kurtosis(data: &[f64]) -> Option<f64> {
    let n = data.len();
    if n < 4 {
        return None;
    }
    if !data.iter().all(|x| x.is_finite()) {
        return None;
    }
    let nf = n as f64;
    let m = kahan_sum(data) / nf;
    let mut sum2 = 0.0;
    let mut sum4 = 0.0;
    for &x in data {
        let d = x - m;
        let d2 = d * d;
        sum2 += d2;
        sum4 += d2 * d2;
    }
    // Sample variance s² = sum2 / (n-1)
    let s2 = sum2 / (nf - 1.0);
    if s2 == 0.0 {
        return None;
    }
    let s4 = s2 * s2;
    // Σ[(xᵢ − x̄)/s]⁴
    let sum_z4 = sum4 / s4;
    let a = nf * (nf + 1.0) / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0));
    let b = 3.0 * (nf - 1.0) * (nf - 1.0) / ((nf - 2.0) * (nf - 3.0));
    Some(a * sum_z4 - b)
}

/// Computes the sample covariance between two datasets.
///
/// # Formula
/// ```text
/// Cov(X, Y) = Σ(xᵢ − x̄)(yᵢ − ȳ) / (n − 1)
/// ```
///
/// Uses Bessel's correction (n−1 denominator) for an unbiased estimator.
///
/// # Complexity
/// Time: O(n), Space: O(1)
///
/// # Returns
/// - `None` if `x.len() != y.len()`, `n < 2`, or data contains NaN/Inf.
///
/// # Examples
/// ```
/// use u_numflow::stats::covariance;
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [2.0, 4.0, 6.0, 8.0, 10.0];
/// let cov = covariance(&x, &y).unwrap();
/// assert!((cov - 5.0).abs() < 1e-14); // perfect positive covariance
/// ```
pub fn covariance(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return None;
    }
    if !x.iter().chain(y.iter()).all(|v| v.is_finite()) {
        return None;
    }
    let nf = n as f64;
    let mean_x = kahan_sum(x) / nf;
    let mean_y = kahan_sum(y) / nf;
    let mut sum = 0.0;
    for i in 0..n {
        sum += (x[i] - mean_x) * (y[i] - mean_y);
    }
    Some(sum / (nf - 1.0))
}

// ---------------------------------------------------------------------------
// Kahan compensated summation
// ---------------------------------------------------------------------------

/// Neumaier compensated summation for O(ε) error independent of `n`.
///
/// This is an improved variant of Kahan summation that also handles the
/// case where the addend is larger in magnitude than the running sum.
///
/// # Algorithm
/// Maintains a running compensation variable `c`. At each step, the
/// branch ensures the smaller operand's low-order bits are captured.
///
/// Reference: Neumaier (1974), "Rundungsfehleranalyse einiger Verfahren
/// zur Summation endlicher Summen", *Zeitschrift für Angewandte
/// Mathematik und Mechanik* 54(1), pp. 39–51.
///
/// # Complexity
/// Time: O(n), Space: O(1)
pub fn kahan_sum(data: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &x in data {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    sum + c
}

// ---------------------------------------------------------------------------
// Welford online accumulator
// ---------------------------------------------------------------------------

/// Streaming accumulator for mean, variance, skewness, and kurtosis.
///
/// Computes running descriptive statistics in a single pass with O(1)
/// memory and guaranteed numerical stability, using the extended
/// Welford algorithm for higher-order moments.
///
/// # Algorithm
/// Maintains central moment sums M₂, M₃, M₄ incrementally. The update
/// order (M₄ → M₃ → M₂) preserves correctness since each uses the
/// *previous* values of lower moments.
///
/// References:
/// - Welford (1962), *Technometrics* 4(3), pp. 419–420.
/// - Pébay (2008), "Formulas for Robust, One-Pass Parallel Computation
///   of Covariances and Arbitrary-Order Statistical Moments",
///   Sandia Report SAND2008-6212.
/// - Terriberry (2007), "Computing Higher-Order Moments Online".
///
/// # Examples
/// ```
/// use u_numflow::stats::WelfordAccumulator;
/// let mut acc = WelfordAccumulator::new();
/// for &x in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
///     acc.update(x);
/// }
/// assert!((acc.mean().unwrap() - 5.0).abs() < 1e-15);
/// assert!((acc.sample_variance().unwrap() - 4.571428571428571).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct WelfordAccumulator {
    count: u64,
    mean_acc: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl WelfordAccumulator {
    /// Creates a new empty accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean_acc: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
        }
    }

    /// Feeds a new sample into the accumulator.
    ///
    /// Updates M₂, M₃, M₄ in the correct order (M₄ → M₃ → M₂) so that
    /// each uses the *previous* values of lower-order moments.
    ///
    /// The first sample is handled as a special case: all moments remain
    /// zero and only the mean is initialized. This avoids intermediate
    /// overflow when `delta² > f64::MAX` (e.g., `value ≈ 1e166`).
    pub fn update(&mut self, value: f64) {
        let n1 = self.count;
        self.count += 1;

        if n1 == 0 {
            // First sample: mean = value, all moments stay zero.
            self.mean_acc = value;
            return;
        }

        let n = self.count as f64;
        let delta = value - self.mean_acc;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1 as f64;

        // M₄ before M₃ before M₂ — order matters!
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
        self.mean_acc += delta_n;
    }

    /// Returns the number of samples seen so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the running mean, or `None` if no samples have been added.
    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.mean_acc)
        }
    }

    /// Returns the sample variance (n − 1 denominator), or `None` if fewer
    /// than 2 samples have been added.
    pub fn sample_variance(&self) -> Option<f64> {
        if self.count < 2 {
            None
        } else {
            Some(self.m2 / (self.count - 1) as f64)
        }
    }

    /// Returns the population variance (n denominator), or `None` if no
    /// samples have been added.
    pub fn population_variance(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.m2 / self.count as f64)
        }
    }

    /// Returns the sample standard deviation, or `None` if fewer than 2
    /// samples have been added.
    pub fn sample_std_dev(&self) -> Option<f64> {
        self.sample_variance().map(f64::sqrt)
    }

    /// Returns the population standard deviation, or `None` if no samples
    /// have been added.
    pub fn population_std_dev(&self) -> Option<f64> {
        self.population_variance().map(f64::sqrt)
    }

    /// Returns Fisher's adjusted sample skewness (G₁), or `None` if
    /// fewer than 3 samples have been added or variance is zero.
    ///
    /// Uses the same bias correction as Excel `SKEW()`.
    pub fn skewness(&self) -> Option<f64> {
        if self.count < 3 {
            return None;
        }
        let n = self.count as f64;
        if self.m2 == 0.0 {
            return None;
        }
        // Biased skewness: g₁ = √n × M₃ / M₂^(3/2)
        let g1 = n.sqrt() * self.m3 / self.m2.powf(1.5);
        // Bias correction: √(n(n−1)) / (n−2)
        let correction = (n * (n - 1.0)).sqrt() / (n - 2.0);
        Some(correction * g1)
    }

    /// Returns Fisher's excess kurtosis (G₂) with bias correction, or
    /// `None` if fewer than 4 samples have been added or variance is zero.
    ///
    /// Uses the same bias correction as Excel `KURT()`.
    /// Returns 0 for a normal distribution, positive for heavy tails.
    pub fn kurtosis(&self) -> Option<f64> {
        if self.count < 4 {
            return None;
        }
        let n = self.count as f64;
        if self.m2 == 0.0 {
            return None;
        }
        // Biased excess kurtosis: g₂ = n × M₄ / M₂² − 3
        let g2 = n * self.m4 / (self.m2 * self.m2) - 3.0;
        // Unbiased (Fisher G₂): [(n−1)/((n−2)(n−3))] × [(n+1)×g₂ + 6]
        let correction = (n - 1.0) / ((n - 2.0) * (n - 3.0));
        Some(correction * ((n + 1.0) * g2 + 6.0))
    }

    /// Merges another accumulator into this one (parallel-friendly).
    ///
    /// Uses Chan's parallel algorithm extended to higher-order moments.
    ///
    /// References:
    /// - Chan, Golub & LeVeque (1979), "Updating Formulae and a
    ///   Pairwise Algorithm for Computing Sample Variances".
    /// - Pébay (2008), SAND2008-6212 (M₃, M₄ merge formulas).
    pub fn merge(&mut self, other: &WelfordAccumulator) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        let na = self.count as f64;
        let nb = other.count as f64;
        let total = self.count + other.count;
        let n = total as f64;
        let delta = other.mean_acc - self.mean_acc;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;

        let new_mean = self.mean_acc + delta * (nb / n);

        let new_m2 = self.m2 + other.m2 + delta2 * na * nb / n;

        let new_m3 = self.m3
            + other.m3
            + delta3 * na * nb * (na - nb) / (n * n)
            + 3.0 * delta * (na * other.m2 - nb * self.m2) / n;

        let new_m4 = self.m4
            + other.m4
            + delta4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
            + 6.0 * delta2 * (na * na * other.m2 + nb * nb * self.m2) / (n * n)
            + 4.0 * delta * (na * other.m3 - nb * self.m3) / n;

        self.count = total;
        self.mean_acc = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.m4 = new_m4;
    }
}

impl Default for WelfordAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- mean ---

    #[test]
    fn test_mean_basic() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), Some(3.0));
    }

    #[test]
    fn test_mean_single() {
        assert_eq!(mean(&[42.0]), Some(42.0));
    }

    #[test]
    fn test_mean_empty() {
        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn test_mean_nan() {
        assert_eq!(mean(&[1.0, f64::NAN, 3.0]), None);
    }

    #[test]
    fn test_mean_inf() {
        assert_eq!(mean(&[1.0, f64::INFINITY, 3.0]), None);
    }

    // --- variance ---

    #[test]
    fn test_variance_basic() {
        let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = variance(&v).unwrap();
        assert!((var - 4.571428571428571).abs() < 1e-10);
    }

    #[test]
    fn test_variance_constant() {
        let v = [5.0; 100];
        assert!((variance(&v).unwrap()).abs() < 1e-15);
    }

    #[test]
    fn test_variance_single() {
        assert_eq!(variance(&[1.0]), None);
    }

    #[test]
    fn test_variance_empty() {
        assert_eq!(variance(&[]), None);
    }

    #[test]
    fn test_population_variance() {
        let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = population_variance(&v).unwrap();
        assert!((var - 4.0).abs() < 1e-10);
    }

    // --- std_dev ---

    #[test]
    fn test_std_dev() {
        let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&v).unwrap();
        let expected = 4.571428571428571_f64.sqrt();
        assert!((sd - expected).abs() < 1e-10);
    }

    // --- min / max ---

    #[test]
    fn test_min_max() {
        let v = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        assert_eq!(min(&v), Some(1.0));
        assert_eq!(max(&v), Some(9.0));
    }

    #[test]
    fn test_min_max_empty() {
        assert_eq!(min(&[]), None);
        assert_eq!(max(&[]), None);
    }

    #[test]
    fn test_min_max_nan() {
        assert_eq!(min(&[1.0, f64::NAN]), None);
        assert_eq!(max(&[1.0, f64::NAN]), None);
    }

    // --- median ---

    #[test]
    fn test_median_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    }

    #[test]
    fn test_median_even() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
    }

    #[test]
    fn test_median_single() {
        assert_eq!(median(&[7.0]), Some(7.0));
    }

    #[test]
    fn test_median_empty() {
        assert_eq!(median(&[]), None);
    }

    // --- quantile ---

    #[test]
    fn test_quantile_extremes() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&data, 0.0), Some(1.0));
        assert_eq!(quantile(&data, 1.0), Some(5.0));
    }

    #[test]
    fn test_quantile_median() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&data, 0.5), Some(3.0));
    }

    #[test]
    fn test_quantile_interpolation() {
        let data = [1.0, 2.0, 3.0, 4.0];
        // h = (4-1)*0.25 = 0.75, j=0, g=0.75
        // (1-0.75)*1.0 + 0.75*2.0 = 0.25 + 1.5 = 1.75
        let q = quantile(&data, 0.25).unwrap();
        assert!((q - 1.75).abs() < 1e-15);
    }

    #[test]
    fn test_quantile_invalid_p() {
        assert_eq!(quantile(&[1.0, 2.0], -0.1), None);
        assert_eq!(quantile(&[1.0, 2.0], 1.1), None);
    }

    #[test]
    fn test_quantile_empty() {
        assert_eq!(quantile(&[], 0.5), None);
    }

    #[test]
    fn test_quantile_single() {
        assert_eq!(quantile(&[42.0], 0.0), Some(42.0));
        assert_eq!(quantile(&[42.0], 0.5), Some(42.0));
        assert_eq!(quantile(&[42.0], 1.0), Some(42.0));
    }

    // --- kahan_sum ---

    #[test]
    fn test_kahan_sum_basic() {
        let v = [1.0, 2.0, 3.0];
        assert!((kahan_sum(&v) - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_kahan_sum_precision() {
        // Sum of 1e16 + 1.0 + (-1e16) with naive sum loses the 1.0
        let v = [1e16, 1.0, -1e16];
        let result = kahan_sum(&v);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "Kahan sum should preserve the 1.0: got {result}"
        );
    }

    // --- WelfordAccumulator ---

    #[test]
    fn test_welford_empty() {
        let acc = WelfordAccumulator::new();
        assert_eq!(acc.count(), 0);
        assert_eq!(acc.mean(), None);
        assert_eq!(acc.sample_variance(), None);
    }

    #[test]
    fn test_welford_single() {
        let mut acc = WelfordAccumulator::new();
        acc.update(5.0);
        assert_eq!(acc.mean(), Some(5.0));
        assert_eq!(acc.sample_variance(), None);
        assert_eq!(acc.population_variance(), Some(0.0));
    }

    #[test]
    fn test_welford_matches_batch() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut acc = WelfordAccumulator::new();
        for &x in &data {
            acc.update(x);
        }
        let batch_mean = mean(&data).unwrap();
        let batch_var = variance(&data).unwrap();
        assert!((acc.mean().unwrap() - batch_mean).abs() < 1e-14);
        assert!((acc.sample_variance().unwrap() - batch_var).abs() < 1e-10);
    }

    #[test]
    fn test_welford_merge() {
        let data_a = [1.0, 2.0, 3.0, 4.0];
        let data_b = [5.0, 6.0, 7.0, 8.0];
        let data_all: Vec<f64> = data_a.iter().chain(data_b.iter()).copied().collect();

        let mut acc_a = WelfordAccumulator::new();
        for &x in &data_a {
            acc_a.update(x);
        }
        let mut acc_b = WelfordAccumulator::new();
        for &x in &data_b {
            acc_b.update(x);
        }
        acc_a.merge(&acc_b);

        let expected_mean = mean(&data_all).unwrap();
        let expected_var = variance(&data_all).unwrap();

        assert!((acc_a.mean().unwrap() - expected_mean).abs() < 1e-14);
        assert!((acc_a.sample_variance().unwrap() - expected_var).abs() < 1e-10);
    }

    // --- skewness ---

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric data → skewness = 0
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(skewness(&data).unwrap().abs() < 1e-14);
    }

    #[test]
    fn test_skewness_right_skewed() {
        // Right-skewed: one large outlier
        let data = [1.0, 2.0, 3.0, 4.0, 50.0];
        let s = skewness(&data).unwrap();
        assert!(s > 0.0, "expected positive skewness, got {s}");
    }

    #[test]
    fn test_skewness_left_skewed() {
        // Left-skewed: one small outlier
        let data = [-50.0, 1.0, 2.0, 3.0, 4.0];
        let s = skewness(&data).unwrap();
        assert!(s < 0.0, "expected negative skewness, got {s}");
    }

    #[test]
    fn test_skewness_edge_cases() {
        assert_eq!(skewness(&[]), None);
        assert_eq!(skewness(&[1.0]), None);
        assert_eq!(skewness(&[1.0, 2.0]), None);
        // Constant data → zero variance → None
        assert_eq!(skewness(&[5.0, 5.0, 5.0]), None);
        // NaN → None
        assert_eq!(skewness(&[1.0, f64::NAN, 3.0]), None);
    }

    #[test]
    fn test_skewness_known_value() {
        // Data: [1, 2, 3, 4, 8]
        // Manual computation:
        //   n=5, mean=3.6
        //   d = [-2.6, -1.6, -0.6, 0.4, 4.4]
        //   m2 = (6.76+2.56+0.36+0.16+19.36)/5 = 5.84
        //   m3 = (-17.576-4.096-0.216+0.064+85.184)/5 = 12.672
        //   g1 = 12.672 / 5.84^1.5 ≈ 0.8982
        //   correction = sqrt(20)/3 ≈ 1.4907
        //   G1 ≈ 1.3388
        let data = [1.0, 2.0, 3.0, 4.0, 8.0];
        let s = skewness(&data).unwrap();
        assert!(
            (s - 1.339).abs() < 0.01,
            "expected skewness ≈ 1.34, got {s}"
        );
    }

    // --- kurtosis ---

    #[test]
    fn test_kurtosis_uniform_negative() {
        // Uniform-ish data should have negative excess kurtosis (platykurtic)
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let k = kurtosis(&data).unwrap();
        assert!(k < 0.0, "uniform data should be platykurtic, got {k}");
    }

    #[test]
    fn test_kurtosis_edge_cases() {
        assert_eq!(kurtosis(&[]), None);
        assert_eq!(kurtosis(&[1.0]), None);
        assert_eq!(kurtosis(&[1.0, 2.0]), None);
        assert_eq!(kurtosis(&[1.0, 2.0, 3.0]), None);
        // Constant → None
        assert_eq!(kurtosis(&[5.0, 5.0, 5.0, 5.0]), None);
        // NaN → None
        assert_eq!(kurtosis(&[1.0, f64::NAN, 3.0, 4.0]), None);
    }

    #[test]
    fn test_kurtosis_heavy_tails() {
        // Heavy-tailed data (leptokurtic) → positive excess kurtosis
        let data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
        let k = kurtosis(&data).unwrap();
        assert!(k > 0.0, "heavy-tailed data should be leptokurtic, got {k}");
    }

    // --- covariance ---

    #[test]
    fn test_covariance_perfect_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let cov = covariance(&x, &y).unwrap();
        assert!((cov - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_covariance_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 8.0, 6.0, 4.0, 2.0];
        let cov = covariance(&x, &y).unwrap();
        assert!((cov - (-5.0)).abs() < 1e-14);
    }

    #[test]
    fn test_covariance_zero() {
        // Independent: x and -x values cancel
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [5.0, 5.0, 5.0, 5.0]; // constant
        let cov = covariance(&x, &y).unwrap();
        assert!(cov.abs() < 1e-14);
    }

    #[test]
    fn test_covariance_self_is_variance() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let cov_xx = covariance(&data, &data).unwrap();
        let var = variance(&data).unwrap();
        assert!((cov_xx - var).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_edge_cases() {
        assert_eq!(covariance(&[], &[]), None);
        assert_eq!(covariance(&[1.0], &[2.0]), None);
        assert_eq!(covariance(&[1.0, 2.0], &[1.0]), None); // different lengths
        assert_eq!(covariance(&[1.0, f64::NAN], &[1.0, 2.0]), None);
    }

    // --- WelfordAccumulator: skewness & kurtosis ---

    #[test]
    fn test_welford_skewness_matches_batch() {
        let data = [1.0, 2.0, 3.0, 4.0, 50.0];
        let mut acc = WelfordAccumulator::new();
        for &x in &data {
            acc.update(x);
        }
        let batch = skewness(&data).unwrap();
        let stream = acc.skewness().unwrap();
        assert!(
            (batch - stream).abs() < 1e-10,
            "batch={batch}, stream={stream}"
        );
    }

    #[test]
    fn test_welford_kurtosis_matches_batch() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];
        let mut acc = WelfordAccumulator::new();
        for &x in &data {
            acc.update(x);
        }
        let batch = kurtosis(&data).unwrap();
        let stream = acc.kurtosis().unwrap();
        assert!(
            (batch - stream).abs() < 1e-8,
            "batch={batch}, stream={stream}"
        );
    }

    #[test]
    fn test_welford_skewness_kurtosis_edge_cases() {
        let mut acc = WelfordAccumulator::new();
        assert_eq!(acc.skewness(), None);
        assert_eq!(acc.kurtosis(), None);
        acc.update(1.0);
        assert_eq!(acc.skewness(), None);
        assert_eq!(acc.kurtosis(), None);
        acc.update(2.0);
        assert_eq!(acc.skewness(), None);
        assert_eq!(acc.kurtosis(), None);
        acc.update(3.0);
        // skewness available at n=3
        assert!(acc.skewness().is_some());
        assert_eq!(acc.kurtosis(), None);
        acc.update(4.0);
        // kurtosis available at n=4
        assert!(acc.kurtosis().is_some());
    }

    #[test]
    fn test_welford_merge_skewness_kurtosis() {
        let data_a = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
        let data_b = [2.0, 50.0, 4.0, 6.0, 8.0, 100.0];
        let all: Vec<f64> = data_a.iter().chain(data_b.iter()).copied().collect();

        let mut acc_a = WelfordAccumulator::new();
        for &x in &data_a {
            acc_a.update(x);
        }
        let mut acc_b = WelfordAccumulator::new();
        for &x in &data_b {
            acc_b.update(x);
        }
        acc_a.merge(&acc_b);

        let expected_skew = skewness(&all).unwrap();
        let expected_kurt = kurtosis(&all).unwrap();
        let merged_skew = acc_a.skewness().unwrap();
        let merged_kurt = acc_a.kurtosis().unwrap();

        assert!(
            (expected_skew - merged_skew).abs() < 1e-8,
            "skewness: expected={expected_skew}, merged={merged_skew}"
        );
        assert!(
            (expected_kurt - merged_kurt).abs() < 1e-6,
            "kurtosis: expected={expected_kurt}, merged={merged_kurt}"
        );
    }

    // --- numerical stability ---

    #[test]
    fn test_variance_large_offset() {
        // Data with large mean: [1e9 + 1, 1e9 + 2, ..., 1e9 + 5]
        // Naive algorithm would suffer catastrophic cancellation.
        let data: Vec<f64> = (1..=5).map(|i| 1e9 + i as f64).collect();
        let var = variance(&data).unwrap();
        // True variance of [1,2,3,4,5] = 2.5
        assert!(
            (var - 2.5).abs() < 1e-5,
            "Variance of offset data should be ~2.5, got {var}"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating finite f64 vectors of reasonable size.
    fn finite_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
        proptest::collection::vec(
            prop::num::f64::NORMAL.prop_filter("finite", |x| x.is_finite() && x.abs() < 1e12),
            min_len..=max_len,
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        // --- Variance is non-negative ---
        #[test]
        fn variance_non_negative(data in finite_vec(2, 100)) {
            let var = variance(&data).unwrap();
            prop_assert!(var >= 0.0, "variance must be >= 0, got {}", var);
        }

        // --- Variance of constant is zero ---
        #[test]
        fn variance_of_constant_is_zero(
            value in prop::num::f64::NORMAL.prop_filter("finite", |x| x.is_finite()),
            n in 2_usize..50,
        ) {
            let data = vec![value; n];
            let var = variance(&data).unwrap();
            prop_assert!(var.abs() < 1e-10, "variance of constant should be ~0, got {}", var);
        }

        // --- std_dev = sqrt(variance) ---
        #[test]
        fn std_dev_is_sqrt_of_variance(data in finite_vec(2, 100)) {
            let var = variance(&data).unwrap();
            let sd = std_dev(&data).unwrap();
            let diff = (sd * sd - var).abs();
            prop_assert!(diff < 1e-10 * var.max(1.0), "sd² should equal variance");
        }

        // --- Mean linearity: mean(a*x + b) = a*mean(x) + b ---
        #[test]
        fn mean_linearity(
            data in finite_vec(1, 100),
            a in -100.0_f64..100.0,
            b in -100.0_f64..100.0,
        ) {
            prop_assume!(a.is_finite() && b.is_finite());
            let m = mean(&data).unwrap();
            let transformed: Vec<f64> = data.iter().map(|&x| a * x + b).collect();
            if let Some(mt) = mean(&transformed) {
                let expected = a * m + b;
                let tol = 1e-8 * expected.abs().max(1.0);
                prop_assert!(
                    (mt - expected).abs() < tol,
                    "mean(a*x+b)={} != a*mean(x)+b={}",
                    mt, expected
                );
            }
        }

        // --- quantile(0) = min, quantile(1) = max ---
        #[test]
        fn quantile_extremes_are_min_max(data in finite_vec(1, 100)) {
            let q0 = quantile(&data, 0.0).unwrap();
            let q1 = quantile(&data, 1.0).unwrap();
            let mn = min(&data).unwrap();
            let mx = max(&data).unwrap();
            prop_assert!((q0 - mn).abs() < 1e-15, "quantile(0) should be min");
            prop_assert!((q1 - mx).abs() < 1e-15, "quantile(1) should be max");
        }

        // --- Quantiles are monotonic ---
        #[test]
        fn quantiles_monotonic(
            data in finite_vec(2, 100),
            p1 in 0.0_f64..=1.0,
            p2 in 0.0_f64..=1.0,
        ) {
            let (lo, hi) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };
            let q_lo = quantile(&data, lo).unwrap();
            let q_hi = quantile(&data, hi).unwrap();
            prop_assert!(q_lo <= q_hi + 1e-15, "quantiles should be monotonic");
        }

        // --- median = quantile(0.5) ---
        #[test]
        fn median_equals_quantile_half(data in finite_vec(1, 100)) {
            let med = median(&data).unwrap();
            let q50 = quantile(&data, 0.5).unwrap();
            prop_assert!(
                (med - q50).abs() < 1e-14,
                "median={} != quantile(0.5)={}",
                med, q50
            );
        }

        // --- Welford merge produces same result as sequential ---
        #[test]
        fn welford_merge_equals_sequential(
            data_a in finite_vec(1, 50),
            data_b in finite_vec(1, 50),
        ) {
            let mut sequential = WelfordAccumulator::new();
            for &x in data_a.iter().chain(data_b.iter()) {
                sequential.update(x);
            }

            let mut acc_a = WelfordAccumulator::new();
            for &x in &data_a { acc_a.update(x); }
            let mut acc_b = WelfordAccumulator::new();
            for &x in &data_b { acc_b.update(x); }
            acc_a.merge(&acc_b);

            let seq_mean = sequential.mean().unwrap();
            let mrg_mean = acc_a.mean().unwrap();
            prop_assert!(
                (seq_mean - mrg_mean).abs() < 1e-10 * seq_mean.abs().max(1.0),
                "merged mean should match sequential"
            );

            if sequential.count() >= 2 {
                let seq_var = sequential.sample_variance().unwrap();
                let mrg_var = acc_a.sample_variance().unwrap();
                prop_assert!(
                    (seq_var - mrg_var).abs() < 1e-8 * seq_var.max(1.0),
                    "merged variance should match sequential"
                );
            }
        }

        // --- Skewness of symmetric data is near zero ---
        #[test]
        fn skewness_of_symmetric_is_zero(
            half in proptest::collection::vec(-1e6_f64..1e6, 2..=50),
        ) {
            prop_assume!(half.iter().all(|x| x.is_finite()));
            // Build truly symmetric data around 0: [a, b, c, -c, -b, -a]
            let mut data: Vec<f64> = half.clone();
            data.extend(half.iter().map(|x| -x));
            if let Some(s) = skewness(&data) {
                prop_assert!(
                    s.abs() < 1e-8,
                    "symmetric data should have ~0 skewness, got {}",
                    s
                );
            }
        }

        // --- Streaming skewness matches batch ---
        // Uses bounded range [-1e6, 1e6] to avoid ill-conditioned data
        // with 100+ order-of-magnitude dynamic range.
        #[test]
        fn welford_skewness_matches_batch(
            data in proptest::collection::vec(-1e6_f64..1e6, 3..=100)
        ) {
            let mut acc = WelfordAccumulator::new();
            for &x in &data { acc.update(x); }
            match (skewness(&data), acc.skewness()) {
                (Some(batch), Some(stream)) if batch.is_finite() && stream.is_finite() => {
                    let tol = 1e-8 * batch.abs().max(1.0);
                    prop_assert!(
                        (batch - stream).abs() < tol,
                        "batch={} stream={}", batch, stream
                    );
                }
                _ => {} // NaN/None cases: skip
            }
        }

        // --- Streaming kurtosis matches batch ---
        #[test]
        fn welford_kurtosis_matches_batch(
            data in proptest::collection::vec(-1e6_f64..1e6, 4..=100)
        ) {
            let mut acc = WelfordAccumulator::new();
            for &x in &data { acc.update(x); }
            match (kurtosis(&data), acc.kurtosis()) {
                (Some(batch), Some(stream)) if batch.is_finite() && stream.is_finite() => {
                    let tol = 1e-6 * batch.abs().max(1.0);
                    prop_assert!(
                        (batch - stream).abs() < tol,
                        "batch={} stream={}", batch, stream
                    );
                }
                _ => {} // NaN/None cases: skip
            }
        }

        // --- Covariance of x with itself equals variance ---
        #[test]
        fn covariance_self_is_variance(data in finite_vec(2, 100)) {
            let cov = covariance(&data, &data).unwrap();
            let var = variance(&data).unwrap();
            let tol = 1e-10 * var.max(1.0);
            prop_assert!(
                (cov - var).abs() < tol,
                "Cov(x,x)={} != Var(x)={}", cov, var
            );
        }

        // --- Covariance is symmetric: Cov(x,y) = Cov(y,x) ---
        #[test]
        fn covariance_symmetric(
            x in finite_vec(2, 50),
            y in finite_vec(2, 50),
        ) {
            let n = x.len().min(y.len());
            if n >= 2 {
                let x_slice = &x[..n];
                let y_slice = &y[..n];
                let cov_xy = covariance(x_slice, y_slice).unwrap();
                let cov_yx = covariance(y_slice, x_slice).unwrap();
                prop_assert!(
                    (cov_xy - cov_yx).abs() < 1e-10 * cov_xy.abs().max(1.0),
                    "Cov(x,y)={} != Cov(y,x)={}", cov_xy, cov_yx
                );
            }
        }
    }
}
