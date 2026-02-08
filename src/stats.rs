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
/// use u_optim::stats::mean;
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
/// use u_optim::stats::variance;
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
/// use u_optim::stats::population_variance;
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
/// use u_optim::stats::std_dev;
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
/// use u_optim::stats::min;
/// assert_eq!(min(&[3.0, 1.0, 4.0, 1.0, 5.0]), Some(1.0));
/// ```
pub fn min(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    data.iter()
        .copied()
        .try_fold(f64::INFINITY, |acc, x| {
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
/// use u_optim::stats::max;
/// assert_eq!(max(&[3.0, 1.0, 4.0, 1.0, 5.0]), Some(5.0));
/// ```
pub fn max(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    data.iter()
        .copied()
        .try_fold(f64::NEG_INFINITY, |acc, x| {
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
/// use u_optim::stats::median;
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
/// use u_optim::stats::quantile;
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

/// Streaming accumulator for mean and variance using Welford's algorithm.
///
/// Computes running mean, variance, and standard deviation in a single
/// pass with O(1) memory and guaranteed numerical stability.
///
/// # Algorithm
/// For each new sample `x_k`:
/// ```text
/// δ₁ = x_k − μ_{k−1}
/// μ_k = μ_{k−1} + δ₁ / k
/// δ₂ = x_k − μ_k
/// M₂_k = M₂_{k−1} + δ₁ × δ₂
/// ```
///
/// Reference: Welford (1962), *Technometrics* 4(3), pp. 419–420.
///
/// # Examples
/// ```
/// use u_optim::stats::WelfordAccumulator;
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
}

impl WelfordAccumulator {
    /// Creates a new empty accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean_acc: 0.0,
            m2: 0.0,
        }
    }

    /// Feeds a new sample into the accumulator.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean_acc;
        self.mean_acc += delta / self.count as f64;
        let delta2 = value - self.mean_acc;
        self.m2 += delta * delta2;
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

    /// Merges another accumulator into this one (parallel-friendly).
    ///
    /// Uses Chan's parallel algorithm for combining partial aggregates.
    ///
    /// Reference: Chan, Golub & LeVeque (1979), "Updating Formulae and a
    /// Pairwise Algorithm for Computing Sample Variances".
    pub fn merge(&mut self, other: &WelfordAccumulator) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        let total = self.count + other.count;
        let delta = other.mean_acc - self.mean_acc;
        let new_mean =
            self.mean_acc + delta * (other.count as f64 / total as f64);
        let new_m2 = self.m2
            + other.m2
            + delta * delta * (self.count as f64 * other.count as f64 / total as f64);
        self.count = total;
        self.mean_acc = new_mean;
        self.m2 = new_m2;
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
    }
}
