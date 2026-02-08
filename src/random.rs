//! Random number generation, shuffling, and weighted sampling.
//!
//! Provides seeded RNG construction, Fisher-Yates shuffle, and
//! weighted random sampling utilities.
//!
//! # Reproducibility
//!
//! For reproducible experiments, use [`create_rng`] with a fixed seed.
//! The underlying algorithm (SmallRng) is deterministic for a given seed
//! on the same platform.

use rand::Rng;

/// Creates a fast, seeded random number generator.
///
/// Uses `SmallRng` (Xoshiro256++) for high performance.
/// The sequence is deterministic for a given seed on the same platform.
///
/// # Examples
/// ```
/// use u_optim::random::create_rng;
/// use rand::Rng;
/// let mut rng = create_rng(42);
/// let x: f64 = rng.random();
/// assert!(x >= 0.0 && x < 1.0);
/// ```
pub fn create_rng(seed: u64) -> rand::rngs::SmallRng {
    use rand::SeedableRng;
    rand::rngs::SmallRng::seed_from_u64(seed)
}

/// Fisher-Yates (Durstenfeld) in-place shuffle.
///
/// Produces a uniformly random permutation: each of the n! permutations
/// is equally likely.
///
/// # Algorithm
/// Modern variant due to Durstenfeld (1964), popularized by Knuth as
/// "Algorithm P". Iterates backwards, swapping each element with a
/// uniformly chosen earlier (or same) position.
///
/// Reference: Knuth (1997), *TAOCP* Vol. 2, §3.4.2, Algorithm P.
///
/// # Complexity
/// Time: O(n), Space: O(1) (in-place)
///
/// # Examples
/// ```
/// use u_optim::random::{create_rng, shuffle};
/// let mut v = vec![1, 2, 3, 4, 5];
/// let mut rng = create_rng(42);
/// shuffle(&mut v, &mut rng);
/// // v is now a permutation of [1, 2, 3, 4, 5]
/// v.sort();
/// assert_eq!(v, vec![1, 2, 3, 4, 5]);
/// ```
pub fn shuffle<T, R: Rng>(slice: &mut [T], rng: &mut R) {
    let n = slice.len();
    if n <= 1 {
        return;
    }
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

/// Returns a shuffled index permutation of `[0, n)`.
///
/// Generates a random permutation of indices without modifying the
/// original data. Useful when you need to iterate over data in random
/// order without cloning.
///
/// # Complexity
/// Time: O(n), Space: O(n)
///
/// # Examples
/// ```
/// use u_optim::random::{create_rng, shuffled_indices};
/// let mut rng = create_rng(42);
/// let indices = shuffled_indices(5, &mut rng);
/// assert_eq!(indices.len(), 5);
/// let mut sorted = indices.clone();
/// sorted.sort();
/// assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
/// ```
pub fn shuffled_indices<R: Rng>(n: usize, rng: &mut R) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, rng);
    indices
}

/// Selects a random index weighted by the given weights.
///
/// Uses the CDF binary search method. For repeated sampling from the
/// same weights, prefer [`WeightedSampler`].
///
/// # Complexity
/// Time: O(n) per sample
///
/// # Returns
/// - `None` if `weights` is empty or all weights are zero.
///
/// # Examples
/// ```
/// use u_optim::random::{create_rng, weighted_choose};
/// let mut rng = create_rng(42);
/// let weights = [1.0, 2.0, 3.0]; // index 2 is most likely
/// let idx = weighted_choose(&weights, &mut rng).unwrap();
/// assert!(idx < 3);
/// ```
pub fn weighted_choose<R: Rng>(weights: &[f64], rng: &mut R) -> Option<usize> {
    if weights.is_empty() {
        return None;
    }

    let total: f64 = weights.iter().filter(|w| **w > 0.0).sum();
    if total <= 0.0 {
        return None;
    }

    let threshold = rng.random_range(0.0..total);
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        if w > 0.0 {
            cumulative += w;
            if cumulative > threshold {
                return Some(i);
            }
        }
    }

    // Fallback (floating-point edge case)
    Some(weights.len() - 1)
}

/// Pre-computed weighted sampler for O(log n) repeated sampling.
///
/// Builds a cumulative distribution table from weights, then uses
/// binary search for each sample.
///
/// # Algorithm
/// CDF-based weighted sampling with binary search.
///
/// # Complexity
/// - Construction: O(n)
/// - Sampling: O(log n)
///
/// # Examples
/// ```
/// use u_optim::random::{create_rng, WeightedSampler};
/// let weights = vec![1.0, 2.0, 3.0, 4.0];
/// let sampler = WeightedSampler::new(&weights).unwrap();
/// let mut rng = create_rng(42);
/// let idx = sampler.sample(&mut rng);
/// assert!(idx < 4);
/// ```
pub struct WeightedSampler {
    cumulative: Vec<f64>,
    total: f64,
}

impl WeightedSampler {
    /// Creates a new weighted sampler from the given weights.
    ///
    /// # Returns
    /// - `None` if `weights` is empty or all weights are zero/negative.
    pub fn new(weights: &[f64]) -> Option<Self> {
        if weights.is_empty() {
            return None;
        }

        let mut cumulative = Vec::with_capacity(weights.len());
        let mut total = 0.0;
        for &w in weights {
            if w > 0.0 {
                total += w;
            }
            cumulative.push(total);
        }

        if total <= 0.0 {
            return None;
        }

        Some(Self { cumulative, total })
    }

    /// Samples a random index according to the weights.
    ///
    /// # Complexity
    /// O(log n) via binary search.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> usize {
        let threshold = rng.random_range(0.0..self.total);
        match self
            .cumulative
            .binary_search_by(|c| c.partial_cmp(&threshold).expect("cumulative values are finite"))
        {
            Ok(i) => i,
            Err(i) => i.min(self.cumulative.len() - 1),
        }
    }

    /// Returns the number of categories.
    pub fn len(&self) -> usize {
        self.cumulative.len()
    }

    /// Returns true if there are no categories.
    pub fn is_empty(&self) -> bool {
        self.cumulative.is_empty()
    }

    /// Returns the total weight.
    pub fn total_weight(&self) -> f64 {
        self.total
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_rng_deterministic() {
        let mut rng1 = create_rng(42);
        let mut rng2 = create_rng(42);
        let vals1: Vec<f64> = (0..10).map(|_| rng1.random()).collect();
        let vals2: Vec<f64> = (0..10).map(|_| rng2.random()).collect();
        assert_eq!(vals1, vals2);
    }

    #[test]
    fn test_shuffle_preserves_elements() {
        let mut v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut rng = create_rng(123);
        shuffle(&mut v, &mut rng);
        v.sort();
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_shuffle_empty() {
        let mut v: Vec<i32> = vec![];
        let mut rng = create_rng(0);
        shuffle(&mut v, &mut rng); // should not panic
    }

    #[test]
    fn test_shuffle_single() {
        let mut v = vec![42];
        let mut rng = create_rng(0);
        shuffle(&mut v, &mut rng);
        assert_eq!(v, vec![42]);
    }

    #[test]
    fn test_shuffle_actually_shuffles() {
        // With 10 elements, probability of identity permutation is 1/10! ≈ 2.8e-7
        let original = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut v = original.clone();
        let mut rng = create_rng(42);
        shuffle(&mut v, &mut rng);
        assert_ne!(v, original, "shuffle should change order (probabilistic)");
    }

    #[test]
    fn test_shuffled_indices() {
        let mut rng = create_rng(42);
        let indices = shuffled_indices(10, &mut rng);
        assert_eq!(indices.len(), 10);
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_weighted_choose_basic() {
        let mut rng = create_rng(42);
        let weights = [0.0, 0.0, 1.0]; // only index 2 has weight
        for _ in 0..100 {
            assert_eq!(weighted_choose(&weights, &mut rng), Some(2));
        }
    }

    #[test]
    fn test_weighted_choose_empty() {
        let mut rng = create_rng(42);
        assert_eq!(weighted_choose(&[], &mut rng), None);
    }

    #[test]
    fn test_weighted_choose_all_zero() {
        let mut rng = create_rng(42);
        assert_eq!(weighted_choose(&[0.0, 0.0], &mut rng), None);
    }

    #[test]
    fn test_weighted_choose_distribution() {
        let mut rng = create_rng(42);
        let weights = [1.0, 3.0]; // index 1 should be ~3x more likely
        let mut counts = [0u32; 2];
        let n = 10000;
        for _ in 0..n {
            let idx = weighted_choose(&weights, &mut rng).unwrap();
            counts[idx] += 1;
        }
        let ratio = counts[1] as f64 / counts[0] as f64;
        assert!(
            (ratio - 3.0).abs() < 0.5,
            "expected ratio ~3.0, got {ratio}"
        );
    }

    #[test]
    fn test_weighted_sampler_basic() {
        let sampler = WeightedSampler::new(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(sampler.len(), 3);
        assert!(!sampler.is_empty());
        assert!((sampler.total_weight() - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_weighted_sampler_deterministic_weight() {
        let sampler = WeightedSampler::new(&[0.0, 0.0, 1.0]).unwrap();
        let mut rng = create_rng(42);
        for _ in 0..100 {
            assert_eq!(sampler.sample(&mut rng), 2);
        }
    }

    #[test]
    fn test_weighted_sampler_distribution() {
        let sampler = WeightedSampler::new(&[1.0, 3.0]).unwrap();
        let mut rng = create_rng(42);
        let mut counts = [0u32; 2];
        let n = 10000;
        for _ in 0..n {
            counts[sampler.sample(&mut rng)] += 1;
        }
        let ratio = counts[1] as f64 / counts[0] as f64;
        assert!(
            (ratio - 3.0).abs() < 0.5,
            "expected ratio ~3.0, got {ratio}"
        );
    }

    #[test]
    fn test_weighted_sampler_empty() {
        assert!(WeightedSampler::new(&[]).is_none());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn shuffle_is_permutation(
            seed in 0_u64..10000,
            data in proptest::collection::vec(0_i32..1000, 0..50),
        ) {
            let mut shuffled = data.clone();
            let mut rng = create_rng(seed);
            shuffle(&mut shuffled, &mut rng);
            let mut sorted_orig = data.clone();
            let mut sorted_shuf = shuffled;
            sorted_orig.sort();
            sorted_shuf.sort();
            prop_assert_eq!(sorted_orig, sorted_shuf);
        }

        #[test]
        fn weighted_choose_returns_valid_index(
            seed in 0_u64..10000,
            weights in proptest::collection::vec(0.0_f64..10.0, 1..20),
        ) {
            let has_positive = weights.iter().any(|&w| w > 0.0);
            let mut rng = create_rng(seed);
            let result = weighted_choose(&weights, &mut rng);
            if has_positive {
                let idx = result.unwrap();
                prop_assert!(idx < weights.len());
            }
        }
    }
}
