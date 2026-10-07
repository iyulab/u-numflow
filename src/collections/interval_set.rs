//! Sets of half-open intervals on an ordered line.
//!
//! An [`IntervalSet`] is a finite union of half-open intervals `[start, end)`,
//! stored in **normal form**: sorted by start, pairwise disjoint, and with no
//! two pieces touching (`[1, 3) ∪ [3, 5)` is stored as `[1, 5)`). Every
//! operation takes normal-form inputs and returns a normal-form result, so the
//! measure of a set is always the plain sum of its pieces — overlapping input
//! can never be counted twice.
//!
//! # Algorithm
//!
//! Construction sorts the pieces and merges in one sweep. Union, intersection
//! and difference of two normal-form sets are linear two-pointer merges.
//!
//! # Complexity
//!
//! | Operation | Cost |
//! |---|---|
//! | [`IntervalSet::from_intervals`] | O(n log n) |
//! | [`union`](IntervalSet::union) · [`intersection`](IntervalSet::intersection) · [`difference`](IntervalSet::difference) | O(n + m) |
//! | [`clip`](IntervalSet::clip) · [`measure`](IntervalSet::measure) | O(n) |
//! | [`contains`](IntervalSet::contains) | O(log n) |
//!
//! # References
//!
//! - de Berg, Cheong, van Kreveld & Overmars (2008), *Computational Geometry*,
//!   3rd ed., §10.1 (interval sets and their sweep-based operations).

use std::cmp::Ordering;
use std::fmt;
use std::ops::Add;

/// A coordinate type an [`IntervalSet`] can be built over.
///
/// Implemented for `f32`, `f64` and the primitive integers. The associated
/// [`Length`](IntervalBound::Length) is the type of a distance between two
/// coordinates: the coordinate type itself for floats, and the unsigned type
/// of the same width for integers — the same choice as the standard library's
/// `i64::abs_diff`, so that the length of `[i64::MIN, i64::MAX)` is
/// representable. Because the pieces of a set are disjoint, the total measure
/// of any set is at most the distance between the type's extremes and never
/// overflows `Length`.
pub trait IntervalBound: Copy + PartialOrd + fmt::Debug {
    /// The type of a distance between two coordinates.
    type Length: Copy + PartialOrd + Default + Add<Output = Self::Length> + fmt::Debug;

    /// Whether the value can bound an interval (false for NaN and ±∞).
    fn is_admissible(self) -> bool;

    /// The distance from `start` to `end`. Callers guarantee `start <= end`.
    fn distance(start: Self, end: Self) -> Self::Length;
}

macro_rules! float_bound {
    ($($t:ty),*) => {$(
        impl IntervalBound for $t {
            type Length = $t;
            fn is_admissible(self) -> bool {
                self.is_finite()
            }
            fn distance(start: Self, end: Self) -> Self::Length {
                end - start
            }
        }
    )*};
}

macro_rules! int_bound {
    ($($t:ty => $len:ty),*) => {$(
        impl IntervalBound for $t {
            type Length = $len;
            fn is_admissible(self) -> bool {
                true
            }
            fn distance(start: Self, end: Self) -> Self::Length {
                end.abs_diff(start)
            }
        }
    )*};
}

float_bound!(f32, f64);
int_bound!(
    i8 => u8, i16 => u16, i32 => u32, i64 => u64, i128 => u128, isize => usize,
    u8 => u8, u16 => u16, u32 => u32, u64 => u64, u128 => u128, usize => usize
);

/// Why an interval was refused.
///
/// `index` is the interval's position in the input of
/// [`IntervalSet::from_intervals`]; operations that take a single interval
/// ([`IntervalSet::insert`], [`IntervalSet::clip`]) report `0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalError {
    /// `start > end`. A reversed interval is refused rather than swapped: the
    /// caller's data says something it did not mean, and guessing which bound
    /// is wrong would hide that.
    Reversed {
        /// Position of the interval in the input.
        index: usize,
    },
    /// A bound is NaN or infinite.
    NotFinite {
        /// Position of the interval in the input.
        index: usize,
    },
}

impl fmt::Display for IntervalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntervalError::Reversed { index } => {
                write!(f, "interval {index} has start > end")
            }
            IntervalError::NotFinite { index } => {
                write!(f, "interval {index} has a NaN or infinite bound")
            }
        }
    }
}

impl std::error::Error for IntervalError {}

fn check<T: IntervalBound>(index: usize, start: T, end: T) -> Result<(), IntervalError> {
    if !start.is_admissible() || !end.is_admissible() {
        return Err(IntervalError::NotFinite { index });
    }
    if start > end {
        return Err(IntervalError::Reversed { index });
    }
    Ok(())
}

/// Total order on admissible bounds (NaN is refused before it gets here).
fn cmp<T: IntervalBound>(a: T, b: T) -> Ordering {
    a.partial_cmp(&b)
        .expect("IntervalSet bounds are admissible, so comparable")
}

fn max<T: IntervalBound>(a: T, b: T) -> T {
    if b > a {
        b
    } else {
        a
    }
}

fn min<T: IntervalBound>(a: T, b: T) -> T {
    if b < a {
        b
    } else {
        a
    }
}

/// A finite union of half-open intervals `[start, end)`, kept in normal form.
///
/// Empty intervals (`start == end`) contribute nothing and are dropped.
///
/// # Examples
///
/// Up-time of an asset in a reporting window, where planned and unplanned
/// stops overlap:
///
/// ```
/// use u_numflow::collections::IntervalSet;
///
/// let planned = IntervalSet::from_intervals([(0.0, 24.0)]).unwrap();
/// let unplanned = IntervalSet::from_intervals([(10.0, 30.0)]).unwrap();
/// let window = IntervalSet::from_intervals([(0.0, 168.0)]).unwrap();
///
/// let down = planned.union(&unplanned); // [0, 30): the 14 h overlap counts once
/// assert_eq!(down.measure(), 30.0);
/// assert_eq!(window.difference(&down).measure(), 138.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalSet<T: IntervalBound> {
    pieces: Vec<(T, T)>,
}

impl<T: IntervalBound> Default for IntervalSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: IntervalBound> IntervalSet<T> {
    /// The empty set.
    pub fn new() -> Self {
        Self { pieces: Vec::new() }
    }

    /// Builds the union of `(start, end)` pairs, each read as `[start, end)`.
    ///
    /// Overlapping and touching pieces are merged; empty ones are dropped.
    ///
    /// # Errors
    /// [`IntervalError::Reversed`] for a pair with `start > end`,
    /// [`IntervalError::NotFinite`] for a NaN or infinite bound — with the
    /// pair's position in `intervals`.
    pub fn from_intervals<I>(intervals: I) -> Result<Self, IntervalError>
    where
        I: IntoIterator<Item = (T, T)>,
    {
        let mut pieces = Vec::new();
        for (index, (start, end)) in intervals.into_iter().enumerate() {
            check(index, start, end)?;
            if start < end {
                pieces.push((start, end));
            }
        }
        pieces.sort_by(|a, b| cmp(a.0, b.0));
        Ok(Self {
            pieces: merge_sorted(pieces),
        })
    }

    /// Adds `[start, end)` to the set.
    ///
    /// # Errors
    /// As [`from_intervals`](Self::from_intervals), with `index` 0.
    pub fn insert(&mut self, start: T, end: T) -> Result<(), IntervalError> {
        check(0, start, end)?;
        if start < end {
            *self = self.union(&Self {
                pieces: vec![(start, end)],
            });
        }
        Ok(())
    }

    /// The normal-form pieces, in increasing order.
    pub fn iter(&self) -> impl Iterator<Item = (T, T)> + '_ {
        self.pieces.iter().copied()
    }

    /// The normal-form pieces as a slice.
    pub fn as_slice(&self) -> &[(T, T)] {
        &self.pieces
    }

    /// Number of disjoint pieces.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether the set contains no point.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Total length of the set — each point counted once.
    pub fn measure(&self) -> T::Length {
        self.pieces
            .iter()
            .fold(T::Length::default(), |acc, &(s, e)| acc + T::distance(s, e))
    }

    /// Whether `t` lies in the set (`start <= t < end` for some piece).
    pub fn contains(&self, t: T) -> bool {
        if !t.is_admissible() {
            return false;
        }
        // First piece whose end is beyond t; t is inside iff that piece starts at or before t.
        let i = self.pieces.partition_point(|&(_, e)| e <= t);
        self.pieces.get(i).is_some_and(|&(s, _)| s <= t)
    }

    /// Points in either set.
    pub fn union(&self, other: &Self) -> Self {
        let mut all = Vec::with_capacity(self.pieces.len() + other.pieces.len());
        let (mut i, mut j) = (0, 0);
        while i < self.pieces.len() || j < other.pieces.len() {
            let take_self = match (self.pieces.get(i), other.pieces.get(j)) {
                (Some(a), Some(b)) => a.0 <= b.0,
                (Some(_), None) => true,
                _ => false,
            };
            if take_self {
                all.push(self.pieces[i]);
                i += 1;
            } else {
                all.push(other.pieces[j]);
                j += 1;
            }
        }
        Self {
            pieces: merge_sorted(all),
        }
    }

    /// Points in both sets.
    pub fn intersection(&self, other: &Self) -> Self {
        let mut out = Vec::new();
        let (mut i, mut j) = (0, 0);
        while let (Some(&(a0, a1)), Some(&(b0, b1))) = (self.pieces.get(i), other.pieces.get(j)) {
            let s = max(a0, b0);
            let e = min(a1, b1);
            if s < e {
                out.push((s, e));
            }
            if a1 < b1 {
                i += 1;
            } else {
                j += 1;
            }
        }
        // Pieces of each input are disjoint and non-touching, so the overlaps are too.
        Self { pieces: out }
    }

    /// Points in `self` but not in `other`.
    pub fn difference(&self, other: &Self) -> Self {
        let mut out = Vec::new();
        let mut j = 0;
        for &(a0, a1) in &self.pieces {
            let mut cur = a0;
            // Skip subtrahend pieces that end at or before this piece starts.
            while other.pieces.get(j).is_some_and(|&(_, b1)| b1 <= cur) {
                j += 1;
            }
            let mut k = j;
            while let Some(&(b0, b1)) = other.pieces.get(k) {
                if b0 >= a1 {
                    break;
                }
                if b0 > cur {
                    out.push((cur, b0));
                }
                cur = max(cur, b1);
                if cur >= a1 {
                    break;
                }
                k += 1;
            }
            if cur < a1 {
                out.push((cur, a1));
            }
            // Every piece before `k` ended inside this minuend piece, so cannot reach the
            // next one; piece `k` itself (if any) may, so the next scan starts there.
            j = k;
        }
        Self { pieces: out }
    }

    /// The part of the set inside `[start, end)`.
    ///
    /// # Errors
    /// As [`from_intervals`](Self::from_intervals), with `index` 0.
    pub fn clip(&self, start: T, end: T) -> Result<Self, IntervalError> {
        check(0, start, end)?;
        if start >= end {
            return Ok(Self::new());
        }
        Ok(self.intersection(&Self {
            pieces: vec![(start, end)],
        }))
    }
}

/// Merges a start-sorted list of non-empty pieces into normal form.
fn merge_sorted<T: IntervalBound>(sorted: Vec<(T, T)>) -> Vec<(T, T)> {
    let mut out: Vec<(T, T)> = Vec::with_capacity(sorted.len());
    for (s, e) in sorted {
        match out.last_mut() {
            Some(last) if s <= last.1 => last.1 = max(last.1, e),
            _ => out.push((s, e)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn set(v: &[(i64, i64)]) -> IntervalSet<i64> {
        IntervalSet::from_intervals(v.iter().copied()).expect("valid test intervals")
    }

    #[test]
    fn normalises_overlap_touch_and_empty() {
        let s = set(&[(5, 7), (1, 3), (3, 4), (2, 2), (6, 9), (11, 12)]);
        assert_eq!(s.as_slice(), &[(1, 4), (5, 9), (11, 12)]);
        assert_eq!(s.measure(), 8);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn refuses_reversed_and_non_finite_with_position() {
        assert_eq!(
            IntervalSet::from_intervals([(0, 1), (3, 2)]),
            Err(IntervalError::Reversed { index: 1 })
        );
        assert_eq!(
            IntervalSet::from_intervals([(0.0, 1.0), (f64::NAN, 2.0)]),
            Err(IntervalError::NotFinite { index: 1 })
        );
        assert_eq!(
            IntervalSet::from_intervals([(0.0, f64::INFINITY)]),
            Err(IntervalError::NotFinite { index: 0 })
        );
        let mut s = IntervalSet::new();
        assert_eq!(s.insert(2, 1), Err(IntervalError::Reversed { index: 0 }));
        assert_eq!(
            set(&[(0, 5)]).clip(4, 3),
            Err(IntervalError::Reversed { index: 0 })
        );
    }

    #[test]
    fn set_operations() {
        let a = set(&[(0, 10), (20, 30)]);
        let b = set(&[(5, 25)]);
        assert_eq!(a.union(&b).as_slice(), &[(0, 30)]);
        assert_eq!(a.intersection(&b).as_slice(), &[(5, 10), (20, 25)]);
        assert_eq!(a.difference(&b).as_slice(), &[(0, 5), (25, 30)]);
        assert_eq!(b.difference(&a).as_slice(), &[(10, 20)]);
        assert_eq!(a.clip(8, 22).unwrap().as_slice(), &[(8, 10), (20, 22)]);
        assert!(a.clip(3, 3).unwrap().is_empty());
    }

    #[test]
    fn difference_where_one_subtrahend_covers_several_pieces() {
        let a = set(&[(0, 2), (3, 5), (6, 8), (9, 12)]);
        let b = set(&[(1, 7), (10, 11)]);
        assert_eq!(
            a.difference(&b).as_slice(),
            &[(0, 1), (7, 8), (9, 10), (11, 12)]
        );
    }

    #[test]
    fn contains_is_half_open() {
        let s = set(&[(1, 3), (5, 6)]);
        assert!(!s.contains(0));
        assert!(s.contains(1));
        assert!(s.contains(2));
        assert!(!s.contains(3));
        assert!(s.contains(5));
        assert!(!s.contains(6));
        assert!(!IntervalSet::<f64>::new().contains(0.0));
        assert!(!IntervalSet::from_intervals([(0.0, 1.0)])
            .unwrap()
            .contains(f64::NAN));
    }

    #[test]
    fn integer_measure_spans_the_whole_type() {
        let s = set(&[(i64::MIN, 0), (1, i64::MAX)]);
        assert_eq!(s.measure(), u64::MAX - 1);
    }

    /// Reported case: 24 h planned + 20 h unplanned stops overlapping 14 h in a
    /// 168 h week give 30 h down (not 44 h).
    #[test]
    fn overlapping_categories_count_once() {
        let planned = IntervalSet::from_intervals([(0.0, 24.0)]).unwrap();
        let unplanned = IntervalSet::from_intervals([(10.0, 30.0)]).unwrap();
        let week = IntervalSet::from_intervals([(0.0, 168.0)]).unwrap();
        let up = week.difference(&planned.union(&unplanned));
        assert_eq!(up.measure(), 138.0);
    }

    fn arb_pieces() -> impl Strategy<Value = Vec<(i64, i64)>> {
        prop::collection::vec((-50i64..50, 0i64..20), 0..12)
            .prop_map(|v| v.into_iter().map(|(s, l)| (s, s + l)).collect())
    }

    /// Membership of every integer point in [-60, 80) — the reference model.
    fn points(s: &IntervalSet<i64>) -> Vec<bool> {
        (-60..80).map(|t| s.contains(t)).collect()
    }

    fn is_normal(s: &IntervalSet<i64>) -> bool {
        s.as_slice().iter().all(|&(a, b)| a < b) && s.as_slice().windows(2).all(|w| w[0].1 < w[1].0)
    }

    proptest! {
        #[test]
        fn operations_agree_with_pointwise_model(a in arb_pieces(), b in arb_pieces()) {
            let (sa, sb) = (set(&a), set(&b));
            let (pa, pb) = (points(&sa), points(&sb));
            for (op, expect) in [
                (sa.union(&sb), pa.iter().zip(&pb).map(|(x, y)| *x || *y).collect::<Vec<_>>()),
                (sa.intersection(&sb), pa.iter().zip(&pb).map(|(x, y)| *x && *y).collect()),
                (sa.difference(&sb), pa.iter().zip(&pb).map(|(x, y)| *x && !*y).collect()),
            ] {
                prop_assert!(is_normal(&op));
                prop_assert_eq!(points(&op), expect.clone());
                // Unit-spaced integer model: measure = number of covered points.
                prop_assert_eq!(op.measure() as usize, expect.iter().filter(|x| **x).count());
            }
        }

        #[test]
        fn measure_is_inclusion_exclusion(a in arb_pieces(), b in arb_pieces()) {
            let (sa, sb) = (set(&a), set(&b));
            prop_assert_eq!(
                sa.union(&sb).measure() + sa.intersection(&sb).measure(),
                sa.measure() + sb.measure()
            );
            prop_assert_eq!(
                sa.difference(&sb).measure() + sa.intersection(&sb).measure(),
                sa.measure()
            );
        }

        #[test]
        fn insert_equals_union(a in arb_pieces(), s in -50i64..50, l in 0i64..20) {
            let mut x = set(&a);
            x.insert(s, s + l).unwrap();
            prop_assert_eq!(x, set(&a).union(&set(&[(s, s + l)])));
        }
    }
}
