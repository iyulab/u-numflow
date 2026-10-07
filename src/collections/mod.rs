//! Specialized data structures for optimization algorithms.
//!
//! # Available Structures
//!
//! - [`UnionFind`]: Disjoint-set forest with path compression and union by rank
//! - [`IntervalSet`]: Finite union of half-open intervals with union,
//!   intersection, difference and an overlap-safe measure

mod interval_set;
mod union_find;

pub use interval_set::{IntervalBound, IntervalError, IntervalSet};
pub use union_find::UnionFind;
