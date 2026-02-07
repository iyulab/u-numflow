//! Specialized data structures for optimization algorithms.
//!
//! # Available Structures
//!
//! - [`UnionFind`]: Disjoint-set forest with path compression and union by rank

mod union_find;

pub use union_find::UnionFind;
