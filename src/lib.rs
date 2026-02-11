//! # u-numflow
//!
//! Mathematical primitives for the U-Engine ecosystem.
//!
//! This crate provides foundational mathematical, statistical, and probabilistic
//! building blocks that are domain-agnostic. It knows nothing about scheduling,
//! nesting, geometry, or any consumer domain.
//!
//! ## Modules
//!
//! - [`stats`] — Descriptive statistics with numerical stability guarantees
//! - [`distributions`] — Probability distributions (Uniform, Triangular, PERT, Normal, LogNormal, Weibull)
//! - [`special`] — Special mathematical functions (normal CDF, inverse normal CDF)
//! - [`matrix`] — Dense matrix operations (multiply, determinant, inverse, Cholesky)
//! - [`random`] — Random number generation, shuffling, and weighted sampling
//! - [`collections`] — Specialized data structures (Union-Find)
//!
//! ## Design Philosophy
//!
//! - **Numerical stability first**: Welford's algorithm for variance,
//!   Neumaier summation for accumulation
//! - **Reproducibility**: Seeded RNG support for deterministic experiments
//! - **Property-based testing**: Mathematical invariants verified via proptest

pub mod collections;
pub mod distributions;
pub mod matrix;
pub mod random;
pub mod special;
pub mod stats;
