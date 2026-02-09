# u-optim

**Mathematical primitives for the U-Engine ecosystem**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

## Overview

u-optim provides foundational mathematical, statistical, and probabilistic building blocks that are entirely domain-agnostic. It sits at **Layer 1 (Foundation)** in the U-Engine architecture and knows nothing about scheduling, nesting, geometry, or any consumer domain.

```
Consumer Layer
  u-aps, u-nesting
Framework Layer
  u-schedule, u-nesting  →  u-metaheur  →  u-geometry
                                 ↓
Foundation Layer              u-optim  ← you are here
```

## Modules

| Module | Description |
|--------|-------------|
| `stats` | Descriptive statistics (mean, variance, skewness, kurtosis) with Welford's online algorithm and Neumaier summation |
| `distributions` | Probability distributions: Uniform, Triangular, PERT, Normal, LogNormal |
| `special` | Special mathematical functions: normal CDF (Abramowitz-Stegun), inverse normal CDF (Beasley-Springer-Moro) |
| `random` | Seeded RNG, Fisher-Yates shuffle, weighted sampling, random subset selection |
| `collections` | Specialized data structures: Union-Find with path compression and union-by-rank |

## Design Philosophy

- **Numerical stability first** — Welford's algorithm for variance, Neumaier summation for accumulation
- **Reproducibility** — Seeded RNG support for deterministic experiments
- **Property-based testing** — Mathematical invariants verified via `proptest`
- **Zero domain coupling** — No scheduling, nesting, or manufacturing concepts

## Quick Start

```toml
[dependencies]
u-optim = { git = "https://github.com/iyulab/u-optim" }
```

```rust
use u_optim::stats::OnlineStats;
use u_optim::distributions::{PertDistribution, Distribution};
use u_optim::random::Rng;

// Online statistics with numerical stability
let mut stats = OnlineStats::new();
for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
    stats.push(x);
}
assert_eq!(stats.mean(), 3.0);

// PERT distribution sampling
let pert = PertDistribution::new(1.0, 4.0, 7.0);
let mut rng = Rng::seed_from_u64(42);
let sample = pert.sample(&mut rng);

// Seeded shuffling for reproducibility
let mut items = vec![1, 2, 3, 4, 5];
u_optim::random::shuffle(&mut items, &mut rng);
```

## Build & Test

```bash
cargo build
cargo test
```

## Dependencies

- `rand` 0.9 — Random number generation
- `proptest` 1.4 — Property-based testing (dev only)

## License

MIT License — see [LICENSE](LICENSE).
