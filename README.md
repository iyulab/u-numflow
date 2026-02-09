# u-numflow

**Domain-agnostic mathematical primitives in Rust**

[![Crates.io](https://img.shields.io/crates/v/u-numflow.svg)](https://crates.io/crates/u-numflow)
[![docs.rs](https://docs.rs/u-numflow/badge.svg)](https://docs.rs/u-numflow)
[![CI](https://github.com/iyulab/u-numflow/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/u-numflow/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

u-numflow provides foundational mathematical, statistical, and probabilistic building blocks. Entirely domain-agnostic with no external dependencies beyond `rand`.

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

## Quick Start

```toml
[dependencies]
u-numflow = { git = "https://github.com/iyulab/u-numflow" }
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

## Related

- [u-metaheur](https://github.com/iyulab/u-metaheur) — Metaheuristic optimization (GA, SA, ALNS, CP)
- [u-geometry](https://github.com/iyulab/u-geometry) — Computational geometry
- [u-schedule](https://github.com/iyulab/u-schedule) — Scheduling framework
- [u-nesting](https://github.com/iyulab/U-Nesting) — 2D/3D nesting and bin packing
