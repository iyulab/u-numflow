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
| `distributions` | Probability distributions: Uniform, Triangular, PERT, Normal, LogNormal, Weibull, Exponential, Gamma, Beta, χ² |
| `special` | Special functions: normal CDF and tail-precise survival function (and their inverses), t/F/chi² CDF and quantiles, the noncentral t CDF (Lenth AS 243 — the distribution power is computed from), regularized incomplete beta/gamma, erf |
| `transforms` | Data transformations: Box-Cox (λ via MLE golden-section search), inverse Box-Cox |
| `fourier` | Discrete Fourier transform of any length (radix-2 for powers of two, Bluestein otherwise): `fft`, `ifft`, `rfft`, `Complex` |
| `matrix` | Dense matrix operations: determinant, inverse, Cholesky decomposition, Jacobi eigenvalue decomposition |
| `random` | Seeded RNG, Fisher-Yates shuffle, weighted sampling, random subset selection |
| `collections` | Specialized data structures: Union-Find with path compression and union-by-rank; `IntervalSet` — unions of half-open intervals with union, intersection, difference, clip and a measure that counts overlapping input once |

## Design Philosophy

- **Numerical stability first** — Welford's algorithm for variance, Neumaier summation for accumulation
- **Reproducibility** — Seeded RNG support for deterministic experiments
- **Property-based testing** — Mathematical invariants verified via `proptest`

## Quick Start

```toml
[dependencies]
u-numflow = "0.8"
```

```rust
use u_numflow::distributions::Pert;
use u_numflow::random::{create_rng, shuffle};
use u_numflow::stats::WelfordAccumulator;

// Online statistics with numerical stability (Welford)
let mut stats = WelfordAccumulator::new();
for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
    stats.update(x);
}
assert_eq!(stats.mean(), Some(3.0));

// PERT distribution: moments and quantiles; sample by inverting a uniform draw
let pert = Pert::new(1.0, 4.0, 7.0).unwrap();
assert_eq!(pert.mean(), 4.0);
let p90 = pert.quantile(0.9).unwrap();
assert!(p90 > 4.0 && p90 < 7.0);

// Seeded shuffling for reproducibility
let mut rng = create_rng(42);
let mut items = vec![1, 2, 3, 4, 5];
shuffle(&mut items, &mut rng);

// Box-Cox transformation (non-normal data normalization)
use u_numflow::transforms::{box_cox, estimate_lambda};
let data = [1.0, 2.0, 4.0, 8.0, 16.0];
let fit = estimate_lambda(&data, -2.0, 2.0).unwrap(); // MLE via golden-section
let transformed = box_cox(&data, fit.lambda).unwrap();
assert_eq!(transformed.len(), data.len());

// Discrete Fourier transform of any length
use u_numflow::fourier::rfft;
let signal: Vec<f64> = (0..30).map(|j| (2.0 * std::f64::consts::PI * 3.0 * j as f64 / 30.0).sin()).collect();
let spectrum = rfft(&signal);           // 30 complex bins; bin 3 carries the energy
assert!(spectrum[3].norm() > 14.0);

// Interval sets: overlapping stops are down-time once, not twice
use u_numflow::collections::IntervalSet;
let planned = IntervalSet::from_intervals([(0.0, 24.0)]).unwrap();
let unplanned = IntervalSet::from_intervals([(10.0, 30.0)]).unwrap();
let week = IntervalSet::from_intervals([(0.0, 168.0)]).unwrap();
let up = week.difference(&planned.union(&unplanned));
assert_eq!(up.measure(), 138.0);
```

`IntervalSet<T>` works over `f32`, `f64` and every primitive integer type. Its
measure has the type of a distance — the unsigned type of the same width for
integers (as `i64::abs_diff` returns `u64`), so it cannot overflow. A reversed
interval (`start > end`) or a NaN/infinite bound is refused with its position,
never swapped or dropped.

## Build & Test

```bash
cargo build
cargo test
```

## Dependencies

- `rand` 0.10 — Random number generation
- `proptest` 1.4 — Property-based testing (dev only)

## License

MIT License — see [LICENSE](LICENSE).

## npm (WebAssembly)

```bash
npm install @iyulab/u-numflow
```

The package resolves per environment via a conditional `exports` map:

| Environment | Entry |
|---|---|
| Bundlers (webpack, Vite, …) | ESM + WebAssembly ESM-integration (`default` condition) |
| Node.js — `require()`, ESM `import`, CJS TS runners (`tsx`, `ts-node`) | CJS glue loading the wasm from the filesystem (`node` condition) — no loader hooks or flags |

A browser **without** a bundler is not supported: the package loads its `.wasm`
file with an ES module import, which browsers refuse (`application/wasm` is not a
module script type), so `<script type="module">` from a CDN fails, and CDN
re-bundling services fail on the same import. Use a bundler or Node.

Exported functions: `mean`, `std_dev`, `variance`, `normal_cdf`, `normal_sf` (upper tail `P(Z > x)`, computed directly so tail probabilities keep ~15 significant digits), `box_cox`, `estimate_lambda` (returns `{ lambda, at_bound }` — `at_bound` is `true` when the likelihood was still rising at an end of the search range, so `lambda` is that limit), and
`rfft(data) -> Float64Array` — the DFT of a real sequence of any length, interleaved as
`[re0, im0, re1, im1, …]` (bins `k` and `n − k` are conjugates, so `0..=n/2` describes the spectrum).

Every `data` argument is a `number[]` or a `Float64Array`, read exactly as sent: an
element that is not a number (`null`, `undefined`, a string) throws `malformed_input`
and a NaN or ±Infinity throws `value_not_finite`, each with the element's `index` —
`mean([1, null, 3])` throws rather than averaging the `null` as 0. `mean` of no values
throws `empty_input`; `std_dev` and `variance` of fewer than 2 throw `insufficient_data`.

Distribution functions for critical values and p-values — each returns a `number` and
**throws** (an `Error` naming the argument — see *Errors* below) when an argument is outside the domain,
rather than returning `NaN`:

| Function | Returns |
|---|---|
| `inverse_normal_cdf(p)` | `z` with `P(Z ≤ z) = p` |
| `t_distribution_cdf(t, df)` / `t_distribution_quantile(p, df)` | Student's t; a two-sided critical value at level α is `t_distribution_quantile(1 − α/2, df)` |
| `f_distribution_cdf(x, df1, df2)` / `f_distribution_quantile(p, df1, df2)` | F |
| `chi_squared_cdf(x, k)` / `chi_squared_quantile(p, k)` | χ² |

`p` must lie strictly between 0 and 1 and every degrees-of-freedom argument must be a
finite number `> 0` (fractional values are allowed).

```js
const { t_distribution_quantile } = require("@iyulab/u-numflow");
t_distribution_quantile(0.975, 10); // 2.2281…
```

Interval sets — each argument is a `[start, end][]` of half-open intervals, and every
result is in normal form (sorted, disjoint, touching pieces merged, empty ones dropped):

| Function | Returns |
|---|---|
| `interval_normalize(intervals)` | The union of `intervals`, as `[number, number][]` |
| `interval_measure(intervals)` | Its total length — each point counted once, however many rows cover it |
| `interval_union(a, b)` / `interval_intersection(a, b)` / `interval_difference(a, b)` | Set operations, as `[number, number][]`; clip to a window with `interval_intersection(a, [[from, to]])` |

```js
const { interval_union, interval_difference, interval_measure } = require("@iyulab/u-numflow");
const down = interval_union([[0, 24]], [[10, 30]]);       // [[0, 30]] — the overlap counts once
const up = interval_difference([[0, 168]], down);          // [[30, 168]]
console.log(interval_measure(up));                         // 138
```

**Errors.** A refusal throws an `Error` whose `message` is readable text and which
carries a `code` naming the reason, next to the values behind it:

```js
const { t_distribution_quantile } = require("@iyulab/u-numflow");
try {
  t_distribution_quantile(1.5, 10);
} catch (err) {
  console.log(err.code, err.parameter, err.min, err.max, err.got); // parameter_out_of_range p 0 1 1.5
}
```

| `code` | Fields | Meaning |
|---|---|---|
| `parameter_out_of_range` | `parameter`, `min`, `max` (or `null`), `got` | `p` not strictly inside (0, 1), or a degrees of freedom that is not a finite number `> 0` (both bounds excluded) |
| `malformed_input` | `parameter`, `index` (or absent) | A `data` argument that is not an array or `Float64Array`, or an element that is not a number; an interval row that is not two numbers (`parameter` is its path, e.g. `a[2]`) |
| `reversed_interval` | `parameter`, `index` | An interval row with start > end — refused, not swapped |
| `value_not_finite` | `parameter`, `index` for an array element | A NaN argument, or a NaN or infinity in any `data` array or interval row (`parameter` is the row's path) |
| `empty_input` | `parameter` | `mean` of no values |
| `non_positive_data` | — | Box-Cox data with a value `≤ 0` |
| `insufficient_data` | `parameter`, `min`, `got` for `std_dev`/`variance`; — for Box-Cox | Fewer values than the function needs (`std_dev`/`variance` 2, Box-Cox 2) |
| `invalid_transform` | — | A Box-Cox result that is not finite |
| `invalid_lambda_range` | — | `estimate_lambda` bounds that are not finite with `lambda_min < lambda_max` |

### TypeScript

Every exported function declares its parameter and return types, and the
declarations are generated from the same structs the binding reads and
serialises, so they cannot drift from what it actually accepts and returns:

```ts
export function mean(data: number[] | Float64Array): number;
export function estimate_lambda(data: number[] | Float64Array, lambda_min: number, lambda_max: number): LambdaEstimateDto;
```

An absent optional value is declared `T | undefined`, which is what the binding
sends. Nothing needs an `as` cast -- and a wrong assumption about a result's
shape is a compile error rather than something that fails at run time.

A `data` parameter is declared `number[] | Float64Array` and read by the
binding itself rather than copied into a typed array by the generated glue, so
a value the declaration does not allow is refused where it sits instead of
being converted. The publishing workflow keeps every declaration free of `any`.

## Related

- [u-metaheur](https://github.com/iyulab/u-metaheur) — Metaheuristic optimization (GA, SA, ALNS, CP)
- [u-geometry](https://github.com/iyulab/u-geometry) — Computational geometry
- [u-schedule](https://github.com/iyulab/u-schedule) — Scheduling framework
- [u-nesting](https://github.com/iyulab/U-Nesting) — 2D/3D nesting and bin packing
