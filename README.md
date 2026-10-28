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
| `optimize` | Derivative-free local minimisation of a function of several variables: `nelder_mead` (Nelder–Mead simplex, dimension-adaptive coefficients, relative tolerances) |
| `matrix` | Dense matrix operations: determinant, inverse, Cholesky decomposition, Jacobi eigenvalue decomposition |
| `random` | Seeded RNG, Fisher-Yates shuffle, weighted sampling, random subset selection |
| `distributions::Sample` | Random variates from every continuous distribution — `sample(&mut rng)` and reproducible `sample_n(n, seed)`; inversion where the quantile is direct, Marsaglia–Tsang for the gamma family (Gamma, χ², Beta, PERT); uniforms from the open interval (0, 1), so no variate is infinite |
| `collections` | Specialized data structures: Union-Find with path compression and union-by-rank; `IntervalSet` — unions of half-open intervals with union, intersection, difference, clip and a measure that counts overlapping input once |
| `units` | Units of measure: dimensional analysis over the seven SI base quantities, UCUM-syntax unit expressions (`kW.h`, `m/s2`, `J/(kg.K)`), conversion of readings (with a scale's offset — 20 °C is 293.15 K) and of differences (without — a rise of 20 °C is a rise of 20 K); the SI built in, every other unit defined by the caller |

## Design Philosophy

- **Numerical stability first** — Welford's algorithm for variance, Neumaier summation for accumulation
- **Reproducibility** — Seeded RNG support for deterministic experiments
- **Property-based testing** — Mathematical invariants verified via `proptest`

## Quick Start

```toml
[dependencies]
u-numflow = "0.16"
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

// PERT distribution: moments, quantiles and reproducible samples
use u_numflow::distributions::Sample;
let pert = Pert::new(1.0, 4.0, 7.0).unwrap();
assert_eq!(pert.mean(), 4.0);
let p90 = pert.quantile(0.9).unwrap();
assert!(p90 > 4.0 && p90 < 7.0);
let draws = pert.sample_n(1000, 7); // same (n, seed), same draws
assert!(draws.iter().all(|&x| (1.0..=7.0).contains(&x)));

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
let spectrum = rfft(&signal);           // bins 0..=15 (n / 2 + 1); bin 3 carries the energy
assert!(spectrum[3].norm() > 14.0);

// Interval sets: overlapping stops are down-time once, not twice
use u_numflow::collections::IntervalSet;
let planned = IntervalSet::from_intervals([(0.0, 24.0)]).unwrap();
let unplanned = IntervalSet::from_intervals([(10.0, 30.0)]).unwrap();
let week = IntervalSet::from_intervals([(0.0, 168.0)]).unwrap();
let up = week.difference(&planned.union(&unplanned));
assert_eq!(up.measure(), 138.0);
```

```rust
// Units: the SI is built in; a unit whose size varies by convention is yours to define
use u_numflow::units::Units;
let mut units = Units::si();
units.define("cal_IT", 4.1868, "J", 0.0).unwrap(); // the international calorie
units.define("toe", 1e7, "kcal_IT", 0.0).unwrap();  // tonne of oil equivalent
assert!((units.convert(1.0, "toe", "GJ").unwrap() - 41.868).abs() < 1e-9);
assert!((units.convert(20.0, "Cel", "K").unwrap() - 293.15).abs() < 1e-9); // a reading
assert_eq!(units.convert_difference(20.0, "Cel", "K").unwrap(), 20.0);    // a change
assert!(units.convert(1.0, "m3", "kg").is_err());                         // volume is not mass
```

Unit expressions follow UCUM's case-sensitive syntax: an atom with an optional
decimal prefix (`km`, `kW`, `uA` — `u` is micro), an integer exponent after it
(`m3`, `s-2`), `.` for a product and `/` for a quotient read left to right
(`kg/m.s` is `(kg/m)·s`), parentheses, integer factors (`1000.m`), and `{…}`
annotations, which have no effect (`m3{gas}`). Built in: the SI base units
(`m`, `g`, `s`, `A`, `K`, `mol`, `cd`), the 22 derived units with special names
(`N`, `Pa`, `J`, `W`, `Cel`, …), `min`, `h`, `d`, `L`/`l`, `t`, and the prefixes
`Q` … `q`. No unit table beyond the SI is embedded: the calorie (three
definitions, up to 0.07 % apart), the tonne of oil equivalent, imperial units
and the normal cubic metre are the caller's `define` calls, so the crate never
picks one convention silently. A scale with an offset (`Cel`) stands alone — a
prefix, exponent or product on it is refused.

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

## .NET (NuGet)

```bash
dotnet add package UNumflow
```

The distributions (CDF, quantile, moments, seeded sampling), sample statistics
(median, R-7 quantiles), interval sets and units, backed by this crate through its C ABI (the `ffi` feature) — see
[bindings/csharp/README.md](bindings/csharp/README.md). Refusals carry the same
`code` and fields as the WebAssembly binding's.

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

Exported functions: `mean`, `std_dev`, `variance`, `median`, `quantiles(data, probabilities) -> Float64Array` (sample quantiles in the order asked, R-7 linear interpolation — the default of R, NumPy and Excel — with one sort for all of them), `normal_cdf`, `normal_sf` (upper tail `P(Z > x)`, computed directly so tail probabilities keep ~15 significant digits), `box_cox`, `estimate_lambda` (returns `{ lambda, at_bound }` — `at_bound` is `true` when the likelihood was still rising at an end of the search range, so `lambda` is that limit), and
`rfft(data) -> Float64Array` — the DFT of a real sequence of any length: bins `0..=n/2`
(`n / 2 + 1` of them, as NumPy's `rfft`), interleaved as `[re0, im0, re1, im1, …]`
(bins `k` and `n − k` are conjugates, so these describe the spectrum in full).

Every `data` argument is a `number[]` or a `Float64Array`, read exactly as sent: an
element that is not a number (`null`, `undefined`, a string) throws `malformed_input`
and a NaN or ±Infinity throws `value_not_finite`, each with the element's `index` —
`mean([1, null, 3])` throws rather than averaging the `null` as 0. `mean` of no values
throws `empty_input`; `std_dev` and `variance` of fewer than 2 throw `insufficient_data`.
`median` and `quantiles` of no values throw `empty_input`; a probability outside `[0, 1]`
throws `parameter_out_of_range` with `parameter` `probabilities[i]`, `min` 0 and `max` 1.

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

Any of the crate's distributions by specification — `distribution` is an object
`{ kind, ...parameters }`, declared as the union type `DistributionSpec`:

| `kind` | Parameters |
|---|---|
| `uniform` | `min`, `max` |
| `triangular` | `min`, `mode`, `max` |
| `pert` | `min`, `mode`, `max`, `lambda?` (default 4) |
| `normal` / `lognormal` | `mu`, `sigma` |
| `weibull` | `shape`, `scale` |
| `exponential` | `rate` |
| `gamma` | `shape`, `rate` |
| `beta` | `alpha`, `beta` |
| `chi_squared` | `k` |

| Function | Returns |
|---|---|
| `distribution_cdf(distribution, x)` | `P(X ≤ x)` |
| `distribution_quantile(distribution, p)` | `x` with `P(X ≤ x) = p`, `p` strictly inside (0, 1) |
| `distribution_sample(distribution, n, seed)` | `Float64Array` of `n` variates — the same `(distribution, n, seed)` always gives the same values |

```js
const { distribution_quantile, distribution_sample } = require("@iyulab/u-numflow");
distribution_quantile({ kind: "chi_squared", k: 4 }, 0.95);              // 9.4877…
const lives = distribution_sample({ kind: "weibull", shape: 2, scale: 100 }, 1000, 42);
console.log(lives.length);                                                // 1000
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

Units of measure — `from`, `to` and `unit` are unit expressions (UCUM syntax, see the
Rust section above) over the SI and the optional `definitions`, an array of
`UnitDefinitionSpec` `{ symbol, unit, factor?, offset? }` (`symbol` = `factor` × `unit`,
plus `offset` SI units at its zero; `factor` defaults to 1, `offset` to 0; a definition
may use the ones before it):

| Function | Returns |
|---|---|
| `unit_convert(value, from, to, definitions?)` | A reading converted — 20 `Cel` is 293.15 `K` |
| `unit_convert_difference(value, from, to, definitions?)` | An amount of change converted — the offset cancels: a rise of 20 `Cel` is a rise of 20 `K` |
| `unit_info(unit, definitions?)` | `UnitInfo` `{ dimension, exponents, scale, offset }` — the dimension in SI base units (`"kg.m2.s-2"`; exponents of m, kg, s, A, K, mol, cd) and `value_in_si = value · scale + offset`. Two units convert into each other exactly when their dimensions are equal |

```js
const { unit_convert, unit_convert_difference, unit_info } = require("@iyulab/u-numflow");
const energy = [
  { symbol: "cal_IT", unit: "J", factor: 4.1868 },
  { symbol: "toe", unit: "kcal_IT", factor: 1e7 },
];
console.log(unit_convert(1, "Gcal_IT", "GJ", energy));     // 4.1868
console.log(unit_convert(20, "Cel", "K"));                 // 293.15
console.log(unit_convert_difference(20, "Cel", "K"));      // 20
console.log(unit_info("toe", energy).dimension);           // "kg.m2.s-2"
try {
  unit_convert(1, "m3", "kg");
} catch (err) {
  console.log(err.code, err.parameter, err.expected, err.got); // dimension_mismatch to m3 kg
}
```

`codes()` returns every refusal `code` the functions throw, with the fields that
can come with it — `{ refusals: [{ code, fields }], violations: [] }` — built from
the table the refusals come from; the table below explains each. The C library
returns the same JSON from `unumflow_codes()` and the .NET package from
`Numflow.Codes`.

**Errors.** A refusal throws an `Error` whose `message` is readable text and which
carries a `code` naming the reason, next to the values behind it:

```js
const { t_distribution_quantile } = require("@iyulab/u-numflow");
try {
  t_distribution_quantile(1.5, 10);
} catch (err) {
  console.log(err.code, err.parameter, err.exclusive_min, err.exclusive_max, err.got);
  // parameter_out_of_range p 0 1 1.5
}
```

| `code` | Fields | Meaning |
|---|---|---|
| `parameter_out_of_range` | `parameter`, `min`, `max`, `got`; `exclusive_min` / `exclusive_max` on an open end | `p` not strictly inside (0, 1) (`exclusive_min` 0, `exclusive_max` 1), or a degrees of freedom that is not a finite number `> 0` (`exclusive_min` 0); a distribution parameter that must be `> 0` (`exclusive_min` 0; `parameter` is its path, e.g. `distribution.sigma`), or that falls outside the range the others leave it: `distribution.max` not above `min` (`exclusive_min` = `min`), `distribution.mode` outside `[min, max]` (`min`, `max`); `estimate_lambda`'s `lambda_min` not below `lambda_max` (`exclusive_max` = `lambda_max`); a whole `n` or `seed` outside its range, a negative one included (`min` 0, `max`); a unit definition's `factor` not `> 0` (`parameter` `definitions[i].factor`, `exclusive_min` 0) |
| `unknown_option` | `parameter`, `got`, `expected` | A `distribution.kind` the crate does not have |
| `malformed_input` | `parameter`, `index` (or absent) | A `data` argument that is not an array or `Float64Array`, or an element that is not a number; a `distribution` that is not an object with a string `kind`, lacks a parameter or has one its kind does not take; an interval row that is not two numbers (`parameter` is its path, e.g. `a[2]`); an `n` or `seed` with a fraction ("must be a whole number"); a unit expression that does not parse (`from`, `to`, `unit`, `definitions[i].unit`), a definition symbol that is empty, starts with a digit or holds an operator character (`definitions[i].symbol`), or a definition of the wrong shape (`definitions[i]`) |
| `unknown_unit` | `parameter`, `got` | A unit symbol that is neither built in nor defined, nor a prefix on one — `got` is the symbol (`hr` in `kW.hr`) |
| `dimension_mismatch` | `parameter`, `expected`, `got` | `unit_convert` between units of different dimensions — `parameter` is `to`, `expected` the dimension of `from` and `got` that of `to`, both in SI base units (`m3`, `kg`) |
| `invalid_option` | `parameter` | A scale with an offset (`Cel`, or a defined one) given a prefix, an exponent or a product — it has a meaning only alone |
| `duplicate_id` | `parameter`, `id`, `index` | A unit definition whose `symbol` is already a unit (built in or defined before) — `index` is the definition's position |
| `reversed_interval` | `parameter`, `index` | An interval row with start > end — refused, not swapped |
| `value_not_finite` | `parameter`, `index` for an array element | A NaN argument, an infinite `n`, `seed`, `lambda_min` or `lambda_max`, or a NaN or infinity in any `data` array or interval row (`parameter` is the row's path); a NaN or infinite `value` to convert or definition `factor` / `offset` |
| `empty_input` | `parameter` | `mean` of no values |
| `non_positive_data` | `parameter`, `index`, `got` | Box-Cox data with a value `≤ 0` — `index` is the first such value |
| `insufficient_data` | `parameter`, `min`, `got` | Fewer values than the function needs (`std_dev`/`variance` 2, Box-Cox 2) |
| `invalid_transform` | — | A Box-Cox result that is not finite |
| `invalid_inverse` | — | An inverse Box-Cox whose result is not finite |
| `internal` | — | A fault in this package rather than in the request — please report it |

In `parameter_out_of_range`, `min` and `max` are inclusive bounds and are always
present — `null` when that end is unbounded or open. An open (exclusive) end is
`exclusive_min` / `exclusive_max`, present only when the bound is open.

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
