# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.1 onward; earlier entries list release dates only (see git history).

## [Unreleased]

### Added

- `special::standard_normal_sf(x)` -- the normal upper tail `P(Z > x)`,
  computed directly as `erfc(x/√2)/2`. Tail probabilities (PPM defect rates,
  one-sided p-values) keep about 15 significant digits up to `x ≈ 38`, where
  `1.0 - standard_normal_cdf(x)` has none left. Also exposed as the WASM
  function `normal_sf`.
- `special::inverse_normal_sf(q)` -- the deviate with upper-tail probability
  `q`, inverting the tail probability as given instead of through `1 - q`.

### Fixed

- **`erfc` was computed as `1.0 - erf(x)`**, contradicting its own
  documentation ("more numerically stable than `1.0 - erf(x)`"): in the tail it
  returned a cancellation residue. `erf` and `erfc` are now the FreeBSD msun
  `s_erf.c` rational approximations (erfc formed from an exponential, no
  subtraction), accurate to about one ulp in `erf` and to full relative
  precision in `erfc` — `erfc(10)` is `2.088487583762545e-45`, not `0`.
- **`standard_normal_cdf` had an absolute, not relative, error bound**
  (Abramowitz & Stegun 26.2.17, `< 7.5e-8`), so its tails had no significant
  digits: relative error 5e-5 at 3σ growing to 6.5e-3 at 7σ. It is now
  `erfc(−x/√2)/2`, with relative error of order 1e-15 in both tails.
- **`inverse_normal_cdf` was accurate only to `4.5e-4`** (A&S 26.2.23). It is
  now Wichura's AS 241 (PPND16), accurate to about 1 part in 1e16.

Every value these functions return changes in its trailing digits; results
that depended on the old approximations' error (tolerance-pinned tests, cached
reference outputs) should be regenerated.

## [0.5.0] - 2026-09-13

### Added

- `fourier` module -- discrete Fourier transform of a complex or real sequence
  of any length: `fft`, `ifft`, `rfft` and a minimal `Complex` type. A
  power-of-two length uses the iterative radix-2 Cooley-Tukey algorithm; any
  other length uses Bluestein's chirp-z algorithm, so every length is
  O(n log n). No new dependencies. Property-tested against the naive DFT for
  every length up to 64, plus Parseval's identity and the Hermitian symmetry of
  a real signal's spectrum. Also exposed as the WASM function `rfft`
  (interleaved `[re, im, …]`).

## [0.4.0] - 2026-09-07

### Changed (breaking)

- **`rand` is now 0.10** (previously 0.9). `rand`'s traits and types appear in
  this crate's public signatures — `shuffle`, `shuffled_indices`,
  `weighted_choose` and `AliasTable::sample` are generic over `R: Rng`, and
  `create_rng` returns `rand::rngs::SmallRng` — so the two versions are not
  interchangeable at the boundary. Callers must move to `rand` 0.10 as well;
  passing a 0.9 generator no longer satisfies these bounds. The generated
  sequences for a given seed are unchanged, so seeded results are identical to
  the previous release.
- **The minimum supported Rust version is now declared as 1.85** and is verified
  by building on that exact toolchain; 1.84 and below fail. The crate previously
  declared no `rust-version` at all, so this makes an existing requirement
  explicit rather than raising one that was already documented.

### Changed

- **`getrandom` is now 0.4** on WebAssembly targets. It reaches the browser
  entropy source through its `wasm_js` crate feature alone; the
  `RUSTFLAGS --cfg getrandom_backend="wasm_js"` that 0.3 required is no longer
  needed.

## [0.3.1] - 2026-07-05

### Fixed

- npm: expose the `./package.json` subpath in the `exports` map so tools
  that `require('<pkg>/package.json')` (license scanners, version
  reporters) keep working alongside the conditional exports introduced in
  the previous release (`ERR_PACKAGE_PATH_NOT_EXPORTED`).

## [0.3.0] - 2026-07-05

### Added

- `wasm` feature — wasm-bindgen bindings backing the `@iyulab/u-numflow`
  npm package (`box_cox`, `estimate_lambda`, `mean`, `normal_cdf`,
  `std_dev`, `variance`). In git since 2026-03-06 but never published to
  crates.io; minor bump per additive-API rule.

### Fixed

- Distribution/special-function input validation hardening + docs
  (in git since 2026-03-06, previously unreleased).
- **npm packaging — Node-compatible entry.** The npm package previously
  shipped only the wasm-bindgen *bundler*-target output, whose static
  `.wasm` import fails on Node's CJS path (`tsx`/`ts-node` in non-ESM
  packages) with an opaque `SyntaxError: Invalid or unexpected token`.
  The package now additionally ships the *nodejs*-target CJS glue under
  `node/` and routes Node consumers to it via a conditional `exports`
  map (`node` → CJS with filesystem wasm loading, `default` → bundler
  ESM). `require()`, native ESM `import`, and CJS TS runners all work
  without loader hooks. A pre-publish smoke test (CJS `require` + ESM
  `import`) now guards this path in CI. Math API unchanged.

## [0.2.1] - 2026-03-05

### Changed

- Publish pipeline: idempotent npm publish (skip when the version already
  exists on the registry).

## Earlier releases

- 0.2.0 — 2026-02-12
- 0.1.0 — 2026-02-09
