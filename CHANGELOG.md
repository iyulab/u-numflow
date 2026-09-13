# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.1 onward; earlier entries list release dates only (see git history).

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
