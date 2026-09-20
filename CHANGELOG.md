# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.1 onward; earlier entries list release dates only (see git history).

## [0.6.2] - 2026-09-20

### Added

- **Every exported WASM function declares its return type.** They were typed
  `(...) => any`, with the output's field *names* in the doc comment and the
  element types only in the README -- so a consumer's wrong assumption about a
  result's shape compiled and shipped. `as` is the only thing that can be
  written against `any`, and it is exactly the construct that silences this.

  The declarations are derived from the structs the binding already
  serialises, so there is no second copy to drift: `tsify` emits the interface
  and `unchecked_return_type` names it in the signature. The runtime path is
  unchanged -- same serializer, same bytes. An optional field is declared
  `T | undefined`, which is what the binding sends.

  A publish-path check (`scripts/check-typed-dts.sh`) fails the release if any
  exported function returns `any`, or if a declaration names a type the file
  does not declare. It runs before publishing rather than beside it in CI,
  because the two run on the same push.

  Inputs remain `any`; they are validated at the boundary.

## [0.6.1] - 2026-09-16

### Added

- **`stats::skewness_moment` and `stats::kurtosis_moment`** — the moment
  coefficients `g₁ = m₃/m₂^(3/2)` and `g₂ = m₄/m₂² − 3`, with no bias
  correction. These are what published moment formulas (Jarque-Bera,
  D'Agostino's K²) are written in, and what SciPy's `bias=True`, R's
  `e1071` type 1 and statsmodels compute. `skewness` and `kurtosis` keep
  returning the bias-adjusted `G₁`/`G₂` that Excel's `SKEW()`/`KURT()` report.

  The crate already computed both pairs — the adjusted ones are the moment ones
  times a factor — and returned only one of them, so a caller who needed the
  other had to recompute the central moments from scratch. At n = 20 the two
  skewness estimators differ by 8 %, and the kurtosis adjustment can carry a
  mildly platykurtic sample across zero.

### Fixed

- **`transforms::estimate_lambda` no longer stops short of the range on a
  narrow sample, and no longer calls that stop an interior estimate.** Two
  faults that hid each other:

  - The profile likelihood was computed from the unnormalised transform
    `(y^λ - 1)/λ`. For a sample clustered well away from 1 -- 250 readings
    inside a 0.13 % band around 25 -- `y^λ` is about `1e-7` at `λ = -5`, so
    every transformed value sits next to `-1` and the differences between them,
    which are all the variance is made of, fall off the bottom of the mantissa.
    The likelihood came out non-monotone at the `1e-5` level and the search
    converged on that noise. It is now computed from the geometric-mean
    normalised transform `g·expm1(λ·ln(y/g))/λ` (Box & Cox 1964, §3), which
    keeps the differences at full precision and folds in the Jacobian.
  - `at_bound` asked the golden-section search whether an end of its bracket
    had ever moved. A likelihood that is monotone across the range leaves the
    bracket in the middle, so a λ on the edge of the range was reported as
    interior. It is now decided by comparing the likelihood at each end against
    the interior candidate.

  On the sample above, `estimate_lambda(y, -5.0, 5.0)` returned
  `{ lambda: -4.992668718202271, at_bound: false }` and now returns
  `{ lambda: -5.0, at_bound: true }`.

- **`transforms::box_cox` computes `expm1(λ·ln y)` rather than `y^λ - 1`.**
  The two agree mathematically; the second loses the result to cancellation
  whenever `y^λ` is near 1, which is every λ near zero and every `y` near one.

## [0.6.0] - 2026-09-15

### Changed (breaking)

- **`transforms::estimate_lambda` returns a `LambdaEstimate { lambda, at_bound }`**
  instead of a bare `f64`. `at_bound` is `true` when the likelihood maximum is on
  an end of the search range — the likelihood was still rising there, so the
  range cut the search short — and `lambda` is then that range limit exactly
  (previously the bracket midpoint, e.g. `1.9999996` for a range ending at 2,
  indistinguishable from an interior estimate). The WASM function
  `estimate_lambda` returns `{ lambda, at_bound }` likewise.
- An empty or non-finite λ range is now `TransformError::InvalidLambdaRange`
  (previously misreported as `InsufficientData`).

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
- **`estimate_lambda` could panic** when `y^λ` overflowed `f64` for a λ in the
  search range (large data, large |λ|): the non-finite transform reached an
  `expect` on its variance. Such a λ is now not a candidate. Data containing NaN
  or an infinity is refused as `TransformError::NonFiniteData` (it passed the
  positivity check before).
- `box_cox` refuses a result that is not finite (`TransformError::InvalidTransform`),
  as `inverse_box_cox` already did for its own direction.

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
