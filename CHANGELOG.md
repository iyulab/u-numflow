# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.1 onward; earlier entries list release dates only (see git history).

## [Unreleased]

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
