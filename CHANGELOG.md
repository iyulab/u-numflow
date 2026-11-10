# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.1 onward; earlier entries list release dates only (see git history).

## [Unreleased]

## [0.16.3] - 2026-11-10

### Added

- `special::inverse_regularized_lower_gamma(a, p)`, `inverse_regularized_upper_gamma(a, q)` and
  `inverse_regularized_incomplete_beta(p, a, b)`: the inverses of `regularized_lower_gamma`, `regularized_upper_gamma`
  and `regularized_incomplete_beta` (SciPy's `gammaincinv`, `gammainccinv` and `betaincinv`), the solver behind the gamma,
  chi-squared and beta quantiles made public. The upper-tail inverse takes `q` as given, so a p-value of 10⁻²⁰⁰ names its
  chi-squared critical value (`2 · inverse_regularized_upper_gamma(k/2, p_value)`) where `1 − q` has no digits; the beta
  inverse mirrors (`inverse_regularized_incomplete_beta(1 − p, b, a) = 1 − x`) for a root within 10⁻¹⁶ of 1. They are not
  in the WASM, C ABI or .NET surfaces: the quantiles built on them already are.

### Fixed

- The normal tails lost digits with the square of the distance: `standard_normal_sf`, `standard_normal_cdf` (and so
  `normal_sf` / `normal_cdf` in every transport) rounded `x/√2` to a double before the exponential, which multiplies
  that rounding by `x²` — a relative error of 2·10⁻¹⁵ at 4 sd, 10⁻¹⁴ at 10 sd, 10⁻¹³ at 25 sd and 1.7·10⁻¹³ at 35 sd, against
  the documented 10⁻¹⁵. `standard_normal_pdf` did the same with `exp(−x²/2)` (6·10⁻¹⁴ at 35 sd). The exponent is now
  built from `x` with an exactly representable square (the construction fdlibm's `erfc` uses for its own argument), and the
  worst relative error over a grid to 38 sd, checked against mpmath at 40 digits, is 4·10⁻¹⁶ for all three. The
  quantiles (`inverse_normal_cdf`, `inverse_normal_sf`) were already at 7·10⁻¹⁶. The doctests' value of Φ(−6) was off in
  its last digits (`…948e-10`; the value is `…981e-10`) and is corrected.
- The radix-2 FFT built each stage's twiddles as a running product (`w ← w·step`), so their rounding grew with the
  stage length: against a transform with directly computed twiddles the relative error was 10⁻¹⁴ at 2¹⁰ points,
  10⁻¹³ at 2¹⁴, 2·10⁻¹² at 2¹⁸ and 4·10⁻¹¹ at 2²² (every `fft`, `ifft` and `rfft`, and Bluestein for any other length,
  which runs three power-of-two transforms; the autocorrelation, periodogram and period estimates built on them). The
  twiddles now come from a table whose entries are reduced to the first octant by integer arithmetic (exact
  reflections and quarter turns for the rest), and the error is 8·10⁻¹⁶ at 2²². The same table-per-stage layout keeps
  the speed (2²² points 0.37 s before, 0.40 s after); the table is `16·n` bytes, held for the one transform.
- `Exponential` and `Weibull` computed their CDF as `1 − exp(−y)`, their survival function (`survival`, `reliability`) as
  `1 − CDF` and their quantile as `−ln(1 − p)`. A small failure probability lost its digits (`F` at `y = 10⁻¹⁴` was
  off by 8·10⁻⁴, at `10⁻¹⁰` by 10⁻⁶), the quantile at `p = 10⁻¹⁵` was 1–3 % off, and the survival function was 0 from
  `y = 37` although it is a number to `y ≈ 745`. They use `expm1`, `ln_1p` and `exp(−y)` directly now: 2·10⁻¹⁶ for the
  CDF and quantile over `y` from 10⁻¹⁴ to 740 (and `p` from 10⁻¹⁵), and 4·10⁻¹⁴ at the far end of the survival function
  (`y ≈ 500`, where the rounding of `y` itself is multiplied by `y`).
- The quantiles of `GammaDistribution`, `ChiSquared`, `BetaDistribution` and `Pert` were wrong for a small probability:
  the gamma one never returned less than 1e-15 (a gamma(0.5) at `p = 10⁻¹⁰` has its quantile at 8·10⁻²¹), and the beta one
  bisected to an absolute width of 1e-14, so a quantile of 8·10⁻⁹ had 6 digits (`p = 10⁻¹⁵`, a PERT 6·10⁻⁴ off) and one
  of 10⁻³⁰ was off by a factor. Both solve for `ln x` now, by Newton steps on `ln P` kept in a bracket — a tail probability
  is nearly a power of `x`, so in the logarithm it is a straight line — and take a `p` above ½ as the tail `1 − p`, which is
  exact: the relative error against 40-digit references is below 4·10⁻¹⁴ for `p` from 10⁻³⁰⁰ to 1 − 10⁻¹⁴ (shapes 0.1 to
  100), and a quantile below the least positive double is reported as that double. `ChiSquared::quantile` no longer returns
  infinity for a `p` within 10⁻¹⁵ of 1. (The free functions `t_distribution_quantile`, `f_distribution_quantile`
  and `chi_squared_quantile` are fixed by the next entry.)
- `special::t_distribution_quantile`, `f_distribution_quantile` and `chi_squared_quantile` failed for a probability
  below about 10⁻¹⁰ (`p` of a p-value, `1 − p` of a critical value at 10⁻¹⁴): the t one was a Newton iteration on the CDF
  from the normal quantile, the other two bisected on the CDF to a width of 10⁻¹²–10⁻¹³ with a bracket that stops at 10¹⁵.
  A Cauchy `t(1)` at `p = 10⁻³⁰` came out as −1.3·10¹⁶ (it is −3.18·10²⁹), `χ²(2)` at `p = 10⁻³⁰⁰` as 6·10⁻⁶¹ (it is
  2·10⁻³⁰⁰), and a quantile beyond 10¹⁵ was returned as 10¹⁵. All three now solve the incomplete function itself, in
  the logarithm of the quantile (the gamma and beta solver above): the chi-squared quantile is
  `2 · inverse_regularized_lower_gamma(k/2, p)`, and t and F take the root `x` of the incomplete beta function together
  with `1 − x` — whichever of the two is the smaller is the one solved for, against whichever of `p` and `1 − p` is the
  smaller — so `t² = df·(1 − x)/x` and `F = d₂x/(d₁(1 − x))` keep their digits when `x` is near 0 or near 1. The relative
  error against 50-digit references (mpmath) is below 6·10⁻¹⁴ for `p` from 10⁻³⁰⁰ to 1 − 10⁻¹⁴ and a shape of 0.5 to 10³
  (4·10⁻¹⁴ at `t(0.5)`, `p = 10⁻¹⁰⁰`, where the quantile is 10¹⁹⁹), and 2·10⁻¹³ for `χ²` to 10⁶ degrees of freedom; where the
  previous code was right the new code is no worse. Past a shape of 10³ the bound is the incomplete beta function's
  own — `1 − x` is a rounded double — about `3·10⁻¹⁷ · shape`: 6·10⁻¹² for `t(10⁶)`, 7·10⁻¹⁰ for `F(3, 10⁸)` and 2·10⁻⁷
  for `t(10¹⁰)` at `p = 0.975` (1·10⁻⁷ before; for `p` from 0.1 to 0.9 it fell from 10⁻⁸–3·10⁻⁶ to 3·10⁻¹⁵). A quantile beyond the largest double is ±∞, an infinite shape is NaN (the CDFs are not defined there either). `t_distribution_quantile(0.5, df)` is
  still exactly 0, and the odd symmetry holds to the bit where `1 − p` is exact.
- `regularized_incomplete_beta` documented "relative error < 1e-10" without a shape: measured against mpmath at 60 digits the error
  of a value in the smaller tail grows with the shape, `≈ 3·10⁻¹⁶ · max(a, b)` of that tail (4·10⁻¹³ at a shape of 10³, 4·10⁻¹¹
  at 10⁶, 3·10⁻⁹ for `I(1.5, 5·10⁷)`), because the continued fraction is handed a `1 − x` rounded from a small `x` (the
  t, F and beta quantiles inherit it). The `# Accuracy` section says so, and a test holds the CDF to that line. Using the
  direct fraction a few times past the mean instead of the symmetric one, to keep the small `x` exact, does not converge
  (errors of order 10³ from 4× the mean), so the fix is a different algorithm for large shapes, not a threshold.

## [UNumflow 0.8.2] - 2026-11-09

The .NET client only; the crate is unchanged at 0.16.2. The package carries the native library built from 0.16.2, where
`UNumflow` 0.8.1 carried the one built from 0.15.0, so it takes every fix of 0.16.0, 0.16.1 and 0.16.2: the regularized
incomplete beta and gamma functions (every beta, gamma and chi-squared CDF, and the F and t p-values) keep their digits at
large shapes, the Student t, gamma and chi-squared densities too, and the Frobenius norm and the symmetric eigen solver are
right in any unit.

### Changed

- A distribution whose mean, variance, quantile or sample is beyond a double now throws `NumflowException` with `ErrorCode`
  `result_overflow` (`Parameter` names the answer, `Index` the first variate that overflows). The native answer used to carry
  `null` where a number was declared, and the client's constructor threw `InvalidOperationException` reading it.

## [0.16.2] - 2026-11-09

### Added

- A new shared refusal, `result_overflow` (`parameter`, `index`): the arguments are valid and the answer exists, but
  it is beyond the largest number a double holds. `distribution_moments`, `distribution_quantile` and
  `distribution_sample` (WebAssembly and C ABI) refuse with it where they used to send an infinity -- which JSON
  cannot carry, so the C ABI sent `null` in a field declared a number. `parameter` names the answer (`mean`,
  `variance`, `quantile`, `values`); for `values`, `index` is the first variate that overflows.

### Fixed

- `Matrix::frobenius_norm` squared every entry, so it underflowed to 0 for entries below about 1e-154 and
  overflowed to infinity above 1e154; the tolerances built on it then called a covariance in 1e100 units singular
  (`inverse` refused it) and `eigen_symmetric` of one in 1e-100 units returned zeros. The norm is now summed relative
  to the largest entry, and `eigen_symmetric` solves a copy divided by a power of two near that entry (an exact
  division) and multiplies the eigenvalues back, so both are right in any unit.
- The regularized incomplete beta and gamma functions (every beta, gamma and chi-squared CDF, and the F and
  t p-values built on them) lost their digits as a shape grew: a fixed 200-term limit gave out near a shape of
  1e4 (`P(1e4, 1e4 − 1/3)` was 0.477, not 0.5; `P(1e6, ·)` was 0.08), and `I_0.5(1e8, 1e8)` was 1.306. The term limit
  now grows with the square root of the shape, and from a shape of 1e7 the normal limit is used (Wilson-Hilferty
  for the gamma, the first Edgeworth term for the beta), so the functions are right to about 1e-9 at any shape.
  A beta CDF with both shapes near 1e300 answered infinity, and a gamma or chi-squared CDF with a shape near 1e300
  answered NaN; they answer now.
- The regularized incomplete beta function lost its digits when one shape was large and the other small -- the
  Student t and F p-values at large degrees of freedom: `I_x(1e14, 1)` was off by a third (`x^a` is the answer),
  and at a shape of 1e8 the error was 1e-7. Four causes: `ln_beta` was `ln Γ(a) + ln Γ(b) − ln Γ(a+b)`, three
  numbers of order `a ln a` whose rounding leaves the sum, of order `b ln a`, with no digit; it now subtracts
  the Stirling series of the large terms first (`ln_1p` of a ratio) and is right to 1e-13 at any size, also for
  `a = 1e300`. The prefix took `ln(1 − x)` of a rounded `1 − x` (an `x` of 5e-16 lost a tenth of itself, which a
  shape of 1e14 multiplies); and the symmetry relation `I_x(a,b) = 1 − I_{1−x}(b,a)` rounded `1 − x` the same
  way, so the function now carries `x` and `1 − x` side by side and takes logarithms of the exact one; the
  continued fraction's first term subtracted two numbers equal to the digits of `1 − x` and now uses the
  exact side. Checked against the closed forms `I_x(a, 1) = x^a` and `I_x(1, b) = 1 − (1−x)^b` for shapes
  from 1e2 to 1e14 (relative error under 1e-9).
- The densities of Student's t, the gamma and chi-squared lost their digits at a large shape the same way, by
  subtracting `ln Γ` values of order `a ln a`: the t density was off by 8 % at 1e14 degrees of freedom and the gamma
  density by 7e-4 at a shape of 1e11. The t density takes `ln Γ(h+½) − ln Γ(h)` as the difference of their Stirling
  series (and `ln_1p` for the `1 + t²/df`), and the gamma and chi-squared densities use the saddle-point form
  `−D(a, z) + ½ ln a − ½ ln 2π − c(a)` with Loader's deviance term, so every one is right to 1e-12 against mpmath at
  shapes from 1e3 to 1e14. The noncentral t CDF's `ln B` term uses the same step.
- Checked against the density integrated at 60 digits (mpmath; its own `gammainc` and `betainc` do not converge past a
  shape of about 1e7): the regularized lower gamma is right to 1e-9 (absolute) from 1e4 to 1e12, as the 0.16.1 notes
  said; the incomplete beta is right to 1e-9 below a shape of 1e7 and to 7e-9 where the normal limit takes over
  (`I_x(1e7, 1e7)`), falling as 1/(a+b) -- "about 1e-9 at any shape" was 7e-9 at its worst.
- `Matrix::eigen_symmetric` took three times as long as its operation count says at 2,000 rows and up: the Householder
  reduction ran its inner loops down the columns of a row-major matrix (a stride of `n`, all cache misses) and then
  transposed the whole transform for the QL step. It stores the transform column by column now, so the loops are
  contiguous and the transpose is gone: 2,049 rows 52.7 s to 27.4 s (1,000 rows 3.0 s to 2.9 s), the eigenvalues and
  eigenvectors identical bit for bit.

## [0.16.1] - 2026-11-08

### Changed

- `fourier::rfft` of an even length transforms a complex sequence of half the length and splits it, instead of
  transforming the real sequence as complex: about half the time, the same bins to rounding.
- `Matrix::eigen_symmetric` of a matrix with 64 or more rows reduces it to tridiagonal form by Householder
  reflections and finds the eigensystem by implicit QL, instead of cyclic Jacobi rotations (several sweeps of n³):
  800 rows take 2.2 s (about 20 s before), 2,049 rows a minute. Smaller matrices keep Jacobi, and its signs. An
  eigenvector's sign is not defined and may differ between the two; eigenvalues, orthonormality and the
  decomposition `A v = λ v` hold to the same tolerance at any scale.

### Fixed

- Distributions at the ends of their parameter range answered NaN or infinity where the value is an ordinary
  number: a triangular distribution over a range near ±1e300 (its CDF, quantiles, density and samples multiplied
  widths that overflow), the variance of a triangular distribution narrow next to its location (cancelled to 0),
  the log-normal variance when `exp(σ²)` overflows or `σ` is tiny (rounded to 0), the gamma variance for a tiny
  rate, the gamma CDF where `rate · x` overflows, the beta variance for tiny or huge shapes, and beta samples for
  shapes below 1 (both gamma draws underflowed, giving 0/0 -- now drawn in logarithms).
- The C ABI writes its answers from typed values: a non-finite number could turn into `null` while the answer was
  assembled, in a field read as a number.
- `optimize::nelder_mead` stopped by a number, not by the problem's scale: the
  tolerances were `tolerance · (1 + |value|)` and `tolerance · (1 + |coordinate|)`,
  and a coordinate that started at zero took a fixed first step of 0.00025. In
  small units the first simplex was already inside the tolerance, so a search
  declared `converged` after three evaluations at the start point (variables near
  1e-12, values near 1e-14); in large ones the value tolerance could not be met and
  the search spent its whole budget unconverged. The value is now judged against
  the larger of the best value and the largest value at the start, a coordinate
  against the larger of itself and its first step, and a zero start's step in
  proportion to the other coordinates. Where the problem is of ordinary size the
  stopping point is as before.

## [0.16.0] - 2026-10-29

### Added

- `optimize::nelder_mead` (with `NelderMeadOptions`, `Minimum`): derivative-free local
  minimisation of a function of several variables by the Nelder–Mead simplex method, with the
  dimension-adaptive coefficients of Gao & Han (2012) and relative stopping tolerances, so the
  stopping point does not depend on the units. Deterministic; a non-finite value of the
  objective refuses the point. Takes a closure, so it has no WebAssembly or C counterpart.
- `VERSION`: the crate's own version (`CARGO_PKG_VERSION`) as a constant at the
  crate root, so a consumer can report what it was built against without
  reading a lock file -- which names the wrong version once the crate is linked
  from another package.

## [0.15.0] - 2026-10-20

### Added

- Upper tails that keep their relative precision: `special::chi_squared_sf`,
  `f_distribution_sf`, `t_distribution_sf` and `regularized_upper_gamma`. A
  p-value taken as `1.0 - cdf` is the difference of two numbers near 1 and has
  no digits left below about 1e-16 (a chi-squared of 200 on 5 degrees of
  freedom is 2.8e-41); these return it to the precision of the incomplete
  beta and gamma functions (relative 1e-9 against SciPy's `sf`).

### Changed

- **Breaking:** `fourier::rfft` and the WebAssembly `rfft` return the
  `n / 2 + 1` bins `0..=n/2`, as NumPy's, SciPy's and FFTW's real transforms
  do, instead of all `n` (the upper half only mirrored the lower:
  `X[n−k] = conj(X[k])`). A caller that needs all `n` bins takes `fft` of the
  values as complex; one that read bins `0..=n/2` is unaffected. The WebAssembly
  array is `2 * (n / 2 + 1)` numbers long.

### Fixed

- `Matrix::eigen_symmetric` does not depend on the scale of the matrix. Its
  convergence test and its skip of tiny off-diagonal entries compared against
  fixed numbers (1e-15), so a matrix whose entries were all smaller than that,
  such as the covariance of data measured in units near 1e-9, was never rotated
  and its diagonal came back as the eigenvalues and the identity as the
  eigenvectors. The tolerance is now 1e-15 times the matrix's Frobenius norm;
  the zero matrix returns zero eigenvalues and the identity.

## [0.14.1] - 2026-10-18

The .NET client is `UNumflow` 0.8.0. The Rust API is unchanged: what is new are
binding entry points to functions the crate already had.

### Added

- Sample median and quantiles reach every binding: WebAssembly `median(data)`
  and `quantiles(data, probabilities)` (R-7, in the order asked, one sort),
  the C ABI `unumflow_median` / `unumflow_quantiles`, and .NET
  `Stats.Median`, `Stats.Quantile` and `Stats.Quantiles`. No values is
  `empty_input`; a probability outside `[0, 1]` is `parameter_out_of_range`
  at `probabilities[i]` with `min` 0 and `max` 1.

## [0.14.0] - 2026-10-16

The .NET client is `UNumflow` 0.7.0.

### Added

- `units` — units of measure: a `Unit` is a dimension (integer exponents of the
  seven SI base quantities) with `value_in_si = value · scale + offset`.
  `Units::si()` reads UCUM-syntax expressions (`kW.h`, `m/s2`, `J/(kg.K)`,
  prefixes, exponents, parentheses, integer factors, `{…}` annotations) over
  the SI base units, the 22 named derived units, `min` `h` `d` `L` `t` and the
  decimal prefixes; `define` adds a unit as a multiple of an expression (with
  an offset for a scale). `convert` converts a reading (20 °C → 293.15 K),
  `convert_difference` an amount of change (20 °C → 20 K); units of different
  dimensions are refused. No unit table beyond the SI is embedded.
- WebAssembly `unit_convert`, `unit_convert_difference`, `unit_info`
  (`UnitDefinitionSpec`, `UnitInfo`); C ABI `unumflow_unit_convert`,
  `unumflow_unit_convert_difference`, `unumflow_unit_info`; .NET `Units`,
  `UnitDefinition`, `UnitInfo`. New refusal codes: `unknown_unit`,
  `dimension_mismatch`, `invalid_option`, `duplicate_id`.

## [0.13.1] - 2026-10-16

The .NET client is unchanged (`UNumflow` 0.6.0).

### Fixed

- A result the package fails to hand back to JavaScript is refused as
  `internal`, not as `malformed_input` with `parameter: "result"` — the fault
  is the package's, not the caller's input.

## [0.13.0] - 2026-10-16

The .NET client is `UNumflow` 0.6.0.

### Added

- `codes()` (WebAssembly): every refusal `code` the package throws, with the
  fields that can come with it -- `{ refusals: [{ code, fields }], violations:
  [] }`, the shape every engine package shares -- built from the table the
  refusals are built from, so a caller that translates codes can check its list
  against the package's. The same catalog over the C ABI (`unumflow_codes()`)
  and in .NET (`Numflow.Codes` → `Codes(Refusals, Violations)`).

### Fixed

- The README lists `invalid_inverse` and `internal`, which the package throws;
  a test keeps the table and `codes()` equal.

## [UNumflow 0.5.0] - 2026-10-14

The .NET client only; the crate is unchanged at 0.12.0.

### Changed

- **Breaking (.NET `UNumflow`):** `NumflowException` names its members the way the other
  .NET bindings' exceptions do — `Status` is the native status (was `Code`),
  `ErrorCode` the error body's `code` (was `Reason`), and `Parameter` / `Index`
  now say which value the refusal is about without reading `Details`. Migrate: `ex.Code` → `ex.Status`, `ex.Reason` → `ex.ErrorCode`.
- **Breaking (.NET `UNumflow`):** `Distribution.Kind` is the `DistributionKind`
  enum (`Weibull`, `ChiSquared`, …) instead of the engine's name as a string.

## [0.12.0] - 2026-10-13

### Added

- `src/ffi/request_keys.json` lists the keys each C entry point reads, the
  parameters of every distribution `kind` included. The crate's tests measure it from
  the entry points themselves -- a request with a key no request has, and a value with
  a name no option has, at every level -- and fail when the file and the entry points
  differ.
- .NET: a test project checks every request the client writes against that list in
  both directions, and runs every call with every setting against the native library,
  with reflection-based JSON disabled. CI runs it.

### Changed

- **Breaking:** each C entry point's request has exactly the keys that call reads. A
  `p` sent to `unumflow_distribution_cdf`, an `x` sent to
  `unumflow_distribution_quantile`, and a `t` sent to `unumflow_interval_measure` or
  `unumflow_interval_normalize` are refused as unknown keys (status -2,
  `malformed_input`); they were accepted and ignored.
- **Breaking:** an unknown `distribution.kind` is refused with `expected` as a list of
  names, as every other unknown option is; it was one comma-separated string. Over
  WebAssembly the property is an array.
- A `distribution` without a `kind` string is refused with `parameter:
  "distribution.kind"` (it was `"distribution"`), over both interfaces.

## [0.11.0] - 2026-10-13

The .NET client is `UNumflow` 0.3.0.

### Changed

- **Breaking:** `parameter_out_of_range` refusals (WebAssembly `Error` properties and
  the C error body) report open bounds as `exclusive_min` / `exclusive_max`. `min` and
  `max` now carry only inclusive bounds and stay present, `null` on an open or
  unbounded end; the `exclusive_*` keys appear only when that end is open. A
  distribution parameter that must be `> 0` and a degrees of freedom `> 0` report
  `min: null, exclusive_min: 0, max: null` (previously `min: 0`); a probability
  outside (0, 1) reports `exclusive_min: 0, exclusive_max: 1` with `min` and `max`
  `null` (previously `min: 0, max: 1`). Closed ranges (`n`, `seed`) are unchanged.
- **Breaking:** `distribution_sample` (WebAssembly and C ABI) tells a count that is not
  a whole number apart from one out of range. `n` or `seed` with a fraction (`2.5`) is
  `malformed_input` with `parameter` ("must be a whole number"), previously
  `parameter_out_of_range`; NaN or ±∞ is `value_not_finite`, previously
  `parameter_out_of_range`. A whole number outside the range, a negative one included,
  stays `parameter_out_of_range` with `min` 0, `max` and `got`.
- **Breaking:** a distribution parameter outside the range the others leave it is
  `parameter_out_of_range` (WebAssembly and C ABI), previously `invalid_option` with no
  bounds: `max` not above `min` reports `parameter: "distribution.max"`,
  `exclusive_min` = `min` and `got`; `mode` outside `[min, max]` reports
  `parameter: "distribution.mode"`, `min`, `max` and `got`.
  `DistributionError::Unordered` carries those bounds (`low`, `high`) and the value
  given (`got`), and its message ends with the value.
- **Breaking:** `estimate_lambda` (WebAssembly) refuses `lambda_min` not below
  `lambda_max` as `parameter_out_of_range` with `parameter: "lambda_min"`,
  `exclusive_max` = `lambda_max` and `got`, and a NaN or infinite end as
  `value_not_finite` naming it. Both were `invalid_lambda_range`, whose `min` and `max`
  held the values given rather than bounds.

## [0.10.0] - 2026-10-07

The .NET client is `UNumflow` 0.2.0.

### Added

- `UNumflow` carries the native library for `linux-arm64` (glibc 2.39 or later).

### Changed

- **Breaking:** `TransformError` is `#[non_exhaustive]` and its input variants carry
  where the input went wrong: `NonPositiveData { index, value }` (the first value ≤ 0),
  `NonFiniteData { index }`, `InsufficientData { min, got }` and
  `InvalidLambdaRange { min, max }`. The WebAssembly refusals report them as fields
  (`parameter`, `index`, `got`, `min`, `max`) instead of only in the message.
- `UNumflow` is marked `IsAotCompatible`: the trimming and NativeAOT analyzers run on
  every build and any finding fails it, so the package stays usable in trimmed and
  NativeAOT applications.

## [0.9.0] - 2026-10-07

The .NET client is `UNumflow` 0.1.0 — the first release of this crate's C# binding.

### Changed

- **Breaking:** `DistributionError` names the parameter and the rule it
  breaks instead of carrying one sentence: `NotFinite { parameter }`,
  `NotPositive { parameter, got }`, and `Unordered { parameter, relation }`
  (`min < max`, `min ≤ mode ≤ max`). `InvalidParameters(String)` is gone.

### Added

- `distributions::Sample` — `sample(&mut rng)` and reproducible
  `sample_n(n, seed)` for Uniform, Triangular, PERT, Normal, LogNormal,
  Weibull, Exponential, Gamma, Beta and χ². Inversion where the quantile is
  direct (a larger uniform always gives a larger variate, so runs on one seed
  are coupled); Marsaglia–Tsang (2000) for the gamma family, whose quantile is a
  Newton iteration. Uniforms come from the open interval (0, 1) — 52 bits
  offset by half a step, since the 53-bit top value `1 − 2⁻⁵⁴` rounds to 1 — so
  no variate is infinite.
- WebAssembly: `distribution_cdf`, `distribution_quantile` and
  `distribution_sample(distribution, n, seed)` over a `{ kind, ...parameters }`
  specification (declared `DistributionSpec`). An unknown `kind` throws
  `unknown_option` with `expected`; a bad parameter throws with its path
  (`distribution.sigma`).
- C ABI behind the `ffi` feature (JSON in, JSON out): distribution moments,
  CDF, quantile and seeded sampling over the same `{ kind, ...parameters }`
  specification, and the interval-set operations. A refusal is a non-zero
  status with `{"error", "code", ...fields}` — the same `code` and fields as
  the WebAssembly binding's.
- WebAssembly: `distribution_moments(distribution)` → `{ mean, variance }`.
- `collections::IntervalSet<T>` — a finite union of half-open intervals
  `[start, end)`, always kept sorted, disjoint and with touching pieces merged:
  `from_intervals`, `insert`, `union`, `intersection`, `difference`, `clip`,
  `measure`, `contains`, iteration. Because the pieces never overlap, `measure`
  counts every point once however much the input overlapped. Works over `f32`,
  `f64` and every primitive integer; the measure of an integer set is the
  unsigned type of the same width (as `i64::abs_diff`), so it cannot overflow.
  A reversed interval (`start > end`) or a NaN/infinite bound is refused as
  `IntervalError` with its position — never swapped or dropped.
- WebAssembly: `interval_normalize`, `interval_measure`, `interval_union`,
  `interval_intersection` and `interval_difference` over `[number, number][]`.
  A row that is not two numbers throws `malformed_input` at its path
  (`a[2]`), a reversed row `reversed_interval` with `parameter` and `index`.

### Fixed

- README: the dependency list named `rand` 0.9 (the crate uses 0.10), and the
  module table listed five of the ten distributions.

## [0.8.0] - 2026-10-04

### Changed

- **Breaking:** the WebAssembly functions that take a `data` array (`mean`,
  `std_dev`, `variance`, `box_cox`, `estimate_lambda`, `rfft`) read it as sent.
  Before, the generated glue copied a plain array into a typed array, so `null`
  became `0` and a string `NaN` — `mean([1, null, 3])` returned `1.333…`. An
  element that is not a number now throws `malformed_input` and a NaN or
  ±Infinity `value_not_finite`, both with `parameter` and `index`. `data` is
  declared `number[] | Float64Array`; anything else throws `malformed_input`.
- **Breaking:** `mean`, `std_dev` and `variance` throw instead of returning
  `NaN`: `mean` of no values is `empty_input`, `std_dev`/`variance` of fewer
  than two values `insufficient_data` (with `min` and `got`).

## [0.7.0] - 2026-09-30

### Changed

- **Breaking:** the WebAssembly functions throw an `Error` carrying a stable
  `code` and the values behind it instead of a bare string: a `p` outside
  (0, 1) or a non-positive degrees of freedom is `parameter_out_of_range` with
  `parameter`, `min`, `max` (or `null`), `got`; a NaN argument is
  `value_not_finite`; the Box-Cox refusals are `non_positive_data`,
  `value_not_finite`, `insufficient_data`, `invalid_transform` and
  `invalid_lambda_range`. `err.message` reads as before, but `String(err)` now
  starts with `Error: `.

- The README says a browser without a bundler is not supported (the package
  loads its `.wasm` through an ES module import, which browsers refuse), instead
  of listing only the environments that work.

### Removed

- **Breaking:** `Pert::quantile_approx`, deprecated in 0.6.3 — use
  `Pert::quantile`, which is exact.

## [0.6.4] - 2026-09-30

### Changed

- The publishing workflow now also fails if an exported function takes a
  parameter typed `any` (`check-typed-dts.sh`). None does; the README
  said inputs were still `any`, which has not been true of this binding.
- The publishing workflow runs the README's JavaScript examples against the
  built package before it publishes, so an example that throws is caught
  before a reader copies it.

### Fixed

- The README's Rust example did not compile: it used an `OnlineStats` type, a
  `PertDistribution` with a `sample` method and an `Rng::seed_from_u64` that
  the crate does not have. It now uses `WelfordAccumulator`, `Pert` and
  `random::create_rng`.
  The README's Rust examples are now compiled and run with the doc-tests,
  so an example that stops matching the API fails CI.

## [0.6.3] - 2026-09-25

### Added

- `special::chi_squared_quantile(p, k)` -- the inverse of `chi_squared_cdf`, by
  bisection on the incomplete gamma function. The library had the t and F
  quantiles but not this one, so a chi-squared critical value had to be
  approximated by the caller; the Wilson-Hilferty cube-root approximation is
  off by 2-3 % in the tails at one or two degrees of freedom.
- `special::noncentral_t_cdf(t, df, delta)` -- the noncentral t CDF (Lenth
  1989, AS 243). It is the distribution a t statistic follows when the
  hypothesis is false, so it is what the power of a t test, or of an effect
  test in a two-level factorial, is computed from.
- `distributions::Pert::quantile(p)` -- the exact PERT quantile, a Beta
  quantile stretched over `[min, max]`. The only quantile was
  `quantile_approx`, a normal approximation that is off for any skewed
  estimate: for min 0, mode 1, max 10 it put the 5th percentile at -0.30
  (below the minimum, then clamped to 0) where the exact value is 0.30.

- WASM: `inverse_normal_cdf`, `t_distribution_cdf`, `t_distribution_quantile`,
  `f_distribution_cdf`, `f_distribution_quantile`, `chi_squared_cdf`,
  `chi_squared_quantile`. The crate already computed these; a TypeScript
  consumer that needed a critical value had to re-implement the inverse. They
  throw, naming the argument, for a `p` outside (0, 1) or a non-positive
  degrees of freedom -- the crate functions return `NaN` there, which across
  the JS boundary draws nothing and raises nothing.

### Deprecated

- `Pert::quantile_approx` -- use `Pert::quantile`. It stays in this release so
  the release is a patch, and is removed in the next minor version.

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
