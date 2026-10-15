# UNumflow

.NET bindings for the u-numflow numerical library.

## Features

- **Distributions**: Uniform, Triangular, PERT, Normal, LogNormal, Weibull,
  Exponential, Gamma, Beta and χ² — `Cdf`, `Quantile`, `Mean`, `Variance`, and
  `Sample(n, seed)`: the same seed always gives the same values, and the
  uniforms behind them come from the open interval (0, 1), so no variate is
  infinite
- **Interval sets**: unions of half-open intervals `[start, end)` with
  `Union`, `Intersection`, `Difference`, `Clip`, `Contains` and a `Measure`
  that counts every point once, however much the input overlapped
- **Parameters are checked when you create a distribution** — a
  `Distribution` that exists is a valid one

## Installation

```bash
dotnet add package UNumflow
```

## Usage

```csharp
using UNumflow;

// Critical values and probabilities
var chi = Distribution.ChiSquared(4);
double critical = chi.Quantile(0.95);       // 9.4877…
double p = chi.Cdf(critical);               // 0.95

// Reproducible samples
double[] lives = Distribution.Weibull(shape: 2, scale: 100).Sample(1000, seed: 42);

// Coverage arithmetic: overlapping stops are down-time once
var planned = IntervalSet.From([(0, 24)]);
var unplanned = IntervalSet.From([(10, 30)]);
var week = IntervalSet.From([(0, 168)]);
double upHours = week.Difference(planned.Union(unplanned)).Measure; // 138
```

## Errors

A refused request raises `NumflowException`. `Message` is readable text;
`ErrorCode` is a stable code to branch on (the error body's `code`), `Parameter`
and `Index` say which value it is about, `Details` is the error body with the
values behind it, and `Status` is the native status:

```csharp
try
{
    Distribution.Normal(0, sigma: -1);
}
catch (NumflowException e)
{
    // e.ErrorCode == "parameter_out_of_range"
    // e.Details: { "parameter": "distribution.sigma", "min": null, "exclusive_min": 0, "max": null, "got": -1, ... }
}
```

| `ErrorCode` | Details | Meaning |
|---|---|---|
| `parameter_out_of_range` | `parameter`, `min`, `max`, `got`; `exclusive_min` / `exclusive_max` on an open end | A distribution parameter that must be `> 0` (`exclusive_min` 0); `distribution.max` not above `min` (`exclusive_min` = `min`) or `distribution.mode` outside `[min, max]` (`min`, `max`); `p` not strictly inside (0, 1) (`exclusive_min` 0, `exclusive_max` 1); a `seed` outside `[0, 2⁵³]` (`min`, `max`) |
| `value_not_finite` | `parameter`, `index` | A NaN or infinite parameter or interval bound |
| `reversed_interval` | `parameter`, `index` | An interval with start > end — refused, not swapped |
| `unknown_option` | `parameter`, `got`, `expected` | A distribution kind the engine does not have |
| `malformed_input` | `parameter` | A request the engine could not read |
| `internal` | — | An engine fault |

`Numflow.Codes` returns the engine's codes as data — `Codes(Refusals, Violations)`, each
refusal with the fields beside it — from the table the engine refuses from (it is the
engine's whole list, so it also names reasons only the WebAssembly package's functions
raise).

In `parameter_out_of_range`, `min` and `max` are inclusive bounds and are always
present — `null` when that end is unbounded or open. An open (exclusive) end is
`exclusive_min` / `exclusive_max`, present only when the bound is open.

## Platforms

The package carries the native library for `win-x64`, `linux-x64` and `linux-arm64`
(glibc 2.39 or later), `osx-x64` and `osx-arm64`; no separate install is needed.

## License

MIT
