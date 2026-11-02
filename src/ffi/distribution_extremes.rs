//! Every distribution at the ends of the parameter range it accepts, through
//! every distribution entry point: moments, the CDF at points from far below
//! to far above, quantiles near 0, at the middle and near 1, and a sample.
//! The answer is a number or a refusal -- never `internal`, which in a test
//! build is also what a non-finite number in a response becomes.
//!
//! Two kinds are still tracked rather than closed. A result beyond the range
//! of a double (the variance of a uniform on ±1e300, a high quantile of a
//! Weibull of shape 1e-8) has no number to be sent as; what the wire says
//! for one is an open decision, so an infinity is counted, not failed. A
//! regularized incomplete gamma at a shape parameter near 1e300 has
//! no asymptotic expansion here yet and answers NaN: those three requests are
//! listed. Any other NaN fails.

use super::degenerate_inputs::{last_panic, quiet_panics};
use super::request_keys::{call, Entry};
use serde_json::{json, Value};

/// The requests that still answer NaN: the incomplete gamma function at a
/// shape parameter near 1e300.
const KNOWN_NAN: [&str; 3] = [
    r#"cdf {"distribution":{"k":1e+300,"kind":"chi_squared"},"x":1e+300}"#,
    r#"cdf {"distribution":{"kind":"gamma","rate":1.0,"shape":1e+300},"x":1e+300}"#,
    r#"cdf {"distribution":{"kind":"gamma","rate":1e+300,"shape":1e+300},"x":1.0}"#,
];

fn specs() -> Vec<Value> {
    let tiny = 1e-300;
    let small = 1e-8;
    let large = 1e8;
    let huge = 1e300;
    let mut out = Vec::new();
    for (a, b) in [
        (0.0, small),
        (-huge, huge),
        (-large, large),
        (tiny, 2.0 * tiny),
    ] {
        out.push(json!({ "kind": "uniform", "min": a, "max": b }));
        out.push(json!({ "kind": "triangular", "min": a, "mode": a, "max": b }));
        out.push(json!({ "kind": "pert", "min": a, "mode": b, "max": b }));
    }
    for s in [tiny, small, large, huge] {
        out.push(json!({ "kind": "normal", "mu": 0, "sigma": s }));
        out.push(json!({ "kind": "normal", "mu": huge, "sigma": s }));
        out.push(json!({ "kind": "exponential", "rate": s }));
        out.push(json!({ "kind": "chi_squared", "k": s }));
        for t in [tiny, small, 1.0, large, huge] {
            out.push(json!({ "kind": "weibull", "shape": s, "scale": t }));
            out.push(json!({ "kind": "gamma", "shape": s, "rate": t }));
            out.push(json!({ "kind": "beta", "alpha": s, "beta": t }));
        }
    }
    for sigma in [small, 1.0, 10.0, 30.0, 300.0] {
        for mu in [-700.0, 0.0, 700.0] {
            out.push(json!({ "kind": "lognormal", "mu": mu, "sigma": sigma }));
        }
    }
    out
}

#[test]
fn every_distribution_at_its_extremes_answers_without_nan() {
    quiet_panics();
    let mut nan = Vec::new();
    let mut other = Vec::new();
    let mut overflow = 0;
    let mut calls = 0;
    for d in specs() {
        let mut requests: Vec<(&str, Entry, Value)> = vec![(
            "moments",
            super::unumflow_distribution_moments,
            json!({ "distribution": d }),
        )];
        for x in [-1e300, -1.0, 0.0, 1e-300, 0.5, 1.0, 1e8, 1e300] {
            requests.push((
                "cdf",
                super::unumflow_distribution_cdf,
                json!({ "distribution": d, "x": x }),
            ));
        }
        for p in [1e-300, 1e-12, 0.5, 1.0 - 1e-12] {
            requests.push((
                "quantile",
                super::unumflow_distribution_quantile,
                json!({ "distribution": d, "p": p }),
            ));
        }
        requests.push((
            "sample",
            super::unumflow_distribution_sample,
            json!({ "distribution": d, "n": 5, "seed": 3 }),
        ));
        for (what, entry, request) in requests {
            last_panic();
            let (_, body) = call(entry, &request);
            calls += 1;
            if body.get("code").and_then(Value::as_str) != Some("internal") {
                continue;
            }
            let why = last_panic();
            let at = format!("{what} {request}");
            if why.contains("NaN at") {
                nan.push(at);
            } else if why.contains("inf at") {
                overflow += 1;
            } else {
                other.push(format!("{at}: {why}"));
            }
        }
    }
    assert!(
        other.is_empty(),
        "panics:
{}",
        other.join(
            "
"
        )
    );
    let mut known: Vec<String> = KNOWN_NAN.iter().map(|s| s.to_string()).collect();
    known.sort();
    nan.sort();
    assert_eq!(nan, known, "requests answering NaN (of {calls})");
    // Results beyond a double: counted until the wire has a way to say so.
    assert!(overflow < 70, "{overflow} infinite results");
}
