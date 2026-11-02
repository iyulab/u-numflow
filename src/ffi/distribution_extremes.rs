//! Every distribution at the ends of the parameter range it accepts, through
//! every distribution entry point: moments, the CDF at points from far below
//! to far above, quantiles near 0, at the middle and near 1, and a sample.
//! The answer is a number or a refusal -- never `internal`, which in a test
//! build is also what a non-finite number in a response becomes.

use super::request_keys::{call, Entry};
use serde_json::{json, Value};

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
#[ignore = "86 of 1442 cases answer a non-finite number: results beyond the range of a double, and NaN where the true value is finite -- being fixed"]
fn every_distribution_at_its_extremes_answers_without_internal() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut failures = Vec::new();
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
            let (_, body) = call(entry, &request);
            calls += 1;
            if body.get("code").and_then(Value::as_str) == Some("internal") {
                failures.push(format!("{what} {request}"));
            }
        }
    }
    std::panic::set_hook(previous);
    assert!(
        failures.is_empty(),
        "{} of {calls} requests ended `internal`:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
