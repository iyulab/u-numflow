//! Every distribution at the ends of the parameter range it accepts, through
//! every distribution entry point: moments, the CDF at points from far below
//! to far above, quantiles near 0, at the middle and near 1, and a sample.
//! The answer is a number or a refusal -- never `internal`, which in a test
//! build is also what a non-finite number in a response becomes.
//!
//! A result beyond the range of a double (the variance of a uniform on ±1e300,
//! a high quantile of a Weibull of shape 1e-8) has no number to be sent as: it
//! is refused as `result_overflow`, naming the answer, and an infinity that
//! reaches a response fails. A NaN fails too.

use super::degenerate_inputs::{last_panic, quiet_panics};
use super::request_keys::{call, Entry};
use serde_json::{json, Value};

/// The requests that still answer NaN: none.
const KNOWN_NAN: [&str; 0] = [];

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
    let mut refused = 0;
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
            if body.get("code").and_then(Value::as_str) == Some("result_overflow") {
                refused += 1;
                if !["mean", "variance", "quantile", "values"]
                    .contains(&body["parameter"].as_str().unwrap_or(""))
                {
                    other.push(format!("{what} {request}: {body}"));
                }
                continue;
            }
            if body.get("code").and_then(Value::as_str) != Some("internal") {
                continue;
            }
            let why = last_panic();
            let at = format!("{what} {request}");
            if why.contains("NaN at") {
                nan.push(at);
            } else if why.contains("inf at") {
                overflow += 1;
                eprintln!("inf: {at}: {why}");
            } else {
                other.push(format!("{at}: {why}"));
            }
        }
    }
    eprintln!("sweep: {calls} calls, {refused} refused as result_overflow, {overflow} infinite, {} NaN, {} other", nan.len(), other.len());
    for n in nan.iter().take(8) {
        eprintln!("nan: {n}");
    }
    for o in other.iter().take(8) {
        eprintln!("other: {o}");
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
    eprintln!(
        "sweep: {calls} calls, {refused} refused as result_overflow, {overflow} infinite, {} NaN",
        nan.len()
    );
    let mut known: Vec<String> = KNOWN_NAN.iter().map(|s| s.to_string()).collect();
    known.sort();
    nan.sort();
    assert_eq!(nan, known, "requests answering NaN (of {calls})");
    // Results beyond a double are refused, not sent.
    assert_eq!(
        overflow, 0,
        "{overflow} infinite results reached a response"
    );
    assert!(refused > 0, "the sweep reaches no result beyond a double");
}

#[test]
fn a_result_beyond_a_double_is_refused_with_the_answer_it_names() {
    let wide = json!({ "kind": "uniform", "min": -1e300, "max": 1e300 });
    let (_, body) = call(
        super::unumflow_distribution_moments,
        &json!({ "distribution": wide }),
    );
    assert_eq!(body["code"], "result_overflow", "{body}");
    assert_eq!(body["parameter"], "variance");

    // A lognormal with a huge location: the variates overflow at some index.
    let heavy = json!({ "kind": "lognormal", "mu": 700.0, "sigma": 30.0 });
    let (_, body) = call(
        super::unumflow_distribution_sample,
        &json!({ "distribution": heavy, "n": 5, "seed": 3 }),
    );
    assert_eq!(body["code"], "result_overflow", "{body}");
    assert_eq!(body["parameter"], "values");
    assert!(body["index"].is_number());

    // The same distribution at an ordinary location answers.
    let (code, body) = call(
        super::unumflow_distribution_moments,
        &json!({ "distribution": { "kind": "uniform", "min": -1.0, "max": 1.0 } }),
    );
    assert_eq!(code, 0, "{body}");
}
