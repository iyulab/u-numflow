//! C ABI for u-numflow — JSON in, JSON out.
//!
//! Every entry point takes one NUL-terminated JSON request and writes one JSON
//! body to `*result_ptr`, which the caller releases with
//! [`unumflow_free_string`]. The returned status is:
//!
//! | Status | Meaning |
//! |---|---|
//! | `0` | success — the body is the result |
//! | `-1` | a null pointer argument (no body) |
//! | `-2` | the request is not JSON of the expected shape |
//! | `-3` | the request was refused (a parameter out of range, an unknown distribution, a reversed interval, …) |
//! | `-4` | internal panic |
//!
//! A non-zero status other than `-1` comes with an error body
//! `{"error": "<readable text>", "code": "<stable reason>", ...}`; `code` and
//! the fields beside it (`parameter`, `index`, `min`, `max`, `got`,
//! `expected`) are the same the WebAssembly binding puts on its thrown
//! `Error` — both are built by `crate::wire`.
//!
//! Requests and results:
//!
//! | Function | Request | Result |
//! |---|---|---|
//! | `unumflow_distribution_moments` | `{distribution}` | `{"mean", "variance"}` |
//! | `unumflow_distribution_cdf` | `{distribution, x}` | `{"value"}` |
//! | `unumflow_distribution_quantile` | `{distribution, p}` | `{"value"}` |
//! | `unumflow_distribution_sample` | `{distribution, n, seed}` | `{"values": [...]}` |
//! | `unumflow_interval_normalize` | `{intervals}` | `{"intervals": [[s, e], ...]}` |
//! | `unumflow_interval_measure` | `{intervals}` | `{"value"}` |
//! | `unumflow_interval_contains` | `{intervals, t}` | `{"value": bool}` |
//! | `unumflow_interval_union` / `_intersection` / `_difference` | `{a, b}` | `{"intervals": ...}` |
//!
//! `distribution` is `{"kind": "weibull", "shape": 2, "scale": 10}` — the
//! same specification as the WebAssembly `DistributionSpec`.

use std::ffi::{CStr, CString};
use std::panic;

use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::collections::IntervalSet;
use crate::wire::{
    check_finite, check_kind, check_probability, distribution_refusal, interval_set_from, whole,
    Distribution, DistributionSpec, Field, Refusal,
};

const ERR_NULL: i32 = -1;
const ERR_PARSE: i32 = -2;
const ERR_REFUSED: i32 = -3;
const ERR_PANIC: i32 = -4;

// ── Transport ─────────────────────────────────────────────────────────────

/// A request that could not be answered: the status and the refusal to write.
struct Failure(i32, Refusal);

fn malformed(parameter: &'static str, message: String) -> Failure {
    Failure(
        ERR_PARSE,
        Refusal {
            code: "malformed_input",
            message,
            fields: vec![("parameter", Field::Str(parameter))],
        },
    )
}

fn refused(r: Refusal) -> Failure {
    Failure(ERR_REFUSED, r)
}

fn field_json(f: Field) -> Value {
    match f {
        Field::Num(n) => json!(n),
        Field::Str(s) => json!(s),
        Field::Path(s) => json!(s),
        Field::Null => Value::Null,
    }
}

/// Writes `value` as JSON to `*result_ptr`; `-1` if `result_ptr` is null.
fn write_json(result_ptr: *mut *mut libc::c_char, value: &Value) -> i32 {
    if result_ptr.is_null() {
        return ERR_NULL;
    }
    // serde_json never writes an interior NUL: it escapes U+0000 as \u0000.
    let text = CString::new(value.to_string()).expect("JSON text has no interior NUL");
    unsafe { *result_ptr = text.into_raw() };
    0
}

/// Runs `body` on the request text, writing its result or its refusal.
fn handle(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
    body: impl FnOnce(&str) -> Result<Value, Failure> + panic::UnwindSafe,
) -> i32 {
    if !result_ptr.is_null() {
        unsafe { *result_ptr = std::ptr::null_mut() };
    }
    if request_json.is_null() || result_ptr.is_null() {
        return ERR_NULL;
    }
    let outcome = panic::catch_unwind(|| {
        let text = unsafe { CStr::from_ptr(request_json) }
            .to_str()
            .map_err(|_| malformed("request", "request: not valid UTF-8".to_string()))?;
        body(text)
    });
    let (status, value) = match outcome {
        Ok(Ok(value)) => (0, value),
        Ok(Err(Failure(status, r))) => (status, error_body(r)),
        Err(_) => (
            ERR_PANIC,
            json!({ "error": "internal panic", "code": "internal" }),
        ),
    };
    match write_json(result_ptr, &value) {
        0 => status,
        write_failure => write_failure,
    }
}

fn error_body(r: Refusal) -> Value {
    let mut body = serde_json::Map::new();
    body.insert("error".into(), json!(r.message));
    body.insert("code".into(), json!(r.code));
    for (k, v) in r.fields {
        body.insert(k.into(), field_json(v));
    }
    Value::Object(body)
}

/// Parses the request into `T`, refusing unknown or missing fields.
fn parse<T: DeserializeOwned>(text: &str) -> Result<T, Failure> {
    serde_json::from_str(text).map_err(|e| malformed("request", format!("request: {e}")))
}

// ── Requests ──────────────────────────────────────────────────────────────

/// Reads `distribution` (checking `kind` before serde, so an unknown name is
/// `unknown_option` with the list) and builds it.
fn distribution(value: Value) -> Result<Distribution, Failure> {
    let kind = value.get("kind").and_then(Value::as_str).ok_or_else(|| {
        malformed(
            "distribution",
            "distribution.kind: expected a string naming the distribution".to_string(),
        )
    })?;
    check_kind(kind).map_err(refused)?;
    let spec: DistributionSpec = serde_json::from_value(value)
        .map_err(|e| malformed("distribution", format!("distribution: {e}")))?;
    spec.build().map_err(|e| refused(distribution_refusal(e)))
}

fn intervals(parameter: &'static str, rows: Vec<Vec<f64>>) -> Result<IntervalSet<f64>, Failure> {
    interval_set_from(parameter, rows).map_err(refused)
}

fn pieces(set: &IntervalSet<f64>) -> Value {
    json!({ "intervals": set.iter().map(|(s, e)| [s, e]).collect::<Vec<_>>() })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DistOnly {
    distribution: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DistAt {
    distribution: Value,
    #[serde(default)]
    x: Option<f64>,
    #[serde(default)]
    p: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DistSample {
    distribution: Value,
    n: f64,
    seed: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct One {
    intervals: Vec<Vec<f64>>,
    #[serde(default)]
    t: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Two {
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
}

fn required(name: &'static str, v: Option<f64>) -> Result<f64, Failure> {
    v.ok_or_else(|| malformed(name, format!("request: missing field `{name}`")))
}

// ── Exports ───────────────────────────────────────────────────────────────

macro_rules! export {
    ($(#[$doc:meta])* $name:ident, |$text:ident| $body:expr) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// `request_json` must be null or point to a NUL-terminated string, and
        /// `result_ptr` must be null or valid for writing one pointer. A string
        /// written there is owned by the caller and must be released with
        /// [`unumflow_free_string`].
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            request_json: *const libc::c_char,
            result_ptr: *mut *mut libc::c_char,
        ) -> i32 {
            handle(request_json, result_ptr, |$text: &str| $body)
        }
    };
}

export!(
    /// Mean and variance of `distribution` — also the way to validate one.
    unumflow_distribution_moments,
    |text| {
        let r: DistOnly = parse(text)?;
        let d = distribution(r.distribution)?;
        Ok(json!({ "mean": d.mean(), "variance": d.variance() }))
    }
);

export!(
    /// `P(X ≤ x)`.
    unumflow_distribution_cdf,
    |text| {
        let r: DistAt = parse(text)?;
        let d = distribution(r.distribution)?;
        let x = required("x", r.x)?;
        check_finite("x", x).map_err(refused)?;
        Ok(json!({ "value": d.cdf(x) }))
    }
);

export!(
    /// The `x` with `P(X ≤ x) = p`, for `p` strictly inside (0, 1).
    unumflow_distribution_quantile,
    |text| {
        let r: DistAt = parse(text)?;
        let d = distribution(r.distribution)?;
        let p = required("p", r.p)?;
        check_probability(p).map_err(refused)?;
        Ok(json!({ "value": d.quantile(p) }))
    }
);

export!(
    /// `n` variates, reproducible from `seed` (a whole number in `[0, 2⁵³]`).
    unumflow_distribution_sample,
    |text| {
        let r: DistSample = parse(text)?;
        let d = distribution(r.distribution)?;
        let n = whole("n", r.n, u32::MAX as f64).map_err(refused)? as usize;
        let seed = whole("seed", r.seed, 9_007_199_254_740_992.0).map_err(refused)? as u64;
        Ok(json!({ "values": d.sample_n(n, seed) }))
    }
);

export!(
    /// The union of `intervals` in normal form.
    unumflow_interval_normalize,
    |text| {
        let r: One = parse(text)?;
        Ok(pieces(&intervals("intervals", r.intervals)?))
    }
);

export!(
    /// Total length of the union of `intervals`.
    unumflow_interval_measure,
    |text| {
        let r: One = parse(text)?;
        Ok(json!({ "value": intervals("intervals", r.intervals)?.measure() }))
    }
);

export!(
    /// Whether `t` lies in the union of `intervals` (half-open).
    unumflow_interval_contains,
    |text| {
        let r: One = parse(text)?;
        let t = required("t", r.t)?;
        Ok(json!({ "value": intervals("intervals", r.intervals)?.contains(t) }))
    }
);

export!(
    /// Points in `a` or `b`.
    unumflow_interval_union,
    |text| {
        let r: Two = parse(text)?;
        Ok(pieces(&intervals("a", r.a)?.union(&intervals("b", r.b)?)))
    }
);

export!(
    /// Points in both `a` and `b`.
    unumflow_interval_intersection,
    |text| {
        let r: Two = parse(text)?;
        Ok(pieces(&intervals("a", r.a)?.intersection(&intervals("b", r.b)?)))
    }
);

export!(
    /// Points in `a` but not in `b`.
    unumflow_interval_difference,
    |text| {
        let r: Two = parse(text)?;
        Ok(pieces(&intervals("a", r.a)?.difference(&intervals("b", r.b)?)))
    }
);

/// Releases a string this library wrote.
///
/// # Safety
///
/// `ptr` must be null or a string returned by this library that has not
/// already been released.
#[no_mangle]
pub unsafe extern "C" fn unumflow_free_string(ptr: *mut libc::c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)) };
    }
}

/// The library version; release it with [`unumflow_free_string`].
#[no_mangle]
pub extern "C" fn unumflow_version() -> *mut libc::c_char {
    CString::new(env!("CARGO_PKG_VERSION"))
        .expect("version string has no interior NUL")
        .into_raw()
}

// ── Tests ─────────────────────────────────────────────────────────────────
//
// These call the exported symbols as a C caller does, so they pin the wire
// contract: a C string in, a status and a JSON body out.

#[cfg(test)]
mod tests {
    use super::*;

    type Export = unsafe extern "C" fn(*const libc::c_char, *mut *mut libc::c_char) -> i32;

    fn call(f: Export, request: &str) -> (i32, Value) {
        let request = CString::new(request).expect("no interior NUL");
        let mut out: *mut libc::c_char = std::ptr::null_mut();
        let status = unsafe { f(request.as_ptr(), &mut out) };
        assert!(!out.is_null(), "every status but -1 comes with a body");
        let body = unsafe { CStr::from_ptr(out) }.to_str().unwrap().to_owned();
        unsafe { unumflow_free_string(out) };
        (status, serde_json::from_str(&body).expect("body is JSON"))
    }

    #[test]
    fn distribution_round_trip() {
        let chi = r#"{"distribution": {"kind": "chi_squared", "k": 4}, "p": 0.95}"#;
        let (s, b) = call(unumflow_distribution_quantile, chi);
        assert_eq!(s, 0, "{b}");
        let q = b["value"].as_f64().unwrap();
        assert!((q - 9.487_729).abs() < 1e-5);

        let (s, b) = call(
            unumflow_distribution_cdf,
            &format!(r#"{{"distribution": {{"kind": "chi_squared", "k": 4}}, "x": {q}}}"#),
        );
        assert_eq!(s, 0, "{b}");
        assert!((b["value"].as_f64().unwrap() - 0.95).abs() < 1e-9);

        let (s, b) = call(
            unumflow_distribution_moments,
            r#"{"distribution": {"kind": "gamma", "shape": 3, "rate": 2}}"#,
        );
        assert_eq!(
            (s, b["mean"].as_f64(), b["variance"].as_f64()),
            (0, Some(1.5), Some(0.75))
        );

        let req = r#"{"distribution": {"kind": "weibull", "shape": 2, "scale": 100}, "n": 4, "seed": 42}"#;
        let (s, a) = call(unumflow_distribution_sample, req);
        let (_, b) = call(unumflow_distribution_sample, req);
        assert_eq!(s, 0);
        assert_eq!(a["values"].as_array().unwrap().len(), 4);
        assert_eq!(a, b, "same seed, same values");
    }

    #[test]
    fn interval_round_trip() {
        let (s, b) = call(
            unumflow_interval_union,
            r#"{"a": [[0, 24]], "b": [[10, 30]]}"#,
        );
        assert_eq!((s, &b["intervals"]), (0, &json!([[0.0, 30.0]])));
        let (_, b) = call(
            unumflow_interval_difference,
            r#"{"a": [[0, 168]], "b": [[0, 30]]}"#,
        );
        assert_eq!(b["intervals"], json!([[30.0, 168.0]]));
        let (_, b) = call(
            unumflow_interval_measure,
            r#"{"intervals": [[0, 24], [10, 30]]}"#,
        );
        assert_eq!(b["value"], json!(30.0));
        let (_, b) = call(
            unumflow_interval_contains,
            r#"{"intervals": [[0, 1]], "t": 1}"#,
        );
        assert_eq!(b["value"], json!(false));
        let (_, b) = call(
            unumflow_interval_intersection,
            r#"{"a": [[0, 10]], "b": [[5, 20]]}"#,
        );
        assert_eq!(b["intervals"], json!([[5.0, 10.0]]));
        let (_, b) = call(
            unumflow_interval_normalize,
            r#"{"intervals": [[3, 4], [1, 3]]}"#,
        );
        assert_eq!(b["intervals"], json!([[1.0, 4.0]]));
    }

    #[test]
    fn refusals_carry_the_same_code_and_fields_as_webassembly() {
        let (s, b) = call(
            unumflow_distribution_cdf,
            r#"{"distribution": {"kind": "weibul", "shape": 1, "scale": 1}, "x": 1}"#,
        );
        assert_eq!((s, b["code"].as_str()), (-3, Some("unknown_option")), "{b}");
        assert_eq!(b["parameter"], "distribution.kind");
        assert_eq!(b["got"], "weibul");

        let (s, b) = call(
            unumflow_distribution_cdf,
            r#"{"distribution": {"kind": "normal", "mu": 0, "sigma": -1}, "x": 1}"#,
        );
        assert_eq!(
            (s, b["code"].as_str()),
            (-3, Some("parameter_out_of_range"))
        );
        assert_eq!(b["parameter"], "distribution.sigma");
        assert_eq!(
            (b["min"].as_f64(), b["max"].is_null(), b["got"].as_f64()),
            (Some(0.0), true, Some(-1.0))
        );

        let (s, b) = call(
            unumflow_distribution_cdf,
            r#"{"distribution": {"kind": "uniform", "min": 3, "max": 1}, "x": 1}"#,
        );
        assert_eq!(
            (s, b["code"].as_str(), b["parameter"].as_str()),
            (-3, Some("invalid_option"), Some("distribution.max"))
        );

        let (s, b) = call(unumflow_interval_union, r#"{"a": [[0, 1]], "b": [[5, 3]]}"#);
        assert_eq!(
            (
                s,
                b["code"].as_str(),
                b["parameter"].as_str(),
                b["index"].as_f64()
            ),
            (-3, Some("reversed_interval"), Some("b"), Some(0.0))
        );

        let (s, b) = call(unumflow_interval_measure, r#"{"intervals": [[0, 1], [2]]}"#);
        assert_eq!(
            (s, b["code"].as_str(), b["parameter"].as_str()),
            (-3, Some("malformed_input"), Some("intervals[1]"))
        );

        let (s, b) = call(
            unumflow_distribution_quantile,
            r#"{"distribution": {"kind": "normal", "mu": 0, "sigma": 1}, "p": 1}"#,
        );
        assert_eq!(
            (s, b["code"].as_str(), b["parameter"].as_str()),
            (-3, Some("parameter_out_of_range"), Some("p"))
        );

        let (s, b) = call(
            unumflow_distribution_sample,
            r#"{"distribution": {"kind": "normal", "mu": 0, "sigma": 1}, "n": 2.5, "seed": 1}"#,
        );
        assert_eq!((s, b["parameter"].as_str()), (-3, Some("n")));

        let (s, b) = call(unumflow_distribution_cdf, "not json");
        assert_eq!((s, b["code"].as_str()), (-2, Some("malformed_input")));
        let (s, b) = call(
            unumflow_distribution_cdf,
            r#"{"distribution": {"kind": "normal", "mu": 0, "sigma": 1}}"#,
        );
        assert_eq!((s, b["parameter"].as_str()), (-2, Some("x")));
        let (s, b) = call(
            unumflow_distribution_cdf,
            r#"{"distribution": {"kind": "normal", "mu": 0, "sigma": 1, "sd": 2}, "x": 0}"#,
        );
        assert_eq!((s, b["parameter"].as_str()), (-2, Some("distribution")));
    }

    #[test]
    fn null_pointers_are_refused_without_a_body() {
        let mut out: *mut libc::c_char = std::ptr::null_mut();
        assert_eq!(
            unsafe { unumflow_interval_measure(std::ptr::null(), &mut out) },
            -1
        );
        assert!(out.is_null());
        let req = CString::new("{}").unwrap();
        assert_eq!(
            unsafe { unumflow_interval_measure(req.as_ptr(), std::ptr::null_mut()) },
            -1
        );
    }

    #[test]
    fn version_is_the_crate_version() {
        let v = unumflow_version();
        let s = unsafe { CStr::from_ptr(v) }.to_str().unwrap().to_owned();
        unsafe { unumflow_free_string(v) };
        assert_eq!(s, env!("CARGO_PKG_VERSION"));
    }
}
