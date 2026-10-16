//! The keys and names each C entry point reads, measured from the entry
//! points themselves and held in `request_keys.json` beside this file -- the
//! file a binding's contract tests check the requests they write against.
//!
//! Every request type refuses a key it does not know and names the keys it
//! does (`unknown field ..., expected one of ...`); every option refuses a name
//! it does not know and lists the names it does (`unknown_option` with
//! `expected`). So a valid request is sent with a key no request has at each
//! level, and with a name no option has at each value, and the refusals are
//! read. Each value is described as one of
//!
//! - `{"properties": {key: value, ...}}` -- an object;
//! - `{"tag": key, "variants": {name: {"properties": ...}, ...}}` -- an object
//!   whose `tag` key names which keys it has;
//! - `{"items": value}` -- an array of objects or of names;
//! - `{"enum": [name, ...]}` -- one of these names, with `"number": true` when
//!   a number is taken there too;
//! - `{"enum": [name, ...], "items": {"enum": [...]}}` -- one of these names,
//!   or an array of the other names (a named set, or the members spelt out);
//! - `{}` -- anything else: a number, an array of numbers.
//!
//! An object that does not refuse an unknown key cannot be measured, and fails
//! the measurement: a misspelt key it ignores would never reach the caller.
//!
//! `UPDATE_REQUEST_KEYS=1 cargo test --features ffi --lib request_keys`
//! rewrites the file; review its diff like any other change to the contract.

use super::*;
// What differs from one crate's copy of this module to another's.
use super::unumflow_free_string as free_string;
const FREE_STRING: &str = "unumflow_free_string";

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::PathBuf;

type Entry = unsafe extern "C" fn(*const libc::c_char, *mut *mut libc::c_char) -> i32;

/// A key and a name no request has.
const PROBE: &str = "__probe__";

#[derive(Clone)]
enum Step {
    Key(String),
    Index(usize),
}

fn call(entry: Entry, request: &Value) -> (i32, Value) {
    let request = CString::new(request.to_string()).expect("no interior NUL");
    let mut out: *mut libc::c_char = std::ptr::null_mut();
    let code = unsafe { entry(request.as_ptr(), &mut out) };
    assert!(!out.is_null(), "every status comes with a JSON body");
    let body = unsafe { CStr::from_ptr(out) }
        .to_str()
        .expect("body is UTF-8")
        .to_owned();
    unsafe { free_string(out) };
    (code, serde_json::from_str(&body).expect("body is JSON"))
}

/// `limits.ucl`, `customers[0].id`: how a refusal's `parameter` names a place.
fn parameter(path: &[Step]) -> String {
    let mut out = String::new();
    for step in path {
        match step {
            Step::Key(k) if out.is_empty() => out.push_str(k),
            Step::Key(k) => {
                out.push('.');
                out.push_str(k);
            }
            Step::Index(i) => out.push_str(&format!("[{i}]")),
        }
    }
    out
}

/// `base` with `value` at `path`; an object on the way is made when absent.
fn put(base: &Value, path: &[Step], value: Value) -> Value {
    let mut request = base.clone();
    let mut here = &mut request;
    for step in path {
        here = match step {
            Step::Key(k) => {
                if here.is_null() {
                    *here = json!({});
                }
                here.as_object_mut()
                    .unwrap_or_else(|| panic!("{} is not an object in {base}", parameter(path)))
                    .entry(k.clone())
                    .or_insert(Value::Null)
            }
            Step::Index(i) => here.get_mut(*i).unwrap_or_else(|| {
                panic!("the base request needs an element at {}", parameter(path))
            }),
        };
    }
    *here = value;
    request
}

fn get<'a>(base: &'a Value, path: &[Step]) -> Option<&'a Value> {
    path.iter().try_fold(base, |v, step| match step {
        Step::Key(k) => v.get(k),
        Step::Index(i) => v.get(*i),
    })
}

/// The keys an unknown-field refusal of the probe key lists.
fn known_keys(body: &Value) -> Option<Vec<String>> {
    let text = body["error"].as_str()?;
    let marker = format!("unknown field `{PROBE}`, ");
    let rest = &text[text.find(&marker)? + marker.len()..];
    let rest = rest.split(" at line ").next().unwrap_or(rest);
    if rest.starts_with("there are no fields") {
        return Some(Vec::new());
    }
    Some(
        rest.split('`')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect(),
    )
}

/// The names an unknown-option refusal at `at` lists.
fn known_names(body: &Value, at: &str) -> Option<Vec<String>> {
    if body["code"] != "unknown_option" || body["parameter"] != at {
        return None;
    }
    let expected = body["expected"]
        .as_array()
        .unwrap_or_else(|| panic!("an unknown option without `expected`: {body}"));
    Some(
        expected
            .iter()
            .map(|n| n.as_str().expect("a name").to_string())
            .collect(),
    )
}

/// An object that took the probe key without a word cannot be measured.
fn refuse_silence(name: &str, at: &str, body: &Value) {
    let text = body["error"].as_str().unwrap_or_default();
    assert!(
        !text.contains("missing field"),
        "{name}: {at} took a key it does not know without refusing it \
         (or the base request lacks a key it needs): {body}"
    );
}

fn object(name: &str, entry: Entry, base: &Value, path: &[Step], keys: Vec<String>) -> Value {
    let properties: BTreeMap<String, Value> = keys
        .into_iter()
        .map(|key| {
            let mut at = path.to_vec();
            at.push(Step::Key(key.clone()));
            let value = describe(name, entry, base, &at);
            (key, value)
        })
        .collect();
    json!({ "properties": properties })
}

/// What the value at `path` of a valid `base` request is.
fn describe(name: &str, entry: Entry, base: &Value, path: &[Step]) -> Value {
    let at = parameter(path);

    // An object: the probe key is refused with the keys it has.
    let (_, body) = call(entry, &put(base, path, json!({ PROBE: 0 })));
    if let Some(keys) = known_keys(&body) {
        return object(name, entry, base, path, keys);
    }
    refuse_silence(name, &at, &body);
    // An object whose tag key names its variant: the object without the tag
    // is refused at that key, and a name no variant has with the names of
    // the variants.
    let tag = body["parameter"]
        .as_str()
        .and_then(|p| p.strip_prefix(&format!("{at}.")))
        .filter(|tag| !tag.contains(['.', '[']))
        .map(str::to_string);
    let variants = tag.as_ref().and_then(|tag| {
        let (_, body) = call(entry, &put(base, path, json!({ tag.as_str(): PROBE })));
        known_names(&body, &format!("{at}.{tag}"))
    });
    if let (Some(tag), Some(variants)) = (tag, variants) {
        let variants: BTreeMap<String, Value> = variants
            .into_iter()
            .map(|variant| {
                let (_, body) = call(
                    entry,
                    &put(base, path, json!({ tag.as_str(): variant, PROBE: 0 })),
                );
                let keys: Vec<String> = known_keys(&body)
                    .unwrap_or_else(|| panic!("{name}: {at} as {variant:?}: {body}"))
                    .into_iter()
                    .filter(|k| *k != tag)
                    .collect();
                let carried = get(base, path).and_then(|o| o.get(&tag)) == Some(&json!(variant));
                let variant_base = if carried {
                    base.clone()
                } else {
                    put(base, path, json!({ tag.as_str(): variant }))
                };
                let described = object(name, entry, &variant_base, path, keys);
                (variant, described)
            })
            .collect();
        return json!({ "tag": tag, "variants": variants });
    }

    // An array of objects.
    let (_, body) = call(entry, &put(base, path, json!([{ PROBE: 0 }])));
    if let Some(keys) = known_keys(&body) {
        let mut item = path.to_vec();
        item.push(Step::Index(0));
        return json!({ "items": object(name, entry, base, &item, keys) });
    }
    refuse_silence(name, &at, &body);

    // A name, or an array of names.
    let (_, body) = call(entry, &put(base, path, json!(PROBE)));
    if let Some(names) = known_names(&body, &at) {
        // A name, or an array of other names: the array is refused with those.
        let (_, listed) = call(entry, &put(base, path, json!([PROBE])));
        let items = known_names(&listed, &at)
            .or_else(|| known_names(&listed, &format!("{at}[0]")))
            .filter(|items| *items != names);
        if let Some(items) = items {
            return json!({ "enum": names, "items": { "enum": items } });
        }
        let (_, body) = call(entry, &put(base, path, json!(1)));
        let text = body["error"].as_str().unwrap_or_default();
        return if text.contains("invalid type") {
            json!({ "enum": names })
        } else {
            json!({ "enum": names, "number": true })
        };
    }
    let (_, body) = call(entry, &put(base, path, json!([PROBE])));
    if let Some(names) = known_names(&body, &at).or_else(|| known_names(&body, &format!("{at}[0]")))
    {
        return json!({ "items": { "enum": names } });
    }
    json!({})
}

fn measure(entry_points: &[(&str, Entry, Value)]) -> Value {
    let measured: BTreeMap<&str, Value> = entry_points
        .iter()
        .map(|(name, entry, base)| {
            let (code, body) = call(*entry, base);
            assert_eq!(code, 0, "{name}: the base request is refused: {body}");
            let (_, body) = call(*entry, &json!({ PROBE: 0 }));
            let keys = known_keys(&body)
                .unwrap_or_else(|| panic!("{name} does not refuse a key it does not know: {body}"));
            (*name, object(name, *entry, base, &[], keys))
        })
        .collect();
    json!({
        "$comment": "The keys and names each C entry point reads, measured from the entry points (src/ffi/request_keys.rs). Regenerate: UPDATE_REQUEST_KEYS=1 cargo test --features ffi --lib request_keys",
        "entry_points": measured,
    })
}

/// The JSON entry points `ffi.rs` exports, read from its source: each
/// `pub unsafe extern "C" fn` and each name an `...export!` macro is given.
fn exported() -> Vec<String> {
    let source = include_str!("../ffi.rs");
    let source = &source[..source.find("// ── Tests").expect("the tests marker")];
    let mut names = Vec::new();
    let mut in_macro_call = false;
    for line in source.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("pub unsafe extern \"C\" fn ") {
            if let Some(name) = rest.split('(').next().filter(|n| !n.starts_with('$')) {
                names.push(name.to_string());
            }
        } else if line.ends_with("export!(") && !line.starts_with("macro_rules!") {
            in_macro_call = true;
        } else if in_macro_call && !line.starts_with("///") && !line.starts_with("#[") {
            let name = line.split([',', ' ']).next().expect("a name");
            names.push(name.to_string());
            in_macro_call = false;
        }
    }
    names.retain(|n| n != FREE_STRING);
    names.sort();
    names
}

// ── The entry points and a valid request to each ──────────────────────────

fn ep(name: &'static str, entry: Entry, base: Value) -> (&'static str, Entry, Value) {
    (name, entry, base)
}

fn entry_points() -> Vec<(&'static str, Entry, Value)> {
    let normal = json!({ "kind": "normal", "mu": 0, "sigma": 1 });
    let intervals = json!([[0, 2], [1, 4], [6, 7]]);
    let calorie = json!([{ "symbol": "cal_IT", "unit": "J", "factor": 4.1868, "offset": 0 }]);
    vec![
        ep(
            "unumflow_distribution_moments",
            unumflow_distribution_moments,
            json!({ "distribution": normal }),
        ),
        ep(
            "unumflow_distribution_cdf",
            unumflow_distribution_cdf,
            json!({ "distribution": normal, "x": 0.5 }),
        ),
        ep(
            "unumflow_distribution_quantile",
            unumflow_distribution_quantile,
            json!({ "distribution": normal, "p": 0.9 }),
        ),
        ep(
            "unumflow_distribution_sample",
            unumflow_distribution_sample,
            json!({ "distribution": normal, "n": 3, "seed": 7 }),
        ),
        ep(
            "unumflow_interval_normalize",
            unumflow_interval_normalize,
            json!({ "intervals": intervals }),
        ),
        ep(
            "unumflow_interval_measure",
            unumflow_interval_measure,
            json!({ "intervals": intervals }),
        ),
        ep(
            "unumflow_interval_contains",
            unumflow_interval_contains,
            json!({ "intervals": intervals, "t": 1.5 }),
        ),
        ep(
            "unumflow_interval_union",
            unumflow_interval_union,
            json!({ "a": intervals, "b": [[3, 5]] }),
        ),
        ep(
            "unumflow_interval_intersection",
            unumflow_interval_intersection,
            json!({ "a": intervals, "b": [[3, 5]] }),
        ),
        ep(
            "unumflow_interval_difference",
            unumflow_interval_difference,
            json!({ "a": intervals, "b": [[3, 5]] }),
        ),
        ep(
            "unumflow_unit_convert",
            unumflow_unit_convert,
            json!({ "value": 1, "from": "kcal_IT", "to": "J", "definitions": calorie }),
        ),
        ep(
            "unumflow_unit_convert_difference",
            unumflow_unit_convert_difference,
            json!({ "value": 1, "from": "kcal_IT", "to": "J", "definitions": calorie }),
        ),
        ep(
            "unumflow_unit_info",
            unumflow_unit_info,
            json!({ "unit": "kcal_IT", "definitions": calorie }),
        ),
    ]
}

fn held_file() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/ffi/request_keys.json")
}

#[test]
fn every_exported_entry_point_is_measured() {
    let mut measured: Vec<String> = entry_points()
        .iter()
        .map(|(n, _, _)| n.to_string())
        .collect();
    measured.sort();
    assert_eq!(measured, exported());
}

#[test]
fn the_held_request_keys_are_what_the_entry_points_read() {
    let text = serde_json::to_string_pretty(&measure(&entry_points())).expect("JSON") + "\n";
    let path = held_file();
    if std::env::var_os("UPDATE_REQUEST_KEYS").is_some() {
        std::fs::write(&path, &text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        return;
    }
    let held = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .replace("\r\n", "\n");
    assert!(
        held == text,
        "{} is not what the entry points read. Rerun with UPDATE_REQUEST_KEYS=1 and review the \
         diff; the bindings' contract tests read this file.\nmeasured:\n{text}",
        path.display()
    );
}
