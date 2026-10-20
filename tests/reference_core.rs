//! Descriptive statistics, seeded sampling, interval sets and unit conversion
//! against NumPy, SciPy and the definitions. The constants were printed by that
//! recipe with the versions in the next line.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

// Generated with NumPy 2.4.6, SciPy 1.17.1.
const ODD: [f64; 41] = [
    51.376733, 44.56596, 63.00053, 55.548272, 48.584042, 54.117955, 48.149411, 46.376227,
    42.455992, 35.116838, 24.968877, 43.091502, 37.065071, 51.239176, 52.146074, 23.954962,
    56.449366, 64.148943, 53.28856, 57.732317, 33.04684, 63.656794, 52.798658, 67.664426,
    47.055656, 71.910725, 45.328196, 44.149553, 18.172515, 41.610049, 46.17218, 55.27447,
    39.592695, 38.417085, 57.272323, 45.807541, 48.269918, 58.387019, 41.451537, 49.905845,
    56.188702,
];
const ODD_STATS: [f64; 4] = [
    48.183159390243894,
    130.25980964238,
    11.413141970657335,
    48.269918,
];
const EVEN: [f64; 40] = [
    51.376733, 44.56596, 63.00053, 55.548272, 48.584042, 54.117955, 48.149411, 46.376227,
    42.455992, 35.116838, 24.968877, 43.091502, 37.065071, 51.239176, 52.146074, 23.954962,
    56.449366, 64.148943, 53.28856, 57.732317, 33.04684, 63.656794, 52.798658, 67.664426,
    47.055656, 71.910725, 45.328196, 44.149553, 18.172515, 41.610049, 46.17218, 55.27447,
    39.592695, 38.417085, 57.272323, 45.807541, 48.269918, 58.387019, 41.451537, 49.905845,
];
const EVEN_STATS: [f64; 4] = [
    47.983020825,
    131.9154219335042,
    11.485443915387172,
    48.2096645,
];
const OFFSET: [f64; 60] = [
    1000000000.540072,
    999999998.074016,
    999999998.483718,
    1000000000.53611,
    1000000000.319291,
    1000000001.835493,
    999999999.335108,
    999999999.594609,
    999999999.632879,
    1000000000.748587,
    1000000000.794876,
    999999999.787746,
    999999998.329783,
    1000000000.401029,
    999999999.685293,
    1000000000.518451,
    1000000000.427501,
    999999999.912807,
    999999999.069652,
    999999999.022261,
    1000000000.639316,
    999999998.810983,
    1000000000.652522,
    999999997.804046,
    999999999.508387,
    999999999.685357,
    1000000000.944797,
    1000000000.584665,
    999999999.480495,
    999999999.212578,
    1000000000.695641,
    999999999.854266,
    999999998.944,
    1000000000.385467,
    999999999.43907,
    1000000000.406521,
    999999999.744951,
    999999999.719281,
    1000000001.894353,
    999999999.284746,
    1000000002.484775,
    1000000000.314538,
    1000000000.484404,
    999999998.342718,
    1000000000.169303,
    999999998.038441,
    1000000000.638764,
    1000000002.032759,
    999999999.794392,
    1000000000.424218,
    999999999.706676,
    1000000000.848164,
    1000000002.818135,
    1000000000.058824,
    1000000000.408426,
    1000000002.087645,
    1000000001.549883,
    999999999.36475,
    999999999.618494,
    999999998.691507,
];
const OFFSET_MEAN: f64 = 1000000000.0436257;
const OFFSET_VAR: f64 = 1.1714172779630763;
const OFFSET_MEDIAN: f64 = 999999999.9858155;
const Q_UNIFORM: [f64; 5] = [2.3, 2.75, 3.5, 4.25, 4.7];
const Q_TRIANGULAR: [f64; 5] = [
    2.1832159566199234,
    2.870828693386971,
    3.8166998673296226,
    5.041960108450192,
    6.12917130661303,
];
const Q_NORMAL: [f64; 5] = [
    7.436896868910799,
    8.651020499607837,
    10.0,
    11.348979500392163,
    12.5631031310892,
];
const Q_LOGNORMAL: [f64; 5] = [
    1.432217893498685,
    1.9401302789180261,
    2.718281828459045,
    3.808536044832714,
    5.159170355622591,
];
const Q_WEIBULL: [f64; 5] = [
    0.8923021025476684,
    1.743151726812095,
    3.1328790750986055,
    4.973135539500686,
    6.974886054385647,
];
const Q_EXPONENTIAL: [f64; 5] = [
    0.15051502236832331,
    0.4109743892168299,
    0.990210257942779,
    1.980420515885558,
    3.2894072757057797,
];
const Q_GAMMA: [f64; 5] = [
    0.6193492257547395,
    1.0286933882431397,
    1.6736385350367415,
    2.548338370703557,
    3.552444961454278,
];
const Q_BETA: [f64; 5] = [
    0.09259525891312874,
    0.16116291679032654,
    0.26444998329566,
    0.3894794852007245,
    0.5103163065514917,
];
const Q_CHISQUARED: [f64; 5] = [
    1.0636232167792241,
    1.9225575262295542,
    3.3566939800333224,
    5.385269057779391,
    7.779440339734858,
];
const Q_PERT: [f64; 5] = [
    1.3619996256367142,
    2.2512320450124763,
    3.5099948494911826,
    4.931707389326274,
    6.207544856651202,
];

use u_numflow::collections::IntervalSet;
use u_numflow::distributions::{
    BetaDistribution, ChiSquared, Exponential, GammaDistribution, LogNormal, Normal, Pert, Sample,
    Triangular, Uniform, Weibull,
};
use u_numflow::stats::{mean, median, std_dev, variance};
use u_numflow::units::{Dimension, Units};

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let scale = want.abs().max(1e-300);
    assert!(
        (got - want).abs() <= rel * scale,
        "{what}: got {got}, want {want}"
    );
}

#[test]
fn descriptive_statistics_match_numpy() {
    for (name, data, want) in [
        ("odd", &ODD[..], ODD_STATS),
        ("even", &EVEN[..], EVEN_STATS),
    ] {
        close(mean(data).unwrap(), want[0], 1e-13, &format!("{name} mean"));
        close(
            variance(data).unwrap(),
            want[1],
            1e-12,
            &format!("{name} variance"),
        );
        close(
            std_dev(data).unwrap(),
            want[2],
            1e-12,
            &format!("{name} std"),
        );
        close(
            median(data).unwrap(),
            want[3],
            1e-14,
            &format!("{name} median"),
        );
    }
}

/// A variance is of the spread, not of the offset: data near 1e9 with unit
/// spread keep their variance (the squares of the raw values are 1e18).
#[test]
fn variance_of_data_on_a_large_offset_is_not_lost() {
    close(mean(&OFFSET).unwrap(), OFFSET_MEAN, 1e-15, "offset mean");
    close(
        variance(&OFFSET).unwrap(),
        OFFSET_VAR,
        1e-6,
        "offset variance",
    );
    close(
        median(&OFFSET).unwrap(),
        OFFSET_MEDIAN,
        1e-15,
        "offset median",
    );
}

/// The empirical fraction below each SciPy quantile is the probability asked
/// for, to four standard errors (the seed is fixed, so the check is exact).
#[test]
fn sampled_values_fall_below_the_scipy_quantiles_with_the_right_frequency() {
    const N: usize = 20_000;
    let probs = [0.1, 0.25, 0.5, 0.75, 0.9];
    let cases: Vec<(&str, Vec<f64>, [f64; 5])> = vec![
        (
            "uniform",
            Uniform::new(2.0, 5.0).unwrap().sample_n(N, 11),
            Q_UNIFORM,
        ),
        (
            "triangular",
            Triangular::new(1.0, 3.0, 8.0).unwrap().sample_n(N, 11),
            Q_TRIANGULAR,
        ),
        (
            "normal",
            Normal::new(10.0, 2.0).unwrap().sample_n(N, 11),
            Q_NORMAL,
        ),
        (
            "lognormal",
            LogNormal::new(1.0, 0.5).unwrap().sample_n(N, 11),
            Q_LOGNORMAL,
        ),
        (
            "weibull",
            Weibull::new(1.5, 4.0).unwrap().sample_n(N, 11),
            Q_WEIBULL,
        ),
        (
            "exponential",
            Exponential::new(0.7).unwrap().sample_n(N, 11),
            Q_EXPONENTIAL,
        ),
        (
            "gamma",
            GammaDistribution::new(2.5, 1.3).unwrap().sample_n(N, 11),
            Q_GAMMA,
        ),
        (
            "beta",
            BetaDistribution::new(2.0, 5.0).unwrap().sample_n(N, 11),
            Q_BETA,
        ),
        (
            "chi-squared",
            ChiSquared::new(4.0).unwrap().sample_n(N, 11),
            Q_CHISQUARED,
        ),
        (
            "pert",
            Pert::new(0.0, 3.0, 10.0).unwrap().sample_n(N, 11),
            Q_PERT,
        ),
    ];
    for (name, sample, quantiles) in cases {
        for (p, q) in probs.iter().zip(quantiles) {
            let frac = sample.iter().filter(|&&x| x <= q).count() as f64 / N as f64;
            let se = (p * (1.0 - p) / N as f64).sqrt();
            assert!(
                (frac - p).abs() < 4.0 * se,
                "{name}: {frac} of the sample is below the {p} quantile {q}"
            );
        }
    }
}

fn random_list(next: &mut dyn FnMut(u64) -> u64) -> Vec<(f64, f64)> {
    let count = next(6) as usize;
    (0..count)
        .map(|_| {
            let a = next(24);
            let b = next(24);
            (a.min(b) as f64, a.max(b) as f64)
        })
        .collect()
}

fn inside(list: &[(f64, f64)], t: f64) -> bool {
    list.iter().any(|&(s, e)| s <= t && t < e)
}

/// Union, intersection and difference are the point-wise set operations, and
/// the measure is the total length, checked on every half-open unit cell.
#[test]
fn interval_sets_are_the_pointwise_set_operations() {
    let mut state = 12345u64;
    let mut next = |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % m
    };
    for _ in 0..200 {
        let la = random_list(&mut next);
        let lb = random_list(&mut next);
        let a = IntervalSet::from_intervals(la.clone()).unwrap();
        let b = IntervalSet::from_intervals(lb.clone()).unwrap();
        let union = a.union(&b);
        let inter = a.intersection(&b);
        let diff = a.difference(&b);
        for k in -2..52 {
            let t = k as f64 * 0.5;
            let (ia, ib) = (inside(&la, t), inside(&lb, t));
            assert_eq!(union.contains(t), ia || ib, "union at {t}");
            assert_eq!(inter.contains(t), ia && ib, "intersection at {t}");
            assert_eq!(diff.contains(t), ia && !ib, "difference at {t}");
        }
        // normal form: sorted, disjoint, non-adjacent, non-empty pieces
        for set in [&a, &b, &union, &inter, &diff] {
            let pieces = set.as_slice();
            assert!(pieces.iter().all(|&(s, e)| s < e));
            assert!(pieces.windows(2).all(|w| w[0].1 < w[1].0), "{pieces:?}");
        }
        let cells = |keep: &dyn Fn(f64) -> bool| (0..24).filter(|&c| keep(c as f64 + 0.25)).count();
        close(
            union.measure(),
            cells(&|t| inside(&la, t) || inside(&lb, t)) as f64,
            1e-12,
            "union measure",
        );
        close(
            inter.measure() + 1.0,
            cells(&|t| inside(&la, t) && inside(&lb, t)) as f64 + 1.0,
            1e-12,
            "intersection measure",
        );
        close(
            diff.measure() + 1.0,
            cells(&|t| inside(&la, t) && !inside(&lb, t)) as f64 + 1.0,
            1e-12,
            "difference measure",
        );
    }
}

/// Conversions that follow from the definitions of the SI units and prefixes.
#[test]
fn si_conversions_follow_from_the_definitions() {
    let units = Units::si();
    let cases: [(f64, &str, &str, f64); 22] = [
        (1.0, "km", "m", 1000.0),
        (1.0, "cm", "m", 0.01),
        (1.0, "kg", "g", 1000.0),
        (2.5, "mmol", "mol", 0.0025),
        (1.0, "kPa", "Pa", 1000.0),
        (1.0, "MHz", "Hz", 1e6),
        (1.0, "uV", "V", 1e-6),
        (1.0, "mT", "T", 1e-3),
        (1.0, "dam", "m", 10.0),
        (1.0, "g/cm3", "kg/m3", 1000.0),
        (1.0, "N.m", "J", 1.0),
        (1.0, "W.s", "J", 1.0),
        (1.0, "Pa", "N/m2", 1.0),
        (1.0, "J", "kg.m2/s2", 1.0),
        (1.0, "V", "W/A", 1.0),
        (1.0, "Ohm", "V/A", 1.0),
        (1.0, "F", "C/V", 1.0),
        (1.0, "H", "Wb/A", 1.0),
        (1.0, "T", "Wb/m2", 1.0),
        (1.0, "Gy", "J/kg", 1.0),
        (25.0, "Cel", "K", 298.15),
        (300.0, "K", "Cel", 26.85),
    ];
    for (value, from, to, want) in cases {
        let got = units
            .convert(value, from, to)
            .unwrap_or_else(|e| panic!("{value} {from} -> {to}: {e:?}"));
        close(got, want, 1e-12, &format!("{value} {from} -> {to}"));
    }
    // a temperature difference has no offset
    close(
        units.convert_difference(10.0, "Cel", "K").unwrap(),
        10.0,
        1e-15,
        "difference",
    );
    // a different dimension, and an offset unit in a product, are refused
    assert!(units.convert(1.0, "m", "s").is_err());
    assert!(units.convert(1.0, "Cel.m", "K.m").is_err());
    // the dimension of the newton is length x mass / time^2
    assert_eq!(
        units.parse("N").unwrap().dimension,
        Dimension([1, 1, -2, 0, 0, 0, 0])
    );
    assert_eq!(
        units.parse("kN").unwrap().dimension,
        units.parse("kg.m/s2").unwrap().dimension
    );
}
