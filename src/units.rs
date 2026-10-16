//! Units of measure: dimensional analysis and conversion between units of the
//! same dimension.
//!
//! A unit is a point in the space of physical dimensions — integer powers of
//! the seven SI base quantities — together with an affine map to the coherent
//! SI unit of that dimension: `value_in_si = value · scale + offset`. Most
//! units have offset 0 (a kilowatt hour is 3.6 × 10⁶ J); a temperature scale
//! such as the degree Celsius has an offset (0 °C = 273.15 K).
//!
//! Two conversions are distinct and both are offered:
//!
//! - [`Units::convert`] converts a **reading** — 20 °C is 293.15 K;
//! - [`Units::convert_difference`] converts an **amount of change** — a rise
//!   of 20 °C is a rise of 20 K. The offset cancels in a difference.
//!
//! Mixing the two is a silent error with an offset unit, so the caller picks.
//!
//! # Expressions
//!
//! Units are written as expressions in the case-sensitive syntax of UCUM
//! (the Unified Code for Units of Measure), restricted to what dimensional
//! conversion needs:
//!
//! - an atom (`m`, `s`, `J`), optionally with a decimal prefix (`km`, `mg`,
//!   `kW`, `uA` — `u` is micro);
//! - an integer exponent written after it (`m3`, `s-2`, `m+2`);
//! - products with `.` and quotients with `/`, evaluated left to right
//!   (`kW.h`, `m/s2`, `kg/m.s` = `(kg/m)·s`), a leading `/` for a reciprocal
//!   (`/s`);
//! - parentheses (`J/(kg.K)`), positive integers as factors (`1000.m`), `1`
//!   for unity;
//! - annotations in braces (`m3{gas}`), which UCUM defines to have no effect
//!   on the unit and which are read as such.
//!
//! An atom is resolved before it is split into prefix and atom (`cd` is the
//! candela, not a centi-day), and a whole token is tried as an atom before a
//! trailing exponent is split off — so an atom whose symbol ends in digits
//! (`Nm3`, defined by a caller) stays whole.
//!
//! A unit with an offset stands alone: a prefix, an exponent or a product
//! applied to it is refused, since none has a meaning for a reading.
//!
//! # Built-in units
//!
//! [`Units::si`] knows the SI: the base units `m`, `g` (with prefix: `kg`),
//! `s`, `A`, `K`, `mol`, `cd`; the 22 coherent derived units with special
//! names (`rad`, `sr`, `Hz`, `N`, `Pa`, `J`, `W`, `C`, `V`, `F`, `Ohm`, `S`,
//! `Wb`, `T`, `H`, `Cel`, `lm`, `lx`, `Bq`, `Gy`, `Sv`, `kat`); the units the SI
//! accepts for use with it that are exact multiples (`min`, `h`, `d`, `L`/`l`,
//! `t`); and the decimal prefixes `Q` … `q`. Everything else — calorie
//! variants, the tonne of oil equivalent, the normal cubic metre, imperial
//! units — is defined by the caller with [`Units::define`], so this crate
//! never chooses between same-named units of different size (the 15 °C,
//! international and thermochemical calories differ by up to 0.07 %).
//!
//! # Example
//!
//! ```
//! use u_numflow::units::Units;
//!
//! let mut units = Units::si();
//! units.define("cal_IT", 4.1868, "J", 0.0).unwrap();
//! assert!((units.convert(1.0, "kW.h", "J").unwrap() - 3.6e6).abs() < 1e-6);
//! assert!((units.convert(1.0, "kcal_IT", "J").unwrap() - 4186.8).abs() < 1e-9);
//! assert!((units.convert(20.0, "Cel", "K").unwrap() - 293.15).abs() < 1e-9);
//! assert_eq!(units.convert_difference(20.0, "Cel", "K").unwrap(), 20.0);
//! assert!(units.convert(1.0, "m3", "kg").is_err()); // volume is not mass
//! ```
//!
//! # References
//! - BIPM (2019), *The International System of Units (SI)*, 9th ed.
//! - Schadow & McDonald, *The Unified Code for Units of Measure*, §2–§5
//!   (syntax only; no UCUM table is embedded).

use std::collections::HashMap;
use std::fmt;

/// Symbols of the SI base units, in the order of a [`Dimension`]'s exponents.
pub const BASE_SYMBOLS: [&str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];

/// Integer exponents of the seven SI base quantities — length, mass, time,
/// electric current, thermodynamic temperature, amount of substance,
/// luminous intensity, in that order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Dimension(pub [i32; 7]);

impl Dimension {
    /// The dimension of a pure number.
    pub const NONE: Dimension = Dimension([0; 7]);

    fn base(index: usize) -> Self {
        let mut e = [0; 7];
        e[index] = 1;
        Dimension(e)
    }

    fn times(self, other: Self) -> Self {
        let mut e = self.0;
        for (a, b) in e.iter_mut().zip(other.0) {
            *a += b;
        }
        Dimension(e)
    }

    fn pow(self, n: i32) -> Self {
        Dimension(self.0.map(|e| e * n))
    }

    /// Whether this is the dimension of a pure number.
    pub fn is_none(&self) -> bool {
        *self == Self::NONE
    }
}

/// Written as a product of SI base units in UCUM syntax, e.g. `kg.m2.s-2`
/// (or `1` for a pure number).
impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Mass first reads as engineers write it: kg.m2.s-2.
        const ORDER: [usize; 7] = [1, 0, 2, 3, 4, 5, 6];
        let parts: Vec<String> = ORDER
            .iter()
            .filter(|&&i| self.0[i] != 0)
            .map(|&i| match self.0[i] {
                1 => BASE_SYMBOLS[i].to_string(),
                e => format!("{}{e}", BASE_SYMBOLS[i]),
            })
            .collect();
        if parts.is_empty() {
            f.write_str("1")
        } else {
            f.write_str(&parts.join("."))
        }
    }
}

/// A unit: its dimension and the affine map to the coherent SI unit of that
/// dimension, `value_in_si = value · scale + offset`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unit {
    /// The physical dimension.
    pub dimension: Dimension,
    /// Size of one unit in coherent SI units (`> 0`).
    pub scale: f64,
    /// SI value of the unit's zero (0 except for scales such as `Cel`).
    pub offset: f64,
}

impl Unit {
    const ONE: Unit = Unit {
        dimension: Dimension::NONE,
        scale: 1.0,
        offset: 0.0,
    };

    /// A reading `value` in this unit, in `to`.
    ///
    /// # Errors
    /// [`UnitError::NotFinite`] for `value`, [`UnitError::DimensionMismatch`]
    /// when `to` measures something else.
    pub fn convert_to(&self, value: f64, to: &Unit) -> Result<f64, UnitError> {
        self.check(value, to)?;
        Ok((value * self.scale + self.offset - to.offset) / to.scale)
    }

    /// An amount of change `value` in this unit, in `to` (offsets cancel).
    ///
    /// # Errors
    /// As [`Unit::convert_to`].
    pub fn difference_to(&self, value: f64, to: &Unit) -> Result<f64, UnitError> {
        self.check(value, to)?;
        Ok(value * self.scale / to.scale)
    }

    fn check(&self, value: f64, to: &Unit) -> Result<(), UnitError> {
        if !value.is_finite() {
            return Err(UnitError::NotFinite { parameter: "value" });
        }
        if self.dimension != to.dimension {
            return Err(UnitError::DimensionMismatch {
                from: self.dimension,
                to: to.dimension,
            });
        }
        Ok(())
    }

    fn times(self, other: Unit) -> Unit {
        Unit {
            dimension: self.dimension.times(other.dimension),
            scale: self.scale * other.scale,
            offset: 0.0,
        }
    }

    fn pow(self, n: i32) -> Unit {
        Unit {
            dimension: self.dimension.pow(n),
            scale: self.scale.powi(n),
            offset: 0.0,
        }
    }
}

/// Why an expression, a definition or a conversion was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum UnitError {
    /// The expression is not well formed (an empty term, an unbalanced
    /// parenthesis, a misplaced operator, an exponent too large).
    Malformed {
        /// The expression as given.
        expression: String,
        /// What is wrong, in words.
        reason: String,
    },
    /// A symbol that is neither a known atom nor a prefix with one.
    UnknownSymbol {
        /// The expression as given.
        expression: String,
        /// The symbol not recognised.
        symbol: String,
    },
    /// A unit with an offset given a prefix, an exponent, or a product.
    OffsetCombined {
        /// The expression as given.
        expression: String,
        /// The offset unit's symbol.
        symbol: String,
    },
    /// A conversion between different dimensions.
    DimensionMismatch {
        /// The dimension of the unit converted from.
        from: Dimension,
        /// The dimension of the unit converted to.
        to: Dimension,
    },
    /// A definition whose symbol is already an atom (built in or defined).
    DuplicateSymbol {
        /// The symbol.
        symbol: String,
    },
    /// A definition whose symbol is not usable as an atom.
    InvalidSymbol {
        /// The symbol.
        symbol: String,
        /// What is wrong, in words.
        reason: String,
    },
    /// A value, factor or offset that is NaN or infinite.
    NotFinite {
        /// What it is (`value`, `factor`, `offset`).
        parameter: &'static str,
    },
    /// A definition's factor that is not `> 0`.
    NonPositiveFactor {
        /// The factor given.
        got: f64,
    },
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitError::Malformed { expression, reason } => {
                write!(f, "unit \"{expression}\": {reason}")
            }
            UnitError::UnknownSymbol { expression, symbol } => {
                write!(f, "unit \"{expression}\": unknown unit symbol \"{symbol}\"")
            }
            UnitError::OffsetCombined { expression, symbol } => write!(
                f,
                "unit \"{expression}\": \"{symbol}\" has an offset (a scale such as Cel) and \
                 must stand alone, without a prefix, an exponent or a product"
            ),
            UnitError::DimensionMismatch { from, to } => {
                write!(f, "cannot convert between dimensions {from} and {to}")
            }
            UnitError::DuplicateSymbol { symbol } => {
                write!(f, "\"{symbol}\" is already a unit")
            }
            UnitError::InvalidSymbol { symbol, reason } => {
                write!(f, "\"{symbol}\" cannot be a unit symbol: {reason}")
            }
            UnitError::NotFinite { parameter } => {
                write!(f, "{parameter} must be a finite number")
            }
            UnitError::NonPositiveFactor { got } => {
                write!(f, "factor must be > 0, got {got}")
            }
        }
    }
}

impl std::error::Error for UnitError {}

/// Decimal prefixes (SI 2022, Q … q), with `u` for micro as UCUM writes it.
const PREFIXES: [(&str, i32); 25] = [
    ("Q", 30),
    ("R", 27),
    ("Y", 24),
    ("Z", 21),
    ("E", 18),
    ("P", 15),
    ("T", 12),
    ("G", 9),
    ("M", 6),
    ("k", 3),
    ("h", 2),
    ("da", 1),
    ("d", -1),
    ("c", -2),
    ("m", -3),
    ("u", -6),
    ("n", -9),
    ("p", -12),
    ("f", -15),
    ("a", -18),
    ("z", -21),
    ("y", -24),
    ("r", -27),
    ("q", -30),
    ("μ", -6),
];

/// A known atom: the unit it stands for, and whether a prefix may go on it.
#[derive(Debug, Clone, Copy)]
struct Atom {
    unit: Unit,
    prefixable: bool,
}

/// A set of unit symbols: the SI, and whatever the caller defines.
///
/// Every operation reads expressions against this set; nothing is global.
#[derive(Debug, Clone)]
pub struct Units {
    atoms: HashMap<String, Atom>,
}

/// The characters that separate tokens in an expression.
fn is_operator(c: char) -> bool {
    matches!(c, '.' | '/' | '(' | ')' | '{' | '}') || c.is_whitespace()
}

impl Units {
    /// The SI units (see the module documentation).
    pub fn si() -> Self {
        let mut atoms = HashMap::new();
        let mut put = |symbol: &str, unit: Unit, prefixable: bool| {
            atoms.insert(symbol.to_string(), Atom { unit, prefixable });
        };
        let d = |e: [i32; 7]| Dimension(e);
        let coherent = |e: [i32; 7]| Unit {
            dimension: d(e),
            scale: 1.0,
            offset: 0.0,
        };
        let scaled = |e: [i32; 7], scale: f64| Unit {
            dimension: d(e),
            scale,
            offset: 0.0,
        };
        put(
            "m",
            Unit {
                dimension: Dimension::base(0),
                scale: 1.0,
                offset: 0.0,
            },
            true,
        );
        put(
            "g",
            Unit {
                dimension: Dimension::base(1),
                scale: 1e-3,
                offset: 0.0,
            },
            true,
        );
        put(
            "s",
            Unit {
                dimension: Dimension::base(2),
                scale: 1.0,
                offset: 0.0,
            },
            true,
        );
        put(
            "A",
            Unit {
                dimension: Dimension::base(3),
                scale: 1.0,
                offset: 0.0,
            },
            true,
        );
        put(
            "K",
            Unit {
                dimension: Dimension::base(4),
                scale: 1.0,
                offset: 0.0,
            },
            true,
        );
        put(
            "mol",
            Unit {
                dimension: Dimension::base(5),
                scale: 1.0,
                offset: 0.0,
            },
            true,
        );
        put(
            "cd",
            Unit {
                dimension: Dimension::base(6),
                scale: 1.0,
                offset: 0.0,
            },
            true,
        );
        put("rad", coherent([0, 0, 0, 0, 0, 0, 0]), true);
        put("sr", coherent([0, 0, 0, 0, 0, 0, 0]), true);
        put("Hz", coherent([0, 0, -1, 0, 0, 0, 0]), true);
        put("N", coherent([1, 1, -2, 0, 0, 0, 0]), true);
        put("Pa", coherent([-1, 1, -2, 0, 0, 0, 0]), true);
        put("J", coherent([2, 1, -2, 0, 0, 0, 0]), true);
        put("W", coherent([2, 1, -3, 0, 0, 0, 0]), true);
        put("C", coherent([0, 0, 1, 1, 0, 0, 0]), true);
        put("V", coherent([2, 1, -3, -1, 0, 0, 0]), true);
        put("F", coherent([-2, -1, 4, 2, 0, 0, 0]), true);
        put("Ohm", coherent([2, 1, -3, -2, 0, 0, 0]), true);
        put("S", coherent([-2, -1, 3, 2, 0, 0, 0]), true);
        put("Wb", coherent([2, 1, -2, -1, 0, 0, 0]), true);
        put("T", coherent([0, 1, -2, -1, 0, 0, 0]), true);
        put("H", coherent([2, 1, -2, -2, 0, 0, 0]), true);
        put("lm", coherent([0, 0, 0, 0, 0, 0, 1]), true);
        put("lx", coherent([-2, 0, 0, 0, 0, 0, 1]), true);
        put("Bq", coherent([0, 0, -1, 0, 0, 0, 0]), true);
        put("Gy", coherent([2, 0, -2, 0, 0, 0, 0]), true);
        put("Sv", coherent([2, 0, -2, 0, 0, 0, 0]), true);
        put("kat", coherent([0, 0, -1, 0, 0, 1, 0]), true);
        put(
            "Cel",
            Unit {
                dimension: Dimension::base(4),
                scale: 1.0,
                offset: 273.15,
            },
            false,
        );
        // Accepted for use with the SI (SI Brochure, Table 8) — exact multiples.
        put("min", scaled([0, 0, 1, 0, 0, 0, 0], 60.0), false);
        put("h", scaled([0, 0, 1, 0, 0, 0, 0], 3600.0), false);
        put("d", scaled([0, 0, 1, 0, 0, 0, 0], 86_400.0), false);
        put("L", scaled([3, 0, 0, 0, 0, 0, 0], 1e-3), true);
        put("l", scaled([3, 0, 0, 0, 0, 0, 0], 1e-3), true);
        put("t", scaled([0, 1, 0, 0, 0, 0, 0], 1e3), true);
        put("1", Unit::ONE, false);
        Units { atoms }
    }

    /// Defines `symbol` as `factor` × `expression` (+ `offset` SI units at
    /// its zero, for a scale): `define("cal_IT", 4.1868, "J", 0.0)`,
    /// `define("toe", 1e7, "kcal_IT", 0.0)`,
    /// `define("degF", 5.0 / 9.0, "K", 459.67 * 5.0 / 9.0)`.
    ///
    /// `expression` may use the built-in units and every symbol defined
    /// before. A defined unit takes decimal prefixes unless it has an offset.
    /// A symbol that reads as a prefix on an atom (`ft`, the femtotonne) is
    /// free to define: atoms are read before prefixes, so the definition wins.
    ///
    /// # Errors
    /// [`UnitError::InvalidSymbol`] for a symbol that is empty, starts with a
    /// digit or a sign, or holds an operator character (`.` `/` `(` `)` `{`
    /// `}` or a space); [`UnitError::DuplicateSymbol`] for one already
    /// defined (built-in or not); [`UnitError::NotFinite`] /
    /// [`UnitError::NonPositiveFactor`] for `factor` and `offset`; and the
    /// errors of [`Units::parse`] for `expression`.
    pub fn define(
        &mut self,
        symbol: &str,
        factor: f64,
        expression: &str,
        offset: f64,
    ) -> Result<(), UnitError> {
        let invalid = |reason: &str| UnitError::InvalidSymbol {
            symbol: symbol.to_string(),
            reason: reason.to_string(),
        };
        let first = symbol
            .chars()
            .next()
            .ok_or_else(|| invalid("it is empty"))?;
        if first.is_ascii_digit() || first == '+' || first == '-' {
            return Err(invalid("it starts with a digit or a sign"));
        }
        if symbol.chars().any(is_operator) {
            return Err(invalid(
                "it holds an operator character (. / ( ) { } or a space)",
            ));
        }
        // A prefix on an atom (`ft` = femtotonne) may be taken over: atoms
        // are read first, and `ft` as the foot is what a caller means.
        if self.atoms.contains_key(symbol) {
            return Err(UnitError::DuplicateSymbol {
                symbol: symbol.to_string(),
            });
        }
        if !factor.is_finite() {
            return Err(UnitError::NotFinite {
                parameter: "factor",
            });
        }
        if factor <= 0.0 {
            return Err(UnitError::NonPositiveFactor { got: factor });
        }
        if !offset.is_finite() {
            return Err(UnitError::NotFinite {
                parameter: "offset",
            });
        }
        let base = self.parse(expression)?;
        let unit = Unit {
            dimension: base.dimension,
            scale: factor * base.scale,
            offset: base.offset + offset,
        };
        self.atoms.insert(
            symbol.to_string(),
            Atom {
                unit,
                prefixable: unit.offset == 0.0,
            },
        );
        Ok(())
    }

    /// The unit `expression` stands for.
    ///
    /// # Errors
    /// [`UnitError::Malformed`], [`UnitError::UnknownSymbol`],
    /// [`UnitError::OffsetCombined`].
    pub fn parse(&self, expression: &str) -> Result<Unit, UnitError> {
        let mut parser = Parser {
            units: self,
            expression,
            chars: expression.char_indices().peekable(),
        };
        let (unit, offset_symbol) = parser.term()?;
        if let Some(&(_, c)) = parser.chars.peek() {
            return Err(parser.malformed(format!("unexpected \"{c}\"")));
        }
        if let OffsetUse::Combined(symbol) = offset_symbol {
            return Err(UnitError::OffsetCombined {
                expression: expression.to_string(),
                symbol,
            });
        }
        Ok(unit)
    }

    /// `value` read in `from`, in `to`: a reading, so a scale's offset
    /// applies (20 `Cel` is 293.15 `K`).
    ///
    /// # Errors
    /// [`UnitError::NotFinite`] for `value`; the errors of [`Units::parse`];
    /// [`UnitError::DimensionMismatch`] when the units measure different
    /// things.
    pub fn convert(&self, value: f64, from: &str, to: &str) -> Result<f64, UnitError> {
        self.parse(from)?.convert_to(value, &self.parse(to)?)
    }

    /// An amount of change `value` in `from`, in `to`: the offset cancels
    /// (a rise of 20 `Cel` is a rise of 20 `K`).
    ///
    /// # Errors
    /// As [`Units::convert`].
    pub fn convert_difference(&self, value: f64, from: &str, to: &str) -> Result<f64, UnitError> {
        self.parse(from)?.difference_to(value, &self.parse(to)?)
    }

    /// The atom `token` names, alone or after a prefix; `None` if neither.
    fn atom<'t>(&self, token: &'t str) -> Option<(Unit, Option<&'t str>)> {
        if let Some(a) = self.atoms.get(token) {
            return Some((a.unit, None));
        }
        PREFIXES.iter().find_map(|&(prefix, exp)| {
            let rest = token.strip_prefix(prefix)?;
            let a = self.atoms.get(rest)?;
            if !a.prefixable && a.unit.offset == 0.0 {
                return None;
            }
            Some((
                Unit {
                    dimension: a.unit.dimension,
                    scale: a.unit.scale * 10f64.powi(exp),
                    offset: a.unit.offset,
                },
                Some(rest),
            ))
        })
    }
}

/// Whether a parsed part involves an offset unit, and how.
#[derive(Debug, Clone, PartialEq)]
enum OffsetUse {
    None,
    /// The whole part is the offset unit, unmodified.
    Alone(String),
    /// An offset unit with a prefix, exponent, or product.
    Combined(String),
}

struct Parser<'a> {
    units: &'a Units,
    expression: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
}

impl Parser<'_> {
    fn malformed(&self, reason: String) -> UnitError {
        UnitError::Malformed {
            expression: self.expression.to_string(),
            reason,
        }
    }

    /// term := ['/'] component (('.' | '/') component)*
    fn term(&mut self) -> Result<(Unit, OffsetUse), UnitError> {
        let mut unit = Unit::ONE;
        let mut parts: Vec<OffsetUse> = Vec::new();
        let mut divide = false;
        if let Some(&(_, '/')) = self.chars.peek() {
            self.chars.next();
            divide = true;
        }
        let mut divided = divide;
        loop {
            let (next, offset) = self.component()?;
            unit = if parts.is_empty() && !divide {
                next
            } else {
                unit.times(if divide { next.pow(-1) } else { next })
            };
            parts.push(offset);
            match self.chars.peek() {
                Some(&(_, '.')) => divide = false,
                Some(&(_, '/')) => divide = true,
                _ => break,
            }
            divided |= divide;
            self.chars.next();
        }
        // An offset unit keeps its meaning only as the whole term.
        let offset = match parts.as_slice() {
            [OffsetUse::Alone(s)] if !divided => OffsetUse::Alone(s.clone()),
            _ => parts
                .into_iter()
                .find_map(|p| match p {
                    OffsetUse::Alone(s) | OffsetUse::Combined(s) => Some(OffsetUse::Combined(s)),
                    OffsetUse::None => None,
                })
                .unwrap_or(OffsetUse::None),
        };
        if !matches!(offset, OffsetUse::Alone(_)) {
            unit.offset = 0.0;
        }
        Ok((unit, offset))
    }

    /// component := '(' term ')' annotation? | token annotation? | annotation
    fn component(&mut self) -> Result<(Unit, OffsetUse), UnitError> {
        match self.chars.peek().copied() {
            None => Err(self.malformed("a unit is missing at the end".to_string())),
            Some((_, '(')) => {
                self.chars.next();
                let (unit, offset) = self.term()?;
                match self.chars.next() {
                    Some((_, ')')) => {}
                    _ => return Err(self.malformed("\"(\" is not closed".to_string())),
                }
                self.annotation()?;
                Ok((unit, offset))
            }
            Some((_, '{')) => {
                self.annotation()?;
                Ok((Unit::ONE, OffsetUse::None))
            }
            Some((_, c)) if is_operator(c) => {
                Err(self.malformed(format!("a unit is missing before \"{c}\"")))
            }
            Some((start, _)) => {
                let mut end = start;
                while let Some(&(i, c)) = self.chars.peek() {
                    if is_operator(c) {
                        break;
                    }
                    end = i + c.len_utf8();
                    self.chars.next();
                }
                let token = &self.expression[start..end];
                let result = self.token(token)?;
                self.annotation()?;
                Ok(result)
            }
        }
    }

    /// An optional `{…}` annotation, which has no effect on the unit.
    fn annotation(&mut self) -> Result<(), UnitError> {
        if let Some(&(_, '{')) = self.chars.peek() {
            self.chars.next();
            loop {
                match self.chars.next() {
                    Some((_, '}')) => return Ok(()),
                    Some((_, '{')) => {
                        return Err(self.malformed("an annotation holds \"{\"".to_string()))
                    }
                    Some(_) => {}
                    None => return Err(self.malformed("\"{\" is not closed".to_string())),
                }
            }
        }
        Ok(())
    }

    /// token := integer | atom exponent? — the whole token is tried as an
    /// atom before a trailing exponent is split off.
    fn token(&self, token: &str) -> Result<(Unit, OffsetUse), UnitError> {
        if token.bytes().all(|b| b.is_ascii_digit()) {
            let n: f64 = token
                .parse()
                .map_err(|_| self.malformed(format!("\"{token}\" is not a number")))?;
            if n <= 0.0 {
                return Err(self.malformed("a factor must be a positive integer".to_string()));
            }
            return Ok((
                Unit {
                    scale: n,
                    ..Unit::ONE
                },
                OffsetUse::None,
            ));
        }
        if let Some(found) = self.lookup(token) {
            return Ok(found);
        }
        let digits = token
            .char_indices()
            .rev()
            .take_while(|(_, c)| c.is_ascii_digit())
            .last()
            .map(|(i, _)| i);
        let Some(digits_at) = digits else {
            return Err(self.unknown(token));
        };
        let (symbol_and_sign, digits) = token.split_at(digits_at);
        let (symbol, sign) = match symbol_and_sign.strip_suffix(['+', '-']) {
            Some(s) => (s, &symbol_and_sign[s.len()..]),
            None => (symbol_and_sign, ""),
        };
        if symbol.is_empty() {
            return Err(self.malformed(format!("\"{token}\" has an exponent but no unit")));
        }
        let n: i32 = format!("{sign}{digits}")
            .parse()
            .ok()
            .filter(|n: &i32| n.unsigned_abs() <= 64)
            .ok_or_else(|| self.malformed(format!("exponent \"{sign}{digits}\" is too large")))?;
        let (unit, offset) = self.lookup(symbol).ok_or_else(|| self.unknown(symbol))?;
        let offset = match offset {
            OffsetUse::Alone(s) if n != 1 => OffsetUse::Combined(s),
            other => other,
        };
        Ok((unit.pow(n), offset))
    }

    fn lookup(&self, symbol: &str) -> Option<(Unit, OffsetUse)> {
        let (unit, prefixed) = self.units.atom(symbol)?;
        let offset = match (unit.offset != 0.0, prefixed) {
            (false, _) => OffsetUse::None,
            (true, None) => OffsetUse::Alone(symbol.to_string()),
            (true, Some(atom)) => OffsetUse::Combined(atom.to_string()),
        };
        Some((unit, offset))
    }

    fn unknown(&self, symbol: &str) -> UnitError {
        UnitError::UnknownSymbol {
            expression: self.expression.to_string(),
            symbol: symbol.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * b.abs().max(1.0)
    }

    /// The units a district-heating or energy report needs, defined as data.
    fn energy() -> Units {
        let mut u = Units::si();
        u.define("bar", 1e5, "Pa", 0.0).unwrap();
        u.define("cal_IT", 4.1868, "J", 0.0).unwrap();
        u.define("cal_th", 4.184, "J", 0.0).unwrap();
        u.define("toe", 1e7, "kcal_IT", 0.0).unwrap();
        u.define("Nm3", 1.0, "m3", 0.0).unwrap();
        u
    }

    // ── the behaviours the request lists ──

    #[test]
    fn one_bar_is_one_hundred_kilopascal() {
        assert!(close(energy().convert(1.0, "bar", "kPa").unwrap(), 100.0));
    }

    #[test]
    fn a_celsius_reading_converts_with_the_offset_and_a_difference_without() {
        let u = Units::si();
        assert!(close(u.convert(20.0, "Cel", "K").unwrap(), 293.15));
        assert!(close(u.convert_difference(20.0, "Cel", "K").unwrap(), 20.0));
        assert!(close(u.convert(300.0, "K", "Cel").unwrap(), 26.85));
    }

    #[test]
    fn a_kilowatt_hour_is_three_point_six_megajoule() {
        assert!(close(Units::si().convert(1.0, "kW.h", "J").unwrap(), 3.6e6));
        assert!(close(Units::si().convert(1.0, "kW.h", "MJ").unwrap(), 3.6));
    }

    #[test]
    fn the_calorie_is_the_one_the_caller_defined() {
        let u = energy();
        assert!(close(u.convert(1.0, "kcal_IT", "J").unwrap(), 4186.8));
        assert!(close(u.convert(1.0, "kcal_th", "J").unwrap(), 4184.0));
        assert!(close(u.convert(1.0, "Gcal_IT", "GJ").unwrap(), 4.1868));
        assert!(close(u.convert(1.0, "toe", "GJ").unwrap(), 41.868));
    }

    #[test]
    fn volume_is_not_mass() {
        assert_eq!(
            energy().convert(1.0, "m3", "kg"),
            Err(UnitError::DimensionMismatch {
                from: Dimension([3, 0, 0, 0, 0, 0, 0]),
                to: Dimension([0, 1, 0, 0, 0, 0, 0]),
            })
        );
    }

    // ── syntax ──

    #[test]
    fn products_and_quotients_read_left_to_right() {
        let u = Units::si();
        assert!(close(
            u.convert(1.0, "km/h", "m/s").unwrap(),
            1000.0 / 3600.0
        ));
        // kg/m.s = (kg/m)·s, as UCUM reads it.
        assert_eq!(
            u.parse("kg/m.s").unwrap().dimension,
            Dimension([-1, 1, 1, 0, 0, 0, 0])
        );
        assert_eq!(
            u.parse("kg/(m.s)").unwrap().dimension,
            Dimension([-1, 1, -1, 0, 0, 0, 0])
        );
        assert_eq!(
            u.parse("/s").unwrap().dimension,
            u.parse("Hz").unwrap().dimension
        );
        assert!(close(u.convert(1.0, "m/s2", "N/kg").unwrap(), 1.0));
        assert!(close(u.convert(1.0, "J/(kg.K)", "m2/(s2.K)").unwrap(), 1.0));
    }

    #[test]
    fn exponents_are_signed_integers() {
        let u = Units::si();
        assert!(close(u.convert(1.0, "m3", "L").unwrap(), 1000.0));
        assert!(close(u.convert(1.0, "cm3", "mL").unwrap(), 1.0));
        assert!(close(u.convert(2.0, "s-1", "Hz").unwrap(), 2.0));
        assert!(close(u.convert(1.0, "m+2", "dm2").unwrap(), 100.0));
    }

    #[test]
    fn integers_are_factors_and_one_is_unity() {
        let u = Units::si();
        assert!(close(u.convert(1.0, "1000.m", "km").unwrap(), 1.0));
        assert!(u.parse("1").unwrap().dimension.is_none());
        assert!(close(u.convert(1.0, "rad", "1").unwrap(), 1.0));
    }

    #[test]
    fn annotations_have_no_effect() {
        let u = energy();
        assert!(close(u.convert(5.0, "m3{gas}", "m3").unwrap(), 5.0));
        assert!(close(u.convert(5.0, "{count}/min", "/h").unwrap(), 300.0));
    }

    #[test]
    fn an_atom_wins_over_a_prefix_split_and_a_symbol_ending_in_digits_stays_whole() {
        let u = energy();
        assert_eq!(u.parse("cd").unwrap().dimension, Dimension::base(6)); // candela
        assert!(close(u.convert(1.0, "Nm3", "m3").unwrap(), 1.0));
        assert!(close(u.convert(1.0, "Nm3", "Nm3").unwrap(), 1.0));
        assert!(close(u.convert(1.0, "min", "s").unwrap(), 60.0)); // not milli-inch
        assert!(close(u.convert(1.0, "mmol", "mol").unwrap(), 1e-3));
        assert!(close(u.convert(1.0, "kg", "g").unwrap(), 1e3));
        assert!(close(u.convert(1.0, "dam", "m").unwrap(), 10.0));
        assert!(close(u.convert(1.0, "um", "m").unwrap(), 1e-6));
        assert!(close(u.convert(1.0, "μm", "m").unwrap(), 1e-6));
    }

    #[test]
    fn hours_and_days_take_no_prefix() {
        // "kh" would be a kilohour; the SI does not prefix the hour.
        assert!(matches!(
            Units::si().parse("kh"),
            Err(UnitError::UnknownSymbol { .. })
        ));
    }

    // ── refusals ──

    #[test]
    fn an_offset_unit_stands_alone() {
        let u = Units::si();
        for e in ["Cel2", "kCel", "Cel/s", "m.Cel", "(Cel)", "/Cel"] {
            let r = u.parse(e);
            if e == "(Cel)" {
                // A parenthesized lone unit is still the unit alone.
                assert!(r.is_ok(), "{e}: {r:?}");
                continue;
            }
            assert!(
                matches!(r, Err(UnitError::OffsetCombined { ref symbol, .. }) if symbol == "Cel"),
                "{e}: {r:?}"
            );
        }
    }

    #[test]
    fn a_malformed_expression_is_refused() {
        let u = Units::si();
        for e in ["", "m.", "m//s", "(m", "m)", "m{x", ".m", "m s", "m99999"] {
            assert!(
                matches!(u.parse(e), Err(UnitError::Malformed { .. })),
                "{e:?}: {:?}",
                u.parse(e)
            );
        }
    }

    #[test]
    fn an_unknown_symbol_is_named() {
        assert_eq!(
            Units::si().parse("kW.hr"),
            Err(UnitError::UnknownSymbol {
                expression: "kW.hr".into(),
                symbol: "hr".into()
            })
        );
        // The exponent is split off before the symbol is named.
        assert!(matches!(
            Units::si().parse("yd2"),
            Err(UnitError::UnknownSymbol { ref symbol, .. }) if symbol == "yd"
        ));
        // A prefix on an atom is a unit even where a reader expects another:
        // `ft` is the femtotonne — imperial units are the caller's to define.
        assert!(Units::si().parse("ft").is_ok());
    }

    #[test]
    fn a_definition_is_checked() {
        let mut u = Units::si();
        assert_eq!(
            u.define("J", 1.0, "J", 0.0),
            Err(UnitError::DuplicateSymbol { symbol: "J".into() })
        );
        // A prefixed atom is not an atom: the foot takes over the femtotonne.
        u.define("ft", 0.3048, "m", 0.0).unwrap();
        assert!(close(u.convert(1.0, "ft2", "m2").unwrap(), 0.3048 * 0.3048));
        for bad in ["", "2x", "-x", "a.b", "a b", "a/b", "{a}"] {
            assert!(
                matches!(
                    u.define(bad, 1.0, "m", 0.0),
                    Err(UnitError::InvalidSymbol { .. })
                ),
                "{bad:?}"
            );
        }
        assert_eq!(
            u.define("x", 0.0, "m", 0.0),
            Err(UnitError::NonPositiveFactor { got: 0.0 })
        );
        assert_eq!(
            u.define("x", f64::NAN, "m", 0.0),
            Err(UnitError::NotFinite {
                parameter: "factor"
            })
        );
        assert!(matches!(
            u.define("x", 1.0, "later", 0.0),
            Err(UnitError::UnknownSymbol { .. })
        ));
    }

    #[test]
    fn a_defined_scale_converts_readings() {
        let mut u = Units::si();
        u.define("degF", 5.0 / 9.0, "K", 459.67 * 5.0 / 9.0)
            .unwrap();
        assert!(close(u.convert(32.0, "degF", "Cel").unwrap(), 0.0));
        assert!(close(u.convert(212.0, "degF", "Cel").unwrap(), 100.0));
        assert!(close(u.convert_difference(9.0, "degF", "K").unwrap(), 5.0));
        assert!(matches!(
            u.parse("kdegF"),
            Err(UnitError::OffsetCombined { .. })
        ));
    }

    #[test]
    fn a_value_must_be_finite() {
        assert_eq!(
            Units::si().convert(f64::INFINITY, "m", "m"),
            Err(UnitError::NotFinite { parameter: "value" })
        );
    }

    #[test]
    fn a_dimension_is_written_in_base_units() {
        let u = Units::si();
        assert_eq!(u.parse("J").unwrap().dimension.to_string(), "kg.m2.s-2");
        assert_eq!(u.parse("rad").unwrap().dimension.to_string(), "1");
        assert_eq!(
            u.parse("Ohm").unwrap().dimension.to_string(),
            "kg.m2.s-3.A-2"
        );
    }

    #[test]
    fn round_trips_are_exact_to_rounding() {
        let u = energy();
        for (a, b) in [("kW.h", "GJ"), ("Cel", "K"), ("bar", "Pa"), ("toe", "MW.h")] {
            let x = 123.456;
            let back = u.convert(u.convert(x, a, b).unwrap(), b, a).unwrap();
            assert!(close(back, x), "{a} ↔ {b}: {back}");
        }
    }
}
