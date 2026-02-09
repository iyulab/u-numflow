//! Probability distributions.
//!
//! Domain-agnostic probability distribution types with analytical
//! moments (mean, variance) and CDF/inverse-CDF evaluation.
//!
//! # Supported Distributions
//!
//! | Distribution | Parameters | Mean | Variance |
//! |---|---|---|---|
//! | [`Uniform`] | min, max | (a+b)/2 | (b−a)²/12 |
//! | [`Triangular`] | min, mode, max | (a+b+c)/3 | (a²+b²+c²−ab−ac−bc)/18 |
//! | [`Normal`] | μ, σ | μ | σ² |
//! | [`LogNormal`] | μ, σ | exp(μ+σ²/2) | (exp(σ²)−1)·exp(2μ+σ²) |
//! | [`Pert`] | min, mode, max | (a+4m+b)/6 | see docs |
//! | [`Weibull`] | shape (β), scale (η) | η·Γ(1+1/β) | η²·[Γ(1+2/β)−Γ(1+1/β)²] |
//! | [`Exponential`] | rate (λ) | 1/λ | 1/λ² |
//! | [`GammaDistribution`] | shape (α), rate (β) | α/β | α/β² |
//! | [`BetaDistribution`] | α, β | α/(α+β) | αβ/((α+β)²(α+β+1)) |
//! | [`ChiSquared`] | k (degrees of freedom) | k | 2k |
//!
//! # Design Notes
//!
//! This module is **domain-agnostic**. There is no concept of "duration",
//! "scheduling", or any consumer domain. Parameters are plain `f64` values.

use crate::special;

/// Error type for invalid distribution parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionError {
    /// Parameters violate distribution constraints.
    InvalidParameters(String),
}

impl std::fmt::Display for DistributionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistributionError::InvalidParameters(msg) => {
                write!(f, "invalid distribution parameters: {msg}")
            }
        }
    }
}

impl std::error::Error for DistributionError {}

// ============================================================================
// Uniform Distribution
// ============================================================================

/// Continuous uniform distribution on `[min, max]`.
///
/// # Mathematical Definition
/// - PDF: f(x) = 1/(max−min) for x ∈ [min, max]
/// - CDF: F(x) = (x−min)/(max−min)
/// - Mean: (min+max)/2
/// - Variance: (max−min)²/12
#[derive(Debug, Clone, PartialEq)]
pub struct Uniform {
    min: f64,
    max: f64,
}

impl Uniform {
    /// Creates a new uniform distribution on `[min, max]`.
    ///
    /// # Errors
    /// Returns `Err` if `min >= max` or either parameter is not finite.
    pub fn new(min: f64, max: f64) -> Result<Self, DistributionError> {
        if !min.is_finite() || !max.is_finite() || min >= max {
            return Err(DistributionError::InvalidParameters(format!(
                "Uniform requires min < max, got min={min}, max={max}"
            )));
        }
        Ok(Self { min, max })
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    pub fn mean(&self) -> f64 {
        (self.min + self.max) / 2.0
    }

    pub fn variance(&self) -> f64 {
        let range = self.max - self.min;
        range * range / 12.0
    }

    /// CDF: F(x) = (x−min)/(max−min), clamped to [0, 1].
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= self.min {
            0.0
        } else if x >= self.max {
            1.0
        } else {
            (x - self.min) / (self.max - self.min)
        }
    }

    /// Inverse CDF (quantile function): x = min + p·(max−min).
    ///
    /// Returns `None` if `p` is outside `[0, 1]`.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }
        Some(self.min + p * (self.max - self.min))
    }

    /// PDF: f(x) = 1/(max−min) for x ∈ [min, max], 0 otherwise.
    pub fn pdf(&self, x: f64) -> f64 {
        if x >= self.min && x <= self.max {
            1.0 / (self.max - self.min)
        } else {
            0.0
        }
    }
}

// ============================================================================
// Triangular Distribution
// ============================================================================

/// Triangular distribution with parameters `[min, mode, max]`.
///
/// # Mathematical Definition
/// - PDF: piecewise linear, peaking at mode
/// - CDF: piecewise quadratic
/// - Mean: (min + mode + max) / 3
/// - Variance: (a² + b² + c² − ab − ac − bc) / 18
///
/// Reference: Johnson, Kotz & Balakrishnan (1995), *Continuous Univariate
/// Distributions*, Vol. 2, Chapter 26.
#[derive(Debug, Clone, PartialEq)]
pub struct Triangular {
    min: f64,
    mode: f64,
    max: f64,
}

impl Triangular {
    /// Creates a new triangular distribution.
    ///
    /// # Errors
    /// Returns `Err` if `min >= max` or `mode` is outside `[min, max]`.
    pub fn new(min: f64, mode: f64, max: f64) -> Result<Self, DistributionError> {
        if !min.is_finite() || !mode.is_finite() || !max.is_finite() {
            return Err(DistributionError::InvalidParameters(
                "Triangular parameters must be finite".into(),
            ));
        }
        if min > mode || mode > max || min >= max {
            return Err(DistributionError::InvalidParameters(format!(
                "Triangular requires min ≤ mode ≤ max and min < max, got {min}, {mode}, {max}"
            )));
        }
        Ok(Self { min, mode, max })
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn mode(&self) -> f64 {
        self.mode
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    /// Mean = (min + mode + max) / 3.
    pub fn mean(&self) -> f64 {
        (self.min + self.mode + self.max) / 3.0
    }

    /// Variance = (a² + b² + c² − ab − ac − bc) / 18.
    pub fn variance(&self) -> f64 {
        let (a, b, c) = (self.min, self.mode, self.max);
        (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
    }

    /// PDF of the triangular distribution.
    ///
    /// ```text
    /// f(x) = 2(x−a) / ((c−a)(b−a))  for a ≤ x ≤ b
    ///      = 2(c−x) / ((c−a)(c−b))  for b < x ≤ c
    ///      = 0                       otherwise
    /// ```
    pub fn pdf(&self, x: f64) -> f64 {
        let (a, b, c) = (self.min, self.mode, self.max);
        if x < a || x > c {
            0.0
        } else if x <= b {
            2.0 * (x - a) / ((c - a) * (b - a).max(f64::MIN_POSITIVE))
        } else {
            2.0 * (c - x) / ((c - a) * (c - b).max(f64::MIN_POSITIVE))
        }
    }

    /// CDF of the triangular distribution.
    ///
    /// ```text
    /// F(x) = (x−a)² / ((c−a)(b−a))       for a ≤ x ≤ b
    ///      = 1 − (c−x)² / ((c−a)(c−b))   for b < x ≤ c
    /// ```
    pub fn cdf(&self, x: f64) -> f64 {
        let (a, b, c) = (self.min, self.mode, self.max);
        if x <= a {
            0.0
        } else if x <= b {
            (x - a) * (x - a) / ((c - a) * (b - a).max(f64::MIN_POSITIVE))
        } else if x < c {
            1.0 - (c - x) * (c - x) / ((c - a) * (c - b).max(f64::MIN_POSITIVE))
        } else {
            1.0
        }
    }

    /// Inverse CDF (quantile function) of the triangular distribution.
    ///
    /// ```text
    /// F⁻¹(p) = a + √(p·(c−a)·(b−a))                 if p < F(b)
    ///        = c − √((1−p)·(c−a)·(c−b))              if p ≥ F(b)
    /// ```
    ///
    /// Returns `None` if `p` is outside `[0, 1]`.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }
        let (a, b, c) = (self.min, self.mode, self.max);
        let fc = (b - a) / (c - a); // CDF at the mode
        if p < fc {
            Some(a + ((c - a) * (b - a) * p).sqrt())
        } else {
            Some(c - ((c - a) * (c - b) * (1.0 - p)).sqrt())
        }
    }
}

// ============================================================================
// Normal Distribution
// ============================================================================

/// Normal (Gaussian) distribution N(μ, σ²).
///
/// # Mathematical Definition
/// - PDF: φ(x) = (1/(σ√(2π))) exp(−(x−μ)²/(2σ²))
/// - CDF: Φ((x−μ)/σ) (via standard normal CDF)
/// - Mean: μ
/// - Variance: σ²
#[derive(Debug, Clone, PartialEq)]
pub struct Normal {
    mu: f64,
    sigma: f64,
}

impl Normal {
    /// Creates a new normal distribution N(μ, σ).
    ///
    /// # Errors
    /// Returns `Err` if `sigma ≤ 0` or parameters are not finite.
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistributionError> {
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(DistributionError::InvalidParameters(format!(
                "Normal requires finite μ and σ > 0, got μ={mu}, σ={sigma}"
            )));
        }
        Ok(Self { mu, sigma })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    pub fn mean(&self) -> f64 {
        self.mu
    }

    pub fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }

    pub fn std_dev(&self) -> f64 {
        self.sigma
    }

    /// PDF: (1/(σ√(2π))) exp(−(x−μ)²/(2σ²)).
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        special::standard_normal_pdf(z) / self.sigma
    }

    /// CDF: Φ((x−μ)/σ).
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        special::standard_normal_cdf(z)
    }

    /// Inverse CDF (quantile): μ + σ·Φ⁻¹(p).
    ///
    /// Returns `None` if `p` is outside `(0, 1)`.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if p <= 0.0 || p >= 1.0 {
            return None;
        }
        Some(self.mu + self.sigma * special::inverse_normal_cdf(p))
    }
}

// ============================================================================
// LogNormal Distribution
// ============================================================================

/// Log-normal distribution: if X ~ LogNormal(μ, σ), then ln(X) ~ N(μ, σ²).
///
/// # Mathematical Definition
/// - PDF: (1/(xσ√(2π))) exp(−(ln(x)−μ)²/(2σ²)) for x > 0
/// - CDF: Φ((ln(x)−μ)/σ)
/// - Mean: exp(μ + σ²/2)
/// - Variance: (exp(σ²) − 1) · exp(2μ + σ²)
///
/// Reference: Johnson, Kotz & Balakrishnan (1994), *Continuous Univariate
/// Distributions*, Vol. 1, Chapter 14.
#[derive(Debug, Clone, PartialEq)]
pub struct LogNormal {
    mu: f64,
    sigma: f64,
}

impl LogNormal {
    /// Creates a new log-normal distribution.
    ///
    /// Parameters `mu` and `sigma` are the mean and std dev of ln(X).
    ///
    /// # Errors
    /// Returns `Err` if `sigma ≤ 0` or parameters are not finite.
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistributionError> {
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(DistributionError::InvalidParameters(format!(
                "LogNormal requires finite μ and σ > 0, got μ={mu}, σ={sigma}"
            )));
        }
        Ok(Self { mu, sigma })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Mean = exp(μ + σ²/2).
    pub fn mean(&self) -> f64 {
        (self.mu + self.sigma * self.sigma / 2.0).exp()
    }

    /// Variance = (exp(σ²) − 1) · exp(2μ + σ²).
    pub fn variance(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        (s2.exp() - 1.0) * (2.0 * self.mu + s2).exp()
    }

    /// PDF for x > 0.
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let ln_x = x.ln();
        let z = (ln_x - self.mu) / self.sigma;
        special::standard_normal_pdf(z) / (x * self.sigma)
    }

    /// CDF: Φ((ln(x)−μ)/σ) for x > 0.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = (x.ln() - self.mu) / self.sigma;
        special::standard_normal_cdf(z)
    }

    /// Inverse CDF: exp(μ + σ·Φ⁻¹(p)).
    ///
    /// Returns `None` if `p` is outside `(0, 1)`.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if p <= 0.0 || p >= 1.0 {
            return None;
        }
        Some((self.mu + self.sigma * special::inverse_normal_cdf(p)).exp())
    }
}

// ============================================================================
// PERT Distribution (Modified Beta)
// ============================================================================

/// PERT distribution (Program Evaluation and Review Technique).
///
/// A modified Beta distribution defined by three points: optimistic (min),
/// most likely (mode), and pessimistic (max).
///
/// # Mathematical Definition
///
/// Shape parameters (with λ = 4):
/// ```text
/// α = 1 + λ · (mode − min) / (max − min)
/// β = 1 + λ · (max − mode) / (max − min)
/// ```
///
/// The underlying variable Y = (X − min)/(max − min) follows Beta(α, β).
///
/// - Mean: (min + λ·mode + max) / (λ + 2) = (min + 4·mode + max) / 6
/// - Std Dev (simplified): (max − min) / (λ + 2) = (max − min) / 6
///
/// # Exact vs Simplified Variance
///
/// The simplified variance `((max−min)/6)²` is an approximation. The exact
/// variance uses the Beta distribution formula:
/// ```text
/// Var = α·β / ((α+β)²·(α+β+1)) × (max−min)²
/// ```
///
/// The simplified formula is exact when the distribution is symmetric
/// (mode = midpoint) and becomes less accurate as skewness increases.
///
/// Reference: Malcolm et al. (1959), "Application of a Technique for
/// Research and Development Program Evaluation", *Operations Research* 7(5).
#[derive(Debug, Clone, PartialEq)]
pub struct Pert {
    min: f64,
    mode: f64,
    max: f64,
    alpha: f64,
    beta: f64,
}

impl Pert {
    /// Creates a standard PERT distribution (λ = 4).
    ///
    /// # Errors
    /// Returns `Err` if `min >= max` or `mode` is outside `[min, max]`.
    pub fn new(min: f64, mode: f64, max: f64) -> Result<Self, DistributionError> {
        Self::with_shape(min, mode, max, 4.0)
    }

    /// Creates a modified PERT distribution with custom shape parameter λ.
    ///
    /// λ controls the weight of the mode:
    /// - λ = 4: standard PERT
    /// - λ > 4: tighter distribution (more peaked)
    /// - λ < 4: flatter distribution (less peaked)
    ///
    /// # Errors
    /// Returns `Err` if parameters are invalid.
    pub fn with_shape(
        min: f64,
        mode: f64,
        max: f64,
        lambda: f64,
    ) -> Result<Self, DistributionError> {
        if !min.is_finite() || !mode.is_finite() || !max.is_finite() || !lambda.is_finite() {
            return Err(DistributionError::InvalidParameters(
                "PERT parameters must be finite".into(),
            ));
        }
        if min > mode || mode > max || min >= max {
            return Err(DistributionError::InvalidParameters(format!(
                "PERT requires min ≤ mode ≤ max and min < max, got {min}, {mode}, {max}"
            )));
        }
        if lambda <= 0.0 {
            return Err(DistributionError::InvalidParameters(format!(
                "PERT λ must be > 0, got {lambda}"
            )));
        }

        let range = max - min;
        let alpha = 1.0 + lambda * (mode - min) / range;
        let beta = 1.0 + lambda * (max - mode) / range;

        Ok(Self {
            min,
            mode,
            max,
            alpha,
            beta,
        })
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn mode(&self) -> f64 {
        self.mode
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn beta_param(&self) -> f64 {
        self.beta
    }

    /// Mean = (min + 4·mode + max) / 6 for standard PERT (λ=4).
    ///
    /// General: (min + λ·mode + max) / (λ + 2).
    pub fn mean(&self) -> f64 {
        let ab = self.alpha + self.beta;
        // Mean of Beta(α,β) on [min,max] = min + (α/(α+β))·(max-min)
        self.min + (self.alpha / ab) * (self.max - self.min)
    }

    /// Exact variance using Beta distribution formula.
    ///
    /// ```text
    /// Var = α·β / ((α+β)²·(α+β+1)) × (max−min)²
    /// ```
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        let range = self.max - self.min;
        (self.alpha * self.beta) / (ab * ab * (ab + 1.0)) * range * range
    }

    /// Standard deviation = √(variance).
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// CDF via regularized incomplete beta function approximation.
    ///
    /// Uses a numerical approximation of the regularized incomplete beta
    /// function I_x(α, β).
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= self.min {
            return 0.0;
        }
        if x >= self.max {
            return 1.0;
        }
        let t = (x - self.min) / (self.max - self.min);
        regularized_incomplete_beta(t, self.alpha, self.beta)
    }

    /// Approximate quantile using normal approximation.
    ///
    /// Uses `μ + σ·Φ⁻¹(p)` with the exact PERT mean and std dev.
    /// This is an approximation; accuracy decreases for highly skewed PERTs.
    ///
    /// Returns `None` if `p` is outside `(0, 1)`.
    pub fn quantile_approx(&self, p: f64) -> Option<f64> {
        if p <= 0.0 || p >= 1.0 {
            return None;
        }
        let z = special::inverse_normal_cdf(p);
        let result = self.mean() + z * self.std_dev();
        // Clamp to [min, max]
        Some(result.clamp(self.min, self.max))
    }
}

// ============================================================================
// Weibull Distribution
// ============================================================================

/// Weibull distribution with shape parameter β (> 0) and scale parameter η (> 0).
///
/// Widely used in reliability engineering and survival analysis.
///
/// # Mathematical Definition
/// - PDF: f(t) = (β/η)·(t/η)^(β−1)·exp(−(t/η)^β) for t ≥ 0
/// - CDF: F(t) = 1 − exp(−(t/η)^β)
/// - Quantile: t = η·(−ln(1−p))^(1/β)
/// - Mean: η·Γ(1 + 1/β)
/// - Variance: η²·[Γ(1 + 2/β) − Γ(1 + 1/β)²]
///
/// # Special Cases
/// - β = 1: Exponential distribution with rate 1/η
/// - β = 2: Rayleigh distribution
/// - β ≈ 3.6: Approximates a normal distribution
///
/// Reference: Weibull (1951), "A Statistical Distribution Function of Wide
/// Applicability", *Journal of Applied Mechanics* 18(3), pp. 293–297.
#[derive(Debug, Clone, PartialEq)]
pub struct Weibull {
    shape: f64, // β
    scale: f64, // η
}

impl Weibull {
    /// Creates a new Weibull distribution with the given shape (β) and scale (η).
    ///
    /// # Errors
    /// Returns `Err` if `shape ≤ 0`, `scale ≤ 0`, or either is not finite.
    pub fn new(shape: f64, scale: f64) -> Result<Self, DistributionError> {
        if !shape.is_finite() || !scale.is_finite() || shape <= 0.0 || scale <= 0.0 {
            return Err(DistributionError::InvalidParameters(format!(
                "Weibull requires shape > 0 and scale > 0, got shape={shape}, scale={scale}"
            )));
        }
        Ok(Self { shape, scale })
    }

    /// Shape parameter β.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Scale parameter η.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Mean = η·Γ(1 + 1/β).
    pub fn mean(&self) -> f64 {
        self.scale * gamma(1.0 + 1.0 / self.shape)
    }

    /// Variance = η²·[Γ(1 + 2/β) − Γ(1 + 1/β)²].
    pub fn variance(&self) -> f64 {
        let g1 = gamma(1.0 + 1.0 / self.shape);
        let g2 = gamma(1.0 + 2.0 / self.shape);
        self.scale * self.scale * (g2 - g1 * g1)
    }

    /// PDF: f(t) = (β/η)·(t/η)^(β−1)·exp(−(t/η)^β) for t ≥ 0.
    pub fn pdf(&self, t: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }
        if t == 0.0 {
            return if self.shape < 1.0 {
                f64::INFINITY
            } else if (self.shape - 1.0).abs() < f64::EPSILON {
                1.0 / self.scale
            } else {
                0.0
            };
        }
        let z = t / self.scale;
        (self.shape / self.scale) * z.powf(self.shape - 1.0) * (-z.powf(self.shape)).exp()
    }

    /// CDF: F(t) = 1 − exp(−(t/η)^β).
    pub fn cdf(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let z = t / self.scale;
        1.0 - (-z.powf(self.shape)).exp()
    }

    /// Quantile (inverse CDF): t = η·(−ln(1−p))^(1/β).
    ///
    /// Returns `None` if `p` is outside `[0, 1)`.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if !(0.0..1.0).contains(&p) {
            return None;
        }
        if p == 0.0 {
            return Some(0.0);
        }
        Some(self.scale * (-(1.0 - p).ln()).powf(1.0 / self.shape))
    }

    /// Hazard rate (failure rate): λ(t) = (β/η)·(t/η)^(β−1).
    ///
    /// - β < 1: Decreasing failure rate (infant mortality)
    /// - β = 1: Constant failure rate (random failures)
    /// - β > 1: Increasing failure rate (wear-out)
    pub fn hazard_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return if t == 0.0 && self.shape >= 1.0 {
                if (self.shape - 1.0).abs() < f64::EPSILON {
                    1.0 / self.scale
                } else {
                    0.0
                }
            } else {
                0.0
            };
        }
        let z = t / self.scale;
        (self.shape / self.scale) * z.powf(self.shape - 1.0)
    }

    /// Reliability (survival) function: R(t) = 1 − F(t) = exp(−(t/η)^β).
    pub fn reliability(&self, t: f64) -> f64 {
        1.0 - self.cdf(t)
    }
}

// ============================================================================
// Exponential Distribution
// ============================================================================

/// Exponential distribution with rate parameter λ.
///
/// Models the time between events in a Poisson process. The rate λ > 0
/// determines the expected frequency; the mean is 1/λ.
///
/// # Examples
///
/// ```
/// use u_numflow::distributions::Exponential;
///
/// let exp = Exponential::new(0.5).unwrap();
/// assert!((exp.mean() - 2.0).abs() < 1e-10);
/// assert!((exp.cdf(2.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    rate: f64,
}

impl Exponential {
    /// Creates a new Exponential distribution with the given rate λ.
    ///
    /// Returns an error if λ ≤ 0 or λ is not finite.
    pub fn new(rate: f64) -> Result<Self, DistributionError> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(DistributionError::InvalidParameters(format!(
                "Exponential rate must be positive and finite, got {rate}"
            )));
        }
        Ok(Self { rate })
    }

    /// Rate parameter λ.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Mean = 1/λ.
    pub fn mean(&self) -> f64 {
        1.0 / self.rate
    }

    /// Variance = 1/λ².
    pub fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }

    /// PDF: f(x) = λ exp(−λx) for x ≥ 0, 0 otherwise.
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate * (-self.rate * x).exp()
        }
    }

    /// CDF: F(x) = 1 − exp(−λx) for x ≥ 0, 0 otherwise.
    pub fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-self.rate * x).exp()
        }
    }

    /// Quantile (inverse CDF): x = −ln(1−p)/λ.
    ///
    /// Returns `None` if p is not in [0, 1].
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }
        if p == 1.0 {
            return Some(f64::INFINITY);
        }
        Some(-(1.0 - p).ln() / self.rate)
    }

    /// Hazard rate (failure rate): h(x) = λ (constant).
    pub fn hazard_rate(&self) -> f64 {
        self.rate
    }

    /// Survival function: S(x) = exp(−λx).
    pub fn survival(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }
}

// ============================================================================
// Gamma Distribution
// ============================================================================

/// Gamma distribution with shape α and rate β (or equivalently scale θ = 1/β).
///
/// Uses the (shape, rate) parametrization: X ~ Gamma(α, β).
///
/// # PDF
///
/// f(x) = β^α / Γ(α) · x^(α−1) · exp(−βx) for x > 0
///
/// # Examples
///
/// ```
/// use u_numflow::distributions::GammaDistribution;
///
/// let g = GammaDistribution::new(2.0, 1.0).unwrap();
/// assert!((g.mean() - 2.0).abs() < 1e-10);
/// assert!((g.variance() - 2.0).abs() < 1e-10);
/// ```
///
/// # Reference
///
/// Johnson, N.L., Kotz, S. & Balakrishnan, N. (1994). *Continuous Univariate
/// Distributions*, Vol. 1, 2nd ed. Wiley.
#[derive(Debug, Clone, Copy)]
pub struct GammaDistribution {
    shape: f64, // α > 0
    rate: f64,  // β > 0
}

impl GammaDistribution {
    /// Creates a Gamma(α, β) distribution with shape α and rate β.
    ///
    /// Returns an error if α ≤ 0, β ≤ 0, or either is not finite.
    pub fn new(shape: f64, rate: f64) -> Result<Self, DistributionError> {
        if !shape.is_finite() || shape <= 0.0 || !rate.is_finite() || rate <= 0.0 {
            return Err(DistributionError::InvalidParameters(format!(
                "Gamma shape and rate must be positive and finite, got shape={shape}, rate={rate}"
            )));
        }
        Ok(Self { shape, rate })
    }

    /// Creates a Gamma distribution from shape and scale θ = 1/β.
    pub fn from_shape_scale(shape: f64, scale: f64) -> Result<Self, DistributionError> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(DistributionError::InvalidParameters(format!(
                "Gamma scale must be positive and finite, got {scale}"
            )));
        }
        Self::new(shape, 1.0 / scale)
    }

    /// Shape parameter α.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Rate parameter β.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Scale parameter θ = 1/β.
    pub fn scale(&self) -> f64 {
        1.0 / self.rate
    }

    /// Mean = α/β.
    pub fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    /// Variance = α/β².
    pub fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }

    /// Mode = (α−1)/β for α ≥ 1, 0 for α < 1.
    pub fn mode(&self) -> f64 {
        if self.shape >= 1.0 {
            (self.shape - 1.0) / self.rate
        } else {
            0.0
        }
    }

    /// PDF: f(x) = β^α / Γ(α) · x^(α−1) · exp(−βx) for x > 0.
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            if x == 0.0 && self.shape < 1.0 {
                return f64::INFINITY;
            }
            return 0.0;
        }
        let log_pdf = self.shape * self.rate.ln() - ln_gamma(self.shape)
            + (self.shape - 1.0) * x.ln()
            - self.rate * x;
        log_pdf.exp()
    }

    /// CDF: F(x) = γ(α, βx) / Γ(α) = regularized lower incomplete gamma.
    ///
    /// Uses series expansion for small x or continued fraction for large x.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        regularized_lower_gamma(self.shape, self.rate * x)
    }

    /// Quantile (inverse CDF) via Newton-Raphson iteration.
    ///
    /// Returns `None` if p is not in [0, 1].
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }
        if p == 0.0 {
            return Some(0.0);
        }
        if p == 1.0 {
            return Some(f64::INFINITY);
        }

        // Initial guess: Wilson-Hilferty approximation
        let z = crate::special::inverse_normal_cdf(p);
        let v = 1.0 / (9.0 * self.shape);
        let chi2_approx = self.shape * (1.0 - v + z * v.sqrt()).powi(3).max(0.001);
        let mut x = chi2_approx / self.rate;

        // Newton-Raphson refinement
        for _ in 0..50 {
            let f = self.cdf(x) - p;
            let fp = self.pdf(x);
            if fp < 1e-300 {
                break;
            }
            let step = f / fp;
            x = (x - step).max(1e-15);
            if step.abs() < 1e-12 * x {
                break;
            }
        }

        Some(x)
    }
}

/// Regularized lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a).
///
/// Uses series expansion for x < a + 1, continued fraction otherwise.
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series expansion
        gamma_series(a, x)
    } else {
        // Continued fraction (complement)
        1.0 - gamma_cf(a, x)
    }
}

/// Series expansion for the regularized lower incomplete gamma.
fn gamma_series(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut ap = a;

    for _ in 0..200 {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * 1e-14 {
            break;
        }
    }

    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Continued fraction for the upper incomplete gamma Q(a, x) = 1 − P(a, x).
fn gamma_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=200 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-14 {
            break;
        }
    }

    h * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Gamma function Γ(x) = exp(ln_gamma(x)).
///
/// Uses the Lanczos approximation via [`ln_gamma`].
fn gamma(x: f64) -> f64 {
    ln_gamma(x).exp()
}

// ============================================================================
// Regularized Incomplete Beta Function
// ============================================================================

/// Regularized incomplete beta function I_x(a, b).
///
/// Uses the continued fraction representation (Lentz's method) for
/// numerical evaluation.
///
/// Reference: Press et al. (2007), *Numerical Recipes*, 3rd ed., §6.4.
///
/// # Accuracy
/// Relative error < 1e-10 for typical parameter ranges.
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation: I_x(a,b) = 1 - I_{1-x}(b,a)
    // Choose the form with better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b);

    // Continued fraction (Lentz's algorithm)
    let cf = beta_cf(x, a, b);

    (ln_prefix.exp() / a) * cf
}

/// Log of the Beta function: ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a+b).
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation of ln Γ(x).
///
/// Reference: Lanczos (1964), "A Precision Approximation of the Gamma
/// Function", *SIAM Journal on Numerical Analysis* 1(1).
///
/// # Accuracy
/// Relative error < 2 × 10⁻¹⁰ for x > 0.
fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients (g = 7)
    #[allow(clippy::excessive_precision)]
    const COEFFICIENTS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        // Reflection formula: Γ(x)·Γ(1−x) = π/sin(πx)
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = COEFFICIENTS[0];
    for (i, &c) in COEFFICIENTS[1..].iter().enumerate() {
        sum += c / (x + i as f64 + 1.0);
    }

    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Continued fraction for the incomplete beta function (Lentz's algorithm).
///
/// Reference: Press et al. (2007), *Numerical Recipes*, §6.4.
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-14;
    const TINY: f64 = 1e-30;

    let mut c = 1.0;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(TINY);
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;

        // Even step: d_{2m}
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 / (1.0 + num_even * d).max(TINY);
        c = (1.0 + num_even / c).max(TINY);
        h *= d * c;

        // Odd step: d_{2m+1}
        let num_odd = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 / (1.0 + num_odd * d).max(TINY);
        c = (1.0 + num_odd / c).max(TINY);
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPS {
            break;
        }
    }

    h
}

// ============================================================================
// Beta Distribution
// ============================================================================

/// Beta distribution on `[0, 1]`.
///
/// # Mathematical Definition
/// - PDF: f(x) = x^(α−1) (1−x)^(β−1) / B(α, β)
/// - CDF: F(x) = I_x(α, β) (regularized incomplete beta)
/// - Mean: α / (α + β)
/// - Variance: αβ / ((α+β)²(α+β+1))
///
/// # Examples
///
/// ```
/// use u_numflow::distributions::BetaDistribution;
///
/// let beta = BetaDistribution::new(2.0, 5.0).unwrap();
/// assert!((beta.mean() - 2.0 / 7.0).abs() < 1e-10);
/// assert!(beta.pdf(0.3) > 0.0);
/// assert!((beta.cdf(0.0)).abs() < 1e-10);
/// assert!((beta.cdf(1.0) - 1.0).abs() < 1e-10);
/// ```
///
/// # References
///
/// Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995).
/// *Continuous Univariate Distributions*, Vol. 2, Chapter 25.
#[derive(Debug, Clone)]
pub struct BetaDistribution {
    /// Shape parameter α > 0.
    pub alpha: f64,
    /// Shape parameter β > 0.
    pub beta: f64,
}

impl BetaDistribution {
    /// Creates a new Beta distribution with shape parameters α and β.
    ///
    /// # Errors
    /// Returns `DistributionError` if α ≤ 0 or β ≤ 0.
    pub fn new(alpha: f64, beta: f64) -> Result<Self, DistributionError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(DistributionError::InvalidParameters(
                format!("alpha must be positive and finite, got {alpha}"),
            ));
        }
        if beta <= 0.0 || !beta.is_finite() {
            return Err(DistributionError::InvalidParameters(
                format!("beta must be positive and finite, got {beta}"),
            ));
        }
        Ok(Self { alpha, beta })
    }

    /// Returns the mean: α / (α + β).
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Returns the variance: αβ / ((α+β)²(α+β+1)).
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        self.alpha * self.beta / (ab * ab * (ab + 1.0))
    }

    /// Returns the mode.
    ///
    /// Defined when α > 1 and β > 1: mode = (α−1)/(α+β−2).
    /// Returns `None` for other parameter combinations (bimodal or boundary modes).
    pub fn mode(&self) -> Option<f64> {
        if self.alpha > 1.0 && self.beta > 1.0 {
            Some((self.alpha - 1.0) / (self.alpha + self.beta - 2.0))
        } else {
            None
        }
    }

    /// Evaluates the PDF at x.
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        let ln_pdf = (self.alpha - 1.0) * x.ln()
            + (self.beta - 1.0) * (1.0 - x).ln()
            - ln_beta(self.alpha, self.beta);
        ln_pdf.exp()
    }

    /// Evaluates the CDF at x: F(x) = I_x(α, β).
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        regularized_incomplete_beta(x, self.alpha, self.beta)
    }

    /// Evaluates the quantile (inverse CDF) at probability p ∈ [0, 1].
    ///
    /// Uses Newton-Raphson iteration with an initial approximation
    /// from the normal approximation to the beta distribution.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }
        if p == 0.0 {
            return Some(0.0);
        }
        if (p - 1.0).abs() < 1e-15 {
            return Some(1.0);
        }

        // Bisection method for robust convergence on [0, 1]
        let mut lo = 0.0_f64;
        let mut hi = 1.0_f64;

        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            if hi - lo < 1e-14 {
                break;
            }
            if self.cdf(mid) < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        Some((lo + hi) / 2.0)
    }
}

// ============================================================================
// Chi-Squared Distribution
// ============================================================================

/// Chi-squared distribution with k degrees of freedom.
///
/// The chi-squared distribution is a special case of the Gamma distribution
/// with shape = k/2 and rate = 1/2.
///
/// # Mathematical Definition
/// - PDF: f(x) = x^(k/2−1) exp(−x/2) / (2^(k/2) Γ(k/2))
/// - CDF: F(x) = P(k/2, x/2) (regularized lower incomplete gamma)
/// - Mean: k
/// - Variance: 2k
///
/// # Examples
///
/// ```
/// use u_numflow::distributions::ChiSquared;
///
/// let chi2 = ChiSquared::new(3.0).unwrap();
/// assert!((chi2.mean() - 3.0).abs() < 1e-10);
/// assert!((chi2.variance() - 6.0).abs() < 1e-10);
/// assert!((chi2.cdf(0.0)).abs() < 1e-10);
/// ```
///
/// # References
///
/// Johnson, N. L., Kotz, S., & Balakrishnan, N. (1994).
/// *Continuous Univariate Distributions*, Vol. 1, Chapter 18.
#[derive(Debug, Clone)]
pub struct ChiSquared {
    /// Degrees of freedom k > 0.
    pub k: f64,
}

impl ChiSquared {
    /// Creates a new Chi-squared distribution with k degrees of freedom.
    ///
    /// # Errors
    /// Returns `DistributionError` if k ≤ 0.
    pub fn new(k: f64) -> Result<Self, DistributionError> {
        if k <= 0.0 || !k.is_finite() {
            return Err(DistributionError::InvalidParameters(
                format!("k must be positive and finite, got {k}"),
            ));
        }
        Ok(Self { k })
    }

    /// Returns the mean: k.
    pub fn mean(&self) -> f64 {
        self.k
    }

    /// Returns the variance: 2k.
    pub fn variance(&self) -> f64 {
        2.0 * self.k
    }

    /// Evaluates the PDF at x.
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let half_k = self.k / 2.0;
        let ln_pdf = (half_k - 1.0) * x.ln() - x / 2.0 - half_k * 2.0_f64.ln() - ln_gamma(half_k);
        ln_pdf.exp()
    }

    /// Evaluates the CDF at x: F(x) = P(k/2, x/2).
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        regularized_lower_gamma(self.k / 2.0, x / 2.0)
    }

    /// Evaluates the quantile (inverse CDF) at probability p ∈ [0, 1].
    ///
    /// Delegates to Gamma(k/2, 1/2) quantile since Chi2(k) = Gamma(k/2, 1/2).
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }
        if p == 0.0 {
            return Some(0.0);
        }
        if (p - 1.0).abs() < 1e-15 {
            return Some(f64::INFINITY);
        }

        // Chi2(k) = Gamma(k/2, rate=1/2)
        let gamma = GammaDistribution {
            shape: self.k / 2.0,
            rate: 0.5,
        };
        gamma.quantile(p)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Uniform ---

    #[test]
    fn test_uniform_basic() {
        let u = Uniform::new(0.0, 10.0).unwrap();
        assert!((u.mean() - 5.0).abs() < 1e-15);
        assert!((u.variance() - 100.0 / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_cdf() {
        let u = Uniform::new(0.0, 10.0).unwrap();
        assert_eq!(u.cdf(-1.0), 0.0);
        assert!((u.cdf(5.0) - 0.5).abs() < 1e-15);
        assert_eq!(u.cdf(11.0), 1.0);
    }

    #[test]
    fn test_uniform_quantile() {
        let u = Uniform::new(2.0, 8.0).unwrap();
        assert_eq!(u.quantile(0.0), Some(2.0));
        assert_eq!(u.quantile(1.0), Some(8.0));
        assert!((u.quantile(0.5).unwrap() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_uniform_pdf() {
        let u = Uniform::new(0.0, 5.0).unwrap();
        assert!((u.pdf(2.5) - 0.2).abs() < 1e-15);
        assert_eq!(u.pdf(-1.0), 0.0);
    }

    #[test]
    fn test_uniform_invalid() {
        assert!(Uniform::new(5.0, 5.0).is_err());
        assert!(Uniform::new(5.0, 3.0).is_err());
        assert!(Uniform::new(f64::NAN, 5.0).is_err());
    }

    // --- Triangular ---

    #[test]
    fn test_triangular_mean() {
        let t = Triangular::new(0.0, 3.0, 6.0).unwrap();
        assert!((t.mean() - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_triangular_symmetric_variance() {
        let t = Triangular::new(0.0, 5.0, 10.0).unwrap();
        // Var = (0+100+25-0-0-50)/18 = 75/18 ≈ 4.1667
        let expected = (0.0 + 25.0 + 100.0 - 0.0 - 0.0 - 50.0) / 18.0;
        assert!((t.variance() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_triangular_cdf() {
        let t = Triangular::new(0.0, 5.0, 10.0).unwrap();
        assert!((t.cdf(0.0)).abs() < 1e-15);
        assert!((t.cdf(10.0) - 1.0).abs() < 1e-15);
        // At mode: F(5) = (5-0)²/((10-0)*(5-0)) = 25/50 = 0.5
        assert!((t.cdf(5.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_triangular_quantile() {
        let t = Triangular::new(0.0, 5.0, 10.0).unwrap();
        assert!((t.quantile(0.0).unwrap() - 0.0).abs() < 1e-15);
        assert!((t.quantile(1.0).unwrap() - 10.0).abs() < 1e-15);
        assert!((t.quantile(0.5).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangular_invalid() {
        assert!(Triangular::new(5.0, 3.0, 10.0).is_err()); // mode < min
        assert!(Triangular::new(0.0, 11.0, 10.0).is_err()); // mode > max
        assert!(Triangular::new(5.0, 5.0, 5.0).is_err()); // min == max
    }

    // --- Normal ---

    #[test]
    fn test_normal_standard() {
        let n = Normal::new(0.0, 1.0).unwrap();
        assert!((n.mean()).abs() < 1e-15);
        assert!((n.variance() - 1.0).abs() < 1e-15);
        assert!((n.cdf(0.0) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_normal_shifted() {
        let n = Normal::new(10.0, 2.0).unwrap();
        assert!((n.mean() - 10.0).abs() < 1e-15);
        assert!((n.variance() - 4.0).abs() < 1e-15);
        assert!((n.cdf(10.0) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_normal_quantile() {
        let n = Normal::new(0.0, 1.0).unwrap();
        assert!((n.quantile(0.5).unwrap()).abs() < 0.01);
        assert!((n.quantile(0.975).unwrap() - 1.96).abs() < 0.01);
    }

    #[test]
    fn test_normal_invalid() {
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
    }

    // --- LogNormal ---

    #[test]
    fn test_lognormal_mean() {
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        let expected = (0.5_f64).exp(); // exp(0 + 1/2)
        assert!((ln.mean() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_lognormal_cdf() {
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        assert_eq!(ln.cdf(0.0), 0.0);
        // Median of LogNormal(0,1) = exp(0) = 1.0
        assert!((ln.cdf(1.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_lognormal_quantile() {
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        // Median = exp(μ) = 1.0
        let q50 = ln.quantile(0.5).unwrap();
        assert!((q50 - 1.0).abs() < 0.01);
    }

    // --- PERT ---

    #[test]
    fn test_pert_mean() {
        let p = Pert::new(1.0, 4.0, 7.0).unwrap();
        // Mean = (1 + 4*4 + 7) / 6 = 24/6 = 4.0
        assert!((p.mean() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_pert_symmetric_variance() {
        let p = Pert::new(0.0, 5.0, 10.0).unwrap();
        // For symmetric PERT: α = β = 3, range = 10
        // Var = 3*3/(6*6*7) * 100 = 900/2520 * 100 ≈ 3.571
        let expected = 9.0 / (36.0 * 7.0) * 100.0;
        assert!(
            (p.variance() - expected).abs() < 1e-10,
            "PERT variance: {} vs expected: {}",
            p.variance(),
            expected
        );
    }

    #[test]
    fn test_pert_shape_params() {
        let p = Pert::new(0.0, 5.0, 10.0).unwrap();
        // α = 1 + 4*(5-0)/(10-0) = 1 + 2 = 3
        // β = 1 + 4*(10-5)/(10-0) = 1 + 2 = 3
        assert!((p.alpha() - 3.0).abs() < 1e-15);
        assert!((p.beta_param() - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_pert_cdf_bounds() {
        let p = Pert::new(1.0, 4.0, 7.0).unwrap();
        assert_eq!(p.cdf(0.0), 0.0);
        assert_eq!(p.cdf(8.0), 1.0);
        // CDF at mean should be close to 0.5 for symmetric PERT
        let p_sym = Pert::new(0.0, 5.0, 10.0).unwrap();
        assert!((p_sym.cdf(5.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_pert_cdf_monotonic() {
        let p = Pert::new(0.0, 3.0, 10.0).unwrap();
        let mut prev = 0.0;
        for i in 0..=100 {
            let x = i as f64 * 0.1;
            let c = p.cdf(x);
            assert!(c >= prev - 1e-15, "CDF not monotonic at x={x}");
            prev = c;
        }
    }

    #[test]
    fn test_pert_quantile_approx() {
        let p = Pert::new(0.0, 5.0, 10.0).unwrap();
        let q50 = p.quantile_approx(0.5).unwrap();
        assert!((q50 - 5.0).abs() < 0.5, "median approx: {q50}");
    }

    #[test]
    fn test_pert_invalid() {
        assert!(Pert::new(5.0, 3.0, 10.0).is_err());
        assert!(Pert::new(5.0, 5.0, 5.0).is_err());
    }

    #[test]
    fn test_pert_with_lambda() {
        // Higher lambda = more peaked
        let p4 = Pert::with_shape(0.0, 5.0, 10.0, 4.0).unwrap();
        let p8 = Pert::with_shape(0.0, 5.0, 10.0, 8.0).unwrap();
        assert!(
            p8.variance() < p4.variance(),
            "higher λ should give lower variance"
        );
    }

    // --- Regularized Incomplete Beta Function ---

    #[test]
    fn test_regularized_beta_bounds() {
        assert_eq!(regularized_incomplete_beta(0.0, 2.0, 3.0), 0.0);
        assert_eq!(regularized_incomplete_beta(1.0, 2.0, 3.0), 1.0);
    }

    #[test]
    fn test_regularized_beta_symmetric() {
        // For Beta(a,a), I_{0.5}(a,a) = 0.5 by symmetry
        let result = regularized_incomplete_beta(0.5, 3.0, 3.0);
        assert!(
            (result - 0.5).abs() < 1e-8,
            "I_0.5(3,3) = {result}, expected 0.5"
        );
    }

    #[test]
    fn test_regularized_beta_known_values() {
        // I_x(1,1) = x (Uniform)
        for &x in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let result = regularized_incomplete_beta(x, 1.0, 1.0);
            assert!(
                (result - x).abs() < 1e-10,
                "I_{x}(1,1) = {result}, expected {x}"
            );
        }

        // I_x(1,b) = 1 - (1-x)^b
        for &x in &[0.1, 0.5, 0.9] {
            let result = regularized_incomplete_beta(x, 1.0, 3.0);
            let expected = 1.0 - (1.0 - x).powi(3);
            assert!(
                (result - expected).abs() < 1e-10,
                "I_{x}(1,3) = {result}, expected {expected}"
            );
        }
    }

    // --- ln_gamma ---

    #[test]
    fn test_ln_gamma_known() {
        // Γ(1) = 1, ln(1) = 0
        assert!((ln_gamma(1.0)).abs() < 1e-10);
        // Γ(2) = 1, ln(1) = 0
        assert!((ln_gamma(2.0)).abs() < 1e-10);
        // Γ(3) = 2, ln(2) ≈ 0.6931
        assert!((ln_gamma(3.0) - 2.0_f64.ln()).abs() < 1e-10);
        // Γ(5) = 24, ln(24) ≈ 3.1781
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
        // Γ(0.5) = √π
        assert!(
            (ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-10,
            "ln Γ(0.5) = {}, expected {}",
            ln_gamma(0.5),
            std::f64::consts::PI.sqrt().ln()
        );
    }

    // --- Weibull ---

    #[test]
    fn test_weibull_exponential_special_case() {
        // β=1 → Exponential(λ=1/η), mean=η, variance=η²
        let w = Weibull::new(1.0, 5.0).unwrap();
        assert!((w.mean() - 5.0).abs() < 1e-10);
        assert!((w.variance() - 25.0).abs() < 1e-8);
    }

    #[test]
    fn test_weibull_rayleigh_special_case() {
        // β=2 → Rayleigh, mean = η√(π/4) = η·Γ(1.5) = η·(√π/2)
        let w = Weibull::new(2.0, 1.0).unwrap();
        let expected_mean = std::f64::consts::PI.sqrt() / 2.0;
        assert!(
            (w.mean() - expected_mean).abs() < 1e-10,
            "Weibull(2,1) mean = {}, expected {}",
            w.mean(),
            expected_mean
        );
    }

    #[test]
    fn test_weibull_cdf_known_values() {
        // F(t) = 1 - exp(-(t/η)^β)
        let w = Weibull::new(2.0, 10.0).unwrap();
        // F(10) = 1 - exp(-1) ≈ 0.6321
        assert!((w.cdf(10.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
        // F(0) = 0
        assert_eq!(w.cdf(0.0), 0.0);
        // F(-1) = 0
        assert_eq!(w.cdf(-1.0), 0.0);
    }

    #[test]
    fn test_weibull_pdf_basic() {
        // β=1, η=1: f(t) = exp(-t)
        let w = Weibull::new(1.0, 1.0).unwrap();
        assert!((w.pdf(0.0) - 1.0).abs() < 1e-10);
        assert!((w.pdf(1.0) - (-1.0_f64).exp()).abs() < 1e-10);
        assert!((w.pdf(2.0) - (-2.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_pdf_negative() {
        let w = Weibull::new(2.0, 5.0).unwrap();
        assert_eq!(w.pdf(-1.0), 0.0);
    }

    #[test]
    fn test_weibull_quantile_roundtrip() {
        let w = Weibull::new(2.5, 100.0).unwrap();
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let t = w.quantile(p).unwrap();
            let p_back = w.cdf(t);
            assert!(
                (p_back - p).abs() < 1e-10,
                "roundtrip: p={p} -> t={t} -> p_back={p_back}"
            );
        }
    }

    #[test]
    fn test_weibull_quantile_edge_cases() {
        let w = Weibull::new(2.0, 10.0).unwrap();
        assert_eq!(w.quantile(0.0), Some(0.0));
        assert_eq!(w.quantile(1.0), None);
        assert_eq!(w.quantile(-0.1), None);
    }

    #[test]
    fn test_weibull_reliability() {
        let w = Weibull::new(2.0, 10.0).unwrap();
        // R(t) = 1 - F(t)
        for &t in &[1.0, 5.0, 10.0, 20.0] {
            let r = w.reliability(t);
            let f = w.cdf(t);
            assert!((r + f - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_weibull_hazard_rate() {
        // β=1 → constant hazard rate = 1/η
        let w = Weibull::new(1.0, 5.0).unwrap();
        for &t in &[1.0, 5.0, 10.0] {
            assert!((w.hazard_rate(t) - 0.2).abs() < 1e-10);
        }
        // β=2 → h(t) = 2t/η², linearly increasing
        let w2 = Weibull::new(2.0, 10.0).unwrap();
        assert!((w2.hazard_rate(5.0) - 2.0 * 5.0 / 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_invalid() {
        assert!(Weibull::new(0.0, 1.0).is_err());
        assert!(Weibull::new(-1.0, 1.0).is_err());
        assert!(Weibull::new(1.0, 0.0).is_err());
        assert!(Weibull::new(1.0, -1.0).is_err());
        assert!(Weibull::new(f64::NAN, 1.0).is_err());
        assert!(Weibull::new(1.0, f64::INFINITY).is_err());
    }

    // --- Exponential ---

    #[test]
    fn test_exponential_basic() {
        let e = Exponential::new(0.5).unwrap();
        assert!((e.rate() - 0.5).abs() < 1e-10);
        assert!((e.mean() - 2.0).abs() < 1e-10);
        assert!((e.variance() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_pdf() {
        let e = Exponential::new(1.0).unwrap();
        assert!((e.pdf(0.0) - 1.0).abs() < 1e-10);
        assert!((e.pdf(1.0) - (-1.0_f64).exp()).abs() < 1e-10);
        assert_eq!(e.pdf(-1.0), 0.0);
    }

    #[test]
    fn test_exponential_cdf() {
        let e = Exponential::new(1.0).unwrap();
        assert_eq!(e.cdf(-1.0), 0.0);
        assert_eq!(e.cdf(0.0), 0.0);
        assert!((e.cdf(1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_quantile() {
        let e = Exponential::new(2.0).unwrap();
        assert_eq!(e.quantile(0.0), Some(0.0));
        assert_eq!(e.quantile(1.0), Some(f64::INFINITY));
        let q = e.quantile(0.5).unwrap();
        // Median = ln(2)/λ = ln(2)/2
        assert!((q - 2.0_f64.ln() / 2.0).abs() < 1e-10);
        assert!(e.quantile(-0.1).is_none());
        assert!(e.quantile(1.1).is_none());
    }

    #[test]
    fn test_exponential_quantile_roundtrip() {
        let e = Exponential::new(3.0).unwrap();
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = e.quantile(p).unwrap();
            let p_back = e.cdf(x);
            assert!((p_back - p).abs() < 1e-10, "p={p}, x={x}, p_back={p_back}");
        }
    }

    #[test]
    fn test_exponential_memoryless() {
        // P(X > s+t | X > s) = P(X > t)
        let e = Exponential::new(1.5).unwrap();
        let s = 2.0;
        let t = 3.0;
        let lhs = e.survival(s + t) / e.survival(s);
        let rhs = e.survival(t);
        assert!((lhs - rhs).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_invalid() {
        assert!(Exponential::new(0.0).is_err());
        assert!(Exponential::new(-1.0).is_err());
        assert!(Exponential::new(f64::NAN).is_err());
        assert!(Exponential::new(f64::INFINITY).is_err());
    }

    // --- Gamma Distribution ---

    #[test]
    fn test_gamma_basic() {
        let g = GammaDistribution::new(2.0, 1.0).unwrap();
        assert!((g.shape() - 2.0).abs() < 1e-10);
        assert!((g.rate() - 1.0).abs() < 1e-10);
        assert!((g.scale() - 1.0).abs() < 1e-10);
        assert!((g.mean() - 2.0).abs() < 1e-10);
        assert!((g.variance() - 2.0).abs() < 1e-10);
        assert!((g.mode() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_from_shape_scale() {
        let g = GammaDistribution::from_shape_scale(3.0, 2.0).unwrap();
        assert!((g.shape() - 3.0).abs() < 1e-10);
        assert!((g.rate() - 0.5).abs() < 1e-10);
        assert!((g.mean() - 6.0).abs() < 1e-10); // α/β = 3/0.5 = 6
    }

    #[test]
    fn test_gamma_pdf_integral() {
        // Numerical integration should ≈ 1
        let g = GammaDistribution::new(3.0, 2.0).unwrap();
        let n = 20_000;
        let dt = 20.0 / n as f64;
        let integral: f64 = (0..n)
            .map(|i| {
                let x = (i as f64 + 0.5) * dt;
                g.pdf(x) * dt
            })
            .sum();
        assert!((integral - 1.0).abs() < 0.01, "PDF integral = {integral}");
    }

    #[test]
    fn test_gamma_cdf_known() {
        // Gamma(1, 1) = Exponential(1): CDF(1) = 1 - e^{-1}
        let g = GammaDistribution::new(1.0, 1.0).unwrap();
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((g.cdf(1.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_cdf_chi2() {
        // Chi-squared(k) = Gamma(k/2, 1/2)
        // For k=2, CDF(x) = 1 - exp(-x/2)
        let g = GammaDistribution::new(1.0, 0.5).unwrap(); // Gamma(1, 0.5) = Exp(0.5)
        let x = 4.0;
        let expected = 1.0 - (-0.5_f64 * x).exp();
        assert!((g.cdf(x) - expected).abs() < 1e-8, "CDF({x}) = {} vs {expected}", g.cdf(x));
    }

    #[test]
    fn test_gamma_quantile_roundtrip() {
        let g = GammaDistribution::new(5.0, 2.0).unwrap();
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = g.quantile(p).unwrap();
            let p_back = g.cdf(x);
            assert!((p_back - p).abs() < 1e-6, "p={p}, x={x}, p_back={p_back}");
        }
    }

    #[test]
    fn test_gamma_mode_small_shape() {
        let g = GammaDistribution::new(0.5, 1.0).unwrap();
        assert_eq!(g.mode(), 0.0); // α < 1
    }

    #[test]
    fn test_gamma_invalid() {
        assert!(GammaDistribution::new(0.0, 1.0).is_err());
        assert!(GammaDistribution::new(-1.0, 1.0).is_err());
        assert!(GammaDistribution::new(1.0, 0.0).is_err());
        assert!(GammaDistribution::new(1.0, -1.0).is_err());
        assert!(GammaDistribution::new(f64::NAN, 1.0).is_err());
    }

    // --- Beta Distribution ---

    #[test]
    fn test_beta_mean_variance() {
        let b = BetaDistribution::new(2.0, 5.0).unwrap();
        // Mean = 2/7
        assert!((b.mean() - 2.0 / 7.0).abs() < 1e-10);
        // Variance = 2*5 / (49 * 8) = 10/392
        let expected_var = 2.0 * 5.0 / (7.0 * 7.0 * 8.0);
        assert!((b.variance() - expected_var).abs() < 1e-10);
    }

    #[test]
    fn test_beta_symmetric() {
        // Beta(3, 3) is symmetric around 0.5
        let b = BetaDistribution::new(3.0, 3.0).unwrap();
        assert!((b.mean() - 0.5).abs() < 1e-10);
        assert!((b.mode().unwrap() - 0.5).abs() < 1e-10);
        // PDF symmetric: f(0.3) = f(0.7)
        assert!((b.pdf(0.3) - b.pdf(0.7)).abs() < 1e-10);
    }

    #[test]
    fn test_beta_uniform_special_case() {
        // Beta(1, 1) = Uniform(0, 1)
        let b = BetaDistribution::new(1.0, 1.0).unwrap();
        assert!((b.mean() - 0.5).abs() < 1e-10);
        assert!((b.cdf(0.5) - 0.5).abs() < 1e-10);
        assert!((b.cdf(0.25) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_beta_cdf_boundaries() {
        let b = BetaDistribution::new(2.0, 3.0).unwrap();
        assert_eq!(b.cdf(0.0), 0.0);
        assert_eq!(b.cdf(-1.0), 0.0);
        assert_eq!(b.cdf(1.0), 1.0);
        assert_eq!(b.cdf(2.0), 1.0);
    }

    #[test]
    fn test_beta_pdf_integral() {
        let b = BetaDistribution::new(2.0, 5.0).unwrap();
        let n = 10_000;
        let dt = 1.0 / n as f64;
        let integral: f64 = (0..n)
            .map(|i| {
                let x = (i as f64 + 0.5) * dt;
                b.pdf(x) * dt
            })
            .sum();
        assert!((integral - 1.0).abs() < 0.01, "PDF integral = {integral}");
    }

    #[test]
    fn test_beta_quantile_roundtrip() {
        let b = BetaDistribution::new(2.0, 5.0).unwrap();
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = b.quantile(p).unwrap();
            let p_back = b.cdf(x);
            assert!((p_back - p).abs() < 1e-6, "p={p}, x={x}, p_back={p_back}");
        }
    }

    #[test]
    fn test_beta_mode() {
        let b = BetaDistribution::new(5.0, 3.0).unwrap();
        // mode = (5-1)/(5+3-2) = 4/6 = 2/3
        assert!((b.mode().unwrap() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_beta_mode_undefined() {
        // Beta(0.5, 0.5) — U-shaped, no interior mode
        let b = BetaDistribution::new(0.5, 0.5).unwrap();
        assert!(b.mode().is_none());
    }

    #[test]
    fn test_beta_invalid() {
        assert!(BetaDistribution::new(0.0, 1.0).is_err());
        assert!(BetaDistribution::new(-1.0, 1.0).is_err());
        assert!(BetaDistribution::new(1.0, 0.0).is_err());
        assert!(BetaDistribution::new(f64::NAN, 1.0).is_err());
    }

    // --- Chi-Squared Distribution ---

    #[test]
    fn test_chi2_mean_variance() {
        let chi2 = ChiSquared::new(5.0).unwrap();
        assert!((chi2.mean() - 5.0).abs() < 1e-10);
        assert!((chi2.variance() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_chi2_exp_special_case() {
        // Chi2(2) = Exp(1/2): CDF(x) = 1 - exp(-x/2)
        let chi2 = ChiSquared::new(2.0).unwrap();
        let x = 4.0;
        let expected = 1.0 - (-x / 2.0_f64).exp();
        assert!(
            (chi2.cdf(x) - expected).abs() < 1e-8,
            "CDF({x}) = {} vs {expected}",
            chi2.cdf(x)
        );
    }

    #[test]
    fn test_chi2_cdf_boundaries() {
        let chi2 = ChiSquared::new(3.0).unwrap();
        assert_eq!(chi2.cdf(0.0), 0.0);
        assert_eq!(chi2.cdf(-1.0), 0.0);
        // CDF should approach 1 for large x
        assert!(chi2.cdf(100.0) > 0.999);
    }

    #[test]
    fn test_chi2_pdf_integral() {
        let chi2 = ChiSquared::new(4.0).unwrap();
        let n = 20_000;
        let dt = 30.0 / n as f64;
        let integral: f64 = (0..n)
            .map(|i| {
                let x = (i as f64 + 0.5) * dt;
                chi2.pdf(x) * dt
            })
            .sum();
        assert!((integral - 1.0).abs() < 0.01, "PDF integral = {integral}");
    }

    #[test]
    fn test_chi2_quantile_roundtrip() {
        let chi2 = ChiSquared::new(5.0).unwrap();
        for &p in &[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let x = chi2.quantile(p).unwrap();
            let p_back = chi2.cdf(x);
            assert!((p_back - p).abs() < 1e-6, "p={p}, x={x}, p_back={p_back}");
        }
    }

    #[test]
    fn test_chi2_known_quantiles() {
        // Common chi-squared critical values (k=5)
        // p=0.05 → 1.1455, p=0.95 → 11.0705 (from tables)
        let chi2 = ChiSquared::new(5.0).unwrap();
        let q05 = chi2.quantile(0.05).unwrap();
        assert!((q05 - 1.1455).abs() < 0.01, "q(0.05) = {q05}");
        let q95 = chi2.quantile(0.95).unwrap();
        assert!((q95 - 11.0705).abs() < 0.01, "q(0.95) = {q95}");
    }

    #[test]
    fn test_chi2_invalid() {
        assert!(ChiSquared::new(0.0).is_err());
        assert!(ChiSquared::new(-1.0).is_err());
        assert!(ChiSquared::new(f64::NAN).is_err());
    }

    #[test]
    fn test_chi2_gamma_consistency() {
        // Chi2(k) = Gamma(k/2, 1/2)
        let k = 6.0;
        let chi2 = ChiSquared::new(k).unwrap();
        let gamma = GammaDistribution::new(k / 2.0, 0.5).unwrap();

        let x = 5.0;
        assert!(
            (chi2.cdf(x) - gamma.cdf(x)).abs() < 1e-10,
            "Chi2 and Gamma CDF should match"
        );
    }

    #[test]
    fn test_weibull_mean_variance_known() {
        // β=3.6, η=1000: known engineering example
        // Γ(1 + 1/3.6) = Γ(1.2778) ≈ 0.8946
        let w = Weibull::new(3.6, 1000.0).unwrap();
        let mean = w.mean();
        assert!(mean > 800.0 && mean < 1000.0, "mean={mean}");
        assert!(w.variance() > 0.0);
    }

    #[test]
    fn test_weibull_pdf_integral_approx() {
        // Numerical integration of PDF should ≈ 1
        let w = Weibull::new(2.0, 10.0).unwrap();
        let n = 10_000;
        let dt = 50.0 / n as f64;
        let integral: f64 = (0..n).map(|i| {
            let t = (i as f64 + 0.5) * dt;
            w.pdf(t) * dt
        }).sum();
        assert!(
            (integral - 1.0).abs() < 0.01,
            "PDF integral = {integral}, expected ≈ 1.0"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        // --- Uniform ---

        #[test]
        fn uniform_cdf_in_01(
            min in -100.0_f64..0.0,
            max in 1.0_f64..100.0,
            x in -200.0_f64..200.0,
        ) {
            let u = Uniform::new(min, max).unwrap();
            let c = u.cdf(x);
            prop_assert!((0.0..=1.0).contains(&c));
        }

        #[test]
        fn uniform_quantile_roundtrip(
            min in -100.0_f64..0.0,
            max in 1.0_f64..100.0,
            p in 0.0_f64..=1.0,
        ) {
            let u = Uniform::new(min, max).unwrap();
            let x = u.quantile(p).unwrap();
            let p_back = u.cdf(x);
            prop_assert!((p_back - p).abs() < 1e-12, "roundtrip: p={p} -> x={x} -> p_back={p_back}");
        }

        // --- Triangular ---

        #[test]
        fn triangular_cdf_in_01(
            min in -100.0_f64..-1.0,
            mode_frac in 0.0_f64..=1.0,
            range in 1.0_f64..100.0,
            x in -200.0_f64..200.0,
        ) {
            let max = min + range;
            let mode = min + mode_frac * range;
            let t = Triangular::new(min, mode, max).unwrap();
            let c = t.cdf(x);
            prop_assert!((0.0..=1.0).contains(&c));
        }

        #[test]
        fn triangular_quantile_roundtrip(
            min in -50.0_f64..0.0,
            mode_frac in 0.01_f64..0.99,
            range in 1.0_f64..50.0,
            p in 0.001_f64..0.999,
        ) {
            let max = min + range;
            let mode = min + mode_frac * range;
            let t = Triangular::new(min, mode, max).unwrap();
            let x = t.quantile(p).unwrap();
            let p_back = t.cdf(x);
            prop_assert!(
                (p_back - p).abs() < 1e-8,
                "roundtrip: p={p} -> x={x} -> p_back={p_back}"
            );
        }

        // --- PERT ---

        #[test]
        fn pert_mean_formula(
            min in -50.0_f64..0.0,
            mode_frac in 0.01_f64..0.99,
            range in 1.0_f64..50.0,
        ) {
            let max = min + range;
            let mode = min + mode_frac * range;
            let p = Pert::new(min, mode, max).unwrap();
            let expected = (min + 4.0 * mode + max) / 6.0;
            prop_assert!(
                (p.mean() - expected).abs() < 1e-10,
                "PERT mean: {} vs expected: {}",
                p.mean(),
                expected
            );
        }

        #[test]
        fn pert_variance_positive(
            min in -50.0_f64..0.0,
            mode_frac in 0.01_f64..0.99,
            range in 1.0_f64..50.0,
        ) {
            let max = min + range;
            let mode = min + mode_frac * range;
            let p = Pert::new(min, mode, max).unwrap();
            prop_assert!(p.variance() > 0.0);
        }

        #[test]
        fn pert_cdf_monotonic(
            min in -50.0_f64..0.0,
            mode_frac in 0.05_f64..0.95,
            range in 2.0_f64..50.0,
        ) {
            let max = min + range;
            let mode = min + mode_frac * range;
            let p = Pert::new(min, mode, max).unwrap();
            let mut prev = 0.0;
            for i in 0..=20 {
                let x = min + (i as f64 / 20.0) * range;
                let c = p.cdf(x);
                prop_assert!(c >= prev - 1e-10, "CDF not monotonic at x={x}");
                prev = c;
            }
        }

        // --- Weibull ---

        #[test]
        fn weibull_cdf_in_01(
            shape in 0.1_f64..10.0,
            scale in 0.1_f64..100.0,
            t in 0.0_f64..200.0,
        ) {
            let w = Weibull::new(shape, scale).unwrap();
            let c = w.cdf(t);
            prop_assert!((0.0..=1.0).contains(&c), "CDF({t}) = {c} out of [0,1]");
        }

        #[test]
        fn weibull_quantile_roundtrip(
            shape in 0.5_f64..10.0,
            scale in 1.0_f64..100.0,
            p in 0.001_f64..0.999,
        ) {
            let w = Weibull::new(shape, scale).unwrap();
            let t = w.quantile(p).unwrap();
            let p_back = w.cdf(t);
            prop_assert!(
                (p_back - p).abs() < 1e-8,
                "roundtrip: p={p} -> t={t} -> p_back={p_back}"
            );
        }

        #[test]
        fn weibull_pdf_non_negative(
            shape in 0.1_f64..10.0,
            scale in 0.1_f64..100.0,
            t in 0.0_f64..200.0,
        ) {
            let w = Weibull::new(shape, scale).unwrap();
            prop_assert!(w.pdf(t) >= 0.0, "PDF({t}) must be >= 0");
        }

        #[test]
        fn weibull_reliability_plus_cdf_is_one(
            shape in 0.5_f64..5.0,
            scale in 1.0_f64..50.0,
            t in 0.001_f64..100.0,
        ) {
            let w = Weibull::new(shape, scale).unwrap();
            let sum = w.cdf(t) + w.reliability(t);
            prop_assert!(
                (sum - 1.0).abs() < 1e-12,
                "CDF + R should = 1, got {sum}"
            );
        }

        #[test]
        fn weibull_variance_positive(
            shape in 0.1_f64..10.0,
            scale in 0.1_f64..100.0,
        ) {
            let w = Weibull::new(shape, scale).unwrap();
            prop_assert!(w.variance() > 0.0, "variance must be positive");
        }

        // --- Exponential ---

        #[test]
        fn exponential_cdf_in_01(
            rate in 0.01_f64..10.0,
            x in 0.0_f64..100.0,
        ) {
            let e = Exponential::new(rate).unwrap();
            let c = e.cdf(x);
            prop_assert!((0.0..=1.0).contains(&c), "CDF({x}) = {c}");
        }

        #[test]
        fn exponential_quantile_roundtrip(
            rate in 0.01_f64..10.0,
            p in 0.001_f64..0.999,
        ) {
            let e = Exponential::new(rate).unwrap();
            let x = e.quantile(p).unwrap();
            let p_back = e.cdf(x);
            prop_assert!((p_back - p).abs() < 1e-10);
        }

        // --- Gamma ---

        #[test]
        fn gamma_cdf_in_01(
            shape in 0.5_f64..10.0,
            rate in 0.1_f64..10.0,
            x in 0.001_f64..50.0,
        ) {
            let g = GammaDistribution::new(shape, rate).unwrap();
            let c = g.cdf(x);
            prop_assert!((0.0..=1.0).contains(&c), "CDF({x}) = {c}");
        }

        #[test]
        fn gamma_variance_positive(
            shape in 0.1_f64..10.0,
            rate in 0.1_f64..10.0,
        ) {
            let g = GammaDistribution::new(shape, rate).unwrap();
            prop_assert!(g.variance() > 0.0, "variance must be positive");
        }

        // --- Beta ---

        #[test]
        fn beta_cdf_in_01(
            alpha in 0.5_f64..10.0,
            beta_param in 0.5_f64..10.0,
            x in 0.001_f64..0.999,
        ) {
            let b = BetaDistribution::new(alpha, beta_param).unwrap();
            let c = b.cdf(x);
            prop_assert!((0.0..=1.0).contains(&c), "CDF({x}) = {c}");
        }

        #[test]
        fn beta_quantile_roundtrip(
            alpha in 1.0_f64..10.0,
            beta_param in 1.0_f64..10.0,
            p in 0.01_f64..0.99,
        ) {
            let b = BetaDistribution::new(alpha, beta_param).unwrap();
            let x = b.quantile(p).unwrap();
            let p_back = b.cdf(x);
            prop_assert!((p_back - p).abs() < 1e-5, "p={p}, p_back={p_back}");
        }

        // --- Chi-Squared ---

        #[test]
        fn chi2_cdf_in_01(
            k in 1.0_f64..20.0,
            x in 0.001_f64..50.0,
        ) {
            let chi2 = ChiSquared::new(k).unwrap();
            let c = chi2.cdf(x);
            prop_assert!((0.0..=1.0).contains(&c), "CDF({x}) = {c}");
        }

        #[test]
        fn chi2_variance_positive(
            k in 0.5_f64..50.0,
        ) {
            let chi2 = ChiSquared::new(k).unwrap();
            prop_assert!(chi2.variance() > 0.0, "variance must be positive");
        }
    }
}
