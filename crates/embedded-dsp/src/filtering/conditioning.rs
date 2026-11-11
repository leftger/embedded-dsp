//! Signal conditioning: Deadband, Hysteresis, Slew Rate Limiting, Running Median, and Savitzky-Golay filtering.

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::types::Status;

#[cfg(feature = "pipeline")]
use crate::pipeline::{Inplace, Process};

// ─────────────────────────────────────────────────────────────────────────────
// Deadband
// ─────────────────────────────────────────────────────────────────────────────

/// Deadband signal conditioning block.
///
/// Suppresses small fluctuations around zero (e.g. sensor noise at rest or control stick centering).
///
/// Supports two operational modes:
/// - **Plain**: passes values outside `[-threshold, threshold]` untouched.
/// - **Recentered**: shifts values outside the band back toward zero by `threshold`,
///   so the output leaves zero continuously without sudden step jolts.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Deadband<T> {
    /// How far from zero a value must be before it is passed.
    pub threshold: T,
    /// Whether values outside the band are shifted back toward zero.
    pub recentered: bool,
}

/// Convenience alias for single-precision float deadband.
pub type DeadbandF32 = Deadband<f32>;
/// Convenience alias for double-precision float deadband.
pub type DeadbandF64 = Deadband<f64>;

impl Deadband<f32> {
    /// Builds a plain deadband that passes values outside `[-threshold, threshold]` untouched.
    ///
    /// Returns [`Status::ArgumentError`] if `threshold < 0.0`, or [`Status::NanInf`] if not finite.
    pub fn plain(threshold: f32) -> Result<Self, Status> {
        Self::new(threshold, false)
    }

    /// Builds a recentered deadband that smoothly subtracts `threshold` on breakout.
    pub fn recentered(threshold: f32) -> Result<Self, Status> {
        Self::new(threshold, true)
    }

    /// Constructs a deadband with explicit recentering configuration.
    pub fn new(threshold: f32, recentered: bool) -> Result<Self, Status> {
        if !threshold.is_finite() {
            return Err(Status::NanInf);
        }
        if threshold < 0.0 {
            return Err(Status::ArgumentError);
        }
        Ok(Self { threshold, recentered })
    }

    /// Applies the deadband to a single sample.
    #[inline]
    pub fn apply(&self, input: f32) -> f32 {
        if input.abs() <= self.threshold {
            0.0
        } else if self.recentered {
            input - self.threshold.copysign(input)
        } else {
            input
        }
    }

    /// Applies the deadband with non-finite sample check.
    #[inline]
    pub fn apply_checked(&self, input: f32) -> Result<f32, Status> {
        if !input.is_finite() {
            return Err(Status::NanInf);
        }
        Ok(self.apply(input))
    }
}

impl Deadband<f64> {
    /// Builds a plain deadband for `f64`.
    pub fn plain(threshold: f64) -> Result<Self, Status> {
        Self::new(threshold, false)
    }

    /// Builds a recentered deadband for `f64`.
    pub fn recentered(threshold: f64) -> Result<Self, Status> {
        Self::new(threshold, true)
    }

    /// Constructs a deadband for `f64`.
    pub fn new(threshold: f64, recentered: bool) -> Result<Self, Status> {
        if !threshold.is_finite() {
            return Err(Status::NanInf);
        }
        if threshold < 0.0 {
            return Err(Status::ArgumentError);
        }
        Ok(Self { threshold, recentered })
    }

    /// Applies the deadband to a single sample.
    #[inline]
    pub fn apply(&self, input: f64) -> f64 {
        if input.abs() <= self.threshold {
            0.0
        } else if self.recentered {
            input - self.threshold.copysign(input)
        } else {
            input
        }
    }

    /// Applies the deadband with non-finite sample check.
    #[inline]
    pub fn apply_checked(&self, input: f64) -> Result<f64, Status> {
        if !input.is_finite() {
            return Err(Status::NanInf);
        }
        Ok(self.apply(input))
    }
}

#[cfg(feature = "pipeline")]
impl Process<f32, f32> for Deadband<f32> {
    #[inline]
    fn process(&mut self, sample: f32) -> f32 {
        self.apply(sample)
    }
}

#[cfg(feature = "pipeline")]
impl Inplace<f32> for Deadband<f32> {
    #[inline]
    fn inplace(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.apply(*s);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hysteresis
// ─────────────────────────────────────────────────────────────────────────────

/// Hysteresis threshold comparator (Schmitt trigger).
///
/// Prevents rapid switching / chatter around a noisy decision threshold.
/// Output switches to `true` when input rises above `high_threshold`,
/// and switches to `false` when input falls below `low_threshold`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Hysteresis<T> {
    /// Low threshold below which output turns false.
    pub low_threshold: T,
    /// High threshold above which output turns true.
    pub high_threshold: T,
    /// Current boolean state.
    pub state: bool,
}

/// Convenience alias for single-precision float hysteresis.
pub type HysteresisF32 = Hysteresis<f32>;
/// Convenience alias for double-precision float hysteresis.
pub type HysteresisF64 = Hysteresis<f64>;

impl Hysteresis<f32> {
    /// Creates a new Hysteresis comparator.
    ///
    /// Returns [`Status::ArgumentError`] if `low_threshold > high_threshold`,
    /// or [`Status::NanInf`] if either threshold is non-finite.
    pub fn new(low_threshold: f32, high_threshold: f32, initial_state: bool) -> Result<Self, Status> {
        if !low_threshold.is_finite() || !high_threshold.is_finite() {
            return Err(Status::NanInf);
        }
        if low_threshold > high_threshold {
            return Err(Status::ArgumentError);
        }
        Ok(Self {
            low_threshold,
            high_threshold,
            state: initial_state,
        })
    }

    /// Updates the comparator with a new sample and returns the resulting state.
    ///
    /// NaN samples are ignored and the previous state silently holds.
    #[inline]
    pub fn update(&mut self, input: f32) -> bool {
        if input.is_nan() {
            return self.state;
        }
        if input >= self.high_threshold {
            self.state = true;
        } else if input <= self.low_threshold {
            self.state = false;
        }
        self.state
    }

    /// Updates the comparator with non-finite sample check.
    #[inline]
    pub fn update_checked(&mut self, input: f32) -> Result<bool, Status> {
        if !input.is_finite() {
            return Err(Status::NanInf);
        }
        Ok(self.update(input))
    }

    /// Manually resets the state.
    #[inline]
    pub fn reset(&mut self, state: bool) {
        self.state = state;
    }

    /// Returns the current state without feeding a sample.
    #[inline]
    pub fn state(&self) -> bool {
        self.state
    }
}

impl Hysteresis<f64> {
    /// Creates a new Hysteresis comparator for `f64`.
    pub fn new(low_threshold: f64, high_threshold: f64, initial_state: bool) -> Result<Self, Status> {
        if !low_threshold.is_finite() || !high_threshold.is_finite() {
            return Err(Status::NanInf);
        }
        if low_threshold > high_threshold {
            return Err(Status::ArgumentError);
        }
        Ok(Self {
            low_threshold,
            high_threshold,
            state: initial_state,
        })
    }

    /// Updates the comparator with a new sample and returns the resulting state.
    #[inline]
    pub fn update(&mut self, input: f64) -> bool {
        if input.is_nan() {
            return self.state;
        }
        if input >= self.high_threshold {
            self.state = true;
        } else if input <= self.low_threshold {
            self.state = false;
        }
        self.state
    }

    /// Updates the comparator with non-finite sample check.
    #[inline]
    pub fn update_checked(&mut self, input: f64) -> Result<bool, Status> {
        if !input.is_finite() {
            return Err(Status::NanInf);
        }
        Ok(self.update(input))
    }

    /// Manually resets the state.
    #[inline]
    pub fn reset(&mut self, state: bool) {
        self.state = state;
    }

    /// Returns the current state without feeding a sample.
    #[inline]
    pub fn state(&self) -> bool {
        self.state
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SlewRateLimiter
// ─────────────────────────────────────────────────────────────────────────────

/// Slew rate limiter signal conditioning block.
///
/// Bounds the maximum rate of change (derivative) of a signal.
/// Prevents mechanical strain, actuator saturation, or sudden current spikes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct SlewRateLimiter<T> {
    /// Maximum positive delta allowed per sample.
    pub max_rise: T,
    /// Maximum negative delta allowed per sample (positive magnitude).
    pub max_fall: T,
    /// Current filter state.
    pub state: T,
    /// Whether the limiter has been seeded with an initial sample.
    pub initialized: bool,
}

/// Convenience alias for single-precision float slew rate limiter.
pub type SlewRateLimiterF32 = SlewRateLimiter<f32>;
/// Convenience alias for double-precision float slew rate limiter.
pub type SlewRateLimiterF64 = SlewRateLimiter<f64>;

impl SlewRateLimiter<f32> {
    /// Constructs a limiter from per-second rate limits and sample rate in Hz.
    pub fn asymmetric(rise_per_sec: f32, fall_per_sec: f32, sample_rate: f32) -> Result<Self, Status> {
        if !rise_per_sec.is_finite() || !fall_per_sec.is_finite() || !sample_rate.is_finite() {
            return Err(Status::NanInf);
        }
        if rise_per_sec <= 0.0 || fall_per_sec <= 0.0 || sample_rate <= 0.0 {
            return Err(Status::ArgumentError);
        }
        let max_rise = rise_per_sec / sample_rate;
        let max_fall = fall_per_sec / sample_rate;
        Self::from_per_sample(max_rise, max_fall)
    }

    /// Constructs a symmetric limiter with identical rise and fall limits.
    pub fn symmetric(rate_per_sec: f32, sample_rate: f32) -> Result<Self, Status> {
        Self::asymmetric(rate_per_sec, rate_per_sec, sample_rate)
    }

    /// Constructs a limiter directly from maximum per-sample step sizes.
    pub fn from_per_sample(max_rise: f32, max_fall: f32) -> Result<Self, Status> {
        if !max_rise.is_finite() || !max_fall.is_finite() {
            return Err(Status::NanInf);
        }
        if max_rise <= 0.0 || max_fall <= 0.0 {
            return Err(Status::ArgumentError);
        }
        Ok(Self {
            max_rise,
            max_fall,
            state: 0.0,
            initialized: false,
        })
    }

    /// Filters a target sample, clamping rate of change to configured bounds.
    ///
    /// The first sample seeds the state immediately, eliminating initial jump delays.
    #[inline]
    pub fn filter(&mut self, target: f32) -> f32 {
        if !self.initialized {
            self.state = target;
            self.initialized = true;
            return self.state;
        }
        let diff = target - self.state;
        if diff > self.max_rise {
            self.state += self.max_rise;
        } else if diff < -self.max_fall {
            self.state -= self.max_fall;
        } else {
            self.state = target;
        }
        self.state
    }

    /// Filters with non-finite sample check.
    #[inline]
    pub fn filter_checked(&mut self, target: f32) -> Result<f32, Status> {
        if !target.is_finite() {
            return Err(Status::NanInf);
        }
        Ok(self.filter(target))
    }

    /// Resets the limiter to unseeded state.
    #[inline]
    pub fn reset(&mut self) {
        self.state = 0.0;
        self.initialized = false;
    }

    /// Returns current state without updating.
    #[inline]
    pub fn value(&self) -> f32 {
        self.state
    }
}

impl SlewRateLimiter<f64> {
    /// Constructs an `f64` asymmetric limiter.
    pub fn asymmetric(rise_per_sec: f64, fall_per_sec: f64, sample_rate: f64) -> Result<Self, Status> {
        if !rise_per_sec.is_finite() || !fall_per_sec.is_finite() || !sample_rate.is_finite() {
            return Err(Status::NanInf);
        }
        if rise_per_sec <= 0.0 || fall_per_sec <= 0.0 || sample_rate <= 0.0 {
            return Err(Status::ArgumentError);
        }
        let max_rise = rise_per_sec / sample_rate;
        let max_fall = fall_per_sec / sample_rate;
        Self::from_per_sample(max_rise, max_fall)
    }

    /// Constructs an `f64` symmetric limiter.
    pub fn symmetric(rate_per_sec: f64, sample_rate: f64) -> Result<Self, Status> {
        Self::asymmetric(rate_per_sec, rate_per_sec, sample_rate)
    }

    /// Constructs an `f64` limiter from per-sample limits.
    pub fn from_per_sample(max_rise: f64, max_fall: f64) -> Result<Self, Status> {
        if !max_rise.is_finite() || !max_fall.is_finite() {
            return Err(Status::NanInf);
        }
        if max_rise <= 0.0 || max_fall <= 0.0 {
            return Err(Status::ArgumentError);
        }
        Ok(Self {
            max_rise,
            max_fall,
            state: 0.0,
            initialized: false,
        })
    }

    /// Filters a target sample for `f64`.
    #[inline]
    pub fn filter(&mut self, target: f64) -> f64 {
        if !self.initialized {
            self.state = target;
            self.initialized = true;
            return self.state;
        }
        let diff = target - self.state;
        if diff > self.max_rise {
            self.state += self.max_rise;
        } else if diff < -self.max_fall {
            self.state -= self.max_fall;
        } else {
            self.state = target;
        }
        self.state
    }

    /// Filters with non-finite sample check for `f64`.
    #[inline]
    pub fn filter_checked(&mut self, target: f64) -> Result<f64, Status> {
        if !target.is_finite() {
            return Err(Status::NanInf);
        }
        Ok(self.filter(target))
    }

    /// Resets the limiter.
    #[inline]
    pub fn reset(&mut self) {
        self.state = 0.0;
        self.initialized = false;
    }

    /// Returns current state.
    #[inline]
    pub fn value(&self) -> f64 {
        self.state
    }
}

#[cfg(feature = "pipeline")]
impl Process<f32, f32> for SlewRateLimiter<f32> {
    #[inline]
    fn process(&mut self, sample: f32) -> f32 {
        self.filter(sample)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RunningMedian
// ─────────────────────────────────────────────────────────────────────────────

/// Real-time streaming sliding-window median filter.
///
/// Features:
/// - Fixed-size const-generic stack storage (zero heap allocations).
/// - Requires an odd window length (`WINDOW >= 1 && WINDOW % 2 == 1`).
/// - Auto-initialization: seeds with the first sample to avoid startup lag.
/// - Outlier/NaN rejection: NaN samples do not pollute the window; the filter holds the previous median.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunningMedian<const WINDOW: usize, T> {
    samples: [T; WINDOW],
    next: usize,
    count: usize,
    initialized: bool,
}

/// Convenience alias for single-precision running median.
pub type RunningMedianF32<const WINDOW: usize> = RunningMedian<WINDOW, f32>;
/// Convenience alias for double-precision running median.
pub type RunningMedianF64<const WINDOW: usize> = RunningMedian<WINDOW, f64>;

impl<const WINDOW: usize> RunningMedian<WINDOW, f32> {
    /// Creates a new empty `RunningMedian` filter.
    ///
    /// Returns [`Status::ArgumentError`] if `WINDOW` is even or zero.
    pub fn new() -> Result<Self, Status> {
        if WINDOW == 0 || WINDOW.is_multiple_of(2) {
            return Err(Status::ArgumentError);
        }
        Ok(Self {
            samples: [0.0; WINDOW],
            next: 0,
            count: 0,
            initialized: false,
        })
    }

    /// Feeds one sample and returns the median of the current window.
    ///
    /// NaN samples are rejected without modifying the window; the filter holds its current median.
    #[inline]
    pub fn filter(&mut self, input: f32) -> f32 {
        if !input.is_nan() {
            if !self.initialized {
                self.samples.fill(input);
                self.initialized = true;
                self.count = WINDOW;
                self.next = 0;
            } else {
                self.samples[self.next] = input;
                self.next = (self.next + 1) % WINDOW;
                if self.count < WINDOW {
                    self.count += 1;
                }
            }
        }
        self.value()
    }

    /// Feeds one sample with non-finite sample check.
    ///
    /// Returns [`Status::NanInf`] if `input` is NaN.
    #[inline]
    pub fn filter_checked(&mut self, input: f32) -> Result<f32, Status> {
        if input.is_nan() {
            return Err(Status::NanInf);
        }
        Ok(self.filter(input))
    }

    /// Returns the current median value of the window.
    #[inline]
    pub fn value(&self) -> f32 {
        if !self.initialized || self.count == 0 {
            return 0.0;
        }
        let mut sorted = self.samples;
        // Insertion sort on stack
        for i in 1..WINDOW {
            let key = sorted[i];
            let mut j = i;
            while j > 0 && (sorted[j - 1] > key || sorted[j - 1].is_nan()) {
                sorted[j] = sorted[j - 1];
                j -= 1;
            }
            sorted[j] = key;
        }
        sorted[WINDOW / 2]
    }

    /// Resets the filter.
    #[inline]
    pub fn reset(&mut self) {
        self.samples = [0.0; WINDOW];
        self.next = 0;
        self.count = 0;
        self.initialized = false;
    }
}

impl<const WINDOW: usize> RunningMedian<WINDOW, f64> {
    /// Creates a new empty `RunningMedian` filter for `f64`.
    pub fn new() -> Result<Self, Status> {
        if WINDOW == 0 || WINDOW.is_multiple_of(2) {
            return Err(Status::ArgumentError);
        }
        Ok(Self {
            samples: [0.0; WINDOW],
            next: 0,
            count: 0,
            initialized: false,
        })
    }

    /// Feeds one sample and returns current median.
    #[inline]
    pub fn filter(&mut self, input: f64) -> f64 {
        if !input.is_nan() {
            if !self.initialized {
                self.samples.fill(input);
                self.initialized = true;
                self.count = WINDOW;
                self.next = 0;
            } else {
                self.samples[self.next] = input;
                self.next = (self.next + 1) % WINDOW;
                if self.count < WINDOW {
                    self.count += 1;
                }
            }
        }
        self.value()
    }

    /// Feeds one sample with non-finite sample check.
    #[inline]
    pub fn filter_checked(&mut self, input: f64) -> Result<f64, Status> {
        if input.is_nan() {
            return Err(Status::NanInf);
        }
        Ok(self.filter(input))
    }

    /// Returns the current median value.
    #[inline]
    pub fn value(&self) -> f64 {
        if !self.initialized || self.count == 0 {
            return 0.0;
        }
        let mut sorted = self.samples;
        for i in 1..WINDOW {
            let key = sorted[i];
            let mut j = i;
            while j > 0 && (sorted[j - 1] > key || sorted[j - 1].is_nan()) {
                sorted[j] = sorted[j - 1];
                j -= 1;
            }
            sorted[j] = key;
        }
        sorted[WINDOW / 2]
    }

    /// Resets the filter.
    #[inline]
    pub fn reset(&mut self) {
        self.samples = [0.0; WINDOW];
        self.next = 0;
        self.count = 0;
        self.initialized = false;
    }
}

#[cfg(feature = "pipeline")]
impl<const WINDOW: usize> Process<f32, f32> for RunningMedian<WINDOW, f32> {
    #[inline]
    fn process(&mut self, sample: f32) -> f32 {
        self.filter(sample)
    }
}

#[inline]
fn int_power(base: f32, exp: usize) -> f32 {
    let mut result = 1.0f32;
    for _ in 0..exp {
        result *= base;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// SavitzkyGolay
// ─────────────────────────────────────────────────────────────────────────────

/// Savitzky-Golay streaming polynomial smoothing and derivative estimator.
///
/// Fits a local polynomial of degree `TERMS - 1` across a moving window of `WINDOW` samples.
/// Simultaneously estimates:
/// - Smoothed signal value
/// - First derivative (velocity / rate of change)
/// - Second derivative (acceleration / curvature)
///
/// Supports two read modes:
/// - **Latest (0-delay)**: evaluates polynomial at the newest sample, introducing zero phase delay for live feedback control loops.
/// - **Centered**: evaluates polynomial at the center of the window, providing optimal smoothing with a constant delay of `(WINDOW - 1) / 2` samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SavitzkyGolay<const WINDOW: usize, const TERMS: usize, T> {
    buffer: [T; WINDOW],
    next: usize,
    count: usize,
    initialized: bool,
    value_coeffs: [T; WINDOW],
    deriv1_coeffs: [T; WINDOW],
    deriv2_coeffs: [T; WINDOW],
    dt: T,
    centered: bool,
}

/// Convenience alias for single-precision Savitzky-Golay filter.
pub type SavitzkyGolayF32<const WINDOW: usize, const TERMS: usize> = SavitzkyGolay<WINDOW, TERMS, f32>;
/// Convenience alias for double-precision Savitzky-Golay filter.
pub type SavitzkyGolayF64<const WINDOW: usize, const TERMS: usize> = SavitzkyGolay<WINDOW, TERMS, f64>;

impl<const WINDOW: usize, const TERMS: usize> SavitzkyGolay<WINDOW, TERMS, f32> {
    /// Constructs a Savitzky-Golay filter in **Latest (zero-latency)** mode.
    ///
    /// Returns [`Status::ArgumentError`] if `WINDOW < TERMS`, `TERMS < 1`, `TERMS > 6`, or `dt <= 0.0`.
    pub fn latest(dt: f32) -> Result<Self, Status> {
        Self::build(dt, false)
    }

    /// Constructs a Savitzky-Golay filter in **Centered** mode (delay = `(WINDOW - 1) / 2 * dt`).
    pub fn centered(dt: f32) -> Result<Self, Status> {
        if WINDOW.is_multiple_of(2) {
            return Err(Status::ArgumentError);
        }
        Self::build(dt, true)
    }

    fn build(dt: f32, centered: bool) -> Result<Self, Status> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Status::ArgumentError);
        }
        if WINDOW < TERMS || TERMS < 1 || TERMS > 6 {
            return Err(Status::ArgumentError);
        }

        // Compute coordinate offsets t_i for each sample in the window.
        // In latest mode: newest sample at index WINDOW - 1 is t = 0; older samples are at negative t.
        // In centered mode: center sample is at t = 0.
        let mut t = [0.0f32; WINDOW];
        for i in 0..WINDOW {
            if centered {
                t[i] = (i as f32) - ((WINDOW - 1) as f32) * 0.5;
            } else {
                t[i] = (i as f32) - ((WINDOW - 1) as f32);
            }
        }

        // Form normal equations matrix M = A^T A (TERMS x TERMS)
        let mut m = [[0.0f32; 6]; 6];
        for r in 0..TERMS {
            for c in 0..TERMS {
                let mut sum = 0.0f32;
                for i in 0..WINDOW {
                    let p_r = int_power(t[i], r);
                    let p_c = int_power(t[i], c);
                    sum += p_r * p_c;
                }
                m[r][c] = sum;
            }
        }

        // Invert M (TERMS x TERMS) using Gauss-Jordan with partial pivoting on stack
        let mut aug = [[0.0f32; 12]; 6];
        for r in 0..TERMS {
            for c in 0..TERMS {
                aug[r][c] = m[r][c];
            }
            aug[r][TERMS + r] = 1.0;
        }

        for i in 0..TERMS {
            let mut pivot_row = i;
            let mut max_val = aug[i][i].abs();
            for r in (i + 1)..TERMS {
                if aug[r][i].abs() > max_val {
                    max_val = aug[r][i].abs();
                    pivot_row = r;
                }
            }
            if max_val < 1e-12 {
                return Err(Status::Singular);
            }
            if pivot_row != i {
                aug.swap(i, pivot_row);
            }

            let pivot = aug[i][i];
            for c in 0..(2 * TERMS) {
                aug[i][c] /= pivot;
            }
            for r in 0..TERMS {
                if r != i {
                    let factor = aug[r][i];
                    for c in 0..(2 * TERMS) {
                        aug[r][c] -= factor * aug[i][c];
                    }
                }
            }
        }

        // Extracted M_inv is in aug[r][TERMS + c]
        let mut m_inv = [[0.0f32; 6]; 6];
        for r in 0..TERMS {
            for c in 0..TERMS {
                m_inv[r][c] = aug[r][TERMS + c];
            }
        }

        // Compute convolution weights: W = M_inv * A^T
        // For value (c_0): row 0 of W
        // For first derivative (c_1 / dt): row 1 of W divided by dt
        // For second derivative (2 * c_2 / dt^2): 2 * row 2 of W divided by dt^2
        let mut value_coeffs = [0.0f32; WINDOW];
        let mut deriv1_coeffs = [0.0f32; WINDOW];
        let mut deriv2_coeffs = [0.0f32; WINDOW];

        for i in 0..WINDOW {
            // value: row 0
            let mut val_w = 0.0f32;
            for k in 0..TERMS {
                val_w += m_inv[0][k] * int_power(t[i], k);
            }
            value_coeffs[i] = val_w;

            // 1st derivative: row 1 / dt
            if TERMS >= 2 {
                let mut d1_w = 0.0f32;
                for k in 0..TERMS {
                    d1_w += m_inv[1][k] * int_power(t[i], k);
                }
                deriv1_coeffs[i] = d1_w / dt;
            }

            // 2nd derivative: 2 * row 2 / dt^2
            if TERMS >= 3 {
                let mut d2_w = 0.0f32;
                for k in 0..TERMS {
                    d2_w += m_inv[2][k] * int_power(t[i], k);
                }
                deriv2_coeffs[i] = (2.0 * d2_w) / (dt * dt);
            }
        }

        Ok(Self {
            buffer: [0.0; WINDOW],
            next: 0,
            count: 0,
            initialized: false,
            value_coeffs,
            deriv1_coeffs,
            deriv2_coeffs,
            dt,
            centered,
        })
    }

    /// Feeds one sample and returns the smoothed value.
    pub fn filter(&mut self, sample: f32) -> f32 {
        if !self.initialized {
            self.buffer.fill(sample);
            self.initialized = true;
            self.count = WINDOW;
            self.next = 0;
        } else {
            self.buffer[self.next] = sample;
            self.next = (self.next + 1) % WINDOW;
            if self.count < WINDOW {
                self.count += 1;
            }
        }
        self.value()
    }

    /// Feeds one sample with non-finite sample check.
    pub fn filter_checked(&mut self, sample: f32) -> Result<f32, Status> {
        if !sample.is_finite() {
            return Err(Status::NanInf);
        }
        Ok(self.filter(sample))
    }

    /// Evaluates linear combination of circular buffer samples with given weights.
    #[inline]
    fn evaluate_weights(&self, weights: &[f32; WINDOW]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..WINDOW {
            // index (self.next + i) % WINDOW points from oldest (0) to newest (WINDOW - 1)
            let buf_idx = (self.next + i) % WINDOW;
            sum += weights[i] * self.buffer[buf_idx];
        }
        sum
    }

    /// Returns current smoothed signal value.
    #[inline]
    pub fn value(&self) -> f32 {
        self.evaluate_weights(&self.value_coeffs)
    }

    /// Returns estimated first derivative (rate of change / velocity).
    #[inline]
    pub fn first_derivative(&self) -> f32 {
        self.evaluate_weights(&self.deriv1_coeffs)
    }

    /// Returns estimated second derivative (acceleration / curvature).
    #[inline]
    pub fn second_derivative(&self) -> f32 {
        self.evaluate_weights(&self.deriv2_coeffs)
    }

    /// Returns group delay in seconds introduced by the filter.
    #[inline]
    pub fn delay(&self) -> f32 {
        if self.centered {
            ((WINDOW - 1) as f32 * 0.5) * self.dt
        } else {
            0.0
        }
    }

    /// Resets the filter buffer.
    pub fn reset(&mut self) {
        self.buffer = [0.0; WINDOW];
        self.next = 0;
        self.count = 0;
        self.initialized = false;
    }
}

#[cfg(feature = "pipeline")]
impl<const WINDOW: usize, const TERMS: usize> Process<f32, f32> for SavitzkyGolay<WINDOW, TERMS, f32> {
    #[inline]
    fn process(&mut self, sample: f32) -> f32 {
        self.filter(sample)
    }
}
