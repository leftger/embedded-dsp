//! Signal synthesis and anti-aliased test signal generators.
//!
//! Includes PolyBLEP band-limited oscillators (sine, saw, square, triangle),
//! Kellett pink noise, Xorshift32 white noise, and linear/exponential chirp sweep generators.

#[cfg(all(not(feature = "std"), feature = "libm"))]
use libm::{expf, logf, sinf};

#[cfg(feature = "std")]
fn sinf(val: f32) -> f32 {
    val.sin()
}
#[cfg(feature = "std")]
fn expf(val: f32) -> f32 {
    val.exp()
}
#[cfg(feature = "std")]
fn logf(val: f32) -> f32 {
    val.ln()
}

use core::f32::consts::TAU;

/// Waveform shape for the PolyBLEP oscillator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyBlepWaveform {
    /// Pure sine wave.
    Sine,
    /// Anti-aliased sawtooth wave.
    Sawtooth,
    /// Anti-aliased square wave (50% duty cycle).
    Square,
    /// Anti-aliased triangle wave.
    Triangle,
}

/// Band-limited oscillator using Polynomial Band-Limited Step (PolyBLEP) anti-aliasing.
#[derive(Debug, Clone)]
pub struct PolyBlepOscillator {
    sample_rate: f32,
    frequency: f32,
    phase: f32,
    phase_step: f32,
    waveform: PolyBlepWaveform,
}

impl PolyBlepOscillator {
    /// Creates a new PolyBLEP oscillator.
    pub fn new(sample_rate: f32, frequency: f32, waveform: PolyBlepWaveform) -> Self {
        let mut osc = Self {
            sample_rate: sample_rate.max(1.0),
            frequency: frequency.max(0.0),
            phase: 0.0,
            phase_step: 0.0,
            waveform,
        };
        osc.update_phase_step();
        osc
    }

    /// Sets the frequency in Hz and updates phase step.
    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency.max(0.0);
        self.update_phase_step();
    }

    /// Sets the oscillator waveform shape.
    pub fn set_waveform(&mut self, waveform: PolyBlepWaveform) {
        self.waveform = waveform;
    }

    /// Returns the current frequency.
    pub fn frequency(&self) -> f32 {
        self.frequency
    }

    /// Resets the phase to zero.
    pub fn reset_phase(&mut self) {
        self.phase = 0.0;
    }

    fn update_phase_step(&mut self) {
        self.phase_step = (self.frequency / self.sample_rate).clamp(0.0, 0.5);
    }

    /// Computes the PolyBLEP correction residual for a discontinuity at phase t.
    fn poly_blep(t: f32, dt: f32) -> f32 {
        if dt <= 0.0 {
            return 0.0;
        }
        if t < dt {
            let t = t / dt;
            t + t - t * t - 1.0
        } else if t > 1.0 - dt {
            let t = (t - 1.0) / dt;
            t * t + t + t + 1.0
        } else {
            0.0
        }
    }

    /// Generates the next floating-point audio sample in [-1.0, 1.0].
    pub fn next_sample(&mut self) -> f32 {
        let t = self.phase;
        let dt = self.phase_step;

        let sample = match self.waveform {
            PolyBlepWaveform::Sine => sinf(t * TAU),
            PolyBlepWaveform::Sawtooth => {
                let naive = 2.0 * t - 1.0;
                naive - Self::poly_blep(t, dt)
            }
            PolyBlepWaveform::Square => {
                let naive = if t < 0.5 { 1.0 } else { -1.0 };
                let mut corr = Self::poly_blep(t, dt);
                let t_shifted = (t + 0.5) % 1.0;
                corr -= Self::poly_blep(t_shifted, dt);
                naive + corr
            }
            PolyBlepWaveform::Triangle => {
                let mut sq_corr = Self::poly_blep(t, dt);
                let t_shifted = (t + 0.5) % 1.0;
                sq_corr -= Self::poly_blep(t_shifted, dt);

                // Leaky integration to form triangle
                let tri_naive = if t < 0.5 { 4.0 * t - 1.0 } else { 3.0 - 4.0 * t };
                tri_naive + 0.5 * sq_corr
            }
        };

        self.phase = (self.phase + self.phase_step) % 1.0;
        sample.clamp(-1.0, 1.0)
    }
}

/// Zero-allocation Xorshift32 Pseudo-Random Noise Generator (White Noise).
#[derive(Debug, Clone)]
pub struct WhiteNoise {
    state: u32,
}

impl WhiteNoise {
    /// Creates a new white noise generator with seed.
    pub fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0x1234_5678 } else { seed },
        }
    }

    /// Returns the next white noise sample in [-1.0, 1.0].
    pub fn next_sample(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        (x as f32 / 2147483648.0) - 1.0
    }
}

/// Paul Kellett 1/f Pink Noise Filter Generator.
#[derive(Debug, Clone)]
pub struct KellettPinkNoise {
    white: WhiteNoise,
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    b4: f32,
    b5: f32,
    b6: f32,
}

impl KellettPinkNoise {
    /// Creates a new pink noise generator with seed.
    pub fn new(seed: u32) -> Self {
        Self {
            white: WhiteNoise::new(seed),
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            b3: 0.0,
            b4: 0.0,
            b5: 0.0,
            b6: 0.0,
        }
    }

    /// Returns the next pink noise sample in [-1.0, 1.0].
    pub fn next_sample(&mut self) -> f32 {
        let white = self.white.next_sample();

        self.b0 = 0.99886 * self.b0 + white * 0.0555179;
        self.b1 = 0.99332 * self.b1 + white * 0.0750759;
        self.b2 = 0.96900 * self.b2 + white * 0.153_852;
        self.b3 = 0.86650 * self.b3 + white * 0.3104856;
        self.b4 = 0.55000 * self.b4 + white * 0.5329522;
        self.b5 = -0.7616 * self.b5 - white * 0.0168980;

        let pink = self.b0 + self.b1 + self.b2 + self.b3 + self.b4 + self.b5 + self.b6 + white * 0.5362;
        self.b6 = white * 0.115926;

        (pink * 0.11).clamp(-1.0, 1.0)
    }
}

/// Chirp frequency sweep generator (Linear and Exponential).
#[derive(Debug, Clone)]
pub struct ChirpSweep {
    sample_rate: f32,
    start_freq: f32,
    end_freq: f32,
    duration_secs: f32,
    exponential: bool,
    current_time: f32,
}

impl ChirpSweep {
    /// Creates a new linear or exponential chirp frequency sweep generator.
    pub fn new(sample_rate: f32, start_freq: f32, end_freq: f32, duration_secs: f32, exponential: bool) -> Self {
        Self {
            sample_rate: sample_rate.max(1.0),
            start_freq: start_freq.max(0.1),
            end_freq: end_freq.max(0.1),
            duration_secs: duration_secs.max(0.001),
            exponential,
            current_time: 0.0,
        }
    }

    /// Resets the sweep generator back to time zero.
    pub fn reset(&mut self) {
        self.current_time = 0.0;
    }

    /// Returns the next chirp sweep sample in [-1.0, 1.0].
    pub fn next_sample(&mut self) -> f32 {
        let t = self.current_time;
        let sweep_duration = self.duration_secs;
        let f0 = self.start_freq;
        let f1 = self.end_freq;

        let phase = if self.exponential {
            let rate = logf(f1 / f0) / sweep_duration;
            if rate.abs() < 1e-6 {
                TAU * f0 * t
            } else {
                TAU * f0 * (expf(rate * t) - 1.0) / rate
            }
        } else {
            let k = (f1 - f0) / sweep_duration;
            TAU * (f0 * t + 0.5 * k * t * t)
        };

        let sample = sinf(phase);
        self.current_time += 1.0 / self.sample_rate;
        sample
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exponential Swept-Sine Chirp Generator & Delta-Sigma Accumulator
// ─────────────────────────────────────────────────────────────────────────────

use crate::math::FloatMath;
use crate::types::Complex;

const Q32_F32: f32 = (1i64 << 32) as f32;

/// Parameter errors for [`Sweep::fit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepError {
    /// Start parameter out of bounds or negative state.
    Start,
    /// Stop parameter out of bounds (must be in `0.0..=0.5` Nyquist).
    Stop,
}

impl core::fmt::Display for SweepError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Start => f.write_str("Sweep start parameter out of bounds"),
            Self::Stop => f.write_str("Sweep stop parameter out of bounds"),
        }
    }
}

/// Exponential sweep generator with integrated 1st-order delta-sigma modulator.
///
/// Sweeps exponentially across frequency decades with exact cycle alignment,
/// providing the ideal excitation stimulus for transfer function and impulse response
/// measurement without spectral leakage.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Sweep {
    /// Rate of exponential frequency increase.
    pub rate: i32,
    /// Current 64-bit state with fractional bits for delta-sigma modulation.
    pub state: i64,
}

impl Iterator for Sweep {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        const BIAS: i64 = 1 << 31;
        let s = self.state;
        self.state = s.checked_add(self.rate as i64 * ((s + BIAS) >> 32))?;
        Some(s)
    }
}

impl core::iter::FusedIterator for Sweep {}

impl Sweep {
    /// Create a new exponential sweep from raw rate and 64-bit initial state.
    #[inline]
    pub const fn new(rate: i32, state: i64) -> Self {
        Self { rate, state }
    }

    /// Continuous-time exponential sweep rate.
    #[inline]
    pub fn rate(&self) -> f64 {
        FloatMath::ln(1.0 + self.rate as f64 / ((1i64 << 32) as f64))
    }

    /// Delay/length in samples for a given harmonic.
    #[inline]
    pub fn delay(&self, harmonic: f64) -> f64 {
        FloatMath::ln(harmonic) / self.rate()
    }

    /// Samples per octave.
    #[inline]
    pub fn octave(&self) -> f64 {
        core::f64::consts::LN_2 / self.rate()
    }

    /// Samples per decade.
    #[inline]
    pub fn decade(&self) -> f64 {
        core::f64::consts::LN_10 / self.rate()
    }

    /// Current continuous-time phase state.
    #[inline]
    pub fn state(&self) -> f64 {
        self.cycles() * self.rate()
    }

    /// Number of cycles per harmonic.
    #[inline]
    pub fn cycles(&self) -> f64 {
        self.state as f64 / ((1i64 << 32) as f64 * self.rate as f64)
    }

    /// Evaluate integrated sweep phase at a given sample time `t`.
    #[inline]
    pub fn continuous(&self, t: f64) -> f64 {
        self.cycles() * FloatMath::exp(self.rate() * t)
    }

    /// Synthesize an exponential swept-sine profile.
    ///
    /// # Arguments
    /// * `stop` - Maximum stop frequency in units of sample rate (e.g. 0.5 for Nyquist).
    /// * `harmonics` - Number of harmonics to sweep across (e.g. 1000.0).
    /// * `cycles` - Number of cycles (phase wraps) per harmonic (`>= 1.0`).
    pub fn fit(stop: f32, harmonics: f32, cycles: f32) -> Result<Self, SweepError> {
        if !(0.0..=0.5).contains(&stop) {
            return Err(SweepError::Stop);
        }
        let exp_term = FloatMath::exp(stop / (cycles * harmonics)) - 1.0;
        let rate = (Q32_F32 * exp_term) as i32;
        let state = (rate as i64 * cycles as i64) << 32;
        if state <= 0 {
            return Err(SweepError::Start);
        }
        Ok(Self::new(rate, state))
    }
}

/// Exponentially swept sine oscillator with 64-bit phase accumulator.
#[derive(Clone, Debug)]
pub struct AccuOsc<T> {
    sweep: T,
    accu: i64,
}

impl<T> AccuOsc<T> {
    /// Create a new swept oscillator wrapping a sweep iterator.
    pub const fn new(sweep: T) -> Self {
        Self {
            sweep,
            accu: 0,
        }
    }

    /// Current 64-bit phase accumulator state.
    pub const fn state(&self) -> i64 {
        self.accu
    }
}

impl<T: Iterator<Item = i64>> Iterator for AccuOsc<T> {
    type Item = Complex<i32>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.sweep.next().map(|p| {
            self.accu = self.accu.wrapping_add(p);
            let phase = (self.accu >> 32) as i32;
            #[cfg(feature = "fast-math")]
            {
                let (c, s) = crate::fast_math::cossin(phase);
                Complex::new(c, s)
            }
            #[cfg(not(feature = "fast-math"))]
            {
                let rad = phase as f32 * (core::f32::consts::PI / 2147483648.0);
                let c = (FloatMath::cos(rad) * 2147483647.0) as i32;
                let s = (FloatMath::sin(rad) * 2147483647.0) as i32;
                Complex::new(c, s)
            }
        })
    }
}

impl<T: core::iter::FusedIterator + Iterator<Item = i64>> core::iter::FusedIterator for AccuOsc<T> {}

// ─────────────────────────────────────────────────────────────────────────────
// Generic Accumulator / NCO iterator (ported from idsp)
// ─────────────────────────────────────────────────────────────────────────────

/// Generic wrapping accumulator / Numerically Controlled Oscillator (NCO).
///
/// An infinite `Iterator` that yields phase values by adding a fixed `step`
/// to its `state` on every call to `next()`. Overflow wraps naturally, making
/// it ideal for phase accumulators in PLLs, NCOs, and test-signal generators.
///
/// # Type
/// Use `core::num::Wrapping<i32>` (or any integer) for a wrapping phase
/// accumulator, or `f32`/`f64` for a floating-point ramp.
///
/// # Algebra
/// `Accu<T>` supports `Add`, `Sub`, and `Mul<T>` so accumulators can be
/// composed: `a + b` gives an accumulator whose state and step are sums of
/// the originals.
///
/// # Example
///
/// ```rust
/// # use embedded_dsp::synthesis::Accu;
/// use core::num::Wrapping;
/// let mut nco = Accu::new(Wrapping(0i32), Wrapping(0x0800_0000i32)); // ~6.25% of full-scale
/// let p0 = nco.next().unwrap();
/// assert_eq!(p0, Wrapping(0x0800_0000i32));
/// ```
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
pub struct Accu<T> {
    /// Current accumulator state (phase).
    pub state: T,
    /// Phase increment per step.
    pub step: T,
}

impl<T> Accu<T> {
    /// Create a new accumulator with the given initial state and step.
    pub const fn new(state: T, step: T) -> Self {
        Self { state, step }
    }
}

impl<T: Copy + core::ops::AddAssign> Iterator for Accu<T> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.state += self.step;
        Some(self.state)
    }
}

impl<T: Copy + core::ops::Mul<Output = T>> core::ops::Mul<T> for Accu<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self::new(self.state * rhs, self.step * rhs)
    }
}

impl<T: Copy + core::ops::Add<Output = T>> core::ops::Add for Accu<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.state + rhs.state, self.step + rhs.step)
    }
}

impl<T: Copy + core::ops::Sub<Output = T>> core::ops::Sub for Accu<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.state - rhs.state, self.step - rhs.step)
    }
}


