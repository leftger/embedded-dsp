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
        self.b2 = 0.96900 * self.b2 + white * 0.1538520;
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
