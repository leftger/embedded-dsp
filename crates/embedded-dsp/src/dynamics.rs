//! Dynamics Range Control: Compressor, Limiter, Expander, and Noise Gate.
//!
//! Provides real-time dynamics processing with soft-knee curves, decoupled attack/release
//! ballistics, and integration with the [`DspNode`] streaming framework.

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::pipeline::DspNode;

/// Dynamics Range Compressor with soft knee and make-up gain.
#[derive(Debug, Clone, Copy)]
pub struct DynamicsCompressor {
    threshold_db: f32,
    ratio: f32,
    knee_db: f32,
    makeup_gain_linear: f32,
    attack_coeff: f32,
    release_coeff: f32,
    envelope_db: f32,
}

impl DynamicsCompressor {
    /// Creates a new dynamics compressor.
    ///
    /// - `threshold_db`: Threshold level in dBFS (e.g. `-20.0`).
    /// - `ratio`: Compression ratio (e.g. `4.0` for 4:1).
    /// - `knee_db`: Soft knee width in dB (e.g. `6.0` for smooth transition, `0.0` for hard knee).
    /// - `attack_s`: Attack time in seconds (e.g. `0.005` for 5 ms).
    /// - `release_s`: Release time in seconds (e.g. `0.1` for 100 ms).
    /// - `makeup_gain_db`: Post-compression make-up gain in dB (e.g. `4.0`).
    /// - `sample_rate_hz`: Audio sample rate in Hz (e.g. `48000.0`).
    pub fn new(
        threshold_db: f32,
        ratio: f32,
        knee_db: f32,
        attack_s: f32,
        release_s: f32,
        makeup_gain_db: f32,
        sample_rate_hz: f32,
    ) -> Self {
        let attack_coeff = (-1.0 / (attack_s.max(1e-5) * sample_rate_hz)).exp();
        let release_coeff = (-1.0 / (release_s.max(1e-5) * sample_rate_hz)).exp();
        let makeup_gain_linear = (10.0f32).powf(makeup_gain_db / 20.0);

        Self {
            threshold_db,
            ratio: ratio.max(1.0),
            knee_db: knee_db.max(0.0),
            makeup_gain_linear,
            attack_coeff,
            release_coeff,
            envelope_db: 0.0,
        }
    }

    /// Process a single audio/signal sample through the compressor.
    pub fn process(&mut self, input: f32) -> f32 {
        let abs_in = input.abs();
        let input_db = if abs_in > 1e-6 {
            20.0 * abs_in.log10()
        } else {
            -120.0
        };

        // Static compression characteristic with quadratic soft knee
        let target_gain_db = if self.knee_db > 0.0
            && (2.0 * (input_db - self.threshold_db)).abs() <= self.knee_db
        {
            let delta = input_db - self.threshold_db + self.knee_db / 2.0;
            -(1.0 - 1.0 / self.ratio) * delta * delta / (2.0 * self.knee_db)
        } else if input_db > self.threshold_db {
            -(input_db - self.threshold_db) * (1.0 - 1.0 / self.ratio)
        } else {
            0.0
        };

        // Smooth gain change via attack/release ballistics
        if target_gain_db < self.envelope_db {
            // Attack (gain decreasing / compressing)
            self.envelope_db = self.attack_coeff * self.envelope_db + (1.0 - self.attack_coeff) * target_gain_db;
        } else {
            // Release (gain restoring)
            self.envelope_db = self.release_coeff * self.envelope_db + (1.0 - self.release_coeff) * target_gain_db;
        }

        let gain_linear = (10.0f32).powf(self.envelope_db / 20.0) * self.makeup_gain_linear;
        input * gain_linear
    }

    /// Reset compressor internal state.
    pub fn reset(&mut self) {
        self.envelope_db = 0.0;
    }
}

impl DspNode<f32> for DynamicsCompressor {
    #[inline(always)]
    fn process_sample(&mut self, input: f32) -> f32 {
        self.process(input)
    }
}

/// Noise Gate for ambient noise and low-level hum suppression.
#[derive(Debug, Clone, Copy)]
pub struct NoiseGate {
    threshold_db: f32,
    reduction_linear: f32,
    attack_coeff: f32,
    release_coeff: f32,
    envelope_linear: f32,
}

impl NoiseGate {
    /// Create a new noise gate.
    ///
    /// - `threshold_db`: Gate open threshold (e.g. `-45.0` dBFS).
    /// - `reduction_db`: Maximum attenuation when closed (e.g. `-40.0` dB).
    /// - `attack_s`: Opening time (e.g. `0.002` s).
    /// - `release_s`: Closing time (e.g. `0.05` s).
    pub fn new(threshold_db: f32, reduction_db: f32, attack_s: f32, release_s: f32, sample_rate_hz: f32) -> Self {
        let attack_coeff = (-1.0 / (attack_s.max(1e-5) * sample_rate_hz)).exp();
        let release_coeff = (-1.0 / (release_s.max(1e-5) * sample_rate_hz)).exp();
        let reduction_linear = (10.0f32).powf(reduction_db / 20.0);

        Self {
            threshold_db,
            reduction_linear,
            attack_coeff,
            release_coeff,
            envelope_linear: 0.0,
        }
    }

    /// Process a sample through the noise gate.
    pub fn process(&mut self, input: f32) -> f32 {
        let abs_in = input.abs();
        let input_db = if abs_in > 1e-6 {
            20.0 * abs_in.log10()
        } else {
            -120.0
        };

        let target_gain = if input_db >= self.threshold_db {
            1.0
        } else {
            self.reduction_linear
        };

        if target_gain > self.envelope_linear {
            self.envelope_linear = self.attack_coeff * self.envelope_linear + (1.0 - self.attack_coeff) * target_gain;
        } else {
            self.envelope_linear = self.release_coeff * self.envelope_linear + (1.0 - self.release_coeff) * target_gain;
        }

        input * self.envelope_linear
    }

    /// Reset noise gate internal states.
    pub fn reset(&mut self) {
        self.envelope_linear = 0.0;
    }
}

impl DspNode<f32> for NoiseGate {
    #[inline(always)]
    fn process_sample(&mut self, input: f32) -> f32 {
        self.process(input)
    }
}

/// Zero-latency Safety Peak Limiter node to prevent output digital clipping or speaker damage.
#[derive(Debug, Clone, Copy)]
pub struct SafetyLimiter {
    ceiling: f32,
    release_coeff: f32,
    current_gain: f32,
}

impl SafetyLimiter {
    /// Creates a new safety peak limiter.
    ///
    /// - `ceiling`: Maximum absolute amplitude limit (e.g. `0.95`).
    /// - `release_s`: Release recovery time in seconds (e.g. `0.05` s).
    /// - `sample_rate_hz`: Audio sample rate in Hz.
    pub fn new(ceiling: f32, release_s: f32, sample_rate_hz: f32) -> Self {
        let release_coeff = (-1.0 / (release_s.max(1e-5) * sample_rate_hz)).exp();
        Self {
            ceiling: ceiling.abs().clamp(0.01, 100.0),
            release_coeff,
            current_gain: 1.0,
        }
    }

    /// Process a sample through the peak limiter.
    pub fn process(&mut self, input: f32) -> f32 {
        let abs_in = input.abs();
        if abs_in > 1e-6 {
            let required_gain = (self.ceiling / abs_in).min(1.0);
            if required_gain < self.current_gain {
                // Instantaneous peak attack
                self.current_gain = required_gain;
            } else {
                // Smooth exponential release recovery
                self.current_gain = self.release_coeff * self.current_gain + (1.0 - self.release_coeff) * 1.0;
            }
        } else {
            self.current_gain = self.release_coeff * self.current_gain + (1.0 - self.release_coeff) * 1.0;
        }

        (input * self.current_gain).clamp(-self.ceiling, self.ceiling)
    }

    /// Returns the current gain attenuation factor (0.0 to 1.0).
    pub fn current_gain(&self) -> f32 {
        self.current_gain
    }

    /// Resets the limiter gain state back to unity.
    pub fn reset(&mut self) {
        self.current_gain = 1.0;
    }
}

/// Streaming automatic gain control targeting unit output energy (liquid-dsp `agc`).
///
/// `y = g * x`, then `g` is updated from a one-pole estimate of `|y|^2` so the
/// long-run output power is 1. Squelch is omitted; lock freezes the gain.
#[derive(Debug, Clone, Copy)]
pub struct AgcF32 {
    g: f32,
    scale: f32,
    alpha: f32,
    y2_prime: f32,
    locked: bool,
}

impl AgcF32 {
    /// Default loop bandwidth used by liquid-dsp (`1e-2`).
    pub const DEFAULT_BANDWIDTH: f32 = 1e-2;

    /// Creates an unlocked AGC with the given loop bandwidth in `(0, 1]`.
    pub fn new(bandwidth: f32) -> Self {
        let mut agc = Self {
            g: 1.0,
            scale: 1.0,
            alpha: 0.0,
            y2_prime: 1.0,
            locked: false,
        };
        agc.set_bandwidth(bandwidth);
        agc
    }

    /// Sets the loop bandwidth (clamped to `[0, 1]`). `0` freezes adaptation.
    pub fn set_bandwidth(&mut self, bandwidth: f32) {
        self.alpha = bandwidth.clamp(0.0, 1.0);
    }

    /// Current loop bandwidth (the internal one-pole coefficient).
    #[inline]
    pub fn bandwidth(&self) -> f32 {
        self.alpha
    }

    /// Multiplier applied after the AGC gain (`1` by default).
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// Freeze gain updates.
    pub fn lock(&mut self) {
        self.locked = true;
    }

    /// Resume gain updates.
    pub fn unlock(&mut self) {
        self.locked = false;
    }

    /// Whether gain updates are frozen.
    #[inline]
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Current linear gain `g` (before [`Self::set_scale`]).
    #[inline]
    pub fn gain(&self) -> f32 {
        self.g
    }

    /// Smoothed output energy estimate, in dB (`10 log10 y2'`).
    pub fn rssi_db(&self) -> f32 {
        10.0 * self.y2_prime.max(1e-12).log10()
    }

    /// Resets gain and energy estimate to unity and unlocks.
    pub fn reset(&mut self) {
        self.g = 1.0;
        self.y2_prime = 1.0;
        self.locked = false;
    }

    /// Applies AGC to one real sample.
    pub fn process(&mut self, x: f32) -> f32 {
        let mut y = x * self.g;
        let y2 = y * y;
        self.y2_prime = (1.0 - self.alpha) * self.y2_prime + self.alpha * y2;
        if !self.locked && self.y2_prime > 1e-6 {
            self.g *= (-0.5 * self.alpha * self.y2_prime.ln()).exp();
            if self.g > 1e6 {
                self.g = 1e6;
            }
        }
        y *= self.scale;
        y
    }
}

impl DspNode<f32> for AgcF32 {
    #[inline(always)]
    fn process_sample(&mut self, input: f32) -> f32 {
        self.process(input)
    }
}

impl DspNode<f32> for SafetyLimiter {
    #[inline(always)]
    fn process_sample(&mut self, input: f32) -> f32 {
        self.process(input)
    }
}

