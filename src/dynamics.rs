//! Dynamics Range Control: Compressor, Limiter, Expander, and Noise Gate.
//!
//! Provides real-time dynamics processing with soft-knee curves, decoupled attack/release
//! ballistics, and integration with the [`DspNode`](crate::pipeline::DspNode) streaming framework.

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
