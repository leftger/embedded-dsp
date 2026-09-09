//! State Variable Filter with simultaneous lowpass, highpass, bandpass, notch, and peak outputs.
//!
//! Unlike a biquad, the cutoff and resonance can be swept every sample without recomputing a
//! coefficient set, and all five responses are available from a single [`process`](StateVariableFilter::process)
//! call. Ported from Andrew Simper's "Double Sampled, Stable State Variable Filter"
//! (musicdsp.org), which internally runs two half-rate passes per sample for stability at high
//! cutoff/resonance settings.

#[allow(unused_imports)]
use crate::math::FloatMath;

/// Simultaneous low/high/band/notch/peak state-variable filter.
#[derive(Debug, Clone, Copy)]
pub struct StateVariableFilter {
    sample_rate_hz: f32,
    cutoff_max_hz: f32,
    resonance: f32,
    pre_drive: f32,
    drive: f32,
    freq: f32,
    damp: f32,
    low: f32,
    high: f32,
    band: f32,
    notch: f32,
    out_low: f32,
    out_high: f32,
    out_band: f32,
    out_notch: f32,
    out_peak: f32,
}

impl StateVariableFilter {
    /// Creates a filter for `sample_rate_hz`, defaulting to a 200 Hz cutoff and 0.5 resonance.
    pub fn new(sample_rate_hz: f32) -> Self {
        let mut svf = Self {
            sample_rate_hz,
            cutoff_max_hz: sample_rate_hz / 3.0,
            resonance: 0.5,
            pre_drive: 0.5,
            drive: 0.0,
            freq: 0.25,
            damp: 0.0,
            low: 0.0,
            high: 0.0,
            band: 0.0,
            notch: 0.0,
            out_low: 0.0,
            out_high: 0.0,
            out_band: 0.0,
            out_notch: 0.0,
            out_peak: 0.0,
        };
        svf.set_cutoff(200.0);
        svf.set_resonance(0.5);
        svf
    }

    /// Sets the cutoff frequency in Hz. Clamped to `(0, sample_rate_hz / 3]` to keep the
    /// double-sampled topology stable.
    pub fn set_cutoff(&mut self, cutoff_hz: f32) {
        let cutoff_hz = cutoff_hz.clamp(1.0e-6, self.cutoff_max_hz);
        // *2.0 because the filter runs two half-rate passes per input sample.
        self.freq = 2.0
            * (core::f32::consts::PI * (cutoff_hz / (self.sample_rate_hz * 2.0)).min(0.25)).sin();
        self.recompute_damp();
    }

    /// Sets the resonance, clamped to `[0.0, 1.0]` to guarantee stability.
    pub fn set_resonance(&mut self, resonance: f32) {
        self.resonance = resonance.clamp(0.0, 1.0);
        self.recompute_damp();
        self.drive = self.pre_drive * self.resonance;
    }

    /// Sets the drive, which shapes how hard the resonant peak saturates. Typical range `0.0..=1.0`.
    pub fn set_drive(&mut self, drive: f32) {
        self.pre_drive = (drive * 0.1).clamp(0.0, 1.0);
        self.drive = self.pre_drive * self.resonance;
    }

    fn recompute_damp(&mut self) {
        let res_damp = 2.0 * (1.0 - self.resonance.powf(0.25));
        let freq_damp = (2.0 / self.freq - self.freq * 0.5).min(2.0);
        self.damp = res_damp.min(freq_damp);
    }

    /// Processes one input sample, updating all five simultaneous outputs.
    pub fn process(&mut self, input: f32) {
        self.pass(input);
        self.out_low = 0.5 * self.low;
        self.out_high = 0.5 * self.high;
        self.out_band = 0.5 * self.band;
        self.out_peak = 0.5 * (self.low - self.high);
        self.out_notch = 0.5 * self.notch;

        self.pass(input);
        self.out_low += 0.5 * self.low;
        self.out_high += 0.5 * self.high;
        self.out_band += 0.5 * self.band;
        self.out_peak += 0.5 * (self.low - self.high);
        self.out_notch += 0.5 * self.notch;
    }

    #[inline]
    fn pass(&mut self, input: f32) {
        self.notch = input - self.damp * self.band;
        self.low += self.freq * self.band;
        self.high = self.notch - self.low;
        self.band += self.freq * self.high - self.drive * self.band * self.band * self.band;

        // At a lightly-damped resonance near the cutoff ceiling (`sample_rate_hz / 3`), the
        // cubic drive term isn't always enough to keep this loop from numerically diverging,
        // particularly at low sample rates where realistic cutoffs sit closer to that ceiling.
        // Clamp generously — well outside any level this filter produces in normal operation —
        // so a runaway degrades to a bounded, loud output instead of NaN/Inf.
        const STATE_LIMIT: f32 = 1.0e3;
        self.low = self.low.clamp(-STATE_LIMIT, STATE_LIMIT);
        self.high = self.high.clamp(-STATE_LIMIT, STATE_LIMIT);
        self.band = self.band.clamp(-STATE_LIMIT, STATE_LIMIT);
        self.notch = self.notch.clamp(-STATE_LIMIT, STATE_LIMIT);
    }

    /// Lowpass output from the most recent [`process`](Self::process) call.
    pub fn low(&self) -> f32 {
        self.out_low
    }

    /// Highpass output from the most recent [`process`](Self::process) call.
    pub fn high(&self) -> f32 {
        self.out_high
    }

    /// Bandpass output from the most recent [`process`](Self::process) call.
    pub fn band(&self) -> f32 {
        self.out_band
    }

    /// Notch (band-stop) output from the most recent [`process`](Self::process) call.
    pub fn notch(&self) -> f32 {
        self.out_notch
    }

    /// Peak output from the most recent [`process`](Self::process) call.
    pub fn peak(&self) -> f32 {
        self.out_peak
    }

    /// Resets the filter's internal state. Cutoff, resonance, and drive are left unchanged.
    pub fn reset(&mut self) {
        self.low = 0.0;
        self.high = 0.0;
        self.band = 0.0;
        self.notch = 0.0;
        self.out_low = 0.0;
        self.out_high = 0.0;
        self.out_band = 0.0;
        self.out_notch = 0.0;
        self.out_peak = 0.0;
    }
}
