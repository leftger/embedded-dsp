//! Audio analysis: the Goertzel single-frequency detector, peak/RMS envelope followers,
//! a Mel filterbank, and MFCC feature extraction. These are DSP front-ends (e.g. for a
//! keyword-spotting pipeline); classifiers and neural nets live in `embedded-nn`.

#[allow(unused_imports)]
use crate::filter_design::single_pole_decay_from_time_constant;
#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::math::isqrt_u64;
use crate::transform::cfft_f32;
use crate::types::{q15, Q8F7, Status};

// --- Goertzel Single-Frequency Detector ---

/// A Goertzel single-frequency detector: computes the DFT magnitude at one target frequency
/// via a simple two-pole recursive filter, without a full FFT. Ideal for detecting a known
/// tone (e.g. DTMF, a pilot tone) from a stream of samples on constrained hardware.
#[derive(Debug, Clone, Copy, Default)]
pub struct GoertzelDetector {
    coeff: f32,
    s_prev: f32,
    s_prev2: f32,
    count: u32,
}

impl GoertzelDetector {
    /// Creates a detector tuned to `target_freq_hz` at the given `sample_rate_hz`.
    pub fn new(target_freq_hz: f32, sample_rate_hz: f32) -> Self {
        let w = 2.0 * core::f32::consts::PI * target_freq_hz / sample_rate_hz;
        Self {
            coeff: 2.0 * w.cos(),
            s_prev: 0.0,
            s_prev2: 0.0,
            count: 0,
        }
    }

    /// Feeds one input sample into the detector.
    #[inline(always)]
    pub fn process_sample(&mut self, x: f32) {
        let s = x + self.coeff * self.s_prev - self.s_prev2;
        self.s_prev2 = self.s_prev;
        self.s_prev = s;
        self.count += 1;
    }

    /// Returns the magnitude of the target-frequency component accumulated so far, normalized
    /// by the number of samples processed so it approximates the input sinusoid's amplitude
    /// regardless of block length.
    pub fn magnitude(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let mag_sq = self.s_prev * self.s_prev + self.s_prev2 * self.s_prev2
            - self.coeff * self.s_prev * self.s_prev2;
        mag_sq.max(0.0).sqrt() / (self.count as f32 / 2.0)
    }

    /// Resets the detector's internal state to start a new detection block.
    pub fn reset(&mut self) {
        self.s_prev = 0.0;
        self.s_prev2 = 0.0;
        self.count = 0;
    }
}

/// Q2.14 fixed-point type for coefficients that can range up to `±2.0`
/// (e.g. `2 cos(ω)`), which does not fit `q15`'s `[-1.0, 1.0)` range.
type Q2F14 = fixed::FixedI16<fixed::types::extra::U14>;

/// Q15 Goertzel detector: same two-pole recurrence as [`GoertzelDetector`], with
/// Q2.14 `2 cos(ω)` and i32 delays so a typical block (`N ≲ 256`) does not wrap.
#[derive(Debug, Clone, Copy, Default)]
pub struct GoertzelDetectorQ15 {
    coeff: Q2F14,
    s_prev: i32,
    s_prev2: i32,
    count: u32,
}

impl GoertzelDetectorQ15 {
    /// Creates a detector tuned to `target_freq_hz` at the given `sample_rate_hz`.
    pub fn new(target_freq_hz: f32, sample_rate_hz: f32) -> Self {
        let w = 2.0 * core::f32::consts::PI * target_freq_hz / sample_rate_hz;
        let coeff = Q2F14::saturating_from_num(2.0 * w.cos());
        Self {
            coeff,
            s_prev: 0,
            s_prev2: 0,
            count: 0,
        }
    }

    /// Feeds one Q15 input sample into the detector.
    #[inline(always)]
    pub fn process_sample(&mut self, x: q15) {
        let s = x.to_bits() as i32
            + ((((self.coeff.to_bits() as i64) * (self.s_prev as i64)) >> 14) as i32)
            - self.s_prev2;
        self.s_prev2 = self.s_prev;
        self.s_prev = s;
        self.count += 1;
    }

    /// Magnitude of the target bin, normalized by `N/2` like [`GoertzelDetector::magnitude`].
    pub fn magnitude(&self) -> q15 {
        if self.count == 0 {
            return q15::ZERO;
        }
        let s = self.s_prev as i64;
        let s2 = self.s_prev2 as i64;
        let c = self.coeff.to_bits() as i64;
        let mag_sq = s * s + s2 * s2 - ((c * s * s2) >> 14);
        if mag_sq <= 0 {
            return q15::ZERO;
        }
        let mag = isqrt_u64(mag_sq as u64);
        let out = (mag * 2) / (self.count as u64);
        q15::from_bits(out.min(32767) as i16)
    }

    /// Resets the detector's internal state to start a new detection block.
    pub fn reset(&mut self) {
        self.s_prev = 0;
        self.s_prev2 = 0;
        self.count = 0;
    }
}

// --- Envelope Followers ---

/// Peak envelope follower with independent attack/release time constants, as used for audio
/// dynamics processing (compressors, limiters, VU-style level meters).
#[derive(Debug, Clone, Copy, Default)]
pub struct PeakEnvelopeFollower {
    attack_coeff: f32,
    release_coeff: f32,
    envelope: f32,
}

impl PeakEnvelopeFollower {
    /// `attack_samples` / `release_samples`: the time constant, in samples, for the envelope
    /// to rise / fall `1 - 1/e` (~63%) of the way to a step change in input level.
    pub fn new(attack_samples: f32, release_samples: f32) -> Self {
        Self {
            attack_coeff: 1.0 - single_pole_decay_from_time_constant(attack_samples),
            release_coeff: 1.0 - single_pole_decay_from_time_constant(release_samples),
            envelope: 0.0,
        }
    }

    /// Processes one input sample and returns the updated envelope value.
    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        let rectified = x.abs();
        let coeff = if rectified > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope += coeff * (rectified - self.envelope);
        self.envelope
    }

    /// Resets the envelope to zero.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

/// RMS envelope follower: a single-pole exponential moving average of instantaneous power,
/// reported as an RMS level.
#[derive(Debug, Clone, Copy, Default)]
pub struct RmsEnvelopeFollower {
    coeff: f32,
    mean_sq: f32,
}

impl RmsEnvelopeFollower {
    /// `time_constant_samples`: the time constant, in samples, of the underlying power
    /// averaging filter.
    pub fn new(time_constant_samples: f32) -> Self {
        Self {
            coeff: 1.0 - single_pole_decay_from_time_constant(time_constant_samples),
            mean_sq: 0.0,
        }
    }

    /// Processes one input sample and returns the updated RMS envelope value.
    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        self.mean_sq += self.coeff * (x * x - self.mean_sq);
        self.mean_sq.max(0.0).sqrt()
    }

    /// Resets the running mean-square to zero.
    pub fn reset(&mut self) {
        self.mean_sq = 0.0;
    }
}

/// Q15 peak envelope follower (same attack/release recurrence as [`PeakEnvelopeFollower`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct PeakEnvelopeFollowerQ15 {
    attack_coeff: q15,
    release_coeff: q15,
    envelope: q15,
}

impl PeakEnvelopeFollowerQ15 {
    pub fn new(attack_samples: f32, release_samples: f32) -> Self {
        let attack = 1.0 - single_pole_decay_from_time_constant(attack_samples);
        let release = 1.0 - single_pole_decay_from_time_constant(release_samples);
        Self {
            attack_coeff: q15::saturating_from_num(attack),
            release_coeff: q15::saturating_from_num(release),
            envelope: q15::ZERO,
        }
    }

    #[inline(always)]
    pub fn process(&mut self, x: q15) -> q15 {
        let rectified = x.to_bits().unsigned_abs() as i32;
        let env = self.envelope.to_bits() as i32;
        let coeff = if rectified > env {
            self.attack_coeff
        } else {
            self.release_coeff
        }
        .to_bits() as i32;
        let y = env + ((coeff * (rectified - env)) >> 15);
        self.envelope = q15::from_bits(y.clamp(0, 32767) as i16);
        self.envelope
    }

    pub fn reset(&mut self) {
        self.envelope = q15::ZERO;
    }
}

/// Q15 RMS envelope follower.
#[derive(Debug, Clone, Copy, Default)]
pub struct RmsEnvelopeFollowerQ15 {
    coeff: q15,
    mean_sq: q15,
}

impl RmsEnvelopeFollowerQ15 {
    pub fn new(time_constant_samples: f32) -> Self {
        let c = 1.0 - single_pole_decay_from_time_constant(time_constant_samples);
        Self {
            coeff: q15::saturating_from_num(c),
            mean_sq: q15::ZERO,
        }
    }

    #[inline(always)]
    pub fn process(&mut self, x: q15) -> q15 {
        let inst = ((x.to_bits() as i32 * x.to_bits() as i32) >> 15).clamp(0, 32767);
        let ms = self.mean_sq.to_bits() as i32;
        let y = ms + ((self.coeff.to_bits() as i32 * (inst - ms)) >> 15);
        self.mean_sq = q15::from_bits(y.clamp(0, 32767) as i16);
        let mag = isqrt_u64((self.mean_sq.to_bits() as u64) << 15);
        q15::from_bits(mag.min(32767) as i16)
    }

    pub fn reset(&mut self) {
        self.mean_sq = q15::ZERO;
    }
}

// --- Mel Filterbank & MFCC ---

/// Converts a frequency in Hz to the Mel scale: `2595 * log10(1 + hz / 700)`.
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Converts a Mel-scale value back to Hz: `700 * (10^(mel / 2595) - 1)`.
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((10.0f32).powf(mel / 2595.0) - 1.0)
}

/// Applies a triangular Mel filterbank to a one-sided power (or magnitude-squared) spectrum,
/// producing one energy value per Mel band — the standard first step of MFCC / speech feature
/// extraction.
///
/// `power_spectrum`: one-sided spectrum of length `fft_size / 2 + 1` (DC through Nyquist).
/// `fft_size`: the FFT length the spectrum was computed with.
/// `sample_rate_hz`: sampling rate in Hz.
/// `low_freq_hz` / `high_freq_hz`: frequency range to cover with Mel bands (`0..=sample_rate/2`).
/// `mel_energies`: destination for the output; its length sets the number of Mel filters `M`
/// (`1..=64`).
pub fn mel_filterbank_f32(
    power_spectrum: &[f32],
    fft_size: usize,
    sample_rate_hz: f32,
    low_freq_hz: f32,
    high_freq_hz: f32,
    mel_energies: &mut [f32],
) -> Status {
    let num_filters = mel_energies.len();
    if num_filters == 0 || num_filters > 64 {
        return Status::ArgumentError;
    }
    let num_bins = fft_size / 2 + 1;
    if power_spectrum.len() < num_bins {
        return Status::LengthError;
    }

    let mel_low = hz_to_mel(low_freq_hz);
    let mel_high = hz_to_mel(high_freq_hz);

    let mut bin_points = [0usize; 66];
    for (i, bp) in bin_points.iter_mut().enumerate().take(num_filters + 2) {
        let mel = mel_low + (mel_high - mel_low) * (i as f32) / (num_filters + 1) as f32;
        let hz = mel_to_hz(mel);
        let bin = (hz * fft_size as f32 / sample_rate_hz) as usize;
        *bp = bin.min(num_bins - 1);
    }

    for (m, out) in mel_energies.iter_mut().enumerate() {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        let mut energy = 0.0f32;
        if center > left {
            let span = (center - left) as f32;
            for bin in left..center {
                energy += ((bin - left) as f32 / span) * power_spectrum[bin];
            }
        }
        if right > center {
            let span = (right - center) as f32;
            for bin in center..=right {
                energy += ((right - bin) as f32 / span) * power_spectrum[bin];
            }
        }
        *out = energy;
    }

    Status::Success
}

/// Computes MFCC (Mel-Frequency Cepstral Coefficient) features from a single real-valued
/// audio frame: FFT power spectrum, Mel filterbank, log compression, and a DCT-II to
/// decorrelate the log-Mel-energies into cepstral coefficients. This is the standard
/// speech/audio feature-extraction pipeline.
///
/// `frame`: `fft_size` real audio samples (already windowed by the caller, e.g. with
/// [`crate::window::hamming_f32`] + [`crate::window::apply_window_f32`]); `fft_size` must be
/// a power of 2, `<= 512`.
/// `mel_energies_scratch`: scratch buffer for the intermediate Mel-filterbank output; its
/// length sets the number of Mel filters used internally (`1..=64`).
/// `mfcc_out`: destination for the resulting cepstral coefficients; its length sets the number
/// of coefficients returned (typically 12-13), and must be `<= mel_energies_scratch.len()`.
pub fn mfcc_f32(
    frame: &[f32],
    sample_rate_hz: f32,
    low_freq_hz: f32,
    high_freq_hz: f32,
    mel_energies_scratch: &mut [f32],
    mfcc_out: &mut [f32],
) -> Status {
    let fft_size = frame.len();
    if fft_size < 2 || (fft_size & (fft_size - 1)) != 0 || 2 * fft_size > 1024 {
        return Status::ArgumentError;
    }
    if mfcc_out.len() > mel_energies_scratch.len() {
        return Status::ArgumentError;
    }

    let mut c_data = [0.0f32; 1024];
    for (i, &x) in frame.iter().enumerate() {
        c_data[2 * i] = x;
        c_data[2 * i + 1] = 0.0;
    }
    cfft_f32(&mut c_data[..2 * fft_size], fft_size, 0, 1);

    let num_bins = fft_size / 2 + 1;
    let mut power_spectrum = [0.0f32; 513];
    for k in 0..num_bins {
        let re = c_data[2 * k];
        let im = c_data[2 * k + 1];
        power_spectrum[k] = re * re + im * im;
    }

    let status = mel_filterbank_f32(
        &power_spectrum[..num_bins],
        fft_size,
        sample_rate_hz,
        low_freq_hz,
        high_freq_hz,
        mel_energies_scratch,
    );
    if status != Status::Success {
        return status;
    }

    for e in mel_energies_scratch.iter_mut() {
        *e = e.max(1e-10).ln();
    }

    let num_mel = mel_energies_scratch.len() as f32;
    for (k, out) in mfcc_out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (m, &log_e) in mel_energies_scratch.iter().enumerate() {
            let angle = core::f32::consts::PI * k as f32 * (m as f32 + 0.5) / num_mel;
            sum += log_e * angle.cos();
        }
        *out = sum;
    }

    Status::Success
}

// ─────────────────────────────────────────────────────────────────────────────
// Generalized Filterbank & Fixed-Point Feature Extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Generalized triangular filterbank applicable to any spectral scale (Linear, Octave, Constant-Q, Bark, Mel).
///
/// Integrates `power_spectrum` against triangular weighting windows defined by parallel slices
/// of `left_bins`, `center_bins`, and `right_bins`.
pub fn generalized_triangular_filterbank(
    power_spectrum: &[f32],
    left_bins: &[usize],
    center_bins: &[usize],
    right_bins: &[usize],
    energies_out: &mut [f32],
) -> Status {
    let num_filters = energies_out.len();
    if left_bins.len() < num_filters
        || center_bins.len() < num_filters
        || right_bins.len() < num_filters
        || num_filters == 0
    {
        return Status::LengthError;
    }

    for (i, energy) in energies_out.iter_mut().enumerate() {
        let left = left_bins[i];
        let center = center_bins[i];
        let right = right_bins[i];

        if left > center || center > right || right >= power_spectrum.len() {
            return Status::ArgumentError;
        }

        let mut sum = 0.0f32;
        if center > left {
            let span = (center - left) as f32;
            for bin in left..=center {
                let weight = (bin - left) as f32 / span;
                sum += weight * power_spectrum[bin];
            }
        }
        if right > center {
            let span = (right - center) as f32;
            for bin in (center + 1)..=right {
                let weight = (right - bin) as f32 / span;
                sum += weight * power_spectrum[bin];
            }
        }
        *energy = sum;
    }

    Status::Success
}

/// Fast integer base-2 logarithm approximation using leading zeros.
///
/// Input is a positive Q15 number (`(0, 32767]`).
/// Returns `log2(x)` scaled to Q8.7 format.
#[inline]
pub fn fast_log2_q15(x: q15) -> Q8F7 {
    if x <= q15::ZERO {
        return Q8F7::MIN;
    }
    let lz = (x.to_bits() as u16).leading_zeros() as i32;
    // Integer part of log2 is 15 - lz
    let int_part = 14 - lz;
    // Fractional part via linear interpolation of remainder bits
    let shifted = (x.to_bits() as i32) << lz;
    let frac = (shifted & 0x7FFF) >> 8; // top 7 bits of fraction
    let log_val = (int_part << 7) + frac;
    Q8F7::from_bits((log_val - (15 << 7)).clamp(i16::MIN as i32, i16::MAX as i32) as i16)
}

/// Simple Voice Activity Detector (VAD) in pure Q15 integer arithmetic.
///
/// Combines Short-Time Energy (STE) and Zero-Crossing Rate (ZCR) thresholds to classify frames
/// as speech/activity vs background noise.
#[derive(Debug, Clone, Copy)]
pub struct VadDetectorQ15 {
    energy_threshold: i32,
    zcr_threshold: u16,
}

impl VadDetectorQ15 {
    /// Create a new VAD detector with energy and zero-crossing rate thresholds.
    pub const fn new(energy_threshold: i32, zcr_threshold: u16) -> Self {
        Self {
            energy_threshold,
            zcr_threshold,
        }
    }

    /// Classify frame as active (`true`) or silence/noise (`false`).
    pub fn is_active(&self, frame: &[q15]) -> bool {
        if frame.is_empty() {
            return false;
        }

        let mut energy_acc: i64 = 0;
        let mut zcr_count: u16 = 0;

        for i in 0..frame.len() {
            let sample = frame[i].to_bits() as i64;
            energy_acc += (sample * sample) >> 15;

            if i > 0 {
                let prev = frame[i - 1];
                let cur = frame[i];
                if (prev >= 0 && cur < 0) || (prev < 0 && cur >= 0) {
                    zcr_count += 1;
                }
            }
        }

        let avg_energy = (energy_acc / frame.len() as i64) as i32;
        avg_energy >= self.energy_threshold && zcr_count >= self.zcr_threshold
    }
}
