//! Audio analysis: the Goertzel single-frequency detector, peak/RMS envelope followers,
//! a Mel filterbank, and MFCC feature extraction. These are DSP front-ends (e.g. for a
//! keyword-spotting pipeline); classifiers and neural nets live in `embedded-nn`.

#[allow(unused_imports)]
use crate::filter_design::single_pole_decay_from_time_constant;
#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::transform::cfft_f32;
use crate::types::Status;

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
