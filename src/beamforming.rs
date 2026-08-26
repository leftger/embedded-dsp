//! Acoustic Array Processing, Delay-and-Sum Beamforming, and GCC-PHAT TDoA Direction-of-Arrival Estimation.
//!
//! Designed for multi-microphone arrays, sonar arrays, and acoustic anomaly triangulation on embedded hardware.

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::transform::cfft_f32;
use crate::types::Status;

/// Delay-and-Sum Beamformer for multi-channel microphone/sensor arrays.
///
/// Implements fractional sample delay interpolation via linear delay lines and weighted spatial summing.
#[derive(Debug, Clone)]
pub struct DelayAndSumBeamformer<const MICS: usize, const MAX_DELAY: usize> {
    delays_samples: [f32; MICS],
    weights: [f32; MICS],
    delay_lines: [[f32; MAX_DELAY]; MICS],
    write_ptrs: [usize; MICS],
}

impl<const MICS: usize, const MAX_DELAY: usize> DelayAndSumBeamformer<MICS, MAX_DELAY> {
    /// Creates a new Delay-and-Sum Beamformer with uniform weights ($1 / M$).
    pub fn new() -> Self {
        let uniform_w = 1.0 / MICS as f32;
        Self {
            delays_samples: [0.0; MICS],
            weights: [uniform_w; MICS],
            delay_lines: [[0.0; MAX_DELAY]; MICS],
            write_ptrs: [0; MICS],
        }
    }

    /// Sets the fractional delay (in samples) for each microphone channel.
    pub fn set_delays(&mut self, delays: &[f32; MICS]) {
        for (i, &d) in delays.iter().enumerate() {
            self.delays_samples[i] = d.clamp(0.0, (MAX_DELAY - 2) as f32);
        }
    }

    /// Sets the spatial weighting / apodization factors for each channel.
    pub fn set_weights(&mut self, weights: &[f32; MICS]) {
        self.weights.copy_from_slice(weights);
    }

    /// Processes a single multi-channel sample vector and returns the steered beamformed output.
    pub fn process_sample(&mut self, mic_inputs: &[f32; MICS]) -> f32 {
        let mut output = 0.0f32;

        for m in 0..MICS {
            // Write input sample into circular delay line
            let w_ptr = self.write_ptrs[m];
            self.delay_lines[m][w_ptr] = mic_inputs[m];
            self.write_ptrs[m] = (w_ptr + 1) % MAX_DELAY;

            // Compute fractional read pointer
            let delay = self.delays_samples[m];
            let int_delay = delay as usize;
            let frac_delay = delay - int_delay as f32;

            // Two-point linear interpolation
            let idx0 = (w_ptr + MAX_DELAY - int_delay) % MAX_DELAY;
            let idx1 = (idx0 + MAX_DELAY - 1) % MAX_DELAY;

            let s0 = self.delay_lines[m][idx0];
            let s1 = self.delay_lines[m][idx1];
            let delayed_sample = s0 + frac_delay * (s1 - s0);

            output += delayed_sample * self.weights[m];
        }

        output
    }

    /// Resets all internal delay lines.
    pub fn reset(&mut self) {
        for m in 0..MICS {
            self.delay_lines[m].fill(0.0);
            self.write_ptrs[m] = 0;
        }
    }
}

impl<const MICS: usize, const MAX_DELAY: usize> Default for DelayAndSumBeamformer<MICS, MAX_DELAY> {
    fn default() -> Self {
        Self::new()
    }
}

/// Generalized Cross-Correlation with Phase Transform (GCC-PHAT) for Time Difference of Arrival (TDoA).
///
/// Computes the normalized cross-correlation:
/// $$R_{\text{PHAT}}(f) = \frac{X_1(f) X_2^*(f)}{|X_1(f) X_2^*(f)|}$$
/// and finds the time lag $\tau \in [-\text{max\_delay}, \text{max\_delay}]$ that maximizes $r_{\text{PHAT}}(\tau)$.
///
/// `sig_a` and `sig_b` must have equal length $N$ (power of two $\le 512$).
/// Returns the estimated fractional delay (in samples) between channel A and channel B.
pub fn gcc_phat_tdoa_f32(sig_a: &[f32], sig_b: &[f32], max_delay: usize) -> Result<f32, Status> {
    let n = sig_a.len();
    if n != sig_b.len() || n < 4 || (n & (n - 1)) != 0 || n > 512 {
        return Err(Status::ArgumentError);
    }
    if max_delay >= n / 2 {
        return Err(Status::ArgumentError);
    }

    let mut buf_a = [0.0f32; 1024];
    let mut buf_b = [0.0f32; 1024];

    for i in 0..n {
        buf_a[2 * i] = sig_a[i];
        buf_a[2 * i + 1] = 0.0;
        buf_b[2 * i] = sig_b[i];
        buf_b[2 * i + 1] = 0.0;
    }

    // Forward FFTs
    cfft_f32(&mut buf_a[..2 * n], n, 0, 1);
    cfft_f32(&mut buf_b[..2 * n], n, 0, 1);

    // Cross-spectrum with Phase Transform normalization: X_a * conj(X_b) / |X_a * conj(X_b)|
    let mut xcorr_spec = [0.0f32; 1024];
    for k in 0..n {
        let a_re = buf_a[2 * k];
        let a_im = buf_a[2 * k + 1];
        let b_re = buf_b[2 * k];
        let b_im = -buf_b[2 * k + 1]; // Conjugate

        let c_re = a_re * b_re - a_im * b_im;
        let c_im = a_re * b_im + a_im * b_re;

        let mag = (c_re * c_re + c_im * c_im).sqrt().max(1e-12);
        xcorr_spec[2 * k] = c_re / mag;
        xcorr_spec[2 * k + 1] = c_im / mag;
    }

    // Inverse FFT to get time-domain cross-correlation
    cfft_f32(&mut xcorr_spec[..2 * n], n, 1, 1);

    // Circular shift to center lag 0 at n/2
    let mut gcc = [0.0f32; 512];
    for i in 0..n {
        let target_idx = (i + n / 2) % n;
        gcc[target_idx] = xcorr_spec[2 * i];
    }

    // Search for maximum peak in [-max_delay, +max_delay] centered around n/2
    let center = n / 2;
    let start_idx = center - max_delay;
    let end_idx = center + max_delay;

    let mut peak_val = f32::MIN;
    let mut peak_idx = center;

    for i in start_idx..=end_idx {
        if gcc[i] > peak_val {
            peak_val = gcc[i];
            peak_idx = i;
        }
    }

    // Parabolic sub-sample interpolation
    let mut frac_offset = 0.0f32;
    if peak_idx > start_idx && peak_idx < end_idx {
        let alpha = gcc[peak_idx - 1];
        let beta = gcc[peak_idx];
        let gamma = gcc[peak_idx + 1];
        let denom = 2.0 * (2.0 * beta - alpha - gamma);
        if denom.abs() > 1e-12 {
            frac_offset = (alpha - gamma) / denom;
        }
    }

    let delay_samples = (peak_idx as f32 + frac_offset) - center as f32;
    Ok(delay_samples)
}
