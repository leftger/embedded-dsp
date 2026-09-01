//! Multi-rate digital signal processing routines: Cascaded Integrator-Comb (CIC) decimation/interpolation and linear fractional resampling.

/// Cascaded Integrator-Comb (CIC) Decimator for downsampling signals in integer arithmetic.
pub struct CicDecimator<const STAGES: usize> {
    r: usize, // Decimation factor
    integrator_state: [i32; STAGES],
    comb_state: [i32; STAGES],
    sample_counter: usize,
}

impl<const STAGES: usize> CicDecimator<STAGES> {
    /// Initialise a new CIC decimator with decimation factor `r`.
    pub fn new(r: usize) -> Self {
        Self {
            r,
            integrator_state: [0; STAGES],
            comb_state: [0; STAGES],
            sample_counter: 0,
        }
    }

    /// Theoretical maximum DC gain: `R^STAGES`.
    pub fn gain(&self) -> u64 {
        let mut g: u64 = 1;
        for _ in 0..STAGES {
            g = g.saturating_mul(self.r as u64);
        }
        g
    }

    /// Number of bits of bit-growth: `ceil(log2(R^STAGES))`.
    pub fn gain_bits(&self) -> u32 {
        let g = self.gain();
        if g <= 1 {
            0
        } else {
            64 - (g - 1).leading_zeros()
        }
    }

    /// Process an input sample. Returns `Some(decimated_sample)` every `R` samples.
    pub fn process_sample(&mut self, input: i32) -> Option<i32> {
        // Integrator stages running at high sample rate
        let mut val = input;
        for i in 0..STAGES {
            self.integrator_state[i] = self.integrator_state[i].wrapping_add(val);
            val = self.integrator_state[i];
        }

        self.sample_counter += 1;
        if self.sample_counter >= self.r {
            self.sample_counter = 0;

            // Comb stages running at low sample rate
            for i in 0..STAGES {
                let diff = val.wrapping_sub(self.comb_state[i]);
                self.comb_state[i] = val;
                val = diff;
            }
            Some(val)
        } else {
            None
        }
    }

    /// Process an input sample and normalize output by bit-growth right-shift to prevent overflow.
    pub fn process_sample_scaled(&mut self, input: i32) -> Option<i32> {
        self.process_sample(input).map(|out| {
            let shift = self.gain_bits();
            if shift > 0 {
                out >> shift
            } else {
                out
            }
        })
    }
}

/// Cascaded Integrator-Comb (CIC) Interpolator for upsampling signals in integer arithmetic.
pub struct CicInterpolator<const STAGES: usize> {
    r: usize, // Interpolation factor
    comb_state: [i32; STAGES],
    integrator_state: [i32; STAGES],
}

impl<const STAGES: usize> CicInterpolator<STAGES> {
    /// Initialise a new CIC interpolator with interpolation factor `r`.
    pub fn new(r: usize) -> Self {
        Self {
            r,
            comb_state: [0; STAGES],
            integrator_state: [0; STAGES],
        }
    }

    /// Theoretical maximum DC gain: `R^(STAGES - 1)`.
    pub fn gain(&self) -> u64 {
        if STAGES <= 1 {
            return 1;
        }
        let mut g: u64 = 1;
        for _ in 0..(STAGES - 1) {
            g = g.saturating_mul(self.r as u64);
        }
        g
    }

    /// Number of bits of bit-growth: `ceil(log2(gain))`.
    pub fn gain_bits(&self) -> u32 {
        let g = self.gain();
        if g <= 1 {
            0
        } else {
            64 - (g - 1).leading_zeros()
        }
    }

    /// Process a single input sample and populate `out_buf` with `R` interpolated output samples.
    pub fn process_sample(&mut self, input: i32, out_buf: &mut [i32]) {
        assert!(
            out_buf.len() >= self.r,
            "out_buf must hold at least R samples"
        );

        // Comb stages at low rate
        let mut val = input;
        for i in 0..STAGES {
            let diff = val.wrapping_sub(self.comb_state[i]);
            self.comb_state[i] = val;
            val = diff;
        }

        // Zero stuffing and integrator stages at high rate
        for step in 0..self.r {
            let in_step = if step == 0 { val } else { 0 };
            let mut stage_val = in_step;

            for i in 0..STAGES {
                self.integrator_state[i] = self.integrator_state[i].wrapping_add(stage_val);
                stage_val = self.integrator_state[i];
            }

            out_buf[step] = stage_val;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Polyphase & Linear Resampling in Q15
// ─────────────────────────────────────────────────────────────────────────────

use crate::types::q15;

/// Polyphase FIR decimation by integer factor `M`.
///
/// Filters and downsamples `src` by factor `M` (`decimation_factor`).
/// `coeffs` is the prototype FIR filter kernel (length typically multiple of `M`).
/// Returns the number of output samples written to `dst`.
pub fn polyphase_decimate_q15(
    src: &[q15],
    coeffs: &[q15],
    decimation_factor: usize,
    dst: &mut [q15],
) -> usize {
    if decimation_factor == 0 || coeffs.is_empty() || src.is_empty() {
        return 0;
    }
    let num_taps = coeffs.len();
    let out_len = dst.len().min(if src.len() >= num_taps { (src.len() - num_taps) / decimation_factor + 1 } else { 0 });

    for i in 0..out_len {
        let src_offset = i * decimation_factor;
        let mut acc: i64 = 0;
        for k in 0..num_taps {
            acc += (src[src_offset + k].to_bits() as i64 * coeffs[k].to_bits() as i64) >> 15;
        }
        dst[i] = q15::from_bits(acc.clamp(i16::MIN as i64, i16::MAX as i64) as i16);
    }

    out_len
}

/// Polyphase FIR interpolation by integer factor `L`.
///
/// Upsamples `src` by factor `L` (`interpolation_factor`) using polyphase decomposition.
/// `coeffs` length must be a multiple of `L`.
/// Returns the number of output samples written to `dst`.
pub fn polyphase_interpolate_q15(
    src: &[q15],
    coeffs: &[q15],
    interpolation_factor: usize,
    dst: &mut [q15],
) -> usize {
    let l = interpolation_factor;
    if l == 0 || coeffs.is_empty() || src.is_empty() || coeffs.len() % l != 0 {
        return 0;
    }
    let taps_per_phase = coeffs.len() / l;
    let max_in_samples = if src.len() >= taps_per_phase { src.len() - taps_per_phase + 1 } else { 0 };
    let out_len = dst.len().min(max_in_samples * l);

    for in_idx in 0..max_in_samples {
        for phase in 0..l {
            let out_idx = in_idx * l + phase;
            if out_idx >= dst.len() {
                break;
            }
            let mut acc: i64 = 0;
            for k in 0..taps_per_phase {
                let coeff = coeffs[k * l + phase].to_bits() as i64;
                let sample = src[in_idx + k].to_bits() as i64;
                acc += (sample * coeff) >> 15;
            }
            dst[out_idx] =
                q15::from_bits((acc * l as i64).clamp(i16::MIN as i64, i16::MAX as i64) as i16);
        }
    }

    out_len
}

/// Linear fractional resampler in Q15.
/// `ratio_q16` is `(src_sample_rate / dst_sample_rate)` in Q16.16 format.
pub fn resample_linear_q15(src: &[q15], dst: &mut [q15], ratio_q16: i32) {
    if src.is_empty() || dst.is_empty() || ratio_q16 <= 0 {
        return;
    }

    let mut phase_acc: i64 = 0;
    for i in 0..dst.len() {
        let idx0 = (phase_acc >> 16) as usize;
        let frac = (phase_acc & 0xFFFF) as i32; // [0, 65535]

        if idx0 >= src.len() {
            dst[i] = src[src.len() - 1];
        } else {
            let s0 = src[idx0].to_bits() as i32;
            let s1 = if idx0 + 1 < src.len() {
                src[idx0 + 1].to_bits() as i32
            } else {
                s0
            };
            let diff = s1 - s0;
            let interp = s0 + ((diff * frac) >> 16);
            dst[i] = q15::from_bits(interp.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
        }

        phase_acc += ratio_q16 as i64;
    }
}

/// Linear fractional resampler.
/// Resamples `src` into `dst` according to `ratio` (`src_sample_rate / dst_sample_rate`).
pub fn resample_linear_f32(src: &[f32], dst: &mut [f32], ratio: f32) {
    if src.is_empty() || dst.is_empty() || ratio <= 0.0 {
        return;
    }

    for i in 0..dst.len() {
        let src_idx_float = i as f32 * ratio;
        let idx0 = src_idx_float as usize;
        let idx1 = (idx0 + 1).min(src.len() - 1);

        if idx0 >= src.len() {
            dst[i] = src[src.len() - 1];
            continue;
        }

        let frac = src_idx_float - idx0 as f32;
        dst[i] = src[idx0] * (1.0 - frac) + src[idx1] * frac;
    }
}

#[cfg(feature = "transform")]
use crate::transform::cfft_f32;
#[cfg(feature = "transform")]
use crate::types::Status;

/// Spectral (Sinc) 2:1 Interpolator using frequency-domain zero-padding via FFT/IFFT.
///
/// `src` length must be a power of 2 (e.g. 16, 32, 64, 128, 256).
/// `dst` must have length at least `2 * src.len()`.
///
/// Requires the `transform` feature (enabled by `full`).
#[cfg(feature = "transform")]
pub fn spectral_interpolate_2x_f32(src: &[f32], dst: &mut [f32]) -> Status {
    let n = src.len();
    if n < 4 || (n & (n - 1)) != 0 {
        return Status::ArgumentError;
    }
    if dst.len() < 2 * n {
        return Status::LengthError;
    }
    if 4 * n > 1024 {
        return Status::LengthError; // Max scratch size limit (256-pt input -> 512-pt complex)
    }

    let mut c_buf = [0.0f32; 1024];

    // Copy src into complex array (size 2 * 2n)
    for i in 0..n {
        c_buf[2 * i] = src[i];
        c_buf[2 * i + 1] = 0.0;
    }

    // FFT of size n
    cfft_f32(&mut c_buf[..2 * n], n, 0, 1);

    // Half Nyquist component
    let nyq_re = 0.5 * c_buf[n];
    let nyq_im = 0.5 * c_buf[n + 1];
    c_buf[n] = nyq_re;
    c_buf[n + 1] = nyq_im;

    // Shift negative frequencies to upper half and zero middle
    let mut expanded = [0.0f32; 1024];
    // Copy 0..=N/2
    for i in 0..=(n / 2) {
        expanded[2 * i] = c_buf[2 * i];
        expanded[2 * i + 1] = c_buf[2 * i + 1];
    }
    // Nyquist conjugate mirror at 3N/2
    expanded[2 * (3 * n / 2)] = nyq_re;
    expanded[2 * (3 * n / 2) + 1] = nyq_im;

    // Negative frequencies
    for i in (n / 2 + 1)..n {
        expanded[2 * (i + n)] = c_buf[2 * i];
        expanded[2 * (i + n) + 1] = c_buf[2 * i + 1];
    }

    // IFFT of size 2n
    cfft_f32(&mut expanded[..4 * n], 2 * n, 1, 1);

    // Copy back scaled real part (factor of 2)
    for i in 0..(2 * n) {
        dst[i] = 2.0 * expanded[2 * i];
    }

    Status::Success
}
