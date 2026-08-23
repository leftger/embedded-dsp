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

use crate::transform::cfft_f32;
use crate::types::Status;

/// Spectral (Sinc) 2:1 Interpolator using frequency-domain zero-padding via FFT/IFFT.
///
/// `src` length must be a power of 2 (e.g. 16, 32, 64, 128, 256).
/// `dst` must have length at least `2 * src.len()`.
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
