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
