//! Digital filtering functions (FIR, Biquad IIR Direct Form I & II, LMS Adaptive Filter, Convolution, Correlation).

use crate::types::*;

// --- FIR Filter ---

/// Instance structure for the floating-point FIR filter.
pub struct FirInstanceF32<'a> {
    pub num_taps: u16,
    pub coeffs: &'a [f32],
    pub state: &'a mut [f32],
}

impl<'a> FirInstanceF32<'a> {
    pub fn init(num_taps: u16, coeffs: &'a [f32], state: &'a mut [f32]) -> Self {
        state.fill(0.0);
        Self {
            num_taps,
            coeffs,
            state,
        }
    }
}

pub fn fir_f32(instance: &mut FirInstanceF32, src: &[f32], dst: &mut [f32]) {
    let num_taps = instance.num_taps as usize;
    let block_size = src.len().min(dst.len());

    for i in 0..block_size {
        // Shift state
        for k in (1..num_taps).rev() {
            instance.state[k] = instance.state[k - 1];
        }
        instance.state[0] = src[i];

        // Compute dot product with coefficients
        let mut acc = 0.0f32;
        for k in 0..num_taps {
            acc += instance.state[k] * instance.coeffs[k];
        }
        dst[i] = acc;
    }
}

/// Instance structure for the Q31 FIR filter.
pub struct FirInstanceQ31<'a> {
    pub num_taps: u16,
    pub coeffs: &'a [q31],
    pub state: &'a mut [q31],
}

impl<'a> FirInstanceQ31<'a> {
    pub fn init(num_taps: u16, coeffs: &'a [q31], state: &'a mut [q31]) -> Self {
        state.fill(0);
        Self {
            num_taps,
            coeffs,
            state,
        }
    }
}

pub fn fir_q31(instance: &mut FirInstanceQ31, src: &[q31], dst: &mut [q31]) {
    let num_taps = instance.num_taps as usize;
    let block_size = src.len().min(dst.len());

    for i in 0..block_size {
        for k in (1..num_taps).rev() {
            instance.state[k] = instance.state[k - 1];
        }
        instance.state[0] = src[i];

        let mut acc: i64 = 0;
        for k in 0..num_taps {
            acc += (instance.state[k] as i64 * instance.coeffs[k] as i64) >> 31;
        }
        dst[i] = acc.clamp(i32::MIN as i64, i32::MAX as i64) as q31;
    }
}

/// Instance structure for the Q15 FIR filter.
pub struct FirInstanceQ15<'a> {
    pub num_taps: u16,
    pub coeffs: &'a [q15],
    pub state: &'a mut [q15],
}

impl<'a> FirInstanceQ15<'a> {
    pub fn init(num_taps: u16, coeffs: &'a [q15], state: &'a mut [q15]) -> Self {
        state.fill(0);
        Self {
            num_taps,
            coeffs,
            state,
        }
    }
}

pub fn fir_q15(instance: &mut FirInstanceQ15, src: &[q15], dst: &mut [q15]) {
    let num_taps = instance.num_taps as usize;
    let block_size = src.len().min(dst.len());

    for i in 0..block_size {
        for k in (1..num_taps).rev() {
            instance.state[k] = instance.state[k - 1];
        }
        instance.state[0] = src[i];

        let mut acc: i32 = 0;
        for k in 0..num_taps {
            acc += (instance.state[k] as i32 * instance.coeffs[k] as i32) >> 15;
        }
        dst[i] = acc.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
    }
}

// --- Biquad Cascade Direct Form I Filter ---

/// Instance structure for the floating-point Biquad Cascade Direct Form I filter.
pub struct BiquadCascadeInstanceF32<'a> {
    pub num_stages: u8,
    pub coeffs: &'a [f32],    // 5 * num_stages: [b0, b1, b2, a1, a2]
    pub state: &'a mut [f32], // 4 * num_stages: [x[n-1], x[n-2], y[n-1], y[n-2]]
}

impl<'a> BiquadCascadeInstanceF32<'a> {
    pub fn init(num_stages: u8, coeffs: &'a [f32], state: &'a mut [f32]) -> Self {
        state.fill(0.0);
        Self {
            num_stages,
            coeffs,
            state,
        }
    }
}

pub fn biquad_cascade_df1_f32(
    instance: &mut BiquadCascadeInstanceF32,
    src: &[f32],
    dst: &mut [f32],
) {
    let num_stages = instance.num_stages as usize;
    let block_size = src.len().min(dst.len());

    let mut in_val;
    let mut out_val;

    for i in 0..block_size {
        in_val = src[i];
        for stage in 0..num_stages {
            let b0 = instance.coeffs[stage * 5];
            let b1 = instance.coeffs[stage * 5 + 1];
            let b2 = instance.coeffs[stage * 5 + 2];
            let a1 = instance.coeffs[stage * 5 + 3];
            let a2 = instance.coeffs[stage * 5 + 4];

            let x1 = instance.state[stage * 4];
            let x2 = instance.state[stage * 4 + 1];
            let y1 = instance.state[stage * 4 + 2];
            let y2 = instance.state[stage * 4 + 3];

            out_val = b0 * in_val + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;

            instance.state[stage * 4 + 1] = x1;
            instance.state[stage * 4] = in_val;
            instance.state[stage * 4 + 3] = y1;
            instance.state[stage * 4 + 2] = out_val;

            in_val = out_val;
        }
        dst[i] = in_val;
    }
}

// --- LMS Adaptive Filter ---

/// Instance structure for the floating-point LMS adaptive filter.
pub struct LmsInstanceF32<'a> {
    pub num_taps: u16,
    pub coeffs: &'a mut [f32],
    pub state: &'a mut [f32],
    pub mu: f32,
}

impl<'a> LmsInstanceF32<'a> {
    pub fn init(num_taps: u16, coeffs: &'a mut [f32], state: &'a mut [f32], mu: f32) -> Self {
        state.fill(0.0);
        coeffs.fill(0.0);
        Self {
            num_taps,
            coeffs,
            state,
            mu,
        }
    }
}

pub fn lms_f32(
    instance: &mut LmsInstanceF32,
    src: &[f32],
    ref_signal: &[f32],
    out: &mut [f32],
    err: &mut [f32],
) {
    let num_taps = instance.num_taps as usize;
    let block_size = src
        .len()
        .min(ref_signal.len())
        .min(out.len())
        .min(err.len());

    for i in 0..block_size {
        for k in (1..num_taps).rev() {
            instance.state[k] = instance.state[k - 1];
        }
        instance.state[0] = src[i];

        let mut acc = 0.0f32;
        for k in 0..num_taps {
            acc += instance.state[k] * instance.coeffs[k];
        }
        out[i] = acc;
        let e = ref_signal[i] - acc;
        err[i] = e;

        // Update coefficients: w[n+1] = w[n] + 2 * mu * e[n] * x[n]
        let alpha = 2.0 * instance.mu * e;
        for k in 0..num_taps {
            instance.coeffs[k] += alpha * instance.state[k];
        }
    }
}

// --- Convolution ---

pub fn conv_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    dst[..out_len].fill(0.0);
    for i in 0..len_a {
        for j in 0..len_b {
            if i + j < out_len {
                dst[i + j] += src_a[i] * src_b[j];
            }
        }
    }
}

pub fn conv_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i64 = 0;
        let k_min = if n >= len_b - 1 { n - (len_b - 1) } else { 0 };
        let k_max = n.min(len_a - 1);
        for k in k_min..=k_max {
            acc += (src_a[k] as i64 * src_b[n - k] as i64) >> 31;
        }
        dst[n] = acc.clamp(i32::MIN as i64, i32::MAX as i64) as q31;
    }
}

pub fn conv_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i32 = 0;
        let k_min = if n >= len_b - 1 { n - (len_b - 1) } else { 0 };
        let k_max = n.min(len_a - 1);
        for k in k_min..=k_max {
            acc += (src_a[k] as i32 * src_b[n - k] as i32) >> 15;
        }
        dst[n] = acc.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
    }
}

pub fn conv_q7(src_a: &[q7], src_b: &[q7], dst: &mut [q7]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i32 = 0;
        let k_min = if n >= len_b - 1 { n - (len_b - 1) } else { 0 };
        let k_max = n.min(len_a - 1);
        for k in k_min..=k_max {
            acc += (src_a[k] as i32 * src_b[n - k] as i32) >> 7;
        }
        dst[n] = acc.clamp(i8::MIN as i32, i8::MAX as i32) as q7;
    }
}

// --- Correlation ---

pub fn correlate_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    dst[..out_len].fill(0.0);
    for n in 0..out_len {
        let mut acc = 0.0f32;
        for k in 0..len_a {
            let idx_b = (k as isize) + (len_b as isize - 1) - (n as isize);
            if idx_b >= 0 && (idx_b as usize) < len_b {
                acc += src_a[k] * src_b[idx_b as usize];
            }
        }
        dst[n] = acc;
    }
}

pub fn correlate_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i64 = 0;
        for k in 0..len_a {
            let idx_b = (k as isize) + (len_b as isize - 1) - (n as isize);
            if idx_b >= 0 && (idx_b as usize) < len_b {
                acc += (src_a[k] as i64 * src_b[idx_b as usize] as i64) >> 31;
            }
        }
        dst[n] = acc.clamp(i32::MIN as i64, i32::MAX as i64) as q31;
    }
}

pub fn correlate_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i32 = 0;
        for k in 0..len_a {
            let idx_b = (k as isize) + (len_b as isize - 1) - (n as isize);
            if idx_b >= 0 && (idx_b as usize) < len_b {
                acc += (src_a[k] as i32 * src_b[idx_b as usize] as i32) >> 15;
            }
        }
        dst[n] = acc.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
    }
}
