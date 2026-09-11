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
        state.fill(q31::ZERO);
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
            acc += (instance.state[k].to_bits() as i64 * instance.coeffs[k].to_bits() as i64) >> 31;
        }
        dst[i] = q31::from_bits(acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
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
        state.fill(q15::ZERO);
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
            acc += (instance.state[k].to_bits() as i32 * instance.coeffs[k].to_bits() as i32) >> 15;
        }
        dst[i] = q15::from_bits(acc.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
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

/// Instance structure for the floating-point Biquad Cascade Transposed Direct Form II filter.
///
/// Same SOS layout `[b0, b1, b2, a1, a2]` as [`BiquadCascadeInstanceF32`].
/// State is two delays per stage (`[s1, s2, ...]`).
pub struct BiquadCascadeDf2tInstanceF32<'a> {
    pub num_stages: u8,
    pub coeffs: &'a [f32],
    pub state: &'a mut [f32],
}

impl<'a> BiquadCascadeDf2tInstanceF32<'a> {
    pub fn init(num_stages: u8, coeffs: &'a [f32], state: &'a mut [f32]) -> Self {
        state.fill(0.0);
        Self {
            num_stages,
            coeffs,
            state,
        }
    }
}

pub fn biquad_cascade_df2t_f32(
    instance: &mut BiquadCascadeDf2tInstanceF32,
    src: &[f32],
    dst: &mut [f32],
) {
    let num_stages = instance.num_stages as usize;
    let block_size = src.len().min(dst.len());

    for i in 0..block_size {
        let mut in_val = src[i];
        for stage in 0..num_stages {
            let b0 = instance.coeffs[stage * 5];
            let b1 = instance.coeffs[stage * 5 + 1];
            let b2 = instance.coeffs[stage * 5 + 2];
            let a1 = instance.coeffs[stage * 5 + 3];
            let a2 = instance.coeffs[stage * 5 + 4];

            let s1 = instance.state[stage * 2];
            let s2 = instance.state[stage * 2 + 1];

            let y = b0 * in_val + s1;
            instance.state[stage * 2] = b1 * in_val + a1 * y + s2;
            instance.state[stage * 2 + 1] = b2 * in_val + a2 * y;
            in_val = y;
        }
        dst[i] = in_val;
    }
}

/// Instance structure for the Q15 Biquad Cascade Direct Form I filter.
///
/// Coeffs are Q1.15 `[b0, b1, b2, a1, a2]` per stage (same layout as the f32 cascade).
/// `post_shift` extra headroom in stored coeffs (`coeff_f32 / 2^{post_shift}` in Q15);
/// the MAC is shifted `15 - post_shift` (CMSIS-style).
pub struct BiquadCascadeInstanceQ15<'a> {
    pub num_stages: u8,
    pub post_shift: u8,
    pub coeffs: &'a [q15],
    pub state: &'a mut [q15],
}

impl<'a> BiquadCascadeInstanceQ15<'a> {
    pub fn init(num_stages: u8, coeffs: &'a [q15], state: &'a mut [q15], post_shift: u8) -> Self {
        state.fill(q15::ZERO);
        Self {
            num_stages,
            post_shift,
            coeffs,
            state,
        }
    }
}

pub fn biquad_cascade_df1_q15(
    instance: &mut BiquadCascadeInstanceQ15,
    src: &[q15],
    dst: &mut [q15],
) {
    let num_stages = instance.num_stages as usize;
    let block_size = src.len().min(dst.len());
    let shift = 15u32.saturating_sub(instance.post_shift as u32).min(31);

    for i in 0..block_size {
        let mut in_val = src[i].to_bits() as i64;
        for stage in 0..num_stages {
            let b0 = instance.coeffs[stage * 5].to_bits() as i64;
            let b1 = instance.coeffs[stage * 5 + 1].to_bits() as i64;
            let b2 = instance.coeffs[stage * 5 + 2].to_bits() as i64;
            let a1 = instance.coeffs[stage * 5 + 3].to_bits() as i64;
            let a2 = instance.coeffs[stage * 5 + 4].to_bits() as i64;

            let x1 = instance.state[stage * 4].to_bits() as i64;
            let x2 = instance.state[stage * 4 + 1].to_bits() as i64;
            let y1 = instance.state[stage * 4 + 2].to_bits() as i64;
            let y2 = instance.state[stage * 4 + 3].to_bits() as i64;

            let acc = b0 * in_val + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
            let out_val = (acc >> shift).clamp(i16::MIN as i64, i16::MAX as i64);

            instance.state[stage * 4 + 1] = q15::from_bits(x1 as i16);
            instance.state[stage * 4] = q15::from_bits(in_val as i16);
            instance.state[stage * 4 + 3] = q15::from_bits(y1 as i16);
            instance.state[stage * 4 + 2] = q15::from_bits(out_val as i16);

            in_val = out_val;
        }
        dst[i] = q15::from_bits(in_val as i16);
    }
}

/// Instance structure for the Q31 Biquad Cascade Direct Form I filter.
pub struct BiquadCascadeInstanceQ31<'a> {
    pub num_stages: u8,
    pub post_shift: u8,
    pub coeffs: &'a [q31],
    pub state: &'a mut [q31],
}

impl<'a> BiquadCascadeInstanceQ31<'a> {
    pub fn init(num_stages: u8, coeffs: &'a [q31], state: &'a mut [q31], post_shift: u8) -> Self {
        state.fill(q31::ZERO);
        Self {
            num_stages,
            post_shift,
            coeffs,
            state,
        }
    }
}

pub fn biquad_cascade_df1_q31(
    instance: &mut BiquadCascadeInstanceQ31,
    src: &[q31],
    dst: &mut [q31],
) {
    let num_stages = instance.num_stages as usize;
    let block_size = src.len().min(dst.len());
    let shift = 31u32.saturating_sub(instance.post_shift as u32).min(63);

    for i in 0..block_size {
        let mut in_val = src[i].to_bits() as i64;
        for stage in 0..num_stages {
            let b0 = instance.coeffs[stage * 5].to_bits() as i64;
            let b1 = instance.coeffs[stage * 5 + 1].to_bits() as i64;
            let b2 = instance.coeffs[stage * 5 + 2].to_bits() as i64;
            let a1 = instance.coeffs[stage * 5 + 3].to_bits() as i64;
            let a2 = instance.coeffs[stage * 5 + 4].to_bits() as i64;

            let x1 = instance.state[stage * 4].to_bits() as i64;
            let x2 = instance.state[stage * 4 + 1].to_bits() as i64;
            let y1 = instance.state[stage * 4 + 2].to_bits() as i64;
            let y2 = instance.state[stage * 4 + 3].to_bits() as i64;

            let acc = b0 * in_val + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
            let out_val = (acc >> shift).clamp(i32::MIN as i64, i32::MAX as i64);

            instance.state[stage * 4 + 1] = q31::from_bits(x1 as i32);
            instance.state[stage * 4] = q31::from_bits(in_val as i32);
            instance.state[stage * 4 + 3] = q31::from_bits(y1 as i32);
            instance.state[stage * 4 + 2] = q31::from_bits(out_val as i32);

            in_val = out_val;
        }
        dst[i] = q31::from_bits(in_val as i32);
    }
}

/// Instance structure for the Q15 Biquad Cascade Transposed Direct Form II filter.
///
/// Same SOS layout `[b0, b1, b2, a1, a2]` and `post_shift` as
/// [`BiquadCascadeInstanceQ15`]. State is two delays per stage (`[s1, s2, ...]`),
/// which is better-conditioned for high-Q poles than Direct Form I.
pub struct BiquadCascadeDf2tInstanceQ15<'a> {
    pub num_stages: u8,
    pub post_shift: u8,
    pub coeffs: &'a [q15],
    pub state: &'a mut [q15],
}

impl<'a> BiquadCascadeDf2tInstanceQ15<'a> {
    pub fn init(num_stages: u8, coeffs: &'a [q15], state: &'a mut [q15], post_shift: u8) -> Self {
        state.fill(q15::ZERO);
        Self {
            num_stages,
            post_shift,
            coeffs,
            state,
        }
    }
}

pub fn biquad_cascade_df2t_q15(
    instance: &mut BiquadCascadeDf2tInstanceQ15,
    src: &[q15],
    dst: &mut [q15],
) {
    let num_stages = instance.num_stages as usize;
    let block_size = src.len().min(dst.len());
    let shift = 15u32.saturating_sub(instance.post_shift as u32).min(31);

    for i in 0..block_size {
        let mut in_val = src[i].to_bits() as i64;
        for stage in 0..num_stages {
            let b0 = instance.coeffs[stage * 5].to_bits() as i64;
            let b1 = instance.coeffs[stage * 5 + 1].to_bits() as i64;
            let b2 = instance.coeffs[stage * 5 + 2].to_bits() as i64;
            let a1 = instance.coeffs[stage * 5 + 3].to_bits() as i64;
            let a2 = instance.coeffs[stage * 5 + 4].to_bits() as i64;

            let s1 = instance.state[stage * 2].to_bits() as i64;
            let s2 = instance.state[stage * 2 + 1].to_bits() as i64;

            let y = (b0 * in_val + (s1 << shift)).clamp(i64::MIN >> 1, i64::MAX >> 1) >> shift;
            let out_val = y.clamp(i16::MIN as i64, i16::MAX as i64);
            let s1_new = (b1 * in_val + a1 * out_val + (s2 << shift)) >> shift;
            let s2_new = (b2 * in_val + a2 * out_val) >> shift;

            instance.state[stage * 2] =
                q15::from_bits(s1_new.clamp(i16::MIN as i64, i16::MAX as i64) as i16);
            instance.state[stage * 2 + 1] =
                q15::from_bits(s2_new.clamp(i16::MIN as i64, i16::MAX as i64) as i16);
            in_val = out_val;
        }
        dst[i] = q15::from_bits(in_val as i16);
    }
}

/// Instance structure for the Q31 Biquad Cascade Transposed Direct Form II filter.
pub struct BiquadCascadeDf2tInstanceQ31<'a> {
    pub num_stages: u8,
    pub post_shift: u8,
    pub coeffs: &'a [q31],
    pub state: &'a mut [q31],
}

impl<'a> BiquadCascadeDf2tInstanceQ31<'a> {
    pub fn init(num_stages: u8, coeffs: &'a [q31], state: &'a mut [q31], post_shift: u8) -> Self {
        state.fill(q31::ZERO);
        Self {
            num_stages,
            post_shift,
            coeffs,
            state,
        }
    }
}

pub fn biquad_cascade_df2t_q31(
    instance: &mut BiquadCascadeDf2tInstanceQ31,
    src: &[q31],
    dst: &mut [q31],
) {
    let num_stages = instance.num_stages as usize;
    let block_size = src.len().min(dst.len());
    let shift = 31u32.saturating_sub(instance.post_shift as u32).min(63);

    for i in 0..block_size {
        let mut in_val = src[i].to_bits() as i64;
        for stage in 0..num_stages {
            let b0 = instance.coeffs[stage * 5].to_bits() as i64;
            let b1 = instance.coeffs[stage * 5 + 1].to_bits() as i64;
            let b2 = instance.coeffs[stage * 5 + 2].to_bits() as i64;
            let a1 = instance.coeffs[stage * 5 + 3].to_bits() as i64;
            let a2 = instance.coeffs[stage * 5 + 4].to_bits() as i64;

            let s1 = instance.state[stage * 2].to_bits() as i64;
            let s2 = instance.state[stage * 2 + 1].to_bits() as i64;

            let y = (b0 * in_val + (s1 << shift)).clamp(i64::MIN >> 1, i64::MAX >> 1) >> shift;
            let out_val = y.clamp(i32::MIN as i64, i32::MAX as i64);
            let s1_new = (b1 * in_val + a1 * out_val + (s2 << shift)) >> shift;
            let s2_new = (b2 * in_val + a2 * out_val) >> shift;

            instance.state[stage * 2] =
                q31::from_bits(s1_new.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
            instance.state[stage * 2 + 1] =
                q31::from_bits(s2_new.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
            in_val = out_val;
        }
        dst[i] = q31::from_bits(in_val as i32);
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

/// Leaky LMS: `w ← (1 - leak) w + 2 μ e x`. `leak = 0` matches [`lms_f32`].
pub fn lms_leaky_f32(
    instance: &mut LmsInstanceF32,
    src: &[f32],
    ref_signal: &[f32],
    out: &mut [f32],
    err: &mut [f32],
    leak: f32,
) {
    let num_taps = instance.num_taps as usize;
    let block_size = src
        .len()
        .min(ref_signal.len())
        .min(out.len())
        .min(err.len());
    let keep = 1.0 - leak;

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

        let alpha = 2.0 * instance.mu * e;
        for k in 0..num_taps {
            instance.coeffs[k] = keep * instance.coeffs[k] + alpha * instance.state[k];
        }
    }
}

/// Normalized LMS instance (`eps` floors the power denominator).
pub struct NlmsInstanceF32<'a> {
    pub num_taps: u16,
    pub coeffs: &'a mut [f32],
    pub state: &'a mut [f32],
    pub mu: f32,
    pub eps: f32,
}

impl<'a> NlmsInstanceF32<'a> {
    pub fn init(
        num_taps: u16,
        coeffs: &'a mut [f32],
        state: &'a mut [f32],
        mu: f32,
        eps: f32,
    ) -> Self {
        state.fill(0.0);
        coeffs.fill(0.0);
        Self {
            num_taps,
            coeffs,
            state,
            mu,
            eps,
        }
    }
}

/// NLMS: `w ← w + μ e x / (eps + ||x||²)`.
pub fn nlms_f32(
    instance: &mut NlmsInstanceF32,
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
        let mut power = instance.eps;
        for k in 0..num_taps {
            acc += instance.state[k] * instance.coeffs[k];
            power += instance.state[k] * instance.state[k];
        }
        out[i] = acc;
        let e = ref_signal[i] - acc;
        err[i] = e;

        let alpha = instance.mu * e / power;
        for k in 0..num_taps {
            instance.coeffs[k] += alpha * instance.state[k];
        }
    }
}

/// Q15 LMS adaptive filter.
pub struct LmsInstanceQ15<'a> {
    pub num_taps: u16,
    pub coeffs: &'a mut [q15],
    pub state: &'a mut [q15],
    pub mu: q15,
}

impl<'a> LmsInstanceQ15<'a> {
    pub fn init(num_taps: u16, coeffs: &'a mut [q15], state: &'a mut [q15], mu: q15) -> Self {
        state.fill(q15::ZERO);
        coeffs.fill(q15::ZERO);
        Self {
            num_taps,
            coeffs,
            state,
            mu,
        }
    }
}

fn lms_q15_inner(
    instance: &mut LmsInstanceQ15,
    src: &[q15],
    ref_signal: &[q15],
    out: &mut [q15],
    err: &mut [q15],
    leak: q15,
) {
    let num_taps = instance.num_taps as usize;
    let block_size = src
        .len()
        .min(ref_signal.len())
        .min(out.len())
        .min(err.len());
    let keep = 32767i32 - leak.to_bits().max(0) as i32;

    for i in 0..block_size {
        for k in (1..num_taps).rev() {
            instance.state[k] = instance.state[k - 1];
        }
        instance.state[0] = src[i];

        let mut acc: i64 = 0;
        for k in 0..num_taps {
            acc += instance.state[k].to_bits() as i64 * instance.coeffs[k].to_bits() as i64;
        }
        let y = (acc >> 15).clamp(i16::MIN as i64, i16::MAX as i64);
        out[i] = q15::from_bits(y as i16);
        let e = (ref_signal[i].to_bits() as i32 - y as i32).clamp(i16::MIN as i32, i16::MAX as i32);
        err[i] = q15::from_bits(e as i16);

        let alpha = (2i64 * instance.mu.to_bits() as i64 * e as i64) >> 15;
        for k in 0..num_taps {
            let leaked = (keep as i64 * instance.coeffs[k].to_bits() as i64) >> 15;
            let upd = leaked + ((alpha * instance.state[k].to_bits() as i64) >> 15);
            instance.coeffs[k] = q15::from_bits(upd.clamp(i16::MIN as i64, i16::MAX as i64) as i16);
        }
    }
}

pub fn lms_q15(
    instance: &mut LmsInstanceQ15,
    src: &[q15],
    ref_signal: &[q15],
    out: &mut [q15],
    err: &mut [q15],
) {
    lms_q15_inner(instance, src, ref_signal, out, err, q15::ZERO);
}

/// Leaky LMS in Q15. `leak` is Q1.15 (`0` matches [`lms_q15`]).
pub fn lms_leaky_q15(
    instance: &mut LmsInstanceQ15,
    src: &[q15],
    ref_signal: &[q15],
    out: &mut [q15],
    err: &mut [q15],
    leak: q15,
) {
    lms_q15_inner(instance, src, ref_signal, out, err, leak);
}

/// Q15 NLMS instance.
pub struct NlmsInstanceQ15<'a> {
    pub num_taps: u16,
    pub coeffs: &'a mut [q15],
    pub state: &'a mut [q15],
    pub mu: q15,
    pub eps: q15,
}

impl<'a> NlmsInstanceQ15<'a> {
    pub fn init(
        num_taps: u16,
        coeffs: &'a mut [q15],
        state: &'a mut [q15],
        mu: q15,
        eps: q15,
    ) -> Self {
        state.fill(q15::ZERO);
        coeffs.fill(q15::ZERO);
        Self {
            num_taps,
            coeffs,
            state,
            mu,
            eps,
        }
    }
}

pub fn nlms_q15(
    instance: &mut NlmsInstanceQ15,
    src: &[q15],
    ref_signal: &[q15],
    out: &mut [q15],
    err: &mut [q15],
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

        let mut acc: i64 = 0;
        let mut power: i64 = instance.eps.to_bits().max(1) as i64;
        for k in 0..num_taps {
            let x = instance.state[k].to_bits() as i64;
            acc += x * instance.coeffs[k].to_bits() as i64;
            power += (x * x) >> 15;
        }
        let y = (acc >> 15).clamp(i16::MIN as i64, i16::MAX as i64);
        out[i] = q15::from_bits(y as i16);
        let e = (ref_signal[i].to_bits() as i32 - y as i32).clamp(i16::MIN as i32, i16::MAX as i32);
        err[i] = q15::from_bits(e as i16);

        let alpha = (instance.mu.to_bits() as i64 * e as i64) / power;
        for k in 0..num_taps {
            let upd = instance.coeffs[k].to_bits() as i64
                + ((alpha * instance.state[k].to_bits() as i64) >> 15);
            instance.coeffs[k] = q15::from_bits(upd.clamp(i16::MIN as i64, i16::MAX as i64) as i16);
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
            acc += (src_a[k].to_bits() as i64 * src_b[n - k].to_bits() as i64) >> 31;
        }
        dst[n] = q31::from_bits(acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
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
            acc += (src_a[k].to_bits() as i32 * src_b[n - k].to_bits() as i32) >> 15;
        }
        dst[n] = q15::from_bits(acc.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
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
            acc += (src_a[k].to_bits() as i32 * src_b[n - k].to_bits() as i32) >> 7;
        }
        dst[n] = q7::from_bits(acc.clamp(i8::MIN as i32, i8::MAX as i32) as i8);
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
                acc += (src_a[k].to_bits() as i64 * src_b[idx_b as usize].to_bits() as i64) >> 31;
            }
        }
        dst[n] = q31::from_bits(acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
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
                acc += (src_a[k].to_bits() as i32 * src_b[idx_b as usize].to_bits() as i32) >> 15;
            }
        }
        dst[n] = q15::from_bits(acc.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    }
}

// --- Non-linear Filtering (Median & Conditional Median) ---

#[allow(unused_imports)]
use crate::math::FloatMath;
#[cfg(feature = "transform")]
use crate::transform::cfft_f32;

/// 1D Conditional / Thresholded Median Filter for f32.
///
/// Replaces sample `src[i]` with the local median only if `|src[i] - median| > threshold`.
/// When `threshold == 0.0`, performs standard median filtering.
///
/// `window_len` must be odd and $\le 63$.
pub fn median_filter_1d_f32(
    src: &[f32],
    dst: &mut [f32],
    window_len: usize,
    threshold: f32,
) -> Status {
    let n = src.len();
    if n == 0 || dst.len() < n {
        return Status::LengthError;
    }
    if window_len == 0 || window_len % 2 == 0 || window_len > 63 {
        return Status::ArgumentError;
    }

    let half = window_len / 2;
    let mut sort_buf = [0.0f32; 64];

    for i in 0..n {
        // Populate window with boundary clamping
        for j in 0..window_len {
            let idx = (i as isize + j as isize - half as isize).clamp(0, (n - 1) as isize) as usize;
            sort_buf[j] = src[idx];
        }

        // Insertion sort on small stack buffer
        for a in 1..window_len {
            let mut b = a;
            while b > 0 && sort_buf[b - 1] > sort_buf[b] {
                sort_buf.swap(b - 1, b);
                b -= 1;
            }
        }

        let med = sort_buf[half];
        let center = src[i];
        if (center - med).abs() >= threshold {
            dst[i] = med;
        } else {
            dst[i] = center;
        }
    }

    Status::Success
}

/// 1D Conditional Median Filter for Q15.
pub fn median_filter_1d_q15(
    src: &[q15],
    dst: &mut [q15],
    window_len: usize,
    threshold: q15,
) -> Status {
    let n = src.len();
    if n == 0 || dst.len() < n {
        return Status::LengthError;
    }
    if window_len == 0 || window_len % 2 == 0 || window_len > 63 {
        return Status::ArgumentError;
    }

    let half = window_len / 2;
    let mut sort_buf = [q15::ZERO; 64];

    for i in 0..n {
        for j in 0..window_len {
            let idx = (i as isize + j as isize - half as isize).clamp(0, (n - 1) as isize) as usize;
            sort_buf[j] = src[idx];
        }

        for a in 1..window_len {
            let mut b = a;
            while b > 0 && sort_buf[b - 1] > sort_buf[b] {
                sort_buf.swap(b - 1, b);
                b -= 1;
            }
        }

        let med = sort_buf[half];
        let center = src[i];
        let diff = (center.to_bits() as i32 - med.to_bits() as i32).abs();
        if diff >= threshold.to_bits() as i32 {
            dst[i] = med;
        } else {
            dst[i] = center;
        }
    }

    Status::Success
}

/// 1D Conditional Median Filter for Q31.
pub fn median_filter_1d_q31(
    src: &[q31],
    dst: &mut [q31],
    window_len: usize,
    threshold: q31,
) -> Status {
    let n = src.len();
    if n == 0 || dst.len() < n {
        return Status::LengthError;
    }
    if window_len == 0 || window_len % 2 == 0 || window_len > 63 {
        return Status::ArgumentError;
    }

    let half = window_len / 2;
    let mut sort_buf = [q31::ZERO; 64];

    for i in 0..n {
        for j in 0..window_len {
            let idx = (i as isize + j as isize - half as isize).clamp(0, (n - 1) as isize) as usize;
            sort_buf[j] = src[idx];
        }

        for a in 1..window_len {
            let mut b = a;
            while b > 0 && sort_buf[b - 1] > sort_buf[b] {
                sort_buf.swap(b - 1, b);
                b -= 1;
            }
        }

        let med = sort_buf[half];
        let center = src[i];
        let diff = (center.to_bits() as i64 - med.to_bits() as i64).abs();
        if diff >= threshold.to_bits() as i64 {
            dst[i] = med;
        } else {
            dst[i] = center;
        }
    }

    Status::Success
}

// --- FFT Fast Convolution ---

/// Performs fast linear convolution of `signal` and `kernel` via FFT multiplication.
/// Output length is `signal.len() + kernel.len() - 1`.
///
/// Requires the `transform` feature (enabled by `full`).
#[cfg(feature = "transform")]
pub fn fast_convolve_f32(signal: &[f32], kernel: &[f32], dst: &mut [f32]) -> Status {
    let len_sig = signal.len();
    let len_ker = kernel.len();
    if len_sig == 0 || len_ker == 0 {
        return Status::LengthError;
    }
    let total_len = len_sig + len_ker - 1;
    if dst.len() < total_len {
        return Status::LengthError;
    }

    // Find next power of 2
    let mut fft_n = 1;
    while fft_n < total_len {
        fft_n <<= 1;
    }

    if fft_n > 512 {
        // Fall back to time-domain convolution if size exceeds stack scratch buffer
        conv_f32(signal, kernel, dst);
        return Status::Success;
    }

    let mut sig_buf = [0.0f32; 1024]; // 2 * fft_n
    let mut ker_buf = [0.0f32; 1024];

    for i in 0..len_sig {
        sig_buf[2 * i] = signal[i];
    }
    for i in 0..len_ker {
        ker_buf[2 * i] = kernel[i];
    }

    cfft_f32(&mut sig_buf[..2 * fft_n], fft_n, 0, 1);
    cfft_f32(&mut ker_buf[..2 * fft_n], fft_n, 0, 1);

    // Pointwise complex multiplication: (a + jb) * (c + jd)
    for i in 0..fft_n {
        let a = sig_buf[2 * i];
        let b = sig_buf[2 * i + 1];
        let c = ker_buf[2 * i];
        let d = ker_buf[2 * i + 1];
        sig_buf[2 * i] = a * c - b * d;
        sig_buf[2 * i + 1] = a * d + b * c;
    }

    // Inverse FFT
    cfft_f32(&mut sig_buf[..2 * fft_n], fft_n, 1, 1);

    for i in 0..total_len {
        dst[i] = sig_buf[2 * i];
    }

    Status::Success
}

// --- Real-time Circular Buffer & Delay Line ---

/// Const-generic zero-allocation circular buffer and delay line for real-time DSP sample streams.
#[derive(Debug, Clone, Copy)]
pub struct CircularBuffer<T, const N: usize> {
    buffer: [T; N],
    head: usize,
    count: usize,
}

impl<T: Copy, const N: usize> CircularBuffer<T, N> {
    /// Creates a new circular buffer initialized with `init_val`.
    pub const fn new(init_val: T) -> Self {
        Self {
            buffer: [init_val; N],
            head: 0,
            count: 0,
        }
    }

    /// Pushes a new sample into the buffer, overwriting the oldest sample when full.
    #[inline(always)]
    pub fn push(&mut self, sample: T) {
        if N == 0 {
            return;
        }
        self.buffer[self.head] = sample;
        self.head = (self.head + 1) % N;
        if self.count < N {
            self.count += 1;
        }
    }

    /// Gets sample with historical lag $k$, where $k = 0$ is the newest sample (`x[n]`), $k = 1$ is `x[n-1]`, etc.
    /// Returns `None` if `lag >= self.len()`.
    #[inline(always)]
    pub fn get(&self, lag: usize) -> Option<T> {
        if lag >= self.count || N == 0 {
            return None;
        }
        let idx = (self.head + N - 1 - (lag % N)) % N;
        Some(self.buffer[idx])
    }

    /// Returns the most recently pushed sample (`x[n]`).
    #[inline(always)]
    pub fn latest(&self) -> Option<T> {
        self.get(0)
    }

    /// Returns the oldest sample stored in the buffer.
    #[inline(always)]
    pub fn oldest(&self) -> Option<T> {
        if self.count == 0 {
            None
        } else {
            self.get(self.count - 1)
        }
    }

    /// Returns the number of valid samples currently stored in the buffer.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns the capacity of the circular buffer (`N`).
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns `true` if the buffer contains no samples.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns `true` if the buffer is filled to capacity `N`.
    #[inline(always)]
    pub const fn is_full(&self) -> bool {
        self.count == N
    }

    /// Clears the circular buffer, resetting sample count and filling with `reset_val`.
    pub fn clear(&mut self, reset_val: T) {
        self.buffer = [reset_val; N];
        self.head = 0;
        self.count = 0;
    }
}

// --- Single-Pole Recursive Filter (Steven W. Smith, Ch. 19) ---

/// The cheapest possible IIR filter: a single-pole recursive low-pass or high-pass filter
/// (Steven W. Smith, Ch. 19, Eq. 19-2 / 19-3), needing only one or two multiplies per sample.
/// Coefficients are designed from a decay factor `x` (see
/// [`crate::filter_design::single_pole_decay_from_cutoff`] /
/// [`crate::filter_design::single_pole_decay_from_time_constant`]).
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct SinglePoleFilter {
    b0: f32,
    b1: f32,
    a1: f32,
    x1: f32,
    y1: f32,
}

impl SinglePoleFilter {
    /// Creates a single-pole low-pass filter from decay factor `x` (`0.0..1.0`); larger `x`
    /// means slower decay (a lower cutoff frequency).
    pub fn lowpass(decay: f32) -> Self {
        Self {
            b0: 1.0 - decay,
            b1: 0.0,
            a1: decay,
            x1: 0.0,
            y1: 0.0,
        }
    }

    /// Creates a single-pole high-pass filter from the same decay factor `x` used by
    /// [`SinglePoleFilter::lowpass`].
    pub fn highpass(decay: f32) -> Self {
        let b0 = (1.0 + decay) / 2.0;
        Self {
            b0,
            b1: -b0,
            a1: decay,
            x1: 0.0,
            y1: 0.0,
        }
    }

    /// Processes a single input sample and returns the filtered output.
    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.a1 * self.y1;
        self.x1 = x;
        self.y1 = y;
        y
    }

    /// Resets the filter's delay state to zero.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

/// Q15 single-pole recursive low-pass or high-pass filter (same recurrence as
/// [`SinglePoleFilter`]). `decay` is Q1.15 in `0..1` (larger → lower cutoff).
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct SinglePoleFilterQ15 {
    b0: q15,
    b1: q15,
    a1: q15,
    x1: q15,
    y1: q15,
}

impl SinglePoleFilterQ15 {
    /// Creates a single-pole low-pass filter from Q15 decay `x`.
    pub fn lowpass(decay: q15) -> Self {
        let decay = decay.max(q15::ZERO);
        Self {
            b0: q15::from_bits((32767i32 - decay.to_bits() as i32) as i16),
            b1: q15::ZERO,
            a1: decay,
            x1: q15::ZERO,
            y1: q15::ZERO,
        }
    }

    /// Creates a single-pole high-pass filter from the same Q15 decay used by
    /// [`SinglePoleFilterQ15::lowpass`].
    pub fn highpass(decay: q15) -> Self {
        let decay = decay.max(q15::ZERO);
        let b0 = q15::from_bits(((32767i32 + decay.to_bits() as i32) / 2) as i16);
        Self {
            b0,
            b1: -b0,
            a1: decay,
            x1: q15::ZERO,
            y1: q15::ZERO,
        }
    }

    /// Quantizes a floating-point decay in `0.0..1.0` to Q15 and builds a low-pass.
    pub fn lowpass_from_f32(decay: f32) -> Self {
        Self::lowpass(q15::saturating_from_num(decay.clamp(0.0, 1.0)))
    }

    /// Quantizes a floating-point decay in `0.0..1.0` to Q15 and builds a high-pass.
    pub fn highpass_from_f32(decay: f32) -> Self {
        Self::highpass(q15::saturating_from_num(decay.clamp(0.0, 1.0)))
    }

    /// Processes a single Q15 input sample and returns the filtered output.
    #[inline(always)]
    pub fn process(&mut self, x: q15) -> q15 {
        let y = (self.b0.to_bits() as i64 * x.to_bits() as i64
            + self.b1.to_bits() as i64 * self.x1.to_bits() as i64
            + self.a1.to_bits() as i64 * self.y1.to_bits() as i64)
            >> 15;
        let y = q15::from_bits(y.clamp(i16::MIN as i64, i16::MAX as i64) as i16);
        self.x1 = x;
        self.y1 = y;
        y
    }

    /// Resets the filter's delay state to zero.
    pub fn reset(&mut self) {
        self.x1 = q15::ZERO;
        self.y1 = q15::ZERO;
    }
}

/// High-pass single-pole used as a DC blocker (Smith Ch. 19).
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct DcBlockerQ15 {
    inner: SinglePoleFilterQ15,
}

impl DcBlockerQ15 {
    /// `decay` is the same Q15 factor as [`SinglePoleFilterQ15::highpass`].
    pub fn new(decay: q15) -> Self {
        Self {
            inner: SinglePoleFilterQ15::highpass(decay),
        }
    }

    /// Quantizes a floating-point decay in `0.0..1.0`.
    pub fn from_f32_decay(decay: f32) -> Self {
        Self {
            inner: SinglePoleFilterQ15::highpass_from_f32(decay),
        }
    }

    #[inline(always)]
    pub fn process(&mut self, x: q15) -> q15 {
        self.inner.process(x)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

// --- Recursive Moving Average Filter (Steven W. Smith, Ch. 15) ---

/// Const-generic `N`-point moving average filter implemented recursively (Steven W. Smith,
/// Ch. 15, Eq. 15-3): each sample is updated with a single add and subtract, instead of an
/// `O(N)` convolution sum.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct RecursiveMovingAverage<const N: usize> {
    history: CircularBuffer<f32, N>,
    sum: f32,
}

impl<const N: usize> RecursiveMovingAverage<N> {
    /// Creates a new `N`-point recursive moving average filter with empty history.
    pub const fn new() -> Self {
        Self {
            history: CircularBuffer::new(0.0),
            sum: 0.0,
        }
    }

    /// Pushes a new input sample and returns the updated moving average. While fewer than `N`
    /// samples have been seen, the average is taken over the (growing) window received so far.
    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        let oldest = if self.history.is_full() {
            self.history.oldest().unwrap_or(0.0)
        } else {
            0.0
        };
        self.sum += x - oldest;
        self.history.push(x);
        if self.history.len() == 0 {
            0.0
        } else {
            self.sum / self.history.len() as f32
        }
    }

    /// Resets the filter to its initial, empty state.
    pub fn reset(&mut self) {
        self.history.clear(0.0);
        self.sum = 0.0;
    }
}

impl<const N: usize> Default for RecursiveMovingAverage<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Q15 recursive `N`-point moving average (same recurrence as [`RecursiveMovingAverage`]).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct RecursiveMovingAverageQ15<const N: usize> {
    history: CircularBuffer<q15, N>,
    sum: i32,
}

impl<const N: usize> RecursiveMovingAverageQ15<N> {
    pub const fn new() -> Self {
        Self {
            history: CircularBuffer::new(q15::ZERO),
            sum: 0,
        }
    }

    #[inline(always)]
    pub fn process(&mut self, x: q15) -> q15 {
        let oldest = if self.history.is_full() {
            self.history.oldest().unwrap_or(q15::ZERO)
        } else {
            q15::ZERO
        };
        self.sum += x.to_bits() as i32 - oldest.to_bits() as i32;
        self.history.push(x);
        if self.history.len() == 0 {
            q15::ZERO
        } else {
            q15::from_bits(
                (self.sum / self.history.len() as i32).clamp(i16::MIN as i32, i16::MAX as i32)
                    as i16,
            )
        }
    }

    pub fn reset(&mut self) {
        self.history.clear(q15::ZERO);
        self.sum = 0;
    }
}

impl<const N: usize> Default for RecursiveMovingAverageQ15<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Robust Second-Order Sections (Biquads) with Anti-Windup & Clamping
// ─────────────────────────────────────────────────────────────────────────────

/// Direct Form 1 filter history state holding delayed inputs and outputs `[x1, x2, y1, y2]`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable))]
pub struct DirectForm1<T> {
    pub xy: [T; 4],
}

impl<T: Default + Copy> Default for DirectForm1<T> {
    fn default() -> Self {
        Self {
            xy: [T::default(); 4],
        }
    }
}

impl<T: Default + Copy> DirectForm1<T> {
    /// Create a new zero-initialized Direct Form 1 state.
    pub fn new() -> Self {
        Self {
            xy: [T::default(); 4],
        }
    }

    /// Reset internal state buffer.
    pub fn reset(&mut self) {
        self.xy = [T::default(); 4];
    }
}

/// Direct Form 2 Transposed filter state holding accumulator registers `[s0, s1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable))]
pub struct DirectForm2Transposed<T> {
    pub s: [T; 2],
}

impl<T: Default + Copy> Default for DirectForm2Transposed<T> {
    fn default() -> Self {
        Self {
            s: [T::default(); 2],
        }
    }
}

impl<T: Default + Copy> DirectForm2Transposed<T> {
    /// Create a new zero-initialized Direct Form 2 Transposed state.
    pub fn new() -> Self {
        Self {
            s: [T::default(); 2],
        }
    }

    /// Reset internal state buffer.
    pub fn reset(&mut self) {
        self.s = [T::default(); 2];
    }
}

/// Second-order section (SOS) biquadratic filter configuration.
///
/// Contains coefficients `ba: [b0, b1, b2, a1, a2]` normalized such that `a0 = 1`.
/// Recurrence relation:
/// `y0 = b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2`
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Biquad<T> {
    pub ba: [T; 5],
}

impl<T: Copy> Biquad<T> {
    /// Create a new Biquad configuration from coefficients `[b0, b1, b2, a1, a2]`.
    pub const fn new(b0: T, b1: T, b2: T, a1: T, a2: T) -> Self {
        Self {
            ba: [b0, b1, b2, a1, a2],
        }
    }
}

impl Biquad<f32> {
    /// Process a single input sample through Direct Form 1 state.
    #[inline(always)]
    pub fn process_df1(&self, state: &mut DirectForm1<f32>, x0: f32) -> f32 {
        let [b0, b1, b2, a1, a2] = self.ba;
        let [x1, x2, y1, y2] = state.xy;
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
        state.xy = [x0, x1, y0, y1];
        y0
    }

    /// Process a single input sample through Direct Form 2 Transposed state.
    #[inline(always)]
    pub fn process_df2t(&self, state: &mut DirectForm2Transposed<f32>, x0: f32) -> f32 {
        let [b0, b1, b2, a1, a2] = self.ba;
        let y0 = b0 * x0 + state.s[0];
        state.s[0] = b1 * x0 + a1 * y0 + state.s[1];
        state.s[1] = b2 * x0 + a2 * y0;
        y0
    }
}

impl Biquad<f64> {
    /// Process a single input sample through Direct Form 1 state.
    #[inline(always)]
    pub fn process_df1(&self, state: &mut DirectForm1<f64>, x0: f64) -> f64 {
        let [b0, b1, b2, a1, a2] = self.ba;
        let [x1, x2, y1, y2] = state.xy;
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
        state.xy = [x0, x1, y0, y1];
        y0
    }

    /// Process a single input sample through Direct Form 2 Transposed state.
    #[inline(always)]
    pub fn process_df2t(&self, state: &mut DirectForm2Transposed<f64>, x0: f64) -> f64 {
        let [b0, b1, b2, a1, a2] = self.ba;
        let y0 = b0 * x0 + state.s[0];
        state.s[0] = b1 * x0 + a1 * y0 + state.s[1];
        state.s[1] = b2 * x0 + a2 * y0;
        y0
    }
}

/// Biquadratic filter configuration with summing junction offset and anti-windup output clamping.
///
/// Clamps output between `[min, max]` at the summing junction before storing into feedback state,
/// preventing integrator windup and derivative kick when used in feedback control or PID applications.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct BiquadClamp<T> {
    pub coeff: Biquad<T>,
    /// Summing junction offset (setpoint)
    pub u: T,
    /// Minimum saturation clamp
    pub min: T,
    /// Maximum saturation clamp
    pub max: T,
}

impl<T: Copy> BiquadClamp<T> {
    /// Create a new clamped Biquad with coefficients, offset, and clamp bounds.
    pub const fn new(coeff: Biquad<T>, min: T, max: T, u: T) -> Self {
        Self { coeff, u, min, max }
    }
}

impl BiquadClamp<f32> {
    /// Process a sample using Direct Form 1 with anti-windup clamping.
    #[inline(always)]
    pub fn process_df1(&self, state: &mut DirectForm1<f32>, x0: f32) -> f32 {
        let [b0, b1, b2, a1, a2] = self.coeff.ba;
        let [x1, x2, y1, y2] = state.xy;
        let unclamped = b0 * x0 + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2 + self.u;
        let y0 = unclamped.clamp(self.min, self.max);
        state.xy = [x0, x1, y0, y1];
        y0
    }

    /// Process a sample using Direct Form 2 Transposed with anti-windup clamping.
    #[inline(always)]
    pub fn process_df2t(&self, state: &mut DirectForm2Transposed<f32>, x0: f32) -> f32 {
        let [b0, b1, b2, a1, a2] = self.coeff.ba;
        let y0 = (b0 * x0 + state.s[0] + self.u).clamp(self.min, self.max);
        state.s[0] = b1 * x0 + a1 * y0 + state.s[1];
        state.s[1] = b2 * x0 + a2 * y0;
        y0
    }
}

impl BiquadClamp<f64> {
    /// Process a sample using Direct Form 1 with anti-windup clamping.
    #[inline(always)]
    pub fn process_df1(&self, state: &mut DirectForm1<f64>, x0: f64) -> f64 {
        let [b0, b1, b2, a1, a2] = self.coeff.ba;
        let [x1, x2, y1, y2] = state.xy;
        let unclamped = b0 * x0 + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2 + self.u;
        let y0 = unclamped.clamp(self.min, self.max);
        state.xy = [x0, x1, y0, y1];
        y0
    }

    /// Process a sample using Direct Form 2 Transposed with anti-windup clamping.
    #[inline(always)]
    pub fn process_df2t(&self, state: &mut DirectForm2Transposed<f64>, x0: f64) -> f64 {
        let [b0, b1, b2, a1, a2] = self.coeff.ba;
        let y0 = (b0 * x0 + state.s[0] + self.u).clamp(self.min, self.max);
        state.s[0] = b1 * x0 + a1 * y0 + state.s[1];
        state.s[1] = b2 * x0 + a2 * y0;
        y0
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f32, f32, DirectForm1<f32>> for Biquad<f32> {
    #[inline(always)]
    fn process(&self, state: &mut DirectForm1<f32>, x: f32) -> f32 {
        self.process_df1(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f32, f32, DirectForm2Transposed<f32>> for Biquad<f32> {
    #[inline(always)]
    fn process(&self, state: &mut DirectForm2Transposed<f32>, x: f32) -> f32 {
        self.process_df2t(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f32, f32, DirectForm1<f32>> for BiquadClamp<f32> {
    #[inline(always)]
    fn process(&self, state: &mut DirectForm1<f32>, x: f32) -> f32 {
        self.process_df1(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f32, f32, DirectForm2Transposed<f32>> for BiquadClamp<f32> {
    #[inline(always)]
    fn process(&self, state: &mut DirectForm2Transposed<f32>, x: f32) -> f32 {
        self.process_df2t(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f64, f64, DirectForm1<f64>> for Biquad<f64> {
    #[inline(always)]
    fn process(&self, state: &mut DirectForm1<f64>, x: f64) -> f64 {
        self.process_df1(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f64, f64, DirectForm2Transposed<f64>> for Biquad<f64> {
    #[inline(always)]
    fn process(&self, state: &mut DirectForm2Transposed<f64>, x: f64) -> f64 {
        self.process_df2t(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f64, f64, DirectForm1<f64>> for BiquadClamp<f64> {
    #[inline(always)]
    fn process(&self, state: &mut DirectForm1<f64>, x: f64) -> f64 {
        self.process_df1(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f64, f64, DirectForm2Transposed<f64>> for BiquadClamp<f64> {
    #[inline(always)]
    fn process(&self, state: &mut DirectForm2Transposed<f64>, x: f64) -> f64 {
        self.process_df2t(state, x)
    }
}

/// Direct Form 1 state with quantization error feedback for noise shaping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable))]
pub struct DirectForm1NoiseShaped {
    pub xy: [i32; 4],
    pub err: i32,
}

impl DirectForm1NoiseShaped {
    /// Create a new zero-initialized state with zero error feedback.
    pub const fn new() -> Self {
        Self {
            xy: [0; 4],
            err: 0,
        }
    }

    /// Reset internal state and error accumulator.
    pub fn reset(&mut self) {
        self.xy = [0; 4];
        self.err = 0;
    }
}

/// Fixed-point 32-bit Biquad with parameterized fractional scaling and anti-windup clamping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct BiquadFixed<const SHIFT: u32 = 30> {
    /// Fixed-point coefficients `[b0, b1, b2, a1, a2]`.
    pub ba: [i32; 5],
    /// Summing junction offset
    pub u: i32,
    /// Minimum saturation clamp
    pub min: i32,
    /// Maximum saturation clamp
    pub max: i32,
}

impl<const SHIFT: u32> BiquadFixed<SHIFT> {
    /// Create a new fixed-point biquad configuration.
    pub const fn new(ba: [i32; 5], min: i32, max: i32, u: i32) -> Self {
        Self { ba, min, max, u }
    }

    /// Process single sample with 1st-order noise shaping to eliminate limit cycles.
    #[inline(always)]
    pub fn process_noise_shaped(&self, state: &mut DirectForm1NoiseShaped, x0: i32) -> i32 {
        let [b0, b1, b2, a1, a2] = self.ba;
        let [x1, x2, y1, y2] = state.xy;
        let acc = (b0 as i64 * x0 as i64)
            + (b1 as i64 * x1 as i64)
            + (b2 as i64 * x2 as i64)
            + (a1 as i64 * y1 as i64)
            + (a2 as i64 * y2 as i64)
            + ((self.u as i64) << SHIFT)
            - state.err as i64; // noise shaping feedback
        let scaled = acc >> SHIFT;
        let y0 = scaled.clamp(self.min as i64, self.max as i64) as i32;
        state.err = (acc - ((y0 as i64) << SHIFT)) as i32;
        state.xy = [x0, x1, y0, y1];
        y0
    }
}

#[cfg(feature = "pipeline")]
impl<const SHIFT: u32> crate::pipeline::SplitProcess<i32, i32, DirectForm1NoiseShaped>
    for BiquadFixed<SHIFT>
{
    #[inline(always)]
    fn process(&self, state: &mut DirectForm1NoiseShaped, x: i32) -> i32 {
        self.process_noise_shaped(state, x)
    }
}

/// Delta-sigma modulator in MASH-(1)^K architecture.
///
/// Converts a 32-bit unsigned input sample `x` into an integer stream with average
/// value `x / 2^32`, shaping quantization noise up by `K * 20 dB/decade`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dsm<const K: usize> {
    pub a: [u32; K],
    pub c: [i8; K],
}

impl<const K: usize> Default for Dsm<K> {
    fn default() -> Self {
        Self {
            a: [0; K],
            c: [0; K],
        }
    }
}

impl<const K: usize> Dsm<K> {
    /// Create a new zeroed Delta-Sigma modulator.
    pub const fn new() -> Self {
        Self {
            a: [0; K],
            c: [0; K],
        }
    }

    /// Process a new 32-bit sample and return modulated output.
    #[inline]
    pub fn process_sample(&mut self, x: u32) -> i8 {
        let mut d = 0i8;
        for a in self.a.iter_mut() {
            let (next_a, c) = a.overflowing_add(x);
            *a = next_a;
            d = (d << 1) | c as i8;
        }
        let mut y = d & 1;
        for c in self.c.iter_mut().take(K.saturating_sub(1)) {
            d >>= 1;
            let next_y = (d & 1) + y - *c;
            *c = y;
            y = next_y;
        }
        y
    }
}

#[cfg(feature = "pipeline")]
impl<const K: usize> crate::pipeline::Process<u32, i8> for Dsm<K> {
    #[inline(always)]
    fn process(&mut self, x: u32) -> i8 {
        self.process_sample(x)
    }
}

/// Lightweight 32-bit XorShift pseudorandom generator for dither synthesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XorShift32(pub u32);

impl Default for XorShift32 {
    fn default() -> Self {
        Self::new(0x12345678)
    }
}

impl XorShift32 {
    /// Create a new XorShift32 PRNG from non-zero seed.
    #[inline(always)]
    pub const fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0x12345678 } else { seed })
    }

    /// Produce next pseudorandom 32-bit word.
    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Produce next uniform float in `[0.0, 1.0)`.
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16777216.0)
    }

    /// Triangular Probability Density Function (TPDF) dither sample in `[-1.0, 1.0]`.
    #[inline(always)]
    pub fn tpdf_dither_f32(&mut self) -> f32 {
        let r1 = self.next_f32();
        let r2 = self.next_f32();
        r1 - r2
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lock-in Demodulation & Lock-in Amplifier
// ─────────────────────────────────────────────────────────────────────────────

use crate::types::Complex;

/// Dual-phase lock-in amplifier mixer and demodulator.
///
/// Combines channel filters `C` with an IQ local oscillator reference to demodulate
/// a noisy input signal into in-phase $I$ and quadrature $Q$ components.
#[derive(Copy, Clone, Default, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Lockin<C>(pub C);

impl<C> Lockin<C> {
    /// Create a new lock-in demodulator with the given low-pass channel filter.
    pub const fn new(filter: C) -> Self {
        Self(filter)
    }
}

#[cfg(feature = "pipeline")]
impl<X, U, C, S> crate::pipeline::SplitProcess<(X, Complex<U>), Complex<X>, [S; 2]> for Lockin<C>
where
    X: Copy + core::ops::Mul<U, Output = X>,
    U: Copy,
    C: crate::pipeline::SplitProcess<X, X, S>,
{
    /// Demodulate a sample `x.0` against a local oscillator `x.1` (in-phase and quadrature).
    #[inline]
    fn process(&self, state: &mut [S; 2], x: (X, Complex<U>)) -> Complex<X> {
        let (sample, lo) = x;
        Complex::new(
            self.0.process(&mut state[0], sample * lo.real),
            self.0.process(&mut state[1], sample * lo.imag),
        )
    }
}

/// Standalone Lock-in Amplifier with integrated single-pole low-pass filtering.
///
/// Multiplies an incoming signal with an internal or external quadrature reference,
/// and low-pass filters both channels to extract amplitude and phase.
#[derive(Clone, Copy, Debug)]
pub struct LockinAmplifier {
    pub filter_i: SinglePoleFilter,
    pub filter_q: SinglePoleFilter,
    pub phase: i32,
    pub phase_inc: i32,
}

impl LockinAmplifier {
    /// Create a new Lock-in Amplifier with carrier frequency, sample rate, and low-pass decay factor.
    pub fn new(carrier_hz: f32, sample_rate: f32, filter_decay: f32) -> Self {
        let phase_inc = ((carrier_hz / sample_rate) * 4294967296.0) as i32;
        Self {
            filter_i: SinglePoleFilter::lowpass(filter_decay),
            filter_q: SinglePoleFilter::lowpass(filter_decay),
            phase: 0,
            phase_inc,
        }
    }

    /// Set carrier frequency in Hz.
    pub fn set_frequency(&mut self, carrier_hz: f32, sample_rate: f32) {
        self.phase_inc = ((carrier_hz / sample_rate) * 4294967296.0) as i32;
    }

    /// Reset internal filter state and phase accumulator.
    pub fn reset(&mut self) {
        self.filter_i.reset();
        self.filter_q.reset();
        self.phase = 0;
    }

    /// Ingest a sample and return demodulated IQ `Complex<f32>`.
    #[inline]
    pub fn process(&mut self, sample: f32) -> Complex<f32> {
        let (cos_ref, sin_ref) = {
            #[cfg(feature = "fast-math")]
            {
                let (c, s) = crate::fast_math::cossin(self.phase);
                (c as f32 * (1.0 / 2147483648.0), s as f32 * (1.0 / 2147483648.0))
            }
            #[cfg(not(feature = "fast-math"))]
            {
                let rad = self.phase as f32 * (core::f32::consts::PI / 2147483648.0);
                (FloatMath::cos(rad), FloatMath::sin(rad))
            }
        };

        self.phase = self.phase.wrapping_add(self.phase_inc);

        let i_filt = self.filter_i.process(sample * cos_ref);
        let q_filt = self.filter_q.process(sample * sin_ref);
        Complex::new(i_filt, q_filt)
    }

    /// Process a sample using an external reference phase angle (in radians).
    #[inline]
    pub fn process_with_phase(&mut self, sample: f32, phase_rad: f32) -> Complex<f32> {
        let (cos_ref, sin_ref) = {
            #[cfg(feature = "fast-math")]
            {
                crate::fast_math::cossin_f32(phase_rad)
            }
            #[cfg(not(feature = "fast-math"))]
            {
                (FloatMath::cos(phase_rad), FloatMath::sin(phase_rad))
            }
        };

        let i_filt = self.filter_i.process(sample * cos_ref);
        let q_filt = self.filter_q.process(sample * sin_ref);
        Complex::new(i_filt, q_filt)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Integer Lowpass Filter (ported from idsp)
// ─────────────────────────────────────────────────────────────────────────────

/// Arbitrary-order integer lowpass filter with high dynamic range. DC gain is 1.
///
/// Supports order `N = 1` (first-order) and `N = 2` (second-order Butterworth).
/// The filter saturates cleanly towards the `i32` range.
///
/// # Coefficient Calculation
///
/// **First-order** (`N = 1`): `k[0] = π * (1 << 31) * f0 / fn`  
/// where `f0` is the 3 dB corner frequency and `fn` is the Nyquist frequency.
///
/// **Second-order Butterworth** (`N = 2`): `k = [k_sq >> 32, -k / q]`  
/// where `q = 1/sqrt(2)` and `k` is as above.
///
/// Both variants have zeros at Nyquist, optimised for Cortex-M7.
///
/// Ported from the `idsp` crate by the Sinara/ARTIQ project.
#[derive(Clone, Debug)]
pub struct IntLowpass<const N: usize> {
    /// Lead/lag gain coefficients in Q1.31 fixed-point.
    pub k: [i32; N],
    /// Wide internal state accumulators.
    state: [i64; N],
}

impl<const N: usize> Default for IntLowpass<N>
where
    [i32; N]: Default,
{
    fn default() -> Self {
        Self { k: Default::default(), state: [0i64; N] }
    }
}

impl<const N: usize> IntLowpass<N> {
    /// Create a new filter from gain coefficients.
    pub fn new(k: [i32; N]) -> Self {
        Self { k, state: [0i64; N] }
    }

    /// Reset internal state to zero.
    pub fn reset(&mut self) {
        self.state = [0i64; N];
    }

    /// Process a single sample and return the filtered output.
    pub fn process(&mut self, x: i32) -> i32 {
        if N == 1 {
            let d = x.saturating_sub((self.state[0] >> 32) as i32) as i64
                * self.k[0] as i64;
            self.state[0] += d;
            let y = (self.state[0] >> 32) as i32;
            self.state[0] += d;
            y
        } else if N == 2 {
            let mut d = x.saturating_sub((self.state[0] >> 32) as i32) as i64
                * self.k[0] as i64;
            d += (self.state[1] >> 32) * self.k[1] as i64;
            self.state[1] += d;
            self.state[0] += self.state[1];
            let y = (self.state[0] >> 32) as i32;
            self.state[0] += self.state[1];
            self.state[1] += d;
            y
        } else {
            unimplemented!()
        }
    }
}

/// First-order integer lowpass (alias for `IntLowpass<1>`).
pub type IntLowpass1 = IntLowpass<1>;
/// Second-order integer lowpass (alias for `IntLowpass<2>`).
pub type IntLowpass2 = IntLowpass<2>;
