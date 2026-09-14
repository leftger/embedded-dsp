//! LMS / NLMS adaptive filters.

use crate::types::*;

// --- LMS Adaptive Filter ---

/// Scalar algebra for the adaptive filters, whose float and fixed-point coefficient updates are
/// genuinely different operations. Only the widths that ship an adaptive filter implement it; the
/// [`LmsInstance`]/[`NlmsInstance`] stages themselves are generic over it.
pub trait AdaptiveSample: DspSample<Coeff = Self> {
    /// Leaky-LMS retention factor `keep` for `w ← keep·w + alpha·x`.
    fn lms_keep(leak: Self::Coeff) -> Self::Accum;
    /// LMS step `2·μ·e`.
    fn lms_alpha(mu: Self::Coeff, e: Self) -> Self::Accum;
    /// One LMS coefficient update: `w ← keep·w + alpha·x`.
    fn lms_apply(w: Self::Coeff, x: Self, alpha: Self::Accum, keep: Self::Accum) -> Self::Coeff;
    /// NLMS power denominator seed from `eps` (fixed-point floors at one LSB).
    fn nlms_power_seed(eps: Self::Coeff) -> Self::Accum;
    /// NLMS step `μ·e / power`.
    fn nlms_alpha(mu: Self::Coeff, e: Self, power: Self::Accum) -> Self::Accum;
    /// One NLMS coefficient update: `w ← w + alpha·x`.
    fn nlms_apply(w: Self::Coeff, x: Self, alpha: Self::Accum) -> Self::Coeff;
}

impl AdaptiveSample for f32 {
    #[inline(always)]
    fn lms_keep(leak: f32) -> f32 {
        1.0 - leak
    }
    #[inline(always)]
    fn lms_alpha(mu: f32, e: f32) -> f32 {
        2.0 * mu * e
    }
    #[inline(always)]
    fn lms_apply(w: f32, x: f32, alpha: f32, keep: f32) -> f32 {
        keep * w + alpha * x
    }
    #[inline(always)]
    fn nlms_power_seed(eps: f32) -> f32 {
        eps
    }
    #[inline(always)]
    fn nlms_alpha(mu: f32, e: f32, power: f32) -> f32 {
        mu * e / power
    }
    #[inline(always)]
    fn nlms_apply(w: f32, x: f32, alpha: f32) -> f32 {
        w + alpha * x
    }
}

impl AdaptiveSample for q15 {
    #[inline(always)]
    fn lms_keep(leak: q15) -> i64 {
        32_767i64 - leak.to_bits().max(0) as i64
    }
    #[inline(always)]
    fn lms_alpha(mu: q15, e: q15) -> i64 {
        (2 * mu.to_bits() as i64 * e.to_bits() as i64) >> 15
    }
    #[inline(always)]
    fn lms_apply(w: q15, x: q15, alpha: i64, keep: i64) -> q15 {
        let leaked = (keep * w.to_bits() as i64) >> 15;
        let upd = leaked + ((alpha * x.to_bits() as i64) >> 15);
        q15::from_bits(upd.clamp(i16::MIN as i64, i16::MAX as i64) as i16)
    }
    #[inline(always)]
    fn nlms_power_seed(eps: q15) -> i64 {
        eps.to_bits().max(1) as i64
    }
    #[inline(always)]
    fn nlms_alpha(mu: q15, e: q15, power: i64) -> i64 {
        (mu.to_bits() as i64 * e.to_bits() as i64) / power
    }
    #[inline(always)]
    fn nlms_apply(w: q15, x: q15, alpha: i64) -> q15 {
        let upd = w.to_bits() as i64 + ((alpha * x.to_bits() as i64) >> 15);
        q15::from_bits(upd.clamp(i16::MIN as i64, i16::MAX as i64) as i16)
    }
}

/// Instance structure for the LMS adaptive filter, generic over the sample width.
pub struct LmsInstance<'a, T: AdaptiveSample> {
    /// Number of filter taps.
    pub num_taps: u16,
    /// Filter coefficients.
    pub coeffs: &'a mut [T::Coeff],
    /// Filter state buffer.
    pub state: &'a mut [T],
    /// Adaptation step size.
    pub mu: T::Coeff,
}

impl<'a, T: AdaptiveSample> LmsInstance<'a, T> {
    /// Initializes the instance.
    pub fn init(
        num_taps: u16,
        coeffs: &'a mut [T::Coeff],
        state: &'a mut [T],
        mu: T::Coeff,
    ) -> Self {
        state.fill(T::ZERO);
        coeffs.fill(T::coeff_from_f32(0.0));
        Self {
            num_taps,
            coeffs,
            state,
            mu,
        }
    }
}

/// LMS adaptive filtering, generic over the sample width.
pub fn lms<T: AdaptiveSample>(
    instance: &mut LmsInstance<'_, T>,
    src: &[T],
    ref_signal: &[T],
    out: &mut [T],
    err: &mut [T],
) {
    lms_leaky(instance, src, ref_signal, out, err, T::coeff_from_f32(0.0));
}

/// Leaky LMS: `w ← keep·w + 2 μ e x`. `leak = 0` matches [`lms`].
pub fn lms_leaky<T: AdaptiveSample>(
    instance: &mut LmsInstance<'_, T>,
    src: &[T],
    ref_signal: &[T],
    out: &mut [T],
    err: &mut [T],
    leak: T::Coeff,
) {
    let num_taps = instance.num_taps as usize;
    let block_size = src
        .len()
        .min(ref_signal.len())
        .min(out.len())
        .min(err.len());
    let keep = T::lms_keep(leak);

    for i in 0..block_size {
        for k in (1..num_taps).rev() {
            instance.state[k] = instance.state[k - 1];
        }
        instance.state[0] = src[i];

        let mut acc = T::Accum::default();
        for k in 0..num_taps {
            acc = T::madd(acc, instance.state[k], instance.coeffs[k]);
        }
        let y = T::from_accum(acc);
        out[i] = y;
        let e = T::sat_sub(ref_signal[i], y);
        err[i] = e;

        let alpha = T::lms_alpha(instance.mu, e);
        for k in 0..num_taps {
            instance.coeffs[k] = T::lms_apply(instance.coeffs[k], instance.state[k], alpha, keep);
        }
    }
}

/// Instance structure for the normalized LMS adaptive filter, generic over the sample width.
pub struct NlmsInstance<'a, T: AdaptiveSample> {
    /// Number of filter taps.
    pub num_taps: u16,
    /// Filter coefficients.
    pub coeffs: &'a mut [T::Coeff],
    /// Filter state buffer.
    pub state: &'a mut [T],
    /// Adaptation step size.
    pub mu: T::Coeff,
    /// Regularization epsilon.
    pub eps: T::Coeff,
}

impl<'a, T: AdaptiveSample> NlmsInstance<'a, T> {
    /// Initializes the instance.
    pub fn init(
        num_taps: u16,
        coeffs: &'a mut [T::Coeff],
        state: &'a mut [T],
        mu: T::Coeff,
        eps: T::Coeff,
    ) -> Self {
        state.fill(T::ZERO);
        coeffs.fill(T::coeff_from_f32(0.0));
        Self {
            num_taps,
            coeffs,
            state,
            mu,
            eps,
        }
    }
}

/// Normalized LMS: `w ← w + μ e x / (eps + ‖x‖²)`, generic over the sample width.
pub fn nlms<T: AdaptiveSample>(
    instance: &mut NlmsInstance<'_, T>,
    src: &[T],
    ref_signal: &[T],
    out: &mut [T],
    err: &mut [T],
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

        let mut acc = T::Accum::default();
        let mut power = T::nlms_power_seed(instance.eps);
        for k in 0..num_taps {
            acc = T::madd(acc, instance.state[k], instance.coeffs[k]);
            power = power + T::mul_high(instance.state[k], instance.state[k]);
        }
        let y = T::from_accum(acc);
        out[i] = y;
        let e = T::sat_sub(ref_signal[i], y);
        err[i] = e;

        let alpha = T::nlms_alpha(instance.mu, e, power);
        for k in 0..num_taps {
            instance.coeffs[k] = T::nlms_apply(instance.coeffs[k], instance.state[k], alpha);
        }
    }
}


