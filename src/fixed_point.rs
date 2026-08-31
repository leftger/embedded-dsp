//! Q16.16 fixed-point math.
//!
//! A saturating fixed-point number format (`i32`, 16 fractional bits) for
//! pipelines that can't afford float math (no FPU, or the accuracy/latency
//! trade-off isn't worth it): rasterizer transforms, angle math, scanline
//! interpolation. Complements the `q7`/`q15`/`q31`/`q63` aliases in
//! [`crate::types`], which are narrower fixed-point formats aimed at
//! CMSIS-DSP-style signal processing rather than screen-space geometry.
//!
//! - Type alias [`Q16`], plus [`FP_ONE`], [`Q16_MAX`], [`Q16_MIN`]
//! - Conversion: `f32 <-> Q16`, `i16 <-> Q16`, `q31 <-> Q16`
//! - Arithmetic: [`mul_q16`], [`mul_n_q16`], [`mul_f_q16`], [`div_q16`], [`div_n_q16`], [`div_f_q16`]
//! - Saturating: [`qadd_q16`], [`qsub_q16`], [`abs_q16`]
//! - Helpers: [`lerp_q16`], [`angle_to_q16`], [`recip_q16`]
//! - Trig (with the `lut` feature): [`crate::lut::sin_q16`], [`crate::lut::cos_q16`]
//! - [`ScanlineInterp`] — accelerated per-scanline z + (u, v) interpolation

use fixed::types::I16F16;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Fractional bit count for Q16.16.
pub const FP_SHIFT: u32 = 16;

/// 1.0 in Q16.16 representation.
pub const FP_ONE: i32 = 1_i32 << FP_SHIFT;

/// Maximum positive value of a Q16.16 number (same as i32::MAX).
pub const Q16_MAX: i32 = i32::MAX;

/// Minimum value of a Q16.16 number (same as i32::MIN).
pub const Q16_MIN: i32 = i32::MIN;

/// Type alias: `Q16` is `i32` stored in Q16.16 fixed-point format.
///
/// The integer part occupies bits 31..16, the fractional part bits 15..0.
pub type Q16 = i32;

// ─────────────────────────────────────────────────────────────────────────────
// Conversions
// ─────────────────────────────────────────────────────────────────────────────

/// Convert `f32` -> `Q16.16` with correct rounding (half-to-even at exact ties).
#[inline(always)]
pub fn to_q16(v: f32) -> Q16 {
    I16F16::saturating_from_num(v).to_bits()
}

/// Convert `Q16.16` -> `f32`.
#[inline(always)]
pub fn from_q16(v: Q16) -> f32 {
    I16F16::from_bits(v).to_num::<f32>()
}

/// Convert `i16` integer -> `Q16.16` (shift left 16).
#[inline(always)]
pub fn from_i16_q16(v: i16) -> Q16 {
    I16F16::from_num(v).to_bits()
}

/// Convert `Q16.16` -> `i16` integer (truncate fractional bits).
#[inline(always)]
pub fn to_i16_q16(v: Q16) -> i16 {
    I16F16::from_bits(v).to_num::<i16>()
}

/// Reinterpret a Q31 value as Q16.16 by shifting right 15 bits.
#[inline(always)]
pub fn q31_to_q16(v: i32) -> Q16 {
    I16F16::from_bits(v >> 15).to_bits()
}

/// Reinterpret Q16.16 as Q31 by shifting left 16 bits (use i64 to avoid overflow).
#[inline(always)]
pub fn q16_to_q31(v: Q16) -> i64 {
    (I16F16::from_bits(v).to_bits() as i64) << 16
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic
// ─────────────────────────────────────────────────────────────────────────────

/// Multiply two Q16.16 values. Uses i64 intermediate (SMULL on Cortex-M).
#[inline(always)]
pub fn mul_q16(a: Q16, b: Q16) -> Q16 {
    I16F16::from_bits(a).wrapping_mul(I16F16::from_bits(b)).to_bits()
}

/// Multiply a Q16.16 value by a plain `i32` integer (no fractional scaling).
#[inline(always)]
pub fn mul_n_q16(a: Q16, n: i32) -> Q16 {
    I16F16::from_bits(a).wrapping_mul_int(n).to_bits()
}

/// Multiply a Q16.16 value by an `f32` scalar.
#[inline(always)]
pub fn mul_f_q16(a: Q16, f: f32) -> Q16 {
    mul_q16(a, to_q16(f))
}

/// Divide two Q16.16 values. Returns 0 on division by zero.
#[inline(always)]
pub fn div_q16(a: Q16, b: Q16) -> Q16 {
    if b == 0 {
        return 0;
    }
    I16F16::from_bits(a).wrapping_div(I16F16::from_bits(b)).to_bits()
}

/// Divide a Q16.16 value by a plain `i32` integer. Returns 0 on division by zero.
#[inline(always)]
pub fn div_n_q16(a: Q16, n: i32) -> Q16 {
    if n == 0 {
        return 0;
    }
    I16F16::from_bits(a).wrapping_div_int(n).to_bits()
}

/// Divide a Q16.16 value by an `f32` scalar.
#[inline(always)]
pub fn div_f_q16(a: Q16, f: f32) -> Q16 {
    div_q16(a, to_q16(f))
}

// ─────────────────────────────────────────────────────────────────────────────
// Saturating arithmetic
// ─────────────────────────────────────────────────────────────────────────────

/// Saturating add: clamps the result to `[i32::MIN, i32::MAX]`.
#[inline(always)]
pub const fn qadd_q16(a: Q16, b: Q16) -> Q16 {
    I16F16::from_bits(a).saturating_add(I16F16::from_bits(b)).to_bits()
}

/// Saturating subtract: clamps the result to `[i32::MIN, i32::MAX]`.
#[inline(always)]
pub const fn qsub_q16(a: Q16, b: Q16) -> Q16 {
    I16F16::from_bits(a).saturating_sub(I16F16::from_bits(b)).to_bits()
}

/// Absolute value of a Q16.16 number.
#[inline(always)]
pub fn abs_q16(a: Q16) -> Q16 {
    I16F16::from_bits(a).abs().to_bits()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Linear interpolation in Q16.16:  `a + (b - a) * t / denom`.
///
/// `t` and `denom` are plain integers (scan-line step counts).
/// Uses i64 to avoid overflow on large `(b - a)` spans.
#[inline(always)]
pub fn lerp_q16(a: Q16, b: Q16, t: i32, denom: i32) -> Q16 {
    if denom == 0 {
        return a;
    }
    let diff = b as i64 - a as i64;
    (a as i64 + diff * t as i64 / denom as i64) as i32
}

/// Convert an angle in degrees to Q16.16 radians.
#[inline(always)]
pub fn angle_to_q16(degrees: f32) -> Q16 {
    to_q16(degrees * core::f32::consts::PI / 180.0)
}

/// Fast reciprocal approximation: `1.0 / v` in Q16.16.
///
/// Uses integer shift: `(1 << 32) / v` — gives approx 6 correct decimal digits.
/// Returns `Q16_MAX` for zero or near-zero input (safe sentinel).
#[inline(always)]
pub fn recip_q16(v: Q16) -> Q16 {
    if v == 0 {
        return Q16_MAX;
    }
    I16F16::from_bits(v).recip().to_bits()
}

// ─────────────────────────────────────────────────────────────────────────────
// ScanlineInterp — accelerated z-buffer + UV interpolation
// ─────────────────────────────────────────────────────────────────────────────

/// Per-scanline interpolator for z-buffer depth and (u, v) texture coordinates.
///
/// Pre-computes Q16.16 per-pixel step values for `z`, `u`, and `v` so the
/// inner loop only does three wrapping additions instead of floating-point
/// divisions per pixel — useful for MCU scanline rasterization.
///
/// # Usage
/// ```rust,no_run
/// # use embedded_dsp::fixed_point::ScanlineInterp;
/// # let left_z = 0u32;
/// # let right_z = 0u32;
/// # let left_u = 0u32;
/// # let right_u = 0u32;
/// # let left_v = 0u32;
/// # let right_v = 0u32;
/// # let span_pixels = 0i32;
/// let mut interp = ScanlineInterp::new(
///     left_z,  right_z,   // u32 depth values (Q16.16)
///     left_u,  right_u,   // u32 U texture coords (Q16.16)
///     left_v,  right_v,   // u32 V texture coords (Q16.16)
///     span_pixels,         // number of pixels across the scanline
/// );
///
/// for _x in 0..=span_pixels {
///     let z = interp.z();
///     let u = interp.u();
///     let v = interp.v();
///     // ... depth test, texture sample, write pixel ...
///     interp.step();
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ScanlineInterp {
    z_cur: u32,
    z_step: i32,
    u_cur: u32,
    u_step: i32,
    v_cur: u32,
    v_step: i32,
}

impl ScanlineInterp {
    /// Create a new scanline interpolator.
    ///
    /// # Arguments
    /// - `z_left`, `z_right` — depth at the left and right scanline endpoints (Q16.16 `u32`)
    /// - `u_left`, `u_right` — U texture coordinates (Q16.16 `u32`, range `[0, 65536]`)
    /// - `v_left`, `v_right` — V texture coordinates (Q16.16 `u32`, range `[0, 65536]`)
    /// - `span` — number of pixels across the scanline (0 is valid — returns left values)
    #[inline]
    pub fn new(
        z_left: u32,
        z_right: u32,
        u_left: u32,
        u_right: u32,
        v_left: u32,
        v_right: u32,
        span: i32,
    ) -> Self {
        let (z_step, u_step, v_step) = if span > 0 {
            let z_step = ((z_right as i64 - z_left as i64) / span as i64) as i32;
            let u_step = ((u_right as i64 - u_left as i64) / span as i64) as i32;
            let v_step = ((v_right as i64 - v_left as i64) / span as i64) as i32;
            (z_step, u_step, v_step)
        } else {
            (0, 0, 0)
        };
        Self {
            z_cur: z_left,
            z_step,
            u_cur: u_left,
            u_step,
            v_cur: v_left,
            v_step,
        }
    }

    /// Create an interpolator for depth-only scanlines (no texture mapping).
    #[inline]
    pub fn depth_only(z_left: u32, z_right: u32, span: i32) -> Self {
        Self::new(z_left, z_right, 0, 0, 0, 0, span)
    }

    /// Current depth value (Q16.16 `u32`).
    #[inline(always)]
    pub fn z(&self) -> u32 {
        self.z_cur
    }

    /// Current U texture coordinate (Q16.16 `u32`).
    #[inline(always)]
    pub fn u(&self) -> u32 {
        self.u_cur
    }

    /// Current V texture coordinate (Q16.16 `u32`).
    #[inline(always)]
    pub fn v(&self) -> u32 {
        self.v_cur
    }

    /// Advance all interpolators by one pixel.
    #[inline(always)]
    pub fn step(&mut self) {
        self.z_cur = self.z_cur.wrapping_add_signed(self.z_step);
        self.u_cur = self.u_cur.wrapping_add_signed(self.u_step);
        self.v_cur = self.v_cur.wrapping_add_signed(self.v_step);
    }

    /// Advance `n` pixels at once (useful for skipping clipped scanline segments).
    #[inline]
    pub fn step_n(&mut self, n: i32) {
        self.z_cur = self.z_cur.wrapping_add_signed(self.z_step.wrapping_mul(n));
        self.u_cur = self.u_cur.wrapping_add_signed(self.u_step.wrapping_mul(n));
        self.v_cur = self.v_cur.wrapping_add_signed(self.v_step.wrapping_mul(n));
    }

    /// Current depth as `f32`.
    #[inline(always)]
    pub fn z_f32(&self) -> f32 {
        self.z_cur as f32 / 65536.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn to_q16_tie_rounding() {
        // 2.5/65536 is an exact tie: `fixed`'s `saturating_from_num` rounds
        // half-to-even (2.5 -> 2), unlike the old hand-rolled
        // half-away-from-zero rounding (2.5 -> 3). This is a confirmed,
        // accepted behavior change that only manifests at exact ties.
        let v: f32 = 2.5 / 65536.0;
        assert_eq!(to_q16(v), 2, "tie value rounds half-to-even under `fixed`");
    }

    #[test]
    fn to_i16_q16_negative_fraction_floors() {
        // -3.5 in Q16.16 must floor to -4 via arithmetic shift, not truncate
        // toward zero (which would give -3).
        let q = to_q16(-3.5);
        assert_eq!(to_i16_q16(q), -4, "negative fraction must floor, not truncate");
    }

    #[test]
    fn div_q16_min_by_small_denominator_wraps() {
        // Q16_MIN << 16 already exceeds i32 range before the division even
        // happens; the final `as i32` cast wraps (truncates to low 32 bits)
        // rather than panicking.
        assert_eq!(div_q16(Q16_MIN, 1), 0, "overflow must wrap, not panic");
    }

    #[test]
    fn roundtrip_f32_q16() {
        let values = [0.0f32, 0.25, -0.5, 1.0, core::f32::consts::PI, -100.0, 32767.0];
        for v in values {
            let q = to_q16(v);
            let back = from_q16(q);
            let lsb = 1.0 / 65536.0_f32;
            assert!(
                (back - v).abs() <= lsb,
                "roundtrip failed for {v}: got {back}"
            );
        }
    }

    #[test]
    fn roundtrip_i16_q16() {
        for v in [-32768i16, -1, 0, 1, 100, 32767] {
            let q = from_i16_q16(v);
            let back = to_i16_q16(q);
            assert_eq!(back, v, "i16 roundtrip failed for {v}");
        }
    }

    #[test]
    fn q31_q16_shifts() {
        let q31_one = 0x7FFF_FFFFi32;
        let q16 = q31_to_q16(q31_one);
        let f = from_q16(q16);
        assert!(
            (f - 1.0).abs() < 1e-4,
            "Q31->Q16 should be near 1.0, got {f}"
        );
    }

    #[test]
    fn mul_q16_basic() {
        let a = to_q16(1.5);
        let b = to_q16(2.0);
        let result = from_q16(mul_q16(a, b));
        assert!((result - 3.0).abs() < 1e-4, "1.5 * 2.0 = {result}");
    }

    #[test]
    fn mul_q16_negative() {
        let a = to_q16(-2.5);
        let b = to_q16(4.0);
        let result = from_q16(mul_q16(a, b));
        assert!((result - (-10.0)).abs() < 1e-4, "-2.5 * 4.0 = {result}");
    }

    #[test]
    fn mul_n_q16_integer_scale() {
        let a = to_q16(3.5);
        let result = from_q16(mul_n_q16(a, 4));
        assert!((result - 14.0).abs() < 1e-4, "3.5 * 4 = {result}");
    }

    #[test]
    fn mul_f_q16_float_scale() {
        let a = to_q16(2.0);
        let result = from_q16(mul_f_q16(a, 1.5));
        assert!((result - 3.0).abs() < 1e-3, "2.0 * 1.5f = {result}");
    }

    #[test]
    fn div_q16_basic() {
        let result = from_q16(div_q16(to_q16(3.0), to_q16(2.0)));
        assert!((result - 1.5).abs() < 1e-4, "3.0 / 2.0 = {result}");
    }

    #[test]
    fn div_q16_zero_denominator() {
        let result = div_q16(to_q16(5.0), 0);
        assert_eq!(result, 0, "division by zero must return 0");
    }

    #[test]
    fn div_n_q16_integer_divisor() {
        let result = from_q16(div_n_q16(to_q16(9.0), 3));
        assert!((result - 3.0).abs() < 1e-4, "9.0 / 3 = {result}");
    }

    #[test]
    fn div_f_q16_float_divisor() {
        let result = from_q16(div_f_q16(to_q16(6.0), 2.0));
        assert!((result - 3.0).abs() < 1e-3, "6.0 / 2.0f = {result}");
    }

    #[test]
    fn qadd_q16_no_overflow() {
        assert_eq!(
            from_q16(qadd_q16(to_q16(1.0), to_q16(2.0))).round() as i32,
            3
        );
    }

    #[test]
    fn qadd_q16_saturates_at_max() {
        let result = qadd_q16(Q16_MAX, Q16_MAX);
        assert_eq!(result, Q16_MAX, "overflow must saturate at Q16_MAX");
    }

    #[test]
    fn qsub_q16_saturates_at_min() {
        let result = qsub_q16(Q16_MIN, Q16_MAX);
        assert_eq!(result, Q16_MIN, "underflow must saturate at Q16_MIN");
    }

    #[test]
    fn abs_q16_positive_unchanged() {
        let v = to_q16(5.75);
        assert_eq!(abs_q16(v), v);
    }

    #[test]
    fn abs_q16_negated() {
        let v = to_q16(-core::f32::consts::PI);
        let expected = to_q16(core::f32::consts::PI);
        assert_eq!(abs_q16(v), expected);
    }

    #[test]
    fn lerp_q16_midpoint() {
        let a = to_q16(0.0);
        let b = to_q16(1.0);
        let mid = from_q16(lerp_q16(a, b, 1, 2));
        assert!((mid - 0.5).abs() < 1e-4, "lerp mid = {mid}");
    }

    #[test]
    fn lerp_q16_endpoints() {
        let a = to_q16(10.0);
        let b = to_q16(20.0);
        assert_eq!(lerp_q16(a, b, 0, 10), a, "t=0 must return left endpoint");
        assert_eq!(
            lerp_q16(a, b, 10, 10),
            b,
            "t=denom must return right endpoint"
        );
    }

    #[test]
    fn lerp_q16_zero_denom_returns_left() {
        let a = to_q16(5.0);
        let b = to_q16(9.0);
        assert_eq!(lerp_q16(a, b, 0, 0), a, "zero denom must return left");
    }

    #[test]
    fn angle_to_q16_90_degrees() {
        let q = angle_to_q16(90.0);
        let rad = from_q16(q);
        assert!(
            (rad - core::f32::consts::FRAC_PI_2).abs() < 1e-4,
            "90 deg should be pi/2, got {rad}"
        );
    }

    #[test]
    fn angle_to_q16_360_degrees() {
        let q = angle_to_q16(360.0);
        let rad = from_q16(q);
        assert!(
            (rad - 2.0 * core::f32::consts::PI).abs() < 1e-4,
            "360 deg should be 2*pi, got {rad}"
        );
    }

    #[test]
    fn recip_q16_one() {
        let result = from_q16(recip_q16(FP_ONE));
        assert!(
            (result - 1.0).abs() < 0.01,
            "recip(1.0) approx 1.0, got {result}"
        );
    }

    #[test]
    fn recip_q16_two() {
        let result = from_q16(recip_q16(to_q16(2.0)));
        assert!(
            (result - 0.5).abs() < 0.01,
            "recip(2.0) approx 0.5, got {result}"
        );
    }

    #[test]
    fn recip_q16_zero_returns_sentinel() {
        assert_eq!(
            recip_q16(0),
            Q16_MAX,
            "recip(0) must return Q16_MAX sentinel"
        );
    }

    #[test]
    fn scanline_interp_starts_at_left() {
        let interp = ScanlineInterp::new(100, 200, 0, 65536, 0, 32768, 10);
        assert_eq!(interp.z(), 100);
        assert_eq!(interp.u(), 0);
        assert_eq!(interp.v(), 0);
    }

    #[test]
    fn scanline_interp_step_reaches_right() {
        let mut interp = ScanlineInterp::depth_only(0, 65536, 10);
        for _ in 0..10 {
            interp.step();
        }
        let diff = (interp.z() as i64 - 65536i64).abs();
        assert!(
            diff <= 10,
            "z after 10 steps should be within 10 of 65536, got {} (diff={})",
            interp.z(),
            diff
        );
    }

    #[test]
    fn scanline_interp_uv_reaches_right() {
        let mut interp = ScanlineInterp::new(0, 0, 0, 65536, 0, 65536, 8);
        for _ in 0..8 {
            interp.step();
        }
        let u_err = (interp.u() as i64 - 65536i64).abs();
        let v_err = (interp.v() as i64 - 65536i64).abs();
        assert!(
            u_err <= 1,
            "u after 8 steps should be 65536 +/- 1, got {}",
            interp.u()
        );
        assert!(
            v_err <= 1,
            "v after 8 steps should be 65536 +/- 1, got {}",
            interp.v()
        );
    }

    #[test]
    fn scanline_interp_zero_span() {
        let mut interp = ScanlineInterp::new(1000, 9999, 0, 65536, 0, 65536, 0);
        interp.step();
        interp.step();
        assert_eq!(interp.z(), 1000, "zero-span z must not move");
        assert_eq!(interp.u(), 0, "zero-span u must not move");
    }

    #[test]
    fn scanline_interp_step_n() {
        let mut interp = ScanlineInterp::depth_only(0, 100000, 10);
        interp.step_n(5);
        let expected = 50000u32;
        let diff = (interp.z() as i64 - expected as i64).abs();
        assert!(
            diff <= 1,
            "step_n(5) should land at 50000 +/- 1, got {}",
            interp.z()
        );
    }

    #[test]
    fn scanline_interp_z_f32() {
        let interp = ScanlineInterp::depth_only(65536, 65536, 0);
        assert!((interp.z_f32() - 1.0).abs() < 1e-5, "z_f32 should be 1.0");
    }
}
