//! Fast math functions (sin, cos, sin_cos, sqrt, vsqrt, divide, log, exp, atan2).

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::math::{isqrt_u32, isqrt_u64};
use crate::types::*;

/// Floating-point sine calculation.
pub fn sin_f32(x: f32) -> f32 {
    x.sin()
}

/// Floating-point cosine calculation.
pub fn cos_f32(x: f32) -> f32 {
    x.cos()
}

/// Floating-point sine and cosine calculation.
pub fn sin_cos_f32(theta: f32, sin_val: &mut f32, cos_val: &mut f32) {
    let rad = theta * (core::f32::consts::PI / 180.0);
    *sin_val = rad.sin();
    *cos_val = rad.cos();
}

const fn atan_taylor(x: f32) -> f32 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    x - x3 / 3.0 + x5 / 5.0 - x7 / 7.0 + x9 / 9.0
}

const fn gen_cordic_atan_q31() -> [i32; 32] {
    let mut t = [0i32; 32];
    let mut i = 0;
    while i < 32 {
        let x = 1.0f32 / ((1u32 << i) as f32);
        let a = atan_taylor(x) / core::f32::consts::PI * 2147483648.0;
        t[i] = a as i32;
        i += 1;
    }
    t
}

/// `atan(2^-i) / π` in Q1.31 (CORDIC angle table).
const CORDIC_ATAN_Q31: [i32; 32] = gen_cordic_atan_q31();

/// CORDIC K ≈ 0.607252935 in Q1.31.
const CORDIC_K_Q31: i32 = 1_304_063_564;

fn cordic_rotate_q31(theta: i32) -> (i32, i32) {
    let mut x = CORDIC_K_Q31;
    let mut y = 0i32;
    let mut z = theta;
    let mut i = 0;
    while i < 31 {
        let x_sh = x >> i;
        let y_sh = y >> i;
        if z >= 0 {
            x = x.saturating_sub(y_sh);
            y = y.saturating_add(x_sh);
            z = z.saturating_sub(CORDIC_ATAN_Q31[i]);
        } else {
            x = x.saturating_add(y_sh);
            y = y.saturating_sub(x_sh);
            z = z.saturating_add(CORDIC_ATAN_Q31[i]);
        }
        i += 1;
    }
    (x, y)
}

/// First-quadrant `atan(y/x) / π` in Q1.31. `x` and `y` must be `>= 0`.
fn cordic_atan_first_q31(mut x: i32, mut y: i32) -> i32 {
    if x == 0 {
        return if y == 0 { 0 } else { 1 << 30 }; // 0.5 → π/2
    }
    if y == 0 {
        return 0;
    }
    while x < (1 << 30) && y < (1 << 30) && (x > 0 || y > 0) {
        let nx = x.saturating_mul(2);
        let ny = y.saturating_mul(2);
        if nx / 2 != x || ny / 2 != y {
            break;
        }
        x = nx;
        y = ny;
    }
    let mut z = 0i32;
    let mut i = 0;
    while i < 31 {
        let x_sh = x >> i;
        let y_sh = y >> i;
        if y >= 0 {
            x = x.saturating_add(y_sh);
            y = y.saturating_sub(x_sh);
            z = z.saturating_add(CORDIC_ATAN_Q31[i]);
        } else {
            x = x.saturating_sub(y_sh);
            y = y.saturating_add(x_sh);
            z = z.saturating_sub(CORDIC_ATAN_Q31[i]);
        }
        i += 1;
    }
    z.max(0)
}

fn atan2_from_xy_q31(y: i32, x: i32) -> i32 {
    if x == 0 && y == 0 {
        return 0;
    }
    let ax = if x == i32::MIN { i32::MAX } else { x.abs() };
    let ay = if y == i32::MIN { i32::MAX } else { y.abs() };
    let a = cordic_atan_first_q31(ax, ay);
    match (x >= 0, y >= 0) {
        (true, true) => a,
        (true, false) => a.saturating_neg(),
        (false, true) => i32::MAX.saturating_sub(a),
        (false, false) => a.saturating_sub(i32::MAX),
    }
}

/// Q31 sine and cosine. `theta` is in CMSIS units: `[-1, 1) → [-π, π)`.
pub fn sin_cos_q31(theta: q31, sin_val: &mut q31, cos_val: &mut q31) {
    let (c, s) = cordic_rotate_q31(theta.to_bits());
    *cos_val = q31::from_bits(c);
    *sin_val = q31::from_bits(s);
}

/// Q31 sine function.
pub fn sin_q31(x: q31) -> q31 {
    let mut s = q31::ZERO;
    let mut c = q31::ZERO;
    sin_cos_q31(x, &mut s, &mut c);
    s
}

/// Q31 cosine function.
pub fn cos_q31(x: q31) -> q31 {
    let mut s = q31::ZERO;
    let mut c = q31::ZERO;
    sin_cos_q31(x, &mut s, &mut c);
    c
}

/// Floating-point square root function.
pub fn sqrt_f32(in_val: f32, out_val: &mut f32) -> Status {
    if in_val < 0.0 {
        *out_val = 0.0;
        Status::ArgumentError
    } else {
        *out_val = in_val.sqrt();
        Status::Success
    }
}

/// Q31 square root (`sqrt(x / 2^31) * 2^31`).
pub fn sqrt_q31(in_val: q31, out_val: &mut q31) -> Status {
    if in_val < q31::ZERO {
        *out_val = q31::ZERO;
        Status::ArgumentError
    } else {
        let n = (in_val.to_bits() as u64) << 31;
        *out_val = q31::from_bits(isqrt_u64(n).min(i32::MAX as u64) as i32);
        Status::Success
    }
}

/// Q15 square root (`sqrt(x / 2^15) * 2^15`).
pub fn sqrt_q15(in_val: q15, out_val: &mut q15) -> Status {
    if in_val < q15::ZERO {
        *out_val = q15::ZERO;
        Status::ArgumentError
    } else {
        let n = (in_val.to_bits() as u32) << 15;
        *out_val = q15::from_bits(isqrt_u32(n).min(i16::MAX as u32) as i16);
        Status::Success
    }
}

/// Vector square root function.
pub fn vsqrt_f32(src: &[f32], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        if src[i] < 0.0 {
            dst[i] = 0.0;
        } else {
            dst[i] = src[i].sqrt();
        }
    }
}

/// Fixed-point division for Q31 types (numerator / denominator).
pub fn divide_q31(numerator: q31, denominator: q31, quotient: &mut q31, shift: &mut i16) -> Status {
    if denominator == q31::ZERO {
        return Status::ArgumentError;
    }
    let n = numerator.to_bits() as i64;
    let d = denominator.to_bits() as i64;

    let res = (n << 31) / d;
    if res > i32::MAX as i64 || res < i32::MIN as i64 {
        *shift = 0;
        *quotient = q31::from_bits(res.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    } else {
        *shift = 0;
        *quotient = q31::from_bits(res as i32);
    }
    Status::Success
}

/// Fixed-point division for Q15 types (numerator / denominator).
pub fn divide_q15(numerator: q15, denominator: q15, quotient: &mut q15, shift: &mut i16) -> Status {
    if denominator == q15::ZERO {
        return Status::ArgumentError;
    }
    let n = numerator.to_bits() as i32;
    let d = denominator.to_bits() as i32;

    let res = (n << 15) / d;
    *shift = 0;
    *quotient = q15::from_bits(res.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    Status::Success
}

/// Floating-point natural logarithm.
pub fn log_f32(x: f32) -> f32 {
    x.ln()
}

/// Floating-point exponential.
pub fn exp_f32(x: f32) -> f32 {
    x.exp()
}

/// Floating-point arc-tangent 2.
pub fn atan2_f32(y: f32, x: f32, res: &mut f32) -> Status {
    *res = y.atan2(x);
    Status::Success
}

/// Q31 arc-tangent 2. Result is `atan2(y, x) / π` in Q1.31 (`[-1, 1)`).
pub fn atan2_q31(y: q31, x: q31, res: &mut q31) -> Status {
    *res = q31::from_bits(atan2_from_xy_q31(y.to_bits(), x.to_bits()));
    Status::Success
}

/// Q15 arc-tangent 2. Result is `atan2(y, x) / π` in Q1.15.
pub fn atan2_q15(y: q15, x: q15, res: &mut q15) -> Status {
    let z = atan2_from_xy_q31((y.to_bits() as i32) << 16, (x.to_bits() as i32) << 16);
    *res = q15::from_bits((z >> 16) as i16);
    Status::Success
}
