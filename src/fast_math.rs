//! Fast math functions (sin, cos, sin_cos, sqrt, vsqrt, divide, log, exp, atan2).

#[allow(unused_imports)]
use crate::math::FloatMath;
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

/// Q31 sine and cosine calculation.
pub fn sin_cos_q31(theta: q31, sin_val: &mut q31, cos_val: &mut q31) {
    let theta_f = (theta as f64) / 2147483648.0 * core::f64::consts::PI;
    let s = theta_f.sin();
    let c = theta_f.cos();

    *sin_val = (s * 2147483647.0).clamp(-2147483648.0, 2147483647.0) as q31;
    *cos_val = (c * 2147483647.0).clamp(-2147483648.0, 2147483647.0) as q31;
}

/// Q31 sine function.
pub fn sin_q31(x: q31) -> q31 {
    let mut s = 0;
    let mut c = 0;
    sin_cos_q31(x, &mut s, &mut c);
    s
}

/// Q31 cosine function.
pub fn cos_q31(x: q31) -> q31 {
    let mut s = 0;
    let mut c = 0;
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

/// Q31 square root function.
pub fn sqrt_q31(in_val: q31, out_val: &mut q31) -> Status {
    if in_val < 0 {
        *out_val = 0;
        Status::ArgumentError
    } else {
        let f = in_val as f64 / 2147483648.0;
        let res = f.sqrt();
        *out_val = (res * 2147483647.0).clamp(0.0, 2147483647.0) as q31;
        Status::Success
    }
}

/// Q15 square root function.
pub fn sqrt_q15(in_val: q15, out_val: &mut q15) -> Status {
    if in_val < 0 {
        *out_val = 0;
        Status::ArgumentError
    } else {
        let f = in_val as f32 / 32768.0;
        let res = f.sqrt();
        *out_val = (res * 32767.0).clamp(0.0, 32767.0) as q15;
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
    if denominator == 0 {
        return Status::ArgumentError;
    }
    let n = numerator as i64;
    let d = denominator as i64;

    let res = (n << 31) / d;
    if res > i32::MAX as i64 || res < i32::MIN as i64 {
        *shift = 0;
        *quotient = res.clamp(i32::MIN as i64, i32::MAX as i64) as q31;
    } else {
        *shift = 0;
        *quotient = res as q31;
    }
    Status::Success
}

/// Fixed-point division for Q15 types (numerator / denominator).
pub fn divide_q15(numerator: q15, denominator: q15, quotient: &mut q15, shift: &mut i16) -> Status {
    if denominator == 0 {
        return Status::ArgumentError;
    }
    let n = numerator as i32;
    let d = denominator as i32;

    let res = (n << 15) / d;
    *shift = 0;
    *quotient = res.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
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

/// Q31 arc-tangent 2.
pub fn atan2_q31(y: q31, x: q31, res: &mut q31) -> Status {
    let y_f = y as f64 / 2147483648.0;
    let x_f = x as f64 / 2147483648.0;
    let ang = y_f.atan2(x_f) / core::f64::consts::PI;

    *res = (ang * 2147483647.0).clamp(-2147483648.0, 2147483647.0) as q31;
    Status::Success
}

/// Q15 arc-tangent 2.
pub fn atan2_q15(y: q15, x: q15, res: &mut q15) -> Status {
    let y_f = y as f32 / 32768.0;
    let x_f = x as f32 / 32768.0;
    let ang = y_f.atan2(x_f) / core::f32::consts::PI;
    *res = (ang * 32767.0).clamp(-32768.0, 32767.0) as q15;
    Status::Success
}
