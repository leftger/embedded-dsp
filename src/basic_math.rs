//! Basic math operations (absolute value, addition, subtraction, multiplication, dot product, negation, offset, scale, shift, clip, logic ops).

use crate::types::*;

// --- Absolute Value ---

pub fn abs_f32(src: &[f32], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].abs();
    }
}

pub fn abs_f64(src: &[f64], dst: &mut [f64]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].abs();
    }
}

pub fn abs_q31(src: &[q31], dst: &mut [q31]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].saturating_abs();
    }
}

pub fn abs_q15(src: &[q15], dst: &mut [q15]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].saturating_abs();
    }
}

pub fn abs_q7(src: &[q7], dst: &mut [q7]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].saturating_abs();
    }
}

// --- Vector Addition ---

pub fn add_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] + src_b[i];
    }
}

pub fn add_f64(src_a: &[f64], src_b: &[f64], dst: &mut [f64]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] + src_b[i];
    }
}

pub fn add_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i].saturating_add(src_b[i]);
    }
}

pub fn add_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    crate::intrinsics::simd_add_q15(src_a, src_b, dst);
}

pub fn add_q7(src_a: &[q7], src_b: &[q7], dst: &mut [q7]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i].saturating_add(src_b[i]);
    }
}

// --- Vector Subtraction ---

pub fn sub_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] - src_b[i];
    }
}

pub fn sub_f64(src_a: &[f64], src_b: &[f64], dst: &mut [f64]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] - src_b[i];
    }
}

pub fn sub_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i].saturating_sub(src_b[i]);
    }
}

pub fn sub_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    crate::intrinsics::simd_sub_q15(src_a, src_b, dst);
}

pub fn sub_q7(src_a: &[q7], src_b: &[q7], dst: &mut [q7]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i].saturating_sub(src_b[i]);
    }
}

// --- Vector Multiplication ---

pub fn mult_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] * src_b[i];
    }
}

pub fn mult_f64(src_a: &[f64], src_b: &[f64], dst: &mut [f64]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] * src_b[i];
    }
}

pub fn mult_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = q31_mult(src_a[i], src_b[i]);
    }
}

pub fn mult_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    crate::intrinsics::simd_mult_q15(src_a, src_b, dst);
}

pub fn mult_q7(src_a: &[q7], src_b: &[q7], dst: &mut [q7]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = q7_mult(src_a[i], src_b[i]);
    }
}

// --- Negate ---

pub fn negate_f32(src: &[f32], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = -src[i];
    }
}

pub fn negate_f64(src: &[f64], dst: &mut [f64]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = -src[i];
    }
}

pub fn negate_q31(src: &[q31], dst: &mut [q31]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].saturating_neg();
    }
}

pub fn negate_q15(src: &[q15], dst: &mut [q15]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].saturating_neg();
    }
}

pub fn negate_q7(src: &[q7], dst: &mut [q7]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].saturating_neg();
    }
}

// --- Offset ---

pub fn offset_f32(src: &[f32], offset: f32, dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i] + offset;
    }
}

pub fn offset_f64(src: &[f64], offset: f64, dst: &mut [f64]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i] + offset;
    }
}

pub fn offset_q31(src: &[q31], offset: q31, dst: &mut [q31]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].saturating_add(offset);
    }
}

pub fn offset_q15(src: &[q15], offset: q15, dst: &mut [q15]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].saturating_add(offset);
    }
}

pub fn offset_q7(src: &[q7], offset: q7, dst: &mut [q7]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].saturating_add(offset);
    }
}

// --- Scale ---

pub fn scale_f32(src: &[f32], scale: f32, dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i] * scale;
    }
}

pub fn scale_f64(src: &[f64], scale: f64, dst: &mut [f64]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i] * scale;
    }
}

pub fn scale_q31(src: &[q31], scale_fract: q31, shift: i8, dst: &mut [q31]) {
    let len = src.len().min(dst.len());
    let k_shift = 31 - shift;
    for i in 0..len {
        let mult = (src[i].to_bits() as i64 * scale_fract.to_bits() as i64)
            >> if k_shift >= 0 { k_shift as u32 } else { 0 };
        let val = if k_shift < 0 {
            mult << (-k_shift as u32)
        } else {
            mult
        };
        dst[i] = q31::from_bits(val.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    }
}

pub fn scale_q15(src: &[q15], scale_fract: q15, shift: i8, dst: &mut [q15]) {
    let len = src.len().min(dst.len());
    let k_shift = 15 - shift;
    for i in 0..len {
        let mult = (src[i].to_bits() as i32 * scale_fract.to_bits() as i32)
            >> if k_shift >= 0 { k_shift as u32 } else { 0 };
        let val = if k_shift < 0 {
            mult << (-k_shift as u32)
        } else {
            mult
        };
        dst[i] = q15::from_bits(val.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    }
}

pub fn scale_q7(src: &[q7], scale_fract: q7, shift: i8, dst: &mut [q7]) {
    let len = src.len().min(dst.len());
    let k_shift = 7 - shift;
    for i in 0..len {
        let mult = (src[i].to_bits() as i32 * scale_fract.to_bits() as i32)
            >> if k_shift >= 0 { k_shift as u32 } else { 0 };
        let val = if k_shift < 0 {
            mult << (-k_shift as u32)
        } else {
            mult
        };
        dst[i] = q7::from_bits(val.clamp(i8::MIN as i32, i8::MAX as i32) as i8);
    }
}

// --- Shift ---

pub fn shift_q31(src: &[q31], shift_bits: i8, dst: &mut [q31]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        if shift_bits >= 0 {
            dst[i] = q31::from_bits(
                ((src[i].to_bits() as i64) << shift_bits).clamp(i32::MIN as i64, i32::MAX as i64)
                    as i32,
            );
        } else {
            dst[i] = q31::from_bits(src[i].to_bits() >> (-shift_bits));
        }
    }
}

pub fn shift_q15(src: &[q15], shift_bits: i8, dst: &mut [q15]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        if shift_bits >= 0 {
            dst[i] = q15::from_bits(
                ((src[i].to_bits() as i32) << shift_bits).clamp(i16::MIN as i32, i16::MAX as i32)
                    as i16,
            );
        } else {
            dst[i] = q15::from_bits(src[i].to_bits() >> (-shift_bits));
        }
    }
}

pub fn shift_q7(src: &[q7], shift_bits: i8, dst: &mut [q7]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        if shift_bits >= 0 {
            dst[i] = q7::from_bits(
                ((src[i].to_bits() as i32) << shift_bits).clamp(i8::MIN as i32, i8::MAX as i32)
                    as i8,
            );
        } else {
            dst[i] = q7::from_bits(src[i].to_bits() >> (-shift_bits));
        }
    }
}

// --- Dot Product ---

pub fn dot_prod_f32(src_a: &[f32], src_b: &[f32]) -> f32 {
    let len = src_a.len().min(src_b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += src_a[i] * src_b[i];
    }
    sum
}

/// Compensated dot product using Neumaier summation to accumulate products.
///
/// Mitigates loss-of-significance and cancellation errors when accumulating long
/// filter vectors or high dynamic range inputs using native `f32` operations.
pub fn dot_prod_f32_compensated(src_a: &[f32], src_b: &[f32]) -> f32 {
    let len = src_a.len().min(src_b.len());
    if len == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut c = 0.0f32;
    for i in 0..len {
        let p = src_a[i] * src_b[i];
        let t = sum + p;
        if sum.abs() >= p.abs() {
            c += (sum - t) + p;
        } else {
            c += (p - t) + sum;
        }
        sum = t;
    }
    sum + c
}

pub fn dot_prod_f64(src_a: &[f64], src_b: &[f64]) -> f64 {
    let len = src_a.len().min(src_b.len());
    let mut sum = 0.0f64;
    for i in 0..len {
        sum += src_a[i] * src_b[i];
    }
    sum
}

pub fn dot_prod_q31(src_a: &[q31], src_b: &[q31]) -> q63 {
    let len = src_a.len().min(src_b.len());
    let mut sum: q63 = 0;
    for i in 0..len {
        sum += (src_a[i].to_bits() as i64 * src_b[i].to_bits() as i64) >> 14;
    }
    sum
}

/// Exact full-precision Q31 dot product accumulating into a 128-bit wide integer
/// without intermediate truncation or bit loss (Quire-inspired exact accumulator).
pub fn dot_prod_q31_wide(src_a: &[q31], src_b: &[q31]) -> i128 {
    let len = src_a.len().min(src_b.len());
    let mut sum: i128 = 0;
    for i in 0..len {
        let a = src_a[i].to_bits() as i128;
        let b = src_b[i].to_bits() as i128;
        sum += a * b;
    }
    sum
}

/// Saturated Q31 dot product with exact 128-bit accumulation (Quire-inspired).
/// Accumulates products with zero rounding error and scales/saturates once at the end.
pub fn dot_prod_q31_saturated(src_a: &[q31], src_b: &[q31]) -> q31 {
    let acc = dot_prod_q31_wide(src_a, src_b);
    let scaled = acc >> 31;
    let clamped = scaled.clamp(i32::MIN as i128, i32::MAX as i128) as i32;
    q31::from_bits(clamped)
}

pub fn dot_prod_q15(src_a: &[q15], src_b: &[q15]) -> q63 {
    crate::intrinsics::simd_dot_prod_q15(src_a, src_b)
}

/// Exact full-precision Q15 dot product accumulating into a 64-bit wide integer
/// without intermediate bit loss.
pub fn dot_prod_q15_wide(src_a: &[q15], src_b: &[q15]) -> i64 {
    crate::intrinsics::simd_dot_prod_q15(src_a, src_b)
}

/// Saturated Q15 dot product with exact 64-bit accumulation.
/// Accumulates products with zero intermediate rounding error and scales/saturates once at the end.
pub fn dot_prod_q15_saturated(src_a: &[q15], src_b: &[q15]) -> q15 {
    let acc = dot_prod_q15_wide(src_a, src_b);
    let scaled = acc >> 15;
    let clamped = scaled.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
    q15::from_bits(clamped)
}

pub fn dot_prod_q7(src_a: &[q7], src_b: &[q7]) -> q31 {
    let len = src_a.len().min(src_b.len());
    let mut sum = q31::ZERO;
    for i in 0..len {
        sum += q31::from_bits(src_a[i].to_bits() as i32 * src_b[i].to_bits() as i32);
    }
    sum
}

// --- Strided Dot Products (Interleaved DMA Buffers) ---

/// Strided `f32` dot product allowing direct execution on interleaved audio (stereo) or sensor (3-axis IMU) DMA buffers.
pub fn dot_prod_f32_strided(
    src_a: &[f32],
    stride_a: usize,
    src_b: &[f32],
    stride_b: usize,
    count: usize,
) -> f32 {
    if stride_a == 0 || stride_b == 0 || count == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut idx_a = 0;
    let mut idx_b = 0;
    for _ in 0..count {
        if idx_a >= src_a.len() || idx_b >= src_b.len() {
            break;
        }
        sum += src_a[idx_a] * src_b[idx_b];
        idx_a += stride_a;
        idx_b += stride_b;
    }
    sum
}

/// Strided `f64` dot product.
pub fn dot_prod_f64_strided(
    src_a: &[f64],
    stride_a: usize,
    src_b: &[f64],
    stride_b: usize,
    count: usize,
) -> f64 {
    if stride_a == 0 || stride_b == 0 || count == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut idx_a = 0;
    let mut idx_b = 0;
    for _ in 0..count {
        if idx_a >= src_a.len() || idx_b >= src_b.len() {
            break;
        }
        sum += src_a[idx_a] * src_b[idx_b];
        idx_a += stride_a;
        idx_b += stride_b;
    }
    sum
}

/// Strided `q31` dot product.
pub fn dot_prod_q31_strided(
    src_a: &[q31],
    stride_a: usize,
    src_b: &[q31],
    stride_b: usize,
    count: usize,
) -> q63 {
    if stride_a == 0 || stride_b == 0 || count == 0 {
        return 0;
    }
    let mut sum: q63 = 0;
    let mut idx_a = 0;
    let mut idx_b = 0;
    for _ in 0..count {
        if idx_a >= src_a.len() || idx_b >= src_b.len() {
            break;
        }
        sum += (src_a[idx_a].to_bits() as i64 * src_b[idx_b].to_bits() as i64) >> 14;
        idx_a += stride_a;
        idx_b += stride_b;
    }
    sum
}

/// Strided `q15` dot product.
pub fn dot_prod_q15_strided(
    src_a: &[q15],
    stride_a: usize,
    src_b: &[q15],
    stride_b: usize,
    count: usize,
) -> q63 {
    if stride_a == 0 || stride_b == 0 || count == 0 {
        return 0;
    }
    let mut sum: q63 = 0;
    let mut idx_a = 0;
    let mut idx_b = 0;
    for _ in 0..count {
        if idx_a >= src_a.len() || idx_b >= src_b.len() {
            break;
        }
        sum += src_a[idx_a].to_bits() as i64 * src_b[idx_b].to_bits() as i64;
        idx_a += stride_a;
        idx_b += stride_b;
    }
    sum
}

// --- Error-Free Transformations (EFT) & Compensated Arithmetic ---

/// Knuth's TwoSum algorithm for f32: computes `s = fl(a + b)` and exact roundoff `r`
/// such that `a + b = s + r` holds exactly in real arithmetic without overflow.
#[inline]
pub fn two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let b_virtual = s - a;
    let a_virtual = s - b_virtual;
    let b_roundoff = b - b_virtual;
    let a_roundoff = a - a_virtual;
    let r = a_roundoff + b_roundoff;
    (s, r)
}

/// Knuth's TwoSum algorithm for f64: computes `s = fl(a + b)` and exact roundoff `r`
/// such that `a + b = s + r` holds exactly in real arithmetic without overflow.
#[inline]
pub fn two_sum_f64(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let b_virtual = s - a;
    let a_virtual = s - b_virtual;
    let b_roundoff = b - b_virtual;
    let a_roundoff = a - a_virtual;
    let r = a_roundoff + b_roundoff;
    (s, r)
}

/// Dekker's QuickTwoSum algorithm for f32: computes `s = fl(a + b)` and exact roundoff `r`.
///
/// # Precondition
/// Requires `|a| >= |b|` and finite operands.
#[inline]
pub fn quick_two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let r = b - (s - a);
    (s, r)
}

/// TwoDiff algorithm for f32: computes `d = fl(a - b)` and exact roundoff `r`
/// such that `a - b = d + r` holds exactly in real arithmetic without overflow.
#[inline]
pub fn two_diff_f32(a: f32, b: f32) -> (f32, f32) {
    let d = a - b;
    let b_virtual = a - d;
    let a_virtual = d + b_virtual;
    let b_roundoff = b_virtual - b;
    let a_roundoff = a - a_virtual;
    let r = a_roundoff + b_roundoff;
    (d, r)
}

/// TwoProd algorithm for f32 using Dekker splitting: computes `p = fl(a * b)` and exact error `r`
/// such that `a * b = p + r` holds in real arithmetic without overflow.
#[inline]
pub fn two_prod_f32(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    let c = 4097.0f32 * a;
    let a_hi = c - (c - a);
    let a_lo = a - a_hi;
    let c = 4097.0f32 * b;
    let b_hi = c - (c - b);
    let b_lo = b - b_hi;
    let r = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, r)
}

/// TwoProd algorithm for f64 using Dekker splitting: computes `p = fl(a * b)` and exact error `r`
/// such that `a * b = p + r` holds in real arithmetic without overflow.
#[inline]
pub fn two_prod_f64(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let c = 134217729.0f64 * a;
    let a_hi = c - (c - a);
    let a_lo = a - a_hi;
    let c = 134217729.0f64 * b;
    let b_hi = c - (c - b);
    let b_lo = b - b_hi;
    let r = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, r)
}

/// TwoDiv algorithm for f32: computes `q = fl(a / b)` and residual error `r`
/// such that `a / b = q + r`.
#[inline]
pub fn two_div_f32(a: f32, b: f32) -> (f32, f32) {
    let q = a / b;
    let (p, r_prod) = two_prod_f32(q, b);
    let err = (a - p) - r_prod;
    (q, err / b)
}

/// Compensated summation using Neumaier's algorithm.
///
/// Tracks round-off errors at every step, providing near-double precision accuracy
/// while executing purely on hardware single-precision (`f32`) FPUs without software `f64` emulation.
pub fn sum_f32_compensated(src: &[f32]) -> f32 {
    if src.is_empty() {
        return 0.0;
    }
    let mut sum = src[0];
    let mut c = 0.0f32;
    for &x in &src[1..] {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    sum + c
}

// --- Polynomial Evaluation & Root Finding ---

/// Evaluates an n-th degree polynomial `P(x) = coeffs[0] + coeffs[1]*x + ... + coeffs[n]*x^n`
/// using Horner's method.
pub fn poly_eval_f32(coeffs: &[f32], x: f32) -> f32 {
    if coeffs.is_empty() {
        return 0.0;
    }
    let n = coeffs.len() - 1;
    let mut r = coeffs[n];
    for &c in coeffs[..n].iter().rev() {
        r = r * x + c;
    }
    r
}

/// Evaluates an n-th degree polynomial using Horner's method in f64.
pub fn poly_eval_f64(coeffs: &[f64], x: f64) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }
    let n = coeffs.len() - 1;
    let mut r = coeffs[n];
    for &c in coeffs[..n].iter().rev() {
        r = r * x + c;
    }
    r
}

/// Evaluates a polynomial with Q15 coefficients and argument using saturating fixed-point arithmetic.
pub fn poly_eval_q15(coeffs: &[q15], x: q15) -> q15 {
    if coeffs.is_empty() {
        return q15::ZERO;
    }
    let n = coeffs.len() - 1;
    let mut r = coeffs[n];
    for &c in coeffs[..n].iter().rev() {
        r = r.saturating_mul(x).saturating_add(c);
    }
    r
}

/// Evaluates a polynomial with Q31 coefficients and argument using saturating fixed-point arithmetic.
pub fn poly_eval_q31(coeffs: &[q31], x: q31) -> q31 {
    if coeffs.is_empty() {
        return q31::ZERO;
    }
    let n = coeffs.len() - 1;
    let mut r = coeffs[n];
    for &c in coeffs[..n].iter().rev() {
        r = r.saturating_mul(x).saturating_add(c);
    }
    r
}

/// Simultaneously evaluates polynomial `P(x)` and its derivative `P'(x)`
/// using Shaw-Traub Horner recurrence in O(N) operations with zero heap allocation.
pub fn poly_eval_with_deriv_f32(coeffs: &[f32], x: f32) -> (f32, f32) {
    if coeffs.is_empty() {
        return (0.0, 0.0);
    }
    let n = coeffs.len() - 1;
    let mut p = coeffs[n];
    let mut d = 0.0f32;
    for &c in coeffs[..n].iter().rev() {
        d = d * x + p;
        p = p * x + c;
    }
    (p, d)
}

/// Finds a root of polynomial `P(x)` near initial guess `x0` using Newton-Raphson iteration
/// with analytical derivative evaluation. Zero dynamic allocations.
pub fn poly_root_f32(coeffs: &[f32], x0: f32, max_iter: usize, tol: f32) -> Result<f32, Status> {
    if coeffs.len() < 2 {
        return Err(Status::LengthError);
    }
    let mut x = x0;
    let tol_pos = if tol > 0.0 { tol } else { 1e-6 };
    for _ in 0..max_iter {
        let (p, d) = poly_eval_with_deriv_f32(coeffs, x);
        if p.abs() < tol_pos {
            return Ok(x);
        }
        if d.abs() < 1e-20 {
            return Err(Status::Singular);
        }
        let step = p / d;
        x -= step;
        if step.abs() < tol_pos {
            return Ok(x);
        }
    }
    let (p, _) = poly_eval_with_deriv_f32(coeffs, x);
    if p.abs() < tol_pos * 10.0 {
        Ok(x)
    } else {
        Err(Status::TestFailure)
    }
}

// --- Clip ---

pub fn clip_f32(src: &[f32], low: f32, high: f32, dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].clamp(low, high);
    }
}

pub fn clip_q31(src: &[q31], low: q31, high: q31, dst: &mut [q31]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].clamp(low, high);
    }
}

pub fn clip_q15(src: &[q15], low: q15, high: q15, dst: &mut [q15]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].clamp(low, high);
    }
}

pub fn clip_q7(src: &[q7], low: q7, high: q7, dst: &mut [q7]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].clamp(low, high);
    }
}

// --- Logic Operations ---

pub fn and_u32(src_a: &[u32], src_b: &[u32], dst: &mut [u32]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] & src_b[i];
    }
}

pub fn and_u16(src_a: &[u16], src_b: &[u16], dst: &mut [u16]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] & src_b[i];
    }
}

pub fn and_u8(src_a: &[u8], src_b: &[u8], dst: &mut [u8]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] & src_b[i];
    }
}

pub fn or_u32(src_a: &[u32], src_b: &[u32], dst: &mut [u32]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] | src_b[i];
    }
}

pub fn or_u16(src_a: &[u16], src_b: &[u16], dst: &mut [u16]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] | src_b[i];
    }
}

pub fn or_u8(src_a: &[u8], src_b: &[u8], dst: &mut [u8]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] | src_b[i];
    }
}

pub fn not_u32(src: &[u32], dst: &mut [u32]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = !src[i];
    }
}

pub fn not_u16(src: &[u16], dst: &mut [u16]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = !src[i];
    }
}

pub fn not_u8(src: &[u8], dst: &mut [u8]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = !src[i];
    }
}

pub fn xor_u32(src_a: &[u32], src_b: &[u32], dst: &mut [u32]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] ^ src_b[i];
    }
}

pub fn xor_u16(src_a: &[u16], src_b: &[u16], dst: &mut [u16]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] ^ src_b[i];
    }
}

pub fn xor_u8(src_a: &[u8], src_b: &[u8], dst: &mut [u8]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] ^ src_b[i];
    }
}
