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

/// Seven-term Maclaurin series for `atan`, accurate to ~3e-6 for `|x| <= 0.5`.
///
/// The next omitted term is `x^15/15`, i.e. `2.0e-6` at `x = 0.5`; that is why callers fold
/// `x >= 0.5` down to `|z| <= 1/3` before calling in.
const fn atan_series(x: f32) -> f32 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    let x11 = x9 * x2;
    let x13 = x11 * x2;
    x - x3 / 3.0 + x5 / 5.0 - x7 / 7.0 + x9 / 9.0 - x11 / 11.0 + x13 / 13.0
}

/// `atan(x)` for `x` in `0.0..=1.0`.
///
/// The Maclaurin series alone only converges well for small arguments -- at `x = 1` it returns
/// 0.835 instead of `pi/4` -- so arguments of 0.5 or more are folded with
/// `atan(x) = pi/4 + atan((x - 1) / (x + 1))`, which maps them into `[-1/3, 0]` where the series
/// is at its most accurate. The fold starts at 0.5 (not above it) so the CORDIC table's
/// `atan(2^-1)` entry takes the folded branch too: the raw series needs ~13 terms to reach one
/// Q31 LSB at `x = 0.5`.
const fn atan_taylor(x: f32) -> f32 {
    if x >= 0.5 {
        core::f32::consts::FRAC_PI_4 + atan_series((x - 1.0) / (x + 1.0))
    } else {
        atan_series(x)
    }
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
    // The result is scale-invariant, so normalise the larger component into `[2^28, 2^29)`.
    // Scaling *down* is what makes this safe for large inputs: the CORDIC update `x ±= y` runs
    // from the normalised magnitude and the rotation grows the vector by the CORDIC gain
    // (~1.6468), so `sqrt(2) * 2^29 * 1.6468 < 2^31` leaves headroom. The previous code only
    // scaled *up* (towards `2^30`), so any input at or above that magnitude made the first
    // update saturate `i32`, corrupting the angle by up to ~10 degrees.
    while x < (1 << 28) && y < (1 << 28) {
        x <<= 1;
        y <<= 1;
    }
    while x >= (1 << 29) || y >= (1 << 29) {
        x >>= 1;
        y >>= 1;
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

/// Fast polynomial rational approximation for hyperbolic tangent `tanh(x)`.
pub fn fast_tanh_f32(x: f32) -> f32 {
    if x < -3.0 {
        -1.0
    } else if x > 3.0 {
        1.0
    } else {
        let x2 = x * x;
        x * (27.0 + x2) / (27.0 + 9.0 * x2)
    }
}

/// Fast 5th-order minimax polynomial approximation for `exp(x)`.
pub fn fast_exp_f32(x: f32) -> f32 {
    if x < -10.0 {
        0.0
    } else if x > 10.0 {
        22026.465
    } else {
        // Pade / rational approximation for e^x
        let x_half = x * 0.5;
        let num = 12.0 + 6.0 * x_half + x_half * x_half;
        let den = 12.0 - 6.0 * x_half + x_half * x_half;
        let res_half = num / den;
        res_half * res_half
    }
}

// --- Fast Bit-Manipulation Logarithms & Exponentials (Mineiro fastapprox) ---

/// Fast bit-manipulation base-2 logarithm `log2(x)` using IEEE-754 mantissa extraction
/// and low-order rational approximation (Paul Mineiro fastapprox).
///
/// Runs in single-digit cycles on ARM Cortex-M/RISC-V hardware `f32` FPUs without software `libm` emulation.
#[inline]
pub fn fast_log2_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let vx = x.to_bits();
    let mx = f32::from_bits((vx & 0x007F_FFFF) | 0x3F00_0000);
    let y = (vx as f32) * 1.192_092_9e-7;
    y - 124.225_52 - 1.498_030_3 * mx - 1.725_88 / (0.352_088_72 + mx)
}

/// Fast bit-manipulation natural logarithm `ln(x)`.
#[inline]
pub fn fast_ln_f32(x: f32) -> f32 {
    core::f32::consts::LN_2 * fast_log2_f32(x)
}

/// Fast bit-manipulation common logarithm `log10(x)`.
#[inline]
pub fn fast_log10_f32(x: f32) -> f32 {
    core::f32::consts::LOG10_2 * fast_log2_f32(x)
}

/// Fast bit-manipulation base-2 exponential `2^p` with underflow/overflow saturation.
#[inline]
pub fn fast_pow2_f32(p: f32) -> f32 {
    if p < -126.0 {
        return 0.0;
    }
    if p > 127.0 {
        return f32::INFINITY;
    }
    let offset = if p < 0.0 { 1.0f32 } else { 0.0f32 };
    let clipp = p;
    let w = clipp as i32;
    let z = clipp - (w as f32) + offset;
    let scaled = (1u32 << 23) as f32
        * (clipp + 121.274_055 + 27.728_024 / (4.842_525_5 - z) - 1.490_129_1 * z);
    f32::from_bits(scaled as u32)
}

/// Fast bit-manipulation base-10 exponential `10^p`.
#[inline]
pub fn fast_pow10_f32(p: f32) -> f32 {
    fast_pow2_f32(core::f32::consts::LOG2_10 * p)
}

/// Fast linear gain to decibels conversion: `20 * log10(gain)`.
#[inline]
pub fn fast_gain_to_db_f32(gain: f32) -> f32 {
    20.0 * fast_log10_f32(gain)
}

/// Fast decibels to linear gain conversion: `10^(db / 20)`.
#[inline]
pub fn fast_db_to_gain_f32(db: f32) -> f32 {
    fast_pow10_f32(db * 0.05)
}

// ─────────────────────────────────────────────────────────────────────────────
// High-Efficiency Fixed-Point Trigonometry & Phase Tracking (cossin & atan2)
// ─────────────────────────────────────────────────────────────────────────────

/// Depth of the cosine/sine midpoint lookup table (128 entries = 512 bytes).
pub const COSSIN_DEPTH: usize = 7;

/// Midpoint lookup table covering `[0, π/4)` with 7-bit depth.
pub const COSSIN: [u32; 128] = [
    0x00c9fffd, 0x025bfff8, 0x03edffef, 0x057fffe0,
    0x0711ffcc, 0x08a3ffb3, 0x0a35ff96, 0x0bc7ff73,
    0x0d58ff4c, 0x0eeaff1f, 0x107bfeee, 0x120dfeb8,
    0x139efe7d, 0x152efe3d, 0x16bffdf8, 0x184ffdae,
    0x19e0fd5f, 0x1b70fd0b, 0x1cfffcb2, 0x1e8ffc55,
    0x201efbf2, 0x21acfb8b, 0x233bfb1f, 0x24c9faae,
    0x2657fa38, 0x27e4f9bd, 0x2971f93d, 0x2afef8b8,
    0x2c8af82f, 0x2e16f7a1, 0x2fa1f70d, 0x312cf675,
    0x32b6f5d8, 0x3440f537, 0x35caf490, 0x3753f3e5,
    0x38dbf335, 0x3a63f280, 0x3beaf1c6, 0x3d71f107,
    0x3ef7f044, 0x407cef7b, 0x4201eeaf, 0x4385eddd,
    0x4509ed06, 0x468cec2b, 0x480eeb4b, 0x498fea66,
    0x4b10e97d, 0x4c90e88f, 0x4e10e79c, 0x4f8ee6a4,
    0x510ce5a8, 0x5289e4a7, 0x5405e3a1, 0x5581e297,
    0x56fbe188, 0x5875e075, 0x59eedf5c, 0x5b66de40,
    0x5cdddd1e, 0x5e53dbf8, 0x5fc9dacd, 0x613dd99e,
    0x62b1d86a, 0x6423d732, 0x6595d5f5, 0x6706d4b4,
    0x6875d36e, 0x69e4d224, 0x6b51d0d5, 0x6cbecf81,
    0x6e29ce29, 0x6f94cccd, 0x70fdcb6c, 0x7266ca07,
    0x73cdc89e, 0x7533c730, 0x7698c5bd, 0x77fcc446,
    0x795ec2cb, 0x7ac0c14c, 0x7c20bfc8, 0x7d7fbe40,
    0x7eddbcb4, 0x803abb23, 0x8195b98e, 0x82efb7f5,
    0x8448b657, 0x85a0b4b6, 0x86f6b310, 0x884bb166,
    0x899fafb7, 0x8af1ae05, 0x8c42ac4e, 0x8d92aa94,
    0x8ee0a8d5, 0x902da712, 0x9179a54b, 0x92c3a380,
    0x940ca1b1, 0x95539fde, 0x96999e07, 0x97dd9c2b,
    0x99209a4c, 0x9a629869, 0x9ba29682, 0x9ce19497,
    0x9e1e92a9, 0x9f5990b6, 0xa0938ebf, 0xa1cb8cc5,
    0xa3028ac7, 0xa43788c5, 0xa56b86bf, 0xa69d84b6,
    0xa7ce82a8, 0xa8fd8097, 0xaa2a7e82, 0xab557c6a,
    0xac7f7a4e, 0xada8782e, 0xaece760b, 0xaff373e4,
    0xb11671b9, 0xb2386f8b, 0xb3586d5a, 0xb4766b24,
];

/// Compute cosine and sine simultaneously from a 32-bit phase input.
///
/// Uses a compact 128-entry (512-byte) midpoint LUT, octant folding/unfolding,
/// and 1st-order linear interpolation.
///
/// # Arguments
/// * `phase` - 32-bit signed phase where `i32::MIN` represents `-π` and `i32::MAX` represents `+π`.
///
/// # Returns
/// `(cos, sin)` in signed 32-bit full scale (`i32::MIN` to `i32::MAX`).
/// Achieves 22-bit accuracy (-120.4 dBc spur suppression) in ~24 cycles on Cortex-M7.
pub fn cossin(mut phase: i32) -> (i32, i32) {
    let mut octant = phase as u32;
    if octant & (1 << 29) != 0 {
        phase = !phase;
    }

    const ALIGN_MSB: usize = 32 - 16 - 1;
    phase = (((phase as u32) << 3) >> (32 - COSSIN_DEPTH - ALIGN_MSB)) as _;

    let lookup = COSSIN[(phase >> ALIGN_MSB) as usize];
    phase &= (1 << ALIGN_MSB) - 1;
    phase -= 1 << (ALIGN_MSB - 1);

    const PI4: i32 = (core::f64::consts::FRAC_PI_4 * (1 << 16) as f64) as _;
    let dphi = (phase * PI4) >> 16;

    let mut cos = lookup as u16 as i32 + (1 << 16);
    let mut sin = (lookup >> 16) as i32;

    let dcos = (sin * dphi) >> COSSIN_DEPTH;
    let dsin = (cos * dphi) >> (COSSIN_DEPTH + 1);

    cos = (cos << (ALIGN_MSB - 1)) - dcos;
    sin = (sin << ALIGN_MSB) + dsin;

    octant ^= octant >> 1;
    if octant & (1 << 29) != 0 {
        core::mem::swap(&mut cos, &mut sin);
    }
    if octant & (1 << 30) != 0 {
        cos = -cos;
    }
    if octant & (1 << 31) != 0 {
        sin = -sin;
    }
    (cos, sin)
}

/// Floating-point wrapper for [`cossin`].
///
/// # Arguments
/// * `rad` - Angle in radians in `[-π, π]`.
///
/// # Returns
/// `(cos, sin)` normalized in `[-1.0, 1.0]`.
#[inline]
pub fn cossin_f32(mut rad: f32) -> (f32, f32) {
    const TAU: f32 = core::f32::consts::TAU;
    const PI: f32 = core::f32::consts::PI;
    rad %= TAU;
    if rad > PI {
        rad -= TAU;
    } else if rad < -PI {
        rad += TAU;
    }
    let phase = (rad * (2147483648.0 / PI)).clamp(i32::MIN as f32, i32::MAX as f32) as i32;
    let (c, s) = cossin(phase);
    (c as f32 * (1.0 / 2147483648.0), s as f32 * (1.0 / 2147483648.0))
}

/// Depth of the reciprocal lookup table for `atan2`.
pub const ATAN2_DIVI_DEPTH: usize = 4;

/// Reciprocal table `(base, slope)` for reciprocal seed interpolation.
pub const ATAN2_DIVI_RECIP: [(u32, i32); 16] = [
    (0x80000000, -126322568),
    (0x78787878, -112286727),
    (0x71c71c72, -100467071),
    (0x6bca1af3, -90420364),
    (0x66666666, -81808901),
    (0x61861862, -74371728),
    (0x5d1745d1, -67904621),
    (0x590b2164, -62245903),
    (0x55555555, -57266231),
    (0x51eb851f, -52861136),
    (0x4ec4ec4f, -48945496),
    (0x4bda12f7, -45449389),
    (0x49249249, -42314949),
    (0x469ee584, -39493952),
    (0x44444444, -36945955),
    (0x42108421, -34636833),
];

#[inline(always)]
fn mul_q31(x: u32, y: u32) -> u32 {
    ((x as u64 * y as u64) >> 31) as u32
}

#[inline(always)]
fn divi(y: u32, x: u32) -> u32 {
    if x == 0 {
        return 0;
    }
    let shift = x.leading_zeros();
    let y = y << shift;
    let x = x << shift;
    const FRAC_BITS: u32 = 31 - ATAN2_DIVI_DEPTH as u32;
    let rem = x & ((1 << FRAC_BITS) - 1);
    let idx = ((x << 1) >> (1 + FRAC_BITS)) as usize;
    let (base, slope) = ATAN2_DIVI_RECIP[idx];
    let step = ((slope as i64 * rem as i64) >> FRAC_BITS) as u32;
    let r0 = base.wrapping_add(step);
    mul_q31(y, mul_q31(r0, mul_q31(x, r0).wrapping_neg()))
}

fn atani(x: u32) -> u32 {
    const ATANI: [i32; 6] = [
        0x0517c2cd,
        -0x06c6496b,
        0x0fbdb021,
        -0x25b32e0a,
        0x43b34c81,
        -0x3bc823dd,
    ];
    let x2 = ((x as i64 * x as i64) >> 32) as i32;
    let mut r: i64 = 0;
    for &a in ATANI.iter().rev() {
        r = ((r * x2 as i64) >> 32) + a as i64;
    }
    ((r * (x as i64)) >> 28) as u32
}

/// 2-argument arctangent `atan2(y, x)` in integer arithmetic.
///
/// # Arguments
/// * `y` - Quadrature component.
/// * `x` - In-phase component.
///
/// # Returns
/// 32-bit phase angle where `i32::MIN` is `-π` and `i32::MAX` is `+π`.
/// Runs in ~52 Cortex-M7 cycles with 1.3 µrad RMS accuracy.
pub fn atan2_i32(mut y: i32, mut x: i32) -> i32 {
    let mut k = 0u32;
    if y < 0 {
        y = y.saturating_neg();
        k ^= u32::MAX;
    }
    if x < 0 {
        x = x.saturating_neg();
        k ^= u32::MAX >> 1;
    }
    if y > x {
        core::mem::swap(&mut y, &mut x);
        k ^= u32::MAX >> 2;
    }
    let r = atani(divi(y as u32, x as u32));
    (r ^ k) as i32
}

/// Fast floating-point arctangent `atan2(y, x)` returning radians in `[-π, π]`.
#[inline]
pub fn fast_atan2_f32(y: f32, x: f32) -> f32 {
    let max_abs = y.abs().max(x.abs());
    if max_abs == 0.0 {
        return 0.0;
    }
    let scale = 2147483647.0 / max_abs;
    let yi = (y * scale) as i32;
    let xi = (x * scale) as i32;
    let phase = atan2_i32(yi, xi);
    phase as f32 * (core::f32::consts::PI / 2147483648.0)
}

/// Continuous phase tracker and phase unwrapper.
///
/// Accumulates cycle turns whenever phase wraps across the ±π boundary,
/// producing a continuous 64-bit unwrapped phase trajectory without discontinuous jumps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Unwrapper {
    last_phase: i32,
    turns: i32,
}

impl Unwrapper {
    /// Create a new phase unwrapper initialized at zero.
    pub const fn new() -> Self {
        Self {
            last_phase: 0,
            turns: 0,
        }
    }

    /// Reset turn counter and history.
    pub fn reset(&mut self) {
        self.last_phase = 0;
        self.turns = 0;
    }

    /// Update with a wrapped 32-bit phase (`i32::MIN` to `i32::MAX`) and return unwrapped 64-bit phase.
    #[inline]
    pub fn update(&mut self, phase: i32) -> i64 {
        let diff = phase.wrapping_sub(self.last_phase);
        if self.last_phase > 0 && phase < 0 && diff > 0 {
            self.turns += 1;
        } else if self.last_phase < 0 && phase > 0 && diff < 0 {
            self.turns -= 1;
        }
        self.last_phase = phase;
        ((self.turns as i64) << 32) | (phase as u32 as i64)
    }

    /// Get current total number of 2π turns.
    #[inline(always)]
    pub const fn turns(&self) -> i32 {
        self.turns
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Integer phase-unwrap utilities (ported from idsp)
// ─────────────────────────────────────────────────────────────────────────────

/// Subtract `y - x` with wrapping, returning `(delta, wrap_sign)` where
/// `wrap_sign` is `+1` (positive overflow), `-1` (negative overflow), or `0` (none).
///
/// Faster than `i32::overflowing_sub` for embedded targets because no branch is needed.
#[inline]
pub fn overflowing_sub_i32(y: i32, x: i32) -> (i32, i8) {
    let delta = y.wrapping_sub(x);
    // If (delta >= 0) XOR (y >= x) then an overflow occurred.
    // wrap_sign = sign of overflow direction.
    let wrap = (delta >= 0) as i8 - (y >= x) as i8;
    (delta, wrap)
}

/// Combine `hi` (MSB) and `lo` (LSB) i32 words into one i32, saturating on overflow.
///
/// `lo` is right-shifted by `shift` bits, `hi` is left-shifted by `32 - shift`.
/// Valid range: `1 <= shift <= 32`.
#[inline]
pub fn saturating_scale_i32(lo: i32, hi: i32, shift: u32) -> i32 {
    debug_assert!(shift > 0 && shift <= 32, "shift must be in 1..=32");
    let hi_range: i32 = i32::MIN >> (shift - 1); // -(1 << (shift-1))
    if hi <= hi_range {
        i32::MIN.wrapping_sub(hi_range)
    } else if hi >= -hi_range {
        i32::MAX.wrapping_add(hi_range.wrapping_add(1))
    } else {
        (lo >> shift).wrapping_add(hi << (32 - shift))
    }
}

/// Stateful integer phase unwrapper.
///
/// Tracks wrapping phase (e.g. from an NCO encoded as i32) and provides the
/// signed increment between successive samples without any floating-point
/// conversion.
///
/// Useful for optical encoders, PLLs, and frequency-discriminator circuits.
///
/// # Example
///
/// ```rust
/// # use embedded_dsp::fast_math::IntPhaseUnwrapper;
/// let mut u = IntPhaseUnwrapper::new();
/// // Simulate an NCO that wraps from i32::MAX -> i32::MIN
/// let dx = u.process(i32::MIN);
/// // delta should be i32::MIN (wrapping add of large positive step)
/// assert_eq!(dx, i32::MIN);
/// ```
#[derive(Copy, Clone, Debug, Default)]
pub struct IntPhaseUnwrapper {
    /// Last accumulated phase value.
    pub y: i32,
}

impl IntPhaseUnwrapper {
    /// Create a new unwrapper starting from zero.
    pub const fn new() -> Self { Self { y: 0 } }

    /// Feed the next wrapped phase sample.
    ///
    /// Returns the signed phase increment `x - x_prev` (wrapping),
    /// and accumulates it into `self.y`.
    #[inline]
    pub fn process(&mut self, x: i32) -> i32 {
        let dx = x.wrapping_sub(self.y);
        self.y = self.y.wrapping_add(dx);
        dx
    }

    /// Current accumulated phase.
    #[inline]
    pub fn phase(&self) -> i32 { self.y }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    /// `atan_taylor` and the CORDIC angle table are `const fn`s evaluated at build time, so they
    /// need an explicit runtime call to register coverage. Also poke the two private helpers that
    /// the public `atan2_i32` path only reaches through one branch.
    #[test]
    fn compile_time_atan_helpers_and_private_divi() {
        let atan_t: fn(f32) -> f32 = core::hint::black_box(atan_taylor);
        let gen_atan: fn() -> [i32; 32] = core::hint::black_box(gen_cordic_atan_q31);
        let first: fn(i32, i32) -> i32 = core::hint::black_box(cordic_atan_first_q31);
        let div: fn(u32, u32) -> u32 = core::hint::black_box(divi);
        let pi = core::f32::consts::PI;

        // The folded series stays accurate all the way to atan(1) = pi/4.
        for &v in &[0.0f32, 0.125, 0.25, 0.5, 0.75, 1.0] {
            let x = core::hint::black_box(v);
            assert!((atan_t(x) - x.atan()).abs() < 2e-4, "atan_taylor({x})");
        }
        assert!((atan_t(1.0) - core::f32::consts::FRAC_PI_4).abs() < 1e-6);

        // Each entry is atan(2^-i)/pi in Q1.31: positive and non-increasing. Beyond i ~= 29 the
        // angle falls below the Q31 resolution, so the tail is exactly zero.
        let table = gen_atan();
        // atan(2^0) = pi/4, i.e. exactly 0.25 in Q1.31. This used to come out ~6% high because
        // the unfurled Maclaurin series was evaluated at x = 1.
        assert_eq!(table[0], 0x2000_0000);
        for i in 0..31 {
            assert!(table[i] >= table[i + 1], "atan table increased at {i}");
        }
        for i in 0..29 {
            assert!(table[i] > table[i + 1], "atan table not decreasing at {i}");
        }
        for i in 0..30 {
            assert!(table[i] > 0, "atan table non-positive at {i}");
        }
        // Every entry matches the true `atan(2^-i)/pi` to within a few Q31 LSBs. The bound is
        // generous only for the `f32` rounding in the truncated series; the truncation itself
        // is now well below one LSB.
        let mut worst = 0.0f32;
        let mut worst_i = 0;
        for i in 1..29 {
            let expect = (2.0f32.powi(-(i as i32))).atan() / pi * 2147483648.0;
            let err = ((table[i] as f32) - expect).abs();
            if err > worst {
                worst = err;
                worst_i = i;
            }
            assert!(
                err < 4.0,
                "table[{i}] = {} vs {expect} (off by {err} Q31 units)",
                table[i]
            );
        }
        assert!(worst < 4.0, "worst table error {worst} at i={worst_i}");

        assert_eq!(first(0, 0), 0);
        assert!(first(1 << 30, 1 << 28) > 0);
        assert_eq!(div(0x4000_0000, 0), 0); // divide-by-zero guard
        let _ = div(0x2000_0000, 0x4000_0000);
    }

    /// `atan2_from_xy_q31` normalises the vector before the CORDIC rotation. The former pre-scale
    /// only scaled *up* (towards `2^30`), so a component at or above that magnitude saturated
    /// `i32` on the first update and the angle could be off by ~10 degrees.
    #[test]
    fn atan2_from_xy_q31_is_accurate_across_the_full_input_range() {
        let vals = [
            0i32,
            1,
            -1,
            2,
            -2,
            17,
            -17,
            256,
            -256,
            1 << 16,
            -(1 << 16),
            1 << 24,
            -(1 << 24),
            1 << 29,
            -(1 << 29),
            i32::MAX,
            i32::MIN,
            i32::MIN + 1,
        ];

        let mut worst = 0.0f64;
        let mut worst_at = (0i32, 0i32);
        for &y in &vals {
            for &x in &vals {
                let got = atan2_from_xy_q31(y, x) as f64 / 2147483648.0;
                let want = (y as f64).atan2(x as f64) / core::f64::consts::PI;
                if (got - want).abs() > worst {
                    worst = (got - want).abs();
                    worst_at = (y, x);
                }
            }
        }

        assert!(
            worst < 1e-6,
            "worst atan2_from_xy_q31 error {worst} (normalized to pi) at (y={}, x={})",
            worst_at.0,
            worst_at.1
        );
    }
}
