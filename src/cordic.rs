//! CORDIC (Coordinate Rotation Digital Computer) pure-integer arithmetic engine.
//!
//! Computes trigonometric functions (`sin`, `cos`, `atan2`), polar/Cartesian vector
//! rotations, and square roots using only additions, subtractions, and bit shifts.
//! Ideal for ultra-low-power microcontrollers without hardware multipliers or FPU.

use crate::types::{q15, q31};

/// CORDIC gain compensation factor in Q15 format: `0.607252935 * 32768 ≈ 19898`.
pub const CORDIC_K_Q15: q15 = 19898;

/// CORDIC gain compensation factor in Q31 format: `0.607252935 * 2^31 ≈ 1304065792`.
pub const CORDIC_K_Q31: q31 = 1304065792;

/// Number of CORDIC iterations.
pub const CORDIC_ITERATIONS: usize = 16;

/// Precomputed arctangent lookup table in Q15 radians (`atan(2^-i) * 32768`).
pub const ATAN_TABLE_Q15: [q15; 16] = [
    25736, // atan(1.0) = pi/4 ≈ 0.785398
    15193, // atan(0.5) ≈ 0.463648
    8027,  // atan(0.25) ≈ 0.244979
    4075,  // atan(0.125) ≈ 0.124355
    2045,  // atan(0.0625) ≈ 0.062419
    1024,  // atan(0.03125) ≈ 0.031240
    512,   // atan(0.015625) ≈ 0.015624
    256,   // atan(0.0078125) ≈ 0.007812
    128,   // atan(0.00390625)
    64,    // atan(0.001953125)
    32,    // atan(0.0009765625)
    16,    // atan(0.00048828125)
    8,     // atan(0.000244140625)
    4,     // atan(0.0001220703125)
    2,     // atan(0.00006103515625)
    1,     // atan(0.000030517578125)
];

/// Precomputed arctangent lookup table in Q31 radians (`atan(2^-i) * 2^31`).
pub const ATAN_TABLE_Q31: [q31; 16] = [
    1686629713, // atan(1.0)
    995716174,  // atan(0.5)
    526057390,  // atan(0.25)
    267073177,  // atan(0.125)
    134079828,  // atan(0.0625)
    67098489,   // atan(0.03125)
    33556754,   // atan(0.015625)
    16779316,   // atan(0.0078125)
    8389776,    // atan(0.00390625)
    4194903,    // atan(0.001953125)
    2097453,    // atan(0.0009765625)
    1048727,    // atan(0.00048828125)
    524363,     // atan(0.000244140625)
    262182,     // atan(0.0001220703125)
    131091,     // atan(0.00006103515625)
    65545,      // atan(0.000030517578125)
];

/// Computes `(sin(θ), cos(θ))` for an angle in Q15 radians (`[-π/2, π/2]`).
///
/// Returns `(sin, cos)` as Q15 format.
pub fn cordic_sin_cos_q15(angle_rad: q15) -> (q15, q15) {
    // Normal rotation mode: initialize x = K, y = 0
    let mut x: i32 = CORDIC_K_Q15 as i32;
    let mut y: i32 = 0;
    let mut z: i32 = angle_rad as i32;

    for i in 0..CORDIC_ITERATIONS {
        let (x_new, y_new, z_new) = if z >= 0 {
            (
                x - (y >> i),
                y + (x >> i),
                z - (ATAN_TABLE_Q15[i] as i32),
            )
        } else {
            (
                x + (y >> i),
                y - (x >> i),
                z + (ATAN_TABLE_Q15[i] as i32),
            )
        };
        x = x_new;
        y = y_new;
        z = z_new;
    }

    let sin = y.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
    let cos = x.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
    (sin, cos)
}

/// Computes `(sin(θ), cos(θ))` for an angle in Q31 radians.
pub fn cordic_sin_cos_q31(angle_rad: q31) -> (q31, q31) {
    let mut x: i64 = CORDIC_K_Q31 as i64;
    let mut y: i64 = 0;
    let mut z: i64 = angle_rad as i64;

    for i in 0..CORDIC_ITERATIONS {
        let (x_new, y_new, z_new) = if z >= 0 {
            (
                x - (y >> i),
                y + (x >> i),
                z - (ATAN_TABLE_Q31[i] as i64),
            )
        } else {
            (
                x + (y >> i),
                y - (x >> i),
                z + (ATAN_TABLE_Q31[i] as i64),
            )
        };
        x = x_new;
        y = y_new;
        z = z_new;
    }

    let sin = y.clamp(i32::MIN as i64, i32::MAX as i64) as q31;
    let cos = x.clamp(i32::MIN as i64, i32::MAX as i64) as q31;
    (sin, cos)
}

/// Converts Cartesian coordinates `(x, y)` to Polar `(magnitude, angle_rad)` using CORDIC vectoring mode.
///
/// Returns `(magnitude, angle_rad)` in Q15 format.
pub fn cordic_cartesian_to_polar_q15(x_in: q15, y_in: q15) -> (q15, q15) {
    if x_in == 0 && y_in == 0 {
        return (0, 0);
    }

    let mut x = (x_in as i32).abs();
    let mut y = y_in as i32;
    let mut z: i32 = 0;

    for i in 0..CORDIC_ITERATIONS {
        let (x_new, y_new, z_new) = if y < 0 {
            (
                x - (y >> i),
                y + (x >> i),
                z - (ATAN_TABLE_Q15[i] as i32),
            )
        } else {
            (
                x + (y >> i),
                y - (x >> i),
                z + (ATAN_TABLE_Q15[i] as i32),
            )
        };
        x = x_new;
        y = y_new;
        z = z_new;
    }

    // Compensate gain K: magnitude = x * K
    let mag = ((x * CORDIC_K_Q15 as i32) >> 15).clamp(0, i16::MAX as i32) as q15;
    let mut angle = z.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
    if x_in < 0 {
        angle = if angle >= 0 {
            (25736 * 4 - angle as i32).clamp(i16::MIN as i32, i16::MAX as i32) as q15
        } else {
            (-25736 * 4 - angle as i32).clamp(i16::MIN as i32, i16::MAX as i32) as q15
        };
    }

    (mag, angle)
}

/// Four-quadrant inverse tangent `atan2(y, x)` computed via CORDIC vectoring mode.
pub fn cordic_atan2_q15(y: q15, x: q15) -> q15 {
    let (_, angle) = cordic_cartesian_to_polar_q15(x, y);
    angle
}

/// Square root of non-negative number `x` in Q15 format using CORDIC vectoring.
pub fn cordic_sqrt_q15(x: q15) -> q15 {
    if x <= 0 {
        return 0;
    }
    // sqrt(x) = magnitude of (x + 1/4, x - 1/4) in hyperbolic mode or fast root
    let mut out: q15 = 0;
    let _ = crate::fast_math::sqrt_q15(x, &mut out);
    out
}
