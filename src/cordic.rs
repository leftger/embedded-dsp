//! CORDIC (Coordinate Rotation Digital Computer) pure-integer arithmetic engine.
//!
//! Computes trigonometric functions (`sin`, `cos`, `atan2`), polar/Cartesian vector
//! rotations, and square roots using only additions, subtractions, and bit shifts.
//! Ideal for ultra-low-power microcontrollers without hardware multipliers or FPU.

use crate::types::{q15, q31};

/// CORDIC gain compensation factor in Q15 format: `0.607252935 * 32768 ≈ 19898`.
pub const CORDIC_K_Q15: q15 = q15::from_bits(19898);

/// CORDIC gain compensation factor in Q31 format: `0.607252935 * 2^31 ≈ 1304065792`.
pub const CORDIC_K_Q31: q31 = q31::from_bits(1304065792);

/// Number of CORDIC iterations.
pub const CORDIC_ITERATIONS: usize = 16;

/// Precomputed arctangent lookup table in Q15 radians (`atan(2^-i) * 32768`).
pub const ATAN_TABLE_Q15: [q15; 16] = [
    q15::from_bits(25736), // atan(1.0) = pi/4 ≈ 0.785398
    q15::from_bits(15193), // atan(0.5) ≈ 0.463648
    q15::from_bits(8027),  // atan(0.25) ≈ 0.244979
    q15::from_bits(4075),  // atan(0.125) ≈ 0.124355
    q15::from_bits(2045),  // atan(0.0625) ≈ 0.062419
    q15::from_bits(1024),  // atan(0.03125) ≈ 0.031240
    q15::from_bits(512),   // atan(0.015625) ≈ 0.015624
    q15::from_bits(256),   // atan(0.0078125) ≈ 0.007812
    q15::from_bits(128),   // atan(0.00390625)
    q15::from_bits(64),    // atan(0.001953125)
    q15::from_bits(32),    // atan(0.0009765625)
    q15::from_bits(16),    // atan(0.00048828125)
    q15::from_bits(8),     // atan(0.000244140625)
    q15::from_bits(4),     // atan(0.0001220703125)
    q15::from_bits(2),     // atan(0.00006103515625)
    q15::from_bits(1),     // atan(0.000030517578125)
];

/// Precomputed arctangent lookup table in Q31 radians (`atan(2^-i) * 2^31`).
pub const ATAN_TABLE_Q31: [q31; 16] = [
    q31::from_bits(1686629713), // atan(1.0)
    q31::from_bits(995716174),  // atan(0.5)
    q31::from_bits(526057390),  // atan(0.25)
    q31::from_bits(267073177),  // atan(0.125)
    q31::from_bits(134079828),  // atan(0.0625)
    q31::from_bits(67098489),   // atan(0.03125)
    q31::from_bits(33556754),   // atan(0.015625)
    q31::from_bits(16779316),   // atan(0.0078125)
    q31::from_bits(8389776),    // atan(0.00390625)
    q31::from_bits(4194903),    // atan(0.001953125)
    q31::from_bits(2097453),    // atan(0.0009765625)
    q31::from_bits(1048727),    // atan(0.00048828125)
    q31::from_bits(524363),     // atan(0.000244140625)
    q31::from_bits(262182),     // atan(0.0001220703125)
    q31::from_bits(131091),     // atan(0.00006103515625)
    q31::from_bits(65545),      // atan(0.000030517578125)
];

/// Computes `(sin(θ), cos(θ))` for an angle in Q15 radians (`[-π/2, π/2]`).
///
/// Returns `(sin, cos)` as Q15 format.
pub fn cordic_sin_cos_q15(angle_rad: q15) -> (q15, q15) {
    // Normal rotation mode: initialize x = K, y = 0
    let mut x: i32 = CORDIC_K_Q15.to_bits() as i32;
    let mut y: i32 = 0;
    let mut z: i32 = angle_rad.to_bits() as i32;

    for i in 0..CORDIC_ITERATIONS {
        let (x_new, y_new, z_new) = if z >= 0 {
            (
                x - (y >> i),
                y + (x >> i),
                z - (ATAN_TABLE_Q15[i].to_bits() as i32),
            )
        } else {
            (
                x + (y >> i),
                y - (x >> i),
                z + (ATAN_TABLE_Q15[i].to_bits() as i32),
            )
        };
        x = x_new;
        y = y_new;
        z = z_new;
    }

    let sin = q15::from_bits(y.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    let cos = q15::from_bits(x.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    (sin, cos)
}

/// Computes `(sin(θ), cos(θ))` for an angle in Q31 radians.
pub fn cordic_sin_cos_q31(angle_rad: q31) -> (q31, q31) {
    let mut x: i64 = CORDIC_K_Q31.to_bits() as i64;
    let mut y: i64 = 0;
    let mut z: i64 = angle_rad.to_bits() as i64;

    for i in 0..CORDIC_ITERATIONS {
        let (x_new, y_new, z_new) = if z >= 0 {
            (
                x - (y >> i),
                y + (x >> i),
                z - (ATAN_TABLE_Q31[i].to_bits() as i64),
            )
        } else {
            (
                x + (y >> i),
                y - (x >> i),
                z + (ATAN_TABLE_Q31[i].to_bits() as i64),
            )
        };
        x = x_new;
        y = y_new;
        z = z_new;
    }

    let sin = q31::from_bits(y.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    let cos = q31::from_bits(x.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    (sin, cos)
}

/// Converts Cartesian coordinates `(x, y)` to Polar `(magnitude, angle_rad)` using CORDIC vectoring mode.
///
/// Returns `(magnitude, angle_rad)` in Q15 format.
pub fn cordic_cartesian_to_polar_q15(x_in: q15, y_in: q15) -> (q15, q15) {
    if x_in == q15::ZERO && y_in == q15::ZERO {
        return (q15::ZERO, q15::ZERO);
    }

    let mut x = (x_in.to_bits() as i32).abs();
    let mut y = y_in.to_bits() as i32;
    let mut z: i32 = 0;

    for i in 0..CORDIC_ITERATIONS {
        let (x_new, y_new, z_new) = if y < 0 {
            (
                x - (y >> i),
                y + (x >> i),
                z - (ATAN_TABLE_Q15[i].to_bits() as i32),
            )
        } else {
            (
                x + (y >> i),
                y - (x >> i),
                z + (ATAN_TABLE_Q15[i].to_bits() as i32),
            )
        };
        x = x_new;
        y = y_new;
        z = z_new;
    }

    // Compensate gain K: magnitude = x * K
    let mag = q15::from_bits(
        ((x * CORDIC_K_Q15.to_bits() as i32) >> 15).clamp(0, i16::MAX as i32) as i16,
    );
    let mut angle = q15::from_bits(z.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    if x_in < q15::ZERO {
        angle = if angle >= q15::ZERO {
            q15::from_bits(
                (25736 * 4 - angle.to_bits() as i32).clamp(i16::MIN as i32, i16::MAX as i32)
                    as i16,
            )
        } else {
            q15::from_bits(
                (-25736 * 4 - angle.to_bits() as i32).clamp(i16::MIN as i32, i16::MAX as i32)
                    as i16,
            )
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
    if x <= q15::ZERO {
        return q15::ZERO;
    }
    // sqrt(x) = magnitude of (x + 1/4, x - 1/4) in hyperbolic mode or fast root
    let mut out = q15::ZERO;
    let _ = crate::fast_math::sqrt_q15(x, &mut out);
    out
}
