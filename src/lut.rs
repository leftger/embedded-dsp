//! Lookup-table trigonometry.
//!
//! Alternative to [`crate::fast_math`]'s polynomial approximations: a
//! 256-entry sin table (1KB), computed at compile time via a Taylor-series
//! `const fn` so no runtime initialization is needed. Trades accuracy
//! (~2% error) for speed on cores without hardware sin/cos or an FPU.

use core::f32::consts::PI;

/// 256-entry sine lookup table covering 0 to 2*pi.
/// Each entry is scaled by 32768 for fixed-point arithmetic.
const SIN_TABLE: [i16; 256] = generate_sin_table();

/// Generate the sine lookup table at compile time.
const fn generate_sin_table() -> [i16; 256] {
    let mut table = [0i16; 256];
    let mut i = 0;
    while i < 256 {
        // sin() isn't const, so approximate with a Taylor series:
        // sin(x) = x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
        // The truncated series only converges well for |x| <= pi, so
        // reduce the table angle into (-pi, pi] first -- without this,
        // entries in the upper quarter of the table (close to 2*pi) blow
        // up well past +/-1.0 and silently saturate when scaled by 32768.
        let angle_rad = (i as f32 * 2.0 * PI) / 256.0;
        let x = if angle_rad > PI {
            angle_rad - 2.0 * PI
        } else {
            angle_rad
        };
        let x2 = x * x;
        let x3 = x2 * x;
        let x5 = x3 * x2;
        let x7 = x5 * x2;
        let x9 = x7 * x2;
        let x11 = x9 * x2;

        let sin_val = x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0 - x11 / 39916800.0;

        table[i] = (sin_val * 32768.0) as i16;
        i += 1;
    }
    table
}

/// Fast sine using the lookup table.
///
/// Input: angle in radians (any range).
/// Output: sine value scaled by 32768 (-32768 to 32767).
///
/// To convert to float: `result as f32 / 32768.0`.
#[inline(always)]
pub fn fast_sin_i16(angle: f32) -> i16 {
    // Normalize angle to 0..2*pi and map to 0..255. `%` keeps the sign of
    // the dividend, so negative angles land in (-1.0, 0.0) here; nudge
    // those into [0.0, 1.0) before the (saturating) cast to usize, or
    // they'd all collapse to index 0 instead of wrapping around the table.
    let mut normalized = (angle % (2.0 * PI)) / (2.0 * PI);
    if normalized < 0.0 {
        normalized += 1.0;
    }
    let index = ((normalized * 256.0) as usize) & 0xFF;
    SIN_TABLE[index]
}

/// Fast cosine using the lookup table (cos(x) = sin(x + pi/2)).
#[inline(always)]
pub fn fast_cos_i16(angle: f32) -> i16 {
    fast_sin_i16(angle + PI / 2.0)
}

/// `2π` in Q16.16.
const TWO_PI_Q16: i32 = 411_775;

/// Sine of a Q16.16 radian angle, returning Q16.16 (LUT + linear interpolation).
pub fn sin_q16(angle: i32) -> i32 {
    let mut a = angle % TWO_PI_Q16;
    if a < 0 {
        a += TWO_PI_Q16;
    }
    let scaled = a as i64 * 256;
    let idx = (scaled / TWO_PI_Q16 as i64) as usize & 0xFF;
    let rem = scaled % TWO_PI_Q16 as i64;
    let s0 = SIN_TABLE[idx] as i32;
    let s1 = SIN_TABLE[(idx + 1) & 0xFF] as i32;
    let s = s0 + ((s1 - s0) as i64 * rem / TWO_PI_Q16 as i64) as i32;
    s.saturating_mul(2)
}

/// Cosine of a Q16.16 radian angle, returning Q16.16.
pub fn cos_q16(angle: i32) -> i32 {
    sin_q16(angle.saturating_add(TWO_PI_Q16 / 4))
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    fn ref_sin(a: f32) -> f32 {
        a.sin()
    }
    fn ref_cos(a: f32) -> f32 {
        a.cos()
    }

    #[test]
    fn sin_accuracy() {
        let test_angles = [
            0.0,
            PI / 6.0,
            PI / 4.0,
            PI / 3.0,
            PI / 2.0,
            PI,
            3.0 * PI / 2.0,
        ];
        for angle in test_angles.iter() {
            let fast = fast_sin_i16(*angle) as f32 / 32768.0;
            let accurate = ref_sin(*angle);
            let error = (fast - accurate).abs();
            assert!(error < 0.02, "error too large at angle {angle}: {error}");
        }
    }

    #[test]
    fn cos_accuracy() {
        let test_angles = [0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, PI];
        for angle in test_angles.iter() {
            let fast = fast_cos_i16(*angle) as f32 / 32768.0;
            let accurate = ref_cos(*angle);
            let error = (fast - accurate).abs();
            assert!(error < 0.02, "error too large at angle {angle}: {error}");
        }
    }

    #[test]
    fn periodicity() {
        let angle = PI / 4.0;
        let sin1 = fast_sin_i16(angle);
        let sin2 = fast_sin_i16(angle + 2.0 * PI);
        let sin3 = fast_sin_i16(angle + 4.0 * PI);
        assert!(
            (sin1 - sin2).abs() <= 1000,
            "periodicity error: {sin1} vs {sin2}"
        );
        assert!(
            (sin1 - sin3).abs() <= 1000,
            "periodicity error: {sin1} vs {sin3}"
        );
    }

    #[test]
    fn pythagorean_identity() {
        let angles = [0.0, 0.5, 1.0, PI / 4.0, PI / 3.0, 2.0, 3.14];
        for &a in &angles {
            let s = fast_sin_i16(a) as f32 / 32768.0;
            let c = fast_cos_i16(a) as f32 / 32768.0;
            let pyth = s * s + c * c;
            assert!(
                (pyth - 1.0).abs() < 0.01,
                "pythagorean identity failed at {a}: sin^2+cos^2 = {pyth}"
            );
        }
    }

    #[test]
    fn trig_symmetry() {
        let angles = [0.1, 0.785, 1.5, 2.3];
        for &a in &angles {
            let sin_pos = fast_sin_i16(a);
            let sin_neg = fast_sin_i16(-a);
            assert!(
                (sin_pos + sin_neg).abs() <= 1000,
                "sin odd symmetry failed at {a}: sin(a)={sin_pos}, sin(-a)={sin_neg}"
            );

            let cos_pos = fast_cos_i16(a);
            let cos_neg = fast_cos_i16(-a);
            assert!(
                (cos_pos - cos_neg).abs() <= 1000,
                "cos even symmetry failed at {a}: cos(a)={cos_pos}, cos(-a)={cos_neg}"
            );
        }
    }
}
