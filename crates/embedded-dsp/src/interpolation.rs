//! Interpolation functions (Linear, Bilinear, Spline).

use crate::types::*;

// --- Linear Interpolation ---

/// Linear interpolation for f32: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0).
pub fn linear_interp_f32(table: &[f32], x: f32, x_step: f32) -> f32 {
    if table.is_empty() || x_step <= 0.0 {
        return 0.0;
    }
    let idx_f = x / x_step;
    let idx = idx_f as usize;
    if idx >= table.len() - 1 {
        return table[table.len() - 1];
    }
    let frac = idx_f - (idx as f32);
    table[idx] + frac * (table[idx + 1] - table[idx])
}

pub fn linear_interp_q31(table: &[q31], x: q31) -> q31 {
    if table.len() < 2 {
        return q31::ZERO;
    }
    let idx_f = ((x.to_bits() as i64 + 2147483648) as u64 * (table.len() - 1) as u64) >> 31;
    let idx = (idx_f as usize).min(table.len() - 2);
    let y0 = table[idx].to_bits() as i64;
    let y1 = table[idx + 1].to_bits() as i64;
    q31::from_bits(((y0 + y1) / 2) as i32)
}

pub fn linear_interp_q15(table: &[q15], x: q15) -> q15 {
    if table.len() < 2 {
        return q15::ZERO;
    }
    let idx_f = ((x.to_bits() as i32 + 32768) as u32 * (table.len() - 1) as u32) >> 15;
    let idx = (idx_f as usize).min(table.len() - 2);
    let y0 = table[idx].to_bits() as i32;
    let y1 = table[idx + 1].to_bits() as i32;
    q15::from_bits(((y0 + y1) / 2) as i16)
}

// --- Bilinear Interpolation ---

/// Bilinear interpolation for 2D grid of size `num_rows x num_cols`.
pub fn bilinear_interp_f32(table: &[f32], num_rows: usize, num_cols: usize, x: f32, y: f32) -> f32 {
    let r = x.clamp(0.0, (num_rows - 1) as f32);
    let c = y.clamp(0.0, (num_cols - 1) as f32);

    let r0 = r as usize;
    let r1 = (r0 + 1).min(num_rows - 1);
    let c0 = c as usize;
    let c1 = (c0 + 1).min(num_cols - 1);

    let fr = r - r0 as f32;
    let fc = c - c0 as f32;

    let f00 = table[r0 * num_cols + c0];
    let f01 = table[r0 * num_cols + c1];
    let f10 = table[r1 * num_cols + c0];
    let f11 = table[r1 * num_cols + c1];

    (1.0 - fr) * ((1.0 - fc) * f00 + fc * f01) + fr * ((1.0 - fc) * f10 + fc * f11)
}

// --- Cubic Spline Interpolation ---

pub struct SplineInstanceF32<'a> {
    pub x: &'a [f32],
    pub y: &'a [f32],
    pub coeffs: &'a [f32], // 4 * (n - 1) coefficients: [a, b, c, d] per segment
}

impl<'a> SplineInstanceF32<'a> {
    pub fn interpolate(&self, x_val: f32) -> f32 {
        let n = self.x.len();
        if n == 0 {
            return 0.0;
        }
        if x_val <= self.x[0] {
            return self.y[0];
        }
        if x_val >= self.x[n - 1] {
            return self.y[n - 1];
        }

        let mut idx = 0;
        for i in 0..(n - 1) {
            if x_val >= self.x[i] && x_val <= self.x[i + 1] {
                idx = i;
                break;
            }
        }
        let dx = x_val - self.x[idx];
        let a = self.coeffs[idx * 4];
        let b = self.coeffs[idx * 4 + 1];
        let c = self.coeffs[idx * 4 + 2];
        let d = self.coeffs[idx * 4 + 3];

        a + b * dx + c * dx * dx + d * dx * dx * dx
    }
}
