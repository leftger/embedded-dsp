//! Matrix operations (addition, subtraction, multiplication, scale, transpose, inverse, complex matrix multiplication).

use crate::types::*;

/// Matrix structure wrapping a slice of data in row-major order.
#[derive(Debug, Clone, Copy)]
pub struct MatrixInstance<'a, T> {
    pub num_rows: u16,
    pub num_cols: u16,
    pub data: &'a [T],
}

impl<'a, T> MatrixInstance<'a, T> {
    pub fn new(num_rows: u16, num_cols: u16, data: &'a [T]) -> Self {
        Self { num_rows, num_cols, data }
    }
}

/// Mutable Matrix structure wrapping a mutable slice of data in row-major order.
#[derive(Debug)]
pub struct MatrixInstanceMut<'a, T> {
    pub num_rows: u16,
    pub num_cols: u16,
    pub data: &'a mut [T],
}

impl<'a, T> MatrixInstanceMut<'a, T> {
    pub fn new(num_rows: u16, num_cols: u16, data: &'a mut [T]) -> Self {
        Self { num_rows, num_cols, data }
    }
}

// --- Matrix Addition ---

pub fn mat_add_f32(a: &MatrixInstance<f32>, b: &MatrixInstance<f32>, out: &mut MatrixInstanceMut<f32>) -> Status {
    if a.num_rows != b.num_rows || a.num_cols != b.num_cols || a.num_rows != out.num_rows || a.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let total = (a.num_rows as usize) * (a.num_cols as usize);
    if a.data.len() < total || b.data.len() < total || out.data.len() < total {
        return Status::LengthError;
    }
    for i in 0..total {
        out.data[i] = a.data[i] + b.data[i];
    }
    Status::Success
}

pub fn mat_add_q31(a: &MatrixInstance<q31>, b: &MatrixInstance<q31>, out: &mut MatrixInstanceMut<q31>) -> Status {
    if a.num_rows != b.num_rows || a.num_cols != b.num_cols || a.num_rows != out.num_rows || a.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let total = (a.num_rows as usize) * (a.num_cols as usize);
    if a.data.len() < total || b.data.len() < total || out.data.len() < total {
        return Status::LengthError;
    }
    for i in 0..total {
        out.data[i] = a.data[i].saturating_add(b.data[i]);
    }
    Status::Success
}

pub fn mat_add_q15(a: &MatrixInstance<q15>, b: &MatrixInstance<q15>, out: &mut MatrixInstanceMut<q15>) -> Status {
    if a.num_rows != b.num_rows || a.num_cols != b.num_cols || a.num_rows != out.num_rows || a.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let total = (a.num_rows as usize) * (a.num_cols as usize);
    if a.data.len() < total || b.data.len() < total || out.data.len() < total {
        return Status::LengthError;
    }
    for i in 0..total {
        out.data[i] = a.data[i].saturating_add(b.data[i]);
    }
    Status::Success
}

// --- Matrix Subtraction ---

pub fn mat_sub_f32(a: &MatrixInstance<f32>, b: &MatrixInstance<f32>, out: &mut MatrixInstanceMut<f32>) -> Status {
    if a.num_rows != b.num_rows || a.num_cols != b.num_cols || a.num_rows != out.num_rows || a.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let total = (a.num_rows as usize) * (a.num_cols as usize);
    if a.data.len() < total || b.data.len() < total || out.data.len() < total {
        return Status::LengthError;
    }
    for i in 0..total {
        out.data[i] = a.data[i] - b.data[i];
    }
    Status::Success
}

pub fn mat_sub_q31(a: &MatrixInstance<q31>, b: &MatrixInstance<q31>, out: &mut MatrixInstanceMut<q31>) -> Status {
    if a.num_rows != b.num_rows || a.num_cols != b.num_cols || a.num_rows != out.num_rows || a.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let total = (a.num_rows as usize) * (a.num_cols as usize);
    if a.data.len() < total || b.data.len() < total || out.data.len() < total {
        return Status::LengthError;
    }
    for i in 0..total {
        out.data[i] = a.data[i].saturating_sub(b.data[i]);
    }
    Status::Success
}

pub fn mat_sub_q15(a: &MatrixInstance<q15>, b: &MatrixInstance<q15>, out: &mut MatrixInstanceMut<q15>) -> Status {
    if a.num_rows != b.num_rows || a.num_cols != b.num_cols || a.num_rows != out.num_rows || a.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let total = (a.num_rows as usize) * (a.num_cols as usize);
    if a.data.len() < total || b.data.len() < total || out.data.len() < total {
        return Status::LengthError;
    }
    for i in 0..total {
        out.data[i] = a.data[i].saturating_sub(b.data[i]);
    }
    Status::Success
}

// --- Matrix Multiplication ---

pub fn mat_mult_f32(a: &MatrixInstance<f32>, b: &MatrixInstance<f32>, out: &mut MatrixInstanceMut<f32>) -> Status {
    if a.num_cols != b.num_rows || a.num_rows != out.num_rows || b.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let rows_a = a.num_rows as usize;
    let cols_a = a.num_cols as usize;
    let cols_b = b.num_cols as usize;

    for r in 0..rows_a {
        for c in 0..cols_b {
            let mut sum = 0.0f32;
            for k in 0..cols_a {
                sum += a.data[r * cols_a + k] * b.data[k * cols_b + c];
            }
            out.data[r * cols_b + c] = sum;
        }
    }
    Status::Success
}

pub fn mat_mult_q31(a: &MatrixInstance<q31>, b: &MatrixInstance<q31>, out: &mut MatrixInstanceMut<q31>) -> Status {
    if a.num_cols != b.num_rows || a.num_rows != out.num_rows || b.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let rows_a = a.num_rows as usize;
    let cols_a = a.num_cols as usize;
    let cols_b = b.num_cols as usize;

    for r in 0..rows_a {
        for c in 0..cols_b {
            let mut sum: i64 = 0;
            for k in 0..cols_a {
                sum += (a.data[r * cols_a + k] as i64 * b.data[k * cols_b + c] as i64) >> 31;
            }
            out.data[r * cols_b + c] = sum.clamp(i32::MIN as i64, i32::MAX as i64) as q31;
        }
    }
    Status::Success
}

pub fn mat_mult_q15(a: &MatrixInstance<q15>, b: &MatrixInstance<q15>, out: &mut MatrixInstanceMut<q15>) -> Status {
    if a.num_cols != b.num_rows || a.num_rows != out.num_rows || b.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let rows_a = a.num_rows as usize;
    let cols_a = a.num_cols as usize;
    let cols_b = b.num_cols as usize;

    for r in 0..rows_a {
        for c in 0..cols_b {
            let mut sum: i32 = 0;
            for k in 0..cols_a {
                sum += (a.data[r * cols_a + k] as i32 * b.data[k * cols_b + c] as i32) >> 15;
            }
            out.data[r * cols_b + c] = sum.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
        }
    }
    Status::Success
}

// --- Matrix Scale ---

pub fn mat_scale_f32(src: &MatrixInstance<f32>, scale: f32, out: &mut MatrixInstanceMut<f32>) -> Status {
    if src.num_rows != out.num_rows || src.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let total = (src.num_rows as usize) * (src.num_cols as usize);
    for i in 0..total {
        out.data[i] = src.data[i] * scale;
    }
    Status::Success
}

pub fn mat_scale_q31(src: &MatrixInstance<q31>, scale_fract: q31, shift: i8, out: &mut MatrixInstanceMut<q31>) -> Status {
    if src.num_rows != out.num_rows || src.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let total = (src.num_rows as usize) * (src.num_cols as usize);
    crate::basic_math::scale_q31(&src.data[..total], scale_fract, shift, &mut out.data[..total]);
    Status::Success
}

pub fn mat_scale_q15(src: &MatrixInstance<q15>, scale_fract: q15, shift: i8, out: &mut MatrixInstanceMut<q15>) -> Status {
    if src.num_rows != out.num_rows || src.num_cols != out.num_cols {
        return Status::SizeMismatch;
    }
    let total = (src.num_rows as usize) * (src.num_cols as usize);
    crate::basic_math::scale_q15(&src.data[..total], scale_fract, shift, &mut out.data[..total]);
    Status::Success
}

// --- Matrix Transpose ---

pub fn mat_trans_f32(src: &MatrixInstance<f32>, out: &mut MatrixInstanceMut<f32>) -> Status {
    if src.num_rows != out.num_cols || src.num_cols != out.num_rows {
        return Status::SizeMismatch;
    }
    let rows = src.num_rows as usize;
    let cols = src.num_cols as usize;

    for r in 0..rows {
        for c in 0..cols {
            out.data[c * rows + r] = src.data[r * cols + c];
        }
    }
    Status::Success
}

pub fn mat_trans_q31(src: &MatrixInstance<q31>, out: &mut MatrixInstanceMut<q31>) -> Status {
    if src.num_rows != out.num_cols || src.num_cols != out.num_rows {
        return Status::SizeMismatch;
    }
    let rows = src.num_rows as usize;
    let cols = src.num_cols as usize;

    for r in 0..rows {
        for c in 0..cols {
            out.data[c * rows + r] = src.data[r * cols + c];
        }
    }
    Status::Success
}

pub fn mat_trans_q15(src: &MatrixInstance<q15>, out: &mut MatrixInstanceMut<q15>) -> Status {
    if src.num_rows != out.num_cols || src.num_cols != out.num_rows {
        return Status::SizeMismatch;
    }
    let rows = src.num_rows as usize;
    let cols = src.num_cols as usize;

    for r in 0..rows {
        for c in 0..cols {
            out.data[c * rows + r] = src.data[r * cols + c];
        }
    }
    Status::Success
}

// --- Matrix Inverse (f32 Gauss-Jordan Elimination with partial pivoting) ---

pub fn mat_inverse_f32(src: &MatrixInstance<f32>, out: &mut MatrixInstanceMut<f32>) -> Status {
    if src.num_rows != src.num_cols || out.num_rows != out.num_cols || src.num_rows != out.num_rows {
        return Status::SizeMismatch;
    }
    let n = src.num_rows as usize;
    if n == 0 {
        return Status::SizeMismatch;
    }

    // Stack-allocated scratch buffer for n <= 16, or array for n x 2n augmented matrix
    // Gauss-Jordan elimination
    let mut aug = [0.0f32; 16 * 32];
    if n > 16 {
        return Status::ArgumentError; // Limit to 16x16 without heap allocation in no_std
    }

    for r in 0..n {
        for c in 0..n {
            aug[r * 2 * n + c] = src.data[r * n + c];
            aug[r * 2 * n + n + c] = if r == c { 1.0 } else { 0.0 };
        }
    }

    for i in 0..n {
        // Pivot selection
        let mut max_row = i;
        let mut max_val = aug[i * 2 * n + i].abs();
        for r in (i + 1)..n {
            let val = aug[r * 2 * n + i].abs();
            if val > max_val {
                max_val = val;
                max_row = r;
            }
        }

        if max_val < 1e-12 {
            return Status::Singular;
        }

        // Swap rows
        if max_row != i {
            for c in 0..(2 * n) {
                aug.swap(i * 2 * n + c, max_row * 2 * n + c);
            }
        }

        let pivot = aug[i * 2 * n + i];
        for c in 0..(2 * n) {
            aug[i * 2 * n + c] /= pivot;
        }

        for r in 0..n {
            if r != i {
                let factor = aug[r * 2 * n + i];
                for c in 0..(2 * n) {
                    let sub = factor * aug[i * 2 * n + c];
                    aug[r * 2 * n + c] -= sub;
                }
            }
        }
    }

    for r in 0..n {
        for c in 0..n {
            out.data[r * n + c] = aug[r * 2 * n + n + c];
        }
    }

    Status::Success
}
