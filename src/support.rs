//! Support functions (copy, fill, format conversions q7/q15/q31/f32, sort, barycenter, weighted sum).

use crate::types::*;

// --- Copy & Fill ---

pub fn copy_f32(src: &[f32], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    dst[..len].copy_from_slice(&src[..len]);
}

pub fn copy_q31(src: &[q31], dst: &mut [q31]) {
    let len = src.len().min(dst.len());
    dst[..len].copy_from_slice(&src[..len]);
}

pub fn copy_q15(src: &[q15], dst: &mut [q15]) {
    let len = src.len().min(dst.len());
    dst[..len].copy_from_slice(&src[..len]);
}

pub fn copy_q7(src: &[q7], dst: &mut [q7]) {
    let len = src.len().min(dst.len());
    dst[..len].copy_from_slice(&src[..len]);
}

pub fn fill_f32(value: f32, dst: &mut [f32]) {
    dst.fill(value);
}

pub fn fill_q31(value: q31, dst: &mut [q31]) {
    dst.fill(value);
}

pub fn fill_q15(value: q15, dst: &mut [q15]) {
    dst.fill(value);
}

pub fn fill_q7(value: q7, dst: &mut [q7]) {
    dst.fill(value);
}

// --- Type Conversions ---

pub fn q7_to_q15(src: &[q7], dst: &mut [q15]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] as i16) << 8;
    }
}

pub fn q7_to_q31(src: &[q7], dst: &mut [q31]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] as i32) << 24;
    }
}

pub fn q7_to_f32(src: &[q7], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] as f32) / 128.0;
    }
}

pub fn q15_to_q7(src: &[q15], dst: &mut [q7]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] >> 8) as q7;
    }
}

pub fn q15_to_q31(src: &[q15], dst: &mut [q31]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] as i32) << 16;
    }
}

pub fn q15_to_f32(src: &[q15], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] as f32) / 32768.0;
    }
}

pub fn q31_to_q7(src: &[q31], dst: &mut [q7]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] >> 24) as q7;
    }
}

pub fn q31_to_q15(src: &[q31], dst: &mut [q15]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] >> 16) as q15;
    }
}

pub fn q31_to_f32(src: &[q31], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] as f32) / 2147483648.0;
    }
}

pub fn f32_to_q7(src: &[f32], dst: &mut [q7]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] * 128.0).clamp(-128.0, 127.0) as q7;
    }
}

pub fn f32_to_q15(src: &[f32], dst: &mut [q15]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] * 32768.0).clamp(-32768.0, 32767.0) as q15;
    }
}

pub fn f32_to_q31(src: &[f32], dst: &mut [q31]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (src[i] * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as q31;
    }
}

// --- Sorting, Barycenter, Weighted Sum ---

/// Insertion sort for f32 (ascending or descending order).
pub fn sort_f32(src: &[f32], dst: &mut [f32], dir_ascending: bool) {
    let len = src.len().min(dst.len());
    dst[..len].copy_from_slice(&src[..len]);
    let slice = &mut dst[..len];
    for i in 1..len {
        let mut j = i;
        while j > 0 {
            let swap_needed = if dir_ascending {
                slice[j - 1] > slice[j]
            } else {
                slice[j - 1] < slice[j]
            };
            if swap_needed {
                slice.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }
}

/// Compute barycenter of points weighted by given weights.
pub fn barycenter_f32(
    in_pts: &[f32],
    weights: &[f32],
    out_center: &mut [f32],
    num_vecs: usize,
    vec_dim: usize,
) -> Status {
    if in_pts.len() < num_vecs * vec_dim || weights.len() < num_vecs || out_center.len() < vec_dim {
        return Status::LengthError;
    }
    out_center[..vec_dim].fill(0.0);
    let mut weight_sum = 0.0f32;
    for i in 0..num_vecs {
        let w = weights[i];
        weight_sum += w;
        for d in 0..vec_dim {
            out_center[d] += in_pts[i * vec_dim + d] * w;
        }
    }
    if weight_sum != 0.0 {
        for d in 0..vec_dim {
            out_center[d] /= weight_sum;
        }
    }
    Status::Success
}

/// Compute weighted sum of values.
pub fn weighted_sum_f32(in_vals: &[f32], weights: &[f32]) -> f32 {
    let len = in_vals.len().min(weights.len());
    let mut sum = 0.0f32;
    let mut w_sum = 0.0f32;
    for i in 0..len {
        sum += in_vals[i] * weights[i];
        w_sum += weights[i];
    }
    if w_sum != 0.0 {
        sum / w_sum
    } else {
        0.0
    }
}
