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

// --- Pseudo-Random Number Generation & Noise ---

#[allow(unused_imports)]
use crate::math::FloatMath;

/// Simple deterministic zero-allocation 64-bit XorShift Pseudo-Random Number Generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XorShift64 {
    pub state: u64,
}

impl XorShift64 {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Generates uniform float in (0, 1]
        let val = (self.next_u64() >> 40) as u32;
        ((val as f32) + 1.0) / 16777217.0
    }
}

/// Fill destination slice with uniformly distributed random noise in `[min_val, max_val]`.
pub fn uniform_noise_f32(dst: &mut [f32], min_val: f32, max_val: f32, seed: &mut u64) {
    let mut rng = XorShift64::new(*seed);
    let span = max_val - min_val;
    for x in dst.iter_mut() {
        *x = min_val + rng.next_f32() * span;
    }
    *seed = rng.state;
}

/// Fill destination slice with Gaussian (White Noise) samples of given `mean` and `std_dev` using the Box-Muller transform.
pub fn gaussian_noise_f32(dst: &mut [f32], mean: f32, std_dev: f32, seed: &mut u64) {
    let mut rng = XorShift64::new(*seed);
    let len = dst.len();
    let mut i = 0;
    while i < len {
        let u1 = rng.next_f32();
        let u2 = rng.next_f32();
        let r = (-2.0f32 * u1.ln()).sqrt() * std_dev;
        let theta = 2.0f32 * core::f32::consts::PI * u2;
        dst[i] = mean + r * theta.cos();
        if i + 1 < len {
            dst[i + 1] = mean + r * theta.sin();
        }
        i += 2;
    }
    *seed = rng.state;
}
