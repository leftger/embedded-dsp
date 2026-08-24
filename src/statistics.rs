//! Statistics functions (mean, variance, standard deviation, RMS, power, min, max, absmax, absmin, entropy, Kullback-Leibler, LogSumExp).

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::types::*;

// --- Mean ---

pub fn mean_f32(src: &[f32], result: &mut f32) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum = 0.0f32;
    for &val in src {
        sum += val;
    }
    *result = sum / (src.len() as f32);
    Status::Success
}

pub fn mean_f64(src: &[f64], result: &mut f64) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum = 0.0f64;
    for &val in src {
        sum += val;
    }
    *result = sum / (src.len() as f64);
    Status::Success
}

pub fn mean_q31(src: &[q31], result: &mut q31) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum: q63 = 0;
    for &val in src {
        sum += val as q63;
    }
    *result = (sum / (src.len() as q63)) as q31;
    Status::Success
}

pub fn mean_q15(src: &[q15], result: &mut q15) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum: i32 = 0;
    for &val in src {
        sum += val as i32;
    }
    *result = (sum / (src.len() as i32)) as q15;
    Status::Success
}

pub fn mean_q7(src: &[q7], result: &mut q7) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum: i32 = 0;
    for &val in src {
        sum += val as i32;
    }
    *result = (sum / (src.len() as i32)) as q7;
    Status::Success
}

// --- Variance ---

pub fn var_f32(src: &[f32], result: &mut f32) -> Status {
    if src.len() <= 1 {
        return Status::LengthError;
    }
    let mut mean = 0.0f32;
    mean_f32(src, &mut mean);
    let mut sum_sq = 0.0f32;
    for &val in src {
        let diff = val - mean;
        sum_sq += diff * diff;
    }
    *result = sum_sq / ((src.len() - 1) as f32);
    Status::Success
}

pub fn var_f64(src: &[f64], result: &mut f64) -> Status {
    if src.len() <= 1 {
        return Status::LengthError;
    }
    let mut mean = 0.0f64;
    mean_f64(src, &mut mean);
    let mut sum_sq = 0.0f64;
    for &val in src {
        let diff = val - mean;
        sum_sq += diff * diff;
    }
    *result = sum_sq / ((src.len() - 1) as f64);
    Status::Success
}

pub fn var_q31(src: &[q31], result: &mut q31) -> Status {
    if src.len() <= 1 {
        return Status::LengthError;
    }
    let mut m = 0;
    mean_q31(src, &mut m);
    let mut sum_sq: u64 = 0;
    for &val in src {
        let diff = (val as i64) - (m as i64);
        sum_sq += ((diff * diff) >> 31) as u64;
    }
    *result = ((sum_sq / (src.len() - 1) as u64) as i64).clamp(0, i32::MAX as i64) as q31;
    Status::Success
}

pub fn var_q15(src: &[q15], result: &mut q15) -> Status {
    if src.len() <= 1 {
        return Status::LengthError;
    }
    let mut m = 0;
    mean_q15(src, &mut m);
    let mut sum_sq: u32 = 0;
    for &val in src {
        let diff = (val as i32) - (m as i32);
        sum_sq += ((diff * diff) >> 15) as u32;
    }
    *result = ((sum_sq / (src.len() - 1) as u32) as i32).clamp(0, i16::MAX as i32) as q15;
    Status::Success
}

pub fn var_q7(src: &[q7], result: &mut q7) -> Status {
    if src.len() <= 1 {
        return Status::LengthError;
    }
    let mut m = 0;
    mean_q7(src, &mut m);
    let mut sum_sq: u32 = 0;
    for &val in src {
        let diff = (val as i32) - (m as i32);
        sum_sq += ((diff * diff) >> 7) as u32;
    }
    *result = ((sum_sq / (src.len() - 1) as u32) as i32).clamp(0, i8::MAX as i32) as q7;
    Status::Success
}

// --- Standard Deviation ---

pub fn std_f32(src: &[f32], result: &mut f32) -> Status {
    let mut v = 0.0f32;
    let status = var_f32(src, &mut v);
    if status == Status::Success {
        *result = v.sqrt();
    }
    status
}

pub fn std_f64(src: &[f64], result: &mut f64) -> Status {
    let mut v = 0.0f64;
    let status = var_f64(src, &mut v);
    if status == Status::Success {
        *result = v.sqrt();
    }
    status
}

pub fn std_q31(src: &[q31], result: &mut q31) -> Status {
    let mut v = 0;
    let status = var_q31(src, &mut v);
    if status == Status::Success {
        let _ = crate::fast_math::sqrt_q31(v, result);
    }
    status
}

pub fn std_q15(src: &[q15], result: &mut q15) -> Status {
    let mut v = 0;
    let status = var_q15(src, &mut v);
    if status == Status::Success {
        let _ = crate::fast_math::sqrt_q15(v, result);
    }
    status
}

pub fn std_q7(src: &[q7], result: &mut q7) -> Status {
    let mut v = 0;
    let status = var_q7(src, &mut v);
    if status == Status::Success {
        let n = (v.max(0) as u32) << 7;
        *result = crate::math::isqrt_u32(n).min(i8::MAX as u32) as q7;
    }
    status
}

// --- RMS ---

pub fn rms_f32(src: &[f32], result: &mut f32) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum_sq = 0.0f32;
    for &val in src {
        sum_sq += val * val;
    }
    *result = (sum_sq / (src.len() as f32)).sqrt();
    Status::Success
}

pub fn rms_q31(src: &[q31], result: &mut q31) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum_sq: u64 = 0;
    for &val in src {
        let v = val as i64;
        sum_sq += ((v * v) >> 31) as u64;
    }
    let mean_sq = (sum_sq / (src.len() as u64)).min(i32::MAX as u64) as q31;
    let _ = crate::fast_math::sqrt_q31(mean_sq, result);
    Status::Success
}

pub fn rms_q15(src: &[q15], result: &mut q15) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum_sq: u32 = 0;
    for &val in src {
        let v = val as i32;
        sum_sq += ((v * v) >> 15) as u32;
    }
    let mean_sq = (sum_sq / (src.len() as u32)).min(i16::MAX as u32) as q15;
    let _ = crate::fast_math::sqrt_q15(mean_sq, result);
    Status::Success
}

// --- Power ---

pub fn power_f32(src: &[f32], result: &mut f32) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum_sq = 0.0f32;
    for &val in src {
        sum_sq += val * val;
    }
    *result = sum_sq;
    Status::Success
}

pub fn power_q31(src: &[q31], result: &mut q63) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum_sq: q63 = 0;
    for &val in src {
        let v = val as i64;
        sum_sq += (v * v) >> 14;
    }
    *result = sum_sq;
    Status::Success
}

pub fn power_q15(src: &[q15], result: &mut q63) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum_sq: q63 = 0;
    for &val in src {
        let v = val as i32;
        sum_sq += (v * v) as q63;
    }
    *result = sum_sq;
    Status::Success
}

pub fn power_q7(src: &[q7], result: &mut q31) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut sum_sq: q31 = 0;
    for &val in src {
        let v = val as i32;
        sum_sq += v * v;
    }
    *result = sum_sq;
    Status::Success
}

// --- Min & Max ---

pub fn min_f32(src: &[f32], result: &mut f32, index: &mut usize) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut min_val = src[0];
    let mut min_idx = 0;
    for (i, &val) in src.iter().enumerate().skip(1) {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    *result = min_val;
    *index = min_idx;
    Status::Success
}

pub fn max_f32(src: &[f32], result: &mut f32, index: &mut usize) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut max_val = src[0];
    let mut max_idx = 0;
    for (i, &val) in src.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    *result = max_val;
    *index = max_idx;
    Status::Success
}

pub fn min_q31(src: &[q31], result: &mut q31, index: &mut usize) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut min_val = src[0];
    let mut min_idx = 0;
    for (i, &val) in src.iter().enumerate().skip(1) {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    *result = min_val;
    *index = min_idx;
    Status::Success
}

pub fn max_q31(src: &[q31], result: &mut q31, index: &mut usize) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut max_val = src[0];
    let mut max_idx = 0;
    for (i, &val) in src.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    *result = max_val;
    *index = max_idx;
    Status::Success
}

pub fn min_q15(src: &[q15], result: &mut q15, index: &mut usize) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut min_val = src[0];
    let mut min_idx = 0;
    for (i, &val) in src.iter().enumerate().skip(1) {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    *result = min_val;
    *index = min_idx;
    Status::Success
}

pub fn max_q15(src: &[q15], result: &mut q15, index: &mut usize) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut max_val = src[0];
    let mut max_idx = 0;
    for (i, &val) in src.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    *result = max_val;
    *index = max_idx;
    Status::Success
}

pub fn min_q7(src: &[q7], result: &mut q7, index: &mut usize) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut min_val = src[0];
    let mut min_idx = 0;
    for (i, &val) in src.iter().enumerate().skip(1) {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    *result = min_val;
    *index = min_idx;
    Status::Success
}

pub fn max_q7(src: &[q7], result: &mut q7, index: &mut usize) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut max_val = src[0];
    let mut max_idx = 0;
    for (i, &val) in src.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    *result = max_val;
    *index = max_idx;
    Status::Success
}

// --- Absmax & Absmin ---

pub fn absmax_f32(src: &[f32], result: &mut f32, index: &mut usize) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut max_val = src[0].abs();
    let mut max_idx = 0;
    for (i, &val) in src.iter().enumerate().skip(1) {
        let abs_val = val.abs();
        if abs_val > max_val {
            max_val = abs_val;
            max_idx = i;
        }
    }
    *result = max_val;
    *index = max_idx;
    Status::Success
}

pub fn absmin_f32(src: &[f32], result: &mut f32, index: &mut usize) -> Status {
    if src.is_empty() {
        return Status::LengthError;
    }
    let mut min_val = src[0].abs();
    let mut min_idx = 0;
    for (i, &val) in src.iter().enumerate().skip(1) {
        let abs_val = val.abs();
        if abs_val < min_val {
            min_val = abs_val;
            min_idx = i;
        }
    }
    *result = min_val;
    *index = min_idx;
    Status::Success
}

// --- Entropy, KL Divergence, LogSumExp ---

pub fn entropy_f32(src: &[f32]) -> f32 {
    let mut ent = 0.0f32;
    for &p in src {
        if p > 0.0 {
            ent -= p * p.ln();
        }
    }
    ent
}

pub fn kullback_leibler_f32(p: &[f32], q: &[f32]) -> f32 {
    let len = p.len().min(q.len());
    let mut kl = 0.0f32;
    for i in 0..len {
        if p[i] > 0.0 && q[i] > 0.0 {
            kl += p[i] * (p[i] / q[i]).ln();
        }
    }
    kl
}

pub fn logsumexp_f32(src: &[f32]) -> f32 {
    if src.is_empty() {
        return 0.0;
    }
    let mut max_v = src[0];
    for &v in src.iter().skip(1) {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum_exp = 0.0f32;
    for &v in src {
        sum_exp += (v - max_v).exp();
    }
    max_v + sum_exp.ln()
}
