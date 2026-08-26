//! Distance metrics between vectors (Euclidean, Cosine, Chebyshev, Manhattan, Minkowski, Jaccard, Hamming, Canberra, Bray-Curtis).

#[allow(unused_imports)]
use crate::math::FloatMath;

/// Euclidean distance: `sqrt(sum((a_i - b_i)^2))`
pub fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum_sq = 0.0f32;
    for i in 0..len {
        let diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Cosine distance: `1 - (a . b) / (||a|| * ||b||)`
pub fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom != 0.0 {
        1.0 - (dot / denom)
    } else {
        1.0
    }
}

/// Chebyshev distance: `max(|a_i - b_i|)`
pub fn chebyshev_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut max_diff = 0.0f32;
    for i in 0..len {
        let diff = (a[i] - b[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}

/// Manhattan distance: `sum(|a_i - b_i|)`
pub fn manhattan_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

/// Minkowski distance: `(sum(|a_i - b_i|^p))^(1/p)`
pub fn minkowski_distance_f32(a: &[f32], b: &[f32], p: f32) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        let diff = (a[i] - b[i]).abs();
        sum += diff.powf(p);
    }
    sum.powf(1.0 / p)
}

/// Jaccard distance for boolean/binary vectors.
pub fn jaccard_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut tf = 0.0f32;
    let mut tt = 0.0f32;

    for i in 0..len {
        let a_bool = a[i] != 0.0;
        let b_bool = b[i] != 0.0;
        if a_bool && b_bool {
            tt += 1.0;
        } else if a_bool || b_bool {
            tf += 1.0;
        }
    }
    if tt + tf > 0.0 { tf / (tt + tf) } else { 0.0 }
}

/// Hamming distance between float vectors (count of mismatched elements).
pub fn hamming_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut diff_count = 0.0f32;
    for i in 0..len {
        if a[i] != b[i] {
            diff_count += 1.0;
        }
    }
    diff_count
}

/// Canberra distance: `sum(|a_i - b_i| / (|a_i| + |b_i|))`
pub fn canberra_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        let num = (a[i] - b[i]).abs();
        let denom = a[i].abs() + b[i].abs();
        if denom != 0.0 {
            sum += num / denom;
        }
    }
    sum
}

/// Bray-Curtis distance: `sum(|a_i - b_i|) / sum(|a_i + b_i|)`
pub fn bray_curtis_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut num = 0.0f32;
    let mut denom = 0.0f32;
    for i in 0..len {
        num += (a[i] - b[i]).abs();
        denom += (a[i] + b[i]).abs();
    }
    if denom != 0.0 { num / denom } else { 0.0 }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixed-Point Distance Metrics (Q15 / Q31)
// ─────────────────────────────────────────────────────────────────────────────

use crate::types::{q15, q31, q63};

/// Euclidean distance in Q15 normalized per sample: `sqrt(sum((a_i - b_i)^2) / N)`.
///
/// Output is guaranteed in `[0, 1.0]` (Q15 format).
pub fn euclidean_distance_q15(a: &[q15], b: &[q15]) -> q15 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0;
    }
    let mut sum_sq: q63 = 0;
    for i in 0..len {
        let diff = (a[i] as i32) - (b[i] as i32);
        sum_sq += (diff * diff) as q63;
    }
    let mean_sq = (sum_sq / (len as q63)) as q31;
    // mean_sq is at most (32768 - (-32768))^2 = (65536)^2 = 4.0 * 2^30.
    // Scale to Q15 range [0, 32767]
    let scaled_mean = (mean_sq >> 17).clamp(0, i16::MAX as i32) as q15;
    let mut root: q15 = 0;
    let _ = crate::fast_math::sqrt_q15(scaled_mean, &mut root);
    // rescale back
    let out = ((root as i32) << 1).clamp(0, i16::MAX as i32) as q15;
    out
}

/// Euclidean distance in Q31 normalized per sample: `sqrt(sum((a_i - b_i)^2) / N)`.
pub fn euclidean_distance_q31(a: &[q31], b: &[q31]) -> q31 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0;
    }
    let mut sum_sq: i128 = 0;
    for i in 0..len {
        let diff = (a[i] as i64) - (b[i] as i64);
        sum_sq += (diff * diff) as i128;
    }
    let mean_sq = (sum_sq / (len as i128)) as i64;
    let scaled_mean = (mean_sq >> 33).clamp(0, i32::MAX as i64) as q31;
    let mut root: q31 = 0;
    let _ = crate::fast_math::sqrt_q31(scaled_mean, &mut root);
    let out = ((root as i64) << 1).clamp(0, i32::MAX as i64) as q31;
    out
}

/// Chebyshev distance in Q15: `max(|a_i - b_i|)`.
pub fn chebyshev_distance_q15(a: &[q15], b: &[q15]) -> q15 {
    let len = a.len().min(b.len());
    let mut max_diff: i32 = 0;
    for i in 0..len {
        let diff = ((a[i] as i32) - (b[i] as i32)).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff.clamp(0, i16::MAX as i32) as q15
}

/// Chebyshev distance in Q31: `max(|a_i - b_i|)`.
pub fn chebyshev_distance_q31(a: &[q31], b: &[q31]) -> q31 {
    let len = a.len().min(b.len());
    let mut max_diff: i64 = 0;
    for i in 0..len {
        let diff = ((a[i] as i64) - (b[i] as i64)).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff.clamp(0, i32::MAX as i64) as q31
}

/// Manhattan distance in Q15: `sum(|a_i - b_i|)`.
pub fn manhattan_distance_q15(a: &[q15], b: &[q15]) -> q31 {
    let len = a.len().min(b.len());
    let mut sum: q31 = 0;
    for i in 0..len {
        let diff = ((a[i] as i32) - (b[i] as i32)).abs();
        sum = sum.saturating_add(diff);
    }
    sum
}

/// Manhattan distance in Q31: `sum(|a_i - b_i|)`.
pub fn manhattan_distance_q31(a: &[q31], b: &[q31]) -> q63 {
    let len = a.len().min(b.len());
    let mut sum: q63 = 0;
    for i in 0..len {
        let diff = ((a[i] as i64) - (b[i] as i64)).abs();
        sum = sum.saturating_add(diff);
    }
    sum
}

/// Cosine distance in Q15: `1.0 - (a . b) / (||a|| * ||b||)` in Q15 representation `[0, 1.0]`.
pub fn cosine_distance_q15(a: &[q15], b: &[q15]) -> q15 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0;
    }
    let mut dot: i64 = 0;
    let mut norm_a_sq: i64 = 0;
    let mut norm_b_sq: i64 = 0;

    for i in 0..len {
        let ai = a[i] as i64;
        let bi = b[i] as i64;
        dot += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
    }

    if norm_a_sq == 0 || norm_b_sq == 0 {
        return 32767; // Distance 1.0
    }

    let norm_a_mean = (norm_a_sq / len as i64).clamp(0, i32::MAX as i64) as q31;
    let norm_b_mean = (norm_b_sq / len as i64).clamp(0, i32::MAX as i64) as q31;

    let mut norm_a: q15 = 0;
    let mut norm_b: q15 = 0;
    let _ = crate::fast_math::sqrt_q15((norm_a_mean >> 15).clamp(0, i16::MAX as i32) as q15, &mut norm_a);
    let _ = crate::fast_math::sqrt_q15((norm_b_mean >> 15).clamp(0, i16::MAX as i32) as q15, &mut norm_b);

    let denom = (norm_a as i64 * norm_b as i64) * len as i64;
    if denom == 0 {
        return 32767;
    }

    let sim = (dot << 15) / denom;
    let sim_clamped = sim.clamp(-32768, 32767) as i32;
    (32767 - sim_clamped).clamp(0, 32767) as q15
}

/// Cosine distance in Q31: `1.0 - (a . b) / (||a|| * ||b||)` in Q31 representation `[0, 1.0]`.
pub fn cosine_distance_q31(a: &[q31], b: &[q31]) -> q31 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0;
    }
    let mut dot: i128 = 0;
    let mut norm_a_sq: i128 = 0;
    let mut norm_b_sq: i128 = 0;

    for i in 0..len {
        let ai = a[i] as i128;
        let bi = b[i] as i128;
        dot += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
    }

    if norm_a_sq == 0 || norm_b_sq == 0 {
        return i32::MAX;
    }

    let norm_a_mean = (norm_a_sq / len as i128) as i64;
    let norm_b_mean = (norm_b_sq / len as i128) as i64;

    let mut norm_a: q31 = 0;
    let mut norm_b: q31 = 0;
    let _ = crate::fast_math::sqrt_q31((norm_a_mean >> 31).clamp(0, i32::MAX as i64) as q31, &mut norm_a);
    let _ = crate::fast_math::sqrt_q31((norm_b_mean >> 31).clamp(0, i32::MAX as i64) as q31, &mut norm_b);

    let denom = (norm_a as i128 * norm_b as i128) * len as i128;
    if denom == 0 {
        return i32::MAX;
    }

    let sim = (dot << 31) / denom;
    let sim_clamped = sim.clamp(i32::MIN as i128, i32::MAX as i128) as i64;
    (i32::MAX as i64 - sim_clamped).clamp(0, i32::MAX as i64) as q31
}

/// Hamming distance between Q15 vectors (mismatch count).
pub fn hamming_distance_q15(a: &[q15], b: &[q15]) -> usize {
    let len = a.len().min(b.len());
    let mut diff = 0;
    for i in 0..len {
        if a[i] != b[i] {
            diff += 1;
        }
    }
    diff
}

/// Hamming distance between Q31 vectors (mismatch count).
pub fn hamming_distance_q31(a: &[q31], b: &[q31]) -> usize {
    let len = a.len().min(b.len());
    let mut diff = 0;
    for i in 0..len {
        if a[i] != b[i] {
            diff += 1;
        }
    }
    diff
}

/// Canberra distance in Q15 normalized: `(1/N) * sum(|a_i - b_i| / (|a_i| + |b_i|))` in `[0, 1.0]` (Q15).
pub fn canberra_distance_q15(a: &[q15], b: &[q15]) -> q15 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0;
    }
    let mut sum_q15: i64 = 0;
    for i in 0..len {
        let num = ((a[i] as i32) - (b[i] as i32)).abs();
        let denom = (a[i] as i32).abs() + (b[i] as i32).abs();
        if denom != 0 {
            let term = ((num as i64) << 15) / (denom as i64);
            sum_q15 += term;
        }
    }
    ((sum_q15 / len as i64).clamp(0, 32767)) as q15
}

/// Bray-Curtis distance in Q15: `sum(|a_i - b_i|) / sum(|a_i + b_i|)` in `[0, 1.0]` (Q15).
pub fn bray_curtis_distance_q15(a: &[q15], b: &[q15]) -> q15 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0;
    }
    let mut num: i64 = 0;
    let mut denom: i64 = 0;
    for i in 0..len {
        num += ((a[i] as i32) - (b[i] as i32)).abs() as i64;
        denom += ((a[i] as i32) + (b[i] as i32)).abs() as i64;
    }
    if denom == 0 {
        0
    } else {
        (((num << 15) / denom).clamp(0, 32767)) as q15
    }
}
