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
