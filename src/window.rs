//! Window functions (Hanning, Hamming, Blackman, Bartlett, Welch, Flat-top window generators).

#[allow(unused_imports)]
use crate::math::FloatMath;

/// Generate Hanning window of length `n`.
pub fn hanning_f32(dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        dst[0] = 1.0;
        return;
    }
    let factor = 2.0 * core::f32::consts::PI / ((n - 1) as f32);
    for i in 0..n {
        dst[i] = 0.5 * (1.0 - ((i as f32) * factor).cos());
    }
}

/// Generate Hamming window of length `n`.
pub fn hamming_f32(dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        dst[0] = 1.0;
        return;
    }
    let factor = 2.0 * core::f32::consts::PI / ((n - 1) as f32);
    for i in 0..n {
        dst[i] = 0.54 - 0.46 * ((i as f32) * factor).cos();
    }
}

/// Generate Blackman window of length `n`.
pub fn blackman_f32(dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        dst[0] = 1.0;
        return;
    }
    let factor = 2.0 * core::f32::consts::PI / ((n - 1) as f32);
    for i in 0..n {
        let a = (i as f32) * factor;
        dst[i] = 0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos();
    }
}

/// Generate Bartlett (Triangular) window of length `n`.
pub fn bartlett_f32(dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        dst[0] = 1.0;
        return;
    }
    let half = ((n - 1) as f32) / 2.0;
    for i in 0..n {
        dst[i] = 1.0 - ((i as f32 - half) / half).abs();
    }
}

/// Generate Welch parabolic window of length `n`.
pub fn welch_f32(dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        dst[0] = 1.0;
        return;
    }
    let half = ((n - 1) as f32) / 2.0;
    for i in 0..n {
        let term = (i as f32 - half) / half;
        dst[i] = 1.0 - term * term;
    }
}

/// Generate Flat-top window of length `n`.
pub fn flattop_f32(dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        dst[0] = 1.0;
        return;
    }
    let factor = 2.0 * core::f32::consts::PI / ((n - 1) as f32);
    for i in 0..n {
        let a = (i as f32) * factor;
        dst[i] = 0.21557895 - 0.41663158 * a.cos() + 0.277263158 * (2.0 * a).cos()
            - 0.083578947 * (3.0 * a).cos()
            + 0.006947368 * (4.0 * a).cos();
    }
}

/// Multiply signal elements in-place by window array.
pub fn apply_window_f32(signal: &mut [f32], window: &[f32]) {
    let len = signal.len().min(window.len());
    for i in 0..len {
        signal[i] *= window[i];
    }
}
