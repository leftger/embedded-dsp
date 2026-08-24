//! Window functions (Hanning, Hamming, Blackman, Blackman-Harris, Bartlett, Welch, Flat-top window generators).

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::types::q15;

#[inline]
fn q15_from_unit(v: f32) -> q15 {
    (v * 32767.0).clamp(-32768.0, 32767.0) as q15
}

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

/// Generate 4-term Blackman-Harris window of length `n` (>92 dB sidelobe rejection).
pub fn blackman_harris_f32(dst: &mut [f32]) {
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
        dst[i] =
            0.35875 - 0.48829 * a.cos() + 0.14128 * (2.0 * a).cos() - 0.01168 * (3.0 * a).cos();
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
        dst[i] = 0.21557895 - 0.41663158 * a.cos() + 0.277_263_16 * (2.0 * a).cos()
            - 0.08357895 * (3.0 * a).cos()
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

fn fill_window_q15(dst: &mut [q15], fill: impl Fn(usize, usize) -> f32) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        dst[0] = 32767;
        return;
    }
    for i in 0..n {
        dst[i] = q15_from_unit(fill(i, n));
    }
}

/// Q15 Hanning window.
pub fn hanning_q15(dst: &mut [q15]) {
    fill_window_q15(dst, |i, n| {
        let factor = 2.0 * core::f32::consts::PI / ((n - 1) as f32);
        0.5 * (1.0 - ((i as f32) * factor).cos())
    });
}

/// Q15 Hamming window.
pub fn hamming_q15(dst: &mut [q15]) {
    fill_window_q15(dst, |i, n| {
        let factor = 2.0 * core::f32::consts::PI / ((n - 1) as f32);
        0.54 - 0.46 * ((i as f32) * factor).cos()
    });
}

/// Q15 Blackman window.
pub fn blackman_q15(dst: &mut [q15]) {
    fill_window_q15(dst, |i, n| {
        let factor = 2.0 * core::f32::consts::PI / ((n - 1) as f32);
        let a = (i as f32) * factor;
        0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos()
    });
}

/// Q15 Bartlett (triangular) window.
pub fn bartlett_q15(dst: &mut [q15]) {
    fill_window_q15(dst, |i, n| {
        let half = (n - 1) as f32 / 2.0;
        1.0 - ((i as f32 - half) / half).abs()
    });
}

/// Multiply Q15 signal in-place by a Q15 window (`>> 15`).
pub fn apply_window_q15(signal: &mut [q15], window: &[q15]) {
    let len = signal.len().min(window.len());
    for i in 0..len {
        signal[i] = (((signal[i] as i32) * (window[i] as i32)) >> 15) as q15;
    }
}
