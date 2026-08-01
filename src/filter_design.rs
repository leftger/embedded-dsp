//! Filter design routines for calculating biquad IIR coefficients (Low-pass, High-pass, Band-pass, Notch, Peaking, All-pass, Butterworth).

#[allow(unused_imports)]
use crate::math::FloatMath;

/// Computes Direct Form I Biquad coefficients `[b0, b1, b2, a1, a2]` for a Low-Pass Filter.
///
/// `cutoff_freq`: Cutoff frequency in Hz.
/// `sample_rate`: Sampling rate in Hz.
/// `q`: Quality factor (e.g. 0.7071 for Butterworth alignment).
pub fn biquad_lowpass_coeffs(cutoff_freq: f32, sample_rate: f32, q: f32) -> [f32; 5] {
    let w0 = 2.0 * core::f32::consts::PI * cutoff_freq / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    let a0 = 1.0 + alpha;
    let b0 = (1.0 - cos_w0) / 2.0 / a0;
    let b1 = (1.0 - cos_w0) / a0;
    let b2 = (1.0 - cos_w0) / 2.0 / a0;
    // In Direct Form I (out = b0*x + b1*x1 + b2*x2 + a1*y1 + a2*y2), sign of feedback terms is flipped:
    let a1 = (2.0 * cos_w0) / a0;
    let a2 = -(1.0 - alpha) / a0;

    [b0, b1, b2, a1, a2]
}

/// Computes Direct Form I Biquad coefficients `[b0, b1, b2, a1, a2]` for a High-Pass Filter.
pub fn biquad_highpass_coeffs(cutoff_freq: f32, sample_rate: f32, q: f32) -> [f32; 5] {
    let w0 = 2.0 * core::f32::consts::PI * cutoff_freq / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    let a0 = 1.0 + alpha;
    let b0 = (1.0 + cos_w0) / 2.0 / a0;
    let b1 = -(1.0 + cos_w0) / a0;
    let b2 = (1.0 + cos_w0) / 2.0 / a0;
    let a1 = (2.0 * cos_w0) / a0;
    let a2 = -(1.0 - alpha) / a0;

    [b0, b1, b2, a1, a2]
}

/// Computes Direct Form I Biquad coefficients `[b0, b1, b2, a1, a2]` for a Band-Pass Filter (constant skirt gain).
pub fn biquad_bandpass_coeffs(center_freq: f32, sample_rate: f32, q: f32) -> [f32; 5] {
    let w0 = 2.0 * core::f32::consts::PI * center_freq / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    let a0 = 1.0 + alpha;
    let b0 = alpha / a0;
    let b1 = 0.0;
    let b2 = -alpha / a0;
    let a1 = (2.0 * cos_w0) / a0;
    let a2 = -(1.0 - alpha) / a0;

    [b0, b1, b2, a1, a2]
}

/// Computes Direct Form I Biquad coefficients `[b0, b1, b2, a1, a2]` for a Notch (Band-Stop) Filter.
pub fn biquad_notch_coeffs(center_freq: f32, sample_rate: f32, q: f32) -> [f32; 5] {
    let w0 = 2.0 * core::f32::consts::PI * center_freq / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    let a0 = 1.0 + alpha;
    let b0 = 1.0 / a0;
    let b1 = (-2.0 * cos_w0) / a0;
    let b2 = 1.0 / a0;
    let a1 = (2.0 * cos_w0) / a0;
    let a2 = -(1.0 - alpha) / a0;

    [b0, b1, b2, a1, a2]
}

/// Computes Direct Form I Biquad coefficients `[b0, b1, b2, a1, a2]` for a Peaking EQ Filter.
pub fn biquad_peaking_coeffs(center_freq: f32, sample_rate: f32, q: f32, gain_db: f32) -> [f32; 5] {
    let w0 = 2.0 * core::f32::consts::PI * center_freq / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let a = (10.0f32).powf(gain_db / 40.0);
    let alpha = sin_w0 / (2.0 * q);

    let a0 = 1.0 + alpha / a;
    let b0 = (1.0 + alpha * a) / a0;
    let b1 = (-2.0 * cos_w0) / a0;
    let b2 = (1.0 - alpha * a) / a0;
    let a1 = (2.0 * cos_w0) / a0;
    let a2 = -(1.0 - alpha / a) / a0;

    [b0, b1, b2, a1, a2]
}

/// Computes Direct Form I Biquad coefficients `[b0, b1, b2, a1, a2]` for an All-Pass Filter.
pub fn biquad_allpass_coeffs(center_freq: f32, sample_rate: f32, q: f32) -> [f32; 5] {
    let w0 = 2.0 * core::f32::consts::PI * center_freq / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    let a0 = 1.0 + alpha;
    let b0 = (1.0 - alpha) / a0;
    let b1 = (-2.0 * cos_w0) / a0;
    let b2 = (1.0 + alpha) / a0;
    let a1 = (2.0 * cos_w0) / a0;
    let a2 = -(1.0 - alpha) / a0;

    [b0, b1, b2, a1, a2]
}

/// Calculates multi-stage Butterworth Low-Pass filter biquad coefficients.
/// `out_coeffs` must be a slice of size `5 * (order / 2)`.
pub fn butterworth_lowpass_biquads(
    cutoff_freq: f32,
    sample_rate: f32,
    order: usize,
    out_coeffs: &mut [f32],
) {
    let num_stages = order / 2;
    assert!(
        out_coeffs.len() >= num_stages * 5,
        "out_coeffs buffer too small"
    );

    for k in 0..num_stages {
        let angle = core::f32::consts::PI * (2 * k + 1) as f32 / (2 * order) as f32;
        let q = 1.0 / (2.0 * angle.sin());
        let coeffs = biquad_lowpass_coeffs(cutoff_freq, sample_rate, q);
        out_coeffs[k * 5..(k + 1) * 5].copy_from_slice(&coeffs);
    }
}
