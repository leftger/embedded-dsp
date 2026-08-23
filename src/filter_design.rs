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

/// Pre-warps continuous cutoff frequency `fc` for the bilinear transform at sampling rate `fs`.
/// Returns pre-warped analog frequency $\omega_p = 2 f_s \tan(\pi f_c / f_s)$.
pub fn prewarp_cutoff_f32(fc: f32, fs: f32) -> f32 {
    let pi_fc_over_fs = core::f32::consts::PI * fc / fs;
    2.0 * fs * pi_fc_over_fs.tan()
}

/// Converts a 2nd-order analog prototype filter section $H(s) = \frac{a_2 s^2 + a_1 s + a_0}{b_2 s^2 + b_1 s + b_0}$
/// into discrete Direct Form I Biquad coefficients `[b0, b1, b2, a1, a2]` using the Bilinear Transform.
pub fn bilinear_transform_biquad(
    a0: f32,
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    sample_rate: f32,
) -> [f32; 5] {
    let fs = sample_rate;
    let fs2 = fs * fs;

    let ad0 = 4.0 * a2 * fs2 + 2.0 * a1 * fs + a0;
    let ad1 = 2.0 * a0 - 8.0 * a2 * fs2;
    let ad2 = 4.0 * a2 * fs2 - 2.0 * a1 * fs + a0;

    let bd0 = 4.0 * b2 * fs2 + 2.0 * b1 * fs + b0;
    let bd1 = 2.0 * b0 - 8.0 * b2 * fs2;
    let bd2 = 4.0 * b2 * fs2 - 2.0 * b1 * fs + b0;

    let inv_bd0 = 1.0 / bd0;

    let b_0 = ad0 * inv_bd0;
    let b_1 = ad1 * inv_bd0;
    let b_2 = ad2 * inv_bd0;
    let a_1 = -bd1 * inv_bd0;
    let a_2 = -bd2 * inv_bd0;
    [b_0, b_1, b_2, a_1, a_2]
}

// --- Windowed-Sinc FIR Filter Design (Steven W. Smith, Ch. 16) ---

use crate::types::Status;

/// Computes a Low-Pass FIR filter kernel using the Blackman-Windowed Sinc method.
///
/// `fc_norm`: Cutoff frequency as a fraction of sampling rate ($0 < f_c < 0.5$).
/// `out_taps`: Destination slice for filter coefficients. Length $M$ must be odd and $\ge 3$.
pub fn fir_windowed_sinc_lowpass(fc_norm: f32, out_taps: &mut [f32]) -> Status {
    let m = out_taps.len();
    if m < 3 || m % 2 == 0 || fc_norm <= 0.0 || fc_norm >= 0.5 {
        return Status::ArgumentError;
    }

    let half = (m - 1) as f32 / 2.0;
    let two_pi_fc = 2.0 * core::f32::consts::PI * fc_norm;
    let two_pi_over_m = 2.0 * core::f32::consts::PI / (m - 1) as f32;

    let mut sum = 0.0f32;
    for i in 0..m {
        let d = (i as f32) - half;
        let sinc = if d == 0.0 {
            two_pi_fc
        } else {
            (two_pi_fc * d).sin() / d
        };

        // Blackman window
        let w = 0.42 - 0.5 * (two_pi_over_m * i as f32).cos()
            + 0.08 * (2.0 * two_pi_over_m * i as f32).cos();
        let tap = sinc * w;
        out_taps[i] = tap;
        sum += tap;
    }

    // Normalize for 0 dB DC gain
    if sum != 0.0 {
        let inv_sum = 1.0 / sum;
        for i in 0..m {
            out_taps[i] *= inv_sum;
        }
    }

    Status::Success
}

/// Computes a High-Pass FIR filter kernel using spectral inversion of the Windowed-Sinc Low-Pass.
///
/// `fc_norm`: Cutoff frequency as a fraction of sampling rate ($0 < f_c < 0.5$).
/// `out_taps`: Destination slice for filter coefficients. Length $M$ must be odd and $\ge 3$.
pub fn fir_windowed_sinc_highpass(fc_norm: f32, out_taps: &mut [f32]) -> Status {
    let status = fir_windowed_sinc_lowpass(fc_norm, out_taps);
    if status != Status::Success {
        return status;
    }

    let m = out_taps.len();
    let center = (m - 1) / 2;

    // Spectral inversion: negate all taps and add 1.0 to center tap
    for i in 0..m {
        out_taps[i] = -out_taps[i];
    }
    out_taps[center] += 1.0;

    Status::Success
}

/// Computes a Band-Pass FIR filter kernel using the difference of two Windowed-Sinc Low-Pass filters.
pub fn fir_windowed_sinc_bandpass(
    f_low_norm: f32,
    f_high_norm: f32,
    out_taps: &mut [f32],
) -> Status {
    let m = out_taps.len();
    if m < 3 || m % 2 == 0 || f_low_norm <= 0.0 || f_high_norm >= 0.5 || f_low_norm >= f_high_norm {
        return Status::ArgumentError;
    }

    let half = (m - 1) as f32 / 2.0;
    let two_pi_flow = 2.0 * core::f32::consts::PI * f_low_norm;
    let two_pi_fhigh = 2.0 * core::f32::consts::PI * f_high_norm;
    let two_pi_over_m = 2.0 * core::f32::consts::PI / (m - 1) as f32;

    for i in 0..m {
        let d = (i as f32) - half;
        let sinc_low = if d == 0.0 {
            two_pi_flow
        } else {
            (two_pi_flow * d).sin() / d
        };
        let sinc_high = if d == 0.0 {
            two_pi_fhigh
        } else {
            (two_pi_fhigh * d).sin() / d
        };
        let w = 0.42 - 0.5 * (two_pi_over_m * i as f32).cos()
            + 0.08 * (2.0 * two_pi_over_m * i as f32).cos();
        out_taps[i] = (sinc_high - sinc_low) * w;
    }

    // Normalize so center passband gain is 1.0
    let f_center = (f_low_norm + f_high_norm) / 2.0;
    let mut real_gain = 0.0f32;
    let mut imag_gain = 0.0f32;
    for i in 0..m {
        let angle = 2.0 * core::f32::consts::PI * f_center * (i as f32);
        real_gain += out_taps[i] * angle.cos();
        imag_gain -= out_taps[i] * angle.sin();
    }
    let mag = (real_gain * real_gain + imag_gain * imag_gain).sqrt();
    if mag > 1e-12 {
        let inv_mag = 1.0 / mag;
        for i in 0..m {
            out_taps[i] *= inv_mag;
        }
    }

    Status::Success
}

/// Computes a Band-Stop (Notch / Band-Reject) FIR filter kernel using spectral inversion of Band-Pass.
pub fn fir_windowed_sinc_bandstop(
    f_low_norm: f32,
    f_high_norm: f32,
    out_taps: &mut [f32],
) -> Status {
    let m = out_taps.len();
    if m < 3 || m % 2 == 0 || f_low_norm <= 0.0 || f_high_norm >= 0.5 || f_low_norm >= f_high_norm {
        return Status::ArgumentError;
    }

    let status = fir_windowed_sinc_bandpass(f_low_norm, f_high_norm, out_taps);
    if status != Status::Success {
        return status;
    }

    let center = (m - 1) / 2;
    for i in 0..m {
        out_taps[i] = -out_taps[i];
    }
    out_taps[center] += 1.0;

    Status::Success
}
