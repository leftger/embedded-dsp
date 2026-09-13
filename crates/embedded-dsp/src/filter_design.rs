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

/// Computes Direct Form I Biquad coefficients `[b0, b1, b2, a1, a2]` for a Band-Pass Filter (constant skirt gain, peak gain = Q).
pub fn biquad_bandpass_skirt_coeffs(center_freq: f32, sample_rate: f32, q: f32) -> [f32; 5] {
    let w0 = 2.0 * core::f32::consts::PI * center_freq / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    let a0 = 1.0 + alpha;
    let b0 = (sin_w0 / 2.0) / a0;
    let b1 = 0.0;
    let b2 = -b0;
    let a1 = (2.0 * cos_w0) / a0;
    let a2 = -(1.0 - alpha) / a0;

    [b0, b1, b2, a1, a2]
}

/// Computes Direct Form I Biquad coefficients `[b0, b1, b2, a1, a2]` for a Low-Shelf Filter (RBJ Audio EQ Cookbook).
pub fn biquad_lowshelf_coeffs(cutoff_freq: f32, sample_rate: f32, q: f32, gain_db: f32) -> [f32; 5] {
    let w0 = 2.0 * core::f32::consts::PI * cutoff_freq / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let a = (10.0f32).powf(gain_db / 40.0);
    let alpha = sin_w0 / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let b0 = (a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha)) / a0;
    let b1 = (2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)) / a0;
    let b2 = (a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha)) / a0;
    let a1 = (2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)) / a0;
    let a2 = -((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha) / a0;

    [b0, b1, b2, a1, a2]
}

/// Computes Direct Form I Biquad coefficients `[b0, b1, b2, a1, a2]` for a High-Shelf Filter (RBJ Audio EQ Cookbook).
pub fn biquad_highshelf_coeffs(cutoff_freq: f32, sample_rate: f32, q: f32, gain_db: f32) -> [f32; 5] {
    let w0 = 2.0 * core::f32::consts::PI * cutoff_freq / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let a = (10.0f32).powf(gain_db / 40.0);
    let alpha = sin_w0 / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let b0 = (a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha)) / a0;
    let b1 = (-2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)) / a0;
    let b2 = (a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha)) / a0;
    let a1 = (-2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)) / a0;
    let a2 = -((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha) / a0;

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

// --- Chebyshev Recursive Filter Design (Steven W. Smith, Ch. 20) ---

/// Computes one two-pole Direct Form I biquad stage `[b0, b1, b2, a1, a2]` of a Chebyshev
/// recursive filter (Steven W. Smith, Ch. 20, Table 20-5), for pole-pair `pole_pair`
/// (1-indexed, `1..=num_poles / 2`) of a `num_poles`-pole filter.
///
/// `cutoff_norm`: cutoff frequency as a fraction of the sample rate (`0.0..0.5`).
/// `high_pass`: `false` for low-pass, `true` for high-pass.
/// `ripple_percent`: passband ripple, `0.0..29.0` (`0.0` gives a maximally-flat/Butterworth
/// response with no ripple).
/// `num_poles`: total pole count for the filter this stage belongs to; must be even, `2..=20`.
///
/// The returned stage is not normalized for unity passband gain; use
/// [`chebyshev_lowpass_biquads`] / [`chebyshev_highpass_biquads`] to design a complete,
/// gain-normalized cascade.
pub fn chebyshev_biquad_stage(
    cutoff_norm: f32,
    high_pass: bool,
    ripple_percent: f32,
    num_poles: u32,
    pole_pair: u32,
) -> [f32; 5] {
    let pi = core::f32::consts::PI;
    let np = num_poles as f32;
    let p = pole_pair as f32;

    // Pole location on the unit circle.
    let angle = pi / (2.0 * np) + (p - 1.0) * pi / np;
    let mut rp = -angle.cos();
    let mut ip = angle.sin();

    // Warp from a circle to an ellipse for a non-zero-ripple Chebyshev response.
    if ripple_percent != 0.0 {
        let es = ((100.0 / (100.0 - ripple_percent)).powf(2.0) - 1.0).sqrt();
        let vx = (1.0 / np) * ((1.0 / es) + ((1.0 / (es * es)) + 1.0).sqrt()).ln();
        let kx_raw = (1.0 / np) * ((1.0 / es) + ((1.0 / (es * es)) - 1.0).sqrt()).ln();
        let kx = (kx_raw.exp() + (-kx_raw).exp()) / 2.0;
        rp *= ((vx.exp() - (-vx).exp()) / 2.0) / kx;
        ip *= ((vx.exp() + (-vx).exp()) / 2.0) / kx;
    }

    // s-domain to z-domain conversion.
    let t = 2.0 * (0.5f32).tan();
    let w = 2.0 * pi * cutoff_norm;
    let m = rp * rp + ip * ip;
    let d = 4.0 - 4.0 * rp * t + m * t * t;
    let x0 = t * t / d;
    let x1 = 2.0 * t * t / d;
    let x2 = t * t / d;
    let y1 = (8.0 - 2.0 * m * t * t) / d;
    let y2 = (-4.0 - 4.0 * rp * t - m * t * t) / d;

    // Low-pass-to-low-pass, or low-pass-to-high-pass, frequency transform.
    let k = if high_pass {
        -(w / 2.0 + 0.5).cos() / (w / 2.0 - 0.5).cos()
    } else {
        (0.5 - w / 2.0).sin() / (0.5 + w / 2.0).sin()
    };

    let d2 = 1.0 + y1 * k - y2 * k * k;
    let b0 = (x0 - x1 * k + x2 * k * k) / d2;
    let mut b1 = (-2.0 * x0 * k + x1 + x1 * k * k - 2.0 * x2 * k) / d2;
    let b2 = (x0 * k * k - x1 * k + x2) / d2;
    let mut a1 = (2.0 * k + y1 + y1 * k * k - 2.0 * y2 * k) / d2;
    let a2 = (-(k * k) - y1 * k + y2) / d2;

    if high_pass {
        b1 = -b1;
        a1 = -a1;
    }

    [b0, b1, b2, a1, a2]
}

/// Designs a complete, gain-normalized Chebyshev low-pass filter as a cascade of Direct Form I
/// biquad stages (Steven W. Smith, Ch. 20). `out_coeffs` must be a slice of size
/// `5 * (num_poles / 2)`. `num_poles` must be even, `2..=20`; `ripple_percent` in `0.0..29.0`.
/// Larger pole counts amplify `f32` round-off error per the book's own guidance, and should be
/// used with care (consider `f64` or splitting into explicit two-pole stages for high orders).
pub fn chebyshev_lowpass_biquads(
    cutoff_norm: f32,
    ripple_percent: f32,
    num_poles: u32,
    out_coeffs: &mut [f32],
) {
    chebyshev_biquads(cutoff_norm, false, ripple_percent, num_poles, out_coeffs);
}

/// Designs a complete, gain-normalized Chebyshev high-pass filter as a cascade of Direct Form I
/// biquad stages (Steven W. Smith, Ch. 20). See [`chebyshev_lowpass_biquads`] for parameters.
pub fn chebyshev_highpass_biquads(
    cutoff_norm: f32,
    ripple_percent: f32,
    num_poles: u32,
    out_coeffs: &mut [f32],
) {
    chebyshev_biquads(cutoff_norm, true, ripple_percent, num_poles, out_coeffs);
}

fn chebyshev_biquads(
    cutoff_norm: f32,
    high_pass: bool,
    ripple_percent: f32,
    num_poles: u32,
    out_coeffs: &mut [f32],
) {
    let num_stages = (num_poles / 2) as usize;
    assert!(
        out_coeffs.len() >= num_stages * 5,
        "out_coeffs buffer too small"
    );

    // Overall passband gain is the product of each stage's gain at the reference frequency
    // (DC for low-pass, Nyquist for high-pass); normalizing the cascade to unity gain there is
    // equivalent to dividing any single stage's numerator by that product.
    let mut total_gain = 1.0f32;
    for k in 0..num_stages {
        let stage = chebyshev_biquad_stage(
            cutoff_norm,
            high_pass,
            ripple_percent,
            num_poles,
            (k + 1) as u32,
        );
        let [b0, b1, b2, a1, a2] = stage;
        total_gain *= if high_pass {
            (b0 - b1 + b2) / (1.0 + a1 - a2)
        } else {
            (b0 + b1 + b2) / (1.0 - a1 - a2)
        };
        out_coeffs[k * 5..(k + 1) * 5].copy_from_slice(&stage);
    }

    if total_gain != 0.0 {
        let inv_gain = 1.0 / total_gain;
        out_coeffs[0] *= inv_gain;
        out_coeffs[1] *= inv_gain;
        out_coeffs[2] *= inv_gain;
    }
}

// --- Single-Pole Recursive Filter Design (Steven W. Smith, Ch. 19) ---

/// Converts a normalized cutoff frequency (`0.0..0.5`, cycles/sample) to the sample-to-sample
/// decay factor `x` used to design a single-pole recursive filter (Eq. 19-5).
pub fn single_pole_decay_from_cutoff(cutoff_norm: f32) -> f32 {
    (-2.0 * core::f32::consts::PI * cutoff_norm).exp()
}

/// Converts a time constant (in samples, the time to decay to `1/e` &asymp; 36.8%) to the
/// sample-to-sample decay factor `x` used to design a single-pole recursive filter (Eq. 19-4).
pub fn single_pole_decay_from_time_constant(time_constant_samples: f32) -> f32 {
    (-1.0 / time_constant_samples).exp()
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
    if m < 3 || m.is_multiple_of(2) || fc_norm <= 0.0 || fc_norm >= 0.5 {
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
    if m < 3 || m.is_multiple_of(2) || f_low_norm <= 0.0 || f_high_norm >= 0.5 || f_low_norm >= f_high_norm {
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
    if m < 3 || m.is_multiple_of(2) || f_low_norm <= 0.0 || f_high_norm >= 0.5 || f_low_norm >= f_high_norm {
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

// --- Custom Filter Design via Frequency Sampling (Steven W. Smith, Ch. 17) ---

/// Designs a custom FIR filter kernel matching an arbitrary desired frequency response, using
/// the frequency-sampling method: build a Hermitian-symmetric spectrum from the desired
/// positive-frequency samples, inverse FFT it into an aliased impulse response, circularly
/// shift, truncate, and apply a Hamming window.
///
/// `desired_real` / `desired_imag`: the desired frequency response in rectangular form,
/// sampled at `fft_len / 2 + 1` points evenly spaced from DC (`0`) to Nyquist (`0.5`). For a
/// well-behaved real filter, `desired_imag[0]` and `desired_imag[fft_len / 2]` should be `0`
/// (the DC and Nyquist bins have no conjugate partner to mirror against).
/// `fft_len`: must be a power of 2, `>= out_taps.len()`, and `<= 512`; larger values better
/// approximate the desired response at the cost of a longer intermediate FFT.
/// `out_taps`: destination for the resulting FIR kernel; its length `M + 1` must be odd.
///
/// Requires the `transform` feature (enabled by `full`).
#[cfg(feature = "transform")]
pub fn fir_custom_frequency_sampling(
    desired_real: &[f32],
    desired_imag: &[f32],
    fft_len: usize,
    out_taps: &mut [f32],
) -> Status {
    let m = out_taps.len();
    if m < 3 || m.is_multiple_of(2) {
        return Status::ArgumentError;
    }
    if fft_len < 2 || (fft_len & (fft_len - 1)) != 0 || fft_len > 512 || fft_len < m {
        return Status::ArgumentError;
    }
    let half_spec = fft_len / 2 + 1;
    if desired_real.len() < half_spec || desired_imag.len() < half_spec {
        return Status::LengthError;
    }

    let mut c_data = [0.0f32; 1024];
    for k in 0..half_spec {
        c_data[2 * k] = desired_real[k];
        c_data[2 * k + 1] = desired_imag[k];
    }
    // Hermitian symmetry: negative-frequency bins are the conjugate mirror of the positive
    // ones, which guarantees a real (not complex) time-domain impulse response.
    for k in half_spec..fft_len {
        let mirror = fft_len - k;
        c_data[2 * k] = desired_real[mirror];
        c_data[2 * k + 1] = -desired_imag[mirror];
    }

    crate::transform::cfft_f32(&mut c_data[..2 * fft_len], fft_len, 1, 1);

    // Circular shift right by M/2 so the (aliased, wrapped-around) impulse response is
    // centered before truncation, then window it.
    let half = m / 2;
    let two_pi_over_m = 2.0 * core::f32::consts::PI / (m - 1) as f32;
    for i in 0..m {
        let src_idx = (i + fft_len - half) % fft_len;
        let w = 0.54 - 0.46 * (two_pi_over_m * i as f32).cos();
        out_taps[i] = c_data[2 * src_idx] * w;
    }

    Status::Success
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter Quantization and Scaling Pipeline (Design in Float, Deploy in Fixed)
// ─────────────────────────────────────────────────────────────────────────────

use crate::types::{q15, q31};

/// Gain scaling strategy for biquad SOS fixed-point quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingStrategy {
    /// Strict peak-gain scaling: guarantees no overflow for any sinusoidal input.
    LInfNorm,
    /// Energy-based root-mean-square gain scaling.
    L2Norm,
    /// Preserves direct coefficient scale (post_shift handles dynamic range).
    Direct,
}

/// Quantizes and scales floating-point biquad cascade coefficients into Q15.
///
/// Returns `Ok(post_shift)` on success, which should be passed directly to
/// [`crate::filtering::BiquadCascadeInstanceQ15`].
pub fn biquad_quantize_and_scale_q15(
    sos_f32: &[f32],
    out_q15: &mut [q15],
    strategy: ScalingStrategy,
) -> Result<u8, Status> {
    if sos_f32.len() != out_q15.len() || sos_f32.is_empty() || !sos_f32.len().is_multiple_of(5) {
        return Err(Status::LengthError);
    }

    let num_stages = sos_f32.len() / 5;
    let mut max_coeff_mag = 0.0f32;

    let mut scaled_f32 = [0.0f32; 128];
    if sos_f32.len() > scaled_f32.len() {
        return Err(Status::ArgumentError);
    }

    for stage in 0..num_stages {
        let idx = stage * 5;
        let mut b0 = sos_f32[idx];
        let mut b1 = sos_f32[idx + 1];
        let mut b2 = sos_f32[idx + 2];
        let a1 = sos_f32[idx + 3];
        let a2 = sos_f32[idx + 4];

        let scale_factor = match strategy {
            ScalingStrategy::LInfNorm => {
                let peak = crate::filter_analysis::biquad_peak_gain(&[b0, b1, b2, a1, a2], 64);
                if peak > 1.0 { 1.0 / peak } else { 1.0 }
            }
            ScalingStrategy::L2Norm => {
                let l2 = crate::filter_analysis::biquad_l2_norm(&[b0, b1, b2, a1, a2], 64);
                if l2 > 1.0 { 1.0 / l2 } else { 1.0 }
            }
            ScalingStrategy::Direct => 1.0,
        };

        b0 *= scale_factor;
        b1 *= scale_factor;
        b2 *= scale_factor;

        scaled_f32[idx] = b0;
        scaled_f32[idx + 1] = b1;
        scaled_f32[idx + 2] = b2;
        scaled_f32[idx + 3] = a1;
        scaled_f32[idx + 4] = a2;

        for k in 0..5 {
            let mag = scaled_f32[idx + k].abs();
            if mag > max_coeff_mag {
                max_coeff_mag = mag;
            }
        }
    }

    let mut post_shift = 0u8;
    let mut limit = 0.9999f32;
    while limit < max_coeff_mag && post_shift < 14 {
        post_shift += 1;
        limit *= 2.0;
    }

    let status = crate::support::biquad_coeffs_f32_to_q15(&scaled_f32[..sos_f32.len()], out_q15, post_shift);
    if status != Status::Success {
        return Err(status);
    }

    Ok(post_shift)
}

/// Quantizes and scales floating-point biquad cascade coefficients into Q31.
///
/// Returns `Ok(post_shift)` on success.
pub fn biquad_quantize_and_scale_q31(
    sos_f32: &[f32],
    out_q31: &mut [q31],
    strategy: ScalingStrategy,
) -> Result<u8, Status> {
    if sos_f32.len() != out_q31.len() || sos_f32.is_empty() || !sos_f32.len().is_multiple_of(5) {
        return Err(Status::LengthError);
    }

    let num_stages = sos_f32.len() / 5;
    let mut max_coeff_mag = 0.0f32;

    let mut scaled_f32 = [0.0f32; 128];
    if sos_f32.len() > scaled_f32.len() {
        return Err(Status::ArgumentError);
    }

    for stage in 0..num_stages {
        let idx = stage * 5;
        let mut b0 = sos_f32[idx];
        let mut b1 = sos_f32[idx + 1];
        let mut b2 = sos_f32[idx + 2];
        let a1 = sos_f32[idx + 3];
        let a2 = sos_f32[idx + 4];

        let scale_factor = match strategy {
            ScalingStrategy::LInfNorm => {
                let peak = crate::filter_analysis::biquad_peak_gain(&[b0, b1, b2, a1, a2], 64);
                if peak > 1.0 { 1.0 / peak } else { 1.0 }
            }
            ScalingStrategy::L2Norm => {
                let l2 = crate::filter_analysis::biquad_l2_norm(&[b0, b1, b2, a1, a2], 64);
                if l2 > 1.0 { 1.0 / l2 } else { 1.0 }
            }
            ScalingStrategy::Direct => 1.0,
        };

        b0 *= scale_factor;
        b1 *= scale_factor;
        b2 *= scale_factor;

        scaled_f32[idx] = b0;
        scaled_f32[idx + 1] = b1;
        scaled_f32[idx + 2] = b2;
        scaled_f32[idx + 3] = a1;
        scaled_f32[idx + 4] = a2;

        for k in 0..5 {
            let mag = scaled_f32[idx + k].abs();
            if mag > max_coeff_mag {
                max_coeff_mag = mag;
            }
        }
    }

    let mut post_shift = 0u8;
    let mut limit = 0.9999f32;
    while limit < max_coeff_mag && post_shift < 14 {
        post_shift += 1;
        limit *= 2.0;
    }

    let status = crate::support::biquad_coeffs_f32_to_q31(&scaled_f32[..sos_f32.len()], out_q31, post_shift);
    if status != Status::Success {
        return Err(status);
    }

    Ok(post_shift)
}

/// Quantizes floating-point FIR filter taps into Q15 format.
pub fn fir_quantize_q15(taps_f32: &[f32], out_q15: &mut [q15]) -> Result<(), Status> {
    if taps_f32.len() != out_q15.len() || taps_f32.is_empty() {
        return Err(Status::LengthError);
    }
    for i in 0..taps_f32.len() {
        out_q15[i] = q15::saturating_from_num(taps_f32[i]);
    }
    Ok(())
}

/// FIR pulse-shaping length: `2 * samples_per_symbol * symbol_span + 1`.
#[inline]
pub const fn pulse_shaping_len(samples_per_symbol: usize, symbol_span: usize) -> usize {
    2 * samples_per_symbol * symbol_span + 1
}

fn pulse_shaping_args(
    samples_per_symbol: usize,
    symbol_span: usize,
    beta: f32,
    out: &[f32],
) -> Status {
    if samples_per_symbol < 1 || symbol_span < 1 {
        return Status::ArgumentError;
    }
    if !(0.0..=1.0).contains(&beta) {
        return Status::ArgumentError;
    }
    if out.len() != pulse_shaping_len(samples_per_symbol, symbol_span) {
        return Status::LengthError;
    }
    Status::Success
}

fn sinc_pi(z: f32) -> f32 {
    let az = z.abs();
    if az < 0.01 {
        let p = core::f32::consts::PI * z;
        (p * 0.5).cos() * (p * 0.25).cos() * (p * 0.125).cos()
    } else {
        let pz = core::f32::consts::PI * z;
        pz.sin() / pz
    }
}

/// Kaiser window β from a target stop-band attenuation in dB (Vaidyanathan).
///
/// Matches liquid-dsp `kaiser_beta_As`. `stopband_atten_db` is taken in absolute
/// value. Attenuation ≤ 21 dB yields `β = 0` (rectangular).
pub fn kaiser_beta_as(stopband_atten_db: f32) -> f32 {
    let as_db = stopband_atten_db.abs();
    let mut beta = if as_db > 50.0 {
        0.1102 * (as_db - 8.7)
    } else if as_db > 21.0 {
        0.5842 * (as_db - 21.0).powf(0.4) + 0.07886 * (as_db - 21.0)
    } else {
        0.0
    };
    if as_db > 110.0 {
        beta *= 1.02;
    }
    beta
}

/// Kaiser estimate of FIR length for transition width `df` (cycles/sample in
/// `(0, 0.5)`) and stop-band attenuation `stopband_atten_db` (> 0).
///
/// Matches liquid-dsp `estimate_req_filter_len` (Kaiser / Vaidyanathan form):
/// `(As − 7.95) / (14.26 · df)`, truncated toward zero.
pub fn estimate_req_filter_len(df: f32, stopband_atten_db: f32) -> Result<usize, Status> {
    if !(df > 0.0 && df <= 0.5) || stopband_atten_db <= 0.0 {
        return Err(Status::ArgumentError);
    }
    let n = (stopband_atten_db - 7.95) / (14.26 * df);
    if !n.is_finite() || n < 0.0 {
        return Err(Status::ArgumentError);
    }
    Ok(n as usize)
}

fn kaiser_window_sample(i: usize, len: usize, beta: f32) -> f32 {
    if len == 0 || i >= len || beta < 0.0 {
        return 0.0;
    }
    if len == 1 {
        return 1.0;
    }
    let t = i as f32 - (len - 1) as f32 / 2.0;
    let r = 2.0 * t / (len - 1) as f32;
    let arg = (1.0 - r * r).max(0.0).sqrt();
    crate::window::bessel_i0(beta * arg) / crate::window::bessel_i0(beta)
}

/// Kaiser-windowed sinc low-pass FIR (liquid-dsp `liquid_firdes_kaiser`).
///
/// `fc_norm` is cycles/sample in `(0, 0.5]`. `stopband_atten_db` selects β.
/// `frac_delay` is a fractional sample offset in `[-0.5, 0.5]`. Taps are **not**
/// DC-normalized (center tap is 1 for odd length and zero delay).
pub fn firdes_kaiser(
    fc_norm: f32,
    stopband_atten_db: f32,
    frac_delay: f32,
    out: &mut [f32],
) -> Status {
    if out.is_empty() {
        return Status::LengthError;
    }
    if !(fc_norm > 0.0 && fc_norm <= 0.5) {
        return Status::ArgumentError;
    }
    if !(-0.5..=0.5).contains(&frac_delay) {
        return Status::ArgumentError;
    }
    let n = out.len();
    let beta = kaiser_beta_as(stopband_atten_db);
    let half = (n - 1) as f32 / 2.0;
    for (i, h) in out.iter_mut().enumerate() {
        let t = i as f32 - half + frac_delay;
        *h = sinc_pi(2.0 * fc_norm * t) * kaiser_window_sample(i, n, beta);
    }
    Status::Success
}

/// Root-raised-cosine FIR (liquid-dsp `liquid_firdes_rrcos`).
///
/// `out` must have length [`pulse_shaping_len`]. `beta` is the excess bandwidth in `[0, 1]`.
/// `frac_delay` is a fractional sample delay (`dt` in liquid-dsp). `beta = 0` yields a sinc.
pub fn firdes_rrc(
    samples_per_symbol: usize,
    symbol_span: usize,
    beta: f32,
    frac_delay: f32,
    out: &mut [f32],
) -> Status {
    let st = pulse_shaping_args(samples_per_symbol, symbol_span, beta, out);
    if st != Status::Success {
        return st;
    }
    let k = samples_per_symbol as f32;
    let m = symbol_span as f32;
    let pi = core::f32::consts::PI;
    for (n, h) in out.iter_mut().enumerate() {
        let z = (n as f32 + frac_delay) / k - m;
        if beta < 1e-6 {
            *h = sinc_pi(z);
            continue;
        }
        if z.abs() < 1e-5 {
            *h = 1.0 - beta + 4.0 * beta / pi;
            continue;
        }
        let g = 1.0 - 16.0 * beta * beta * z * z;
        if (g * g) < 1e-5 {
            let g1 = 1.0 + 2.0 / pi;
            let g2 = (0.25 * pi / beta).sin();
            let g3 = 1.0 - 2.0 / pi;
            let g4 = (0.25 * pi / beta).cos();
            *h = beta / 2.0f32.sqrt() * (g1 * g2 + g3 * g4);
        } else {
            let t1 = ((1.0 + beta) * pi * z).cos();
            let t2 = ((1.0 - beta) * pi * z).sin();
            let t3 = 1.0 / (4.0 * beta * z);
            let t4 = 4.0 * beta / (pi * (1.0 - 16.0 * beta * beta * z * z));
            *h = t4 * (t1 + t2 * t3);
        }
    }
    Status::Success
}

/// Raised-cosine FIR (liquid-dsp `liquid_firdes_rcos`).
///
/// See [`firdes_rrc`] for arguments. `beta = 0` yields a sinc.
pub fn firdes_rc(
    samples_per_symbol: usize,
    symbol_span: usize,
    beta: f32,
    frac_delay: f32,
    out: &mut [f32],
) -> Status {
    let st = pulse_shaping_args(samples_per_symbol, symbol_span, beta, out);
    if st != Status::Success {
        return st;
    }
    let k = samples_per_symbol as f32;
    let m = symbol_span as f32;
    let pi = core::f32::consts::PI;
    for (n, h) in out.iter_mut().enumerate() {
        let z = (n as f32 + frac_delay) / k - m;
        if beta < 1e-6 {
            *h = sinc_pi(z);
            continue;
        }
        let t3 = 1.0 - 4.0 * beta * beta * z * z;
        if t3.abs() < 1e-3 {
            *h = (pi / (2.0 * beta)).sin() * beta * 0.5;
        } else {
            *h = (beta * pi * z).cos() * sinc_pi(z) / t3;
        }
    }
    Status::Success
}

#[allow(clippy::excessive_precision)]
fn erf_f32(x: f32) -> f32 {
    // Abramowitz and Stegun 7.1.26
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

fn gauss_q(z: f32) -> f32 {
    0.5 * (1.0 - erf_f32(z * core::f32::consts::FRAC_1_SQRT_2))
}

/// GMSK *transmit* pulse (liquid-dsp `liquid_firdes_gmsktx`).
///
/// Difference of Gaussian Q-functions, normalized so the discrete integral is `π/2`
/// then scaled by `samples_per_symbol`. Receive-side GMSK design (FFT / heap) is not
/// ported.
pub fn firdes_gmsk_tx(
    samples_per_symbol: usize,
    symbol_span: usize,
    beta: f32,
    frac_delay: f32,
    out: &mut [f32],
) -> Status {
    let st = pulse_shaping_args(samples_per_symbol, symbol_span, beta, out);
    if st != Status::Success {
        return st;
    }
    if beta < 1e-6 {
        return Status::ArgumentError;
    }
    let k = samples_per_symbol as f32;
    let m = symbol_span as f32;
    let c0 = 1.0 / 2.0f32.ln().sqrt();
    let two_pi = 2.0 * core::f32::consts::PI;
    for (i, h) in out.iter_mut().enumerate() {
        let t = i as f32 / k - m + frac_delay;
        *h = gauss_q(two_pi * beta * (t - 0.5) * c0) - gauss_q(two_pi * beta * (t + 0.5) * c0);
    }
    let e: f32 = out.iter().copied().sum();
    if e.abs() < 1e-12 {
        return Status::ArgumentError;
    }
    let scale = core::f32::consts::PI / (2.0 * e) * k;
    for h in out.iter_mut() {
        *h *= scale;
    }
    Status::Success
}

const ELLIP_LANDEN: usize = 7;

#[derive(Clone, Copy)]
struct C32 {
    re: f32,
    im: f32,
}

impl C32 {
    const ONE: Self = Self { re: 1.0, im: 0.0 };
    const I: Self = Self { re: 0.0, im: 1.0 };

    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }

    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }

    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }

    fn scale(self, s: f32) -> Self {
        Self::new(self.re * s, self.im * s)
    }

    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }

    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    fn mag2(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    fn sqrt(self) -> Self {
        let r = self.mag2().sqrt();
        let sr = ((r + self.re) * 0.5).max(0.0).sqrt();
        let mut si = ((r - self.re) * 0.5).max(0.0).sqrt();
        if self.im < 0.0 {
            si = -si;
        }
        Self::new(sr, si)
    }

    fn ln(self) -> Self {
        Self::new(0.5 * self.mag2().ln(), self.im.atan2(self.re))
    }

    fn cos(self) -> Self {
        let (sin_x, cos_x) = (self.re.sin(), self.re.cos());
        let e = self.im.exp();
        let ei = (-self.im).exp();
        let cosh_y = (e + ei) * 0.5;
        let sinh_y = (e - ei) * 0.5;
        Self::new(cos_x * cosh_y, -sin_x * sinh_y)
    }

    fn acos(self) -> Self {
        // acos(z) = -i ln(z + i sqrt(1 - z^2))
        let one_minus = C32::ONE.sub(self.mul(self));
        let inner = self.add(C32::I.mul(one_minus.sqrt()));
        C32::I.scale(-1.0).mul(inner.ln())
    }
}

fn landenf(k: f32, v: &mut [f32; ELLIP_LANDEN]) {
    let mut kk = k;
    for slot in v.iter_mut() {
        let kp = (1.0 - kk * kk).max(0.0).sqrt();
        kk = (1.0 - kp) / (1.0 + kp);
        *slot = kk;
    }
}

fn ellipkf(k: f32) -> (f32, f32) {
    let kmin = 4e-4f32;
    let kmax = (1.0 - kmin * kmin).sqrt();
    let kp = (1.0 - k * k).max(0.0).sqrt();
    let big_k = if k > kmax {
        let l = -(0.25 * kp).ln();
        l + 0.25 * (l - 1.0) * kp * kp
    } else {
        let mut v = [0.0f32; ELLIP_LANDEN];
        landenf(k, &mut v);
        let mut acc = core::f32::consts::FRAC_PI_2;
        for vi in v {
            acc *= 1.0 + vi;
        }
        acc
    };
    let big_kp = if k < kmin {
        let l = -(k * 0.25).ln();
        l + 0.25 * (l - 1.0) * k * k
    } else {
        let mut vp = [0.0f32; ELLIP_LANDEN];
        landenf(kp, &mut vp);
        let mut acc = core::f32::consts::FRAC_PI_2;
        for vi in vp {
            acc *= 1.0 + vi;
        }
        acc
    };
    (big_k, big_kp)
}

fn ellipdegf(order: f32, k1: f32) -> f32 {
    let (k1k, k1p) = ellipkf(k1);
    let q1 = (-core::f32::consts::PI * k1p / k1k).exp();
    let q = q1.powf(1.0 / order);
    let n = ELLIP_LANDEN as i32;
    let mut b = 0.0f32;
    for m in 0..n {
        b += q.powf((m * (m + 1)) as f32);
    }
    let mut a = 0.0f32;
    for m in 1..n {
        a += q.powf((m * m) as f32);
    }
    let g = b / (1.0 + 2.0 * a);
    4.0 * q.sqrt() * g * g
}

fn ellip_cd(u: C32, k: f32) -> C32 {
    let mut wn = u.scale(core::f32::consts::FRAC_PI_2).cos();
    let mut v = [0.0f32; ELLIP_LANDEN];
    landenf(k, &mut v);
    for i in (0..ELLIP_LANDEN).rev() {
        let vi = v[i];
        wn = wn.scale(1.0 + vi).div(C32::ONE.add(wn.mul(wn).scale(vi)));
    }
    wn
}

fn ellip_acd(w: C32, k: f32) -> C32 {
    let mut v = [0.0f32; ELLIP_LANDEN];
    landenf(k, &mut v);
    let mut w = w;
    for i in 0..ELLIP_LANDEN {
        let v1 = if i == 0 { k } else { v[i - 1] };
        let inner = C32::ONE.sub(w.mul(w).scale(v1 * v1)).sqrt();
        w = w.div(C32::ONE.add(inner)).scale(2.0 / (1.0 + v[i]));
    }
    w.acos().scale(2.0 / core::f32::consts::PI)
}

fn ellip_asn(w: C32, k: f32) -> C32 {
    C32::ONE.sub(ellip_acd(w, k))
}

/// Second-order elliptic low-pass biquad `[b0, b1, b2, a1, a2]` (Direct Form I).
///
/// Analog prototype from Orfanidis / liquid-dsp `ellip_azpkf` (order 2), bilinear
/// transformed with prewarp `tan(π fc_norm)`. `fc_norm` is cycles/sample in `(0, 0.5)`.
/// `passband_ripple_db` and `stopband_atten_db` must be positive with stop-band
/// attenuation larger than the pass-band ripple.
pub fn elliptic_lowpass_biquad(
    fc_norm: f32,
    passband_ripple_db: f32,
    stopband_atten_db: f32,
) -> Result<[f32; 5], Status> {
    if !(fc_norm > 0.0 && fc_norm < 0.5) {
        return Err(Status::ArgumentError);
    }
    if passband_ripple_db <= 0.0 || stopband_atten_db <= passband_ripple_db {
        return Err(Status::ArgumentError);
    }
    let gp = 10.0f32.powf(-passband_ripple_db / 20.0);
    let gs = 10.0f32.powf(-stopband_atten_db / 20.0);
    let ep = (1.0 / (gp * gp) - 1.0).sqrt();
    let es = (1.0 / (gs * gs) - 1.0).sqrt();
    if !ep.is_finite() || !es.is_finite() || ep <= 0.0 || es <= ep {
        return Err(Status::ArgumentError);
    }
    let k1 = ep / es;
    let k = ellipdegf(2.0, k1);
    if !(k > 0.0 && k < 1.0) {
        return Err(Status::ArgumentError);
    }
    let wp = 1.0f32;
    let u = C32::new(0.5, 0.0);
    let zeta = ellip_cd(u, k);
    let za = C32::I.scale(wp).div(zeta.scale(k));
    let v0 = C32::I
        .scale(-1.0)
        .mul(ellip_asn(C32::I.scale(1.0 / ep), k1))
        .scale(0.5);
    let pa = C32::I.scale(wp).mul(ellip_cd(u.sub(C32::I.mul(v0)), k));
    let k0 = 1.0 / (1.0 + ep * ep).sqrt();
    let m = (core::f32::consts::PI * fc_norm).tan();
    let zm = za.scale(m);
    let zd = C32::ONE.add(zm).div(C32::ONE.sub(zm));
    let zd_c = zd.conj();
    let pm = pa.scale(m);
    let pd = C32::ONE.add(pm).div(C32::ONE.sub(pm));
    let pd_c = pd.conj();
    let mut kd = C32::new(k0, 0.0);
    kd = kd.mul(C32::ONE.sub(pd).div(C32::ONE.sub(zd)));
    kd = kd.mul(C32::ONE.sub(pd_c).div(C32::ONE.sub(zd_c)));
    let b0 = kd.re;
    let b1 = -kd.re * 2.0 * zd.re;
    let b2 = kd.re * zd.mag2();
    let a1 = 2.0 * pd.re;
    let a2 = -pd.mag2();
    if ![b0, b1, b2, a1, a2].iter().all(|c| c.is_finite()) {
        return Err(Status::NanInf);
    }
    Ok([b0, b1, b2, a1, a2])
}
