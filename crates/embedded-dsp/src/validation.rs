//! A/B Differential testing, SQNR analysis, and phase inversion cancellation metrics.

#[cfg(all(not(feature = "std"), feature = "libm"))]
use libm::{log10f, sqrtf};

#[cfg(feature = "std")]
fn sqrtf(val: f32) -> f32 {
    val.sqrt()
}
#[cfg(feature = "std")]
fn log10f(val: f32) -> f32 {
    val.log10()
}

/// Comprehensive numerical metrics comparing a reference signal against a test algorithm output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferentialMetrics {
    /// Maximum absolute point-by-point error.
    pub peak_absolute_error: f32,
    /// Root Mean Square (RMS) difference error.
    pub rms_error: f32,
    /// Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB), based on total signal energy.
    pub sqnr_db: f32,
    /// Quantization Signal-to-Noise Ratio (QSNR) in decibels (dB), based on AC signal variance (mean-subtracted).
    pub qsnr_db: f32,
    /// Total Harmonic Distortion / noise percentage.
    pub thd_percent: f32,
    /// Returns true if the two signals cancel out perfectly (bit-exact or zero error).
    pub is_exact_match: bool,
}

/// Computes phase-inverted sum (cancellation difference buffer): `out[i] = ref[i] - target[i]`.
pub fn phase_invert_sum(reference: &[f32], test_target: &[f32], diff_out: &mut [f32]) {
    let len = reference.len().min(test_target.len()).min(diff_out.len());
    for i in 0..len {
        diff_out[i] = reference[i] - test_target[i];
    }
}

/// Computes Root Mean Square Error (RMSE) between a reference signal and a test signal.
pub fn root_mean_square_error(reference: &[f32], test_target: &[f32]) -> f32 {
    let len = reference.len().min(test_target.len());
    if len == 0 {
        return 0.0;
    }
    let mut sum_sq_err = 0.0f64;
    for i in 0..len {
        let err = (reference[i] as f64) - (test_target[i] as f64);
        sum_sq_err += err * err;
    }
    sqrtf((sum_sq_err / (len as f64)) as f32)
}

/// Computes Signal-to-Noise Ratio (SNR in dB): `10 * log10( sum(ref^2) / sum((ref - target)^2) )`.
pub fn signal_to_noise_ratio(reference: &[f32], test_target: &[f32]) -> f32 {
    let len = reference.len().min(test_target.len());
    if len == 0 {
        return 140.0;
    }
    let mut sum_sq_sig = 0.0f64;
    let mut sum_sq_err = 0.0f64;
    for i in 0..len {
        let r = reference[i] as f64;
        let diff = r - (test_target[i] as f64);
        sum_sq_sig += r * r;
        sum_sq_err += diff * diff;
    }
    if sum_sq_err < 1e-12 {
        140.0
    } else if sum_sq_sig < 1e-12 {
        0.0
    } else {
        10.0 * log10f((sum_sq_sig / sum_sq_err) as f32)
    }
}

/// Computes Quantization Signal-to-Noise Ratio (QSNR in dB) using signal variance:
/// `10 * log10( Variance(ref) / MSE(ref - target) )`.
///
/// Unlike standard raw SNR, QSNR subtracts the signal DC mean, ensuring DC offsets
/// (e.g. sensor zero-bias) do not artificially inflate the quantization quality metric.
pub fn quantization_snr(reference: &[f32], test_target: &[f32]) -> f32 {
    let len = reference.len().min(test_target.len());
    if len == 0 {
        return 140.0;
    }

    let mut sum_ref = 0.0f64;
    for i in 0..len {
        sum_ref += reference[i] as f64;
    }
    let mean_ref = sum_ref / (len as f64);

    let mut var_sig = 0.0f64;
    let mut mse_err = 0.0f64;
    for i in 0..len {
        let r = reference[i] as f64;
        let diff_mean = r - mean_ref;
        let err = r - (test_target[i] as f64);
        var_sig += diff_mean * diff_mean;
        mse_err += err * err;
    }
    var_sig /= len as f64;
    mse_err /= len as f64;

    if mse_err < 1e-12 {
        140.0
    } else if var_sig < 1e-12 {
        0.0
    } else {
        10.0 * log10f((var_sig / mse_err) as f32)
    }
}

/// Evaluates differential metrics between a reference signal and a test implementation output.
pub fn evaluate_differential(reference: &[f32], test_target: &[f32]) -> DifferentialMetrics {
    let len = reference.len().min(test_target.len());
    if len == 0 {
        return DifferentialMetrics {
            peak_absolute_error: 0.0,
            rms_error: 0.0,
            sqnr_db: 140.0,
            qsnr_db: 140.0,
            thd_percent: 0.0,
            is_exact_match: true,
        };
    }

    let mut max_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_sig = 0.0f64;
    let mut sum_sig = 0.0f64;

    for i in 0..len {
        let r = reference[i] as f64;
        let t = test_target[i] as f64;
        let err = (r - t).abs();

        if (err as f32) > max_err {
            max_err = err as f32;
        }

        sum_sq_err += err * err;
        sum_sq_sig += r * r;
        sum_sig += r;
    }

    let mean_sq_err = sum_sq_err / (len as f64);
    let rms_err = sqrtf(mean_sq_err as f32);
    let is_exact = max_err < 1e-7;

    let sqnr = if sum_sq_err < 1e-12 {
        140.0 // Effective digital silence / clean cancellation cap
    } else if sum_sq_sig < 1e-12 {
        0.0
    } else {
        let ratio = (sum_sq_sig / sum_sq_err) as f32;
        10.0 * log10f(ratio)
    };

    let mean_sig = sum_sig / (len as f64);
    let mut var_sig = 0.0f64;
    for i in 0..len {
        let diff = (reference[i] as f64) - mean_sig;
        var_sig += diff * diff;
    }

    let qsnr = if sum_sq_err < 1e-12 {
        140.0
    } else if var_sig < 1e-12 {
        0.0
    } else {
        let ratio = (var_sig / sum_sq_err) as f32;
        10.0 * log10f(ratio)
    };

    let thd = if sum_sq_sig < 1e-12 {
        0.0
    } else {
        (sqrtf((sum_sq_err / sum_sq_sig) as f32) * 100.0).clamp(0.0, 100.0)
    };

    DifferentialMetrics {
        peak_absolute_error: max_err,
        rms_error: rms_err,
        sqnr_db: sqnr,
        qsnr_db: qsnr,
        thd_percent: thd,
        is_exact_match: is_exact,
    }
}
