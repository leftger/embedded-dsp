//! Power Spectral Density (PSD) estimation using Welch's Method (Averaged Overlapped Periodogram) and Bartlett/standard periodograms.

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::transform::cfft_f32;
use crate::types::Status;
use crate::window::*;

/// Window function choice for spectral estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WelchWindow {
    Rectangular,
    Hamming,
    Hanning,
    Blackman,
    BlackmanHarris,
    Bartlett,
    Welch,
}

/// Computes the Power Spectral Density (PSD) using Welch's Method (Averaged Overlapped Segment Periodograms).
///
/// `src`: continuous input signal time-series.
/// `dst_psd`: destination slice receiving the one-sided PSD of length `fft_len / 2 + 1` (or `fft_len / 2`).
/// `fft_len`: FFT segment size (must be power of 2, $\le 512$).
/// `overlap`: number of overlapping samples between successive FFT segments (must be $< \text{fft\_len}$).
/// `sample_rate`: sampling frequency in Hz (e.g. 1000.0, 44100.0).
/// `window`: window function applied to each segment.
/// `return_db`: if `true`, returns PSD in decibels ($10 \log_{10}(\text{PSD})$). If `false`, returns linear power.
pub fn welch_psd_f32(
    src: &[f32],
    dst_psd: &mut [f32],
    fft_len: usize,
    overlap: usize,
    sample_rate: f32,
    window: WelchWindow,
    return_db: bool,
) -> Status {
    let out_bins = fft_len / 2;
    if fft_len < 4 || (fft_len & (fft_len - 1)) != 0 || fft_len > 512 {
        return Status::ArgumentError;
    }
    if overlap >= fft_len || sample_rate <= 0.0 {
        return Status::ArgumentError;
    }
    if src.len() < fft_len || dst_psd.len() < out_bins {
        return Status::LengthError;
    }

    let step = fft_len - overlap;
    let num_segments = (src.len() - fft_len) / step + 1;
    if num_segments == 0 {
        return Status::LengthError;
    }

    // Generate window
    let mut win = [1.0f32; 512];
    match window {
        WelchWindow::Rectangular => win[..fft_len].fill(1.0),
        WelchWindow::Hamming => hamming_f32(&mut win[..fft_len]),
        WelchWindow::Hanning => hanning_f32(&mut win[..fft_len]),
        WelchWindow::Blackman => blackman_f32(&mut win[..fft_len]),
        WelchWindow::BlackmanHarris => blackman_harris_f32(&mut win[..fft_len]),
        WelchWindow::Bartlett => bartlett_f32(&mut win[..fft_len]),
        WelchWindow::Welch => welch_f32(&mut win[..fft_len]),
    }

    // Window power sum for normalization
    let mut win_power = 0.0f32;
    for i in 0..fft_len {
        win_power += win[i] * win[i];
    }
    if win_power == 0.0 {
        win_power = 1.0;
    }

    dst_psd[..out_bins].fill(0.0);

    let mut scratch = [0.0f32; 1024];

    for seg in 0..num_segments {
        let start_idx = seg * step;
        for i in 0..fft_len {
            scratch[2 * i] = src[start_idx + i] * win[i];
            scratch[2 * i + 1] = 0.0;
        }

        cfft_f32(&mut scratch[..2 * fft_len], fft_len, 0, 1);

        for k in 0..out_bins {
            let re = scratch[2 * k];
            let im = scratch[2 * k + 1];
            let mag_sq = re * re + im * im;
            dst_psd[k] += mag_sq;
        }
    }

    // Normalization factor for one-sided PSD:
    // 2.0 / (num_segments * sample_rate * win_power)
    let norm = 2.0f32 / (num_segments as f32 * sample_rate * win_power);
    for k in 0..out_bins {
        let linear_psd = dst_psd[k] * norm;
        if return_db {
            let clamped = if linear_psd > 1e-14 {
                linear_psd
            } else {
                1e-14
            };
            dst_psd[k] = 10.0 * clamped.log10();
        } else {
            dst_psd[k] = linear_psd;
        }
    }

    Status::Success
}

/// Computes the single-segment Periodogram Power Spectral Density.
pub fn periodogram_f32(
    src: &[f32],
    dst_psd: &mut [f32],
    fft_len: usize,
    sample_rate: f32,
    window: WelchWindow,
    return_db: bool,
) -> Status {
    welch_psd_f32(src, dst_psd, fft_len, 0, sample_rate, window, return_db)
}
