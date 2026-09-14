//! Convolution, correlation, median, and FFT-based convolution.

use crate::types::*;
// `fast_convolve_f32` is `transform`-gated, so is its import.
#[cfg(feature = "transform")]
use crate::transform::cfft_f32;

// --- Convolution ---

/// Convolution (`f32`) into `dst`.
pub fn conv_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    dst[..out_len].fill(0.0);
    for i in 0..len_a {
        for j in 0..len_b {
            if i + j < out_len {
                dst[i + j] += src_a[i] * src_b[j];
            }
        }
    }
}

/// Convolution (`q31`) into `dst`.
pub fn conv_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i64 = 0;
        let k_min = n.saturating_sub(len_b - 1);
        let k_max = n.min(len_a - 1);
        for k in k_min..=k_max {
            acc += (src_a[k].to_bits() as i64 * src_b[n - k].to_bits() as i64) >> 31;
        }
        dst[n] = q31::from_bits(acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    }
}

/// Convolution (`q15`) into `dst`.
pub fn conv_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i32 = 0;
        let k_min = n.saturating_sub(len_b - 1);
        let k_max = n.min(len_a - 1);
        for k in k_min..=k_max {
            acc += (src_a[k].to_bits() as i32 * src_b[n - k].to_bits() as i32) >> 15;
        }
        dst[n] = q15::from_bits(acc.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    }
}

/// Convolution (`q7`) into `dst`.
pub fn conv_q7(src_a: &[q7], src_b: &[q7], dst: &mut [q7]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i32 = 0;
        let k_min = n.saturating_sub(len_b - 1);
        let k_max = n.min(len_a - 1);
        for k in k_min..=k_max {
            acc += (src_a[k].to_bits() as i32 * src_b[n - k].to_bits() as i32) >> 7;
        }
        dst[n] = q7::from_bits(acc.clamp(i8::MIN as i32, i8::MAX as i32) as i8);
    }
}

// --- Correlation ---

/// Correlation (`f32`) into `dst`.
pub fn correlate_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    dst[..out_len].fill(0.0);
    for n in 0..out_len {
        let mut acc = 0.0f32;
        for k in 0..len_a {
            let idx_b = (k as isize) + (len_b as isize - 1) - (n as isize);
            if idx_b >= 0 && (idx_b as usize) < len_b {
                acc += src_a[k] * src_b[idx_b as usize];
            }
        }
        dst[n] = acc;
    }
}

/// Correlation (`q31`) into `dst`.
pub fn correlate_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i64 = 0;
        for k in 0..len_a {
            let idx_b = (k as isize) + (len_b as isize - 1) - (n as isize);
            if idx_b >= 0 && (idx_b as usize) < len_b {
                acc += (src_a[k].to_bits() as i64 * src_b[idx_b as usize].to_bits() as i64) >> 31;
            }
        }
        dst[n] = q31::from_bits(acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    }
}

/// Correlation (`q15`) into `dst`.
pub fn correlate_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i32 = 0;
        for k in 0..len_a {
            let idx_b = (k as isize) + (len_b as isize - 1) - (n as isize);
            if idx_b >= 0 && (idx_b as usize) < len_b {
                acc += (src_a[k].to_bits() as i32 * src_b[idx_b as usize].to_bits() as i32) >> 15;
            }
        }
        dst[n] = q15::from_bits(acc.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    }
}

// --- Non-linear Filtering (Median & Conditional Median) ---

/// 1D Conditional / Thresholded Median Filter for f32.
///
/// Replaces sample `src[i]` with the local median only if `|src[i] - median| > threshold`.
/// When `threshold == 0.0`, performs standard median filtering.
///
/// `window_len` must be odd and $\le 63$.
pub fn median_filter_1d_f32(
    src: &[f32],
    dst: &mut [f32],
    window_len: usize,
    threshold: f32,
) -> Status {
    let n = src.len();
    if n == 0 || dst.len() < n {
        return Status::LengthError;
    }
    if window_len == 0 || window_len.is_multiple_of(2) || window_len > 63 {
        return Status::ArgumentError;
    }

    let half = window_len / 2;
    let mut sort_buf = [0.0f32; 64];

    for i in 0..n {
        // Populate window with boundary clamping
        for j in 0..window_len {
            let idx = (i as isize + j as isize - half as isize).clamp(0, (n - 1) as isize) as usize;
            sort_buf[j] = src[idx];
        }

        // Insertion sort on small stack buffer
        for a in 1..window_len {
            let mut b = a;
            while b > 0 && sort_buf[b - 1] > sort_buf[b] {
                sort_buf.swap(b - 1, b);
                b -= 1;
            }
        }

        let med = sort_buf[half];
        let center = src[i];
        if (center - med).abs() >= threshold {
            dst[i] = med;
        } else {
            dst[i] = center;
        }
    }

    Status::Success
}

/// 1D Conditional Median Filter for Q15.
pub fn median_filter_1d_q15(
    src: &[q15],
    dst: &mut [q15],
    window_len: usize,
    threshold: q15,
) -> Status {
    let n = src.len();
    if n == 0 || dst.len() < n {
        return Status::LengthError;
    }
    if window_len == 0 || window_len.is_multiple_of(2) || window_len > 63 {
        return Status::ArgumentError;
    }

    let half = window_len / 2;
    let mut sort_buf = [q15::ZERO; 64];

    for i in 0..n {
        for j in 0..window_len {
            let idx = (i as isize + j as isize - half as isize).clamp(0, (n - 1) as isize) as usize;
            sort_buf[j] = src[idx];
        }

        for a in 1..window_len {
            let mut b = a;
            while b > 0 && sort_buf[b - 1] > sort_buf[b] {
                sort_buf.swap(b - 1, b);
                b -= 1;
            }
        }

        let med = sort_buf[half];
        let center = src[i];
        let diff = (center.to_bits() as i32 - med.to_bits() as i32).abs();
        if diff >= threshold.to_bits() as i32 {
            dst[i] = med;
        } else {
            dst[i] = center;
        }
    }

    Status::Success
}

/// 1D Conditional Median Filter for Q31.
pub fn median_filter_1d_q31(
    src: &[q31],
    dst: &mut [q31],
    window_len: usize,
    threshold: q31,
) -> Status {
    let n = src.len();
    if n == 0 || dst.len() < n {
        return Status::LengthError;
    }
    if window_len == 0 || window_len.is_multiple_of(2) || window_len > 63 {
        return Status::ArgumentError;
    }

    let half = window_len / 2;
    let mut sort_buf = [q31::ZERO; 64];

    for i in 0..n {
        for j in 0..window_len {
            let idx = (i as isize + j as isize - half as isize).clamp(0, (n - 1) as isize) as usize;
            sort_buf[j] = src[idx];
        }

        for a in 1..window_len {
            let mut b = a;
            while b > 0 && sort_buf[b - 1] > sort_buf[b] {
                sort_buf.swap(b - 1, b);
                b -= 1;
            }
        }

        let med = sort_buf[half];
        let center = src[i];
        let diff = (center.to_bits() as i64 - med.to_bits() as i64).abs();
        if diff >= threshold.to_bits() as i64 {
            dst[i] = med;
        } else {
            dst[i] = center;
        }
    }

    Status::Success
}

// --- FFT Fast Convolution ---

/// Performs fast linear convolution of `signal` and `kernel` via FFT multiplication.
/// Output length is `signal.len() + kernel.len() - 1`.
///
/// Requires the `transform` feature (enabled by `full`).
#[cfg(feature = "transform")]
pub fn fast_convolve_f32(signal: &[f32], kernel: &[f32], dst: &mut [f32]) -> Status {
    let len_sig = signal.len();
    let len_ker = kernel.len();
    if len_sig == 0 || len_ker == 0 {
        return Status::LengthError;
    }
    let total_len = len_sig + len_ker - 1;
    if dst.len() < total_len {
        return Status::LengthError;
    }

    // Find next power of 2
    let mut fft_n = 1;
    while fft_n < total_len {
        fft_n <<= 1;
    }

    if fft_n > 512 {
        // Fall back to time-domain convolution if size exceeds stack scratch buffer
        conv_f32(signal, kernel, dst);
        return Status::Success;
    }

    let mut sig_buf = [0.0f32; 1024]; // 2 * fft_n
    let mut ker_buf = [0.0f32; 1024];

    for i in 0..len_sig {
        sig_buf[2 * i] = signal[i];
    }
    for i in 0..len_ker {
        ker_buf[2 * i] = kernel[i];
    }

    cfft_f32(&mut sig_buf[..2 * fft_n], fft_n, 0, 1);
    cfft_f32(&mut ker_buf[..2 * fft_n], fft_n, 0, 1);

    // Pointwise complex multiplication: (a + jb) * (c + jd)
    for i in 0..fft_n {
        let a = sig_buf[2 * i];
        let b = sig_buf[2 * i + 1];
        let c = ker_buf[2 * i];
        let d = ker_buf[2 * i + 1];
        sig_buf[2 * i] = a * c - b * d;
        sig_buf[2 * i + 1] = a * d + b * c;
    }

    // Inverse FFT
    cfft_f32(&mut sig_buf[..2 * fft_n], fft_n, 1, 1);

    for i in 0..total_len {
        dst[i] = sig_buf[2 * i];
    }

    Status::Success
}
