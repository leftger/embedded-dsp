//! Fast Fourier Transform (FFT), Real FFT (RFFT), Discrete Cosine Transform (DCT-IV), and Bit Reversal functions.

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::types::*;

/// Bit reversal function for interleaved complex array of size `2 * n`.
pub fn bit_reversal(data: &mut [f32], n: usize) {
    let mut j = 0;
    for i in 0..n {
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

/// In-place Complex FFT for floating point 32-bit (`f32`).
/// `data` is interleaved complex array of size `2 * n` (`[re0, im0, re1, im1, ...]`).
/// `ifft_flag`: 0 for forward FFT, 1 for inverse FFT (IFFT).
/// `bit_reverse_flag`: 1 to enable bit reversal, 0 to disable.
pub fn cfft_f32(data: &mut [f32], n: usize, ifft_flag: u8, bit_reverse_flag: u8) {
    if n < 2 || (n & (n - 1)) != 0 {
        return;
    }

    if bit_reverse_flag != 0 {
        bit_reversal(data, n);
    }

    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let angle = (if ifft_flag != 0 { 2.0 } else { -2.0 }) * core::f32::consts::PI / (len as f32);
        let w_step_re = angle.cos();
        let w_step_im = angle.sin();

        let mut i = 0;
        while i < n {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;

            for j in 0..half_len {
                let u_idx = 2 * (i + j);
                let v_idx = 2 * (i + j + half_len);

                let u_re = data[u_idx];
                let u_im = data[u_idx + 1];

                let v_re = data[v_idx];
                let v_im = data[v_idx + 1];

                let t_re = v_re * w_re - v_im * w_im;
                let t_im = v_re * w_im + v_im * w_re;

                data[u_idx] = u_re + t_re;
                data[u_idx + 1] = u_im + t_im;

                data[v_idx] = u_re - t_re;
                data[v_idx + 1] = u_im - t_im;

                let next_w_re = w_re * w_step_re - w_im * w_step_im;
                let next_w_im = w_re * w_step_im + w_im * w_step_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
            i += len;
        }
        len <<= 1;
    }

    if ifft_flag != 0 {
        let norm = 1.0 / (n as f32);
        for i in 0..(2 * n) {
            data[i] *= norm;
        }
    }
}

/// In-place Complex FFT for Q31 fixed-point.
pub fn cfft_q31(data: &mut [q31], n: usize, ifft_flag: u8, _bit_reverse_flag: u8) {
    if n < 2 { return; }
    // Convert to f32 scratch, run cfft_f32, convert back
    let mut scratch = [0.0f32; 1024];
    let total = 2 * n;
    if total > scratch.len() { return; }

    for i in 0..total {
        scratch[i] = data[i] as f32 / 2147483648.0;
    }
    cfft_f32(&mut scratch[..total], n, ifft_flag, 1);
    for i in 0..total {
        data[i] = (scratch[i] * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as q31;
    }
}

/// In-place Complex FFT for Q15 fixed-point.
pub fn cfft_q15(data: &mut [q15], n: usize, ifft_flag: u8, _bit_reverse_flag: u8) {
    if n < 2 { return; }
    let mut scratch = [0.0f32; 1024];
    let total = 2 * n;
    if total > scratch.len() { return; }

    for i in 0..total {
        scratch[i] = data[i] as f32 / 32768.0;
    }
    cfft_f32(&mut scratch[..total], n, ifft_flag, 1);
    for i in 0..total {
        data[i] = (scratch[i] * 32768.0).clamp(-32768.0, 32767.0) as q15;
    }
}

/// Real FFT for floating point 32-bit (`f32`).
/// `src` has `n` real samples. `dst` receives `2 * n` complex outputs.
pub fn rfft_f32(src: &[f32], dst: &mut [f32], n: usize, ifft_flag: u8) {
    let len = src.len().min(n);
    let mut c_data = [0.0f32; 1024];
    if 2 * len > c_data.len() || dst.len() < 2 * len { return; }

    for i in 0..len {
        c_data[2 * i] = src[i];
        c_data[2 * i + 1] = 0.0;
    }

    cfft_f32(&mut c_data[..2 * len], len, ifft_flag, 1);
    dst[..2 * len].copy_from_slice(&c_data[..2 * len]);
}

/// Real FFT for Q31 fixed-point.
pub fn rfft_q31(src: &[q31], dst: &mut [q31], n: usize, ifft_flag: u8) {
    let len = src.len().min(n);
    let mut c_data = [0; 1024];
    if 2 * len > c_data.len() || dst.len() < 2 * len { return; }

    for i in 0..len {
        c_data[2 * i] = src[i];
        c_data[2 * i + 1] = 0;
    }
    cfft_q31(&mut c_data[..2 * len], len, ifft_flag, 1);
    dst[..2 * len].copy_from_slice(&c_data[..2 * len]);
}

/// Real FFT for Q15 fixed-point.
pub fn rfft_q15(src: &[q15], dst: &mut [q15], n: usize, ifft_flag: u8) {
    let len = src.len().min(n);
    let mut c_data = [0; 1024];
    if 2 * len > c_data.len() || dst.len() < 2 * len { return; }

    for i in 0..len {
        c_data[2 * i] = src[i];
        c_data[2 * i + 1] = 0;
    }
    cfft_q15(&mut c_data[..2 * len], len, ifft_flag, 1);
    dst[..2 * len].copy_from_slice(&c_data[..2 * len]);
}

/// Discrete Cosine Transform Type IV (DCT-IV) for f32.
pub fn dct4_f32(src: &[f32], dst: &mut [f32], n: usize) {
    let len = src.len().min(dst.len()).min(n);
    let pi_over_n = core::f32::consts::PI / (len as f32);

    for k in 0..len {
        let mut sum = 0.0f32;
        let k_factor = (k as f32 + 0.5) * pi_over_n;
        for n_idx in 0..len {
            let angle = (n_idx as f32 + 0.5) * k_factor;
            sum += src[n_idx] * angle.cos();
        }
        let norm = (2.0 / len as f32).sqrt();
        dst[k] = sum * norm;
    }
}
