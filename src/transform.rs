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
        let angle =
            (if ifft_flag != 0 { 2.0 } else { -2.0 }) * core::f32::consts::PI / (len as f32);
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
    if n < 2 {
        return;
    }
    // Convert to f32 scratch, run cfft_f32, convert back
    let mut scratch = [0.0f32; 1024];
    let total = 2 * n;
    if total > scratch.len() {
        return;
    }

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
    if n < 2 {
        return;
    }
    let mut scratch = [0.0f32; 1024];
    let total = 2 * n;
    if total > scratch.len() {
        return;
    }

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
    if 2 * len > c_data.len() || dst.len() < 2 * len {
        return;
    }

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
    if 2 * len > c_data.len() || dst.len() < 2 * len {
        return;
    }

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
    if 2 * len > c_data.len() || dst.len() < 2 * len {
        return;
    }

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

// --- Fast Walsh-Hadamard Transform (FWHT) ---

/// In-place Fast Walsh-Hadamard Transform (FWHT) for floating point `f32`.
///
/// `data.len()` must be a power of 2 (e.g. 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024).
pub fn fwht_f32(data: &mut [f32]) -> Status {
    let n = data.len();
    if n < 2 || (n & (n - 1)) != 0 {
        return Status::ArgumentError;
    }

    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..(i + h) {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }

    Status::Success
}

/// In-place Inverse Fast Walsh-Hadamard Transform (IFWHT) for floating point `f32` (normalized by $1/N$).
pub fn ifwht_f32(data: &mut [f32]) -> Status {
    let status = fwht_f32(data);
    if status != Status::Success {
        return status;
    }
    let norm = 1.0f32 / (data.len() as f32);
    for val in data.iter_mut() {
        *val *= norm;
    }
    Status::Success
}

/// In-place Fast Walsh-Hadamard Transform (FWHT) for 32-bit integers (`i32`).
pub fn fwht_i32(data: &mut [i32]) -> Status {
    let n = data.len();
    if n < 2 || (n & (n - 1)) != 0 {
        return Status::ArgumentError;
    }

    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..(i + h) {
                let x = data[j];
                let y = data[j + h];
                data[j] = x.wrapping_add(y);
                data[j + h] = x.wrapping_sub(y);
            }
            i += h * 2;
        }
        h *= 2;
    }

    Status::Success
}

// --- Haar Transform (Jörg Arndt, "Matters Computational", Ch. 24) ---

/// In-place, orthogonal Haar Transform for `f32`: an `O(n)` multiresolution transform using
/// only additions, subtractions, and a `sqrt(0.5)` scale factor per stage, with no
/// trigonometric factors at all (unlike the Fourier/Hartley transforms).
///
/// `data.len()` must be a power of 2 (e.g. 2, 4, 8, ..., 1024).
pub fn haar_transform_f32(data: &mut [f32]) -> Status {
    let n = data.len();
    if n < 2 || (n & (n - 1)) != 0 {
        return Status::ArgumentError;
    }

    let s2 = (0.5f32).sqrt();
    let mut v = 1.0f32;
    let mut js = 2;
    while js <= n {
        v *= s2;
        let half = js >> 1;
        let mut j = 0;
        while j < n {
            let t = j + half;
            let x = data[j];
            let y = data[t];
            data[j] = x + y;
            data[t] = (x - y) * v;
            j += js;
        }
        js <<= 1;
    }
    data[0] *= v; // v == 1 / sqrt(n)

    Status::Success
}

/// In-place Inverse Haar Transform for `f32`, undoing [`haar_transform_f32`].
pub fn inverse_haar_transform_f32(data: &mut [f32]) -> Status {
    let n = data.len();
    if n < 2 || (n & (n - 1)) != 0 {
        return Status::ArgumentError;
    }

    let s2 = 2.0f32.sqrt();
    let mut v = 1.0f32 / (n as f32).sqrt();
    data[0] *= v;

    let mut js = n;
    while js >= 2 {
        let half = js >> 1;
        let mut j = 0;
        while j < n {
            let t = j + half;
            let x = data[j];
            let y = data[t] * v;
            data[j] = x + y;
            data[t] = x - y;
            j += js;
        }
        v *= s2;
        js >>= 1;
    }

    Status::Success
}

/// In-place, non-normalized Haar Transform for `i32`: a forward-only, integer-exact
/// decomposition using only wrapping add/subtract (no scaling), analogous to
/// [`fwht_i32`]. Because the transform is non-normalized, an exact-integer inverse does not
/// exist in general (undoing it requires dividing by powers of 2 that may not evenly divide
/// intermediate sums); use [`haar_transform_f32`] / [`inverse_haar_transform_f32`] when an
/// invertible round trip is required.
///
/// `data.len()` must be a power of 2 (e.g. 2, 4, 8, ..., 1024).
pub fn haar_transform_i32(data: &mut [i32]) -> Status {
    let n = data.len();
    if n < 2 || (n & (n - 1)) != 0 {
        return Status::ArgumentError;
    }

    let mut js = 2;
    while js <= n {
        let half = js >> 1;
        let mut j = 0;
        while j < n {
            let t = j + half;
            let x = data[j];
            let y = data[t];
            data[j] = x.wrapping_add(y);
            data[t] = x.wrapping_sub(y);
            j += js;
        }
        js <<= 1;
    }

    Status::Success
}

// --- Hartley Transform (Jörg Arndt, "Matters Computational", Ch. 25) ---

/// In-place Discrete Hartley Transform for `f32`.
///
/// Computed via the identity relating the Hartley and Fourier transforms (Ch. 25):
/// `H[a] = (Re(F[a]) - Im(F[a])) / sqrt(n)`, built on top of [`cfft_f32`] rather than a
/// dedicated real-only butterfly network, so it costs a full complex FFT internally
/// (`n <= 512`) even though its inputs and outputs are purely real.
///
/// The Hartley transform is its own inverse (`H[H[a]] = a`): call this function a second time
/// on its output to invert it, with no separate inverse routine needed.
///
/// `data.len()` must be a power of 2 (e.g. 2, 4, 8, ..., 512).
pub fn hartley_transform_f32(data: &mut [f32]) -> Status {
    let n = data.len();
    if n < 2 || (n & (n - 1)) != 0 {
        return Status::ArgumentError;
    }
    if 2 * n > 1024 {
        return Status::LengthError;
    }

    let mut c_data = [0.0f32; 1024];
    for i in 0..n {
        c_data[2 * i] = data[i];
        c_data[2 * i + 1] = 0.0;
    }

    cfft_f32(&mut c_data[..2 * n], n, 0, 1);

    let inv_sqrt_n = 1.0 / (n as f32).sqrt();
    for i in 0..n {
        data[i] = (c_data[2 * i] - c_data[2 * i + 1]) * inv_sqrt_n;
    }

    Status::Success
}

// --- Generalized Wavelet Transform (Jörg Arndt, "Matters Computational", Ch. 27) ---

/// The Daubechies-4 orthogonal wavelet low-pass filter taps (Ch. 27.1), verified to satisfy
/// the wavelet conditions `sum(h_j^2) = 1` and `sum(h_j * h_{j+2}) = 0`. Using
/// `[sqrt(0.5), sqrt(0.5)]` instead recovers the Haar wavelet as a special case.
pub const DAUBECHIES_4: [f32; 4] = [0.482_962_9, 0.836_516_3, 0.224_143_87, -0.129_409_52];

/// The high-pass filter tap derived from low-pass filter `h` (Ch. 27.1, Eq. 27.1-2):
/// `g[k] = (-1)^k * h[n - 1 - k]`.
#[inline(always)]
fn wavelet_high_pass_tap(h: &[f32], k: usize) -> f32 {
    let v = h[h.len() - 1 - k];
    if k % 2 == 0 { v } else { -v }
}

/// Performs one level of a fast wavelet transform step on the first `m` elements of `data`,
/// using wavelet filter `h` (low-pass) and its derived high-pass filter. Writes the low-pass
/// ("scaling") coefficients to `data[0..m/2]` and the high-pass ("wavelet") coefficients to
/// `data[m/2..m]`; the underlying convolution wraps around cyclically at the block boundary.
///
/// `m` must be a power of 2; `h.len()` must be even and `<= m`.
pub fn wavelet_step_f32(data: &mut [f32], m: usize, h: &[f32]) -> Status {
    let taps = h.len();
    if m < 2 || (m & (m - 1)) != 0 || taps == 0 || taps % 2 != 0 || taps > m || data.len() < m {
        return Status::ArgumentError;
    }
    if m > 1024 {
        return Status::LengthError;
    }

    let mut scratch = [0.0f32; 1024];
    let nh = m >> 1;
    let mut i = 0;
    while i < m {
        let mut s = 0.0f32;
        let mut d = 0.0f32;
        for k in 0..taps {
            let idx = (i + k) % m;
            let x = data[idx];
            s += h[k] * x;
            d += wavelet_high_pass_tap(h, k) * x;
        }
        let j = i / 2;
        scratch[j] = s;
        scratch[nh + j] = d;
        i += 2;
    }
    data[..m].copy_from_slice(&scratch[..m]);

    Status::Success
}

/// Performs the exact inverse of one [`wavelet_step_f32`] level.
///
/// `m` must be a power of 2; `h.len()` must be even and `<= m`.
pub fn inverse_wavelet_step_f32(data: &mut [f32], m: usize, h: &[f32]) -> Status {
    let taps = h.len();
    if m < 2 || (m & (m - 1)) != 0 || taps == 0 || taps % 2 != 0 || taps > m || data.len() < m {
        return Status::ArgumentError;
    }
    if m > 1024 {
        return Status::LengthError;
    }

    let mut scratch = [0.0f32; 1024];
    let nh = m >> 1;
    for j in 0..nh {
        let s = data[j];
        let d = data[nh + j];
        for k in 0..taps {
            let idx = (2 * j + k) % m;
            scratch[idx] += h[k] * s + wavelet_high_pass_tap(h, k) * d;
        }
    }
    data[..m].copy_from_slice(&scratch[..m]);

    Status::Success
}

/// Performs a full multi-level fast wavelet transform (Ch. 27): repeatedly applies
/// [`wavelet_step_f32`] to the lower half of the array, halving the active block length each
/// time, stopping once the block would be smaller than the filter itself (mirroring the Haar
/// transform's pyramid structure).
///
/// `data.len()` must be a power of 2 and `>= h.len()`.
pub fn wavelet_transform_f32(data: &mut [f32], h: &[f32]) -> Status {
    let n = data.len();
    if n < 2 || (n & (n - 1)) != 0 || h.len() > n {
        return Status::ArgumentError;
    }

    let mut m = n;
    while m >= h.len() {
        let status = wavelet_step_f32(&mut data[..m], m, h);
        if status != Status::Success {
            return status;
        }
        m >>= 1;
    }

    Status::Success
}

/// Performs the exact inverse of [`wavelet_transform_f32`].
///
/// `data.len()` must be a power of 2 and `>= h.len()`.
pub fn inverse_wavelet_transform_f32(data: &mut [f32], h: &[f32]) -> Status {
    let n = data.len();
    if n < 2 || (n & (n - 1)) != 0 || h.len() > n {
        return Status::ArgumentError;
    }

    let mut smallest = n;
    while smallest >= h.len() {
        smallest >>= 1;
    }
    smallest <<= 1;

    let mut m = smallest;
    while m <= n {
        let status = inverse_wavelet_step_f32(&mut data[..m], m, h);
        if status != Status::Success {
            return status;
        }
        m <<= 1;
    }

    Status::Success
}
