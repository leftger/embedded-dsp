//! Digital filtering functions (FIR, Biquad IIR Direct Form I & II, LMS Adaptive Filter, Convolution, Correlation).

use crate::types::*;

// --- FIR Filter ---

/// Instance structure for the floating-point FIR filter.
pub struct FirInstanceF32<'a> {
    pub num_taps: u16,
    pub coeffs: &'a [f32],
    pub state: &'a mut [f32],
}

impl<'a> FirInstanceF32<'a> {
    pub fn init(num_taps: u16, coeffs: &'a [f32], state: &'a mut [f32]) -> Self {
        state.fill(0.0);
        Self {
            num_taps,
            coeffs,
            state,
        }
    }
}

pub fn fir_f32(instance: &mut FirInstanceF32, src: &[f32], dst: &mut [f32]) {
    let num_taps = instance.num_taps as usize;
    let block_size = src.len().min(dst.len());

    for i in 0..block_size {
        // Shift state
        for k in (1..num_taps).rev() {
            instance.state[k] = instance.state[k - 1];
        }
        instance.state[0] = src[i];

        // Compute dot product with coefficients
        let mut acc = 0.0f32;
        for k in 0..num_taps {
            acc += instance.state[k] * instance.coeffs[k];
        }
        dst[i] = acc;
    }
}

/// Instance structure for the Q31 FIR filter.
pub struct FirInstanceQ31<'a> {
    pub num_taps: u16,
    pub coeffs: &'a [q31],
    pub state: &'a mut [q31],
}

impl<'a> FirInstanceQ31<'a> {
    pub fn init(num_taps: u16, coeffs: &'a [q31], state: &'a mut [q31]) -> Self {
        state.fill(0);
        Self {
            num_taps,
            coeffs,
            state,
        }
    }
}

pub fn fir_q31(instance: &mut FirInstanceQ31, src: &[q31], dst: &mut [q31]) {
    let num_taps = instance.num_taps as usize;
    let block_size = src.len().min(dst.len());

    for i in 0..block_size {
        for k in (1..num_taps).rev() {
            instance.state[k] = instance.state[k - 1];
        }
        instance.state[0] = src[i];

        let mut acc: i64 = 0;
        for k in 0..num_taps {
            acc += (instance.state[k] as i64 * instance.coeffs[k] as i64) >> 31;
        }
        dst[i] = acc.clamp(i32::MIN as i64, i32::MAX as i64) as q31;
    }
}

/// Instance structure for the Q15 FIR filter.
pub struct FirInstanceQ15<'a> {
    pub num_taps: u16,
    pub coeffs: &'a [q15],
    pub state: &'a mut [q15],
}

impl<'a> FirInstanceQ15<'a> {
    pub fn init(num_taps: u16, coeffs: &'a [q15], state: &'a mut [q15]) -> Self {
        state.fill(0);
        Self {
            num_taps,
            coeffs,
            state,
        }
    }
}

pub fn fir_q15(instance: &mut FirInstanceQ15, src: &[q15], dst: &mut [q15]) {
    let num_taps = instance.num_taps as usize;
    let block_size = src.len().min(dst.len());

    for i in 0..block_size {
        for k in (1..num_taps).rev() {
            instance.state[k] = instance.state[k - 1];
        }
        instance.state[0] = src[i];

        let mut acc: i32 = 0;
        for k in 0..num_taps {
            acc += (instance.state[k] as i32 * instance.coeffs[k] as i32) >> 15;
        }
        dst[i] = acc.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
    }
}

// --- Biquad Cascade Direct Form I Filter ---

/// Instance structure for the floating-point Biquad Cascade Direct Form I filter.
pub struct BiquadCascadeInstanceF32<'a> {
    pub num_stages: u8,
    pub coeffs: &'a [f32],    // 5 * num_stages: [b0, b1, b2, a1, a2]
    pub state: &'a mut [f32], // 4 * num_stages: [x[n-1], x[n-2], y[n-1], y[n-2]]
}

impl<'a> BiquadCascadeInstanceF32<'a> {
    pub fn init(num_stages: u8, coeffs: &'a [f32], state: &'a mut [f32]) -> Self {
        state.fill(0.0);
        Self {
            num_stages,
            coeffs,
            state,
        }
    }
}

pub fn biquad_cascade_df1_f32(
    instance: &mut BiquadCascadeInstanceF32,
    src: &[f32],
    dst: &mut [f32],
) {
    let num_stages = instance.num_stages as usize;
    let block_size = src.len().min(dst.len());

    let mut in_val;
    let mut out_val;

    for i in 0..block_size {
        in_val = src[i];
        for stage in 0..num_stages {
            let b0 = instance.coeffs[stage * 5];
            let b1 = instance.coeffs[stage * 5 + 1];
            let b2 = instance.coeffs[stage * 5 + 2];
            let a1 = instance.coeffs[stage * 5 + 3];
            let a2 = instance.coeffs[stage * 5 + 4];

            let x1 = instance.state[stage * 4];
            let x2 = instance.state[stage * 4 + 1];
            let y1 = instance.state[stage * 4 + 2];
            let y2 = instance.state[stage * 4 + 3];

            out_val = b0 * in_val + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;

            instance.state[stage * 4 + 1] = x1;
            instance.state[stage * 4] = in_val;
            instance.state[stage * 4 + 3] = y1;
            instance.state[stage * 4 + 2] = out_val;

            in_val = out_val;
        }
        dst[i] = in_val;
    }
}

// --- LMS Adaptive Filter ---

/// Instance structure for the floating-point LMS adaptive filter.
pub struct LmsInstanceF32<'a> {
    pub num_taps: u16,
    pub coeffs: &'a mut [f32],
    pub state: &'a mut [f32],
    pub mu: f32,
}

impl<'a> LmsInstanceF32<'a> {
    pub fn init(num_taps: u16, coeffs: &'a mut [f32], state: &'a mut [f32], mu: f32) -> Self {
        state.fill(0.0);
        coeffs.fill(0.0);
        Self {
            num_taps,
            coeffs,
            state,
            mu,
        }
    }
}

pub fn lms_f32(
    instance: &mut LmsInstanceF32,
    src: &[f32],
    ref_signal: &[f32],
    out: &mut [f32],
    err: &mut [f32],
) {
    let num_taps = instance.num_taps as usize;
    let block_size = src
        .len()
        .min(ref_signal.len())
        .min(out.len())
        .min(err.len());

    for i in 0..block_size {
        for k in (1..num_taps).rev() {
            instance.state[k] = instance.state[k - 1];
        }
        instance.state[0] = src[i];

        let mut acc = 0.0f32;
        for k in 0..num_taps {
            acc += instance.state[k] * instance.coeffs[k];
        }
        out[i] = acc;
        let e = ref_signal[i] - acc;
        err[i] = e;

        // Update coefficients: w[n+1] = w[n] + 2 * mu * e[n] * x[n]
        let alpha = 2.0 * instance.mu * e;
        for k in 0..num_taps {
            instance.coeffs[k] += alpha * instance.state[k];
        }
    }
}

// --- Convolution ---

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

pub fn conv_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i64 = 0;
        let k_min = if n >= len_b - 1 { n - (len_b - 1) } else { 0 };
        let k_max = n.min(len_a - 1);
        for k in k_min..=k_max {
            acc += (src_a[k] as i64 * src_b[n - k] as i64) >> 31;
        }
        dst[n] = acc.clamp(i32::MIN as i64, i32::MAX as i64) as q31;
    }
}

pub fn conv_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i32 = 0;
        let k_min = if n >= len_b - 1 { n - (len_b - 1) } else { 0 };
        let k_max = n.min(len_a - 1);
        for k in k_min..=k_max {
            acc += (src_a[k] as i32 * src_b[n - k] as i32) >> 15;
        }
        dst[n] = acc.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
    }
}

pub fn conv_q7(src_a: &[q7], src_b: &[q7], dst: &mut [q7]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i32 = 0;
        let k_min = if n >= len_b - 1 { n - (len_b - 1) } else { 0 };
        let k_max = n.min(len_a - 1);
        for k in k_min..=k_max {
            acc += (src_a[k] as i32 * src_b[n - k] as i32) >> 7;
        }
        dst[n] = acc.clamp(i8::MIN as i32, i8::MAX as i32) as q7;
    }
}

// --- Correlation ---

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

pub fn correlate_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i64 = 0;
        for k in 0..len_a {
            let idx_b = (k as isize) + (len_b as isize - 1) - (n as isize);
            if idx_b >= 0 && (idx_b as usize) < len_b {
                acc += (src_a[k] as i64 * src_b[idx_b as usize] as i64) >> 31;
            }
        }
        dst[n] = acc.clamp(i32::MIN as i64, i32::MAX as i64) as q31;
    }
}

pub fn correlate_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let len_a = src_a.len();
    let len_b = src_b.len();
    let out_len = (len_a + len_b - 1).min(dst.len());

    for n in 0..out_len {
        let mut acc: i32 = 0;
        for k in 0..len_a {
            let idx_b = (k as isize) + (len_b as isize - 1) - (n as isize);
            if idx_b >= 0 && (idx_b as usize) < len_b {
                acc += (src_a[k] as i32 * src_b[idx_b as usize] as i32) >> 15;
            }
        }
        dst[n] = acc.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
    }
}

// --- Non-linear Filtering (Median & Conditional Median) ---

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::transform::cfft_f32;

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
    if window_len == 0 || window_len % 2 == 0 || window_len > 63 {
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
    if window_len == 0 || window_len % 2 == 0 || window_len > 63 {
        return Status::ArgumentError;
    }

    let half = window_len / 2;
    let mut sort_buf = [0i16; 64];

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
        let diff = (center as i32 - med as i32).abs();
        if diff >= threshold as i32 {
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
    if window_len == 0 || window_len % 2 == 0 || window_len > 63 {
        return Status::ArgumentError;
    }

    let half = window_len / 2;
    let mut sort_buf = [0i32; 64];

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
        let diff = (center as i64 - med as i64).abs();
        if diff >= threshold as i64 {
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

// --- Real-time Circular Buffer & Delay Line ---

/// Const-generic zero-allocation circular buffer and delay line for real-time DSP sample streams.
#[derive(Debug, Clone, Copy)]
pub struct CircularBuffer<T, const N: usize> {
    buffer: [T; N],
    head: usize,
    count: usize,
}

impl<T: Copy, const N: usize> CircularBuffer<T, N> {
    /// Creates a new circular buffer initialized with `init_val`.
    pub const fn new(init_val: T) -> Self {
        Self {
            buffer: [init_val; N],
            head: 0,
            count: 0,
        }
    }

    /// Pushes a new sample into the buffer, overwriting the oldest sample when full.
    #[inline(always)]
    pub fn push(&mut self, sample: T) {
        if N == 0 {
            return;
        }
        self.buffer[self.head] = sample;
        self.head = (self.head + 1) % N;
        if self.count < N {
            self.count += 1;
        }
    }

    /// Gets sample with historical lag $k$, where $k = 0$ is the newest sample (`x[n]`), $k = 1$ is `x[n-1]`, etc.
    /// Returns `None` if `lag >= self.len()`.
    #[inline(always)]
    pub fn get(&self, lag: usize) -> Option<T> {
        if lag >= self.count || N == 0 {
            return None;
        }
        let idx = (self.head + N - 1 - (lag % N)) % N;
        Some(self.buffer[idx])
    }

    /// Returns the most recently pushed sample (`x[n]`).
    #[inline(always)]
    pub fn latest(&self) -> Option<T> {
        self.get(0)
    }

    /// Returns the oldest sample stored in the buffer.
    #[inline(always)]
    pub fn oldest(&self) -> Option<T> {
        if self.count == 0 {
            None
        } else {
            self.get(self.count - 1)
        }
    }

    /// Returns the number of valid samples currently stored in the buffer.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns the capacity of the circular buffer (`N`).
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns `true` if the buffer contains no samples.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns `true` if the buffer is filled to capacity `N`.
    #[inline(always)]
    pub const fn is_full(&self) -> bool {
        self.count == N
    }

    /// Clears the circular buffer, resetting sample count and filling with `reset_val`.
    pub fn clear(&mut self, reset_val: T) {
        self.buffer = [reset_val; N];
        self.head = 0;
        self.count = 0;
    }
}
