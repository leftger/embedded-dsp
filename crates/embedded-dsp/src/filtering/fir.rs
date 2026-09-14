//! Finite impulse response filters (FIR, overlap-scrap, circular buffer).

use crate::types::*;
// `fast_convolve_f32` and `FastFirF32` are `transform`-gated, so is their import.
#[cfg(feature = "transform")]
use crate::transform::cfft_f32;

// --- FIR Filter ---

/// Instance structure for the FIR filter, generic over the sample width.
///
/// Coefficients are stored in the sample's coefficient type ([`DspSample::Coeff`]) and the state in
/// the sample type. Accumulation runs through [`DspSample::mul_high`] — the per-term narrow the
/// fixed-point kernels use — so `f32`, `q15`, and `q31` share one loop.
pub struct FirInstance<'a, T: DspSample> {
    /// Number of filter taps.
    pub num_taps: u16,
    /// Filter coefficients.
    pub coeffs: &'a [T::Coeff],
    /// Filter state buffer.
    pub state: &'a mut [T],
}

impl<'a, T: DspSample> FirInstance<'a, T> {
    /// Initializes the instance.
    pub fn init(num_taps: u16, coeffs: &'a [T::Coeff], state: &'a mut [T]) -> Self {
        state.fill(T::ZERO);
        Self {
            num_taps,
            coeffs,
            state,
        }
    }
}

/// FIR filtering into `dst`, generic over the sample width.
pub fn fir<T: DspSample>(instance: &mut FirInstance<'_, T>, src: &[T], dst: &mut [T]) {
    let num_taps = instance.num_taps as usize;
    let block_size = src.len().min(dst.len());

    for i in 0..block_size {
        // Shift state
        for k in (1..num_taps).rev() {
            instance.state[k] = instance.state[k - 1];
        }
        instance.state[0] = src[i];

        // Compute the dot product with the per-term high product, exactly as the fixed-point
        // kernels did: `acc += (state * coeff) >> FRAC`.
        let mut acc = T::Accum::default();
        for k in 0..num_taps {
            acc = acc + T::mul_high(instance.state[k], instance.coeffs[k]);
        }
        dst[i] = T::from_accum_shifted(acc, 0);
    }
}



/// Streaming overlap-scrap FIR (`kiss_fastfir`): scrap at the tail of each
/// inverse FFT so consecutive hops overlap by `n_taps - 1` samples.
///
/// `NFFT` is the real FFT size. It must be a length [`cfft_f32`]
/// accepts (`<= 512` because convolution uses a stack scratch of 1024 floats),
/// and must be `>=` the impulse length. Hop size is `NFFT - n_taps + 1`.
///
/// History is primed with `n_taps - 1` zeros so the first hop aligns with
/// linear convolution (no extra delay). Call [`FastFirF32::flush`] after the
/// last input block to emit the filter tail.
#[cfg(feature = "transform")]
#[derive(Clone, Copy)]
pub struct FastFirF32<const NFFT: usize> {
    n_taps: usize,
    ngood: usize,
    fir_re: [f32; NFFT],
    fir_im: [f32; NFFT],
    pending: [f32; NFFT],
    pending_len: usize,
    spec_re: [f32; NFFT],
}

#[cfg(feature = "transform")]
impl<const NFFT: usize> FastFirF32<NFFT> {
    /// Builds a streaming FIR from a real impulse response.
    ///
    /// Returns `None` if `impulse` is empty, longer than `NFFT`, or `NFFT` is
    /// not a supported FFT length.
    pub fn new(impulse: &[f32]) -> Option<Self> {
        use crate::transform::cfft_f32_len_ok;
        if impulse.is_empty() || impulse.len() > NFFT || !cfft_f32_len_ok(NFFT) {
            return None;
        }

        let n_taps = impulse.len();
        let ngood = NFFT - n_taps + 1;
        let pending = [0.0f32; NFFT];
        let mut spec = [0.0f32; 1024];
        if 2 * NFFT > spec.len() {
            return None;
        }

        spec[0] = impulse[n_taps - 1];
        for i in 0..n_taps.saturating_sub(1) {
            spec[2 * (ngood + i)] = impulse[i];
        }
        cfft_f32(&mut spec[..2 * NFFT], NFFT, 0, 1);

        let mut fir_re = [0.0f32; NFFT];
        let mut fir_im = [0.0f32; NFFT];
        for i in 0..NFFT {
            fir_re[i] = spec[2 * i];
            fir_im[i] = spec[2 * i + 1];
        }

        Some(Self {
            n_taps,
            ngood,
            fir_re,
            fir_im,
            pending,
            pending_len: n_taps.saturating_sub(1),
            spec_re: [0.0f32; NFFT],
        })
    }

    /// Valid samples produced per full FFT hop (`NFFT - n_taps + 1`).
    #[inline]
    pub const fn ngood(&self) -> usize {
        self.ngood
    }

    /// Impulse length used at construction.
    #[inline]
    pub const fn n_taps(&self) -> usize {
        self.n_taps
    }

    fn convolve_pending(&mut self) {
        let mut spec = [0.0f32; 1024];
        for i in 0..NFFT {
            spec[2 * i] = self.pending[i];
        }
        cfft_f32(&mut spec[..2 * NFFT], NFFT, 0, 1);
        for i in 0..NFFT {
            let a = spec[2 * i];
            let b = spec[2 * i + 1];
            let c = self.fir_re[i];
            let d = self.fir_im[i];
            spec[2 * i] = a * c - b * d;
            spec[2 * i + 1] = a * d + b * c;
        }
        cfft_f32(&mut spec[..2 * NFFT], NFFT, 1, 1);
        for i in 0..NFFT {
            self.spec_re[i] = spec[2 * i];
        }
    }

    fn shift_scrap(&mut self) {
        let scrap = NFFT - self.ngood;
        for i in 0..scrap {
            self.pending[i] = self.pending[self.ngood + i];
        }
        self.pending_len = scrap;
    }

    /// Consumes `input` and writes as many hop-aligned outputs as fit in
    /// `output`. Returns the number of samples written.
    ///
    /// Provide `output.len() >= ngood` (ideally several hops) so full FFT
    /// blocks are not stalled for lack of output space.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let mut in_i = 0;
        let mut out_i = 0;
        loop {
            while self.pending_len < NFFT && in_i < input.len() {
                self.pending[self.pending_len] = input[in_i];
                self.pending_len += 1;
                in_i += 1;
            }
            if self.pending_len < NFFT || out_i + self.ngood > output.len() {
                break;
            }
            self.convolve_pending();
            output[out_i..out_i + self.ngood].copy_from_slice(&self.spec_re[..self.ngood]);
            out_i += self.ngood;
            self.shift_scrap();
        }
        out_i
    }

    /// Appends `n_taps - 1` zeros and drains a final padded hop so a finite
    /// input of length `L` yields the `L + n_taps - 1` linear-convolution samples.
    pub fn flush(&mut self, output: &mut [f32]) -> usize {
        let pad = self.n_taps.saturating_sub(1);
        let mut written = 0;
        let mut remaining_pad = pad;
        while remaining_pad > 0 && written < output.len() {
            let chunk = remaining_pad.min(32);
            let zeros = [0.0f32; 32];
            let n = self.process(&zeros[..chunk], &mut output[written..]);
            written += n;
            remaining_pad -= chunk;
            if n == 0 && self.pending_len < NFFT {
                break;
            }
        }

        if self.pending_len == 0 || written >= output.len() {
            return written;
        }

        let n = self.pending_len;
        let zpad = NFFT - n;
        for i in n..NFFT {
            self.pending[i] = 0.0;
        }
        self.pending_len = NFFT;
        let nout = self.ngood.saturating_sub(zpad);
        if nout == 0 || written + nout > output.len() {
            self.pending_len = n;
            return written;
        }
        self.convolve_pending();
        output[written..written + nout].copy_from_slice(&self.spec_re[..nout]);
        self.pending_len = 0;
        written + nout
    }

    /// Clears history back to `n_taps - 1` zeros.
    pub fn reset(&mut self) {
        self.pending.fill(0.0);
        self.pending_len = self.n_taps.saturating_sub(1);
    }
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
