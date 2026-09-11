//! Multi-rate digital signal processing routines: Cascaded Integrator-Comb (CIC) decimation/interpolation and linear fractional resampling.

/// Cascaded Integrator-Comb (CIC) Decimator for downsampling signals in integer arithmetic.
pub struct CicDecimator<const STAGES: usize> {
    r: usize, // Decimation factor
    integrator_state: [i32; STAGES],
    comb_state: [i32; STAGES],
    sample_counter: usize,
}

impl<const STAGES: usize> CicDecimator<STAGES> {
    /// Initialise a new CIC decimator with decimation factor `r`.
    pub fn new(r: usize) -> Self {
        Self {
            r,
            integrator_state: [0; STAGES],
            comb_state: [0; STAGES],
            sample_counter: 0,
        }
    }

    /// Theoretical maximum DC gain: `R^STAGES`.
    pub fn gain(&self) -> u64 {
        let mut g: u64 = 1;
        for _ in 0..STAGES {
            g = g.saturating_mul(self.r as u64);
        }
        g
    }

    /// Number of bits of bit-growth: `ceil(log2(R^STAGES))`.
    pub fn gain_bits(&self) -> u32 {
        let g = self.gain();
        if g <= 1 {
            0
        } else {
            64 - (g - 1).leading_zeros()
        }
    }

    /// Process an input sample. Returns `Some(decimated_sample)` every `R` samples.
    pub fn process_sample(&mut self, input: i32) -> Option<i32> {
        // Integrator stages running at high sample rate
        let mut val = input;
        for i in 0..STAGES {
            self.integrator_state[i] = self.integrator_state[i].wrapping_add(val);
            val = self.integrator_state[i];
        }

        self.sample_counter += 1;
        if self.sample_counter >= self.r {
            self.sample_counter = 0;

            // Comb stages running at low sample rate
            for i in 0..STAGES {
                let diff = val.wrapping_sub(self.comb_state[i]);
                self.comb_state[i] = val;
                val = diff;
            }
            Some(val)
        } else {
            None
        }
    }

    /// Process an input sample and normalize output by bit-growth right-shift to prevent overflow.
    pub fn process_sample_scaled(&mut self, input: i32) -> Option<i32> {
        self.process_sample(input).map(|out| {
            let shift = self.gain_bits();
            if shift > 0 {
                out >> shift
            } else {
                out
            }
        })
    }
}

/// Cascaded Integrator-Comb (CIC) Interpolator for upsampling signals in integer arithmetic.
pub struct CicInterpolator<const STAGES: usize> {
    r: usize, // Interpolation factor
    comb_state: [i32; STAGES],
    integrator_state: [i32; STAGES],
}

impl<const STAGES: usize> CicInterpolator<STAGES> {
    /// Initialise a new CIC interpolator with interpolation factor `r`.
    pub fn new(r: usize) -> Self {
        Self {
            r,
            comb_state: [0; STAGES],
            integrator_state: [0; STAGES],
        }
    }

    /// Theoretical maximum DC gain: `R^(STAGES - 1)`.
    pub fn gain(&self) -> u64 {
        if STAGES <= 1 {
            return 1;
        }
        let mut g: u64 = 1;
        for _ in 0..(STAGES - 1) {
            g = g.saturating_mul(self.r as u64);
        }
        g
    }

    /// Number of bits of bit-growth: `ceil(log2(gain))`.
    pub fn gain_bits(&self) -> u32 {
        let g = self.gain();
        if g <= 1 {
            0
        } else {
            64 - (g - 1).leading_zeros()
        }
    }

    /// Process a single input sample and populate `out_buf` with `R` interpolated output samples.
    pub fn process_sample(&mut self, input: i32, out_buf: &mut [i32]) {
        assert!(
            out_buf.len() >= self.r,
            "out_buf must hold at least R samples"
        );

        // Comb stages at low rate
        let mut val = input;
        for i in 0..STAGES {
            let diff = val.wrapping_sub(self.comb_state[i]);
            self.comb_state[i] = val;
            val = diff;
        }

        // Zero stuffing and integrator stages at high rate
        for step in 0..self.r {
            let in_step = if step == 0 { val } else { 0 };
            let mut stage_val = in_step;

            for i in 0..STAGES {
                self.integrator_state[i] = self.integrator_state[i].wrapping_add(stage_val);
                stage_val = self.integrator_state[i];
            }

            out_buf[step] = stage_val;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic CIC Filter (ported from idsp)
// ─────────────────────────────────────────────────────────────────────────────

/// Generic Cascaded Integrator-Comb (CIC) filter.
///
/// - `T`: sample type (e.g. `i32`, `i64`). Must be `Copy + Default + Add + Sub`.
/// - `N`: filter order (1 = linear, 2 = quadratic, 3 = cubic).
/// - `M`: comb delay (usually 1; larger values create a unit-rate lowpass).
///
/// Use [`CicFilter::process_interpolate`] for upsampling: call with `Some(x)` at
/// the slow input rate and `None` every other cycle. Use
/// [`CicFilter::process_decimate`] for downsampling: call with every high-rate
/// input; returns `Some(y)` at the slow output rate.
///
/// # Type Aliases
/// [`CicDec3`] and [`CicInt3`] are convenient aliases for a 3rd-order, M=1 filter.
#[derive(Clone, Debug)]
pub struct CicFilter<T, const N: usize, const M: usize = 1> {
    /// Rate change (fast/slow - 1 for decimator; slow/fast - 1 for interpolator)
    rate: u32,
    /// Up/down-sampler state (counts down to 0)
    index: u32,
    /// Zero-order hold behind comb sections
    zoh: T,
    /// Comb / differentiator state
    combs: [[T; M]; N],
    /// Integrator state
    integrators: [T; N],
}

impl<T, const N: usize, const M: usize> CicFilter<T, N, M>
where
    T: Copy + Default + core::ops::Add<Output = T> + core::ops::Sub<Output = T>,
{
    /// Create a new zero-initialised CIC filter with the given rate change.
    ///
    /// `rate` is `fast/slow - 1` (decimator) or `slow/fast - 1` (interpolator).
    pub fn new(rate: u32) -> Self {
        // SAFETY: T: Default so zeroing is equivalent to calling Default::default()
        // for each element; we use unsafe here only to avoid the constraint
        // `T: Copy` being required for array-repeat expressions with const generics.
        let combs: [[T; M]; N] = core::array::from_fn(|_| core::array::from_fn(|_| T::default()));
        let integrators: [T; N] = core::array::from_fn(|_| T::default());
        Self {
            rate,
            index: 0,
            zoh: T::default(),
            combs,
            integrators,
        }
    }

    /// Filter order.
    pub const fn order(&self) -> usize { N }

    /// Comb delay.
    pub const fn comb_delay(&self) -> usize { M }

    /// Rate change.
    pub const fn rate(&self) -> u32 { self.rate }

    /// Update the rate change.
    pub fn set_rate(&mut self, rate: u32) { self.rate = rate; }

    /// Reset all filter state.
    pub fn clear(&mut self) { *self = Self::new(self.rate); }

    /// Returns `true` when a new slow-rate sample is expected (interpolator)
    /// or a new slow-rate output is available (decimator).
    pub const fn tick(&self) -> bool { self.index == 0 }

    /// Current interpolator output (last integrator value).
    pub fn get_interpolate(&self) -> T {
        *self.integrators.last().unwrap_or(&self.zoh)
    }

    /// Current decimator output (value captured at last decimation tick).
    pub fn get_decimate(&self) -> T { self.zoh }

    /// **Interpolator**: optionally ingest a new low-rate sample and return the
    /// next output.
    ///
    /// Supply `Some(x)` when [`CicFilter::tick`] is `true` (slow input rate),
    /// otherwise supply `None`.
    pub fn process_interpolate(&mut self, x: Option<T>) -> T {
        if let Some(x) = x {
            // Comb stages at slow rate
            self.index = self.rate;
            self.zoh = self.combs.iter_mut().fold(x, |acc, c| {
                let y = acc - c[0];
                c.copy_within(1.., 0);
                c[M - 1] = acc;
                y
            });
        } else {
            self.index -= 1;
        }
        // Integrators run every cycle at the fast rate
        self.integrators.iter_mut().fold(self.zoh, |acc, i| {
            *i = *i + acc;
            *i
        })
    }

    /// **Decimator**: ingest a new high-rate sample and optionally return the
    /// decimated output.
    ///
    /// Returns `Some(y)` at the slow rate (every `rate + 1` calls).
    pub fn process_decimate(&mut self, x: T) -> Option<T>
    where
        T: core::ops::AddAssign,
    {
        // Integrators run at high rate
        let x = self.integrators.iter_mut().fold(x, |acc, i| {
            *i = *i + acc;
            *i
        });
        if let Some(index) = self.index.checked_sub(1) {
            self.index = index;
            None
        } else {
            self.index = self.rate;
            // Comb stages at slow rate
            self.zoh = self.combs.iter_mut().fold(x, |acc, c| {
                let y = acc - c[0];
                c.copy_within(1.., 0);
                c[M - 1] = acc;
                y
            });
            Some(self.zoh)
        }
    }
}

/// Third-order CIC filter with M=1 (typical for decimation).
pub type CicDec3<T> = CicFilter<T, 3, 1>;
/// Third-order CIC filter with M=1 (typical for interpolation).
pub type CicInt3<T> = CicFilter<T, 3, 1>;

// ─────────────────────────────────────────────────────────────────────────────
// Polyphase & Linear Resampling in Q15
// ─────────────────────────────────────────────────────────────────────────────

use crate::types::q15;

/// Polyphase FIR decimation by integer factor `M`.
///
/// Filters and downsamples `src` by factor `M` (`decimation_factor`).
/// `coeffs` is the prototype FIR filter kernel (length typically multiple of `M`).
/// Returns the number of output samples written to `dst`.
pub fn polyphase_decimate_q15(
    src: &[q15],
    coeffs: &[q15],
    decimation_factor: usize,
    dst: &mut [q15],
) -> usize {
    if decimation_factor == 0 || coeffs.is_empty() || src.is_empty() {
        return 0;
    }
    let num_taps = coeffs.len();
    let out_len = dst.len().min(if src.len() >= num_taps { (src.len() - num_taps) / decimation_factor + 1 } else { 0 });

    for i in 0..out_len {
        let src_offset = i * decimation_factor;
        let mut acc: i64 = 0;
        for k in 0..num_taps {
            acc += (src[src_offset + k].to_bits() as i64 * coeffs[k].to_bits() as i64) >> 15;
        }
        dst[i] = q15::from_bits(acc.clamp(i16::MIN as i64, i16::MAX as i64) as i16);
    }

    out_len
}

/// Polyphase FIR interpolation by integer factor `L`.
///
/// Upsamples `src` by factor `L` (`interpolation_factor`) using polyphase decomposition.
/// `coeffs` length must be a multiple of `L`.
/// Returns the number of output samples written to `dst`.
pub fn polyphase_interpolate_q15(
    src: &[q15],
    coeffs: &[q15],
    interpolation_factor: usize,
    dst: &mut [q15],
) -> usize {
    let l = interpolation_factor;
    if l == 0 || coeffs.is_empty() || src.is_empty() || coeffs.len() % l != 0 {
        return 0;
    }
    let taps_per_phase = coeffs.len() / l;
    let max_in_samples = if src.len() >= taps_per_phase { src.len() - taps_per_phase + 1 } else { 0 };
    let out_len = dst.len().min(max_in_samples * l);

    for in_idx in 0..max_in_samples {
        for phase in 0..l {
            let out_idx = in_idx * l + phase;
            if out_idx >= dst.len() {
                break;
            }
            let mut acc: i64 = 0;
            for k in 0..taps_per_phase {
                let coeff = coeffs[k * l + phase].to_bits() as i64;
                let sample = src[in_idx + k].to_bits() as i64;
                acc += (sample * coeff) >> 15;
            }
            dst[out_idx] =
                q15::from_bits((acc * l as i64).clamp(i16::MIN as i64, i16::MAX as i64) as i16);
        }
    }

    out_len
}

/// Linear fractional resampler in Q15.
/// `ratio_q16` is `(src_sample_rate / dst_sample_rate)` in Q16.16 format.
pub fn resample_linear_q15(src: &[q15], dst: &mut [q15], ratio_q16: i32) {
    if src.is_empty() || dst.is_empty() || ratio_q16 <= 0 {
        return;
    }

    let mut phase_acc: i64 = 0;
    for i in 0..dst.len() {
        let idx0 = (phase_acc >> 16) as usize;
        let frac = (phase_acc & 0xFFFF) as i32; // [0, 65535]

        if idx0 >= src.len() {
            dst[i] = src[src.len() - 1];
        } else {
            let s0 = src[idx0].to_bits() as i32;
            let s1 = if idx0 + 1 < src.len() {
                src[idx0 + 1].to_bits() as i32
            } else {
                s0
            };
            let diff = s1 - s0;
            let interp = s0 + ((diff * frac) >> 16);
            dst[i] = q15::from_bits(interp.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
        }

        phase_acc += ratio_q16 as i64;
    }
}

/// Linear fractional resampler.
/// Resamples `src` into `dst` according to `ratio` (`src_sample_rate / dst_sample_rate`).
pub fn resample_linear_f32(src: &[f32], dst: &mut [f32], ratio: f32) {
    if src.is_empty() || dst.is_empty() || ratio <= 0.0 {
        return;
    }

    for i in 0..dst.len() {
        let src_idx_float = i as f32 * ratio;
        let idx0 = src_idx_float as usize;
        let idx1 = (idx0 + 1).min(src.len() - 1);

        if idx0 >= src.len() {
            dst[i] = src[src.len() - 1];
            continue;
        }

        let frac = src_idx_float - idx0 as f32;
        dst[i] = src[idx0] * (1.0 - frac) + src[idx1] * frac;
    }
}

#[cfg(feature = "transform")]
use crate::transform::cfft_f32;
#[cfg(feature = "transform")]
use crate::types::Status;

/// Spectral (Sinc) 2:1 Interpolator using frequency-domain zero-padding via FFT/IFFT.
///
/// `src` length must be a power of 2 (e.g. 16, 32, 64, 128, 256).
/// `dst` must have length at least `2 * src.len()`.
///
/// Requires the `transform` feature (enabled by `full`).
#[cfg(feature = "transform")]
pub fn spectral_interpolate_2x_f32(src: &[f32], dst: &mut [f32]) -> Status {
    let n = src.len();
    if n < 4 || (n & (n - 1)) != 0 {
        return Status::ArgumentError;
    }
    if dst.len() < 2 * n {
        return Status::LengthError;
    }
    if 4 * n > 1024 {
        return Status::LengthError; // Max scratch size limit (256-pt input -> 512-pt complex)
    }

    let mut c_buf = [0.0f32; 1024];

    // Copy src into complex array (size 2 * 2n)
    for i in 0..n {
        c_buf[2 * i] = src[i];
        c_buf[2 * i + 1] = 0.0;
    }

    // FFT of size n
    cfft_f32(&mut c_buf[..2 * n], n, 0, 1);

    // Half Nyquist component
    let nyq_re = 0.5 * c_buf[n];
    let nyq_im = 0.5 * c_buf[n + 1];
    c_buf[n] = nyq_re;
    c_buf[n + 1] = nyq_im;

    // Shift negative frequencies to upper half and zero middle
    let mut expanded = [0.0f32; 1024];
    // Copy 0..=N/2
    for i in 0..=(n / 2) {
        expanded[2 * i] = c_buf[2 * i];
        expanded[2 * i + 1] = c_buf[2 * i + 1];
    }
    // Nyquist conjugate mirror at 3N/2
    expanded[2 * (3 * n / 2)] = nyq_re;
    expanded[2 * (3 * n / 2) + 1] = nyq_im;

    // Negative frequencies
    for i in (n / 2 + 1)..n {
        expanded[2 * (i + n)] = c_buf[2 * i];
        expanded[2 * (i + n) + 1] = c_buf[2 * i + 1];
    }

    // IFFT of size 2n
    cfft_f32(&mut expanded[..4 * n], 2 * n, 1, 1);

    // Copy back scaled real part (factor of 2)
    for i in 0..(2 * n) {
        dst[i] = 2.0 * expanded[2 * i];
    }

    Status::Success
}

// ─────────────────────────────────────────────────────────────────────────────
// Half-Band FIR Decimation & Interpolation Filters (HbfDec & HbfInt)
// ─────────────────────────────────────────────────────────────────────────────

/// Optimal known-good half-band filter coefficients for power-of-2 rate change cascades.
///
/// Stage 0 (highest rate, transition width 0.47 fs) to Stage 4 (relaxed narrow transition).
pub const HBF_CASCADE_COEFFS: [&[f32]; 5] = [
    // Stage 0: 6 symmetric taps (23-tap filter)
    &[
        -0.00086943, 0.00577837, -0.02201674, 0.06357869, -0.16627679, 0.61979312,
    ],
    // Stage 1: 3 symmetric taps (11-tap filter)
    &[
        0.01414651, -0.10439639, 0.59026742,
    ],
    // Stage 2: 3 symmetric taps (11-tap filter)
    &[
        0.01227974, -0.09930782, 0.58702834,
    ],
    // Stage 3: 2 symmetric taps (7-tap filter)
    &[
        -0.06291796, 0.5629161,
    ],
    // Stage 4: 2 symmetric taps (7-tap filter)
    &[
        -0.0625, 0.5625,
    ],
];

/// Single-stage half-band decimation filter (decimate by 2).
///
/// Exploits the half-band symmetry property (all even taps except center are zero)
/// to reduce multiplications by ~75% compared to a conventional direct-form FIR filter.
/// `M` is the number of symmetric tap pairs; effective filter tap length is `4*M - 1`.
#[derive(Clone, Debug)]
pub struct HbfDec<const M: usize> {
    coeffs: [f32; M],
    state: [f32; 32],
}

impl<const M: usize> HbfDec<M> {
    /// Create a new half-band decimator from symmetric tap coefficients.
    pub const fn new(coeffs: [f32; M]) -> Self {
        assert!(M <= 8, "M must be <= 8 for 32-element state");
        Self {
            coeffs,
            state: [0.0; 32],
        }
    }

    /// Reset internal delay line.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Process a block of samples, decimating by a factor of 2.
    ///
    /// `src.len()` must be twice `dst.len()`.
    pub fn process(&mut self, src: &[f32], dst: &mut [f32]) {
        let n_out = dst.len().min(src.len() / 2);
        let state_len = 4 * M;

        for i in 0..n_out {
            // Shift state by 2 samples and ingest 2 new input samples
            self.state.copy_within(2..state_len, 0);
            self.state[state_len - 2] = src[2 * i];
            self.state[state_len - 1] = src[2 * i + 1];

            // Center tap and odd symmetric taps
            let center = self.state[2 * M - 1];
            let mut acc = center;

            // Fold symmetric pairs before multiplication: h[k] * (x[2k] + x[4M - 2 - 2k])
            for k in 0..M {
                let s_left = self.state[2 * k];
                let s_right = self.state[4 * M - 2 - 2 * k];
                acc += self.coeffs[k] * (s_left + s_right);
            }

            dst[i] = 0.5 * acc;
        }
    }
}

/// Single-stage half-band interpolation filter (interpolate by 2).
///
/// Upsamples input by inserting zeros and filtering with symmetric half-band coefficients.
/// `dst.len()` must be twice `src.len()`.
#[derive(Clone, Debug)]
pub struct HbfInt<const M: usize> {
    coeffs: [f32; M],
    state: [f32; 32],
}

impl<const M: usize> HbfInt<M> {
    /// Create a new half-band interpolator from symmetric tap coefficients.
    pub const fn new(coeffs: [f32; M]) -> Self {
        assert!(M <= 16, "M must be <= 16 for 32-element state");
        Self {
            coeffs,
            state: [0.0; 32],
        }
    }

    /// Reset internal delay line.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Process a block of samples, interpolating by a factor of 2.
    pub fn process(&mut self, src: &[f32], dst: &mut [f32]) {
        let n_in = src.len().min(dst.len() / 2);
        let state_len = 2 * M;

        for i in 0..n_in {
            // Shift state by 1 and insert new sample
            self.state.copy_within(1..state_len, 0);
            self.state[state_len - 1] = src[i];

            // Even output sample: passes through direct center sample
            dst[2 * i] = self.state[M];

            // Odd output sample: filtered interpolation tap sum
            let mut acc = 0.0f32;
            for k in 0..M {
                let s_left = self.state[k];
                let s_right = self.state[2 * M - 1 - k];
                acc += self.coeffs[k] * (s_left + s_right);
            }
            dst[2 * i + 1] = acc;
        }
    }
}

/// Multi-stage cascaded half-band decimation filter.
///
/// Decimates by `2^STAGES` (e.g. STAGES=1 => 2x, STAGES=2 => 4x, STAGES=3 => 8x, STAGES=4 => 16x, STAGES=5 => 32x)
/// using optimal staged coefficients with progressively relaxed transition bands.
#[derive(Clone, Debug)]
pub struct HbfDecCascade<const STAGES: usize> {
    stage0: HbfDec<6>,
    stage1: HbfDec<3>,
    stage2: HbfDec<3>,
    stage3: HbfDec<2>,
    stage4: HbfDec<2>,
}

impl<const STAGES: usize> Default for HbfDecCascade<STAGES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const STAGES: usize> HbfDecCascade<STAGES> {
    /// Create a new pre-tuned multi-stage half-band decimation cascade.
    pub fn new() -> Self {
        const { assert!(STAGES >= 1 && STAGES <= 5, "STAGES must be between 1 and 5") };
        Self {
            stage0: HbfDec::new([
                -0.00086943, 0.00577837, -0.02201674, 0.06357869, -0.16627679, 0.61979312,
            ]),
            stage1: HbfDec::new([
                0.01414651, -0.10439639, 0.59026742,
            ]),
            stage2: HbfDec::new([
                0.01227974, -0.09930782, 0.58702834,
            ]),
            stage3: HbfDec::new([
                -0.06291796, 0.5629161,
            ]),
            stage4: HbfDec::new([
                -0.0625, 0.5625,
            ]),
        }
    }

    /// Reset all cascade stages.
    pub fn reset(&mut self) {
        self.stage0.reset();
        self.stage1.reset();
        self.stage2.reset();
        self.stage3.reset();
        self.stage4.reset();
    }

    /// Decimate input slice `src` into `dst`. `src.len()` must be `(1 << STAGES) * dst.len()`.
    pub fn process(&mut self, src: &[f32], dst: &mut [f32]) {
        let dec_factor = 1 << STAGES;
        let n_out = dst.len().min(src.len() / dec_factor);
        if n_out == 0 {
            return;
        }

        // Ping-pong scratch buffers on stack (up to 1024 samples)
        let mut buf_a = [0.0f32; 1024];
        let mut buf_b = [0.0f32; 1024];

        let len1 = n_out * (1 << (STAGES - 1));
        self.stage0.process(&src[..len1 * 2], &mut buf_a[..len1]);

        if STAGES == 1 {
            dst[..n_out].copy_from_slice(&buf_a[..n_out]);
            return;
        }

        let len2 = n_out * (1 << (STAGES - 2));
        self.stage1.process(&buf_a[..len1], &mut buf_b[..len2]);

        if STAGES == 2 {
            dst[..n_out].copy_from_slice(&buf_b[..n_out]);
            return;
        }

        let len3 = n_out * (1 << (STAGES - 3));
        self.stage2.process(&buf_b[..len2], &mut buf_a[..len3]);

        if STAGES == 3 {
            dst[..n_out].copy_from_slice(&buf_a[..n_out]);
            return;
        }

        let len4 = n_out * (1 << (STAGES - 4));
        self.stage3.process(&buf_a[..len3], &mut buf_b[..len4]);

        if STAGES == 4 {
            dst[..n_out].copy_from_slice(&buf_b[..n_out]);
            return;
        }

        self.stage4.process(&buf_b[..len4], &mut dst[..n_out]);
    }
}

