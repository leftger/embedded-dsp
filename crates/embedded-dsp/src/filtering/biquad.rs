//! Biquad and direct-form sections: cascades, clamping, and fixed/integer forms.

use crate::types::*;

// --- Biquad Cascade Direct Form I Filter ---

/// Instance structure for the biquad cascade, Direct Form I, generic over the sample width.
///
/// Coefficients are `5 * num_stages` values `[b0, b1, b2, a1, a2]` per stage; state is
/// `4 * num_stages` values `[x[n-1], x[n-2], y[n-1], y[n-2]]` per stage. `post_shift` gives extra
/// coefficient headroom (CMSIS-style) and the MAC is narrowed by `FRAC - post_shift`; floats leave
/// it at `0` and take the plain per-stage sum.
pub struct BiquadCascadeInstance<'a, T: DspSample> {
    /// Number of biquad stages.
    pub num_stages: u8,
    /// Output right-shift.
    pub post_shift: u8,
    /// Filter coefficients.
    pub coeffs: &'a [T::Coeff],
    /// Filter state buffer.
    pub state: &'a mut [T],
}

impl<'a, T: DspSample> BiquadCascadeInstance<'a, T> {
    /// Initializes the instance with no coefficient headroom (`post_shift = 0`).
    pub fn init(num_stages: u8, coeffs: &'a [T::Coeff], state: &'a mut [T]) -> Self {
        state.fill(T::ZERO);
        Self {
            num_stages,
            post_shift: 0,
            coeffs,
            state,
        }
    }

    /// Initializes the instance with a coefficient `post_shift` (fixed-point headroom).
    pub fn with_post_shift(
        num_stages: u8,
        coeffs: &'a [T::Coeff],
        state: &'a mut [T],
        post_shift: u8,
    ) -> Self {
        state.fill(T::ZERO);
        Self {
            num_stages,
            post_shift,
            coeffs,
            state,
        }
    }
}

/// Biquad cascade, Direct Form I, generic over the sample width.
pub fn biquad_cascade_df1<T: DspSample>(
    instance: &mut BiquadCascadeInstance<'_, T>,
    src: &[T],
    dst: &mut [T],
) {
    let num_stages = instance.num_stages as usize;
    let block_size = src.len().min(dst.len());
    let shift = T::FRAC.saturating_sub(instance.post_shift as u32).min(63);

    for i in 0..block_size {
        let mut in_val = src[i];
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

            let acc = T::madd(
                T::madd(
                    T::madd(T::madd(T::madd(T::Accum::default(), in_val, b0), x1, b1), x2, b2),
                    y1,
                    a1,
                ),
                y2,
                a2,
            );
            let out_val = T::from_accum_shifted(acc, shift);

            instance.state[stage * 4 + 1] = x1;
            instance.state[stage * 4] = in_val;
            instance.state[stage * 4 + 3] = y1;
            instance.state[stage * 4 + 2] = out_val;

            in_val = out_val;
        }
        dst[i] = in_val;
    }
}



/// Instance structure for the biquad cascade, transposed Direct Form II, generic over the sample
/// width.
///
/// Same `[b0, b1, b2, a1, a2]` and `post_shift` conventions as [`BiquadCascadeInstance`]; state is
/// two delays per stage (`[s1, s2, ...]`), which is better-conditioned for high-Q poles.
pub struct BiquadCascadeDf2tInstance<'a, T: DspSample> {
    /// Number of biquad stages.
    pub num_stages: u8,
    /// Output right-shift.
    pub post_shift: u8,
    /// Filter coefficients.
    pub coeffs: &'a [T::Coeff],
    /// Filter state buffer.
    pub state: &'a mut [T],
}

impl<'a, T: DspSample> BiquadCascadeDf2tInstance<'a, T> {
    /// Initializes the instance with no coefficient headroom (`post_shift = 0`).
    pub fn init(num_stages: u8, coeffs: &'a [T::Coeff], state: &'a mut [T]) -> Self {
        state.fill(T::ZERO);
        Self {
            num_stages,
            post_shift: 0,
            coeffs,
            state,
        }
    }

    /// Initializes the instance with a coefficient `post_shift` (fixed-point headroom).
    pub fn with_post_shift(
        num_stages: u8,
        coeffs: &'a [T::Coeff],
        state: &'a mut [T],
        post_shift: u8,
    ) -> Self {
        state.fill(T::ZERO);
        Self {
            num_stages,
            post_shift,
            coeffs,
            state,
        }
    }
}

/// Biquad cascade, transposed Direct Form II, generic over the sample width.
pub fn biquad_cascade_df2t<T: DspSample>(
    instance: &mut BiquadCascadeDf2tInstance<'_, T>,
    src: &[T],
    dst: &mut [T],
) {
    let num_stages = instance.num_stages as usize;
    let block_size = src.len().min(dst.len());
    let shift = T::FRAC.saturating_sub(instance.post_shift as u32).min(63);

    for i in 0..block_size {
        let mut in_val = src[i];
        for stage in 0..num_stages {
            let b0 = instance.coeffs[stage * 5];
            let b1 = instance.coeffs[stage * 5 + 1];
            let b2 = instance.coeffs[stage * 5 + 2];
            let a1 = instance.coeffs[stage * 5 + 3];
            let a2 = instance.coeffs[stage * 5 + 4];

            let s1 = instance.state[stage * 2];
            let s2 = instance.state[stage * 2 + 1];

            // The accumulator is ordered to match the original float association exactly:
            // `y = b0*in + s1`, `s1' = (b1*in + a1*y) + s2`, `s2' = b2*in + a2*y`.
            let y_acc =
                T::madd(T::Accum::default(), in_val, b0) + T::accum_from_shifted(s1, shift);
            let y = T::from_accum_shifted(y_acc, shift);
            let s1_acc = T::madd(T::madd(T::Accum::default(), in_val, b1), y, a1)
                + T::accum_from_shifted(s2, shift);
            let s1_new = T::from_accum_shifted(s1_acc, shift);
            let s2_acc = T::madd(T::madd(T::Accum::default(), in_val, b2), y, a2);
            let s2_new = T::from_accum_shifted(s2_acc, shift);

            instance.state[stage * 2] = s1_new;
            instance.state[stage * 2 + 1] = s2_new;
            in_val = y;
        }
        dst[i] = in_val;
    }
}



// ─────────────────────────────────────────────────────────────────────────────
// Robust Second-Order Sections (Biquads) with Anti-Windup & Clamping
// ─────────────────────────────────────────────────────────────────────────────

/// Direct Form 1 filter history state holding delayed inputs and outputs `[x1, x2, y1, y2]`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable))]
pub struct DirectForm1<T> {
    /// Input/output history.
    pub xy: [T; 4],
}

impl<T: Default + Copy> Default for DirectForm1<T> {
    fn default() -> Self {
        Self {
            xy: [T::default(); 4],
        }
    }
}

impl<T: Default + Copy> DirectForm1<T> {
    /// Create a new zero-initialized Direct Form 1 state.
    pub fn new() -> Self {
        Self {
            xy: [T::default(); 4],
        }
    }

    /// Reset internal state buffer.
    pub fn reset(&mut self) {
        self.xy = [T::default(); 4];
    }
}

/// Direct Form 2 Transposed filter state holding accumulator registers `[s0, s1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable))]
pub struct DirectForm2Transposed<T> {
    /// S.
    pub s: [T; 2],
}

impl<T: Default + Copy> Default for DirectForm2Transposed<T> {
    fn default() -> Self {
        Self {
            s: [T::default(); 2],
        }
    }
}

impl<T: Default + Copy> DirectForm2Transposed<T> {
    /// Create a new zero-initialized Direct Form 2 Transposed state.
    pub fn new() -> Self {
        Self {
            s: [T::default(); 2],
        }
    }

    /// Reset internal state buffer.
    pub fn reset(&mut self) {
        self.s = [T::default(); 2];
    }
}

/// Second-order section (SOS) biquadratic filter configuration.
///
/// Contains coefficients `ba: [b0, b1, b2, a1, a2]` normalized such that `a0 = 1`.
/// Recurrence relation:
/// `y0 = b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2`
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Biquad<T> {
    /// Second-order-section coefficients `[b0, b1, b2, a1, a2]`.
    pub ba: [T; 5],
}

impl<T: Copy> Biquad<T> {
    /// Create a new Biquad configuration from coefficients `[b0, b1, b2, a1, a2]`.
    pub const fn new(b0: T, b1: T, b2: T, a1: T, a2: T) -> Self {
        Self {
            ba: [b0, b1, b2, a1, a2],
        }
    }
}

impl Biquad<f32> {
    /// Process a single input sample through Direct Form 1 state.
    #[inline(always)]
    pub fn process_df1(&self, state: &mut DirectForm1<f32>, x0: f32) -> f32 {
        let [b0, b1, b2, a1, a2] = self.ba;
        let [x1, x2, y1, y2] = state.xy;
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
        state.xy = [x0, x1, y0, y1];
        y0
    }

    /// Process a single input sample through Direct Form 2 Transposed state.
    #[inline(always)]
    pub fn process_df2t(&self, state: &mut DirectForm2Transposed<f32>, x0: f32) -> f32 {
        let [b0, b1, b2, a1, a2] = self.ba;
        let y0 = b0 * x0 + state.s[0];
        state.s[0] = b1 * x0 + a1 * y0 + state.s[1];
        state.s[1] = b2 * x0 + a2 * y0;
        y0
    }
}

impl Biquad<f64> {
    /// Process a single input sample through Direct Form 1 state.
    #[inline(always)]
    pub fn process_df1(&self, state: &mut DirectForm1<f64>, x0: f64) -> f64 {
        let [b0, b1, b2, a1, a2] = self.ba;
        let [x1, x2, y1, y2] = state.xy;
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
        state.xy = [x0, x1, y0, y1];
        y0
    }

    /// Process a single input sample through Direct Form 2 Transposed state.
    #[inline(always)]
    pub fn process_df2t(&self, state: &mut DirectForm2Transposed<f64>, x0: f64) -> f64 {
        let [b0, b1, b2, a1, a2] = self.ba;
        let y0 = b0 * x0 + state.s[0];
        state.s[0] = b1 * x0 + a1 * y0 + state.s[1];
        state.s[1] = b2 * x0 + a2 * y0;
        y0
    }
}

/// Biquadratic filter configuration with summing junction offset and anti-windup output clamping.
///
/// Clamps output between `[min, max]` at the summing junction before storing into feedback state,
/// preventing integrator windup and derivative kick when used in feedback control or PID applications.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct BiquadClamp<T> {
    /// Coeff.
    pub coeff: Biquad<T>,
    /// Summing junction offset (setpoint)
    pub u: T,
    /// Minimum saturation clamp
    pub min: T,
    /// Maximum saturation clamp
    pub max: T,
}

impl<T: Copy> BiquadClamp<T> {
    /// Create a new clamped Biquad with coefficients, offset, and clamp bounds.
    pub const fn new(coeff: Biquad<T>, min: T, max: T, u: T) -> Self {
        Self { coeff, u, min, max }
    }
}

impl BiquadClamp<f32> {
    /// Process a sample using Direct Form 1 with anti-windup clamping.
    #[inline(always)]
    pub fn process_df1(&self, state: &mut DirectForm1<f32>, x0: f32) -> f32 {
        let [b0, b1, b2, a1, a2] = self.coeff.ba;
        let [x1, x2, y1, y2] = state.xy;
        let unclamped = b0 * x0 + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2 + self.u;
        let y0 = unclamped.clamp(self.min, self.max);
        state.xy = [x0, x1, y0, y1];
        y0
    }

    /// Process a sample using Direct Form 2 Transposed with anti-windup clamping.
    #[inline(always)]
    pub fn process_df2t(&self, state: &mut DirectForm2Transposed<f32>, x0: f32) -> f32 {
        let [b0, b1, b2, a1, a2] = self.coeff.ba;
        let y0 = (b0 * x0 + state.s[0] + self.u).clamp(self.min, self.max);
        state.s[0] = b1 * x0 + a1 * y0 + state.s[1];
        state.s[1] = b2 * x0 + a2 * y0;
        y0
    }
}

impl BiquadClamp<f64> {
    /// Process a sample using Direct Form 1 with anti-windup clamping.
    #[inline(always)]
    pub fn process_df1(&self, state: &mut DirectForm1<f64>, x0: f64) -> f64 {
        let [b0, b1, b2, a1, a2] = self.coeff.ba;
        let [x1, x2, y1, y2] = state.xy;
        let unclamped = b0 * x0 + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2 + self.u;
        let y0 = unclamped.clamp(self.min, self.max);
        state.xy = [x0, x1, y0, y1];
        y0
    }

    /// Process a sample using Direct Form 2 Transposed with anti-windup clamping.
    #[inline(always)]
    pub fn process_df2t(&self, state: &mut DirectForm2Transposed<f64>, x0: f64) -> f64 {
        let [b0, b1, b2, a1, a2] = self.coeff.ba;
        let y0 = (b0 * x0 + state.s[0] + self.u).clamp(self.min, self.max);
        state.s[0] = b1 * x0 + a1 * y0 + state.s[1];
        state.s[1] = b2 * x0 + a2 * y0;
        y0
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f32, f32, DirectForm1<f32>> for Biquad<f32> {
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut DirectForm1<f32>, x: f32) -> f32 {
        self.process_df1(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f32, f32, DirectForm2Transposed<f32>> for Biquad<f32> {
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut DirectForm2Transposed<f32>, x: f32) -> f32 {
        self.process_df2t(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f32, f32, DirectForm1<f32>> for BiquadClamp<f32> {
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut DirectForm1<f32>, x: f32) -> f32 {
        self.process_df1(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f32, f32, DirectForm2Transposed<f32>> for BiquadClamp<f32> {
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut DirectForm2Transposed<f32>, x: f32) -> f32 {
        self.process_df2t(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f64, f64, DirectForm1<f64>> for Biquad<f64> {
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut DirectForm1<f64>, x: f64) -> f64 {
        self.process_df1(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f64, f64, DirectForm2Transposed<f64>> for Biquad<f64> {
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut DirectForm2Transposed<f64>, x: f64) -> f64 {
        self.process_df2t(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f64, f64, DirectForm1<f64>> for BiquadClamp<f64> {
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut DirectForm1<f64>, x: f64) -> f64 {
        self.process_df1(state, x)
    }
}

#[cfg(feature = "pipeline")]
impl crate::pipeline::SplitProcess<f64, f64, DirectForm2Transposed<f64>> for BiquadClamp<f64> {
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut DirectForm2Transposed<f64>, x: f64) -> f64 {
        self.process_df2t(state, x)
    }
}

/// Direct Form 1 state with quantization error feedback for noise shaping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable))]
pub struct DirectForm1NoiseShaped {
    /// Input/output history.
    pub xy: [i32; 4],
    /// Quantization error feedback accumulator.
    pub err: i32,
}

impl DirectForm1NoiseShaped {
    /// Create a new zero-initialized state with zero error feedback.
    pub const fn new() -> Self {
        Self {
            xy: [0; 4],
            err: 0,
        }
    }

    /// Reset internal state and error accumulator.
    pub fn reset(&mut self) {
        self.xy = [0; 4];
        self.err = 0;
    }
}

/// Fixed-point 32-bit Biquad with parameterized fractional scaling and anti-windup clamping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct BiquadFixed<const SHIFT: u32 = 30> {
    /// Fixed-point coefficients `[b0, b1, b2, a1, a2]`.
    pub ba: [i32; 5],
    /// Summing junction offset
    pub u: i32,
    /// Minimum saturation clamp
    pub min: i32,
    /// Maximum saturation clamp
    pub max: i32,
}

impl<const SHIFT: u32> BiquadFixed<SHIFT> {
    /// Create a new fixed-point biquad configuration.
    pub const fn new(ba: [i32; 5], min: i32, max: i32, u: i32) -> Self {
        Self { ba, min, max, u }
    }

    /// Process single sample with 1st-order noise shaping to eliminate limit cycles.
    #[inline(always)]
    pub fn process_noise_shaped(&self, state: &mut DirectForm1NoiseShaped, x0: i32) -> i32 {
        let [b0, b1, b2, a1, a2] = self.ba;
        let [x1, x2, y1, y2] = state.xy;
        let acc = (b0 as i64 * x0 as i64)
            + (b1 as i64 * x1 as i64)
            + (b2 as i64 * x2 as i64)
            + (a1 as i64 * y1 as i64)
            + (a2 as i64 * y2 as i64)
            + ((self.u as i64) << SHIFT)
            - state.err as i64; // noise shaping feedback
        let scaled = acc >> SHIFT;
        let y0 = scaled.clamp(self.min as i64, self.max as i64) as i32;
        state.err = (acc - ((y0 as i64) << SHIFT)) as i32;
        state.xy = [x0, x1, y0, y1];
        y0
    }

    /// Process a single sample with a 64-bit (`Q32.32`) output accumulator.
    ///
    /// Compared with [`Self::process_noise_shaped`], the feedback path carries
    /// 32 fractional bits instead of being rounded each sample, which removes
    /// the need for dithering at the cost of a wider state. This is the
    /// embedded-dsp equivalent of `idsp`'s `DirectForm1Wide` processing.
    ///
    /// # Panics
    /// Fails to compile unless `1 <= SHIFT <= 32`.
    #[inline(always)]
    pub fn process_wide(&self, state: &mut DirectForm1Wide, x0: i32) -> i32 {
        const {
            assert!(
                SHIFT >= 1 && SHIFT <= 32,
                "BiquadFixed::process_wide requires 1 <= SHIFT <= 32"
            )
        };
        let [b0, b1, b2, a1, a2] = self.ba;
        let [x1, x2] = state.x;
        let [y1, y2] = state.y;

        // Numerator: full-width products, no truncation.
        let mut acc = (b0 as i64)
            .wrapping_mul(x0 as i64)
            .wrapping_add((b1 as i64).wrapping_mul(x1 as i64))
            .wrapping_add((b2 as i64).wrapping_mul(x2 as i64));

        // Denominator: 32x32 split multiply of the wide states by the bits.
        acc = acc.wrapping_add(((y1 as u32 as i64).wrapping_mul(a1 as i64)) >> 32);
        acc = acc.wrapping_add(((y1 >> 32) as i32 as i64).wrapping_mul(a1 as i64));
        acc = acc.wrapping_add(((y2 as u32 as i64).wrapping_mul(a2 as i64)) >> 32);
        acc = acc.wrapping_add(((y2 >> 32) as i32 as i64).wrapping_mul(a2 as i64));

        // Promote from the `SHIFT`-bit coefficient scale to Q32.32.
        acc <<= 32 - SHIFT;

        let y0 = ((acc >> 32) as i32 as i64)
            .wrapping_add(self.u as i64)
            .clamp(self.min as i64, self.max as i64) as i32;

        // Keep the fractional low word of the accumulator, overwrite the output word.
        state.y = [((y0 as i64) << 32) | (acc as u32 as i64), y1];
        state.x = [x0, x1];
        y0
    }
}

#[cfg(feature = "pipeline")]
impl<const SHIFT: u32> crate::pipeline::SplitProcess<i32, i32, DirectForm1NoiseShaped>
    for BiquadFixed<SHIFT>
{
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut DirectForm1NoiseShaped, x: i32) -> i32 {
        self.process_noise_shaped(state, x)
    }
}

/// Direct Form 1 state with a 64-bit (`Q32.32`) output accumulator.
///
/// This is the embedded-dsp equivalent of `idsp`'s `DirectForm1Wide`: the
/// recursion is carried at 32 fractional bits so coefficient rounding does not
/// accumulate inside the feedback path. Use it with
/// [`BiquadFixed::process_wide`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct DirectForm1Wide {
    /// Input history `[x1, x2]`.
    pub x: [i32; 2],
    /// Output accumulator history `[y1, y2]` in `Q32.32`.
    pub y: [i64; 2],
}

impl DirectForm1Wide {
    /// Create a new zero-initialized wide state.
    pub const fn new() -> Self {
        Self {
            x: [0; 2],
            y: [0; 2],
        }
    }

    /// Reset the state and accumulator history.
    pub fn reset(&mut self) {
        self.x = [0; 2];
        self.y = [0; 2];
    }
}

#[cfg(feature = "pipeline")]
impl<const SHIFT: u32> crate::pipeline::SplitProcess<i32, i32, DirectForm1Wide>
    for BiquadFixed<SHIFT>
{
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut DirectForm1Wide, x: i32) -> i32 {
        self.process_wide(state, x)
    }
}

/// Integer sample type usable with [`BiquadInt`].
///
/// Implemented for `i8`, `i16`, `i32` and `i64`, each with a wider accumulator
/// (`i16`, `i32`, `i64` and `i128` respectively). This mirrors `idsp`'s generic
/// integer `Biquad<C>` over the primitive integer widths.
pub trait BiquadIntSample: Copy + PartialOrd {
    /// Wider accumulator type used for the recursion.
    type Wide: Copy + PartialOrd;
    /// Most negative value.
    const MIN: Self;
    /// Most positive value.
    const MAX: Self;

    /// Widen to the accumulator type.
    fn widen(self) -> Self::Wide;
    /// Saturate an accumulator value back to the sample range.
    fn narrow(w: Self::Wide) -> Self;
    /// Accumulator zero.
    fn wide_zero() -> Self::Wide;
    /// Wrapping accumulator multiply.
    fn wide_mul(a: Self::Wide, b: Self::Wide) -> Self::Wide;
    /// Wrapping accumulator add.
    fn wide_add(a: Self::Wide, b: Self::Wide) -> Self::Wide;
    /// Arithmetic right shift of the accumulator.
    fn wide_shr(a: Self::Wide, n: u32) -> Self::Wide;
}

macro_rules! impl_biquad_int_sample {
    ($sample:ty, $wide:ty) => {
        impl BiquadIntSample for $sample {
            type Wide = $wide;
            const MIN: Self = <$sample>::MIN;
            const MAX: Self = <$sample>::MAX;

            #[inline(always)]
            fn widen(self) -> $wide {
                self as $wide
            }

            #[inline(always)]
            fn narrow(w: $wide) -> Self {
                w.clamp(<$sample>::MIN as $wide, <$sample>::MAX as $wide) as $sample
            }

            #[inline(always)]
            fn wide_zero() -> $wide {
                0
            }

            #[inline(always)]
            fn wide_mul(a: $wide, b: $wide) -> $wide {
                a.wrapping_mul(b)
            }

            #[inline(always)]
            fn wide_add(a: $wide, b: $wide) -> $wide {
                a.wrapping_add(b)
            }

            #[inline(always)]
            fn wide_shr(a: $wide, n: u32) -> $wide {
                a >> n
            }
        }
    };
}

impl_biquad_int_sample!(i8, i16);
impl_biquad_int_sample!(i16, i32);
impl_biquad_int_sample!(i32, i64);
impl_biquad_int_sample!(i64, i128);

/// Direct Form 1 state for [`BiquadInt`]: `[x1, x2, y1, y2]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct DirectForm1Int<T> {
    /// `[x1, x2, y1, y2]`.
    pub xy: [T; 4],
}

impl<T: Copy + Default> Default for DirectForm1Int<T> {
    fn default() -> Self {
        Self {
            xy: [T::default(); 4],
        }
    }
}

impl<T: Copy + Default> DirectForm1Int<T> {
    /// Create a new zeroed state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the state to zero.
    pub fn reset(&mut self) {
        self.xy = [T::default(); 4];
    }
}

/// Fixed-point second-order section generic over the integer sample type.
///
/// Coefficients `ba = [b0, b1, b2, a1, a2]` are scaled by `2^SHIFT`, the
/// recurrence runs in the wider [`BiquadIntSample::Wide`] accumulator, and the
/// output is saturated to `[min, max]`. This closes the `idsp` gap of a biquad
/// that works over `i8`/`i16`/`i32`/`i64` samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct BiquadInt<T: BiquadIntSample, const SHIFT: u32 = 30> {
    /// Fixed-point coefficients `[b0, b1, b2, a1, a2]`.
    pub ba: [T; 5],
    /// Summing-junction offset, in output units.
    pub u: T,
    /// Minimum saturation clamp.
    pub min: T,
    /// Maximum saturation clamp.
    pub max: T,
}

impl<T: BiquadIntSample, const SHIFT: u32> BiquadInt<T, SHIFT> {
    /// Create a new integer biquad configuration.
    pub const fn new(ba: [T; 5], min: T, max: T, u: T) -> Self {
        Self { ba, min, max, u }
    }

    /// Process a single sample through Direct Form 1.
    ///
    /// # Panics
    /// Fails to compile unless `SHIFT < 32`.
    #[inline(always)]
    pub fn process_df1(&self, state: &mut DirectForm1Int<T>, x0: T) -> T {
        const { assert!(SHIFT < 32, "BiquadInt requires SHIFT < 32") };
        let [b0, b1, b2, a1, a2] = self.ba;
        let [x1, x2, y1, y2] = state.xy;

        let mut acc = T::wide_zero();
        for (c, s) in [(b0, x0), (b1, x1), (b2, x2), (a1, y1), (a2, y2)] {
            acc = T::wide_add(acc, T::wide_mul(c.widen(), s.widen()));
        }

        // Scale down, add the offset, then clamp to the configured output range.
        let scaled = T::narrow(T::wide_shr(acc, SHIFT));
        let y_raw = T::narrow(T::wide_add(scaled.widen(), self.u.widen()));
        let y0 = if y_raw < self.min {
            self.min
        } else if y_raw > self.max {
            self.max
        } else {
            y_raw
        };

        state.xy = [x0, x1, y0, y1];
        y0
    }
}

#[cfg(feature = "pipeline")]
impl<T: BiquadIntSample, const SHIFT: u32>
    crate::pipeline::SplitProcess<T, T, DirectForm1Int<T>> for BiquadInt<T, SHIFT>
{
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut DirectForm1Int<T>, x: T) -> T {
        self.process_df1(state, x)
    }
}
