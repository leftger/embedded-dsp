//! Single-pole, DC-blocker, and recursive moving-average filters.

use crate::types::*;
use super::fir::CircularBuffer;

// --- Single-Pole Recursive Filter (Steven W. Smith, Ch. 19) ---

/// The cheapest possible IIR filter: a single-pole recursive low-pass or high-pass filter
/// (Steven W. Smith, Ch. 19, Eq. 19-2 / 19-3), needing only one or two multiplies per sample.
///
/// This is the generic template: the recurrence is shared across every [`DspSample`] width and
/// runs through [`DspSample::madd`] / [`DspSample::from_accum`], so a `q15` stage keeps its wide
/// accumulator and an `f32` stage keeps its plain multiply without either being a separate type.
/// Coefficients are still built per width (a fixed-point decay quantizes differently from an `f32`
/// one); the constructors below are the per-width specializations that sit *behind* the generic
/// type. Decay factors come from
/// [`crate::filter_design::single_pole_decay_from_cutoff`] /
/// [`crate::filter_design::single_pole_decay_from_time_constant`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct SinglePoleFilter<T: DspSample> {
    b0: T::Coeff,
    b1: T::Coeff,
    a1: T::Coeff,
    x1: T,
    y1: T,
}

impl<T: DspSample> SinglePoleFilter<T> {
    /// Creates a filter directly from the feed-forward coefficients `b0`/`b1` and the feed-back
    /// coefficient `a1`.
    pub fn new(b0: T::Coeff, b1: T::Coeff, a1: T::Coeff) -> Self {
        Self {
            b0,
            b1,
            a1,
            x1: T::ZERO,
            y1: T::ZERO,
        }
    }

    /// Processes a single input sample and returns the filtered output.
    #[inline(always)]
    pub fn process(&mut self, x: T) -> T {
        let acc = T::madd(
            T::madd(T::madd(T::Accum::default(), x, self.b0), self.x1, self.b1),
            self.y1,
            self.a1,
        );
        let y = T::from_accum(acc);
        self.x1 = x;
        self.y1 = y;
        y
    }

    /// Resets the filter's delay state to zero.
    pub fn reset(&mut self) {
        self.x1 = T::ZERO;
        self.y1 = T::ZERO;
    }
}

impl<T: DspSample> Default for SinglePoleFilter<T>
where
    T::Coeff: Default,
{
    fn default() -> Self {
        Self {
            b0: T::Coeff::default(),
            b1: T::Coeff::default(),
            a1: T::Coeff::default(),
            x1: T::ZERO,
            y1: T::ZERO,
        }
    }
}

impl SinglePoleFilter<f32> {
    /// Creates a single-pole low-pass filter from decay factor `x` (`0.0..1.0`); larger `x`
    /// means slower decay (a lower cutoff frequency).
    pub fn lowpass(decay: f32) -> Self {
        Self::new(1.0 - decay, 0.0, decay)
    }

    /// Creates a single-pole high-pass filter from the same decay factor `x` used by
    /// [`SinglePoleFilter::lowpass`].
    pub fn highpass(decay: f32) -> Self {
        let b0 = (1.0 + decay) / 2.0;
        Self::new(b0, -b0, decay)
    }
}

impl SinglePoleFilter<q15> {
    /// Creates a single-pole low-pass filter from Q15 decay `x` (larger → lower cutoff).
    pub fn lowpass(decay: q15) -> Self {
        let decay = decay.max(q15::ZERO);
        Self::new(
            q15::from_bits((32767i32 - decay.to_bits() as i32) as i16),
            q15::ZERO,
            decay,
        )
    }

    /// Creates a single-pole high-pass filter from the same Q15 decay used by
    /// [`SinglePoleFilter::lowpass`].
    pub fn highpass(decay: q15) -> Self {
        let decay = decay.max(q15::ZERO);
        let b0 = q15::from_bits(((32767i32 + decay.to_bits() as i32) / 2) as i16);
        Self::new(b0, -b0, decay)
    }

    /// Quantizes a floating-point decay in `0.0..1.0` to Q15 and builds a low-pass.
    pub fn lowpass_from_f32(decay: f32) -> Self {
        Self::lowpass(q15::saturating_from_num(decay.clamp(0.0, 1.0)))
    }

    /// Quantizes a floating-point decay in `0.0..1.0` to Q15 and builds a high-pass.
    pub fn highpass_from_f32(decay: f32) -> Self {
        Self::highpass(q15::saturating_from_num(decay.clamp(0.0, 1.0)))
    }
}

/// The stateless-`SplitProcess` bridge for [`SinglePoleFilter`], kept next to the type so the
/// pipeline layer does not have to reach outward to wrap it. `Process` and
/// [`DspNode`](crate::pipeline::DspNode) follow from the pipeline blankets.
#[cfg(feature = "pipeline")]
impl<T: DspSample> crate::pipeline::SplitProcess<T, T, ()> for SinglePoleFilter<T> {
    #[inline(always)]
    fn process_with_state(&mut self, _state: &mut (), input: T) -> T {
        SinglePoleFilter::process(self, input)
    }
}

/// High-pass single-pole used as a DC blocker (Smith Ch. 19).
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct DcBlockerQ15 {
    inner: SinglePoleFilter<q15>,
}

impl DcBlockerQ15 {
    /// `decay` is the same Q15 factor as [`SinglePoleFilter::highpass`].
    pub fn new(decay: q15) -> Self {
        Self {
            inner: SinglePoleFilter::<q15>::highpass(decay),
        }
    }

    /// Quantizes a floating-point decay in `0.0..1.0`.
    pub fn from_f32_decay(decay: f32) -> Self {
        Self {
            inner: SinglePoleFilter::<q15>::highpass_from_f32(decay),
        }
    }

    #[inline(always)]
    /// Processes a single input sample.
    pub fn process(&mut self, x: q15) -> q15 {
        self.inner.process(x)
    }

    /// Resets the internal state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

// --- Recursive Moving Average Filter (Steven W. Smith, Ch. 15) ---

/// Const-generic `N`-point moving average filter implemented recursively (Steven W. Smith,
/// Ch. 15, Eq. 15-3): each sample is updated with a single add and subtract instead of an
/// `O(N)` convolution sum. Generic over the sample width.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct RecursiveMovingAverage<T: DspSample, const N: usize> {
    history: CircularBuffer<T, N>,
    sum: T::Accum,
}

impl<T: DspSample, const N: usize> RecursiveMovingAverage<T, N> {
    /// Creates a new `N`-point recursive moving average filter with empty history.
    pub fn new() -> Self {
        Self {
            history: CircularBuffer::new(T::ZERO),
            sum: T::Accum::default(),
        }
    }

    /// Pushes a new input sample and returns the updated moving average. While fewer than `N`
    /// samples have been seen, the average is taken over the (growing) window received so far.
    #[inline(always)]
    pub fn process(&mut self, x: T) -> T {
        let oldest = if self.history.is_full() {
            self.history.oldest().unwrap_or(T::ZERO)
        } else {
            T::ZERO
        };
        let delta = T::accum_from_shifted(x, 0) - T::accum_from_shifted(oldest, 0);
        self.sum = self.sum + delta;
        self.history.push(x);
        if self.history.is_empty() {
            T::ZERO
        } else {
            T::average_accum(self.sum, self.history.len())
        }
    }

    /// Resets the filter to its initial, empty state.
    pub fn reset(&mut self) {
        self.history.clear(T::ZERO);
        self.sum = T::Accum::default();
    }
}

impl<T: DspSample, const N: usize> Default for RecursiveMovingAverage<T, N> {
    fn default() -> Self {
        Self::new()
    }
}


