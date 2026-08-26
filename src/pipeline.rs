//! Zero-allocation composable DSP pipeline and streaming abstraction.
//!
//! Provides the [`DspNode`] trait and zero-overhead pipeline combinators ([`Chain`], [`Gain`], [`Limiter`])
//! for real-time sample-by-sample or block DMA stream processing.

use crate::types::{q15, DspSample};

/// A processing element in a real-time digital signal processing pipeline.
pub trait DspNode<T: Copy> {
    /// Process a single input sample and produce one output sample.
    fn process_sample(&mut self, input: T) -> T;

    /// Process a block of samples from `in_buf` into `out_buf`.
    #[inline]
    fn process_block(&mut self, in_buf: &[T], out_buf: &mut [T]) {
        let len = in_buf.len().min(out_buf.len());
        for i in 0..len {
            out_buf[i] = self.process_sample(in_buf[i]);
        }
    }

    /// Process a block of samples in place.
    #[inline]
    fn process_in_place(&mut self, buf: &mut [T]) {
        for sample in buf.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Chains this node with another processing node into a sequential pipeline.
    #[inline]
    fn then<Next>(self, next: Next) -> Chain<Self, Next>
    where
        Self: Sized,
        Next: DspNode<T>,
    {
        Chain {
            first: self,
            second: next,
        }
    }
}

/// A sequential composition of two DSP nodes `A` and `B` with zero runtime overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct Chain<A, B> {
    pub first: A,
    pub second: B,
}

impl<T: Copy, A: DspNode<T>, B: DspNode<T>> DspNode<T> for Chain<A, B> {
    #[inline(always)]
    fn process_sample(&mut self, input: T) -> T {
        let intermediate = self.first.process_sample(input);
        self.second.process_sample(intermediate)
    }
}

/// Linear gain scaling node.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gain<T> {
    pub gain: T,
}

impl<T> Gain<T> {
    #[inline(always)]
    pub const fn new(gain: T) -> Self {
        Self { gain }
    }
}

impl<T: DspSample> DspNode<T> for Gain<T> {
    #[inline(always)]
    fn process_sample(&mut self, input: T) -> T {
        input.sat_mul(self.gain)
    }
}

impl DspNode<i16> for Gain<i16> {
    #[inline(always)]
    fn process_sample(&mut self, input: i16) -> i16 {
        crate::types::q15_mult(input, self.gain)
    }
}

impl DspNode<i32> for Gain<i32> {
    #[inline(always)]
    fn process_sample(&mut self, input: i32) -> i32 {
        let prod = (input as i64 * self.gain as i64) >> 31;
        prod.clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }
}

/// Hard saturation limiter node clamping between `[min, max]`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Limiter<T> {
    pub min: T,
    pub max: T,
}

impl<T> Limiter<T> {
    #[inline(always)]
    pub const fn new(min: T, max: T) -> Self {
        Self { min, max }
    }
}

impl<T: PartialOrd + Copy> DspNode<T> for Limiter<T> {
    #[inline(always)]
    fn process_sample(&mut self, input: T) -> T {
        if input < self.min {
            self.min
        } else if input > self.max {
            self.max
        } else {
            input
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Node Implementations for Built-in Filters and Controllers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "controller")]
impl DspNode<f32> for crate::controller::PidInstanceF32 {
    #[inline(always)]
    fn process_sample(&mut self, input: f32) -> f32 {
        self.process(input)
    }
}

#[cfg(feature = "controller")]
impl DspNode<q15> for crate::controller::PidInstanceQ15 {
    #[inline(always)]
    fn process_sample(&mut self, input: q15) -> q15 {
        self.process(input)
    }
}

#[cfg(feature = "filtering")]
impl DspNode<f32> for crate::filtering::SinglePoleFilter {
    #[inline(always)]
    fn process_sample(&mut self, input: f32) -> f32 {
        self.process(input)
    }
}

#[cfg(feature = "filtering")]
impl DspNode<q15> for crate::filtering::SinglePoleFilterQ15 {
    #[inline(always)]
    fn process_sample(&mut self, input: q15) -> q15 {
        self.process(input)
    }
}

#[cfg(feature = "filtering")]
impl DspNode<q15> for crate::filtering::DcBlockerQ15 {
    #[inline(always)]
    fn process_sample(&mut self, input: q15) -> q15 {
        self.process(input)
    }
}
