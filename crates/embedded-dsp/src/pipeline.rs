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

// ─────────────────────────────────────────────────────────────────────────────
// Composable Streaming and Split Config/State Architecture
// ─────────────────────────────────────────────────────────────────────────────

/// Single-input synchronous DSP processing trait with state held in `&mut self`.
pub trait Process<X: Copy, Y = X> {
    /// Update internal state with a new input sample and produce an output sample.
    fn process(&mut self, x: X) -> Y;

    /// Process a block of input samples into an output slice.
    #[inline]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        debug_assert_eq!(x.len(), y.len());
        let len = x.len().min(y.len());
        for i in 0..len {
            y[i] = self.process(x[i]);
        }
    }
}

/// Inplace synchronous DSP processing trait where input and output buffers share the same storage.
pub trait Inplace<X: Copy>: Process<X> {
    /// Overwrite slice `xy` with processed output samples.
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        for sample in xy.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

/// Processing with split state and configuration.
///
/// An immutable configuration `&self` operates on an explicit mutable runtime state `&mut S`.
/// This enables multi-channel lane sharing without duplicating coefficients, and lock-free/glitch-free
/// coefficient swapping.
pub trait SplitProcess<X: Copy, Y = X, S: ?Sized = ()> {
    /// Process an input into an output using explicit state `state`.
    fn process(&self, state: &mut S, x: X) -> Y;

    /// Process a block of input samples using explicit state `state`.
    #[inline]
    fn block(&self, state: &mut S, x: &[X], y: &mut [Y]) {
        debug_assert_eq!(x.len(), y.len());
        let len = x.len().min(y.len());
        for i in 0..len {
            y[i] = self.process(state, x[i]);
        }
    }
}

/// Inplace processing with split state and configuration.
pub trait SplitInplace<X: Copy, S: ?Sized = ()>: SplitProcess<X, X, S> {
    /// Process slice `xy` in-place using explicit state `state`.
    #[inline]
    fn inplace(&self, state: &mut S, xy: &mut [X]) {
        for sample in xy.iter_mut() {
            *sample = self.process(state, *sample);
        }
    }
}

/// Combines an immutable configuration `config` and a mutable state `state` into an ordinary [`Process`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Split<C, S> {
    /// Node configuration.
    pub config: C,
    /// Filter state buffer.
    pub state: S,
}

impl<C, S> Split<C, S> {
    #[inline(always)]
    /// Creates a new instance.
    pub const fn new(config: C, state: S) -> Self {
        Self { config, state }
    }
}

impl<C, S, X: Copy, Y> Process<X, Y> for Split<C, S>
where
    C: SplitProcess<X, Y, S>,
{
    #[inline(always)]
    fn process(&mut self, x: X) -> Y {
        self.config.process(&mut self.state, x)
    }

    #[inline(always)]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        self.config.block(&mut self.state, x, y);
    }
}

impl<C, S, X: Copy> Inplace<X> for Split<C, S>
where
    C: SplitInplace<X, S>,
{
    #[inline(always)]
    fn inplace(&mut self, xy: &mut [X]) {
        self.config.inplace(&mut self.state, xy);
    }
}

impl<T: Copy, C, S> DspNode<T> for Split<C, S>
where
    C: SplitProcess<T, T, S>,
{
    #[inline(always)]
    fn process_sample(&mut self, input: T) -> T {
        self.process(input)
    }

    #[inline(always)]
    fn process_block(&mut self, in_buf: &[T], out_buf: &mut [T]) {
        self.block(in_buf, out_buf);
    }
}

/// Multi-channel processing wrapper where one shared configuration operates across N independent state lanes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Lanes<C>(pub C);

impl<C> Lanes<C> {
    #[inline(always)]
    /// Creates a new instance.
    pub const fn new(config: C) -> Self {
        Self(config)
    }

    #[inline(always)]
    /// Into inner.
    pub fn into_inner(self) -> C {
        self.0
    }
}

impl<X: Copy, Y: Copy, C, S, const N: usize> SplitProcess<[X; N], [Y; N], [S; N]> for Lanes<C>
where
    C: SplitProcess<X, Y, S>,
{
    #[inline]
    fn process(&self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        core::array::from_fn(|i| self.0.process(&mut state[i], x[i]))
    }
}

/// Parallel dual-branch processing pair (e.g. Hilbert transforms, allpass branch pairs, I/Q branches).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Pair<C0, C1>(pub C0, pub C1);

impl<C0, C1> Pair<C0, C1> {
    #[inline(always)]
    /// Creates a new instance.
    pub const fn new(branch0: C0, branch1: C1) -> Self {
        Self(branch0, branch1)
    }
}

impl<X: Copy, Y: Copy, C0, C1, S0, S1> SplitProcess<[X; 2], [Y; 2], (S0, S1)>
    for Pair<C0, C1>
where
    C0: SplitProcess<X, Y, S0>,
    C1: SplitProcess<X, Y, S1>,
{
    #[inline]
    fn process(&self, (s0, s1): &mut (S0, S1), [x0, x1]: [X; 2]) -> [Y; 2] {
        [self.0.process(s0, x0), self.1.process(s1, x1)]
    }
}

/// Offset addition node: `y = x + offset`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Offset<T>(pub T);

impl<T: DspSample> Process<T, T> for Offset<T> {
    #[inline(always)]
    fn process(&mut self, x: T) -> T {
        x.sat_add(self.0)
    }
}

impl<T: DspSample> Inplace<T> for Offset<T> {}

impl<T: DspSample, S: ?Sized> SplitProcess<T, T, S> for Offset<T> {
    #[inline(always)]
    fn process(&self, _state: &mut S, x: T) -> T {
        x.sat_add(self.0)
    }
}

impl<T: DspSample, S: ?Sized> SplitInplace<T, S> for Offset<T> {}

/// Identity passthrough node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Identity;

impl<X: Copy> Process<X, X> for Identity {
    #[inline(always)]
    fn process(&mut self, x: X) -> X {
        x
    }
}

impl<X: Copy> Inplace<X> for Identity {}

impl<X: Copy, S: ?Sized> SplitProcess<X, X, S> for Identity {
    #[inline(always)]
    fn process(&self, _state: &mut S, x: X) -> X {
        x
    }
}

impl<X: Copy, S: ?Sized> SplitInplace<X, S> for Identity {}

/// A sequential composition of two DSP nodes `A` and `B` with zero runtime overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct Chain<A, B> {
    /// First.
    pub first: A,
    /// Second.
    pub second: B,
}

impl<T: Copy, A: DspNode<T>, B: DspNode<T>> DspNode<T> for Chain<A, B> {
    #[inline(always)]
    fn process_sample(&mut self, input: T) -> T {
        let intermediate = self.first.process_sample(input);
        self.second.process_sample(intermediate)
    }
}

impl<T: Copy, A: Process<T, T>, B: Process<T, T>> Process<T, T> for Chain<A, B> {
    #[inline(always)]
    fn process(&mut self, input: T) -> T {
        let intermediate = self.first.process(input);
        self.second.process(intermediate)
    }
}

impl<T: Copy, A: Inplace<T>, B: Inplace<T>> Inplace<T> for Chain<A, B> {
    #[inline(always)]
    fn inplace(&mut self, xy: &mut [T]) {
        self.first.inplace(xy);
        self.second.inplace(xy);
    }
}

/// Linear gain scaling node.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gain<T> {
    /// Gain.
    pub gain: T,
}

impl<T> Gain<T> {
    #[inline(always)]
    /// Creates a new instance.
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

impl<T: DspSample> Process<T, T> for Gain<T> {
    #[inline(always)]
    fn process(&mut self, input: T) -> T {
        input.sat_mul(self.gain)
    }
}

impl<T: DspSample> Inplace<T> for Gain<T> {}

impl DspNode<i16> for Gain<i16> {
    #[inline(always)]
    fn process_sample(&mut self, input: i16) -> i16 {
        crate::types::q15_mult(q15::from_bits(input), q15::from_bits(self.gain)).to_bits()
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
    /// Minimum saturation clamp.
    pub min: T,
    /// Maximum saturation clamp.
    pub max: T,
}

impl<T> Limiter<T> {
    #[inline(always)]
    /// Creates a new instance.
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

impl<T: PartialOrd + Copy> Process<T, T> for Limiter<T> {
    #[inline(always)]
    fn process(&mut self, input: T) -> T {
        if input < self.min {
            self.min
        } else if input > self.max {
            self.max
        } else {
            input
        }
    }
}

impl<T: PartialOrd + Copy> Inplace<T> for Limiter<T> {}

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
