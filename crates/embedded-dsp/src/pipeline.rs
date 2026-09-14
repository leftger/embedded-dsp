//! Zero-allocation composable DSP pipeline and streaming abstraction.
//!
//! Provides the [`DspNode`] trait and zero-overhead pipeline combinators ([`Chain`], [`Gain`], [`Limiter`])
//! for real-time sample-by-sample or block DMA stream processing.

use core::marker::PhantomData;

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
    fn process_with_state(&mut self, state: &mut S, x: X) -> Y;

    /// Process a block of input samples using explicit state `state`.
    #[inline]
    fn block_with_state(&mut self, state: &mut S, x: &[X], y: &mut [Y]) {
        debug_assert_eq!(x.len(), y.len());
        let len = x.len().min(y.len());
        for i in 0..len {
            y[i] = self.process_with_state(state, x[i]);
        }
    }
}

/// Inplace processing with split state and configuration.
pub trait SplitInplace<X: Copy, S: ?Sized = ()>: SplitProcess<X, X, S> {
    /// Process slice `xy` in-place using explicit state `state`.
    #[inline]
    fn inplace_with_state(&mut self, state: &mut S, xy: &mut [X]) {
        for sample in xy.iter_mut() {
            *sample = self.process_with_state(state, *sample);
        }
    }
}

/// Any stateless [`SplitProcess`] (`S = ()`) is an ordinary [`Process`]: the split vocabulary is the
/// general one, and sample streaming is its degenerate case. A stateless stage implements
/// `SplitProcess` once and inherits `Process` and, through it, [`DspNode`].
impl<X: Copy, Y, C> Process<X, Y> for C
where
    C: SplitProcess<X, Y>,
{
    #[inline(always)]
    fn process(&mut self, x: X) -> Y {
        SplitProcess::process_with_state(self, &mut (), x)
    }

    #[inline(always)]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        SplitProcess::block_with_state(self, &mut (), x, y);
    }
}

/// Any [`Process`] is a single-sample [`DspNode`], so the node vocabulary is derived from the
/// streaming one rather than implemented per node.
impl<T: Copy, N: Process<T, T>> DspNode<T> for N {
    #[inline(always)]
    fn process_sample(&mut self, input: T) -> T {
        Process::process(self, input)
    }

    #[inline(always)]
    fn process_block(&mut self, in_buf: &[T], out_buf: &mut [T]) {
        // `DspNode::process_block` clamps to the shorter buffer, whereas `Process::block` requires
        // equal lengths and may take a configuration's specialised path. Dispatch on that.
        if in_buf.len() == out_buf.len() {
            Process::block(self, in_buf, out_buf);
        } else {
            let len = in_buf.len().min(out_buf.len());
            for i in 0..len {
                out_buf[i] = self.process_sample(in_buf[i]);
            }
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

/// [`Split`] is itself a stateless `SplitProcess` (`S = ()`): it owns the runtime state and drives
/// its configuration with it, so it inherits `Process`/`DspNode` from the blankets above while
/// still routing block/in-place work to the configuration's specialised paths.
impl<C, S, X: Copy, Y> SplitProcess<X, Y, ()> for Split<C, S>
where
    C: SplitProcess<X, Y, S>,
{
    #[inline(always)]
    fn process_with_state(&mut self, _state: &mut (), x: X) -> Y {
        self.config.process_with_state(&mut self.state, x)
    }

    #[inline(always)]
    fn block_with_state(&mut self, _state: &mut (), x: &[X], y: &mut [Y]) {
        self.config.block_with_state(&mut self.state, x, y);
    }
}

impl<C, S, X: Copy> SplitInplace<X, ()> for Split<C, S>
where
    C: SplitInplace<X, S>,
{
    #[inline(always)]
    fn inplace_with_state(&mut self, _state: &mut (), xy: &mut [X]) {
        self.config.inplace_with_state(&mut self.state, xy);
    }
}

impl<C, S, X: Copy> Inplace<X> for Split<C, S>
where
    C: SplitInplace<X, S>,
{
    #[inline(always)]
    fn inplace(&mut self, xy: &mut [X]) {
        self.config.inplace_with_state(&mut self.state, xy);
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
    fn process_with_state(&mut self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        core::array::from_fn(|i| self.0.process_with_state(&mut state[i], x[i]))
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
    fn process_with_state(&mut self, (s0, s1): &mut (S0, S1), [x0, x1]: [X; 2]) -> [Y; 2] {
        [self.0.process_with_state(s0, x0), self.1.process_with_state(s1, x1)]
    }
}

/// Offset addition node: `y = x + offset`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Offset<T>(pub T);

impl<T: DspSample> Inplace<T> for Offset<T> {}

impl<T: DspSample, S: ?Sized> SplitProcess<T, T, S> for Offset<T> {
    #[inline(always)]
    fn process_with_state(&mut self, _state: &mut S, x: T) -> T {
        x.sat_add(self.0)
    }
}

impl<T: DspSample, S: ?Sized> SplitInplace<T, S> for Offset<T> {}

/// Identity passthrough node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Identity;

impl<X: Copy> Inplace<X> for Identity {}

impl<X: Copy, S: ?Sized> SplitProcess<X, X, S> for Identity {
    #[inline(always)]
    fn process_with_state(&mut self, _state: &mut S, x: X) -> X {
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

impl<T: Copy, A: Process<T, T>, B: Process<T, T>> SplitProcess<T, T, ()> for Chain<A, B> {
    #[inline(always)]
    fn process_with_state(&mut self, _state: &mut (), input: T) -> T {
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

impl<T: DspSample> SplitProcess<T, T, ()> for Gain<T> {
    #[inline(always)]
    fn process_with_state(&mut self, _state: &mut (), input: T) -> T {
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

impl<T: PartialOrd + Copy> SplitProcess<T, T, ()> for Limiter<T> {
    #[inline(always)]
    fn process_with_state(&mut self, _state: &mut (), input: T) -> T {
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

#[cfg(feature = "filtering")]
impl SplitProcess<q15, q15, ()> for crate::filtering::DcBlockerQ15 {
    #[inline(always)]
    fn process_with_state(&mut self, _state: &mut (), input: q15) -> q15 {
        self.process(input)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Typed layout views
// ─────────────────────────────────────────────────────────────────────────────

/// Frame-major layout marker.
///
/// Frames are contiguous and each frame holds `L` lane values, which is the ordinary
/// `[[T; L]]` interpretation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameMajor;

/// Lane-major layout marker.
///
/// Storage is `L` contiguous lane slices of equal length, which is what you want when a lane
/// should be processed as one long run rather than interleaved frame by frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneMajor;

/// Immutable typed view of a sample slice.
///
/// `Layout` records how the flat storage should be read and `L` is the lane count. A view never
/// allocates and never transposes: [`as_layout`](Self::as_layout) reinterprets the *same* storage
/// under the other layout, so a processor written for one layout can be handed a buffer built in
/// the other. Both layouts describe `frames * L` samples, which is what makes that reinterpretation
/// sound.
#[derive(Debug)]
pub struct View<'a, T, Layout, const L: usize> {
    flat: &'a [T],
    frames: usize,
    _layout: PhantomData<Layout>,
}

/// Mutable typed view of a sample slice. See [`View`].
#[derive(Debug)]
pub struct ViewMut<'a, T, Layout, const L: usize> {
    flat: &'a mut [T],
    frames: usize,
    _layout: PhantomData<Layout>,
}

impl<T, Layout, const L: usize> Clone for View<'_, T, Layout, L> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, Layout, const L: usize> Copy for View<'_, T, Layout, L> {}

impl<'a, T, Layout, const L: usize> View<'a, T, Layout, L> {
    /// The flat storage behind the view.
    #[inline(always)]
    #[must_use]
    pub fn flat(self) -> &'a [T] {
        self.flat
    }

    /// Number of frames in the view.
    #[inline(always)]
    #[must_use]
    pub const fn frames(self) -> usize {
        self.frames
    }

    /// Reinterpret the same storage under another layout marker.
    #[inline(always)]
    #[must_use]
    pub fn as_layout<Other>(self) -> View<'a, T, Other, L> {
        View {
            flat: self.flat,
            frames: self.frames,
            _layout: PhantomData,
        }
    }
}

impl<'a, T, Layout, const L: usize> ViewMut<'a, T, Layout, L> {
    /// The flat storage behind the view.
    #[inline(always)]
    #[must_use]
    pub fn flat(&self) -> &[T] {
        self.flat
    }

    /// The flat storage behind the view, mutably.
    #[inline(always)]
    #[must_use]
    pub fn flat_mut(&mut self) -> &mut [T] {
        self.flat
    }

    /// Number of frames in the view.
    #[inline(always)]
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Reinterpret the same storage under another layout marker.
    #[inline(always)]
    #[must_use]
    pub fn as_layout<Other>(self) -> ViewMut<'a, T, Other, L> {
        let Self { flat, frames, .. } = self;
        ViewMut {
            flat,
            frames,
            _layout: PhantomData,
        }
    }
}

impl<'a, T, const L: usize> View<'a, T, FrameMajor, L> {
    /// Borrow a conventional frame-major buffer.
    #[must_use]
    pub fn from_frames(frames: &'a [[T; L]]) -> Self {
        Self {
            flat: frames.as_flattened(),
            frames: frames.len(),
            _layout: PhantomData,
        }
    }

    /// The view as exact frames.
    #[must_use]
    pub fn as_frames(self) -> &'a [[T; L]] {
        let (frames, []) = self.flat.as_chunks::<L>() else {
            unreachable!()
        };
        frames
    }

    /// One frame.
    #[must_use]
    pub fn frame(self, i: usize) -> &'a [T; L] {
        &self.as_frames()[i]
    }
}

impl<'a, T, const L: usize> ViewMut<'a, T, FrameMajor, L> {
    /// Borrow a conventional frame-major buffer, mutably.
    #[must_use]
    pub fn from_frames(frames: &'a mut [[T; L]]) -> Self {
        let len = frames.len();
        Self {
            flat: frames.as_flattened_mut(),
            frames: len,
            _layout: PhantomData,
        }
    }

    /// The view as exact frames.
    #[must_use]
    pub fn as_frames(&self) -> &[[T; L]] {
        let (frames, []) = self.flat.as_chunks::<L>() else {
            unreachable!()
        };
        frames
    }

    /// The view as exact frames, mutably.
    #[must_use]
    pub fn as_frames_mut(&mut self) -> &mut [[T; L]] {
        let (frames, []) = self.flat.as_chunks_mut::<L>() else {
            unreachable!()
        };
        frames
    }

    /// One frame.
    #[must_use]
    pub fn frame(&self, i: usize) -> &[T; L] {
        &self.as_frames()[i]
    }

    /// One frame, mutably.
    #[must_use]
    pub fn frame_mut(&mut self, i: usize) -> &mut [T; L] {
        &mut self.as_frames_mut()[i]
    }
}

impl<'a, T, const L: usize> View<'a, T, LaneMajor, L> {
    /// Borrow a lane-major view over `frames * L` samples.
    ///
    /// # Panics
    ///
    /// If `flat.len() != frames * L`.
    #[must_use]
    pub fn from_flat(flat: &'a [T], frames: usize) -> Self {
        assert_eq!(flat.len(), frames * L);
        Self {
            flat,
            frames,
            _layout: PhantomData,
        }
    }

    /// One contiguous lane.
    #[must_use]
    pub fn lane(self, i: usize) -> &'a [T] {
        let start = i * self.frames;
        &self.flat[start..start + self.frames]
    }
}

impl<'a, T, const L: usize> ViewMut<'a, T, LaneMajor, L> {
    /// Borrow a lane-major view over `frames * L` samples, mutably.
    ///
    /// # Panics
    ///
    /// If `flat.len() != frames * L`.
    #[must_use]
    pub fn from_flat(flat: &'a mut [T], frames: usize) -> Self {
        assert_eq!(flat.len(), frames * L);
        Self {
            flat,
            frames,
            _layout: PhantomData,
        }
    }

    /// One contiguous lane.
    #[must_use]
    pub fn lane(&self, i: usize) -> &[T] {
        let start = i * self.frames;
        &self.flat[start..start + self.frames]
    }

    /// One contiguous lane, mutably.
    #[must_use]
    pub fn lane_mut(&mut self, i: usize) -> &mut [T] {
        let start = i * self.frames;
        &mut self.flat[start..start + self.frames]
    }
}

/// Processing over typed views, the view-aware companion to [`Process`].
pub trait ViewProcess<X, Y = X> {
    /// Process one typed input view into one typed output view.
    fn process_view(&mut self, x: X, y: Y);
}

/// In-place processing over typed views.
pub trait ViewInplace<X> {
    /// Process one typed view in place.
    fn inplace_view(&mut self, xy: X);
}

/// Split-state processing over typed views, the view-aware companion to [`SplitProcess`].
pub trait SplitViewProcess<X, Y = X, S: ?Sized = ()> {
    /// Process one typed input view into one typed output view using `state`.
    fn process_view_with_state(&mut self, state: &mut S, x: X, y: Y);
}

/// In-place split-state processing over typed views.
pub trait SplitViewInplace<X, S: ?Sized = ()> {
    /// Process one typed view in place using `state`.
    fn inplace_view_with_state(&mut self, state: &mut S, xy: X);
}

/// Frame-major: each frame is one sample of the underlying processor, so the flat storage is a
/// plain block and this is just [`SplitProcess::block_with_state`].
impl<'a, 'b, X, Y, S: ?Sized, T, const L: usize>
    SplitViewProcess<View<'a, X, FrameMajor, L>, ViewMut<'b, Y, FrameMajor, L>, S> for T
where
    X: Copy,
    T: SplitProcess<X, Y, S>,
{
    #[inline]
    fn process_view_with_state(
        &mut self,
        state: &mut S,
        x: View<'a, X, FrameMajor, L>,
        mut y: ViewMut<'b, Y, FrameMajor, L>,
    ) {
        debug_assert_eq!(x.frames(), y.frames());
        self.block_with_state(state, x.flat(), y.flat_mut());
    }
}

/// In-place, frame-major.
impl<'a, X, S: ?Sized, T, const L: usize> SplitViewInplace<ViewMut<'a, X, FrameMajor, L>, S> for T
where
    X: Copy,
    T: SplitInplace<X, S>,
{
    #[inline]
    fn inplace_view_with_state(&mut self, state: &mut S, mut xy: ViewMut<'a, X, FrameMajor, L>) {
        self.inplace_with_state(state, xy.flat_mut());
    }
}

impl<X, Y, C, S> ViewProcess<X, Y> for Split<C, S>
where
    C: SplitViewProcess<X, Y, S>,
{
    #[inline]
    fn process_view(&mut self, x: X, y: Y) {
        self.config.process_view_with_state(&mut self.state, x, y);
    }
}

impl<X, C, S> ViewInplace<X> for Split<C, S>
where
    C: SplitViewInplace<X, S>,
{
    #[inline]
    fn inplace_view(&mut self, xy: X) {
        self.config.inplace_view_with_state(&mut self.state, xy);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Composition combinators
// ─────────────────────────────────────────────────────────────────────────────

/// Marks a tuple as *state* rather than configuration.
///
/// Keeps a state-only [`Split`] distinct from a configuration-only one, since both would otherwise
/// sit in the same unit-typed position. Prefer [`Split::stateless`] and [`Split::stateful`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct Unsplit<P>(pub P);

impl<C> Split<C, ()> {
    #[inline(always)]
    /// A configuration-only split.
    pub const fn stateless(config: C) -> Self {
        Self::new(config, ())
    }
}

impl<S> Split<(), Unsplit<S>> {
    #[inline(always)]
    /// A state-only split.
    pub const fn stateful(state: S) -> Self {
        Self::new((), Unsplit(state))
    }
}

/// Butterfly: `[a, b] -> [a + b, a - b]`, saturating like the other sample-stage combinators.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Butterfly;

impl<T: DspSample> Process<[T; 2], [T; 2]> for Butterfly {
    #[inline(always)]
    fn process(&mut self, [a, b]: [T; 2]) -> [T; 2] {
        [a.sat_add(b), a.sat_sub(b)]
    }
}

impl<T: DspSample> Inplace<[T; 2]> for Butterfly {}

/// Parallel branches over tuple- or array-shaped data: each branch owns one lane and lanes never
/// interact. Reduce the branches afterwards with e.g. [`Pair`]'s consumers or your own stage.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Parallel<P>(pub P);

impl<P> Parallel<P> {
    #[inline(always)]
    /// Creates a new instance.
    pub const fn new(inner: P) -> Self {
        Self(inner)
    }
}

impl<X: Copy, Y, C0, C1, S0, S1> SplitProcess<[X; 2], [Y; 2], (S0, S1)> for Parallel<(C0, C1)>
where
    C0: SplitProcess<X, Y, S0>,
    C1: SplitProcess<X, Y, S1>,
{
    #[inline]
    fn process_with_state(&mut self, (s0, s1): &mut (S0, S1), [x0, x1]: [X; 2]) -> [Y; 2] {
        [self.0 .0.process_with_state(s0, x0), self.0 .1.process_with_state(s1, x1)]
    }
}

impl<X: Copy, Y, C, S, const N: usize> SplitProcess<[X; N], [Y; N], [S; N]> for Parallel<[C; N]>
where
    C: SplitProcess<X, Y, S>,
{
    #[inline]
    fn process_with_state(&mut self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        core::array::from_fn(|i| self.0[i].process_with_state(&mut state[i], x[i]))
    }
}

impl<X: Copy, C, S: ?Sized> SplitInplace<X, S> for Parallel<C> where Self: SplitProcess<X, X, S> {}

/// Per-lane configuration: each lane gets its own processor, run over a lane-major buffer.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct ByLane<C>(pub C);

impl<C> ByLane<C> {
    #[inline(always)]
    /// Creates a new instance.
    pub const fn new(lanes: C) -> Self {
        Self(lanes)
    }
}

impl<X: Copy, Y, C0, C1, S0, S1> SplitProcess<[X; 2], [Y; 2], (S0, S1)> for ByLane<(C0, C1)>
where
    C0: SplitProcess<X, Y, S0>,
    C1: SplitProcess<X, Y, S1>,
{
    #[inline]
    fn process_with_state(&mut self, (s0, s1): &mut (S0, S1), [x0, x1]: [X; 2]) -> [Y; 2] {
        [self.0 .0.process_with_state(s0, x0), self.0 .1.process_with_state(s1, x1)]
    }
}

impl<X: Copy, Y, C, S, const N: usize> SplitProcess<[X; N], [Y; N], [S; N]> for ByLane<[C; N]>
where
    C: SplitProcess<X, Y, S>,
{
    #[inline]
    fn process_with_state(&mut self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        core::array::from_fn(|i| self.0[i].process_with_state(&mut state[i], x[i]))
    }
}

// Lane-major views. Each lane is one contiguous run, so it is one block of its own. This is the
// case the frame-major blanket impl deliberately leaves free, which is what lets `Lanes` and
// `ByLane` own it.
impl<'a, 'b, X, Y, C, S, const N: usize>
    SplitViewProcess<View<'a, X, LaneMajor, N>, ViewMut<'b, Y, LaneMajor, N>, [S; N]> for Lanes<C>
where
    X: Copy,
    C: SplitProcess<X, Y, S>,
{
    #[inline]
    fn process_view_with_state(
        &mut self,
        state: &mut [S; N],
        x: View<'a, X, LaneMajor, N>,
        mut y: ViewMut<'b, Y, LaneMajor, N>,
    ) {
        debug_assert_eq!(x.frames(), y.frames());
        for (i, s) in state.iter_mut().enumerate() {
            self.0.block_with_state(s, x.lane(i), y.lane_mut(i));
        }
    }
}

impl<'a, X, C, S, const N: usize> SplitViewInplace<ViewMut<'a, X, LaneMajor, N>, [S; N]>
    for Lanes<C>
where
    X: Copy,
    C: SplitInplace<X, S>,
{
    #[inline]
    fn inplace_view_with_state(&mut self, state: &mut [S; N], mut xy: ViewMut<'a, X, LaneMajor, N>) {
        for (i, s) in state.iter_mut().enumerate() {
            self.0.inplace_with_state(s, xy.lane_mut(i));
        }
    }
}

impl<'a, 'b, X, Y, C0, C1, S0, S1>
    SplitViewProcess<View<'a, X, LaneMajor, 2>, ViewMut<'b, Y, LaneMajor, 2>, (S0, S1)>
    for ByLane<(C0, C1)>
where
    X: Copy,
    C0: SplitProcess<X, Y, S0>,
    C1: SplitProcess<X, Y, S1>,
{
    #[inline]
    fn process_view_with_state(
        &mut self,
        state: &mut (S0, S1),
        x: View<'a, X, LaneMajor, 2>,
        mut y: ViewMut<'b, Y, LaneMajor, 2>,
    ) {
        debug_assert_eq!(x.frames(), y.frames());
        self.0 .0.block_with_state(&mut state.0, x.lane(0), y.lane_mut(0));
        self.0 .1.block_with_state(&mut state.1, x.lane(1), y.lane_mut(1));
    }
}

impl<'a, 'b, X, Y, C, S, const N: usize>
    SplitViewProcess<View<'a, X, LaneMajor, N>, ViewMut<'b, Y, LaneMajor, N>, [S; N]>
    for ByLane<[C; N]>
where
    X: Copy,
    C: SplitProcess<X, Y, S>,
{
    #[inline]
    fn process_view_with_state(
        &mut self,
        state: &mut [S; N],
        x: View<'a, X, LaneMajor, N>,
        mut y: ViewMut<'b, Y, LaneMajor, N>,
    ) {
        debug_assert_eq!(x.frames(), y.frames());
        for ((c, s), i) in self.0.iter_mut().zip(state.iter_mut()).zip(0..) {
            c.block_with_state(s, x.lane(i), y.lane_mut(i));
        }
    }
}

/// Adapts a closure into a [`SplitProcess`], convenient for short stages and tests.
///
/// The closure receives the state and one input sample: `Fn(&mut S, X) -> Y`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FnSplitProcess<F>(pub F);

impl<F, X, Y, S> SplitProcess<X, Y, S> for FnSplitProcess<F>
where
    F: Fn(&mut S, X) -> Y,
    X: Copy,
{
    #[inline(always)]
    fn process_with_state(&mut self, state: &mut S, x: X) -> Y {
        (self.0)(state, x)
    }
}

impl<F, X, S> SplitInplace<X, S> for FnSplitProcess<F>
where
    X: Copy,
    Self: SplitProcess<X, X, S>,
{
}

/// Fixed-size sample buffer, used as a delay line or as a chunk accumulator.
///
/// Which job it does is chosen by the [`Process`] implementation in play:
///
/// * `X -> X` — a delay line of `N` samples, the simplest stateful FIFO here.
/// * `X -> Option<[X; N]>` — accumulate into chunks, yielding `Some` every `N` samples.
/// * `Option<[X; N]> -> X` — stream a chunk back out one sample at a time; a `None` on the input
///   means "keep draining the current chunk", so a chunk stays valid across a gap.
///
/// Both `Option` directions also have `block` forms that cross a gapped stream in runs rather than
/// sample by sample.
///
/// # Panics
///
/// The streaming direction reads within the chunk that was last loaded, so a run of `None`s longer
/// than that chunk's remaining samples is out of contract and panics. Feed a fresh `Some` before the
/// current one is exhausted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Buffer<B> {
    buffer: B,
    idx: usize,
}

impl<X, const N: usize> Buffer<[X; N]> {
    /// Whether the write index is at zero.
    ///
    /// For delay-line use this says nothing about whether the earlier samples are still defaults.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.idx == 0
    }
}

/// Delay line.
impl<X: Copy, const N: usize> SplitProcess<X, X, ()> for Buffer<[X; N]> {
    #[inline]
    fn process_with_state(&mut self, _state: &mut (), x: X) -> X {
        const { assert!(N > 0) };
        let y = core::mem::replace(&mut self.buffer[self.idx], x);
        self.idx = (self.idx + 1) % N;
        y
    }

    fn block_with_state(&mut self, _state: &mut (), x: &[X], y: &mut [X]) {
        const { assert!(N > 0) };
        debug_assert_eq!(x.len(), y.len());
        let mut x = x;
        let mut y = y;

        // Finish the partially-filled ring first, so the bulk swaps below start at index zero.
        if self.idx != 0 {
            let n = x.len().min(N - self.idx);
            let (xh, xr) = x.split_at(n);
            let (yh, yr) = y.split_at_mut(n);
            yh.copy_from_slice(&self.buffer[self.idx..self.idx + n]);
            self.buffer[self.idx..self.idx + n].copy_from_slice(xh);
            self.idx = (self.idx + n) % N;
            x = xr;
            y = yr;
        }

        let (chunks, tail) = x.as_chunks::<N>();
        let (out, out_tail) = y.as_chunks_mut::<N>();
        for (chunk, slot) in chunks.iter().zip(out) {
            *slot = self.buffer;
            self.buffer = *chunk;
        }

        out_tail.copy_from_slice(&self.buffer[..tail.len()]);
        self.buffer[..tail.len()].copy_from_slice(tail);
        self.idx = tail.len();
    }
}

impl<X: Copy, const N: usize> Inplace<X> for Buffer<[X; N]> {
    fn inplace(&mut self, xy: &mut [X]) {
        const { assert!(N > 0) };
        let mut xy = xy;

        if self.idx != 0 {
            let n = xy.len().min(N - self.idx);
            let (head, rest) = xy.split_at_mut(n);
            for (sample, held) in head
                .iter_mut()
                .zip(self.buffer[self.idx..self.idx + n].iter_mut())
            {
                core::mem::swap(sample, held);
            }
            self.idx = (self.idx + n) % N;
            xy = rest;
        }

        let (chunks, tail) = xy.as_chunks_mut::<N>();
        for chunk in chunks {
            core::mem::swap(chunk, &mut self.buffer);
        }

        let n = tail.len();
        for (sample, held) in tail.iter_mut().zip(self.buffer[..n].iter_mut()) {
            core::mem::swap(sample, held);
        }
        self.idx = n;
    }
}

/// The delay line applied to an array shape.
impl<X: Copy, const N: usize, const M: usize> SplitProcess<[X; M], [X; M], ()>
    for Buffer<[X; N]>
{
    #[inline]
    fn process_with_state(&mut self, _state: &mut (), x: [X; M]) -> [X; M] {
        let mut y = x;
        <Self as Process<X>>::block(self, &x, &mut y);
        y
    }
}

/// Accumulate into chunks: `Some(chunk)` every `N` samples, `None` otherwise.
impl<X: Copy, const N: usize> SplitProcess<X, Option<[X; N]>, ()> for Buffer<[X; N]> {
    #[inline]
    fn process_with_state(&mut self, _state: &mut (), x: X) -> Option<[X; N]> {
        const { assert!(N > 0) };
        self.buffer[self.idx] = x;
        self.idx += 1;
        if self.idx == N {
            self.idx = 0;
            Some(self.buffer)
        } else {
            None
        }
    }

    fn block_with_state(&mut self, _state: &mut (), x: &[X], y: &mut [Option<[X; N]>]) {
        const { assert!(N > 0) };
        debug_assert_eq!(x.len(), y.len());
        let mut x = x;
        let mut y = y;

        if self.idx != 0 {
            let n = x.len().min(N - self.idx);
            let (xh, xr) = x.split_at(n);
            let (yh, yr) = y.split_at_mut(n);
            self.buffer[self.idx..self.idx + n].copy_from_slice(xh);
            yh.fill(None);
            self.idx += n;
            if self.idx == N {
                self.idx = 0;
                yh[n - 1] = Some(self.buffer);
            }
            x = xr;
            y = yr;
        }

        let (chunks, tail) = x.as_chunks::<N>();
        let (out, out_tail) = y.as_chunks_mut::<N>();
        for (chunk, slot) in chunks.iter().zip(out) {
            let Some((last, rest)) = slot.split_last_mut() else {
                unreachable!()
            };
            rest.fill(None);
            *last = Some(*chunk);
        }

        self.buffer[..tail.len()].copy_from_slice(tail);
        out_tail.fill(None);
        self.idx = tail.len();
    }
}

/// Stream a chunk back out: `Some(chunk)` loads the buffer, `None` keeps draining it.
impl<X: Copy, const N: usize> SplitProcess<Option<[X; N]>, X, ()> for Buffer<[X; N]> {
    #[inline]
    fn process_with_state(&mut self, _state: &mut (), x: Option<[X; N]>) -> X {
        const { assert!(N > 0) };
        if let Some(chunk) = x {
            self.buffer = chunk;
            self.idx = 0;
        } else {
            self.idx += 1;
        }
        self.buffer[self.idx]
    }

    fn block_with_state(&mut self, _state: &mut (), x: &[Option<[X; N]>], y: &mut [X]) {
        const { assert!(N > 0) };
        debug_assert_eq!(x.len(), y.len());
        let mut i = 0;
        while i < x.len() {
            if let Some(chunk) = x[i] {
                self.buffer = chunk;
                self.idx = 0;
                y[i] = self.buffer[0];
                i += 1;
                continue;
            }

            // Drain in one run up to the next chunk rather than per sample.
            let run = x[i..]
                .iter()
                .position(Option::is_some)
                .unwrap_or(x.len() - i);
            y[i..i + run].copy_from_slice(&self.buffer[self.idx + 1..self.idx + 1 + run]);
            self.idx += run;
            i += run;
        }
    }
}

/// Bridges a chunk processor to a block processor.
///
/// Wraps a processor that consumes one `[X; Q]` chunk and emits one `[Y; R]` chunk, exposing it over
/// a block of `[X; N]` into `[Y; M]`, where `N` and `M` hold the same whole number of chunks. A
/// chunk is the unit of work, so a rate-changing stage sees whole chunks rather than a partial one.
///
/// The chunk counts are compile-time checks: the build fails if `Q` or `R` is zero, if `N` is not a
/// multiple of `Q`, or if `N / Q != M / R`. Pair with [`PerFrame`] to run the same stage once per
/// frame of a frame-major view.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChunkInOut<P, const Q: usize, const R: usize>(pub P);

impl<C, S, X, Y, const Q: usize, const N: usize, const R: usize, const M: usize>
    SplitProcess<[X; N], [Y; M], S> for ChunkInOut<C, Q, R>
where
    X: Copy,
    Y: Default + Copy,
    C: SplitProcess<[X; Q], [Y; R], S>,
{
    #[inline]
    fn process_with_state(&mut self, state: &mut S, x: [X; N]) -> [Y; M] {
        const { assert!(Q > 0) };
        const { assert!(R > 0) };
        const { assert!(N.is_multiple_of(Q)) };
        const { assert!(M.is_multiple_of(R)) };
        const { assert!(N / Q == M / R) };

        let mut y = [Y::default(); M];
        let (out, []) = y.as_chunks_mut::<R>() else {
            unreachable!()
        };
        let (chunks, []) = x.as_chunks::<Q>() else {
            unreachable!()
        };
        for (chunk, slot) in chunks.iter().zip(out) {
            *slot = self.0.process_with_state(state, *chunk);
        }
        y
    }

    #[inline]
    fn block_with_state(&mut self, state: &mut S, x: &[[X; N]], y: &mut [[Y; M]]) {
        const { assert!(Q > 0) };
        const { assert!(R > 0) };
        const { assert!(N.is_multiple_of(Q)) };
        const { assert!(M.is_multiple_of(R)) };
        const { assert!(N / Q == M / R) };

        let (in_chunks, []) = x.as_flattened().as_chunks::<Q>() else {
            unreachable!()
        };
        let (out_chunks, []) = y.as_flattened_mut().as_chunks_mut::<R>() else {
            unreachable!()
        };
        self.0.block_with_state(state, in_chunks, out_chunks);
    }
}

impl<C, S, X, const N: usize> SplitInplace<[X; N], S> for ChunkInOut<C, 1, 1>
where
    X: Copy + Default,
    C: SplitInplace<[X; 1], S>,
    Self: SplitProcess<[X; N], [X; N], S>,
{
    #[inline]
    fn inplace_with_state(&mut self, state: &mut S, xy: &mut [[X; N]]) {
        let (samples, []) = xy.as_flattened_mut().as_chunks_mut::<1>() else {
            unreachable!()
        };
        self.0.inplace_with_state(state, samples);
    }
}

/// Runs a chunk processor once per frame of a frame-major view.
///
/// This is the bridge between chunk semantics and the typed views: a
/// `SplitProcess<[X; Q], [Y; R], S>` becomes frame-wise processing from `View<FrameMajor, Q>` to
/// `ViewMut<FrameMajor, R>`. See [`Split::process_frames`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PerFrame<C>(pub C);

impl<C, S> Split<C, S> {
    /// Wrap the configuration in [`PerFrame`], so it runs once per frame of a frame-major view.
    #[inline(always)]
    #[must_use]
    pub fn per_frame(self) -> Split<PerFrame<C>, S> {
        Split::new(PerFrame(self.config), self.state)
    }
}

impl<C, S> Split<PerFrame<C>, S> {
    /// Process a frame-major view one frame at a time, `Q` samples in and `R` samples out.
    ///
    /// # Panics
    ///
    /// If the two views disagree on their frame count.
    #[inline]
    pub fn process_frames<'a, 'b, X, Y, const Q: usize, const R: usize>(
        &mut self,
        x: View<'a, X, FrameMajor, Q>,
        mut y: ViewMut<'b, Y, FrameMajor, R>,
    ) where
        X: Copy,
        C: SplitProcess<[X; Q], [Y; R], S>,
    {
        debug_assert_eq!(x.frames(), y.frames());
        self.config
            .0
            .block_with_state(&mut self.state, x.as_frames(), y.as_frames_mut());
    }

    /// Process a frame-major view in place, one frame at a time.
    #[inline]
    pub fn inplace_frames<'a, X, const L: usize>(&mut self, mut xy: ViewMut<'a, X, FrameMajor, L>)
    where
        X: Copy,
        C: SplitInplace<[X; L], S>,
    {
        self.config.0.inplace_with_state(&mut self.state, xy.as_frames_mut());
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    /// Running sum with explicit state, so the view plumbing is actually exercised rather than
    /// trivially passing samples through.
    struct RunningSum;

    impl SplitProcess<i32, i32, i32> for RunningSum {
        fn process_with_state(&mut self, state: &mut i32, x: i32) -> i32 {
            *state += x;
            *state
        }
    }

    impl SplitInplace<i32, i32> for RunningSum {}

    #[test]
    fn frame_major_views_run_sample_major_over_the_flat_buffer() {
        let x = [[1, 10], [2, 20], [3, 30]];
        let mut y = [[0; 2]; 3];
        let mut p = Split::new(RunningSum, 0);

        p.process_view(View::from_frames(&x), ViewMut::from_frames(&mut y));

        // Flat order is 1, 10, 2, 20, 3, 30 and the running sum carries across frames:
        // 1, 11, 13, 33, 36, 66 -- written back in frame-major order.
        assert_eq!(y, [[1, 11], [13, 33], [36, 66]]);
        assert_eq!(p.state, 66);
    }

    #[test]
    fn lane_major_views_give_each_lane_its_own_run_and_state() {
        let x = [1, 2, 3, 10, 20, 30];
        let mut y = [0; 6];

        // `Lanes` shares one configuration but keeps independent per-lane state.
        let mut p = Split::new(Lanes(RunningSum), [0, 0]);

        p.process_view(
            View::<_, LaneMajor, 2>::from_flat(&x, 3),
            ViewMut::<_, LaneMajor, 2>::from_flat(&mut y, 3),
        );

        // Each lane starts from zero, so lane 0 is 1,3,6 and lane 1 is 10,30,60 -- unlike the
        // frame-major (sample-major) order, which would carry one state across the whole buffer.
        assert_eq!(y, [1, 3, 6, 10, 30, 60]);
        assert_eq!(p.state, [6, 60]);

        let x = View::<_, LaneMajor, 2>::from_flat(&x, 3);
        assert_eq!(x.lane(0), &[1, 2, 3]);
        assert_eq!(x.lane(1), &[10, 20, 30]);
        assert_eq!(x.frames(), 3);
    }

    #[test]
    fn by_lane_gives_each_lane_its_own_configuration() {
        let x = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut y = [0.0f32; 6];

        // Lane 0 is offset by 1, lane 1 by 100 -- per-lane configuration rather than a shared one.
        let mut p = Split::new(ByLane::new((Offset(1.0f32), Offset(100.0f32))), (0.0f32, 0.0f32));

        p.process_view(
            View::<_, LaneMajor, 2>::from_flat(&x, 3),
            ViewMut::<_, LaneMajor, 2>::from_flat(&mut y, 3),
        );

        assert_eq!(y, [2.0, 3.0, 4.0, 110.0, 120.0, 130.0]);
    }

    #[test]
    fn butterfly_and_parallel_compose() {
        // Two branches on a lane each, then a butterfly reduction of the pair.
        let mut branches = Split::new(
            Parallel::new((Offset(1.0f32), Offset(0.0f32))),
            (0.0f32, 0.0f32),
        );
        assert_eq!(
            <Split<Parallel<(Offset<f32>, Offset<f32>)>, (f32, f32)> as Process<
                [f32; 2],
                [f32; 2],
            >>::process(&mut branches, [1.0, 2.0]),
            [2.0, 2.0]
        );

        let mut b = Butterfly;
        assert_eq!(b.process([4.0, 3.0]), [7.0, 1.0]);
    }

    #[test]
    fn as_layout_reinterprets_without_moving_anything() {
        let buf = [1, 2, 3, 4, 5, 6];

        let lm = View::<_, LaneMajor, 2>::from_flat(&buf, 3);
        assert_eq!(lm.frames(), 3);
        assert_eq!(lm.lane(0), &[1, 2, 3]);
        assert_eq!(lm.lane(1), &[4, 5, 6]);

        // The same six samples, read as three frames of two lanes.
        let fm: View<'_, i32, FrameMajor, 2> = lm.as_layout();
        assert_eq!(fm.flat(), &buf[..]);
        assert_eq!(fm.as_frames(), &[[1, 2], [3, 4], [5, 6]]);
        assert_eq!(fm.frame(1), &[3, 4]);
    }

    #[test]
    fn mutable_views_process_through_either_layout() {
        // Frame-major in place.
        let mut frames = [[1, 10], [2, 20]];
        let mut p = Split::new(RunningSum, 0);
        p.inplace_view(ViewMut::from_frames(&mut frames));
        assert_eq!(frames, [[1, 11], [13, 33]]);

        // The same four samples as two lanes of two, each lane keeping its own state.
        let mut flat = [1, 10, 2, 20];
        let mut q = Split::new(Lanes(RunningSum), [0, 0]);
        q.inplace_view(ViewMut::<_, LaneMajor, 2>::from_flat(&mut flat, 2));
        assert_eq!(flat, [1, 11, 2, 22]);

        // A lane-major buffer reinterpreted as frame-major is one block with one shared state.
        let mut flat = [1, 10, 2, 20];
        let mut r = Split::new(RunningSum, 0);
        r.inplace_view(ViewMut::<_, LaneMajor, 2>::from_flat(&mut flat, 2).as_layout::<FrameMajor>());
        assert_eq!(flat, [1, 11, 13, 33]);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn lane_major_rejects_a_mismatched_length() {
        let flat = [1, 2, 3, 4, 5];
        let _ = View::<_, LaneMajor, 2>::from_flat(&flat, 3);
    }

    #[test]
    fn parallel_covers_the_tuple_and_array_shapes() {
        // Tuple-shaped: one branch per lane, each contributing its own (unit) state.
        let mut pair = Split::new(Parallel::new((Offset(1.0f32), Offset(2.0f32))), ((), ()));
        assert_eq!(pair.process([1.0, 1.0]), [2.0, 3.0]);

        let mut xy = [[1.0f32, 1.0], [2.0, 2.0]];
        pair.inplace(&mut xy);
        assert_eq!(xy, [[2.0, 3.0], [3.0, 4.0]]);

        // Array-shaped: one config and one state per lane.
        let mut quad = Split::new(
            Parallel::new([Offset(1.0f32), Offset(2.0), Offset(3.0)]),
            [(); 3],
        );
        assert_eq!(quad.process([0.0, 0.0, 0.0]), [1.0, 2.0, 3.0]);

        let mut xy = [[0.0f32; 3]; 2];
        quad.inplace(&mut xy);
        assert_eq!(xy, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    }

    #[test]
    fn by_lane_covers_the_tuple_and_array_shapes() {
        let mut pair = Split::new(ByLane::new((Offset(1.0f32), Offset(10.0f32))), ((), ()));
        assert_eq!(pair.process([1.0, 2.0]), [2.0, 12.0]);

        let mut quad = Split::new(ByLane::new([Offset(1.0f32), Offset(10.0f32)]), [(); 2]);
        assert_eq!(quad.process([1.0, 2.0]), [2.0, 12.0]);

        // The array form is the one wired to lane-major views.
        let x = [1.0f32, 2.0, 3.0, 40.0, 50.0, 60.0];
        let mut y = [0.0f32; 6];
        let mut lanes = Split::new(ByLane::new([Offset(1.0f32), Offset(10.0f32)]), [(); 2]);
        lanes.process_view(
            View::<_, LaneMajor, 2>::from_flat(&x, 3),
            ViewMut::<_, LaneMajor, 2>::from_flat(&mut y, 3),
        );
        assert_eq!(y, [2.0, 3.0, 4.0, 50.0, 60.0, 70.0]);
    }

    #[test]
    fn stateless_and_stateful_splits() {
        // `stateless` pairs a configuration with unit state.
        let mut stateless = Split::stateless(Offset(3.0f32));
        assert_eq!(stateless.process(1.0), 4.0);

        // `stateful` wraps the state so it cannot be confused with a configuration.
        let stateful = Split::stateful(RunningSum);
        let Unsplit(RunningSum) = stateful.state;
        assert_eq!(stateful.config, ());
    }

    #[test]
    fn mutable_view_accessors() {
        let mut frames = [[1.0f32, 2.0], [3.0, 4.0]];
        let mut vm = ViewMut::from_frames(&mut frames);
        assert_eq!(vm.frames(), 2);
        assert_eq!(vm.flat(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(vm.frame(0), &[1.0, 2.0]);
        vm.frame_mut(0)[0] = 9.0;
        vm.as_frames_mut()[1][1] = 8.0;
        assert_eq!(vm.as_frames()[0], [9.0, 2.0]);
        assert_eq!(vm.flat_mut()[3], 8.0);

        let mut flat = [1.0f32, 2.0, 3.0, 4.0];
        let mut lm = ViewMut::<_, LaneMajor, 2>::from_flat(&mut flat, 2);
        assert_eq!(lm.lane(0), &[1.0, 2.0]);
        lm.lane_mut(1)[1] = 7.0;
        assert_eq!(lm.lane(1), &[3.0, 7.0]);
        assert_eq!(lm.frames(), 2);

        let v: View<'_, f32, LaneMajor, 2> = View::from_flat(&flat, 2);
        assert_eq!(v.flat(), &flat[..]);
        assert_eq!(v.frames(), 2);
    }

    /// Swaps the two samples of a frame, in place.
    struct SwapFrame;

    impl SplitProcess<[f32; 2], [f32; 2], ()> for SwapFrame {
        fn process_with_state(&mut self, _state: &mut (), [a, b]: [f32; 2]) -> [f32; 2] {
            [b, a]
        }
    }

    impl SplitInplace<[f32; 2], ()> for SwapFrame {}

    #[test]
    fn fn_split_process_adapts_a_closure() {
        let mut proc = FnSplitProcess(|state: &mut i32, x: i32| {
            *state += x;
            *state
        });
        let mut state = 0;
        assert_eq!(proc.process_with_state(&mut state, 2), 2);
        assert_eq!(proc.process_with_state(&mut state, 3), 5);
    }

    #[test]
    fn chunk_in_out_bridges_a_chunk_stage_to_a_block() {
        // 2 samples in, 1 out: a pairwise sum.
        let mut p = Split::stateless(ChunkInOut::<_, 2, 1>(FnSplitProcess(
            |_: &mut (), [a, b]: [i32; 2]| [a + b],
        )));
        assert_eq!(p.process([1, 2, 3, 4]), [3, 7]);

        // The same stage over a block of frames, one chunk each.
        let x = [[1, 2], [3, 4], [5, 6]];
        let mut y = [[0]; 3];
        p.block(&x, &mut y);
        assert_eq!(y, [[3], [7], [11]]);

        // 1 in, 2 out: a doubling stage, four samples at a time.
        let mut q = Split::stateless(ChunkInOut::<_, 1, 2>(FnSplitProcess(
            |_: &mut (), [a]: [i32; 1]| [a, -a],
        )));
        assert_eq!(q.process([1, 2]), [1, -1, 2, -2]);
    }

    #[test]
    fn chunk_in_out_is_inplace_for_unit_chunks() {
        // 1:1 chunks are the case that stays in place.
        let mut p = Split::stateless(ChunkInOut::<_, 1, 1>(FnSplitProcess(
            |_: &mut (), [a]: [f32; 1]| [-a],
        )));
        let mut xy = [[1.0f32], [2.0], [3.0]];
        p.inplace(&mut xy);
        assert_eq!(xy, [[-1.0], [-2.0], [-3.0]]);
    }

    #[test]
    fn per_frame_runs_a_chunk_stage_once_per_frame() {
        // A frame holds 2 samples; the stage reduces each frame to 1.
        let mut p = Split::stateless(ChunkInOut::<_, 2, 1>(FnSplitProcess(
            |_: &mut (), [a, b]: [i32; 2]| [a + b],
        )))
        .per_frame();

        let x = View::from_frames(&[[1, 2], [3, 4]]);
        let mut y = [[0; 1]; 2];
        p.process_frames(x, ViewMut::from_frames(&mut y));
        assert_eq!(y, [[3], [7]]);
    }

    #[test]
    fn per_frame_can_process_in_place() {
        let mut p = Split::new(PerFrame(SwapFrame), ());
        let mut frames = [[1.0f32, 2.0], [3.0, 4.0]];
        p.inplace_frames(ViewMut::from_frames(&mut frames));
        assert_eq!(frames, [[2.0, 1.0], [4.0, 3.0]]);
    }

    #[test]
    fn buffer_delay_line_matches_its_sample_path() {
        let x: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];

        let mut per_sample = Buffer::<[i32; 3]>::default();
        let mut expected = [0; 7];
        for (slot, v) in expected.iter_mut().zip(x) {
            *slot = per_sample.process(v);
        }
        assert_eq!(expected, [0, 0, 0, 1, 2, 3, 4]);

        // The block form has to agree with the sample-at-a-time one, ring wrap included.
        let mut blocked = Buffer::<[i32; 3]>::default();
        let mut got = [0; 7];
        blocked.block(&x, &mut got);
        assert_eq!(got, expected);
        assert!(!blocked.is_empty());
    }

    #[test]
    fn buffer_delay_line_inplace_matches_the_out_of_place_form() {
        let x: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];

        let mut a = Buffer::<[i32; 3]>::default();
        let mut out_of_place = [0; 7];
        a.block(&x, &mut out_of_place);

        let mut b = Buffer::<[i32; 3]>::default();
        let mut in_place = x;
        b.inplace(&mut in_place);

        assert_eq!(in_place, out_of_place);
    }

    #[test]
    fn buffer_applies_the_delay_to_array_shapes() {
        // One continuous 2-sample delay across the whole stream 1,2,3,4,5,6, so the outputs are
        // 0,0,1,2,3,4 read off three at a time.
        let mut b = Buffer::<[i32; 2]>::default();
        assert_eq!(b.process([1, 2, 3]), [0, 0, 1]);
        assert_eq!(b.process([4, 5, 6]), [2, 3, 4]);
    }

    #[test]
    fn buffer_collects_chunks_and_streams_them_back_with_gaps() {
        // Collect: one `Some` every N samples, `None` in between.
        let mut collect = Buffer::<[i32; 2]>::default();
        let mut y = [None; 5];
        collect.block(&[1, 2, 3, 4, 5], &mut y);
        assert_eq!(y, [None, Some([1, 2]), None, Some([3, 4]), None]);

        // Stream: a `Some` loads a chunk, `None` keeps draining it across the gap.
        let mut stream = Buffer::<[i32; 2]>::default();
        let mut z = [0; 5];
        stream.block(&[Some([1, 2]), None, Some([3, 4]), None, Some([5, 6])], &mut z);
        assert_eq!(z, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn buffer_option_paths_agree_between_process_and_block() {
        let mut per_sample = Buffer::<[i32; 2]>::default();
        let collected: [Option<[i32; 2]>; 4] = core::array::from_fn(|i| per_sample.process(i as i32));
        assert_eq!(collected, [None, Some([0, 1]), None, Some([2, 3])]);

        let mut per_sample = Buffer::<[i32; 2]>::default();
        let inputs = [Some([1, 2]), None, Some([3, 4]), None];
        let streamed: [i32; 4] = core::array::from_fn(|i| per_sample.process(inputs[i]));
        assert_eq!(streamed, [1, 2, 3, 4]);
    }

    #[test]
    fn buffer_delay_agrees_across_repeated_calls() {
        // Two calls, so the second has to re-enter at a non-zero index: the partial-ring head path
        // in both `block` and `inplace`.
        let first: [i32; 5] = [1, 2, 3, 4, 5];
        let second: [i32; 5] = [6, 7, 8, 9, 10];

        let mut reference = Buffer::<[i32; 3]>::default();
        let mut expected = [0; 10];
        for (slot, v) in expected.iter_mut().zip(first.into_iter().chain(second)) {
            *slot = reference.process(v);
        }

        let mut blocked = Buffer::<[i32; 3]>::default();
        let mut got = [0; 10];
        blocked.block(&first, &mut got[..5]);
        blocked.block(&second, &mut got[5..]);
        assert_eq!(got, expected);

        let mut in_place = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut ip = Buffer::<[i32; 3]>::default();
        ip.inplace(&mut in_place[..5]);
        ip.inplace(&mut in_place[5..]);
        assert_eq!(in_place, expected);
    }

    #[test]
    fn buffer_collect_agrees_across_repeated_calls() {
        let first: [i32; 5] = [1, 2, 3, 4, 5];
        let second: [i32; 5] = [6, 7, 8, 9, 10];

        let mut reference = Buffer::<[i32; 3]>::default();
        let mut expected = [None; 10];
        for (slot, v) in expected.iter_mut().zip(first.into_iter().chain(second)) {
            *slot = reference.process(v);
        }

        let mut collected = Buffer::<[i32; 3]>::default();
        let mut got = [None; 10];
        collected.block(&first, &mut got[..5]);
        collected.block(&second, &mut got[5..]);
        assert_eq!(got, expected);
    }
}
