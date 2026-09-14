//! Lock-in demodulation and the standalone lock-in amplifier.

use crate::types::*;
// Only the `not(fast-math)` branch calls `FloatMath::{cos, sin}` directly; with
// `fast-math` on it uses `fast_math::cossin*` and this import would be unused.
#[cfg(not(feature = "fast-math"))]
use crate::math::FloatMath;
use super::recursive::SinglePoleFilter;

// ─────────────────────────────────────────────────────────────────────────────
// Lock-in Demodulation & Lock-in Amplifier
// ─────────────────────────────────────────────────────────────────────────────


/// Dual-phase lock-in amplifier mixer and demodulator.
///
/// Combines channel filters `C` with an IQ local oscillator reference to demodulate
/// a noisy input signal into in-phase $I$ and quadrature $Q$ components.
#[derive(Copy, Clone, Default, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Lockin<C>(pub C);

impl<C> Lockin<C> {
    /// Create a new lock-in demodulator with the given low-pass channel filter.
    pub const fn new(filter: C) -> Self {
        Self(filter)
    }
}

#[cfg(feature = "pipeline")]
impl<X, U, C, S> crate::pipeline::SplitProcess<(X, Complex<U>), Complex<X>, [S; 2]> for Lockin<C>
where
    X: Copy + core::ops::Mul<U, Output = X>,
    U: Copy,
    C: crate::pipeline::SplitProcess<X, X, S>,
{
    /// Demodulate a sample `x.0` against a local oscillator `x.1` (in-phase and quadrature).
    #[inline]
    fn process_with_state(&mut self, state: &mut [S; 2], x: (X, Complex<U>)) -> Complex<X> {
        let (sample, lo) = x;
        Complex::new(
            self.0.process_with_state(&mut state[0], sample * lo.real),
            self.0.process_with_state(&mut state[1], sample * lo.imag),
        )
    }
}

/// Standalone Lock-in Amplifier with integrated single-pole low-pass filtering.
///
/// Multiplies an incoming signal with an internal or external quadrature reference,
/// and low-pass filters both channels to extract amplitude and phase.
#[derive(Clone, Copy, Debug)]
pub struct LockinAmplifier {
    /// Filter i.
    pub filter_i: SinglePoleFilter<f32>,
    /// Filter q.
    pub filter_q: SinglePoleFilter<f32>,
    /// Phase.
    pub phase: i32,
    /// Phase inc.
    pub phase_inc: i32,
}

impl LockinAmplifier {
    /// Create a new Lock-in Amplifier with carrier frequency, sample rate, and low-pass decay factor.
    pub fn new(carrier_hz: f32, sample_rate: f32, filter_decay: f32) -> Self {
        let phase_inc = ((carrier_hz / sample_rate) * 4294967296.0) as i32;
        Self {
            filter_i: SinglePoleFilter::<f32>::lowpass(filter_decay),
            filter_q: SinglePoleFilter::<f32>::lowpass(filter_decay),
            phase: 0,
            phase_inc,
        }
    }

    /// Set carrier frequency in Hz.
    pub fn set_frequency(&mut self, carrier_hz: f32, sample_rate: f32) {
        self.phase_inc = ((carrier_hz / sample_rate) * 4294967296.0) as i32;
    }

    /// Reset internal filter state and phase accumulator.
    pub fn reset(&mut self) {
        self.filter_i.reset();
        self.filter_q.reset();
        self.phase = 0;
    }

    /// Ingest a sample and return demodulated IQ `Complex<f32>`.
    #[inline]
    pub fn process(&mut self, sample: f32) -> Complex<f32> {
        let (cos_ref, sin_ref) = {
            #[cfg(feature = "fast-math")]
            {
                let (c, s) = crate::fast_math::cossin(self.phase);
                (c as f32 * (1.0 / 2147483648.0), s as f32 * (1.0 / 2147483648.0))
            }
            #[cfg(not(feature = "fast-math"))]
            {
                let rad = self.phase as f32 * (core::f32::consts::PI / 2147483648.0);
                (FloatMath::cos(rad), FloatMath::sin(rad))
            }
        };

        self.phase = self.phase.wrapping_add(self.phase_inc);

        let i_filt = self.filter_i.process(sample * cos_ref);
        let q_filt = self.filter_q.process(sample * sin_ref);
        Complex::new(i_filt, q_filt)
    }

    /// Process a sample using an external reference phase angle (in radians).
    #[inline]
    pub fn process_with_phase(&mut self, sample: f32, phase_rad: f32) -> Complex<f32> {
        let (cos_ref, sin_ref) = {
            #[cfg(feature = "fast-math")]
            {
                crate::fast_math::cossin_f32(phase_rad)
            }
            #[cfg(not(feature = "fast-math"))]
            {
                (FloatMath::cos(phase_rad), FloatMath::sin(phase_rad))
            }
        };

        let i_filt = self.filter_i.process(sample * cos_ref);
        let q_filt = self.filter_q.process(sample * sin_ref);
        Complex::new(i_filt, q_filt)
    }
}
