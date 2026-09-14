//! Normal-form (Rader-Gold / Chamberlain) second-order sections.

use crate::types::*;
// With `std` linked, `f32::{sqrt, cos, sin}` are inherent; without it the
// `FloatMath` trait supplies them.
#[cfg(not(feature = "std"))]
use crate::math::FloatMath;

// ─────────────────────────────────────────────────────────────────────────────
// Normal Form Second-Order Section (Rader-Gold / Chamberlain oscillator)
// ─────────────────────────────────────────────────────────────────────────────

/// Normal form (Rader-Gold / Chamberlain) second-order IIR section with an
/// **arbitrary numerator**.
///
/// Unlike a standard direct-form biquad, the normal form has **constant pole
/// resolution** everywhere in the z-plane rather than clustering resolution
/// near the real axis. This makes it ideal for:
///
/// - Precise narrow-band bandpass filters close to DC or Nyquist.
/// - Quadrature sinusoidal oscillators (the two state variables are
///   in-phase and 90°-shifted copies of the oscillation).
/// - Notch filters requiring very high Q.
///
/// # Architecture
///
/// The two state variables `(u, v)` are updated by a rotation through the
/// conjugate pole pair:
///
/// ```text
/// u[n] =  p.re * u[n-1] - p.im * v[n-1] + x[n]
/// v[n] =  p.im * u[n-1] + p.re * v[n-1]
/// ```
///
/// The filtered output is an arbitrary linear combination of the states and
/// the current input:
///
/// ```text
/// y[n] = c0 * u[n] + c1 * v[n] + c2 * x[n]
/// ```
///
/// With `c` chosen by [`NormalForm::from_ba`] this realizes **exactly**
/// `H(z) = (b0 + b1*z⁻¹ + b2*z⁻²) / (a0 + a1*z⁻¹ + a2*z⁻²)`, while keeping
/// the superior pole resolution of the normal form. (This is more general than
/// the `idsp` `Normal` form, whose numerator is forced to `p.im * z⁻¹ * B(z)`.)
///
/// # Example: quadrature NCO
///
/// ```rust
/// # use embedded_dsp::filtering::{NormalForm, NormalFormState};
/// // 1 kHz oscillator at 48 kHz sample rate
/// let f = 1000.0_f32 / 48000.0;
/// let nco = NormalForm::oscillator(f);
/// let mut state = NormalFormState::default();
/// // Kick the oscillator with a unit impulse
/// let (i_out, q_out) = nco.process_quadrature(&mut state, 1.0);
/// assert!(i_out.abs() > 0.0);
/// ```
#[derive(Clone, Debug, Default)]
pub struct NormalForm {
    /// Output combination coefficients `[c0, c1, c2]`:
    /// `y = c0 * u + c1 * v + c2 * x`.
    pub c: [f32; 3],
    /// Conjugate pole pair: `p.re ± j·p.im`.
    pub p: Complex<f32>,
}

/// State for [`NormalForm`]: the two rotating state variables.
#[derive(Clone, Debug, Default)]
pub struct NormalFormState {
    /// Real (in-phase) state variable `u`.
    pub y_re: f32,
    /// Imaginary (quadrature) state variable `v`.
    pub y_im: f32,
}

impl NormalForm {
    /// Construct from raw output-combination coefficients `c` and pole `p`.
    pub fn new(c: [f32; 3], p: Complex<f32>) -> Self {
        Self { c, p }
    }

    /// Construct from a standard `[b; a]` biquad coefficient matrix, exactly
    /// realizing `H(z) = (b0 + b1*z⁻¹ + b2*z⁻²) / (a0 + a1*z⁻¹ + a2*z⁻²)`.
    ///
    /// `ba[0]` = `[b0, b1, b2]` numerator coefficients.  
    /// `ba[1]` = `[a0, a1, a2]` denominator coefficients (a0 usually 1.0).
    ///
    /// # Panics
    ///
    /// Panics if the poles are not a complex-conjugate pair (i.e. the
    /// discriminant `a1² - 4*a0*a2` must be negative).
    pub fn from_ba(ba: &[[f32; 3]; 2]) -> Self {
        let a0_inv = ba[1][0].recip();
        let b = [ba[0][0] * a0_inv, ba[0][1] * a0_inv, ba[0][2] * a0_inv];
        // Roots of a0*z² + a1*z + a2: p = -a1/(2a0) ± sqrt((a1/(2a0))² - a2/a0)
        let p_re = -0.5 * ba[1][1] * a0_inv;
        let disc = p_re * p_re - ba[1][2] * a0_inv;
        assert!(
            disc < 0.0,
            "NormalForm::from_ba: poles must be a complex-conjugate pair (use a direct-form biquad for real poles)"
        );
        let p_im = (-disc).sqrt();
        let r2 = p_re * p_re + p_im * p_im;

        // Solve for the output combination that realizes B(z)/A(z).
        // u = x*(1 - p_re*z⁻¹)/D, v = x*p_im*z⁻¹/D, D = 1 - 2p_re*z⁻¹ + r2*z⁻².
        // y = c0*u + c1*v + c2*x has numerator
        // c0 + c2 + (c1*p_im - c0*p_re - 2*c2*p_re)*z⁻¹ + c2*r2*z⁻².
        let c2 = b[2] / r2;
        let c0 = b[0] - c2;
        let c1 = (b[1] + p_re * b[0] + p_re * c2) / p_im;
        Self {
            c: [c0, c1, c2],
            p: Complex::new(p_re, p_im),
        }
    }

    /// Construct a pure quadrature sinusoidal oscillator at normalised
    /// frequency `f` (0 < f < 0.5, where 0.5 is Nyquist).
    ///
    /// [`NormalForm::process`] returns the in-phase component (`cos`).
    /// [`NormalForm::process_quadrature`] returns both in-phase and 90°-shifted
    /// components. A unit impulse starts the oscillation.
    ///
    /// # Example
    /// ```rust
    /// # use embedded_dsp::filtering::{NormalForm, NormalFormState};
    /// let nco = NormalForm::oscillator(0.1); // 10% of sample rate
    /// let mut s = NormalFormState::default();
    /// let _ = nco.process_quadrature(&mut s, 1.0); // impulse start
    /// ```
    pub fn oscillator(f: f32) -> Self {
        let theta = 2.0 * core::f32::consts::PI * f;
        Self {
            c: [1.0, 0.0, 0.0],
            p: Complex::new(theta.cos(), theta.sin()),
        }
    }

    /// Construct a narrow-band bandpass filter centred at normalised frequency
    /// `f` with quality factor `q`.
    ///
    /// Realizes `H(z) = g*(1 - z⁻²) / (1 - 2*r*cos(θ)*z⁻¹ + r²*z⁻²)` with
    /// `θ = 2πf`, `r = 1 - πf/q`, and `g = 1 - r` (unity passband gain).
    pub fn bandpass(f: f32, q: f32) -> Self {
        let theta = 2.0 * core::f32::consts::PI * f;
        let r = 1.0 - core::f32::consts::PI * f / q; // pole radius ≈ 1 - π·bw/fs
        let g = 1.0 - r; // unity passband normalisation
        let p_re = r * theta.cos();
        let p_im = r * theta.sin();
        let r2 = r * r;
        // from_ba() solution with b = [g, 0, -g]:
        let c2 = -g / r2;
        let c0 = g - c2;
        let c1 = p_re * g * (1.0 - 1.0 / r2) / p_im;
        Self {
            c: [c0, c1, c2],
            p: Complex::new(p_re, p_im),
        }
    }

    /// Advance the internal state by one sample and return the two rotating
    /// state variables `(u, v)`: the in-phase and quadrature components.
    #[inline]
    pub fn process_quadrature(&self, state: &mut NormalFormState, x0: f32) -> (f32, f32) {
        // Normal-form feedback (conjugate pole pair rotation)
        let u = self.p.re() * state.y_re - self.p.im() * state.y_im + x0;
        let v = self.p.im() * state.y_re + self.p.re() * state.y_im;
        state.y_re = u;
        state.y_im = v;
        (u, v)
    }

    /// Process a single sample and return the filtered output `y`.
    #[inline]
    pub fn process(&self, state: &mut NormalFormState, x0: f32) -> f32 {
        let (u, v) = self.process_quadrature(state, x0);
        self.c[0] * u + self.c[1] * v + self.c[2] * x0
    }

    /// Reset state to zero.
    pub fn reset(state: &mut NormalFormState) {
        *state = NormalFormState::default();
    }
}
