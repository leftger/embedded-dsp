//! Frequency-response, group-delay, and pole-based stability analysis for filters
//! produced by [`crate::filter_design`] or hand-written FIR/biquad coefficients.
//!
//! These routines evaluate the DTFT `H(e^{jω})` of a coefficient set directly (no FFT
//! required), so a filter design can be inspected at arbitrary frequencies before it is
//! deployed to a real-time processing path.

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::types::Complex;

// --- Frequency Response (DTFT evaluation, H(e^{jw})) ---

/// Evaluates the frequency response `H(e^{jω}) = Σ h[k] e^{-jkω}` of an FIR filter (or any
/// raw coefficient sequence) at a single normalized frequency `freq_norm` (cycles/sample,
/// `0.0..=0.5`, where `0.5` is Nyquist).
pub fn fir_frequency_response(taps: &[f32], freq_norm: f32) -> Complex<f32> {
    let omega = 2.0 * core::f32::consts::PI * freq_norm;
    let mut real = 0.0f32;
    let mut imag = 0.0f32;
    for (k, &tap) in taps.iter().enumerate() {
        let angle = omega * k as f32;
        real += tap * angle.cos();
        imag -= tap * angle.sin();
    }
    Complex::new(real, imag)
}

/// Evaluates the frequency response `H(e^{jω})` of a single Direct Form I biquad section
/// `[b0, b1, b2, a1, a2]` (as produced by [`crate::filter_design`] and consumed by
/// [`crate::filtering::biquad_cascade_df1_f32`], where `y(n) = b0 x(n) + b1 x(n-1) + b2 x(n-2)
/// + a1 y(n-1) + a2 y(n-2)`) at a single normalized frequency `freq_norm` (cycles/sample,
/// `0.0..=0.5`).
pub fn biquad_frequency_response(coeffs: &[f32; 5], freq_norm: f32) -> Complex<f32> {
    let omega = 2.0 * core::f32::consts::PI * freq_norm;
    let cos1 = omega.cos();
    let sin1 = omega.sin();
    let cos2 = (2.0 * omega).cos();
    let sin2 = (2.0 * omega).sin();

    let num = Complex::new(
        coeffs[0] + coeffs[1] * cos1 + coeffs[2] * cos2,
        -coeffs[1] * sin1 - coeffs[2] * sin2,
    );
    // Denominator of H(z) = 1 - a1*z^-1 - a2*z^-2, matching the recurrence's sign convention.
    let den = Complex::new(
        1.0 - coeffs[3] * cos1 - coeffs[4] * cos2,
        coeffs[3] * sin1 + coeffs[4] * sin2,
    );
    complex_divide(num, den)
}

/// Evaluates the combined frequency response of a cascade of Direct Form I biquad sections
/// (`coeffs.len()` must be a multiple of 5, as produced by e.g.
/// [`crate::filter_design::butterworth_lowpass_biquads`]) at a single normalized frequency
/// `freq_norm` (cycles/sample, `0.0..=0.5`).
pub fn biquad_cascade_frequency_response(coeffs: &[f32], freq_norm: f32) -> Complex<f32> {
    let mut total = Complex::new(1.0f32, 0.0f32);
    for stage in coeffs.chunks_exact(5) {
        let section: [f32; 5] = [stage[0], stage[1], stage[2], stage[3], stage[4]];
        total = complex_multiply(total, biquad_frequency_response(&section, freq_norm));
    }
    total
}

/// Returns the linear magnitude `|H(e^{jω})|` of a complex frequency-response value.
pub fn response_magnitude(h: Complex<f32>) -> f32 {
    (h.real * h.real + h.imag * h.imag).sqrt()
}

/// Returns the magnitude of a complex frequency-response value in decibels: `20 * log10(|H|)`.
pub fn response_magnitude_db(h: Complex<f32>) -> f32 {
    20.0 * response_magnitude(h).max(1e-20).log10()
}

/// Returns the phase (argument) of a complex frequency-response value, in radians, wrapped to
/// `(-π, π]`.
pub fn response_phase(h: Complex<f32>) -> f32 {
    h.imag.atan2(h.real)
}

fn complex_multiply(a: Complex<f32>, b: Complex<f32>) -> Complex<f32> {
    Complex::new(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real,
    )
}

fn complex_divide(a: Complex<f32>, b: Complex<f32>) -> Complex<f32> {
    let denom = b.real * b.real + b.imag * b.imag;
    if denom < 1e-20 {
        return Complex::new(0.0, 0.0);
    }
    let inv_denom = 1.0 / denom;
    Complex::new(
        (a.real * b.real + a.imag * b.imag) * inv_denom,
        (a.imag * b.real - a.real * b.imag) * inv_denom,
    )
}

// --- Group Delay (Discrete-Time Fourier Transform, linear-phase analysis) ---

/// Computes the group delay (in samples), `τ(ω) = Re[B(e^{jω}) / H(e^{jω})]` where
/// `B(e^{jω}) = Σ k·h[k]·e^{-jkω}`, of an FIR filter at a single normalized frequency
/// `freq_norm` (cycles/sample, `0.0..=0.5`). For a linear-phase (symmetric) FIR of length `M`,
/// this is constant and equal to `(M - 1) / 2` at every frequency.
pub fn fir_group_delay(taps: &[f32], freq_norm: f32) -> f32 {
    let omega = 2.0 * core::f32::consts::PI * freq_norm;
    let mut h_re = 0.0f32;
    let mut h_im = 0.0f32;
    let mut b_re = 0.0f32;
    let mut b_im = 0.0f32;
    for (k, &tap) in taps.iter().enumerate() {
        let n = k as f32;
        let angle = omega * n;
        let c = angle.cos();
        let s = angle.sin();
        h_re += tap * c;
        h_im -= tap * s;
        b_re += n * tap * c;
        b_im -= n * tap * s;
    }
    let denom = h_re * h_re + h_im * h_im;
    if denom < 1e-20 {
        return 0.0;
    }
    (b_re * h_re + b_im * h_im) / denom
}

// --- Pole-Based Stability Analysis (Z-transform) ---

/// Computes the pole radius (largest pole magnitude on the z-plane) of a single Direct Form I
/// biquad section `[b0, b1, b2, a1, a2]`, whose poles are the roots of
/// `z^2 - a1*z - a2 = 0`. A causal LTI system is stable if and only if all poles lie strictly
/// inside the unit circle (`pole_radius < 1.0`).
pub fn biquad_pole_radius(coeffs: &[f32; 5]) -> f32 {
    let a1 = coeffs[3];
    let a2 = coeffs[4];
    let discriminant = a1 * a1 + 4.0 * a2;
    if discriminant >= 0.0 {
        let sqrt_d = discriminant.sqrt();
        let p1 = (a1 + sqrt_d) / 2.0;
        let p2 = (a1 - sqrt_d) / 2.0;
        p1.abs().max(p2.abs())
    } else {
        // Complex-conjugate pole pair: |pole|^2 equals the product of the roots, -a2.
        (-a2).sqrt()
    }
}

/// Returns `true` if the single biquad section `[b0, b1, b2, a1, a2]` is stable, i.e. both
/// poles lie strictly inside the unit circle.
pub fn biquad_is_stable(coeffs: &[f32; 5]) -> bool {
    biquad_pole_radius(coeffs) < 1.0
}

/// Returns `true` if every stage of a biquad cascade (`coeffs.len()` a multiple of 5) is
/// stable.
pub fn biquad_cascade_is_stable(coeffs: &[f32]) -> bool {
    coeffs
        .chunks_exact(5)
        .all(|stage| biquad_is_stable(&[stage[0], stage[1], stage[2], stage[3], stage[4]]))
}
