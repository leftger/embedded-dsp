//! Python bindings for `embedded-dsp`, built with PyO3's stable ABI (`abi3-py39`).
//!
//! A thin surface over the same verified Rust kernels the C ABI and the Rust API
//! expose: the audio-EQ designer, FIR, and biquad cascades. Wheels are built by
//! maturin from this crate (`module-name = "embedded_dsp"`).

use embedded_dsp_core::filter_design::{BiquadType, EqFilter};
use embedded_dsp_core::filtering::{BiquadCascadeInstance, FirInstance, biquad_cascade_df1, fir};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Crate version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Maps a response name to a [`BiquadType`].
fn biquad_type(name: &str) -> PyResult<BiquadType> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "lowpass" => BiquadType::Lowpass,
        "highpass" => BiquadType::Highpass,
        "bandpass" => BiquadType::Bandpass,
        "allpass" => BiquadType::Allpass,
        "notch" => BiquadType::Notch,
        "peaking" => BiquadType::Peaking,
        "lowshelf" => BiquadType::Lowshelf,
        "highshelf" => BiquadType::Highshelf,
        "iho" => BiquadType::Iho,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown biquad type {other:?} (expected lowpass, highpass, bandpass, \
                 allpass, notch, peaking, lowshelf, highshelf, or iho)"
            )));
        }
    })
}

/// Designs `[b0, b1, b2, a1, a2]` (Direct Form I) for an audio-EQ response.
#[pyfunction]
#[pyo3(signature = (typ, frequency_hz, sample_rate_hz, q, gain_db = 0.0))]
fn eq_coeffs(
    typ: &str,
    frequency_hz: f32,
    sample_rate_hz: f32,
    q: f32,
    gain_db: f32,
) -> PyResult<Vec<f32>> {
    let kind = biquad_type(typ)?;
    EqFilter::new(frequency_hz, sample_rate_hz)
        .q(q)
        .gain_db(gain_db)
        .try_build(kind)
        .map(|c| c.to_vec())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// FIR-filter `signal` with `coeffs` (zero initial state), returning the block.
#[pyfunction]
fn fir_f32(coeffs: Vec<f32>, signal: Vec<f32>) -> PyResult<Vec<f32>> {
    let taps = coeffs.len();
    if taps == 0 || taps > u16::MAX as usize {
        return Err(PyValueError::new_err("coeffs must hold 1..=65535 taps"));
    }
    let mut state = vec![0.0f32; taps];
    let mut dst = vec![0.0f32; signal.len()];
    let mut instance = FirInstance::<f32> {
        num_taps: taps as u16,
        coeffs: &coeffs,
        state: &mut state,
    };
    fir(&mut instance, &signal, &mut dst);
    Ok(dst)
}

/// Direct Form I biquad cascade over `signal`.
///
/// `coeffs` holds `5 * num_stages` values `[b0, b1, b2, a1, a2]` per stage.
#[pyfunction]
fn biquad_cascade_f32(coeffs: Vec<f32>, signal: Vec<f32>) -> PyResult<Vec<f32>> {
    if coeffs.is_empty() || coeffs.len() % 5 != 0 {
        return Err(PyValueError::new_err(
            "coeffs must hold 5 * num_stages values",
        ));
    }
    let stages = coeffs.len() / 5;
    if stages > u8::MAX as usize {
        return Err(PyValueError::new_err("too many stages"));
    }
    let mut state = vec![0.0f32; stages * 4];
    let mut dst = vec![0.0f32; signal.len()];
    let mut instance = BiquadCascadeInstance::<f32> {
        num_stages: stages as u8,
        post_shift: 0,
        coeffs: &coeffs,
        state: &mut state,
    };
    biquad_cascade_df1(&mut instance, &signal, &mut dst);
    Ok(dst)
}

/// The `embedded_dsp` Python module.
#[pymodule]
fn embedded_dsp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(eq_coeffs, m)?)?;
    m.add_function(wrap_pyfunction!(fir_f32, m)?)?;
    m.add_function(wrap_pyfunction!(biquad_cascade_f32, m)?)?;
    m.add(
        "BIQUAD_TYPES",
        [
            "lowpass",
            "highpass",
            "bandpass",
            "allpass",
            "notch",
            "peaking",
            "lowshelf",
            "highshelf",
            "iho",
        ],
    )?;
    Ok(())
}
