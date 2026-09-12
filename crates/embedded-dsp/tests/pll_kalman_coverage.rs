//! Coverage for the PLL accessors / integer-PLL helpers and the EKF model
//! input-forwarding defaults.
//!
//! The PLL loop bodies were already exercised by the main test suite, but the
//! reporting accessors, the RPLL timestamp path, `ClampWrap`'s low-rail
//! branch, and the `EkfModel::*_with_input` defaults were not.

use embedded_dsp::kalman::EkfModel;
use embedded_dsp::pll::{ClampWrap, CostasLoop, IntPllState, Rpll, RpllConfig, SogiPll};

// ─────────────────────────────────────────────────────────────────────────────
// SOGI-PLL
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sogi_pll_reports_frequency_and_handles_negative_wrap() {
    // A negative centre frequency makes the NCO run backwards, which drives the
    // phase below -π and exercises the low-side phase wrap.
    let mut pll = SogiPll::new(-100.0, 10_000.0, 1.414, 60.0, 1400.0);
    for _ in 0..500 {
        pll.process(0.0);
    }
    assert!(
        pll.frequency_hz() < 0.0,
        "backward-tuned loop should report a negative frequency, got {}",
        pll.frequency_hz()
    );

    let (alpha, beta) = pll.orthogonal_components();
    assert!(alpha.is_finite() && beta.is_finite());
}

#[test]
fn sogi_pll_locks_onto_a_matching_tone() {
    let sample_rate = 10_000.0f32;
    let f = 50.0f32;
    let mut pll = SogiPll::new(f, sample_rate, 1.414, 60.0, 1400.0);

    for n in 0..4_000 {
        let t = n as f32 / sample_rate;
        pll.process((2.0 * core::f32::consts::PI * f * t).sin());
    }
    assert!(
        (pll.frequency_hz() - f).abs() < 5.0,
        "expected to lock near {f} Hz, got {}",
        pll.frequency_hz()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Costas loop
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn costas_loop_accessors_and_negative_wrap() {
    // A strongly backward-tuned NCO drives the phase below -π quickly, which
    // exercises the low-side phase wrap.
    let mut costas = CostasLoop::new(-4_000.0, 10_000.0, 50.0, 0.707);

    // `frequency_hz` reports the current rad/sample estimate; the constructor
    // seeds it from the (negative) centre frequency.
    assert!(costas.frequency_hz() < 0.0);

    for _ in 0..64 {
        let (i, q) = costas.process_sample(1.0);
        assert!(i.is_finite() && q.is_finite());
    }

    assert!(
        (costas.center_frequency_hz() + 4_000.0).abs() < 1e-2,
        "centre frequency should be -4000 Hz, got {}",
        costas.center_frequency_hz()
    );
}

#[test]
fn costas_loop_demodulates_a_bpsk_carrier() {
    let sample_rate = 10_000.0f32;
    let f = 100.0f32;
    let mut costas = CostasLoop::new(f, sample_rate, 50.0, 0.707);

    // The returned I/Q arms are the low-pass filtered mixer outputs. Each is
    // bounded by the input amplitude (1.0) rather than by the instantaneous
    // sample: the arm filter holds energy across the input's zero crossings, so
    // it can exceed |x| at a given step.
    let amplitude = 1.0f32;
    for n in 0..2_000 {
        let t = n as f32 / sample_rate;
        let x = amplitude * (2.0 * core::f32::consts::PI * f * t).sin();
        let (i, q) = costas.process_sample(x);
        assert!(i.abs() <= amplitude + 1e-5, "step {n}: i = {i}");
        assert!(q.abs() <= amplitude + 1e-5, "step {n}: q = {q}");
    }
    assert!((costas.center_frequency_hz() - f).abs() < 1e-3);
}

// ─────────────────────────────────────────────────────────────────────────────
// Integer PLL helpers
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn clamp_wrap_reports_the_low_rail_after_a_forward_wrap() {
    let mut clamp = ClampWrap::default();
    // {i32::MIN -> i32::MAX} is a forward wrap: the wrapped delta looks
    // negative while the value increased, so the output clamps low.
    clamp.process(i32::MIN);
    assert_eq!(clamp.process(i32::MAX), i32::MIN);
    assert_eq!(clamp.dir, -1);
}

#[test]
fn int_pll_state_reports_phase() {
    let state = IntPllState {
        f: 3i64 << 32,
        y: 42,
        ..Default::default()
    };
    assert_eq!(state.phase(), 42);
    assert_eq!(state.frequency(), 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// Reciprocal PLL
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rpll_advances_with_and_without_timestamps() {
    let cfg = RpllConfig {
        dt2: 0,
        shift_frequency: 1,
        shift_phase: 1,
    };
    let mut rpll = Rpll::default();
    assert_eq!(rpll.phase(), 0);
    assert_eq!(rpll.frequency(), 0);

    // With no reference edge only the phase accumulator advances.
    let (y, f) = rpll.process(&cfg, None);
    assert_eq!(rpll.phase(), y);
    assert_eq!(rpll.frequency(), f);

    // Periodic reference edges feed the frequency and phase loops.
    let mut seen_edge = false;
    for i in 0..64 {
        let ts = if i % 8 == 0 { Some(i * 1_000) } else { None };
        seen_edge |= ts.is_some();
        let (y, f) = rpll.process(&cfg, ts);
        assert_eq!((rpll.phase(), rpll.frequency()), (y, f));
    }
    assert!(seen_edge);
}

// ─────────────────────────────────────────────────────────────────────────────
// EKF model input forwarding
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal 2-state / 1-measurement identity model used to exercise the
/// `EkfModel::*_with_input` default methods.
struct IdentityModel;

impl EkfModel<2, 1> for IdentityModel {
    fn f(&self, x: &[f32; 2], _dt: f32, out: &mut [f32; 2]) {
        *out = *x;
    }

    fn h(&self, x: &[f32; 2], out: &mut [f32; 1]) {
        out[0] = x[0];
    }

    fn jacobian_f(&self, _x: &[f32; 2], _dt: f32, out: &mut [[f32; 2]; 2]) {
        *out = [[1.0, 0.0], [0.0, 1.0]];
    }

    fn jacobian_h(&self, _x: &[f32; 2], out: &mut [[f32; 2]; 1]) {
        *out = [[1.0, 0.0]];
    }
}

#[test]
fn ekf_input_variants_defer_to_the_input_free_models() {
    let model = IdentityModel;
    let x = [1.0f32, 2.0];
    let u = [0.5f32];

    let mut f_out = [0.0f32; 2];
    model.f_with_input(&x, &u, 0.1, &mut f_out);
    assert_eq!(f_out, x);

    let mut jac_f = [[0.0f32; 2]; 2];
    model.jacobian_f_with_input(&x, &u, 0.1, &mut jac_f);
    assert_eq!(jac_f, [[1.0, 0.0], [0.0, 1.0]]);

    let mut h_out = [0.0f32; 1];
    model.h_with_input(&x, &u, &mut h_out);
    assert_eq!(h_out, [1.0]);

    let mut jac_h = [[0.0f32; 2]; 1];
    model.jacobian_h_with_input(&x, &u, &mut jac_h);
    assert_eq!(jac_h, [[1.0, 0.0]]);
}
