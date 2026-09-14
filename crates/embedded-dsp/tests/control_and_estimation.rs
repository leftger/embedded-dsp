//! Consolidated tests: types_controller_coverage, pll_kalman_coverage.

use embedded_dsp::controller::{PidAction, PidBuilder, PidError};
use embedded_dsp::kalman::EkfModel;
use embedded_dsp::pipeline::{DspNode, Process, Split, SplitProcess};
use embedded_dsp::pll::{ClampWrap, CostasLoop, IntPll, IntPllState, Rpll, RpllConfig, SogiPll};
use embedded_dsp::types::{DspSample, q15, q31};

// ─── from types_controller_coverage.rs ────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// DspSample impls (present in both `fixed` configurations)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dsp_sample_saturating_ops_for_float_types() {
    assert_eq!(<f32 as DspSample>::sat_div(1.0, 4.0), 0.25);
    assert_eq!(<f64 as DspSample>::to_f32(2.5), 2.5);
}

#[test]
fn dsp_sample_saturating_ops_for_q15_and_q31() {
    let one = q15::from_bits(32_767);
    let half = q15::from_bits(16_384);
    let _ = <q15 as DspSample>::sat_mul(half, one);
    let _ = <q15 as DspSample>::sat_div(half, one);
    assert!((<q15 as DspSample>::to_f32(half) - 0.5).abs() < 1e-3);

    let one31 = q31::from_bits(i32::MAX);
    let half31 = q31::from_bits(1 << 30);
    let _ = <q31 as DspSample>::sat_mul(half31, one31);
    let _ = <q31 as DspSample>::sat_div(half31, one31);
    assert!((<q31 as DspSample>::to_f32(half31) - 0.5).abs() < 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// PID builder
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn pid_builder_accessors_chain_into_a_valid_configuration() {
    let builder = PidBuilder::default()
        .gain(PidAction::P, 1.0)
        .limit(PidAction::I, 0.5)
        .kd2(0.25)
        .limit_d2(0.125)
        .offset(0.75)
        .output_limits(-1.0, 1.0);

    assert!(builder.validate(0.001).is_ok());
}

#[test]
fn pid_builder_validate_rejects_non_finite_period() {
    assert!(matches!(
        PidBuilder::default().validate(f32::NAN),
        Err(PidError::NonFinite("period"))
    ));
}

#[test]
fn pid_builder_validate_rejects_non_finite_limit() {
    assert!(matches!(
        PidBuilder::default()
            .limit(PidAction::I, f32::NAN)
            .validate(0.001),
        Err(PidError::NonFinite("limit"))
    ));
}

#[test]
fn pid_builder_validate_rejects_zero_limit() {
    assert!(matches!(
        PidBuilder::default()
            .limit(PidAction::D, 0.0)
            .validate(0.001),
        Err(PidError::NonPositive("limit"))
    ));
}

#[test]
fn pid_builder_validate_rejects_inverted_output_limits() {
    assert!(matches!(
        PidBuilder::default()
            .output_limits(1.0, -1.0)
            .validate(0.001),
        Err(PidError::InvertedRange("output_limits"))
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// Fallback fixed-point types (`fixed` feature off)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(feature = "fixed"))]
mod fallback_fixed_point {
    use embedded_dsp::types::{FixedNum, I16F16, q15};

    #[test]
    fn fixed_num_f64_conversion_edge_cases() {
        // NaN maps to zero.
        assert_eq!(
            <f64 as FixedNum>::to_raw_fixed(f64::NAN, 8, -1000, 1000, false),
            0
        );

        // Saturating conversion clamps to the low rail.
        assert_eq!(
            <f64 as FixedNum>::to_raw_fixed(-1.0e9, 8, -100, 100, true),
            -100
        );
        assert_eq!(
            <f64 as FixedNum>::to_raw_fixed(1.0e9, 8, -100, 100, true),
            100
        );

        // Exact .5 tie with an odd integer part rounds away from zero.
        // 1.5 / 256 scaled by 256 gives abs_int = 1 (odd) -> rounds to 2.
        let tie = 1.5f64 / 256.0;
        assert_eq!(
            <f64 as FixedNum>::to_raw_fixed(tie, 8, -1000, 1000, true),
            2
        );

        assert_eq!(<f64 as FixedNum>::from_raw_fixed(256, 8), 1.0);
    }

    #[test]
    fn fixed_num_integer_conversion_saturates() {
        assert_eq!(
            <i32 as FixedNum>::to_raw_fixed(1000, 8, -100, 100, true),
            100
        );
        assert_eq!(
            <i32 as FixedNum>::to_raw_fixed(-1000, 8, -100, 100, true),
            -100
        );
        assert_eq!(<i32 as FixedNum>::from_raw_fixed(256, 8), 1);
    }

    #[test]
    fn q15_wrapping_div_and_recip_edge_cases() {
        let half = q15::from_bits(16_384);
        let zero = q15::from_bits(0);

        // Division by zero short-circuits to zero instead of dividing.
        assert_eq!(half.wrapping_div(zero), zero);
        assert_eq!(half.wrapping_div_int(0), zero);

        // Reciprocal of zero saturates to MAX.
        assert_eq!(zero.recip(), q15::MAX);
    }

    #[test]
    fn q15_checked_div_reports_zero_and_overflow() {
        let zero = q15::from_bits(0);
        assert_eq!(q15::from_bits(16_384).checked_div(zero), None);

        // 1.0 / tiny overflows the i16 backing store.
        assert_eq!(q15::from_bits(32_767).checked_div(q15::from_bits(1)), None);

        // 0.25 / 0.5 = 0.5, which fits.
        assert_eq!(
            q15::from_bits(8_192).checked_div(q15::from_bits(16_384)),
            Some(q15::from_bits(16_384))
        );
    }

    #[test]
    fn fallback_fixed_num_scaling_saturates() {
        // q15 has 15 fractional bits; converting into a narrow window clamps.
        assert_eq!(
            q15::from_bits(32_767).to_raw_fixed(15, -100, 100, true),
            100
        );
        assert_eq!(
            q15::from_bits(-32_768).to_raw_fixed(15, -100, 100, true),
            -100
        );

        // Raw values at or below the type's own precision shift left...
        assert_eq!(<q15 as FixedNum>::from_raw_fixed(1, 15), q15::from_bits(1));
        assert_eq!(
            <q15 as FixedNum>::from_raw_fixed(128, 8),
            q15::from_bits(16_384)
        );
        // ...and coarser inputs shift right.
        assert_eq!(<q15 as FixedNum>::from_raw_fixed(32, 20), q15::from_bits(1));
    }

    #[test]
    fn fallback_arithmetic_operators() {
        let a = q15::from_bits(1_000);
        let b = q15::from_bits(200);

        assert_eq!(a + b, q15::from_bits(1_200));
        assert_eq!(a - b, q15::from_bits(800));

        // Q15 multiply: (1000 * 200) >> 15 == 6.
        assert_eq!(a * b, q15::from_bits(6));

        let mut sub = a;
        sub -= b;
        assert_eq!(sub, q15::from_bits(800));

        let mut mul = a;
        mul *= b;
        assert_eq!(mul, q15::from_bits(6));
    }

    #[test]
    fn fallback_display_shows_raw_bits() {
        assert_eq!(format!("{}", q15::from_bits(1_000)), "1000");
        assert_eq!(format!("{}", q15::from_bits(-1)), "-1");
    }

    #[test]
    fn fallback_integer_comparisons() {
        // I16F16 is backed by i32 with 16 fractional bits, so 65536 raw == 1.
        let one = I16F16::from_bits(1 << 16);

        assert!(one == 1i32);
        assert!(1i32 == one);
        assert_eq!(1i32.partial_cmp(&one), Some(core::cmp::Ordering::Equal));

        let two = I16F16::from_bits(2 << 16);
        assert!(1i32 < two);
    }
}

// ─── from pll_kalman_coverage.rs ────────────────────────────────────────
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
    assert!(pll.phase().is_finite());
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
// PLL pipeline composability
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sogi_pll_reaches_dsp_node_through_the_pipeline_blankets() {
    // f32 -> f32 with no external state: the stateless `SplitProcess` bridge should reach
    // `Process` and `DspNode` through the blankets without any bespoke wiring.
    let mut via_trait = SogiPll::new(50.0, 10_000.0, 1.414, 60.0, 1400.0);
    let mut via_inherent = SogiPll::new(50.0, 10_000.0, 1.414, 60.0, 1400.0);
    for n in 0..256 {
        let x = (n as f32 * 0.01).sin();
        assert_eq!(
            DspNode::process_sample(&mut via_trait, x),
            via_inherent.process(x)
        );
    }
}

#[test]
fn costas_loop_reaches_process_but_not_dsp_node() {
    // Output type `(f32, f32)` differs from the `f32` input, so this reaches `Process` but is
    // not eligible for `DspNode` (which requires matching input/output types).
    let mut via_trait = CostasLoop::new(100.0, 10_000.0, 50.0, 0.707);
    let mut via_inherent = CostasLoop::new(100.0, 10_000.0, 50.0, 0.707);
    for n in 0..256 {
        let x = (n as f32 * 0.01).sin();
        assert_eq!(
            Process::process(&mut via_trait, x),
            via_inherent.process_sample(x)
        );
    }
}

#[test]
fn int_pll_split_process_matches_the_inherent_method() {
    // `IntPll` (config) + `IntPllState` (explicit state) is exactly the split shape the trait
    // models: verify the bridge and confirm `Split` lifts it into a self-contained `DspNode`.
    let mut cfg = IntPll::from_bandwidth(5e-2, 4.0);
    let mut state = IntPllState::default();
    let mut state_ref = IntPllState::default();
    let mut node = Split::new(cfg, IntPllState::default());

    for i in 0..256 {
        let input = i * 12_345;
        let expected = cfg.process(&mut state_ref, input);
        assert_eq!(
            SplitProcess::process_with_state(&mut cfg, &mut state, input),
            expected
        );
        assert_eq!(DspNode::process_sample(&mut node, input), expected);
    }
}

#[test]
fn rpll_split_process_matches_the_inherent_method() {
    // Roles are reversed from `IntPll`: `RpllConfig` is `Self`, `Rpll` is the explicit state.
    let mut cfg = RpllConfig {
        dt2: 0,
        shift_frequency: 1,
        shift_phase: 1,
    };
    let mut state = Rpll::default();
    let mut state_ref = Rpll::default();
    let mut node = Split::new(cfg, Rpll::default());

    for i in 0..64i32 {
        let ts = if i % 8 == 0 { Some(i * 1_000) } else { None };
        let expected = state_ref.process(&cfg, ts);
        assert_eq!(
            SplitProcess::process_with_state(&mut cfg, &mut state, ts),
            expected
        );
        assert_eq!(Process::process(&mut node, ts), expected);
    }
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
