//! Coverage for the streaming biquad state APIs and the CIC resampler.
//!
//! `DirectForm1`/`DirectForm2Transposed` state, the `Biquad`/`BiquadClamp`
//! sample processors, and the generic `CicFilter` were only reachable through
//! paths exercised elsewhere; these tests drive them directly.

use embedded_dsp::filtering::{Biquad, BiquadClamp, DirectForm1, DirectForm2Transposed};
use embedded_dsp::resampling::{CicDec3, CicFilter, CicInt3};

// ─────────────────────────────────────────────────────────────────────────────
// Direct form state
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn direct_form_state_defaults_new_and_reset() {
    let mut df1 = DirectForm1::<f32>::default();
    assert_eq!(df1.xy, [0.0; 4]);
    assert_eq!(df1, DirectForm1::new());

    df1.xy = [1.0, 2.0, 3.0, 4.0];
    df1.reset();
    assert_eq!(df1.xy, [0.0; 4]);

    let mut df2 = DirectForm2Transposed::<f64>::default();
    assert_eq!(df2.s, [0.0; 2]);
    assert_eq!(df2, DirectForm2Transposed::new());

    df2.s = [1.0, 2.0];
    df2.reset();
    assert_eq!(df2.s, [0.0; 2]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Biquad: direct form 1 and direct form 2 transposed
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn biquad_df1_follows_the_documented_recurrence() {
    // y0 = 0.5*x0 + 0.5*x1, i.e. a two-point moving average (a1 = a2 = 0).
    let bq = Biquad::new(0.5f32, 0.5, 0.0, 0.0, 0.0);
    let mut state = DirectForm1::<f32>::new();

    assert_eq!(bq.process_df1(&mut state, 1.0), 0.5);
    assert_eq!(bq.process_df1(&mut state, 2.0), 1.5);
    assert_eq!(bq.process_df1(&mut state, 4.0), 3.0);
    // History is [x0, x1, y0, y1] from the most recent call.
    assert_eq!(state.xy, [4.0, 2.0, 3.0, 1.5]);
}

#[test]
fn biquad_df2t_agrees_with_df1() {
    // Direct Form 1 and Direct Form 2 Transposed are equivalent realisations
    // of the same transfer function, so they must agree sample for sample.
    let ba = Biquad::new(0.5f32, 0.25, 0.125, 0.5, -0.25);
    let mut df1 = DirectForm1::<f32>::new();
    let mut df2t = DirectForm2Transposed::<f32>::new();

    for n in 0..32 {
        let x = (n as f32) * 0.25 - 2.0;
        let y1 = ba.process_df1(&mut df1, x);
        let y2 = ba.process_df2t(&mut df2t, x);
        assert!((y1 - y2).abs() < 1e-5, "step {n}: df1={y1} df2t={y2}");
    }
}

#[test]
fn biquad_f64_forms_agree() {
    let ba = Biquad::new(0.5f64, 0.25, 0.125, 0.5, -0.25);
    let mut df1 = DirectForm1::<f64>::new();
    let mut df2t = DirectForm2Transposed::<f64>::new();

    for n in 0..32 {
        let x = (n as f64) * 0.25 - 2.0;
        let y1 = ba.process_df1(&mut df1, x);
        let y2 = ba.process_df2t(&mut df2t, x);
        assert!((y1 - y2).abs() < 1e-12, "step {n}: df1={y1} df2t={y2}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BiquadClamp: anti-windup clamping
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn biquad_clamp_f32_limits_output() {
    // Pure gain of 2, clamped to +/-1.
    let clamp = BiquadClamp::new(Biquad::new(2.0f32, 0.0, 0.0, 0.0, 0.0), -1.0, 1.0, 0.0);

    let mut df1 = DirectForm1::<f32>::new();
    assert_eq!(clamp.process_df1(&mut df1, 0.25), 0.5);
    assert_eq!(clamp.process_df1(&mut df1, 5.0), 1.0);
    assert_eq!(clamp.process_df1(&mut df1, -5.0), -1.0);

    let mut df2t = DirectForm2Transposed::<f32>::new();
    assert_eq!(clamp.process_df2t(&mut df2t, 0.25), 0.5);
    assert_eq!(clamp.process_df2t(&mut df2t, 5.0), 1.0);
    assert_eq!(clamp.process_df2t(&mut df2t, -5.0), -1.0);
}

#[test]
fn biquad_clamp_f32_applies_the_summing_offset() {
    // Zero coefficients: the output is just the `u` offset, then clamped.
    let clamp = BiquadClamp::new(Biquad::new(0.0f32, 0.0, 0.0, 0.0, 0.0), -0.25, 0.25, 0.5);

    let mut df1 = DirectForm1::<f32>::new();
    assert_eq!(clamp.process_df1(&mut df1, 0.0), 0.25);

    let mut df2t = DirectForm2Transposed::<f32>::new();
    assert_eq!(clamp.process_df2t(&mut df2t, 0.0), 0.25);
}

#[test]
fn biquad_clamp_f64_limits_output() {
    let clamp = BiquadClamp::new(Biquad::new(2.0f64, 0.0, 0.0, 0.0, 0.0), -1.0, 1.0, 0.0);

    let mut df1 = DirectForm1::<f64>::new();
    assert_eq!(clamp.process_df1(&mut df1, 5.0), 1.0);
    assert_eq!(clamp.process_df1(&mut df1, -5.0), -1.0);

    let mut df2t = DirectForm2Transposed::<f64>::new();
    assert_eq!(clamp.process_df2t(&mut df2t, 5.0), 1.0);
    assert_eq!(clamp.process_df2t(&mut df2t, -5.0), -1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// CIC filter
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cic_accessors_report_configuration_and_clear_keeps_rate() {
    let mut filter = CicFilter::<i32, 3, 1>::new(2);
    assert_eq!(filter.order(), 3);
    assert_eq!(filter.comb_delay(), 1);
    assert_eq!(filter.rate(), 2);
    assert!(filter.tick(), "index starts at zero");
    assert_eq!(filter.get_decimate(), 0);
    assert_eq!(filter.get_interpolate(), 0);

    filter.set_rate(4);
    assert_eq!(filter.rate(), 4);

    // A decimation call advances the phase, so the next tick is not due.
    let _ = filter.process_decimate(1);
    assert!(!filter.tick());

    filter.clear();
    assert!(filter.tick(), "clear resets the phase");
    assert_eq!(filter.rate(), 4, "clear preserves the configured rate");
    assert_eq!(filter.get_decimate(), 0);
}

#[test]
fn cic_decimator_zero_input_stays_zero() {
    let mut dec = CicDec3::<i32>::new(2);
    for _ in 0..30 {
        if let Some(y) = dec.process_decimate(0) {
            assert_eq!(y, 0);
        }
    }
}

#[test]
fn cic_decimator_emits_one_output_per_rate_plus_one_samples() {
    let rate = 2u32;
    let mut dec = CicFilter::<i32, 3, 1>::new(rate);
    let mut outputs = 0;
    for _ in 0..(3 * (rate as usize + 1)) {
        if dec.process_decimate(1).is_some() {
            outputs += 1;
        }
    }
    assert_eq!(outputs, 3, "9 inputs at rate 2 decimate to 3 outputs");
}

#[test]
fn cic_decimator_is_linear_and_reports_last_output() {
    let mut one = CicFilter::<i32, 3, 1>::new(2);
    let mut three = CicFilter::<i32, 3, 1>::new(2);
    let mut last = 0;

    for n in 0..24 {
        let x = n % 5 - 2;
        let y1 = one.process_decimate(x);
        let y3 = three.process_decimate(x * 3);
        // Linear, zero-state filter: tripling the input triples the output.
        assert_eq!(y3, y1.map(|v| v * 3), "step {n}");
        if let Some(y) = y1 {
            last = y;
        }
    }
    assert_eq!(one.get_decimate(), last);
}

#[test]
fn cic_interpolator_emits_rate_plus_one_samples_per_input() {
    let rate = 3u32;
    let mut interp: CicInt3<i32> = CicInt3::new(rate);
    let period = rate as usize + 1;

    let mut inputs = 0;
    let mut last_output = 0;
    for _ in 0..(period * 4) {
        let slow = interp.tick();
        if slow {
            inputs += 1;
        }
        last_output = interp.process_interpolate(if slow { Some(1) } else { None });
    }

    assert_eq!(inputs, 4, "one slow input every {period} fast cycles");
    assert_eq!(
        interp.get_interpolate(),
        last_output,
        "accessor must expose the most recent integrator value"
    );
}
