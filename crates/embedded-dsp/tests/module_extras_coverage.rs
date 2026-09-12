//! Coverage for shift/scale branches (`basic_math`), CORDIC edge cases,
//! dynamics-processor branches, and the remaining PSD window types.

use embedded_dsp::basic_math::{scale_q7, scale_q15, scale_q31, shift_q7, shift_q15, shift_q31};
use embedded_dsp::cordic::{cordic_cartesian_to_polar_q15, cordic_sqrt_q15};
use embedded_dsp::dynamics::{DynamicsCompressor, SafetyLimiter};
use embedded_dsp::pipeline::DspNode;
use embedded_dsp::psd::{WelchWindow, welch_psd_f32};
use embedded_dsp::types::{Status, q7, q15, q31};

// ─────────────────────────────────────────────────────────────────────────────
// basic_math: shifts beyond the storage width / negative shift counts
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scale_functions_handle_shifts_beyond_the_storage_width() {
    // `shift` larger than the type's bit width makes the residual shift
    // negative, which takes the left-shift branch.
    let mut out31 = [q31::from_bits(0); 2];
    scale_q31(
        &[q31::from_bits(1 << 20); 2],
        q31::from_bits(1 << 20),
        40,
        &mut out31,
    );

    let mut out15 = [q15::from_bits(0); 2];
    scale_q15(
        &[q15::from_bits(1_000); 2],
        q15::from_bits(1_000),
        20,
        &mut out15,
    );

    let mut out7 = [q7::from_bits(0); 2];
    scale_q7(&[q7::from_bits(10); 2], q7::from_bits(10), 10, &mut out7);
}

#[test]
fn shift_functions_handle_two_complement_counts() {
    // Negative counts shift right, positive counts shift left with saturation.
    let mut right31 = [q31::from_bits(0); 2];
    shift_q31(&[q31::from_bits(1_024); 2], -2, &mut right31);
    assert_eq!(right31[0], q31::from_bits(256));

    let mut left15 = [q15::from_bits(0); 2];
    shift_q15(&[q15::from_bits(16); 2], 2, &mut left15);
    assert_eq!(left15[0], q15::from_bits(64));

    // Saturation on the left shift for the narrowest type.
    let mut sat15 = [q15::from_bits(0); 1];
    shift_q15(&[q15::from_bits(20_000)], 4, &mut sat15);
    assert_eq!(sat15[0], q15::from_bits(i16::MAX));

    let mut right7 = [q7::from_bits(0); 2];
    shift_q7(&[q7::from_bits(16); 2], -2, &mut right7);
    assert_eq!(right7[0], q7::from_bits(4));
}

// ─────────────────────────────────────────────────────────────────────────────
// cordic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cordic_cartesian_to_polar_handles_the_origin() {
    assert_eq!(
        cordic_cartesian_to_polar_q15(q15::ZERO, q15::ZERO),
        (q15::ZERO, q15::ZERO)
    );
}

#[test]
fn cordic_cartesian_to_polar_handles_negative_angles() {
    // A negative y component yields a negative angle, taking the second arm of
    // the quadrant-folding branch.
    let positive = cordic_cartesian_to_polar_q15(q15::from_bits(16_384), q15::from_bits(16_384));
    let negative = cordic_cartesian_to_polar_q15(q15::from_bits(16_384), q15::from_bits(-16_384));

    assert!(
        positive.1 > q15::ZERO,
        "positive y should give a positive angle"
    );
    assert!(
        negative.1 < q15::ZERO,
        "negative y should give a negative angle"
    );
    // Magnitude is sign-independent.
    assert_eq!(positive.0, negative.0);
}

#[test]
fn cordic_sqrt_of_zero_is_zero() {
    assert_eq!(cordic_sqrt_q15(q15::ZERO), q15::ZERO);
}

// ─────────────────────────────────────────────────────────────────────────────
// dynamics
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compressor_soft_knee_and_silence_floor() {
    // threshold -20 dB, 4:1, 6 dB knee.
    let mut comp = DynamicsCompressor::new(-20.0, 4.0, 6.0, 0.01, 0.1, 0.0, 48_000.0);

    // Just inside the knee: the soft-knee quadratic branch runs.
    let knee = comp.process(0.1);
    assert!(knee.is_finite());

    // A vanishingly small sample hits the -120 dB silence floor, where the
    // compressor applies no gain reduction and passes the sample through.
    let silence = comp.process(1.0e-9);
    assert!(silence.is_finite());
    assert!(silence.abs() <= 1.0e-8, "got {silence}");

    comp.reset();
    comp.process(0.5);
}

#[test]
fn safety_limiter_release_branch_and_dsp_node() {
    let mut limiter = SafetyLimiter::new(1.0, 0.01, 48_000.0);

    // Loud sample attacks the gain down...
    let loud = limiter.process(10.0);
    assert!(loud.is_finite());
    let after_attack = limiter.current_gain();
    assert!(after_attack < 1.0);

    // ...then a quiet sample relaxes the gain back toward unity.
    let quiet = limiter.process(0.001);
    assert!(quiet.is_finite());
    assert!(limiter.current_gain() >= after_attack);

    // `DspNode` adapters delegate to the inherent `process`.
    let mut via_node = SafetyLimiter::new(1.0, 0.01, 48_000.0);
    let mut direct = SafetyLimiter::new(1.0, 0.01, 48_000.0);
    assert_eq!(
        DspNode::process_sample(&mut via_node, 0.5),
        direct.process(0.5)
    );

    limiter.reset();
}

// ─────────────────────────────────────────────────────────────────────────────
// psd: remaining window types
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn welch_psd_supports_every_window_variant() {
    let src = [0.5f32; 128];
    for window in [
        WelchWindow::Rectangular,
        WelchWindow::Hamming,
        WelchWindow::Hanning,
        WelchWindow::Blackman,
        WelchWindow::BlackmanHarris,
        WelchWindow::Bartlett,
        WelchWindow::Welch,
    ] {
        let mut psd = [0.0f32; 33];
        assert_eq!(
            welch_psd_f32(&src, &mut psd, 64, 32, 1_000.0, window, false),
            Status::Success,
            "window {window:?} failed"
        );
    }
}
