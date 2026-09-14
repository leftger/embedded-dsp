//! Consolidated tests: misc_modules_coverage, module_extras_coverage.

use embedded_dsp::basic_math::{scale_q7, scale_q15, scale_q31, shift_q7, shift_q15, shift_q31};
use embedded_dsp::cordic::{cordic_cartesian_to_polar_q15, cordic_sqrt_q15};
use embedded_dsp::distance::{
    bray_curtis_distance_q15, canberra_distance_q15, cosine_distance_q15, cosine_distance_q31,
    euclidean_distance_q15, euclidean_distance_q31,
};
use embedded_dsp::dynamics::{DynamicsCompressor, SafetyLimiter};
use embedded_dsp::pipeline::DspNode;
use embedded_dsp::psd::{WelchWindow, welch_psd_f32};
use embedded_dsp::spatial::{convolve2d_f32, dct2d_f32, histogram_2d_f32, idct2d_f32, mse_2d_f32};
use embedded_dsp::synthesis::{AccuOsc, Sweep, SweepError};
use embedded_dsp::types::{Status, q7, q15, q31};

// ─── from misc_modules_coverage.rs ────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// distance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn distance_functions_handle_empty_slices() {
    assert_eq!(euclidean_distance_q15(&[], &[]), q15::ZERO);
    assert_eq!(euclidean_distance_q31(&[], &[]), q31::ZERO);
    assert_eq!(cosine_distance_q15(&[], &[]), q15::ZERO);
    assert_eq!(cosine_distance_q31(&[], &[]), q31::ZERO);
    assert_eq!(canberra_distance_q15(&[], &[]), q15::ZERO);
    assert_eq!(bray_curtis_distance_q15(&[], &[]), q15::ZERO);
}

#[test]
fn cosine_distance_reports_maximum_for_zero_vectors() {
    // Cosine similarity is undefined for a zero-norm vector; the API reports
    // the maximum distance (1.0) in both Q15 and Q31.
    let zeros15 = [q15::ZERO; 4];
    assert_eq!(
        cosine_distance_q15(&zeros15, &zeros15),
        q15::from_bits(32_767)
    );

    let zeros31 = [q31::ZERO; 4];
    assert_eq!(
        cosine_distance_q31(&zeros31, &zeros31),
        q31::from_bits(i32::MAX)
    );
}

#[test]
fn bray_curtis_distance_is_zero_for_zero_vectors() {
    let zeros = [q15::ZERO; 4];
    assert_eq!(bray_curtis_distance_q15(&zeros, &zeros), q15::ZERO);
}

#[test]
fn cosine_distance_ranks_identical_below_orthogonal_q15() {
    let a = [q15::from_bits(16_384), q15::from_bits(8_192)];
    let identical = [q15::from_bits(16_384), q15::from_bits(8_192)];
    // a · c == 0.5*0.25 + 0.25*(-0.5) == 0, i.e. exactly orthogonal.
    let orthogonal = [q15::from_bits(8_192), q15::from_bits(-16_384)];

    let same = cosine_distance_q15(&a, &identical);
    let orth = cosine_distance_q15(&a, &orthogonal);
    assert!(
        same < orth,
        "identical vectors ({same:?}) must be closer than orthogonal ones ({orth:?})"
    );
}

#[test]
fn cosine_distance_ranks_identical_below_orthogonal_q31() {
    let a = [q31::from_bits(1 << 30), q31::from_bits(1 << 29)];
    let identical = [q31::from_bits(1 << 30), q31::from_bits(1 << 29)];
    let orthogonal = [q31::from_bits(1 << 29), q31::from_bits(-(1 << 30))];

    let same = cosine_distance_q31(&a, &identical);
    let orth = cosine_distance_q31(&a, &orthogonal);
    assert!(
        same < orth,
        "identical vectors ({same:?}) must be closer than orthogonal ones ({orth:?})"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// synthesis
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sweep_derived_quantities_are_consistent() {
    // Positive rate: the sweep runs upward in frequency.
    let sweep = Sweep::new(1, 1i64 << 40);
    assert!(sweep.rate() > 0.0);
    assert!(
        sweep.delay(2.0) > 0.0,
        "upward sweep has a positive harmonic delay"
    );
    assert!(sweep.octave() > 0.0 && sweep.decade() > 0.0);

    assert_eq!(sweep.state(), sweep.cycles() * sweep.rate());
    assert_eq!(sweep.continuous(0.0), sweep.cycles());
}

#[test]
fn sweep_fit_rejects_out_of_range_parameters() {
    // `stop` must lie in 0.0..=0.5 (Nyquist).
    assert_eq!(Sweep::fit(0.75, 1_000.0, 1.0), Err(SweepError::Stop));
    assert_eq!(Sweep::fit(-0.1, 1_000.0, 1.0), Err(SweepError::Stop));

    // A stop frequency this low rounds the rate to zero, leaving a
    // non-positive initial state.
    assert_eq!(Sweep::fit(0.5, 1.0e12, 1.0), Err(SweepError::Start));
}

#[test]
fn sweep_error_display_describes_the_bad_parameter() {
    assert_eq!(
        format!("{}", SweepError::Start),
        "Sweep start parameter out of bounds"
    );
    assert_eq!(
        format!("{}", SweepError::Stop),
        "Sweep stop parameter out of bounds"
    );
}

#[test]
fn accu_osc_starts_with_a_zero_phase_accumulator() {
    let osc = AccuOsc::new(Sweep::new(1, 1i64 << 40));
    assert_eq!(osc.state(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// spatial
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn spatial_transforms_reject_degenerate_shapes() {
    // Zero rows/cols.
    assert_eq!(dct2d_f32(&[], &mut [], 0, 0), Status::LengthError);
    assert_eq!(idct2d_f32(&[], &mut [], 0, 0), Status::LengthError);

    // A zero-sized kernel is an argument error...
    assert_eq!(
        convolve2d_f32(&[], &mut [], 0, 0, &[], 0, 0, false),
        Status::ArgumentError
    );
    // ...while a positive shape with undersized buffers is a length error.
    assert_eq!(
        convolve2d_f32(&[], &mut [], 1, 1, &[], 1, 1, false),
        Status::LengthError
    );
}

#[test]
fn histogram_rejects_empty_input_or_bins() {
    let mut bins = [0usize; 4];
    assert_eq!(
        histogram_2d_f32(&[], &mut bins, 0.0, 1.0),
        Status::ArgumentError
    );

    let mut no_bins: [usize; 0] = [];
    assert_eq!(
        histogram_2d_f32(&[0.5], &mut no_bins, 0.0, 1.0),
        Status::ArgumentError
    );

    // An inverted range is also rejected.
    assert_eq!(
        histogram_2d_f32(&[0.5], &mut bins, 1.0, 0.0),
        Status::ArgumentError
    );
}

#[test]
fn mse_of_empty_images_is_zero() {
    assert_eq!(mse_2d_f32(&[], &[]), 0.0);
    assert_eq!(mse_2d_f32(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
}

// ─── from module_extras_coverage.rs ────────────────────────────────────────
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
