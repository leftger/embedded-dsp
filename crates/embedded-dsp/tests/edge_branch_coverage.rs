//! Coverage for the small argument-validation and default paths that the happy-path suites skip.
//!
//! Converting helpers, the GCC-PHAT estimator, and the quantization-SNR utilities all have early
//! `LengthError` / `ArgumentError` returns and `Default` impls that only fire on inputs the
//! existing tests never produce. This file drives them.

use embedded_dsp::beamforming::{DelayAndSumBeamformer, gcc_phat_tdoa_f32};
use embedded_dsp::filter_analysis::{
    biquad_quantization_snr_db, fir_group_delay, fir_quantization_snr_db,
};
use embedded_dsp::resampling::{
    hbf_dec_response_length, hbf_int_response_length, resample_linear_f32, resample_linear_q15,
    spectral_interpolate_2x_f32,
};
use embedded_dsp::support::{
    barycenter_f32, biquad_coeffs_f32_to_q15, biquad_coeffs_f32_to_q31, fir_taps_f32_to_q15,
};
use embedded_dsp::types::{Status, q15, q31};
use embedded_dsp::validation::evaluate_differential;

#[test]
fn beamformer_default_and_gcc_phat_error_paths() {
    // `Default` just forwards to `new`.
    let _beamformer = DelayAndSumBeamformer::<2, 64>::default();

    // Mismatched channel lengths.
    assert_eq!(
        gcc_phat_tdoa_f32(&[0.0; 4], &[0.0; 8], 1),
        Err(Status::ArgumentError)
    );
    // `max_delay` must stay below n / 2.
    assert_eq!(
        gcc_phat_tdoa_f32(&[0.0; 4], &[0.0; 4], 4),
        Err(Status::ArgumentError)
    );
    // A valid (zero) pair still resolves.
    assert!(gcc_phat_tdoa_f32(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0], 1).is_ok());
}

#[test]
fn support_conversion_error_branches() {
    // Destination shorter than the source.
    assert_eq!(
        fir_taps_f32_to_q15(&[0.1, 0.2], &mut [q15::ZERO; 1]),
        Status::LengthError
    );
    assert_eq!(
        fir_taps_f32_to_q15(&[0.1, 0.2], &mut [q15::ZERO; 2]),
        Status::Success
    );

    // Biquad conversion requires a non-empty, 5-tap-aligned slice.
    assert_eq!(
        biquad_coeffs_f32_to_q15(&[1.0, 0.0, 0.0], &mut [q15::ZERO; 3], 0),
        Status::LengthError
    );
    let mut q15_coeffs = [q15::ZERO; 5];
    assert_eq!(
        biquad_coeffs_f32_to_q15(&[1.0, 0.0, 0.0, 0.0, 0.0], &mut q15_coeffs, 0),
        Status::Success
    );

    assert_eq!(
        biquad_coeffs_f32_to_q31(&[1.0, 0.0, 0.0], &mut [q31::ZERO; 3], 0),
        Status::LengthError
    );
    let mut q31_coeffs = [q31::ZERO; 5];
    assert_eq!(
        biquad_coeffs_f32_to_q31(&[1.0, 0.0, 0.0, 0.0, 0.0], &mut q31_coeffs, 0),
        Status::Success
    );
}

#[test]
fn barycenter_handles_its_input_paths() {
    let points = [0.0f32, 0.0, 2.0, 2.0];
    let weights = [1.0f32, 1.0];
    let mut center = [0.0f32; 2];

    assert_eq!(
        barycenter_f32(&points, &weights, &mut center, 2, 2),
        Status::Success
    );
    assert!((center[0] - 1.0).abs() < 1e-6 && (center[1] - 1.0).abs() < 1e-6);

    // Zero vectors short-circuits without touching the output.
    assert_eq!(
        barycenter_f32(&points, &weights, &mut center, 0, 2),
        Status::Success
    );
}

#[test]
fn filter_analysis_error_and_group_delay_paths() {
    // A symmetric 3-tap FIR has a flat group delay of (N-1)/2 = 1 sample.
    let delay = fir_group_delay(&[0.25, 0.5, 0.25], 0.1);
    assert!((delay - 1.0).abs() < 1e-3, "group delay = {delay}");

    // Length mismatches short-circuit the SNR helpers to 0 dB.
    assert_eq!(
        biquad_quantization_snr_db(&[1.0, 0.0, 0.0], &[q15::ZERO; 3], 0, 64),
        0.0
    );
    assert_eq!(
        fir_quantization_snr_db(&[1.0, 0.5], &[q15::ZERO; 3], 64),
        0.0
    );

    // Valid inputs run the real estimator.
    let mut coeffs = [q15::ZERO; 5];
    assert_eq!(
        biquad_coeffs_f32_to_q15(&[1.0, 0.0, 0.0, 0.0, 0.0], &mut coeffs, 0),
        Status::Success
    );
    let snr = biquad_quantization_snr_db(&[1.0, 0.0, 0.0, 0.0, 0.0], &coeffs, 0, 64);
    assert!(snr.is_finite());
    let fir_snr = fir_quantization_snr_db(&[1.0, 0.5], &[q15::ZERO; 2], 64);
    assert!(fir_snr.is_finite());
}

#[test]
fn resampler_edge_paths() {
    // Linear resamplers: degenerate args return early, and a ratio that overshoots the source
    // exercises the "hold the last sample" clamp.
    let src_q15 = [q15::from_bits(1000), q15::from_bits(2000)];
    let mut dst_q15 = [q15::ZERO; 8];
    resample_linear_q15(&src_q15, &mut dst_q15, 0);
    resample_linear_q15(&src_q15, &mut dst_q15, 3 << 16);

    let src_f32 = [1.0f32, 2.0];
    let mut dst_f32 = [0.0f32; 8];
    resample_linear_f32(&src_f32, &mut dst_f32, 0.0);
    resample_linear_f32(&src_f32, &mut dst_f32, 3.0);

    // Spectral 2x interpolator: all three validation branches.
    let mut short = [0.0f32; 8];
    assert_eq!(
        spectral_interpolate_2x_f32(&[0.0; 3], &mut short),
        Status::ArgumentError
    );
    let mut too_small = [0.0f32; 4];
    assert_eq!(
        spectral_interpolate_2x_f32(&[0.0; 4], &mut too_small),
        Status::LengthError
    );
    let mut big = [0.0f32; 1024];
    assert_eq!(
        spectral_interpolate_2x_f32(&[0.0; 512], &mut big),
        Status::LengthError
    );

    // The deepest supported cascade exercises the tail of both response-length formulas.
    let depth = core::hint::black_box(5usize);
    assert!(hbf_dec_response_length(depth) > 0);
    assert!(hbf_int_response_length(depth) > 0);
}

#[test]
fn differential_metrics_handle_silent_references() {
    // A silent reference with a non-silent target drives every `sum_sq_sig == 0` / `var_sig == 0`
    // arm of the metric computation.
    let metrics = evaluate_differential(&[0.0, 0.0], &[1.0, 1.0]);
    assert_eq!(metrics.sqnr_db, 0.0);
    assert_eq!(metrics.qsnr_db, 0.0);
    assert_eq!(metrics.thd_percent, 0.0);
    assert!(!metrics.is_exact_match);

    // Identical, non-silent signals are an exact match.
    let identical = evaluate_differential(&[0.5, -0.5, 0.25], &[0.5, -0.5, 0.25]);
    assert!(identical.is_exact_match);
    assert_eq!(identical.peak_absolute_error, 0.0);
}

/// The fixed-point `atan2` runs through the CORDIC vectoring path. Its former pre-scale only
/// scaled *up*, so large inputs saturated `i32` during the rotation and the angle could be off
/// by up to ~10 degrees; every entry here was wrong by more than a degree before the fix.
#[test]
fn q31_atan2_is_accurate_across_magnitudes_and_quadrants() {
    use embedded_dsp::fast_math::atan2_q31;

    let vals = [
        1i32,
        -1,
        17,
        -17,
        256,
        -256,
        1 << 16,
        -(1 << 16),
        1 << 24,
        -(1 << 24),
        1 << 29,
        -(1 << 29),
        i32::MAX,
        i32::MIN,
    ];

    let mut worst = 0.0f64;
    let mut worst_at = (0i32, 0i32);
    for &iy in &vals {
        for &ix in &vals {
            let mut out = q31::ZERO;
            assert_eq!(
                atan2_q31(q31::from_bits(iy), q31::from_bits(ix), &mut out),
                Status::Success
            );

            // Result is `atan2(y, x) / pi` in Q1.31.
            let got = out.to_bits() as f64 / 2147483648.0;
            let want = (iy as f64).atan2(ix as f64) / core::f64::consts::PI;
            let err = (got - want).abs();
            if err > worst {
                worst = err;
                worst_at = (iy, ix);
            }
        }
    }

    assert!(
        worst < 1e-6,
        "worst atan2_q31 error {worst} (normalized to pi) at (y={}, x={})",
        worst_at.0,
        worst_at.1
    );
}
