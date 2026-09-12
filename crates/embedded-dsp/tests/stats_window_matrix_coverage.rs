//! Coverage for the guard/error branches of the `statistics`, `window`, and
//! `matrix` modules.
//!
//! These modules validate their inputs (`Status::LengthError`,
//! `Status::SizeMismatch`, ...) and short-circuit degenerate lengths. The
//! happy paths were already exercised; these tests pin down the rejection
//! paths and the min/abs-min index tracking branches.

use embedded_dsp::matrix::{
    MatrixInstance, MatrixInstanceMut, mat_add_f32, mat_add_q15, mat_add_q31, mat_inverse_f32,
    mat_mult_f32, mat_mult_q15, mat_mult_q31, mat_scale_f32, mat_scale_q15, mat_scale_q31,
    mat_sub_f32, mat_sub_q15, mat_sub_q31, mat_trans_f32, mat_trans_q15, mat_trans_q31,
    polynomial_eval_f32, polynomial_least_squares_fit,
};
use embedded_dsp::statistics::{
    absmax_f32, absmin_f32, logsumexp_f32, max_f32, max_q7, max_q15, max_q31, mean_f32, mean_f64,
    mean_q7, mean_q15, mean_q31, min_f32, min_q7, min_q15, min_q31, power_f32, power_q7, power_q15,
    power_q31, rms_f32, rms_q15, rms_q31, std_f32, std_f64, std_q7, std_q15, std_q31, var_f32,
    var_f64, var_q7, var_q15, var_q31,
};
use embedded_dsp::types::{Status, q7, q15, q31, q63};
use embedded_dsp::window::{
    bartlett_f32, blackman_f32, blackman_harris_f32, flattop_f32, hamming_f32, hanning_f32,
    welch_f32,
};

// ─────────────────────────────────────────────────────────────────────────────
// statistics: empty-input rejection
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn descriptive_stats_reject_empty_input() {
    let mut f32_out = 0.0f32;
    let mut f64_out = 0.0f64;
    let mut q31_out = q31::from_bits(0);
    let mut q15_out = q15::from_bits(0);
    let mut q7_out = q7::from_bits(0);
    let mut q63_out: q63 = 0;

    assert_eq!(mean_f32(&[], &mut f32_out), Status::LengthError);
    assert_eq!(mean_f64(&[], &mut f64_out), Status::LengthError);
    assert_eq!(mean_q31(&[], &mut q31_out), Status::LengthError);
    assert_eq!(mean_q15(&[], &mut q15_out), Status::LengthError);
    assert_eq!(mean_q7(&[], &mut q7_out), Status::LengthError);

    assert_eq!(var_f32(&[], &mut f32_out), Status::LengthError);
    assert_eq!(var_f64(&[], &mut f64_out), Status::LengthError);
    assert_eq!(var_q31(&[], &mut q31_out), Status::LengthError);
    assert_eq!(var_q15(&[], &mut q15_out), Status::LengthError);
    assert_eq!(var_q7(&[], &mut q7_out), Status::LengthError);

    assert_eq!(std_f32(&[], &mut f32_out), Status::LengthError);
    assert_eq!(std_f64(&[], &mut f64_out), Status::LengthError);
    assert_eq!(std_q31(&[], &mut q31_out), Status::LengthError);
    assert_eq!(std_q15(&[], &mut q15_out), Status::LengthError);
    assert_eq!(std_q7(&[], &mut q7_out), Status::LengthError);

    assert_eq!(rms_f32(&[], &mut f32_out), Status::LengthError);
    assert_eq!(rms_q31(&[], &mut q31_out), Status::LengthError);
    assert_eq!(rms_q15(&[], &mut q15_out), Status::LengthError);

    assert_eq!(power_f32(&[], &mut f32_out), Status::LengthError);
    assert_eq!(power_q31(&[], &mut q63_out), Status::LengthError);
    assert_eq!(power_q15(&[], &mut q63_out), Status::LengthError);
    assert_eq!(power_q7(&[], &mut q31_out), Status::LengthError);
}

#[test]
fn min_max_and_abs_bounds_reject_empty_input() {
    let mut f32_out = 0.0f32;
    let mut q31_out = q31::from_bits(0);
    let mut q15_out = q15::from_bits(0);
    let mut q7_out = q7::from_bits(0);
    let mut idx = 0usize;

    assert_eq!(min_f32(&[], &mut f32_out, &mut idx), Status::LengthError);
    assert_eq!(max_f32(&[], &mut f32_out, &mut idx), Status::LengthError);
    assert_eq!(min_q31(&[], &mut q31_out, &mut idx), Status::LengthError);
    assert_eq!(max_q31(&[], &mut q31_out, &mut idx), Status::LengthError);
    assert_eq!(min_q15(&[], &mut q15_out, &mut idx), Status::LengthError);
    assert_eq!(max_q15(&[], &mut q15_out, &mut idx), Status::LengthError);
    assert_eq!(min_q7(&[], &mut q7_out, &mut idx), Status::LengthError);
    assert_eq!(max_q7(&[], &mut q7_out, &mut idx), Status::LengthError);
    assert_eq!(absmax_f32(&[], &mut f32_out, &mut idx), Status::LengthError);
    assert_eq!(absmin_f32(&[], &mut f32_out, &mut idx), Status::LengthError);
}

#[test]
fn min_tracking_updates_on_descending_input() {
    // The "new minimum" branch only fires when a later sample is smaller.
    let mut out = 0.0f32;
    let mut idx = usize::MAX;
    assert_eq!(
        min_f32(&[3.0, 2.0, 1.0], &mut out, &mut idx),
        Status::Success
    );
    assert_eq!(out, 1.0);
    assert_eq!(idx, 2);

    let mut out = q31::from_bits(0);
    let mut idx = usize::MAX;
    assert_eq!(
        min_q31(
            &[q31::from_bits(3), q31::from_bits(2), q31::from_bits(1)],
            &mut out,
            &mut idx
        ),
        Status::Success
    );
    assert_eq!(out, q31::from_bits(1));
    assert_eq!(idx, 2);

    let mut out = q15::from_bits(0);
    let mut idx = usize::MAX;
    assert_eq!(
        min_q15(
            &[q15::from_bits(3), q15::from_bits(2), q15::from_bits(1)],
            &mut out,
            &mut idx
        ),
        Status::Success
    );
    assert_eq!(out, q15::from_bits(1));
    assert_eq!(idx, 2);

    let mut out = q7::from_bits(0);
    let mut idx = usize::MAX;
    assert_eq!(
        min_q7(
            &[q7::from_bits(3), q7::from_bits(2), q7::from_bits(1)],
            &mut out,
            &mut idx
        ),
        Status::Success
    );
    assert_eq!(out, q7::from_bits(1));
    assert_eq!(idx, 2);
}

#[test]
fn absmin_tracking_updates_on_decreasing_magnitude() {
    let mut out = 0.0f32;
    let mut idx = usize::MAX;
    assert_eq!(
        absmin_f32(&[3.0, -1.0, 2.0], &mut out, &mut idx),
        Status::Success
    );
    assert_eq!(out, 1.0);
    assert_eq!(idx, 1);
}

#[test]
fn logsumexp_of_empty_slice_is_zero() {
    assert_eq!(logsumexp_f32(&[]), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// window: degenerate lengths
// ─────────────────────────────────────────────────────────────────────────────

/// Every window generator must no-op for `n == 0` and emit `1.0` for `n == 1`.
fn check_window_edges(generate: fn(&mut [f32])) {
    let mut empty: [f32; 0] = [];
    generate(&mut empty);

    let mut single = [0.0f32];
    generate(&mut single);
    assert_eq!(single[0], 1.0);
}

#[test]
fn window_generators_handle_zero_and_unit_lengths() {
    check_window_edges(hanning_f32);
    check_window_edges(hamming_f32);
    check_window_edges(blackman_f32);
    check_window_edges(blackman_harris_f32);
    check_window_edges(bartlett_f32);
    check_window_edges(welch_f32);
    check_window_edges(flattop_f32);
}

// ─────────────────────────────────────────────────────────────────────────────
// matrix: shape and length validation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn matrix_add_and_sub_validate_shapes_and_lengths() {
    let ad = [1.0f32; 6];
    let bd = [1.0f32; 6];
    let mut od = [0.0f32; 6];
    let mut shortd = [0.0f32; 2];
    let badd = [1.0f32; 6];

    let a = MatrixInstance::new(2, 3, &ad);
    let b = MatrixInstance::new(2, 3, &bd);
    let b_wrong_shape = MatrixInstance::new(3, 2, &badd);
    let mut out_bad_shape = MatrixInstanceMut::new(2, 3, &mut od);
    assert_eq!(
        mat_add_f32(&a, &b_wrong_shape, &mut out_bad_shape),
        Status::SizeMismatch
    );

    let mut out_short = MatrixInstanceMut::new(2, 3, &mut shortd);
    assert_eq!(mat_add_f32(&a, &b, &mut out_short), Status::LengthError);
    assert_eq!(mat_sub_f32(&a, &b, &mut out_short), Status::LengthError);
    assert_eq!(
        mat_sub_f32(&a, &b_wrong_shape, &mut out_bad_shape),
        Status::SizeMismatch
    );
}

#[test]
fn matrix_add_and_sub_reject_empty_shapes() {
    // Zero-sized shapes hit the same guard on the q31/q15 paths.
    let ad: [q31; 0] = [];
    let bd: [q31; 0] = [];
    let mut od: [q31; 0] = [];
    let bad: [q31; 1] = [q31::from_bits(0)];

    let a = MatrixInstance::new(0, 0, &ad);
    let b = MatrixInstance::new(0, 0, &bd);
    let b2 = MatrixInstance::new(1, 1, &bad);
    let mut out = MatrixInstanceMut::new(0, 0, &mut od);
    // Matching zero-sized shapes are accepted (nothing to copy).
    assert_eq!(mat_add_q31(&a, &b, &mut out), Status::Success);
    assert_eq!(mat_sub_q31(&a, &b, &mut out), Status::Success);
    assert_eq!(mat_add_q31(&a, &b2, &mut out), Status::SizeMismatch);
    assert_eq!(mat_sub_q31(&a, &b2, &mut out), Status::SizeMismatch);

    let a15d: [q15; 0] = [];
    let b15d: [q15; 1] = [q15::from_bits(0)];
    let mut o15d: [q15; 0] = [];
    let a15 = MatrixInstance::new(0, 0, &a15d);
    let b15 = MatrixInstance::new(1, 1, &b15d);
    let mut o15 = MatrixInstanceMut::new(0, 0, &mut o15d);
    assert_eq!(mat_add_q15(&a15, &b15, &mut o15), Status::SizeMismatch);
    assert_eq!(mat_sub_q15(&a15, &b15, &mut o15), Status::SizeMismatch);
}

#[test]
fn matrix_mult_and_scale_validate_shapes() {
    let ad = [1.0f32; 6];
    let bd = [1.0f32; 6];
    let mut od = [0.0f32; 6];
    let bad = [1.0f32; 6];

    let a = MatrixInstance::new(2, 3, &ad);
    let b = MatrixInstance::new(3, 2, &bd);
    let square = MatrixInstance::new(2, 2, &bad);
    let mut out = MatrixInstanceMut::new(2, 2, &mut od);

    // a(2x3) * b(3x2) needs a 2x2 destination; give it a 2x2 for the happy shape
    // but a mismatched one for the guard.
    let mut small = [0.0f32; 4];
    let mut out_ok = MatrixInstanceMut::new(2, 2, &mut small);
    assert_eq!(mat_mult_f32(&a, &b, &mut out_ok), Status::Success);
    assert_eq!(mat_mult_f32(&a, &square, &mut out), Status::SizeMismatch);
    assert_eq!(mat_scale_f32(&square, 2.0, &mut out), Status::Success);
    assert_eq!(mat_scale_f32(&a, 2.0, &mut out), Status::SizeMismatch);
}

#[test]
fn matrix_mult_and_scale_q31_q15_validate_shapes() {
    let ad = [q31::from_bits(1); 6];
    let bd = [q31::from_bits(1); 6];
    let mut od = [q31::from_bits(0); 4];
    let sq = [q31::from_bits(1); 4];

    let a = MatrixInstance::new(2, 3, &ad);
    let b = MatrixInstance::new(3, 2, &bd);
    let square = MatrixInstance::new(2, 2, &sq);
    let mut out = MatrixInstanceMut::new(2, 2, &mut od);
    // a(2x3) * b(3x2) into 2x2 is the valid shape.
    assert_eq!(mat_mult_q31(&a, &b, &mut out), Status::Success);
    assert_eq!(mat_mult_q31(&a, &square, &mut out), Status::SizeMismatch);
    assert_eq!(
        mat_scale_q31(&a, q31::from_bits(2), 0, &mut out),
        Status::SizeMismatch
    );

    let ad = [q15::from_bits(1); 6];
    let sq = [q15::from_bits(1); 4];
    let mut od = [q15::from_bits(0); 4];
    let a = MatrixInstance::new(2, 3, &ad);
    let square = MatrixInstance::new(2, 2, &sq);
    let mut out = MatrixInstanceMut::new(2, 2, &mut od);
    assert_eq!(mat_mult_q15(&a, &square, &mut out), Status::SizeMismatch);
    assert_eq!(
        mat_scale_q15(&a, q15::from_bits(2), 0, &mut out),
        Status::SizeMismatch
    );
}

#[test]
fn matrix_transpose_validates_shapes() {
    let ad = [1.0f32; 6];
    let mut od = [0.0f32; 6];
    let a = MatrixInstance::new(2, 3, &ad);
    let mut out_same = MatrixInstanceMut::new(2, 3, &mut od);
    assert_eq!(mat_trans_f32(&a, &mut out_same), Status::SizeMismatch);

    let a31 = [q31::from_bits(1); 6];
    let mut o31 = [q31::from_bits(0); 6];
    let a = MatrixInstance::new(2, 3, &a31);
    let mut out_same = MatrixInstanceMut::new(2, 3, &mut o31);
    assert_eq!(mat_trans_q31(&a, &mut out_same), Status::SizeMismatch);

    let a15 = [q15::from_bits(1); 6];
    let mut o15 = [q15::from_bits(0); 6];
    let a = MatrixInstance::new(2, 3, &a15);
    let mut out_same = MatrixInstanceMut::new(2, 3, &mut o15);
    assert_eq!(mat_trans_q15(&a, &mut out_same), Status::SizeMismatch);
}

#[test]
fn matrix_inverse_rejects_bad_shapes_and_oversized_input() {
    // Non-square -> SizeMismatch.
    let ad = [1.0f32; 6];
    let mut od = [0.0f32; 4];
    let a = MatrixInstance::new(2, 3, &ad);
    let mut out = MatrixInstanceMut::new(2, 2, &mut od);
    assert_eq!(mat_inverse_f32(&a, &mut out), Status::SizeMismatch);

    // Zero-sized -> SizeMismatch.
    let z: [f32; 0] = [];
    let mut zo: [f32; 0] = [];
    let zsrc = MatrixInstance::new(0, 0, &z);
    let mut zdst = MatrixInstanceMut::new(0, 0, &mut zo);
    assert_eq!(mat_inverse_f32(&zsrc, &mut zdst), Status::SizeMismatch);

    // Larger than the 16x16 stack limit -> ArgumentError.
    let big = [1.0f32; 17 * 17];
    let mut big_out = [0.0f32; 17 * 17];
    let big_src = MatrixInstance::new(17, 17, &big);
    let mut big_dst = MatrixInstanceMut::new(17, 17, &mut big_out);
    assert_eq!(
        mat_inverse_f32(&big_src, &mut big_dst),
        Status::ArgumentError
    );
}

#[test]
fn polynomial_fit_validates_arguments() {
    let x = [0.0f32, 1.0, 2.0, 3.0];
    let y = [1.0f32, 3.0, 5.0, 7.0];
    let mut coeffs = [0.0f32; 2];

    // Empty input, mismatched y, undersized output, and too few samples.
    assert_eq!(
        polynomial_least_squares_fit(&[], &[], None, 1, &mut coeffs),
        Status::LengthError
    );
    assert_eq!(
        polynomial_least_squares_fit(&x, &[1.0, 2.0], None, 1, &mut coeffs),
        Status::LengthError
    );
    assert_eq!(
        polynomial_least_squares_fit(&x, &y, None, 5, &mut coeffs),
        Status::LengthError
    );
    assert_eq!(
        polynomial_least_squares_fit(&[1.0], &[1.0], None, 3, &mut [0.0; 4]),
        Status::LengthError
    );

    // Weights must match the sample count.
    assert_eq!(
        polynomial_least_squares_fit(&x, &y, Some(&[1.0, 1.0]), 1, &mut coeffs),
        Status::LengthError
    );

    // Degree above the stack-allocated limit. Needs enough samples to clear the
    // earlier `n < degree + 1` length check.
    let x17: Vec<f32> = (0..17).map(|i| i as f32).collect();
    let y17: Vec<f32> = (0..17).map(|i| i as f32).collect();
    assert_eq!(
        polynomial_least_squares_fit(&x17, &y17, None, 16, &mut [0.0; 17]),
        Status::ArgumentError
    );
}

#[test]
fn polynomial_fit_with_weights_recovers_a_line() {
    let x = [0.0f32, 1.0, 2.0, 3.0];
    let y = [1.0f32, 3.0, 5.0, 7.0];
    let w = [1.0f32; 4];
    let mut coeffs = [0.0f32; 2];

    // y = 1 + 2x, with unit weights exercising the weighted code path.
    assert_eq!(
        polynomial_least_squares_fit(&x, &y, Some(&w), 1, &mut coeffs),
        Status::Success
    );
    assert!((coeffs[0] - 1.0).abs() < 1e-4, "c0 = {}", coeffs[0]);
    assert!((coeffs[1] - 2.0).abs() < 1e-4, "c1 = {}", coeffs[1]);
    assert!((polynomial_eval_f32(&coeffs, 2.0) - 5.0).abs() < 1e-4);
}

#[test]
fn polynomial_fit_reports_singular_systems() {
    // Duplicate abscissae make the normal equations rank-deficient.
    let x = [1.0f32, 1.0, 1.0];
    let y = [1.0f32, 2.0, 3.0];
    let mut coeffs = [0.0f32; 2];
    assert_eq!(
        polynomial_least_squares_fit(&x, &y, None, 1, &mut coeffs),
        Status::Singular
    );
}

#[test]
fn polynomial_eval_of_empty_coefficients_is_zero() {
    assert_eq!(polynomial_eval_f32(&[], 5.0), 0.0);
}
