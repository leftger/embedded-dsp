//! Coverage for guard/edge branches in `distance`, `synthesis`, and `spatial`.
//!
//! These are the degenerate-input rejections and small accessors that the
//! existing suites left uncovered, plus the normal path of the Q15/Q31 cosine
//! distance.

use embedded_dsp::distance::{
    bray_curtis_distance_q15, canberra_distance_q15, cosine_distance_q15, cosine_distance_q31,
    euclidean_distance_q15, euclidean_distance_q31,
};
use embedded_dsp::spatial::{convolve2d_f32, dct2d_f32, histogram_2d_f32, idct2d_f32, mse_2d_f32};
use embedded_dsp::synthesis::{AccuOsc, Sweep, SweepError};
use embedded_dsp::types::{Status, q15, q31};

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
