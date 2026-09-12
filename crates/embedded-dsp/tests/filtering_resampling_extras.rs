//! Remaining guard/edge coverage for `filtering` and the half-band resamplers.
//!
//! Covers the `median_filter_1d_f32` / `fast_convolve_f32` validation and
//! time-domain fallback, the `Biquad`/`BiquadClamp` `SplitProcess` adapters,
//! and the half-band interpolator reset / decimation cascade.

use embedded_dsp::filtering::{
    Biquad, BiquadClamp, DirectForm1, DirectForm2Transposed, fast_convolve_f32,
    median_filter_1d_f32,
};
use embedded_dsp::pipeline::SplitProcess;
use embedded_dsp::resampling::{HbfDecCascade, HbfInt};
use embedded_dsp::types::Status;

// ─────────────────────────────────────────────────────────────────────────────
// filtering: median filter validation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn median_filter_validates_length_and_window() {
    assert_eq!(
        median_filter_1d_f32(&[], &mut [], 3, 0.0),
        Status::LengthError
    );

    // Even window lengths and zero are rejected.
    assert_eq!(
        median_filter_1d_f32(&[1.0, 2.0], &mut [0.0, 0.0], 4, 0.0),
        Status::ArgumentError
    );
    assert_eq!(
        median_filter_1d_f32(&[1.0, 2.0], &mut [0.0, 0.0], 0, 0.0),
        Status::ArgumentError
    );

    // Odd window within the 63-tap stack limit is accepted.
    let mut out = [0.0f32; 5];
    assert_eq!(
        median_filter_1d_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut out, 3, 0.0),
        Status::Success
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// filtering: FFT convolution and its time-domain fallback
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fast_convolve_validates_inputs() {
    assert_eq!(fast_convolve_f32(&[], &[1.0], &mut []), Status::LengthError);
    assert_eq!(
        fast_convolve_f32(&[1.0], &[1.0], &mut []),
        Status::LengthError
    );
}

#[test]
fn fast_convolve_falls_back_to_time_domain_when_fft_is_too_large() {
    // total_len = 300 + 300 - 1 = 599, so the next power of two is 1024, which
    // exceeds the 512-point stack scratch buffer and takes the time-domain path.
    let signal = [1.0f32; 300];
    let kernel = [1.0f32; 300];
    let mut dst = [0.0f32; 599];

    assert_eq!(
        fast_convolve_f32(&signal, &kernel, &mut dst),
        Status::Success
    );
    // Convolving two runs of 300 ones peaks at 300 in the middle.
    assert!(
        (dst[299] - 300.0).abs() < 1e-3,
        "expected peak 300 at the centre, got {}",
        dst[299]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// filtering: SplitProcess adapters for the biquad state machine
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn biquad_split_process_adapters_delegate_to_the_direct_forms() {
    let biquad32 = Biquad::new(0.5f32, 0.25, 0.125, 0.5, -0.25);
    let clamp32 = BiquadClamp::new(Biquad::new(2.0f32, 0.0, 0.0, 0.0, 0.0), -1.0, 1.0, 0.0);
    let biquad64 = Biquad::new(0.5f64, 0.25, 0.125, 0.5, -0.25);
    let clamp64 = BiquadClamp::new(Biquad::new(2.0f64, 0.0, 0.0, 0.0, 0.0), -1.0, 1.0, 0.0);

    let mut df1_32 = DirectForm1::<f32>::new();
    let mut df2t_32 = DirectForm2Transposed::<f32>::new();
    let mut df1_64 = DirectForm1::<f64>::new();
    let mut df2t_64 = DirectForm2Transposed::<f64>::new();

    // y = b0 * x with zero history.
    assert_eq!(SplitProcess::process(&biquad32, &mut df1_32, 1.0), 0.5);
    assert_eq!(SplitProcess::process(&biquad32, &mut df2t_32, 1.0), 0.5);
    assert_eq!(SplitProcess::process(&biquad64, &mut df1_64, 1.0), 0.5);
    assert_eq!(SplitProcess::process(&biquad64, &mut df2t_64, 1.0), 0.5);

    // Gain of 2 with +/-1 clamps.
    assert_eq!(SplitProcess::process(&clamp32, &mut df1_32, 0.5), 1.0);
    assert_eq!(SplitProcess::process(&clamp32, &mut df2t_32, 0.5), 1.0);
    assert_eq!(SplitProcess::process(&clamp64, &mut df1_64, 0.5), 1.0);
    assert_eq!(SplitProcess::process(&clamp64, &mut df2t_64, 0.5), 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// resampling: half-band interpolator / decimation cascade
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hbf_interpolator_reset_and_cascade_default() {
    let mut interpolator = HbfInt::<3>::new([0.0f32; 3]);
    interpolator.reset();

    let mut cascade = HbfDecCascade::<2>::default();
    cascade.reset();
}

#[test]
fn hbf_dec_cascade_processes_four_and_five_stages() {
    // `process` requires src.len() == (1 << STAGES) * dst.len(); the internal
    // buffer plumbing differs for the 4-stage and 5-stage specialisations.
    let mut four = HbfDecCascade::<4>::new();
    let src4 = [0.0f32; 16];
    let mut dst4 = [0.0f32; 1];
    four.process(&src4, &mut dst4);
    assert_eq!(dst4[0], 0.0);

    let mut five = HbfDecCascade::<5>::new();
    let src5 = [0.0f32; 32];
    let mut dst5 = [0.0f32; 1];
    five.process(&src5, &mut dst5);
    assert_eq!(dst5[0], 0.0);
}
