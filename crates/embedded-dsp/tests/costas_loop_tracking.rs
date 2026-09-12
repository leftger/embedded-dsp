//! Regression tests for `CostasLoop` carrier tracking.
//!
//! Before the fix the loop had no arm filter and an unscaled phase detector, so
//! it only tracked within a narrow band of loop bandwidths: it worked around
//! 5-20 Hz and collapsed to ~0 Hz at 50 Hz or above, which is the bandwidth the
//! repository's own test in `dsp_tests.rs` uses. With a 100 Hz carrier and a
//! 50 Hz loop bandwidth the estimate decayed from 100 Hz to 0.000 Hz.
//!
//! These tests use `BW = 50.0` deliberately: that is the configuration that
//! failed, and it is the operating point the rest of the suite exercises.
//!
//! They also pin the properties measured while reworking the loop: the estimate
//! is independent of input amplitude, survives BPSK data inversions, holds
//! across the pull-in range, and does not depend on the carrier-to-sample-rate
//! ratio.

use core::f32::consts::PI;
use embedded_dsp::pll::CostasLoop;

const FS: f32 = 10_000.0;
const FC: f32 = 100.0;
/// Loop bandwidth requested by the failing configuration (`dsp_tests.rs`).
const BW: f32 = 50.0;
const SAMPLES: usize = 30_000;

/// Lock a loop onto a pure tone and return the final frequency estimate.
fn lock_hz(carrier_hz: f32, amplitude: f32) -> f32 {
    lock_at(FC, FS, BW, carrier_hz, amplitude)
}

fn lock_at(center_hz: f32, fs: f32, bw: f32, carrier_hz: f32, amplitude: f32) -> f32 {
    let mut costas = CostasLoop::new(center_hz, fs, bw, 0.707);
    for k in 0..SAMPLES {
        let t = k as f32 / fs;
        costas.process_sample(amplitude * (2.0 * PI * carrier_hz * t).sin());
    }
    costas.frequency_hz()
}

#[test]
fn does_not_collapse_when_fed_the_centre_frequency() {
    // The original bug in one assertion: a 100 Hz carrier into a loop tuned to
    // 100 Hz with a 50 Hz bandwidth drove the estimate to 0.000 Hz.
    let estimate = lock_hz(FC, 1.0);
    assert!(
        estimate > 0.5 * FC,
        "estimate collapsed to {estimate} Hz instead of tracking {FC} Hz"
    );
    assert!(
        (estimate - FC).abs() < 0.01 * FC,
        "estimate {estimate} Hz is not the {FC} Hz carrier"
    );
}

#[test]
fn locks_onto_carriers_across_the_pull_in_range() {
    for carrier in [90.0f32, 95.0, 100.0, 105.0, 110.0] {
        let estimate = lock_hz(carrier, 1.0);
        let error = estimate - carrier;
        assert!(
            error.abs() < 0.01 * carrier,
            "carrier {carrier} Hz: estimate {estimate} Hz is off by {error} Hz"
        );
    }
}

#[test]
fn frequency_estimate_is_amplitude_independent() {
    // The phase detector is normalised, so a 20x smaller input must converge to
    // the same estimate. The shipped loop was erratic here: at a 50 Hz
    // bandwidth it produced 0.0 Hz for a full-scale input but 97.8 Hz for a
    // half-scale one.
    let loud = lock_hz(FC, 1.0);
    let quiet = lock_hz(FC, 0.05);
    assert!(
        (loud - quiet).abs() < 0.05,
        "loud input locked to {loud} Hz but quiet input to {quiet} Hz"
    );
}

#[test]
fn tracks_a_bpsk_modulated_carrier() {
    // 200 baud inversion on a 100 Hz carrier. A Costas detector is blind to the
    // data sign, so the carrier estimate must be unaffected.
    let mut costas = CostasLoop::new(FC, FS, BW, 0.707);
    for k in 0..SAMPLES {
        let t = k as f32 / FS;
        let mut sample = (2.0 * PI * FC * t).sin();
        if (k / 25) % 2 == 1 {
            sample = -sample;
        }
        costas.process_sample(sample);
    }
    let estimate = costas.frequency_hz();
    assert!(
        (estimate - FC).abs() < 0.05 * FC,
        "BPSK carrier estimate {estimate} Hz is not {FC} Hz"
    );
}

#[test]
fn tracking_does_not_depend_on_the_carrier_to_sample_rate_ratio() {
    // A 1 kHz carrier at 48 kHz: the arm filter cutoff scales with the centre
    // frequency, so the design must not be tuned to the 100 Hz/10 kHz case.
    let (fs, center) = (48_000.0f32, 1_000.0f32);
    for carrier in [900.0f32, 1_000.0, 1_100.0] {
        let estimate = lock_at(center, fs, 500.0, carrier, 1.0);
        let error = estimate - carrier;
        assert!(
            error.abs() < 0.01 * carrier,
            "carrier {carrier} Hz at {fs} Hz: estimate {estimate} Hz is off by {error} Hz"
        );
    }
}

#[test]
fn zero_input_leaves_the_estimate_at_the_centre_frequency() {
    let mut costas = CostasLoop::new(FC, FS, BW, 0.707);
    for _ in 0..1_000 {
        // Exercises the zero-power guard in the phase detector.
        assert_eq!(costas.process_sample(0.0), (0.0, 0.0));
    }
    assert!(costas.frequency_hz().is_finite());
    assert!((costas.frequency_hz() - FC).abs() < 1.0e-3);
}

#[test]
fn negative_centre_frequency_tracks_a_negative_carrier() {
    let mut costas = CostasLoop::new(-FC, FS, BW, 0.707);
    assert!(
        costas.frequency_hz() < 0.0,
        "initial estimate must be negative"
    );
    for k in 0..SAMPLES {
        let t = k as f32 / FS;
        costas.process_sample((2.0 * PI * -FC * t).sin());
    }
    let estimate = costas.frequency_hz();
    assert!(
        (estimate + FC).abs() < 0.05 * FC,
        "estimate {estimate} Hz is not the -{FC} Hz carrier"
    );
}

#[test]
fn reset_restores_the_initial_state() {
    let mut costas = CostasLoop::new(FC, FS, BW, 0.707);
    let initial = costas.frequency_hz();

    for k in 0..5_000 {
        let t = k as f32 / FS;
        costas.process_sample((2.0 * PI * (FC + 5.0) * t).sin());
    }
    assert!(
        (costas.frequency_hz() - initial).abs() > 0.1,
        "loop should have moved"
    );

    costas.reset();
    assert_eq!(costas.frequency_hz(), initial);

    // Identical post-reset behaviour to a freshly constructed loop.
    let mut fresh = CostasLoop::new(FC, FS, BW, 0.707);
    for _ in 0..100 {
        assert_eq!(costas.process_sample(0.7), fresh.process_sample(0.7));
    }
}
