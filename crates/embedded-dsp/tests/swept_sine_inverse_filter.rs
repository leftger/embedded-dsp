//! Tests for the swept-sine inverse filter ([`Sweep::inverse_filter`]).
//!
//! An exponential swept sine is the standard stimulus for impulse-response and distortion
//! measurement precisely because deconvolving it costs one complex multiply per FFT bin. The
//! main test drives exactly that loop -- synthesize a sweep, FFT, multiply by the inverse
//! filter, inverse FFT -- and requires the result to be an impulse rather than a smear.
//!
//! The two analytic tests pin the halves of the design separately: the `sqrt(f)` magnitude law
//! that cancels the sweep's `1/sqrt(f)` spectrum, and the phase slope that cancels the sweep's
//! arrival time.

use embedded_dsp::synthesis::Sweep;
use embedded_dsp::transform::cfft_f32;

/// A ~10 octave sweep ending at Nyquist, long enough to be representative but still fast.
const STOP: f32 = 0.5;
const HARMONICS: f32 = 1000.0;
const CYCLES: f32 = 1.0;

fn sweep() -> Sweep {
    Sweep::fit(STOP, HARMONICS, CYCLES).expect("valid sweep parameters")
}

/// Magnitude of the inverse filter at normalized frequency `f`.
fn mag_at(sweep: &Sweep, f: f32) -> f32 {
    let h = sweep.inverse_filter(f);
    h.re().hypot(h.im())
}

/// The value the inverse filter must take at `bin` of an `n`-point FFT of a *real* signal: bins
/// above Nyquist are the conjugate mirror, so the filter is conjugated there too.
fn filter_at(sweep: &Sweep, bin: usize, n: usize) -> (f32, f32) {
    let signed = if bin <= n / 2 {
        bin as f32
    } else {
        bin as f32 - n as f32
    };
    let h = sweep.inverse_filter(signed / n as f32);
    (h.re(), if signed >= 0.0 { h.im() } else { -h.im() })
}

#[test]
fn deconvolving_a_sweep_with_its_inverse_filter_yields_an_impulse() {
    let sweep = sweep();
    let sweep_len = sweep.delay(HARMONICS as f64).ceil() as usize;
    let n = sweep_len.next_power_of_two();

    // Stimulus: the analytic sweep phase, i.e. what `AccuOsc` renders sample by sample.
    let x: Vec<f32> = (0..n)
        .map(|i| {
            if i < sweep_len {
                (core::f64::consts::TAU * sweep.continuous(i as f64)).cos() as f32
            } else {
                0.0
            }
        })
        .collect();

    // A chirp makes a poor impulse: its peak-to-RMS sits near a sine's sqrt(2).
    let raw_rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
    let raw_peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));

    let mut buf: Vec<f32> = x.iter().flat_map(|v| [*v, 0.0]).collect();
    cfft_f32(&mut buf, n, 0, 1);
    for bin in 0..n {
        let (re, im) = (buf[2 * bin], buf[2 * bin + 1]);
        let (hr, hi) = filter_at(&sweep, bin, n);
        buf[2 * bin] = re * hr - im * hi;
        buf[2 * bin + 1] = re * hi + im * hr;
    }
    cfft_f32(&mut buf, n, 1, 1);

    let mag: Vec<f32> = (0..n).map(|i| buf[2 * i].hypot(buf[2 * i + 1])).collect();
    let rms = (mag.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
    let (peak_bin, peak) =
        mag.iter().enumerate().fold(
            (0usize, 0.0f32),
            |(bi, bv), (i, v)| if *v > bv { (i, *v) } else { (bi, bv) },
        );

    // The whole point: deconvolution collapses the chirp into a spike. In practice this lands
    // two orders of magnitude above the raw sweep, so a 10x floor leaves plenty of margin.
    assert!(
        peak / rms > 10.0 * (raw_peak / raw_rms),
        "deconvolved peak/rms {:.1} vs raw sweep {:.2} -- not impulse-like",
        peak / rms,
        raw_peak / raw_rms
    );

    // The 1/8 turn offset baked into the design puts the impulse at t = 0 rather than at the
    // end of the sweep.
    assert!(
        peak_bin <= 4 || peak_bin >= n - 4,
        "expected the impulse at t = 0, found the peak at bin {peak_bin} of {n}"
    );
}

#[test]
fn inverse_filter_magnitude_rises_as_sqrt_f() {
    let sweep = sweep();
    let f0 = sweep.state() as f32;
    let f1 = f0 * HARMONICS;

    // |H(f)| = 2*sqrt(rate*f), so |H(f)|/sqrt(f) is constant across the swept band.
    let reference = mag_at(&sweep, f0) / f0.sqrt();
    let mut f = f0;
    while f <= f1 {
        let ratio = mag_at(&sweep, f) / f.sqrt();
        assert!(
            (ratio / reference - 1.0).abs() < 1e-3,
            "|H({f})|/sqrt(f) = {ratio}, expected {reference}"
        );
        f *= 1.5;
    }
}

#[test]
fn inverse_filter_phase_slope_is_the_sweep_arrival_time() {
    let sweep = sweep();
    let f0 = sweep.state();
    let f = f0 * 500.0;
    // Large enough that the phase step dwarfs f32 rounding in the (large) absolute phase, small
    // enough that the step stays far inside (-pi, pi] and needs no unwrapping.
    let df = 1e-5f64;

    let phase = |f: f64| {
        let h = sweep.inverse_filter(f as f32);
        (h.im().atan2(h.re())) as f64
    };
    let mut step = phase(f + df / 2.0) - phase(f - df / 2.0);
    while step > core::f64::consts::PI {
        step -= core::f64::consts::TAU;
    }
    while step < -core::f64::consts::PI {
        step += core::f64::consts::TAU;
    }

    // The sweep's own spectral phase is -2*pi*arrival_time*f, so inverting it means the filter's
    // phase must advance at +arrival_time per unit frequency.
    let slope = step / (core::f64::consts::TAU * df);
    let expected = sweep.delay(f / f0);
    assert!(
        (slope / expected - 1.0).abs() < 0.05,
        "phase slope {slope} vs arrival time {expected}"
    );
}

#[test]
fn inverse_filter_degenerate_arguments_return_zero_not_nan() {
    let sweep = sweep();
    for f in [0.0f32, -0.25, f32::NAN] {
        let h = sweep.inverse_filter(f);
        assert_eq!((h.re(), h.im()), (0.0, 0.0), "f = {f}");
    }

    // Band edges and Nyquist stay finite.
    for f in [1e-6f32, 0.5, 1.0] {
        let h = sweep.inverse_filter(f);
        assert!(h.re().is_finite() && h.im().is_finite(), "f = {f}");
    }

    // A zero-rate sweep has no exponential law to invert.
    let degenerate = Sweep::new(0, 0);
    let h = degenerate.inverse_filter(0.1);
    assert_eq!((h.re(), h.im()), (0.0, 0.0));
}
