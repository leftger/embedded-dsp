//! Property tests for the RBJ Audio EQ Cookbook parameter conversions in
//! `embedded_dsp::filter_design`: octave bandwidth (`BW`) and shelf slope (`S`).
//!
//! These lock down the *behavioural* definitions from the cookbook rather than the raw
//! coefficient arithmetic: a bandwidth of `n` octaves must produce a filter whose -3 dB
//! (band-pass / notch) or half-gain (peaking EQ) edges are `n` octaves apart, and `S = 1` must
//! give a monotonic Butterworth-slope shelf whose midpoint at `f0` sits at half its gain.

use embedded_dsp::filter_analysis::{biquad_frequency_response, response_magnitude_db};
use embedded_dsp::filter_design::{
    biquad_bandpass_coeffs, biquad_bandpass_skirt_coeffs, biquad_highshelf_coeffs,
    biquad_lowshelf_coeffs, biquad_notch_coeffs, biquad_peaking_coeffs, biquad_q_from_bw,
    biquad_q_from_shelf_slope,
};

/// -3.0103 dB, the half-power point.
const MINUS_3_DB: f32 = -3.010_3;

fn mag_db(coeffs: [f32; 5], freq: f32, sample_rate: f32) -> f32 {
    response_magnitude_db(biquad_frequency_response(&coeffs, freq / sample_rate))
}

/// Bisects for the frequency in `[lo, hi]` where the magnitude crosses `target_db`.
/// The caller must ensure the endpoints bracket a single crossing.
fn crossing_freq(
    coeffs: [f32; 5],
    sample_rate: f32,
    mut lo: f32,
    mut hi: f32,
    target_db: f32,
) -> f32 {
    let mut g_lo = mag_db(coeffs, lo, sample_rate) - target_db;
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let g_mid = mag_db(coeffs, mid, sample_rate) - target_db;
        if g_lo * g_mid <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            g_lo = g_mid;
        }
    }
    0.5 * (lo + hi)
}

/// Measured octave span of a symmetric band: `log2(f_hi / f_lo)` at the given threshold.
fn measured_octave_span(coeffs: [f32; 5], sample_rate: f32, center: f32, target_db: f32) -> f32 {
    let f_lo = crossing_freq(
        coeffs,
        sample_rate,
        sample_rate * 1e-4,
        center * 0.999,
        target_db,
    );
    let f_hi = crossing_freq(
        coeffs,
        sample_rate,
        center * 1.001,
        sample_rate * 0.4999,
        target_db,
    );
    (f_hi / f_lo).log2()
}

#[test]
fn bw_sets_bandpass_and_notch_octave_bandwidth() {
    let fs = 48_000.0;

    for (f0, bw) in [
        (200.0f32, 0.5f32),
        (1_000.0, 1.0),
        (6_000.0, 1.0),
        (12_000.0, 1.0),
        (300.0, 3.0),
    ] {
        let q = biquad_q_from_bw(bw, f0, fs);
        assert!(q.is_finite() && q > 0.0, "q={q} for f0={f0} bw={bw}");

        let cases = [
            ("band-pass", biquad_bandpass_coeffs(f0, fs, q)),
            ("notch", biquad_notch_coeffs(f0, fs, q)),
        ];
        for (name, coeffs) in cases {
            let actual = measured_octave_span(coeffs, fs, f0, MINUS_3_DB);
            assert!(
                (actual - bw).abs() < 0.05,
                "{name} f0={f0} requested {bw} oct, measured {actual:.4} oct"
            );
        }
    }
}

#[test]
fn bw_sets_peaking_half_gain_bandwidth() {
    let fs = 48_000.0;

    for (f0, bw, gain_db) in [
        (1_000.0f32, 1.0f32, 6.0f32),
        (2_000.0, 2.0, 12.0),
        (6_000.0, 0.5, -9.0),
        (12_000.0, 1.0, 6.0),
    ] {
        let q = biquad_q_from_bw(bw, f0, fs);
        let coeffs = biquad_peaking_coeffs(f0, fs, q, gain_db);

        // Centred exactly on the requested gain; the bandwidth is measured at half of it.
        assert!(
            (mag_db(coeffs, f0, fs) - gain_db).abs() < 0.05,
            "f0={f0}: centre gain {} dB != {gain_db} dB",
            mag_db(coeffs, f0, fs)
        );

        let actual = measured_octave_span(coeffs, fs, f0, gain_db * 0.5);
        assert!(
            (actual - bw).abs() < 0.05,
            "peaking f0={f0} gain={gain_db} dB requested {bw} oct, measured {actual:.4} oct"
        );
    }
}

/// The bilinear-transform correction is what keeps the bandwidth honest near Nyquist; the
/// uncorrected analog-prototype shorthand collapses the band as `f0` rises.
#[test]
fn digital_bw_correction_preserves_width_near_nyquist() {
    let fs = 48_000.0;
    let (f0, bw) = (12_000.0f32, 1.0f32);

    let digital_q = biquad_q_from_bw(bw, f0, fs);
    let analog_q = 1.0 / (2.0 * (0.5 * core::f32::consts::LN_2 * bw).sinh());

    let digital_span = measured_octave_span(
        biquad_bandpass_coeffs(f0, fs, digital_q),
        fs,
        f0,
        MINUS_3_DB,
    );
    let analog_span =
        measured_octave_span(biquad_bandpass_coeffs(f0, fs, analog_q), fs, f0, MINUS_3_DB);

    assert!(
        (digital_span - bw).abs() < 0.05,
        "digital relation should hold near Nyquist, measured {digital_span:.4} oct"
    );
    assert!(
        analog_span < bw - 0.2,
        "analog shorthand should under-state the width, measured {analog_span:.4} oct"
    );
}

#[test]
fn shelf_slope_one_is_butterworth_and_centres_at_half_gain() {
    let fs = 48_000.0;
    let f0 = 1_000.0;

    for gain_db in [6.0f32, 12.0, -6.0, -12.0] {
        let q = biquad_q_from_shelf_slope(1.0, gain_db);
        assert!(
            (q - core::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
            "S=1 should give Q=1/sqrt(2), got {q} for {gain_db} dB"
        );

        let low = biquad_lowshelf_coeffs(f0, fs, q, gain_db);
        let high = biquad_highshelf_coeffs(f0, fs, q, gain_db);

        // The shelf midpoint at f0 is half the shelf gain.
        assert!((mag_db(low, f0, fs) - gain_db * 0.5).abs() < 0.05);
        assert!((mag_db(high, f0, fs) - gain_db * 0.5).abs() < 0.05);

        // A low shelf acts at DC and is transparent at Nyquist; a high shelf is the reverse.
        assert!((mag_db(low, 0.0, fs) - gain_db).abs() < 0.05);
        assert!(mag_db(low, fs * 0.5, fs).abs() < 0.05);
        assert!(mag_db(high, 0.0, fs).abs() < 0.05);
        assert!((mag_db(high, fs * 0.5, fs) - gain_db).abs() < 0.05);
    }
}

#[test]
fn shelf_slope_controls_steepness_and_stays_monotonic() {
    let fs = 48_000.0;
    let f0 = 1_000.0;
    let gain_db = 6.0f32;

    let mut previous_drop = None;
    for slope in [0.3f32, 0.6, 1.0] {
        let q = biquad_q_from_shelf_slope(slope, gain_db);
        let coeffs = biquad_lowshelf_coeffs(f0, fs, q, gain_db);

        // A boost low shelf must fall monotonically from DC to Nyquist for S <= 1.
        let mut previous = f32::INFINITY;
        let mut f = 20.0f32;
        while f < fs * 0.5 {
            let m = mag_db(coeffs, f, fs);
            assert!(
                m <= previous + 1e-3,
                "slope={slope}: non-monotonic at {f} Hz ({m} dB > {previous} dB)"
            );
            previous = m;
            f *= 1.1;
        }

        // Steeper slope => a larger gain drop across the octave around f0.
        let drop = mag_db(coeffs, f0 * 0.5, fs) - mag_db(coeffs, f0 * 2.0, fs);
        if let Some(prev) = previous_drop {
            assert!(
                drop > prev,
                "slope {slope} should drop more than the gentler slope ({drop} vs {prev} dB)"
            );
        }
        previous_drop = Some(drop);
    }
}

/// The cookbook's peaking definition (`A*Q`) makes +N dB and -N dB at identical `Q`/`f0` an
/// exact unity "wire".
#[test]
fn peaking_boost_then_cut_is_a_unity_wire() {
    let fs = 48_000.0;
    let q = 1.2f32;

    // The identity is exact mathematically but only approximate in `f32`: at low `f0` the poles
    // sit close to z = 1 and rounding in the near-cancelling coefficients costs a little
    // accuracy. The bound below covers the sampled grid.
    let mut worst_db = 0.0f32;

    for (f0, gain_db) in [(60.0f32, 6.0f32), (1_000.0, 6.0), (10_000.0, 12.0)] {
        let boost = biquad_peaking_coeffs(f0, fs, q, gain_db);
        let cut = biquad_peaking_coeffs(f0, fs, q, -gain_db);

        let mut f = 5.0f32;
        while f < fs * 0.5 {
            let combined_db = mag_db(boost, f, fs) + mag_db(cut, f, fs);
            worst_db = worst_db.max(combined_db.abs());
            f *= 1.07;
        }
    }

    assert!(
        worst_db < 0.05,
        "worst boost/cut cascade deviation was {worst_db} dB, expected unity"
    );
}

/// Locks the two band-pass variants to the cookbook's distinct gain conventions.
#[test]
fn bandpass_variants_match_their_rbj_conventions() {
    let fs = 48_000.0;
    let f0 = 1_000.0;

    for q in [0.5f32, 1.0, 2.0, 5.0] {
        // Constant 0 dB peak gain.
        let peak = biquad_bandpass_coeffs(f0, fs, q);
        assert!(
            mag_db(peak, f0, fs).abs() < 0.01,
            "q={q}: constant-peak band-pass is {} dB at f0",
            mag_db(peak, f0, fs)
        );

        // Constant skirt gain, so the peak gain equals Q.
        let skirt = biquad_bandpass_skirt_coeffs(f0, fs, q);
        let expected = 20.0 * q.log10();
        assert!(
            (mag_db(skirt, f0, fs) - expected).abs() < 0.01,
            "q={q}: skirt band-pass is {} dB at f0, expected {expected} dB",
            mag_db(skirt, f0, fs)
        );
    }
}

#[test]
fn parameter_converters_stay_finite_on_degenerate_inputs() {
    for (bw, f0, fs) in [
        (0.0f32, 0.0f32, 48_000.0f32),
        (1.0, 24_000.0, 48_000.0), // Nyquist
        (-1.0, 1_000.0, 48_000.0),
        (1.0, -5.0, 48_000.0),
    ] {
        let q = biquad_q_from_bw(bw, f0, fs);
        assert!(
            q.is_finite() && q > 0.0,
            "q_from_bw({bw}, {f0}, {fs}) = {q}"
        );
    }

    for (slope, gain_db) in [(0.0f32, 6.0f32), (-1.0, 6.0), (1.0, 0.0), (0.5, -60.0)] {
        let q = biquad_q_from_shelf_slope(slope, gain_db);
        assert!(
            q.is_finite() && q > 0.0,
            "q_from_shelf_slope({slope}, {gain_db}) = {q}"
        );
    }
}
