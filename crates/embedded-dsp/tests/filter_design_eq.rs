//! Consolidated tests: filter_design_eq_params, voxengo_oracle.

use embedded_dsp::filter_analysis::{biquad_frequency_response, response_magnitude_db};
use embedded_dsp::filter_design::{
    biquad_bandpass_coeffs, biquad_bandpass_skirt_coeffs, biquad_highshelf_coeffs,
    biquad_lowshelf_coeffs, biquad_notch_coeffs, biquad_peaking_coeffs, biquad_q_from_bw,
    biquad_q_from_shelf_slope,
};

// ─── from filter_design_eq_params.rs ────────────────────────────────────────
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

// ─── from voxengo_oracle.rs ────────────────────────────────────────
/// Filter family understood by [`cook_biquad_voxengo`].
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// Bell (`gain > 1`) or notch (`gain < 1`); the reference's `BT_PEQ`.
    Peq,
    /// Constant-peak-gain band-pass; the reference's `BT_BPF`.
    Bpf,
}

/// A biquad with an explicit, unnormalised `a0`, as the reference emits it.
#[derive(Clone, Copy, Debug)]
struct Biquad6 {
    b0: f64,
    b1: f64,
    b2: f64,
    a0: f64,
    a1: f64,
    a2: f64,
}

impl Biquad6 {
    /// Magnitude `|H(e^{jω})|` at normalised frequency `freq_norm` (cycles/sample).
    fn magnitude(&self, freq_norm: f64) -> f64 {
        let omega = 2.0 * core::f64::consts::PI * freq_norm;
        let (s1, c1) = omega.sin_cos();
        let (s2, c2) = (2.0 * omega).sin_cos();

        let num_re = self.b0 + self.b1 * c1 + self.b2 * c2;
        let num_im = -(self.b1 * s1 + self.b2 * s2);
        let den_re = self.a0 + self.a1 * c1 + self.a2 * c2;
        let den_im = -(self.a1 * s1 + self.a2 * s2);

        ((num_re * num_re + num_im * num_im) / (den_re * den_re + den_im * den_im)).sqrt()
    }

    /// Normalises to this crate's Direct Form I `[b0, b1, b2, a1, a2]` convention, i.e.
    /// `H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 - a1 z^-1 - a2 z^-2)`.
    fn to_df1_f32(self) -> [f32; 5] {
        let inv = 1.0 / self.a0;
        [
            (self.b0 * inv) as f32,
            (self.b1 * inv) as f32,
            (self.b2 * inv) as f32,
            (-self.a1 * inv) as f32,
            (-self.a2 * inv) as f32,
        ]
    }

    /// Largest pole modulus of `a0 + a1 z^-1 + a2 z^-2 = 0`.
    fn max_pole_radius(&self) -> f64 {
        let disc = self.a1 * self.a1 - 4.0 * self.a0 * self.a2;
        if disc >= 0.0 {
            let root = disc.sqrt();
            let p1 = (-self.a1 + root) / (2.0 * self.a0);
            let p2 = (-self.a1 - root) / (2.0 * self.a0);
            p1.abs().max(p2.abs())
        } else {
            (self.a2 / self.a0).sqrt()
        }
    }
}

/// Faithful `f64` transcription of `cookBiquadVoxengo` (see module docs).
///
/// `gain` is linear (`2.0` is +6 dB) and `bw` is the -3 dB bandwidth in octaves.
fn cook_biquad_voxengo(kind: Kind, sample_rate: f64, freq: f64, gain: f64, bw: f64) -> Biquad6 {
    // Normalised centre frequency, clamped away from 0 and Nyquist (`tan` blows up otherwise).
    let fp = (freq / sample_rate).clamp(1e-9, 0.499_999_9);
    let fb = fp * 2f64.powf(-bw * 0.5);

    // `rs` is a shift parameter with 2.0 yielding the intended design (the reference notes a
    // value of 1.7 tracks the analog prototype more closely).
    const RS: f64 = 2.0;
    let r = (fp * fp - fb * fb) / (RS * fb * (0.25 - fp * fp));
    let y = r * r;

    // Family anchors: `gn` (Nyquist), `g0` (DC), `gb` (band edge), `gp` (peak), and the skew `v2`.
    let (gn, g0, gb, gp, v2): (f64, f64, f64, f64, f64) = match kind {
        Kind::Peq => {
            if (gain - 1.0).abs() < 1e-9 {
                return Biquad6 {
                    b0: 1.0,
                    b1: 0.0,
                    b2: 0.0,
                    a0: 1.0,
                    a1: 0.0,
                    a2: 0.0,
                };
            }
            (
                (1.0 + gain * y) / (1.0 + y / gain),
                1.0,
                gain,
                gain * gain,
                gain / (gain + y),
            )
        }
        Kind::Bpf => (y / (1.0 + y), 0.0, 0.5, 1.0, 1.0 / (1.0 + y)),
    };

    // Warped frequency axis.
    let xp = (core::f64::consts::PI * fp).tan().powi(2);
    let xb = (core::f64::consts::PI * fb).tan().powi(2);

    let w = xp * v2.sqrt();
    let gn_sqrt = gn.sqrt();
    let g0w = g0.sqrt() * w;

    // 2x2 linear solve for the numerator/denominator quadratic terms.
    let t = w - xp;
    let u = g0w - gn_sqrt * xp;
    let r1 = (gp * t * t - u * u) / xp;

    let t = w - xb;
    let u = g0w - gn_sqrt * xb;
    let r2 = (gb * t * t - u * u) / xb;

    let den = gb - gp;
    let a_sq = (r1 - r2) / den;
    let b_sq = (gb * r1 - gp * r2) / den;
    let a = a_sq.sqrt();
    let b = b_sq.sqrt();

    Biquad6 {
        b0: gn_sqrt + g0w + b,
        b1: 2.0 * (g0w - gn_sqrt),
        b2: gn_sqrt + g0w - b,
        a0: 1.0 + w + a,
        a1: 2.0 * (w - 1.0),
        a2: 1.0 + w - a,
    }
}

/// `Q` that yields a -3 dB octave bandwidth `bw`, per the reference's own relation
/// `BW = 2/ln(2) * asinh(1/(2Q))`.
fn q_from_bw(bw: f64) -> f64 {
    1.0 / (2.0 * (bw * core::f64::consts::LN_2 / 2.0).sinh())
}

/// Linear magnitude -> decibels.
fn db(mag: f64) -> f64 {
    20.0 * mag.log10()
}

/// Magnitude in dB of a crate-format Direct Form I section, via the crate's own analyzer.
fn rbj_mag_db(coeffs: [f32; 5], freq_norm: f32) -> f64 {
    response_magnitude_db(biquad_frequency_response(&coeffs, freq_norm)) as f64
}

#[test]
fn oracle_bell_hits_center_gain_exactly_and_targets_octave_edges() {
    // (sample_rate, centre_hz, linear_gain, bandwidth_octaves)
    let cases = [
        (48_000.0, 1_000.0, 2.0, 1.0),
        (48_000.0, 1_000.0, 3.981_071_7, 3.0),
        (48_000.0, 250.0, 10.0, 0.5),
        (44_100.0, 5_000.0, 0.5, 1.5),
    ];

    for (fs, f0, gain, bw) in cases {
        let f = cook_biquad_voxengo(Kind::Peq, fs, f0, gain, bw);

        let center = f.magnitude(f0 / fs);
        assert!(
            (center - gain).abs() / gain < 1e-9,
            "centre gain {center} != {gain}"
        );

        // Band edges target -3 dB below the peak, i.e. sqrt(gain) in linear terms. The
        // reference is approximate here: residual edge error is small (well under 0.2 dB for
        // these mid-band cases) but grows for very high-frequency, narrow, deep bands.
        let edge_target_db = db(gain.sqrt());
        let lo = f0 * 2f64.powf(-bw / 2.0);
        let hi = f0 * 2f64.powf(bw / 2.0);
        for freq in [lo, hi] {
            let m_db = db(f.magnitude(freq / fs));
            assert!(
                (m_db - edge_target_db).abs() < 0.25,
                "f0={f0} bw={bw} gain={gain}: edge {freq} Hz at {m_db:.3} dB, \
                 expected {edge_target_db:.3} dB"
            );
        }

        // A bell is transparent at DC.
        assert!((f.magnitude(0.0) - 1.0).abs() < 1e-9);
    }
}

#[test]
fn oracle_bandpass_is_unity_peak_with_minus_3db_edges() {
    let (fs, f0, bw) = (48_000.0, 1_000.0, 1.0);
    let f = cook_biquad_voxengo(Kind::Bpf, fs, f0, 1.0, bw);

    assert!((f.magnitude(f0 / fs) - 1.0).abs() < 1e-9);

    let lo = f0 * 2f64.powf(-bw / 2.0);
    let hi = f0 * 2f64.powf(bw / 2.0);
    for freq in [lo, hi] {
        let m_db = db(f.magnitude(freq / fs));
        assert!(
            (m_db + 3.010_3).abs() < 0.1,
            "edge {freq} Hz at {m_db:.3} dB, expected -3.01 dB"
        );
    }

    // A band-pass rejects DC.
    assert!(f.magnitude(0.0) < 1e-6);
}

#[test]
fn oracle_df1_conversion_agrees_with_crate_frequency_response() {
    let f = cook_biquad_voxengo(Kind::Peq, 48_000.0, 2_000.0, 2.0, 1.5);
    let df1 = f.to_df1_f32();

    for k in 0..=100 {
        let freq_norm = 0.5 * f64::from(k) / 100.0;
        let oracle_db = db(f.magnitude(freq_norm));
        let crate_db = rbj_mag_db(df1, freq_norm as f32);
        assert!(
            (oracle_db - crate_db).abs() < 0.01,
            "at {freq_norm} cyc/sample: oracle {oracle_db:.4} dB vs crate {crate_db:.4} dB"
        );
    }
}

#[test]
fn rbj_peaking_tracks_oracle_for_midband_bands() {
    let fs = 48_000.0;

    for f0 in [200.0, 1_000.0, 2_000.0] {
        for bw in [0.5, 1.0, 2.0] {
            for gain_db in [-12.0, -6.0, 6.0, 12.0] {
                let gain = 10f64.powf(gain_db / 20.0);
                let oracle = cook_biquad_voxengo(Kind::Peq, fs, f0, gain, bw);
                let rbj = biquad_peaking_coeffs(
                    f0 as f32,
                    fs as f32,
                    q_from_bw(bw) as f32,
                    gain_db as f32,
                );

                let lo = f0 * 2f64.powf(-bw / 2.0);
                let hi = f0 * 2f64.powf(bw / 2.0);
                for freq in [lo, f0, hi] {
                    let oracle_db = db(oracle.magnitude(freq / fs));
                    let rbj_db = rbj_mag_db(rbj, (freq / fs) as f32);
                    assert!(
                        (oracle_db - rbj_db).abs() < 0.2,
                        "f0={f0} bw={bw} gain={gain_db} dB at {freq} Hz: \
                         oracle {oracle_db:.3} dB vs RBJ {rbj_db:.3} dB"
                    );
                }
            }
        }
    }
}

/// Characterization test: documents *where* RBJ and the oracle diverge, so a future change in
/// either design is caught rather than silently absorbed.
#[test]
fn rbj_nyquist_anchor_pulls_wide_high_band_edges_off_spec() {
    // A 2-octave bell at 16 kHz on a 48 kHz rate has a nominal upper -3 dB edge at 32 kHz, above
    // Nyquist (24 kHz). RBJ forces the Nyquist gain to 0 dB, which drags the lower edge off its
    // nominal 8 kHz / +3 dB location; the oracle keeps the edge and lets Nyquist follow the skirt.
    let (fs, f0, gain_db, bw) = (48_000.0, 16_000.0, 6.0, 2.0);
    let gain = 10f64.powf(gain_db / 20.0);
    let oracle = cook_biquad_voxengo(Kind::Peq, fs, f0, gain, bw);
    let rbj = biquad_peaking_coeffs(f0 as f32, fs as f32, q_from_bw(bw) as f32, gain_db as f32);

    let lo = f0 * 2f64.powf(-bw / 2.0);
    assert!((lo - 8_000.0).abs() < 1e-9);

    let oracle_lo = db(oracle.magnitude(lo / fs));
    let rbj_lo = rbj_mag_db(rbj, (lo / fs) as f32);

    // The oracle honours the nominal spec: -3 dB relative to the +6 dB peak => +3.01 dB.
    assert!(
        (oracle_lo - 3.010_3).abs() < 0.1,
        "oracle lower edge {oracle_lo:.3} dB"
    );

    // RBJ undershoots that edge by more than a decibel.
    assert!(
        rbj_lo < 2.0,
        "expected RBJ lower edge below +2 dB, got {rbj_lo:.3} dB"
    );
    assert!(
        oracle_lo - rbj_lo > 1.0,
        "oracle {oracle_lo:.3} dB vs RBJ {rbj_lo:.3} dB"
    );

    // Conversely RBJ pins Nyquist to 0 dB while the oracle's skirt lifts it.
    let oracle_nyq = db(oracle.magnitude(0.5));
    let rbj_nyq = rbj_mag_db(rbj, 0.5);
    assert!(rbj_nyq.abs() < 0.05, "RBJ Nyquist {rbj_nyq:.3} dB");
    assert!(oracle_nyq > 3.5, "oracle Nyquist {oracle_nyq:.3} dB");
}

#[test]
fn oracle_is_finite_and_stable_over_extreme_parameters() {
    let fs = 48_000.0;
    let mut cases = 0;

    for kind in [Kind::Peq, Kind::Bpf] {
        let mut f0 = 10.0;
        while f0 < fs / 2.0 {
            let mut bw = 0.1;
            while bw <= 4.0 {
                for gain in [0.01, 0.25, 1.0, 4.0, 100.0] {
                    let f = cook_biquad_voxengo(kind, fs, f0, gain, bw);
                    let coeffs = [f.b0, f.b1, f.b2, f.a0, f.a1, f.a2];
                    assert!(
                        coeffs.iter().all(|c| c.is_finite()),
                        "non-finite coefficients at f0={f0} bw={bw} gain={gain}"
                    );
                    let radius = f.max_pole_radius();
                    assert!(
                        radius < 1.0,
                        "unstable at f0={f0} bw={bw} gain={gain}: pole radius {radius}"
                    );
                    cases += 1;
                }
                bw += 0.3;
            }
            f0 *= 1.3;
        }
    }

    assert!(cases > 1_000, "expected a broad sweep, ran {cases} cases");
}

// ─── EQ builder, IHo & WebAudio export ─────────────────────────────────────

use embedded_dsp::filter_design::{BiquadType, EqError, EqFilter, WebAudioFilter};
use idsp::iir::coefficients::{Filter as IdspFilter, Type as IdspType};

/// Convert an `idsp` cookbook `[b, a]` pair to this crate's normalised Direct Form I
/// `[b0, b1, b2, a1, a2]` convention.
fn idsp_ba_to_df1(ba: [[f32; 3]; 2]) -> [f32; 5] {
    let [b, a] = ba;
    let inv = 1.0 / a[0];
    [b[0] * inv, b[1] * inv, b[2] * inv, -a[1] * inv, -a[2] * inv]
}

/// Design `typ` through the `idsp` oracle, with the shelf gain set where it applies.
fn idsp_design(typ: IdspType, f0: f32, fs: f32, q: f32, gain_db: f32) -> [f32; 5] {
    let mut filter = IdspFilter::<f32>::default();
    filter.frequency(f0, fs).q(q);
    if matches!(
        typ,
        IdspType::Peaking | IdspType::Lowshelf | IdspType::Highshelf | IdspType::IHo
    ) {
        filter.shelf_db(gain_db);
    }
    idsp_ba_to_df1(filter.build(typ))
}

fn assert_coeffs_close(actual: [f32; 5], oracle: [f32; 5], ctx: &str) {
    for i in 0..5 {
        let scale = oracle[i].abs().max(1e-3);
        assert!(
            (actual[i] - oracle[i]).abs() / scale < 5e-5,
            "{ctx}: coeff {i}: {} vs idsp oracle {}",
            actual[i],
            oracle[i]
        );
    }
}

/// The unified builder must reproduce `idsp`'s cookbook coefficients for every response
/// type, including the `IHo` section this crate previously lacked.
#[test]
fn eq_builder_matches_idsp_oracle_for_every_type() {
    let fs = 48_000.0f32;
    let types = [
        (BiquadType::Lowpass, IdspType::Lowpass),
        (BiquadType::Highpass, IdspType::Highpass),
        (BiquadType::Bandpass, IdspType::Bandpass),
        (BiquadType::Allpass, IdspType::Allpass),
        (BiquadType::Notch, IdspType::Notch),
        (BiquadType::Peaking, IdspType::Peaking),
        (BiquadType::Lowshelf, IdspType::Lowshelf),
        (BiquadType::Highshelf, IdspType::Highshelf),
        (BiquadType::Iho, IdspType::IHo),
    ];

    let mut cases = 0;
    for (typ, idsp_typ) in types {
        for f0 in [100.0f32, 1_000.0, 5_000.0, 12_000.0] {
            for q in [0.5f32, 0.707, 2.0, 8.0] {
                for gain_db in [-12.0f32, 0.0, 6.0] {
                    let ours = EqFilter::new(f0, fs).q(q).gain_db(gain_db).build(typ);
                    let oracle = idsp_design(idsp_typ, f0, fs, q, gain_db);
                    assert_coeffs_close(
                        ours,
                        oracle,
                        &format!("{typ:?} f0={f0} q={q} gain={gain_db}"),
                    );
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 400, "expected a broad sweep, ran {cases} cases");
}

#[test]
fn eq_builder_validates_and_sanitizes() {
    let fs = 48_000.0f32;

    assert_eq!(
        EqFilter::new(0.0, fs).validate(),
        Err(EqError::OutOfRange("frequency_hz"))
    );
    assert_eq!(
        EqFilter::new(fs * 0.5, fs).validate(),
        Err(EqError::OutOfRange("frequency_hz"))
    );
    assert_eq!(
        EqFilter::new(1_000.0, 0.0).validate(),
        Err(EqError::NonPositive("sample_rate_hz"))
    );
    assert_eq!(
        EqFilter::new(1_000.0, fs).q(0.0).validate(),
        Err(EqError::NonPositive("q"))
    );
    assert_eq!(
        EqFilter::new(f32::NAN, fs).validate(),
        Err(EqError::NonFinite("frequency_hz"))
    );

    // `build` falls back to a passthrough biquad instead of emitting non-finite
    // coefficients; `try_build` reports the error instead.
    assert_eq!(
        EqFilter::new(0.0, fs).build(BiquadType::Lowpass),
        [1.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert!(
        EqFilter::new(1_000.0, fs)
            .try_build(BiquadType::Lowpass)
            .is_ok()
    );
}

#[test]
fn eq_builder_resolves_bandwidth_and_slope_shapes() {
    let (f0, fs) = (1_000.0f32, 48_000.0f32);

    let bw = 1.0f32;
    let by_bw = EqFilter::new(f0, fs).bandwidth_octaves(bw);
    let q_bw = biquad_q_from_bw(bw, f0, fs);
    assert!((by_bw.q_value() - q_bw).abs() < 1e-6);
    assert_eq!(
        by_bw.build(BiquadType::Bandpass),
        EqFilter::new(f0, fs).q(q_bw).bandpass()
    );

    let (slope, gain_db) = (0.8f32, 6.0f32);
    let by_slope = EqFilter::new(f0, fs).shelf_slope(slope).gain_db(gain_db);
    let q_slope = biquad_q_from_shelf_slope(slope, gain_db);
    assert!((by_slope.q_value() - q_slope).abs() < 1e-9);
    assert_eq!(
        by_slope.build(BiquadType::Lowshelf),
        EqFilter::new(f0, fs).q(q_slope).gain_db(gain_db).lowshelf()
    );
}

#[test]
fn webaudio_filter_applies_detune_and_names_types() {
    // +1200 cents is one octave, exactly as a `BiquadFilterNode` detune.
    let wa = WebAudioFilter {
        frequency_hz: 1_000.0,
        detune_cents: 1_200.0,
        ..Default::default()
    };
    assert!((wa.effective_frequency_hz() - 2_000.0).abs() < 1e-3);
    assert_eq!(wa.type_name(), Some("lowpass"));

    let via_wa = wa.try_build().unwrap();
    let direct = EqFilter::new(2_000.0, 48_000.0)
        .q(1.0)
        .try_build(BiquadType::Lowpass)
        .unwrap();
    assert_eq!(via_wa, direct);

    // `IHo` has no native WebAudio node type.
    assert_eq!(
        WebAudioFilter {
            typ: BiquadType::Iho,
            ..Default::default()
        }
        .type_name(),
        None
    );

    // Detune that pushes the effective frequency past Nyquist is rejected.
    assert_eq!(
        WebAudioFilter {
            frequency_hz: 20_000.0,
            detune_cents: 1_200.0,
            ..Default::default()
        }
        .validate(),
        Err(EqError::OutOfRange("effective_frequency_hz"))
    );
}
