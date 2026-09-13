//! Independent-oracle tests for bell / notch / band-pass biquad design.
//!
//! `biquad_peaking_coeffs` follows the RBJ Audio EQ Cookbook, which anchors the response to
//! 0 dB at both DC and Nyquist. That anchor is an approximation of an analog bell: for *wide*
//! bands whose upper edge approaches or passes Nyquist, forcing the Nyquist gain to 0 dB drags
//! the lower -3 dB edge away from its nominal `f0 * 2^(-BW/2)` location.
//!
//! These tests cross-check the RBJ design against an independent reference that targets the
//! `f0 * 2^(-BW/2)` octave-band edges directly instead of pinning Nyquist. The reference is a
//! test-only `f64` port of Aleksey Vaneev's
//! `cookBiquadVoxengo` ("perfect biquad", `biquad_voxengo.h` v1.2). It is deliberately *not*
//! part of the public API — it exists to validate and characterize the crate's own filters.
//!
//! Reference: <https://www.voxengo.com/public/biquad/>
//!
//! ```text
//! Copyright (c) 2026 Aleksey Vaneev
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy of this
//! software and associated documentation files (the "Software"), to deal in the Software
//! without restriction, including without limitation the rights to use, copy, modify, merge,
//! publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
//! to whom the Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all copies or
//! substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//! INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//! PURPOSE AND NONINFRINGEMENT.
//! ```

use embedded_dsp::filter_analysis::{biquad_frequency_response, response_magnitude_db};
use embedded_dsp::filter_design::biquad_peaking_coeffs;

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
