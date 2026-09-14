//! Deterministic randomized differential checks: the `f32` kernels against
//! in-process `f64` references.
//!
//! The libFuzzer targets in `fuzz/` run the same comparisons with coverage
//! guidance; this keeps a fast, always-on version in the normal test suite.

use embedded_dsp::filter_design::{BiquadType, EqFilter};
use embedded_dsp::filtering::{BiquadCascadeInstance, FirInstance, biquad_cascade_df1, fir};

const LEN: usize = 64;

/// Tiny deterministic LCG, so the sweep is reproducible without a dev-dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// A value in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / 2_147_483_648.0
    }

    /// A value in `[0, 1)`.
    fn frac(&mut self) -> f32 {
        self.next_u32() as f32 / 4_294_967_296.0
    }
}

#[test]
fn fir_f32_tracks_f64_reference() {
    let mut rng = Lcg::new(0x1234_5678_9abc_def0);
    let mut worst = 0.0f64;

    for _ in 0..500 {
        let taps = 1 + (rng.next_u32() as usize % 32);
        let coeffs: Vec<f32> = (0..taps).map(|_| rng.unit()).collect();
        let src: Vec<f32> = (0..LEN).map(|_| rng.unit()).collect();
        let mut state = vec![0.0f32; taps];
        let mut dst = vec![0.0f32; LEN];

        let mut instance = FirInstance::<f32> {
            num_taps: taps as u16,
            coeffs: &coeffs,
            state: &mut state,
        };
        fir(&mut instance, &src, &mut dst);
        assert!(dst.iter().all(|v| v.is_finite()));

        for i in 0..LEN {
            let mut reference = 0.0f64;
            for k in 0..taps {
                if i >= k {
                    reference += coeffs[k] as f64 * src[i - k] as f64;
                }
            }
            let rel = (dst[i] as f64 - reference).abs() / (1.0 + reference.abs());
            worst = worst.max(rel);
        }
    }

    assert!(worst < 1e-5, "worst FIR relative gap {worst}");
}

#[test]
fn biquad_f32_tracks_f64_reference() {
    let mut rng = Lcg::new(0xdead_beef_0bad_f00d);
    let fs = 48_000.0f32;
    let types = [
        BiquadType::Lowpass,
        BiquadType::Highpass,
        BiquadType::Bandpass,
        BiquadType::Allpass,
        BiquadType::Notch,
        BiquadType::Peaking,
        BiquadType::Lowshelf,
        BiquadType::Highshelf,
        BiquadType::Iho,
    ];
    let mut worst = 0.0f64;

    for _ in 0..500 {
        let typ = types[rng.next_u32() as usize % types.len()];
        let f0 = 20.0 + rng.frac() * (fs * 0.45 - 20.0);
        let q = 0.1 + rng.frac() * 9.9;
        let gain_db = rng.unit() * 24.0;
        let coeffs = EqFilter::new(f0, fs)
            .q(q)
            .gain_db(gain_db)
            .try_build(typ)
            .expect("in-range design must validate");

        let src: Vec<f32> = (0..LEN).map(|_| rng.unit()).collect();
        let mut state = [0.0f32; 4];
        let mut dst = vec![0.0f32; LEN];

        let mut instance = BiquadCascadeInstance::<f32> {
            num_stages: 1,
            post_shift: 0,
            coeffs: &coeffs,
            state: &mut state,
        };
        biquad_cascade_df1(&mut instance, &src, &mut dst);
        assert!(dst.iter().all(|v| v.is_finite()));

        let (b0, b1, b2) = (coeffs[0] as f64, coeffs[1] as f64, coeffs[2] as f64);
        let (a1, a2) = (coeffs[3] as f64, coeffs[4] as f64);
        let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0, 0.0, 0.0);
        for i in 0..LEN {
            let x = src[i] as f64;
            let y = b0 * x + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;
            let rel = (dst[i] as f64 - y).abs() / (1.0 + y.abs());
            worst = worst.max(rel);
        }
    }

    assert!(worst < 5e-4, "worst biquad relative gap {worst}");
}
