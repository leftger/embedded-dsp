#![no_main]
//! Differential fuzz: audio-EQ-designed biquad cascades (`f32`) against an `f64`
//! Direct Form I reference, one to three stages.

use libfuzzer_sys::fuzz_target;

use embedded_dsp::filter_design::{BiquadType, EqFilter};
use embedded_dsp::filtering::{BiquadCascadeInstance, biquad_cascade_df1};

fn typ(d: u8) -> BiquadType {
    match d % 9 {
        0 => BiquadType::Lowpass,
        1 => BiquadType::Highpass,
        2 => BiquadType::Bandpass,
        3 => BiquadType::Allpass,
        4 => BiquadType::Notch,
        5 => BiquadType::Peaking,
        6 => BiquadType::Lowshelf,
        7 => BiquadType::Highshelf,
        _ => BiquadType::Iho,
    }
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn i32_at(b: &[u8], i: usize) -> i32 {
    i32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// A well-scaled value in `[-1, 1)`.
fn decode(b: &[u8], i: usize) -> f32 {
    i32_at(b, i) as f32 / 2_147_483_648.0
}

fuzz_target!(|data: &[u8]| {
    const LEN: usize = 64;
    const FS: f32 = 48_000.0;
    if data.len() < 1 + 3 * 8 + LEN * 4 {
        return;
    }

    let stages = 1 + (data[0] as usize % 3);
    let mut coeffs = vec![0.0f32; stages * 5];
    for s in 0..stages {
        let base = 1 + s * 8;
        let f0 = 20.0 + (u16_at(data, base) as f32 / 65_535.0) * (FS * 0.45 - 20.0);
        let q = 0.1 + (data[base + 2] as f32 / 255.0) * 9.9;
        let gain_db = decode(data, base + 4) * 24.0;
        let Ok(section) = EqFilter::new(f0, FS)
            .q(q)
            .gain_db(gain_db)
            .try_build(typ(data[base + 3]))
        else {
            return;
        };
        coeffs[s * 5..s * 5 + 5].copy_from_slice(&section);
    }

    let src: Vec<f32> = (0..LEN).map(|i| decode(data, 25 + i * 4)).collect();
    let mut state = vec![0.0f32; stages * 4];
    let mut dst = vec![0.0f32; LEN];
    let mut instance = BiquadCascadeInstance::<f32> {
        num_stages: stages as u8,
        post_shift: 0,
        coeffs: &coeffs,
        state: &mut state,
    };
    biquad_cascade_df1(&mut instance, &src, &mut dst);
    assert!(
        dst.iter().all(|v| v.is_finite()),
        "non-finite biquad output"
    );

    // f64 reference: one stage at a time over the whole block.
    let mut signal: Vec<f64> = src.iter().map(|&x| x as f64).collect();
    for s in 0..stages {
        let c = &coeffs[s * 5..s * 5 + 5];
        let (b0, b1, b2) = (c[0] as f64, c[1] as f64, c[2] as f64);
        let (a1, a2) = (c[3] as f64, c[4] as f64);
        let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0, 0.0, 0.0);
        for x in signal.iter_mut() {
            let y = b0 * *x + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
            x2 = x1;
            x1 = *x;
            y2 = y1;
            y1 = y;
            *x = y;
        }
    }

    for i in 0..LEN {
        let reference = signal[i];
        let diff = (dst[i] as f64 - reference).abs();
        assert!(
            diff <= 2e-3 + 1e-3 * reference.abs(),
            "biquad f32 {} vs f64 {} (stages={stages}, i={i})",
            dst[i],
            reference
        );
    }
});
