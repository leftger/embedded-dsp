#![no_main]
//! Differential fuzz: the generic `fir::<f32>` kernel against an `f64` causal
//! convolution reference, on well-scaled inputs.

use libfuzzer_sys::fuzz_target;

/// Maps four bytes to a value in `[-1, 1)` so the `f32`/`f64` gap stays a pure
/// rounding difference.
fn decode(b: &[u8]) -> f32 {
    i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0
}

fuzz_target!(|data: &[u8]| {
    const SRC_LEN: usize = 64;
    if data.is_empty() {
        return;
    }
    let taps = (data[0] as usize % 32) + 1;
    if data.len() < 1 + (taps + SRC_LEN) * 4 {
        return;
    }

    let mut coeffs = [0.0f32; 32];
    let mut src = [0.0f32; SRC_LEN];
    let mut p = 1;
    for c in coeffs.iter_mut().take(taps) {
        *c = decode(&data[p..p + 4]);
        p += 4;
    }
    for s in src.iter_mut() {
        *s = decode(&data[p..p + 4]);
        p += 4;
    }

    let mut state = [0.0f32; 32];
    let mut dst = [0.0f32; SRC_LEN];
    let mut instance = embedded_dsp::filtering::FirInstance::<f32> {
        num_taps: taps as u16,
        coeffs: &coeffs[..taps],
        state: &mut state[..taps],
    };
    embedded_dsp::filtering::fir(&mut instance, &src, &mut dst);
    assert!(dst.iter().all(|v| v.is_finite()), "non-finite FIR output");

    for i in 0..SRC_LEN {
        let mut reference = 0.0f64;
        for k in 0..taps {
            if i >= k {
                reference += coeffs[k] as f64 * src[i - k] as f64;
            }
        }
        let diff = (dst[i] as f64 - reference).abs();
        assert!(
            diff <= 1e-4 + 1e-5 * reference.abs(),
            "FIR f32 {} vs f64 {} (taps={taps}, i={i})",
            dst[i],
            reference
        );
    }
});
