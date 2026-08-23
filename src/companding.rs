//! Audio companding: the non-linear µ-law and "A"-law compression curves used by the ITU-T
//! G.711 telephony standard to compress a wide-dynamic-range linear audio sample down to a
//! lower bit depth with minimal perceptual loss (Steven W. Smith, "The Scientist and
//! Engineer's Guide to DSP", Ch. 22, Eq. 22-1 / 22-2).
//!
//! All functions operate on samples normalized to `-1.0..=1.0`.

#[allow(unused_imports)]
use crate::math::FloatMath;

const MU: f32 = 255.0;
const A_LAW: f32 = 87.6;
// Precomputed so the per-sample hot path only needs one transcendental call, not two.
const LN_1P_MU: f32 = 5.545_177_5; // ln(1 + MU)
const LN_A_LAW: f32 = 4.472_781; // ln(A_LAW)
const A_LAW_DENOM: f32 = 1.0 + LN_A_LAW; // 1 + ln(A_LAW)

#[inline(always)]
fn sign(x: f32) -> f32 {
    if x < 0.0 { -1.0 } else { 1.0 }
}

/// Compresses a normalized linear sample `x` (`-1.0..=1.0`) using µ255-law companding
/// (Eq. 22-1), expanding resolution for small amplitudes at the expense of large ones.
pub fn mu_law_compress_f32(x: f32) -> f32 {
    let ax = x.abs();
    sign(x) * (1.0 + MU * ax).ln() / LN_1P_MU
}

/// Expands a µ-law-compressed sample `y` (`-1.0..=1.0`) back to a normalized linear sample,
/// inverting [`mu_law_compress_f32`].
pub fn mu_law_expand_f32(y: f32) -> f32 {
    let ay = y.abs();
    sign(y) * ((1.0 + MU).powf(ay) - 1.0) / MU
}

/// Compresses a normalized linear sample `x` (`-1.0..=1.0`) using "A"-law companding
/// (Eq. 22-2): a linear segment near zero, transitioning to a logarithmic curve.
pub fn a_law_compress_f32(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 1.0 / A_LAW {
        sign(x) * (A_LAW * ax) / A_LAW_DENOM
    } else {
        sign(x) * (1.0 + (A_LAW * ax).ln()) / A_LAW_DENOM
    }
}

/// Expands an "A"-law-compressed sample `y` (`-1.0..=1.0`) back to a normalized linear sample,
/// inverting [`a_law_compress_f32`].
pub fn a_law_expand_f32(y: f32) -> f32 {
    let ay = y.abs();
    if ay < 1.0 / A_LAW_DENOM {
        sign(y) * ay * A_LAW_DENOM / A_LAW
    } else {
        sign(y) * (ay * A_LAW_DENOM - 1.0).exp() / A_LAW
    }
}
