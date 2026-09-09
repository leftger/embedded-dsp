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

// --- ITU-T G.711 8-bit bytes (Sun/CCITT linear ↔ μ-law / A-law) ---

fn top_bit(x: i32) -> i32 {
    if x <= 0 {
        -1
    } else {
        31 - (x as u32).leading_zeros() as i32
    }
}

/// Encode a 16-bit linear PCM sample to a G.711 μ-law byte.
pub fn linear_to_ulaw(sample: i16) -> u8 {
    const BIAS: i32 = 0x84;
    let mut pcm = sample as i32;
    let mask = if pcm < 0 {
        pcm = BIAS - pcm;
        0x7F
    } else {
        pcm += BIAS;
        0xFF
    };
    if pcm > 0x7FFF {
        pcm = 0x7FFF;
    }
    let seg = top_bit(pcm | 0xFF) - 7;
    if seg >= 8 {
        (0x7F ^ mask) as u8
    } else {
        let uval = (seg << 4) | ((pcm >> (seg + 3)) & 0x0F);
        (uval ^ mask) as u8
    }
}

/// Decode a G.711 μ-law byte to 16-bit linear PCM.
pub fn ulaw_to_linear(u: u8) -> i16 {
    const BIAS: i32 = 0x84;
    let u = (!u) as i32;
    let mut t = ((u & 0x0F) << 3) + BIAS;
    t <<= (u & 0x70) >> 4;
    let out = if u & 0x80 != 0 { BIAS - t } else { t - BIAS };
    out.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Encode a 16-bit linear PCM sample to a G.711 A-law byte.
pub fn linear_to_alaw(sample: i16) -> u8 {
    let mut pcm = sample as i32;
    let mask = if pcm >= 0 {
        0xD5
    } else {
        pcm = -pcm - 8;
        0x55
    };
    let seg = top_bit(pcm | 0xFF) - 7;
    if seg >= 8 {
        (0x7F ^ mask) as u8
    } else {
        let shift = if seg != 0 { seg + 3 } else { 4 };
        let aval = (seg << 4) | ((pcm >> shift) & 0x0F);
        (aval ^ mask) as u8
    }
}

/// Decode a G.711 A-law byte to 16-bit linear PCM.
pub fn alaw_to_linear(a: u8) -> i16 {
    let a = (a ^ 0x55) as i32;
    let mut t = (a & 0x0F) << 4;
    let seg = (a & 0x70) >> 4;
    if seg != 0 {
        t = (t + 0x108) << (seg - 1);
    } else {
        t += 8;
    }
    let out = if a & 0x80 != 0 { t } else { -t };
    out.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}
