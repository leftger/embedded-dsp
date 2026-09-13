//! Mixed-radix Cooley–Tukey FFT for 2/3/4/5-smooth lengths (Kiss FFT butterflies).
//!
//! Power-of-two sizes stay on the in-place radix-2 path in [`super::cfft_f32`].
//! Non-power-of-two 2/3/4/5-smooth sizes up to [`MIXED_RADIX_MAX`] use an
//! out-of-place factor walk with dedicated radix-2/3/4/5 butterflies.

#![allow(unused_imports)]
use crate::math::FloatMath;

/// Largest mixed-radix length that fits the stack scratch buffer (`2 * n` floats).
pub(super) const MIXED_RADIX_MAX: usize = 512;

const MAX_FACTORS: usize = 32;

#[inline]
fn cmul(ar: f32, ai: f32, br: f32, bi: f32) -> (f32, f32) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

#[inline]
fn cis(k: usize, nfft: usize, inverse: bool) -> (f32, f32) {
    let k = k % nfft;
    let sign = if inverse { 1.0 } else { -1.0 };
    let a = sign * 2.0 * core::f32::consts::PI * (k as f32) / (nfft as f32);
    (a.cos(), a.sin())
}

/// True when `n` factors completely into 2, 3, and 5.
pub fn fft_is_235_smooth(mut n: usize) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2usize, 3, 5] {
        while n.is_multiple_of(p) {
            n /= p;
        }
    }
    n == 1
}

/// Smallest 2/3/4/5-smooth length `>= n` (Kiss `kiss_fft_next_fast_size`).
pub fn next_fast_fft_size(n: usize) -> usize {
    let mut n = n.max(2);
    while !fft_is_235_smooth(n) {
        n += 1;
    }
    n
}

/// Lengths [`cfft_f32`](super::cfft_f32) will transform.
pub fn cfft_f32_len_ok(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_power_of_two() {
        return true;
    }
    n <= MIXED_RADIX_MAX && fft_is_235_smooth(n)
}

fn kf_factor(mut n: usize, fac: &mut [usize]) -> Option<usize> {
    let mut i = 0;
    let mut p = 4usize;
    while n > 1 {
        while !n.is_multiple_of(p) {
            p = match p {
                4 => 2,
                2 => 3,
                _ => p + 2,
            };
            if p * p > n {
                p = n;
            }
        }
        if p > 5 {
            return None;
        }
        n /= p;
        if i + 1 >= fac.len() {
            return None;
        }
        fac[i] = p;
        fac[i + 1] = n;
        i += 2;
    }
    Some(i)
}

fn bfly2(fout: &mut [f32], off: usize, fstride: usize, m: usize, nfft: usize, inverse: bool) {
    for i in 0..m {
        let (wr, wi) = cis(i * fstride, nfft, inverse);
        let a = off + i;
        let b = a + m;
        let (tr, ti) = cmul(fout[2 * b], fout[2 * b + 1], wr, wi);
        let ur = fout[2 * a];
        let ui = fout[2 * a + 1];
        fout[2 * b] = ur - tr;
        fout[2 * b + 1] = ui - ti;
        fout[2 * a] = ur + tr;
        fout[2 * a + 1] = ui + ti;
    }
}

fn bfly3(fout: &mut [f32], off: usize, fstride: usize, m: usize, nfft: usize, inverse: bool) {
    let (_epi3_r, epi3_i) = cis(fstride * m, nfft, inverse);
    let m2 = 2 * m;
    for u in 0..m {
        let i0 = off + u;
        let i1 = i0 + m;
        let i2 = i0 + m2;
        let (wr1, wi1) = cis(u * fstride, nfft, inverse);
        let (wr2, wi2) = cis(2 * u * fstride, nfft, inverse);
        let (s1r, s1i) = cmul(fout[2 * i1], fout[2 * i1 + 1], wr1, wi1);
        let (s2r, s2i) = cmul(fout[2 * i2], fout[2 * i2 + 1], wr2, wi2);
        let s3r = s1r + s2r;
        let s3i = s1i + s2i;
        let s0r = s1r - s2r;
        let s0i = s1i - s2i;
        fout[2 * i1] = fout[2 * i0] - 0.5 * s3r;
        fout[2 * i1 + 1] = fout[2 * i0 + 1] - 0.5 * s3i;
        let s0r = s0r * epi3_i;
        let s0i = s0i * epi3_i;
        fout[2 * i0] += s3r;
        fout[2 * i0 + 1] += s3i;
        fout[2 * i2] = fout[2 * i1] + s0i;
        fout[2 * i2 + 1] = fout[2 * i1 + 1] - s0r;
        fout[2 * i1] -= s0i;
        fout[2 * i1 + 1] += s0r;
    }
}

fn bfly4(fout: &mut [f32], off: usize, fstride: usize, m: usize, nfft: usize, inverse: bool) {
    let m2 = 2 * m;
    let m3 = 3 * m;
    for u in 0..m {
        let i0 = off + u;
        let i1 = i0 + m;
        let i2 = i0 + m2;
        let i3 = i0 + m3;
        let (wr1, wi1) = cis(u * fstride, nfft, inverse);
        let (wr2, wi2) = cis(2 * u * fstride, nfft, inverse);
        let (wr3, wi3) = cis(3 * u * fstride, nfft, inverse);
        let (s0r, s0i) = cmul(fout[2 * i1], fout[2 * i1 + 1], wr1, wi1);
        let (s1r, s1i) = cmul(fout[2 * i2], fout[2 * i2 + 1], wr2, wi2);
        let (s2r, s2i) = cmul(fout[2 * i3], fout[2 * i3 + 1], wr3, wi3);

        let s5r = fout[2 * i0] - s1r;
        let s5i = fout[2 * i0 + 1] - s1i;
        fout[2 * i0] += s1r;
        fout[2 * i0 + 1] += s1i;
        let s3r = s0r + s2r;
        let s3i = s0i + s2i;
        let s4r = s0r - s2r;
        let s4i = s0i - s2i;
        fout[2 * i2] = fout[2 * i0] - s3r;
        fout[2 * i2 + 1] = fout[2 * i0 + 1] - s3i;
        fout[2 * i0] += s3r;
        fout[2 * i0 + 1] += s3i;
        if inverse {
            fout[2 * i1] = s5r - s4i;
            fout[2 * i1 + 1] = s5i + s4r;
            fout[2 * i3] = s5r + s4i;
            fout[2 * i3 + 1] = s5i - s4r;
        } else {
            fout[2 * i1] = s5r + s4i;
            fout[2 * i1 + 1] = s5i - s4r;
            fout[2 * i3] = s5r - s4i;
            fout[2 * i3 + 1] = s5i + s4r;
        }
    }
}

fn bfly5(fout: &mut [f32], off: usize, fstride: usize, m: usize, nfft: usize, inverse: bool) {
    let (yar, yai) = cis(fstride * m, nfft, inverse);
    let (ybr, ybi) = cis(fstride * 2 * m, nfft, inverse);
    for u in 0..m {
        let i0 = off + u;
        let i1 = i0 + m;
        let i2 = i0 + 2 * m;
        let i3 = i0 + 3 * m;
        let i4 = i0 + 4 * m;
        let s0r = fout[2 * i0];
        let s0i = fout[2 * i0 + 1];
        let (s1r, s1i) = cmul(
            fout[2 * i1],
            fout[2 * i1 + 1],
            cis(u * fstride, nfft, inverse).0,
            cis(u * fstride, nfft, inverse).1,
        );
        let (s2r, s2i) = cmul(
            fout[2 * i2],
            fout[2 * i2 + 1],
            cis(2 * u * fstride, nfft, inverse).0,
            cis(2 * u * fstride, nfft, inverse).1,
        );
        let (s3r, s3i) = cmul(
            fout[2 * i3],
            fout[2 * i3 + 1],
            cis(3 * u * fstride, nfft, inverse).0,
            cis(3 * u * fstride, nfft, inverse).1,
        );
        let (s4r, s4i) = cmul(
            fout[2 * i4],
            fout[2 * i4 + 1],
            cis(4 * u * fstride, nfft, inverse).0,
            cis(4 * u * fstride, nfft, inverse).1,
        );

        let s7r = s1r + s4r;
        let s7i = s1i + s4i;
        let s10r = s1r - s4r;
        let s10i = s1i - s4i;
        let s8r = s2r + s3r;
        let s8i = s2i + s3i;
        let s9r = s2r - s3r;
        let s9i = s2i - s3i;

        fout[2 * i0] = s0r + s7r + s8r;
        fout[2 * i0 + 1] = s0i + s7i + s8i;

        let s5r = s0r + s7r * yar + s8r * ybr;
        let s5i = s0i + s7i * yar + s8i * ybr;
        let s6r = s10i * yai + s9i * ybi;
        let s6i = -s10r * yai - s9r * ybi;

        fout[2 * i1] = s5r - s6r;
        fout[2 * i1 + 1] = s5i - s6i;
        fout[2 * i4] = s5r + s6r;
        fout[2 * i4 + 1] = s5i + s6i;

        let s11r = s0r + s7r * ybr + s8r * yar;
        let s11i = s0i + s7i * ybr + s8i * yar;
        let s12r = -s10i * ybi + s9i * yai;
        let s12i = s10r * ybi - s9r * yai;

        fout[2 * i2] = s11r + s12r;
        fout[2 * i2 + 1] = s11i + s12i;
        fout[2 * i3] = s11r - s12r;
        fout[2 * i3 + 1] = s11i - s12i;
    }
}

#[allow(clippy::too_many_arguments)]
fn kf_work(
    fout: &mut [f32],
    fin: &[f32],
    fout_off: usize,
    fin_off: usize,
    fstride: usize,
    in_stride: usize,
    factors: &[usize],
    nfft: usize,
    inverse: bool,
) {
    let p = factors[0];
    let m = factors[1];
    let rest = &factors[2..];

    if m == 1 {
        let mut foff = fin_off;
        for i in 0..p {
            let o = fout_off + i;
            fout[2 * o] = fin[2 * foff];
            fout[2 * o + 1] = fin[2 * foff + 1];
            foff += fstride * in_stride;
        }
    } else {
        for k in 0..p {
            kf_work(
                fout,
                fin,
                fout_off + k * m,
                fin_off + k * fstride * in_stride,
                fstride * p,
                in_stride,
                rest,
                nfft,
                inverse,
            );
        }
    }

    match p {
        2 => bfly2(fout, fout_off, fstride, m, nfft, inverse),
        3 => bfly3(fout, fout_off, fstride, m, nfft, inverse),
        4 => bfly4(fout, fout_off, fstride, m, nfft, inverse),
        5 => bfly5(fout, fout_off, fstride, m, nfft, inverse),
        _ => {}
    }
}

/// Out-of-place mixed-radix CFFT into a stack temp, then copy back.
/// Returns `false` if `n` is not a supported mixed-radix length.
pub(super) fn mixed_radix_cfft(data: &mut [f32], n: usize, inverse: bool) -> bool {
    if !(2..=MIXED_RADIX_MAX).contains(&n) || data.len() < 2 * n || !fft_is_235_smooth(n) {
        return false;
    }
    let mut factors = [0usize; MAX_FACTORS];
    let nfac = match kf_factor(n, &mut factors) {
        Some(v) => v,
        None => return false,
    };
    let mut tmp = [0.0f32; 2 * MIXED_RADIX_MAX];
    kf_work(
        &mut tmp,
        data,
        0,
        0,
        1,
        1,
        &factors[..nfac],
        n,
        inverse,
    );
    data[..2 * n].copy_from_slice(&tmp[..2 * n]);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kf_factor_rejects_primes_and_tiny_buffers() {
        let mut fac = [0usize; MAX_FACTORS];
        assert!(kf_factor(7, &mut fac).is_none());
        assert!(kf_factor(1, &mut fac).is_some());
        let mut tiny = [0usize; 2];
        assert!(kf_factor(12, &mut tiny).is_none());
        let mut two = [0usize; 2];
        assert_eq!(kf_factor(4, &mut two), Some(2));
    }

    #[test]
    fn mixed_radix_cfft_guard_clauses() {
        let mut too_short = [1.0f32; 4];
        assert!(!mixed_radix_cfft(&mut too_short, 12, false));
        let mut seven = [1.0f32; 14];
        assert!(!mixed_radix_cfft(&mut seven, 7, false));
        let mut one = [1.0f32; 4];
        assert!(!mixed_radix_cfft(&mut one, 1, false));
        let mut ok = [0.0f32; 10];
        ok[0] = 1.0;
        assert!(mixed_radix_cfft(&mut ok, 5, false));
        for i in 0..5 {
            assert!((ok[2 * i] - 1.0).abs() < 1e-5);
            assert!(ok[2 * i + 1].abs() < 1e-5);
        }
    }
}
