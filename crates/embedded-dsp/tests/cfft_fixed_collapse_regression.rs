//! Bit-exact regression coverage for the Stage 6d collapse of `cfft_q15`/`cfft_q31` and
//! `cfft_bfp_q15`/`cfft_bfp_q31` into a shared generic core (see `transform.rs`).
//!
//! These reimplement the exact hand-written algorithms the generic core replaced (as they existed
//! immediately before the collapse), independently of `transform.rs`, and compare outputs
//! element-by-element across many sizes, both FFT directions, and both bit-reversal settings.

use embedded_dsp::transform::{cfft_bfp_q15, cfft_bfp_q31, cfft_q15, cfft_q31};
use embedded_dsp::types::{q15, q31};

const TWIDDLE_N: usize = 512;

fn cos_taylor(x: f32) -> f32 {
    let mut x = x;
    while x > core::f32::consts::PI {
        x -= 2.0 * core::f32::consts::PI;
    }
    while x < -core::f32::consts::PI {
        x += 2.0 * core::f32::consts::PI;
    }
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0
}

fn sin_taylor(x: f32) -> f32 {
    let mut x = x;
    while x > core::f32::consts::PI {
        x -= 2.0 * core::f32::consts::PI;
    }
    while x < -core::f32::consts::PI {
        x += 2.0 * core::f32::consts::PI;
    }
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0
}

fn twiddle_q15_legacy(k: usize, n: usize) -> (i16, i16) {
    let idx = k.wrapping_mul(TWIDDLE_N / n) & (TWIDDLE_N - 1);
    let a = (idx as f32) * 2.0 * core::f32::consts::PI / TWIDDLE_N as f32;
    let quant = |v: f32| -> i16 {
        if v >= 32767.0 {
            32767
        } else if v <= -32768.0 {
            -32768
        } else {
            v as i16
        }
    };
    (
        quant(cos_taylor(a) * 32767.0),
        quant(sin_taylor(a) * 32767.0),
    )
}

fn bit_reversal_legacy<T: Copy>(data: &mut [T], n: usize) {
    let mut j = 0;
    for i in 0..n {
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

fn sat_q15_legacy(v: i32) -> q15 {
    q15::from_bits(v.clamp(i16::MIN as i32, i16::MAX as i32) as i16)
}

fn sat_q31_legacy(v: i64) -> q31 {
    q31::from_bits(v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
}

/// Faithful reimplementation of the pre-collapse `cfft_q31`.
fn cfft_q31_legacy(data: &mut [q31], n: usize, ifft_flag: u8, bit_reverse_flag: u8) {
    if !(2..=TWIDDLE_N).contains(&n) || (n & (n - 1)) != 0 || data.len() < 2 * n {
        return;
    }
    if bit_reverse_flag != 0 {
        bit_reversal_legacy(data, n);
    }
    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let mut i = 0;
        while i < n {
            for j in 0..half_len {
                let (w_re_s, w_im_s) = twiddle_q15_legacy(j, len);
                let w_re = (w_re_s as i32) << 16;
                let mut w_im = (w_im_s as i32) << 16;
                if ifft_flag == 0 {
                    w_im = -w_im;
                }
                let u_idx = 2 * (i + j);
                let v_idx = 2 * (i + j + half_len);
                let u_re = data[u_idx].to_bits() as i64;
                let u_im = data[u_idx + 1].to_bits() as i64;
                let v_re = data[v_idx].to_bits() as i64;
                let v_im = data[v_idx + 1].to_bits() as i64;
                let wr = w_re as i64;
                let wi = w_im as i64;
                let t_re = (v_re * wr - v_im * wi) >> 31;
                let t_im = (v_re * wi + v_im * wr) >> 31;
                data[u_idx] = sat_q31_legacy((u_re + t_re) >> 1);
                data[u_idx + 1] = sat_q31_legacy((u_im + t_im) >> 1);
                data[v_idx] = sat_q31_legacy((u_re - t_re) >> 1);
                data[v_idx + 1] = sat_q31_legacy((u_im - t_im) >> 1);
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Faithful reimplementation of the pre-collapse `cfft_q15`.
fn cfft_q15_legacy(data: &mut [q15], n: usize, ifft_flag: u8, bit_reverse_flag: u8) {
    if !(2..=TWIDDLE_N).contains(&n) || (n & (n - 1)) != 0 || data.len() < 2 * n {
        return;
    }
    if bit_reverse_flag != 0 {
        bit_reversal_legacy(data, n);
    }
    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let mut i = 0;
        while i < n {
            for j in 0..half_len {
                let (w_re_s, mut w_im_s) = twiddle_q15_legacy(j, len);
                if ifft_flag == 0 {
                    w_im_s = w_im_s.saturating_neg();
                }
                let u_idx = 2 * (i + j);
                let v_idx = 2 * (i + j + half_len);
                let u_re = data[u_idx].to_bits() as i32;
                let u_im = data[u_idx + 1].to_bits() as i32;
                let v_re = data[v_idx].to_bits() as i32;
                let v_im = data[v_idx + 1].to_bits() as i32;
                let wr = w_re_s as i32;
                let wi = w_im_s as i32;
                let t_re = (v_re * wr - v_im * wi) >> 15;
                let t_im = (v_re * wi + v_im * wr) >> 15;
                data[u_idx] = sat_q15_legacy((u_re + t_re) >> 1);
                data[u_idx + 1] = sat_q15_legacy((u_im + t_im) >> 1);
                data[v_idx] = sat_q15_legacy((u_re - t_re) >> 1);
                data[v_idx + 1] = sat_q15_legacy((u_im - t_im) >> 1);
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Faithful reimplementation of the pre-collapse `cfft_bfp_q15`.
fn cfft_bfp_q15_legacy(data: &mut [q15], n: usize, ifft_flag: u8, bit_reverse_flag: u8) -> u16 {
    if !(2..=TWIDDLE_N).contains(&n) || (n & (n - 1)) != 0 || data.len() < 2 * n {
        return 0;
    }
    if bit_reverse_flag != 0 {
        bit_reversal_legacy(data, n);
    }
    let mut scale_count: u16 = 0;
    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let mut max_val: i16 = 0;
        for i in 0..2 * n {
            let val = data[i].abs().to_bits();
            if val > max_val {
                max_val = val;
            }
        }
        let stage_shift = if max_val > 16383 {
            scale_count += 1;
            1
        } else {
            0
        };
        let mut i = 0;
        while i < n {
            for j in 0..half_len {
                let (w_re_s, mut w_im_s) = twiddle_q15_legacy(j, len);
                if ifft_flag == 0 {
                    w_im_s = w_im_s.saturating_neg();
                }
                let u_idx = 2 * (i + j);
                let v_idx = 2 * (i + j + half_len);
                let u_re = data[u_idx].to_bits() as i32;
                let u_im = data[u_idx + 1].to_bits() as i32;
                let v_re = data[v_idx].to_bits() as i32;
                let v_im = data[v_idx + 1].to_bits() as i32;
                let wr = w_re_s as i32;
                let wi = w_im_s as i32;
                let t_re = (v_re * wr - v_im * wi) >> 15;
                let t_im = (v_re * wi + v_im * wr) >> 15;
                data[u_idx] = sat_q15_legacy((u_re + t_re) >> stage_shift);
                data[u_idx + 1] = sat_q15_legacy((u_im + t_im) >> stage_shift);
                data[v_idx] = sat_q15_legacy((u_re - t_re) >> stage_shift);
                data[v_idx + 1] = sat_q15_legacy((u_im - t_im) >> stage_shift);
            }
            i += len;
        }
        len <<= 1;
    }
    scale_count
}

/// Faithful reimplementation of the pre-collapse `cfft_bfp_q31`.
fn cfft_bfp_q31_legacy(data: &mut [q31], n: usize, ifft_flag: u8, bit_reverse_flag: u8) -> u16 {
    if !(2..=TWIDDLE_N).contains(&n) || (n & (n - 1)) != 0 || data.len() < 2 * n {
        return 0;
    }
    if bit_reverse_flag != 0 {
        bit_reversal_legacy(data, n);
    }
    let mut scale_count: u16 = 0;
    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let mut max_val: i32 = 0;
        for i in 0..2 * n {
            let val = data[i].abs().to_bits();
            if val > max_val {
                max_val = val;
            }
        }
        let stage_shift = if max_val > 1073741823 {
            scale_count += 1;
            1
        } else {
            0
        };
        let mut i = 0;
        while i < n {
            for j in 0..half_len {
                let (w_re_s, w_im_s) = twiddle_q15_legacy(j, len);
                let w_re = (w_re_s as i32) << 16;
                let mut w_im = (w_im_s as i32) << 16;
                if ifft_flag == 0 {
                    w_im = -w_im;
                }
                let u_idx = 2 * (i + j);
                let v_idx = 2 * (i + j + half_len);
                let u_re = data[u_idx].to_bits() as i64;
                let u_im = data[u_idx + 1].to_bits() as i64;
                let v_re = data[v_idx].to_bits() as i64;
                let v_im = data[v_idx + 1].to_bits() as i64;
                let wr = w_re as i64;
                let wi = w_im as i64;
                let t_re = (v_re * wr - v_im * wi) >> 31;
                let t_im = (v_re * wi + v_im * wr) >> 31;
                data[u_idx] = sat_q31_legacy((u_re + t_re) >> stage_shift);
                data[u_idx + 1] = sat_q31_legacy((u_im + t_im) >> stage_shift);
                data[v_idx] = sat_q31_legacy((u_re - t_re) >> stage_shift);
                data[v_idx + 1] = sat_q31_legacy((u_im - t_im) >> stage_shift);
            }
            i += len;
        }
        len <<= 1;
    }
    scale_count
}

/// Deterministic pseudo-random test signal, full-scale enough to exercise BFP's headroom scan.
fn signal_q15(n: usize) -> Vec<q15> {
    let mut state = 0x1234_5678u32;
    (0..2 * n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            q15::from_bits((state >> 16) as i16)
        })
        .collect()
}

fn signal_q31(n: usize) -> Vec<q31> {
    let mut state = 0x9E37_79B9u64;
    (0..2 * n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            q31::from_bits((state >> 32) as i32)
        })
        .collect()
}

const SIZES: [usize; 9] = [2, 4, 8, 16, 32, 64, 128, 256, 512];

#[test]
fn cfft_q15_matches_the_legacy_kernel_across_sizes_and_directions() {
    for &n in &SIZES {
        for ifft_flag in [0u8, 1u8] {
            for bit_reverse_flag in [0u8, 1u8] {
                let src = signal_q15(n);
                let mut a = src.clone();
                let mut b = src.clone();
                cfft_q15(&mut a, n, ifft_flag, bit_reverse_flag);
                cfft_q15_legacy(&mut b, n, ifft_flag, bit_reverse_flag);
                assert_eq!(
                    a, b,
                    "cfft_q15 diverged at n={n}, ifft={ifft_flag}, bitrev={bit_reverse_flag}"
                );
            }
        }
    }
}

#[test]
fn cfft_q31_matches_the_legacy_kernel_across_sizes_and_directions() {
    for &n in &SIZES {
        for ifft_flag in [0u8, 1u8] {
            for bit_reverse_flag in [0u8, 1u8] {
                let src = signal_q31(n);
                let mut a = src.clone();
                let mut b = src.clone();
                cfft_q31(&mut a, n, ifft_flag, bit_reverse_flag);
                cfft_q31_legacy(&mut b, n, ifft_flag, bit_reverse_flag);
                assert_eq!(
                    a, b,
                    "cfft_q31 diverged at n={n}, ifft={ifft_flag}, bitrev={bit_reverse_flag}"
                );
            }
        }
    }
}

#[test]
fn cfft_bfp_q15_matches_the_legacy_kernel_across_sizes_and_directions() {
    for &n in &SIZES {
        for ifft_flag in [0u8, 1u8] {
            for bit_reverse_flag in [0u8, 1u8] {
                let src = signal_q15(n);
                let mut a = src.clone();
                let mut b = src.clone();
                let scale_a = cfft_bfp_q15(&mut a, n, ifft_flag, bit_reverse_flag);
                let scale_b = cfft_bfp_q15_legacy(&mut b, n, ifft_flag, bit_reverse_flag);
                assert_eq!(
                    a, b,
                    "cfft_bfp_q15 data diverged at n={n}, ifft={ifft_flag}, bitrev={bit_reverse_flag}"
                );
                assert_eq!(
                    scale_a, scale_b,
                    "cfft_bfp_q15 scale_count diverged at n={n}, ifft={ifft_flag}, bitrev={bit_reverse_flag}"
                );
            }
        }
    }
}

#[test]
fn cfft_bfp_q31_matches_the_legacy_kernel_across_sizes_and_directions() {
    for &n in &SIZES {
        for ifft_flag in [0u8, 1u8] {
            for bit_reverse_flag in [0u8, 1u8] {
                let src = signal_q31(n);
                let mut a = src.clone();
                let mut b = src.clone();
                let scale_a = cfft_bfp_q31(&mut a, n, ifft_flag, bit_reverse_flag);
                let scale_b = cfft_bfp_q31_legacy(&mut b, n, ifft_flag, bit_reverse_flag);
                assert_eq!(
                    a, b,
                    "cfft_bfp_q31 data diverged at n={n}, ifft={ifft_flag}, bitrev={bit_reverse_flag}"
                );
                assert_eq!(
                    scale_a, scale_b,
                    "cfft_bfp_q31 scale_count diverged at n={n}, ifft={ifft_flag}, bitrev={bit_reverse_flag}"
                );
            }
        }
    }
}
