//! SIMD and DSP hardware acceleration intrinsics hooks.
//!
//! Provides hardware-accelerated dual 16-bit MAC (`smlad`), saturating vector arithmetic (`qadd16`, `qsub16`),
//! and 32/64-bit saturators (`ssat`), with seamless fallbacks for non-ARM/FPU architectures.

use crate::types::{q15, q15_mult, q31, q63};

/// Hardware or SWAR dual 16-bit signed multiply-accumulate: `acc + (a_lo * b_lo) + (a_hi * b_hi)`.
#[inline(always)]
pub fn dual_mac_q15(a_packed: u32, b_packed: u32, acc: i32) -> i32 {
    #[cfg(all(
        target_arch = "arm",
        any(target_feature = "dsp", feature = "cortex-m-dsp")
    ))]
    {
        let res: i32;
        unsafe {
            core::arch::asm!(
                "smlad {out}, {a}, {b}, {acc}",
                a = in(reg) a_packed,
                b = in(reg) b_packed,
                acc = in(reg) acc,
                out = lateout(reg) res,
                options(pure, nomem, nostack)
            );
        }
        res
    }
    #[cfg(not(all(
        target_arch = "arm",
        any(target_feature = "dsp", feature = "cortex-m-dsp")
    )))]
    {
        let a_lo = (a_packed as i16) as i32;
        let a_hi = ((a_packed >> 16) as i16) as i32;
        let b_lo = (b_packed as i16) as i32;
        let b_hi = ((b_packed >> 16) as i16) as i32;
        acc.wrapping_add(a_lo * b_lo).wrapping_add(a_hi * b_hi)
    }
}

/// Hardware or SWAR dual 16-bit signed multiply-accumulate into a 64-bit accumulator (`smlald`).
#[inline(always)]
pub fn dual_mac_q63(a_packed: u32, b_packed: u32, acc: i64) -> i64 {
    #[cfg(all(
        target_arch = "arm",
        any(target_feature = "dsp", feature = "cortex-m-dsp")
    ))]
    {
        let acc_lo = acc as u32;
        let acc_hi = (acc >> 32) as u32;
        let out_lo: u32;
        let out_hi: u32;
        unsafe {
            core::arch::asm!(
                "smlald {out_lo}, {out_hi}, {a}, {b}",
                a = in(reg) a_packed,
                b = in(reg) b_packed,
                out_lo = inout(reg) acc_lo => out_lo,
                out_hi = inout(reg) acc_hi => out_hi,
                options(pure, nomem, nostack)
            );
        }
        ((out_hi as i64) << 32) | (out_lo as i64)
    }
    #[cfg(not(all(
        target_arch = "arm",
        any(target_feature = "dsp", feature = "cortex-m-dsp")
    )))]
    {
        let a_lo = (a_packed as i16) as i64;
        let a_hi = ((a_packed >> 16) as i16) as i64;
        let b_lo = (b_packed as i16) as i64;
        let b_hi = ((b_packed >> 16) as i16) as i64;
        acc.wrapping_add(a_lo * b_lo).wrapping_add(a_hi * b_hi)
    }
}

/// Dual 16-bit saturating addition (`qadd16`).
#[inline(always)]
pub fn dual_saturating_add_q15(a_packed: u32, b_packed: u32) -> u32 {
    #[cfg(all(
        target_arch = "arm",
        any(target_feature = "dsp", feature = "cortex-m-dsp")
    ))]
    {
        let res: u32;
        unsafe {
            core::arch::asm!(
                "qadd16 {out}, {a}, {b}",
                a = in(reg) a_packed,
                b = in(reg) b_packed,
                out = lateout(reg) res,
                options(pure, nomem, nostack)
            );
        }
        res
    }
    #[cfg(not(all(
        target_arch = "arm",
        any(target_feature = "dsp", feature = "cortex-m-dsp")
    )))]
    {
        let a_lo = a_packed as i16;
        let a_hi = (a_packed >> 16) as i16;
        let b_lo = b_packed as i16;
        let b_hi = (b_packed >> 16) as i16;
        let r_lo = (a_lo.saturating_add(b_lo) as u16) as u32;
        let r_hi = (a_hi.saturating_add(b_hi) as u16) as u32;
        r_lo | (r_hi << 16)
    }
}

/// Dual 16-bit saturating subtraction (`qsub16`).
#[inline(always)]
pub fn dual_saturating_sub_q15(a_packed: u32, b_packed: u32) -> u32 {
    #[cfg(all(
        target_arch = "arm",
        any(target_feature = "dsp", feature = "cortex-m-dsp")
    ))]
    {
        let res: u32;
        unsafe {
            core::arch::asm!(
                "qsub16 {out}, {a}, {b}",
                a = in(reg) a_packed,
                b = in(reg) b_packed,
                out = lateout(reg) res,
                options(pure, nomem, nostack)
            );
        }
        res
    }
    #[cfg(not(all(
        target_arch = "arm",
        any(target_feature = "dsp", feature = "cortex-m-dsp")
    )))]
    {
        let a_lo = a_packed as i16;
        let a_hi = (a_packed >> 16) as i16;
        let b_lo = b_packed as i16;
        let b_hi = (b_packed >> 16) as i16;
        let r_lo = (a_lo.saturating_sub(b_lo) as u16) as u32;
        let r_hi = (a_hi.saturating_sub(b_hi) as u16) as u32;
        r_lo | (r_hi << 16)
    }
}

/// Signed 16-bit saturation.
#[inline(always)]
pub fn saturate_q15(val: i32) -> q15 {
    #[cfg(all(
        target_arch = "arm",
        any(target_feature = "dsp", feature = "cortex-m-dsp")
    ))]
    {
        let res: i32;
        unsafe {
            core::arch::asm!(
                "ssat {out}, #16, {val}",
                val = in(reg) val,
                out = lateout(reg) res,
                options(pure, nomem, nostack)
            );
        }
        res as q15
    }
    #[cfg(not(all(
        target_arch = "arm",
        any(target_feature = "dsp", feature = "cortex-m-dsp")
    )))]
    {
        val.clamp(i16::MIN as i32, i16::MAX as i32) as q15
    }
}

/// Signed 32-bit saturation for Q31 results.
#[inline(always)]
pub fn saturate_q31(val: i64) -> q31 {
    val.clamp(i32::MIN as i64, i32::MAX as i64) as q31
}

/// High-throughput unrolled dot product in Q15 with SIMD / SWAR acceleration.
pub fn simd_dot_prod_q15(src_a: &[q15], src_b: &[q15]) -> q63 {
    let len = src_a.len().min(src_b.len());
    let mut sum: q63 = 0;
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        let p_a0 = (src_a[idx] as u16 as u32) | ((src_a[idx + 1] as u16 as u32) << 16);
        let p_b0 = (src_b[idx] as u16 as u32) | ((src_b[idx + 1] as u16 as u32) << 16);
        let p_a1 = (src_a[idx + 2] as u16 as u32) | ((src_a[idx + 3] as u16 as u32) << 16);
        let p_b1 = (src_b[idx + 2] as u16 as u32) | ((src_b[idx + 3] as u16 as u32) << 16);

        sum = dual_mac_q63(p_a0, p_b0, sum);
        sum = dual_mac_q63(p_a1, p_b1, sum);
    }

    let offset = chunks * 4;
    for i in 0..remainder {
        sum += (src_a[offset + i] as i32 * src_b[offset + i] as i32) as q63;
    }

    sum
}

/// High-throughput saturating addition in Q15 with 2-way packed SIMD.
pub fn simd_add_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    let pairs = len / 2;
    let remainder = len % 2;

    for i in 0..pairs {
        let idx = i * 2;
        let p_a = (src_a[idx] as u16 as u32) | ((src_a[idx + 1] as u16 as u32) << 16);
        let p_b = (src_b[idx] as u16 as u32) | ((src_b[idx + 1] as u16 as u32) << 16);
        let res = dual_saturating_add_q15(p_a, p_b);
        dst[idx] = res as q15;
        dst[idx + 1] = (res >> 16) as q15;
    }

    if remainder != 0 {
        let last = len - 1;
        dst[last] = src_a[last].saturating_add(src_b[last]);
    }
}

/// High-throughput saturating subtraction in Q15 with 2-way packed SIMD.
pub fn simd_sub_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    let pairs = len / 2;
    let remainder = len % 2;

    for i in 0..pairs {
        let idx = i * 2;
        let p_a = (src_a[idx] as u16 as u32) | ((src_a[idx + 1] as u16 as u32) << 16);
        let p_b = (src_b[idx] as u16 as u32) | ((src_b[idx + 1] as u16 as u32) << 16);
        let res = dual_saturating_sub_q15(p_a, p_b);
        dst[idx] = res as q15;
        dst[idx + 1] = (res >> 16) as q15;
    }

    if remainder != 0 {
        let last = len - 1;
        dst[last] = src_a[last].saturating_sub(src_b[last]);
    }
}

/// High-throughput saturating elementwise multiplication in Q15 with 4-way unrolling.
pub fn simd_mult_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        dst[idx] = q15_mult(src_a[idx], src_b[idx]);
        dst[idx + 1] = q15_mult(src_a[idx + 1], src_b[idx + 1]);
        dst[idx + 2] = q15_mult(src_a[idx + 2], src_b[idx + 2]);
        dst[idx + 3] = q15_mult(src_a[idx + 3], src_b[idx + 3]);
    }

    let offset = chunks * 4;
    for i in 0..remainder {
        dst[offset + i] = q15_mult(src_a[offset + i], src_b[offset + i]);
    }
}
