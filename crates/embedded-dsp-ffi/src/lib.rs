//! C ABI for `embedded-dsp`.
//!
//! Exposes the hot kernels — FIR, Direct Form I biquad cascades, and the complex
//! FFT — plus the audio-EQ designer as a plain C API, so C and C++ firmware can
//! link the same verified Rust implementations the Rust API uses.
//!
//! Built as both a static library (`libembedded_dsp_ffi.a`) and a shared library
//! (`libembedded_dsp_ffi.so`/`.dylib`/`.dll`). The matching header is
//! [`include/embedded_dsp.h`](../../../include/embedded_dsp.h).
//!
//! Every fallible entry point returns [`EDS_OK`] (`0`) on success and a negative
//! error code on bad input. A panic is caught at the boundary and reported as
//! [`EDS_ERROR_PANIC`] rather than unwinding into C.
//!
//! The kernels are *streaming*: `state` is read and updated in place and must be
//! zeroed by the caller exactly once, before the first call. The functions rebuild
//! a zero-cost instance around the caller's buffers each call; they never reset
//! the state themselves.

#![deny(missing_docs)]

use core::ffi::c_char;
use core::panic::AssertUnwindSafe;

use embedded_dsp::filter_design::{BiquadType, EqFilter};
use embedded_dsp::filtering::{BiquadCascadeInstance, FirInstance, biquad_cascade_df1, fir};
use embedded_dsp::transform::cfft_f32;

/// Success.
pub const EDS_OK: i32 = 0;
/// A required pointer was null.
pub const EDS_ERROR_NULL: i32 = -1;
/// A length or index was out of range.
pub const EDS_ERROR_LENGTH: i32 = -2;
/// An enum discriminant was not recognised.
pub const EDS_ERROR_TYPE: i32 = -3;
/// The call panicked; it was caught at the FFI boundary.
pub const EDS_ERROR_PANIC: i32 = -4;
/// A parameter failed validation (e.g. frequency outside `(0, fs/2)`).
pub const EDS_ERROR_ARG: i32 = -5;

const VERSION_C: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// Returns the library version as a NUL-terminated string (e.g. `"0.6.0"`).
///
/// The pointer is to a `'static` string owned by the library and must not be freed.
#[unsafe(no_mangle)]
pub extern "C" fn eds_version() -> *const c_char {
    VERSION_C.as_ptr().cast()
}

/// Runs `f`, converting a panic into [`EDS_ERROR_PANIC`].
fn catch(f: impl FnOnce() -> i32) -> i32 {
    std::panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or(EDS_ERROR_PANIC)
}

/// # Safety
/// `ptr` must be valid for `len` consecutive `f32` reads.
unsafe fn slice_in<'a>(ptr: *const f32, len: usize) -> Option<&'a [f32]> {
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(ptr, len) })
    }
}

/// # Safety
/// `ptr` must be valid for `len` consecutive `f32` writes.
unsafe fn slice_out<'a>(ptr: *mut f32, len: usize) -> Option<&'a mut [f32]> {
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
    }
}

/// Maps the C discriminant to a [`BiquadType`].
fn eq_type(disc: u32) -> Option<BiquadType> {
    Some(match disc {
        0 => BiquadType::Lowpass,
        1 => BiquadType::Highpass,
        2 => BiquadType::Bandpass,
        3 => BiquadType::Allpass,
        4 => BiquadType::Notch,
        5 => BiquadType::Peaking,
        6 => BiquadType::Lowshelf,
        7 => BiquadType::Highshelf,
        8 => BiquadType::Iho,
        _ => return None,
    })
}

/// FIR filter, `f32`, in place over a block.
///
/// `coeffs` holds `num_taps` coefficients, `state` holds `num_taps` values (zeroed
/// by the caller before the first call), and `src`/`dst` are `len` samples each.
///
/// # Safety
/// All pointers must be non-null and valid for the lengths above; `coeffs`/`src`
/// are read, `state`/`dst` are written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eds_fir_f32(
    num_taps: u32,
    coeffs: *const f32,
    state: *mut f32,
    src: *const f32,
    dst: *mut f32,
    len: u32,
) -> i32 {
    catch(|| {
        let taps = num_taps as usize;
        if taps == 0 || taps > u16::MAX as usize {
            return EDS_ERROR_LENGTH;
        }
        let (Some(coeffs), Some(state), Some(src), Some(dst)) = (
            unsafe { slice_in(coeffs, taps) },
            unsafe { slice_out(state, taps) },
            unsafe { slice_in(src, len as usize) },
            unsafe { slice_out(dst, len as usize) },
        ) else {
            return EDS_ERROR_NULL;
        };
        let mut instance = FirInstance::<f32> {
            num_taps: taps as u16,
            coeffs,
            state,
        };
        fir(&mut instance, src, dst);
        EDS_OK
    })
}

/// Direct Form I biquad cascade, `f32`, in place over a block.
///
/// `coeffs` holds `5 * num_stages` values `[b0, b1, b2, a1, a2]` per stage,
/// `state` holds `4 * num_stages` values, and `src`/`dst` are `len` samples each.
/// `post_shift` provides extra fixed-point headroom; `f32` ignores it (pass `0`).
///
/// # Safety
/// All pointers must be non-null and valid for the lengths above.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eds_biquad_cascade_df1_f32(
    num_stages: u32,
    post_shift: u32,
    coeffs: *const f32,
    state: *mut f32,
    src: *const f32,
    dst: *mut f32,
    len: u32,
) -> i32 {
    catch(|| {
        let stages = num_stages as usize;
        if stages == 0 || stages > u8::MAX as usize || post_shift > u8::MAX as u32 {
            return EDS_ERROR_LENGTH;
        }
        let (Some(coeffs), Some(state), Some(src), Some(dst)) = (
            unsafe { slice_in(coeffs, stages * 5) },
            unsafe { slice_out(state, stages * 4) },
            unsafe { slice_in(src, len as usize) },
            unsafe { slice_out(dst, len as usize) },
        ) else {
            return EDS_ERROR_NULL;
        };
        let mut instance = BiquadCascadeInstance::<f32> {
            num_stages: stages as u8,
            post_shift: post_shift as u8,
            coeffs,
            state,
        };
        biquad_cascade_df1(&mut instance, src, dst);
        EDS_OK
    })
}

/// In-place complex FFT of `n_complex` interleaved `(re, im)` pairs.
///
/// `data` holds `2 * n_complex` floats; `ifft_flag` selects the inverse transform.
/// Scaling matches [`cfft_f32`]: the forward transform is unnormalised.
///
/// # Safety
/// `data` must be non-null and valid for `2 * n_complex` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eds_cfft_f32(data: *mut f32, n_complex: u32, ifft_flag: u32) -> i32 {
    catch(|| {
        let n = n_complex as usize;
        let Some(data) = (unsafe { slice_out(data, 2 * n) }) else {
            return EDS_ERROR_NULL;
        };
        cfft_f32(data, n, ifft_flag as u8, 1);
        EDS_OK
    })
}

/// Designs audio-EQ biquad coefficients `[b0, b1, b2, a1, a2]` (Direct Form I).
///
/// `typ` is the response: `0` low-pass, `1` high-pass, `2` band-pass, `3` all-pass,
/// `4` notch, `5` peaking, `6` low shelf, `7` high shelf, `8` `IHo`. `gain_db`
/// applies to peaking, the shelves, and `IHo`. Writes the passthrough biquad
/// `[1, 0, 0, 0, 0]` on [`EDS_ERROR_ARG`].
///
/// # Safety
/// `out_coeffs` must be non-null and valid for 5 floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eds_eq_coeffs(
    typ: u32,
    frequency_hz: f32,
    sample_rate_hz: f32,
    q: f32,
    gain_db: f32,
    out_coeffs: *mut f32,
) -> i32 {
    catch(|| {
        let Some(kind) = eq_type(typ) else {
            return EDS_ERROR_TYPE;
        };
        let Some(out) = (unsafe { slice_out(out_coeffs, 5) }) else {
            return EDS_ERROR_NULL;
        };
        match EqFilter::new(frequency_hz, sample_rate_hz)
            .q(q)
            .gain_db(gain_db)
            .try_build(kind)
        {
            Ok(coeffs) => {
                out.copy_from_slice(&coeffs);
                EDS_OK
            }
            Err(_) => {
                out.copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0]);
                EDS_ERROR_ARG
            }
        }
    })
}
