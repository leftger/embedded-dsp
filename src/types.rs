//! Data types, status codes, complex structures, and fixed-point helper types.

#[allow(non_camel_case_types)]
pub type q7 = i8;
#[allow(non_camel_case_types)]
pub type q15 = i16;
#[allow(non_camel_case_types)]
pub type q31 = i32;
#[allow(non_camel_case_types)]
pub type q63 = i64;
#[allow(non_camel_case_types)]
pub type f32_t = f32;
#[allow(non_camel_case_types)]
pub type f64_t = f64;

/// Error status returned by functions in `embedded-dsp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Status {
    /// Operation succeeded without error.
    Success = 0,
    /// One or more arguments are invalid.
    ArgumentError = -1,
    /// Length of data buffer is invalid or mismatching.
    LengthError = -2,
    /// Matrix dimensions are incompatible.
    SizeMismatch = -3,
    /// NaN or Infinity was produced during computation.
    NanInf = -4,
    /// Matrix is singular and cannot be inverted.
    Singular = -5,
    /// Test or verification failed.
    TestFailure = -6,
    /// Matrix decomposition failed.
    DecompositionFailure = -7,
}

/// Representation of a complex number with real and imaginary components.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Complex<T> {
    pub real: T,
    pub imag: T,
}

impl<T> Complex<T> {
    #[inline(always)]
    pub const fn new(real: T, imag: T) -> Self {
        Self { real, imag }
    }
}

/// Helper function for saturating multiplication in Q15 format.
#[inline(always)]
pub fn q15_mult(a: q15, b: q15) -> q15 {
    let mul = (a as i32 * b as i32) >> 15;
    mul.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Helper function for saturating multiplication in Q31 format.
#[inline(always)]
pub fn q31_mult(a: q31, b: q31) -> q31 {
    let mul = (a as i64 * b as i64) >> 31;
    mul.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Helper function for saturating multiplication in Q7 format.
#[inline(always)]
pub fn q7_mult(a: q7, b: q7) -> q7 {
    let mul = (a as i32 * b as i32) >> 7;
    mul.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}
