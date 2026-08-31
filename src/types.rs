//! Data types, status codes, complex structures, and fixed-point helper types.

use fixed::types::{I1F7, I1F15, I1F31};

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
    I1F15::from_bits(a)
        .saturating_mul(I1F15::from_bits(b))
        .to_bits()
}

/// Helper function for saturating multiplication in Q31 format.
#[inline(always)]
pub fn q31_mult(a: q31, b: q31) -> q31 {
    I1F31::from_bits(a)
        .saturating_mul(I1F31::from_bits(b))
        .to_bits()
}

/// Helper function for saturating multiplication in Q7 format.
#[inline(always)]
pub fn q7_mult(a: q7, b: q7) -> q7 {
    I1F7::from_bits(a)
        .saturating_mul(I1F7::from_bits(b))
        .to_bits()
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified DSP Sample Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Unified numerical sample trait implemented for floating-point sample types.
///
/// Enables writing generic filters, delay lines, oscillators, and processing blocks that operate
/// seamlessly with `f32` and `f64`.
pub trait DspSample:
    Copy
    + Default
    + PartialEq
    + PartialOrd
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Neg<Output = Self>
{
    /// Additive identity (`0.0`).
    const ZERO: Self;
    /// Multiplicative identity or normalized unity (`1.0`).
    const ONE: Self;

    /// Saturating addition.
    fn sat_add(self, rhs: Self) -> Self;
    /// Saturating subtraction.
    fn sat_sub(self, rhs: Self) -> Self;
    /// Saturating multiplication.
    fn sat_mul(self, rhs: Self) -> Self;
    /// Saturating division.
    fn sat_div(self, rhs: Self) -> Self;
    /// Absolute value.
    fn abs_val(self) -> Self;
    /// Convert to floating-point `f32`.
    fn to_f32(self) -> f32;
    /// Convert from floating-point `f32`.
    fn from_f32(val: f32) -> Self;
}

impl DspSample for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline(always)]
    fn sat_add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn sat_sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn sat_mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn sat_div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn abs_val(self) -> Self {
        if self < 0.0 { -self } else { self }
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        val
    }
}

impl DspSample for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline(always)]
    fn sat_add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn sat_sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn sat_mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn sat_div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn abs_val(self) -> Self {
        if self < 0.0 { -self } else { self }
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        val as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Complex Number Operations
// ─────────────────────────────────────────────────────────────────────────────

impl<T: core::ops::Add<Output = T>> core::ops::Add for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        }
    }
}

impl<T: core::ops::Sub<Output = T>> core::ops::Sub for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            real: self.real - rhs.real,
            imag: self.imag - rhs.imag,
        }
    }
}

impl<T: core::ops::Neg<Output = T>> core::ops::Neg for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            real: -self.real,
            imag: -self.imag,
        }
    }
}

impl<T: Copy + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T>>
    core::ops::Mul for Complex<T>
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        }
    }
}

impl<T: Copy + core::ops::Mul<Output = T>> core::ops::Mul<T> for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, scalar: T) -> Self {
        Self {
            real: self.real * scalar,
            imag: self.imag * scalar,
        }
    }
}
