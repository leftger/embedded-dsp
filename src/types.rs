//! Data types, status codes, complex structures, and fixed-point helper types.

pub use fixed::types::{I1F7 as q7, I1F15 as q15, I1F31 as q31};

/// Q8.7 fixed-point type (8 integer bits, 7 fractional bits, `i16`-backed).
///
/// Used by results that don't fit `q15`'s `[-1.0, 1.0)` range, such as
/// [`crate::audio::fast_log2_q15`]'s base-2 logarithm output.
pub type Q8F7 = fixed::FixedI16<fixed::types::extra::U7>;

/// Wide accumulator type used for dot products, sums-of-squares, and other
/// reductions that need headroom beyond `i32`. This is a plain integer, not
/// a Q1.63 fixed-point value: nothing here divides by a scale factor, and
/// several call sites divide by an element count, which would be meaningless
/// under a `[-1.0, 1.0)`-range fixed-point interpretation.
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
    a.saturating_mul(b)
}

/// Helper function for saturating multiplication in Q31 format.
#[inline(always)]
pub fn q31_mult(a: q31, b: q31) -> q31 {
    a.saturating_mul(b)
}

/// Helper function for saturating multiplication in Q7 format.
#[inline(always)]
pub fn q7_mult(a: q7, b: q7) -> q7 {
    a.saturating_mul(b)
}

/// Saturating division: returns `MAX`/`MIN` (matching the sign of `a`) on
/// division by zero, rather than panicking.
#[inline(always)]
fn saturating_div_q<F>(a: F, b: F) -> F
where
    F: fixed::traits::Fixed + PartialOrd,
{
    match a.checked_div(b) {
        Some(v) => v,
        None if b == F::ZERO => {
            if a >= F::ZERO {
                F::MAX
            } else {
                F::MIN
            }
        }
        None => {
            if (a >= F::ZERO) == (b >= F::ZERO) {
                F::MAX
            } else {
                F::MIN
            }
        }
    }
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

impl DspSample for q15 {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::MAX;

    #[inline(always)]
    fn sat_add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }

    #[inline(always)]
    fn sat_sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }

    #[inline(always)]
    fn sat_mul(self, rhs: Self) -> Self {
        self.saturating_mul(rhs)
    }

    #[inline(always)]
    fn sat_div(self, rhs: Self) -> Self {
        saturating_div_q(self, rhs)
    }

    #[inline(always)]
    fn abs_val(self) -> Self {
        self.saturating_abs()
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_num()
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::saturating_from_num(val)
    }
}

impl DspSample for q31 {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::MAX;

    #[inline(always)]
    fn sat_add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }

    #[inline(always)]
    fn sat_sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }

    #[inline(always)]
    fn sat_mul(self, rhs: Self) -> Self {
        self.saturating_mul(rhs)
    }

    #[inline(always)]
    fn sat_div(self, rhs: Self) -> Self {
        saturating_div_q(self, rhs)
    }

    #[inline(always)]
    fn abs_val(self) -> Self {
        self.saturating_abs()
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_num()
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::saturating_from_num(val)
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
