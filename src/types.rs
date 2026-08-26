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

// ─────────────────────────────────────────────────────────────────────────────
// Strongly-typed Fixed-Point Newtypes (Q15, Q31)
// ─────────────────────────────────────────────────────────────────────────────

/// Strongly-typed Q1.15 fixed-point number stored in an `i16`.
///
/// Range is `[-1.0, 0.999969482421875]`.
/// Provides operator overloading (`+`, `-`, `*`, `/`) with saturating fixed-point arithmetic.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Q15(pub i16);

impl Q15 {
    /// Zero representation (`0.0`).
    pub const ZERO: Self = Self(0);
    /// Maximum positive value (`0.999969482...` = `32767`).
    pub const ONE: Self = Self(i16::MAX);
    /// Minimum representable value (`-1.0` = `-32768`).
    pub const MIN: Self = Self(i16::MIN);
    /// Maximum representable value (`0.999969482...` = `32767`).
    pub const MAX: Self = Self(i16::MAX);
    /// Smallest positive step (`1 / 32768`).
    pub const DELTA: Self = Self(1);

    /// Create from raw integer bits.
    #[inline(always)]
    pub const fn from_bits(bits: i16) -> Self {
        Self(bits)
    }

    /// Retrieve raw integer bits.
    #[inline(always)]
    pub const fn to_bits(self) -> i16 {
        self.0
    }

    /// Convert from `f32` with rounding and saturation to `[-1.0, 0.999969]`.
    #[inline(always)]
    pub fn from_f32(v: f32) -> Self {
        let scaled = v * 32768.0 + if v >= 0.0 { 0.5 } else { -0.5 };
        Self(scaled.clamp(-32768.0, 32767.0) as i16)
    }

    /// Convert to `f32`.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / 32768.0
    }

    /// Saturating addition.
    #[inline(always)]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction.
    #[inline(always)]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Saturating Q15 multiplication.
    #[inline(always)]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        Self(q15_mult(self.0, rhs.0))
    }

    /// Saturating Q15 division with division-by-zero protection.
    #[inline(always)]
    pub fn saturating_div(self, rhs: Self) -> Self {
        if rhs.0 == 0 {
            if self.0 >= 0 { Self::MAX } else { Self::MIN }
        } else {
            let num = (self.0 as i32) << 15;
            let res = num / (rhs.0 as i32);
            Self(res.clamp(i16::MIN as i32, i16::MAX as i32) as i16)
        }
    }

    /// Absolute value (saturating at `Self::MAX` for `-1.0`).
    #[inline(always)]
    pub const fn abs(self) -> Self {
        if self.0 == i16::MIN {
            Self::MAX
        } else {
            Self(self.0.abs())
        }
    }

    /// Clamp to `[min, max]`.
    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }
}

impl core::ops::Add for Q15 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }
}

impl core::ops::Sub for Q15 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }
}

impl core::ops::Mul for Q15 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.saturating_mul(rhs)
    }
}

impl core::ops::Div for Q15 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        self.saturating_div(rhs)
    }
}

impl core::ops::Neg for Q15 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        if self.0 == i16::MIN {
            Self::MAX
        } else {
            Self(-self.0)
        }
    }
}

impl core::ops::AddAssign for Q15 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for Q15 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for Q15 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::DivAssign for Q15 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl From<i16> for Q15 {
    #[inline(always)]
    fn from(v: i16) -> Self {
        Self(v)
    }
}

impl From<Q15> for i16 {
    #[inline(always)]
    fn from(v: Q15) -> Self {
        v.0
    }
}

impl From<f32> for Q15 {
    #[inline(always)]
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl From<Q15> for f32 {
    #[inline(always)]
    fn from(v: Q15) -> Self {
        v.to_f32()
    }
}

/// Strongly-typed Q1.31 fixed-point number stored in an `i32`.
///
/// Range is `[-1.0, 0.9999999995343387]`.
/// Provides operator overloading (`+`, `-`, `*`, `/`) with saturating fixed-point arithmetic.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Q31(pub i32);

impl Q31 {
    /// Zero representation (`0.0`).
    pub const ZERO: Self = Self(0);
    /// Maximum positive value (`~1.0` = `i32::MAX`).
    pub const ONE: Self = Self(i32::MAX);
    /// Minimum representable value (`-1.0` = `i32::MIN`).
    pub const MIN: Self = Self(i32::MIN);
    /// Maximum representable value (`~1.0` = `i32::MAX`).
    pub const MAX: Self = Self(i32::MAX);
    /// Smallest positive step (`1 / 2^31`).
    pub const DELTA: Self = Self(1);

    /// Create from raw integer bits.
    #[inline(always)]
    pub const fn from_bits(bits: i32) -> Self {
        Self(bits)
    }

    /// Retrieve raw integer bits.
    #[inline(always)]
    pub const fn to_bits(self) -> i32 {
        self.0
    }

    /// Convert from `f32` with rounding and saturation.
    #[inline(always)]
    pub fn from_f32(v: f32) -> Self {
        let scaled = v as f64 * 2147483648.0 + if v >= 0.0 { 0.5 } else { -0.5 };
        Self(scaled.clamp(-2147483648.0, 2147483647.0) as i32)
    }

    /// Convert to `f32`.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        (self.0 as f64 / 2147483648.0) as f32
    }

    /// Convert from `f64` with rounding and saturation.
    #[inline(always)]
    pub fn from_f64(v: f64) -> Self {
        let scaled = v * 2147483648.0 + if v >= 0.0 { 0.5 } else { -0.5 };
        Self(scaled.clamp(-2147483648.0, 2147483647.0) as i32)
    }

    /// Convert to `f64`.
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / 2147483648.0
    }

    /// Saturating addition.
    #[inline(always)]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction.
    #[inline(always)]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Saturating Q31 multiplication.
    #[inline(always)]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        Self(q31_mult(self.0, rhs.0))
    }

    /// Saturating Q31 division with division-by-zero protection.
    #[inline(always)]
    pub fn saturating_div(self, rhs: Self) -> Self {
        if rhs.0 == 0 {
            if self.0 >= 0 { Self::MAX } else { Self::MIN }
        } else {
            let num = (self.0 as i64) << 31;
            let res = num / (rhs.0 as i64);
            Self(res.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
        }
    }

    /// Absolute value (saturating at `Self::MAX` for `-1.0`).
    #[inline(always)]
    pub const fn abs(self) -> Self {
        if self.0 == i32::MIN {
            Self::MAX
        } else {
            Self(self.0.abs())
        }
    }

    /// Clamp to `[min, max]`.
    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }
}

impl core::ops::Add for Q31 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }
}

impl core::ops::Sub for Q31 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }
}

impl core::ops::Mul for Q31 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.saturating_mul(rhs)
    }
}

impl core::ops::Div for Q31 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        self.saturating_div(rhs)
    }
}

impl core::ops::Neg for Q31 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        if self.0 == i32::MIN {
            Self::MAX
        } else {
            Self(-self.0)
        }
    }
}

impl core::ops::AddAssign for Q31 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for Q31 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for Q31 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::DivAssign for Q31 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl From<i32> for Q31 {
    #[inline(always)]
    fn from(v: i32) -> Self {
        Self(v)
    }
}

impl From<Q31> for i32 {
    #[inline(always)]
    fn from(v: Q31) -> Self {
        v.0
    }
}

impl From<f32> for Q31 {
    #[inline(always)]
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl From<Q31> for f32 {
    #[inline(always)]
    fn from(v: Q31) -> Self {
        v.to_f32()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified DSP Sample Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Unified numerical sample trait implemented for both floating-point and fixed-point numbers.
///
/// Enables writing generic filters, delay lines, oscillators, and processing blocks that operate
/// seamlessly with `f32`, `f64`, `Q15`, and `Q31`.
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

impl DspSample for Q15 {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

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
        self.saturating_div(rhs)
    }

    #[inline(always)]
    fn abs_val(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::from_f32(val)
    }
}

impl DspSample for Q31 {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

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
        self.saturating_div(rhs)
    }

    #[inline(always)]
    fn abs_val(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::from_f32(val)
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
