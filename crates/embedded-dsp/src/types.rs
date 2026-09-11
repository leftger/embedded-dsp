//! Data types, status codes, complex structures, and fixed-point helper types.

#[cfg(feature = "fixed")]
pub use fixed::types::{I1F7 as q7, I1F15 as q15, I1F31 as q31, I16F16};

#[cfg(feature = "fixed")]
/// Q8.7 fixed-point type (8 integer bits, 7 fractional bits, `i16`-backed).
pub type Q8F7 = fixed::FixedI16<fixed::types::extra::U7>;

#[cfg(feature = "fixed")]
/// Q2.14 fixed-point type (2 integer bits, 14 fractional bits, `i16`-backed).
pub type Q2F14 = fixed::FixedI16<fixed::types::extra::U14>;

#[cfg(not(feature = "fixed"))]
pub use fallback::*;

#[cfg(not(feature = "fixed"))]
mod fallback {
    pub trait FixedNum: Copy {
        fn to_raw_fixed(self, frac: u32, min_val: i64, max_val: i64, saturate: bool) -> i64;
        fn from_raw_fixed(raw: i64, frac: u32) -> Self;
    }

    impl FixedNum for f32 {
        #[inline(always)]
        fn to_raw_fixed(self, frac: u32, min_val: i64, max_val: i64, saturate: bool) -> i64 {
            (self as f64).to_raw_fixed(frac, min_val, max_val, saturate)
        }
        #[inline(always)]
        fn from_raw_fixed(raw: i64, frac: u32) -> Self {
            f64::from_raw_fixed(raw, frac) as f32
        }
    }

    impl FixedNum for f64 {
        #[inline(always)]
        fn to_raw_fixed(self, frac: u32, min_val: i64, max_val: i64, saturate: bool) -> i64 {
            if self.is_nan() {
                return 0;
            }
            let scale = (1u64 << frac) as f64;
            let scaled = self * scale;
            if saturate {
                if scaled >= max_val as f64 {
                    return max_val;
                }
                if scaled <= min_val as f64 {
                    return min_val;
                }
            }
            let sign = if scaled < 0.0 { -1i64 } else { 1i64 };
            let abs_val = if scaled < 0.0 { -scaled } else { scaled };
            let abs_int = abs_val as i64;
            let frac_part = abs_val - (abs_int as f64);
            let rounded_abs = if frac_part > 0.5 {
                abs_int + 1
            } else if frac_part < 0.5 {
                abs_int
            } else {
                if (abs_int & 1) != 0 {
                    abs_int + 1
                } else {
                    abs_int
                }
            };
            let res = sign.wrapping_mul(rounded_abs);
            if saturate {
                res.clamp(min_val, max_val)
            } else {
                res
            }
        }

        #[inline(always)]
        fn from_raw_fixed(raw: i64, frac: u32) -> Self {
            (raw as f64) / ((1u64 << frac) as f64)
        }
    }

    macro_rules! impl_fixed_num_int {
        ($($t:ty),*) => {
            $(
                impl FixedNum for $t {
                    #[inline(always)]
                    fn to_raw_fixed(self, frac: u32, min_val: i64, max_val: i64, saturate: bool) -> i64 {
                        let shifted = (self as i64).wrapping_shl(frac);
                        if saturate {
                            shifted.clamp(min_val, max_val)
                        } else {
                            shifted
                        }
                    }
                    #[inline(always)]
                    fn from_raw_fixed(raw: i64, frac: u32) -> Self {
                        (raw >> frac) as $t
                    }
                }
            )*
        };
    }
    impl_fixed_num_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

    macro_rules! define_fallback_type {
        ($name:ident, $raw:ident, $wide:ident, $frac:expr) => {
            #[allow(non_camel_case_types)]
            #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
            #[repr(transparent)]
            #[cfg_attr(feature = "defmt", derive(defmt::Format))]
            #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
            #[cfg_attr(feature = "serde", serde(transparent))]
            pub struct $name(pub $raw);

            impl $name {
                pub const ZERO: Self = Self(0);
                pub const MAX: Self = Self($raw::MAX);
                pub const MIN: Self = Self($raw::MIN);

                #[inline(always)]
                pub const fn from_bits(bits: $raw) -> Self {
                    Self(bits)
                }

                #[inline(always)]
                pub const fn to_bits(self) -> $raw {
                    self.0
                }

                #[inline(always)]
                pub fn from_num<T: FixedNum>(v: T) -> Self {
                    Self(
                        T::to_raw_fixed(v, $frac, $raw::MIN as i64, $raw::MAX as i64, false)
                            as $raw,
                    )
                }

                #[inline(always)]
                pub fn saturating_from_num<T: FixedNum>(v: T) -> Self {
                    Self(
                        T::to_raw_fixed(v, $frac, $raw::MIN as i64, $raw::MAX as i64, true) as $raw,
                    )
                }

                #[inline(always)]
                pub fn to_num<T: FixedNum>(self) -> T {
                    T::from_raw_fixed(self.0 as i64, $frac)
                }

                #[inline(always)]
                pub const fn saturating_add(self, rhs: Self) -> Self {
                    Self(self.0.saturating_add(rhs.0))
                }

                #[inline(always)]
                pub const fn saturating_sub(self, rhs: Self) -> Self {
                    Self(self.0.saturating_sub(rhs.0))
                }

                #[inline(always)]
                pub fn saturating_mul(self, rhs: Self) -> Self {
                    let prod = ((self.0 as $wide) * (rhs.0 as $wide)) >> $frac;
                    Self(prod.clamp($raw::MIN as $wide, $raw::MAX as $wide) as $raw)
                }

                #[inline(always)]
                pub const fn saturating_neg(self) -> Self {
                    Self(self.0.saturating_neg())
                }

                #[inline(always)]
                pub const fn saturating_abs(self) -> Self {
                    Self(self.0.saturating_abs())
                }

                #[inline(always)]
                pub const fn abs(self) -> Self {
                    Self(self.0.wrapping_abs())
                }

                #[inline(always)]
                pub const fn wrapping_add(self, rhs: Self) -> Self {
                    Self(self.0.wrapping_add(rhs.0))
                }

                #[inline(always)]
                pub const fn wrapping_sub(self, rhs: Self) -> Self {
                    Self(self.0.wrapping_sub(rhs.0))
                }

                #[inline(always)]
                pub const fn wrapping_neg(self) -> Self {
                    Self(self.0.wrapping_neg())
                }

                #[inline(always)]
                pub fn wrapping_mul(self, rhs: Self) -> Self {
                    let prod = (self.0 as $wide).wrapping_mul(rhs.0 as $wide);
                    Self((prod >> $frac) as $raw)
                }

                #[inline(always)]
                pub const fn wrapping_mul_int(self, n: i32) -> Self {
                    Self(self.0.wrapping_mul(n as $raw))
                }

                #[inline(always)]
                pub fn wrapping_div(self, rhs: Self) -> Self {
                    if rhs.0 == 0 {
                        return Self(0);
                    }
                    let a = (self.0 as $wide) << $frac;
                    let b = rhs.0 as $wide;
                    Self((a / b) as $raw)
                }

                #[inline(always)]
                pub fn wrapping_div_int(self, n: i32) -> Self {
                    if n == 0 {
                        return Self(0);
                    }
                    Self(self.0.wrapping_div(n as $raw))
                }

                #[inline(always)]
                pub fn recip(self) -> Self {
                    if self.0 == 0 {
                        return Self::MAX;
                    }
                    let one_shifted = 1i64 << (2 * $frac);
                    Self((one_shifted / (self.0 as i64)) as $raw)
                }

                #[inline(always)]
                pub fn checked_div(self, rhs: Self) -> Option<Self> {
                    if rhs.0 == 0 {
                        return None;
                    }
                    let a = (self.0 as $wide) << $frac;
                    let b = rhs.0 as $wide;
                    let res = a / b;
                    if res > $raw::MAX as $wide || res < $raw::MIN as $wide {
                        None
                    } else {
                        Some(Self(res as $raw))
                    }
                }
            }

            impl FixedNum for $name {
                #[inline(always)]
                fn to_raw_fixed(
                    self,
                    frac: u32,
                    min_val: i64,
                    max_val: i64,
                    saturate: bool,
                ) -> i64 {
                    let shifted = if frac >= $frac {
                        (self.0 as i64) << (frac - $frac)
                    } else {
                        (self.0 as i64) >> ($frac - frac)
                    };
                    if saturate {
                        shifted.clamp(min_val, max_val)
                    } else {
                        shifted
                    }
                }

                #[inline(always)]
                fn from_raw_fixed(raw: i64, frac: u32) -> Self {
                    let bits = if $frac >= frac {
                        (raw << ($frac - frac)) as $raw
                    } else {
                        (raw >> (frac - $frac)) as $raw
                    };
                    Self(bits)
                }
            }

            impl core::ops::Add for $name {
                type Output = Self;
                #[inline(always)]
                fn add(self, rhs: Self) -> Self {
                    Self(self.0.wrapping_add(rhs.0))
                }
            }

            impl core::ops::AddAssign for $name {
                #[inline(always)]
                fn add_assign(&mut self, rhs: Self) {
                    self.0 = self.0.wrapping_add(rhs.0);
                }
            }

            impl core::ops::Sub for $name {
                type Output = Self;
                #[inline(always)]
                fn sub(self, rhs: Self) -> Self {
                    Self(self.0.wrapping_sub(rhs.0))
                }
            }

            impl core::ops::SubAssign for $name {
                #[inline(always)]
                fn sub_assign(&mut self, rhs: Self) {
                    self.0 = self.0.wrapping_sub(rhs.0);
                }
            }

            impl core::ops::Mul for $name {
                type Output = Self;
                #[inline(always)]
                fn mul(self, rhs: Self) -> Self {
                    let prod = (self.0 as $wide) * (rhs.0 as $wide);
                    Self((prod >> $frac) as $raw)
                }
            }

            impl core::ops::MulAssign for $name {
                #[inline(always)]
                fn mul_assign(&mut self, rhs: Self) {
                    *self = *self * rhs;
                }
            }

            impl core::ops::Neg for $name {
                type Output = Self;
                #[inline(always)]
                fn neg(self) -> Self {
                    Self(self.0.wrapping_neg())
                }
            }

            impl core::fmt::Display for $name {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    write!(f, "{}", self.0)
                }
            }

            macro_rules! impl_cmp_int {
                ($int:ident) => {
                    impl PartialEq<$int> for $name {
                        #[inline(always)]
                        fn eq(&self, other: &$int) -> bool {
                            let other_fixed = (*other as i64) << $frac;
                            (self.0 as i64) == other_fixed
                        }
                    }
                    impl PartialOrd<$int> for $name {
                        #[inline(always)]
                        fn partial_cmp(&self, other: &$int) -> Option<core::cmp::Ordering> {
                            let other_fixed = (*other as i64) << $frac;
                            (self.0 as i64).partial_cmp(&other_fixed)
                        }
                    }
                    impl PartialEq<$name> for $int {
                        #[inline(always)]
                        fn eq(&self, other: &$name) -> bool {
                            other.eq(self)
                        }
                    }
                    impl PartialOrd<$name> for $int {
                        #[inline(always)]
                        fn partial_cmp(&self, other: &$name) -> Option<core::cmp::Ordering> {
                            let self_fixed = (*self as i64) << $frac;
                            self_fixed.partial_cmp(&(other.0 as i64))
                        }
                    }
                };
            }
            impl_cmp_int!(i8);
            impl_cmp_int!(i16);
            impl_cmp_int!(i32);
            impl_cmp_int!(i64);
            impl_cmp_int!(isize);
            impl_cmp_int!(u8);
            impl_cmp_int!(u16);
            impl_cmp_int!(u32);
            impl_cmp_int!(u64);
            impl_cmp_int!(usize);
        };
    }

    define_fallback_type!(q7, i8, i32, 7);
    define_fallback_type!(q15, i16, i32, 15);
    define_fallback_type!(q31, i32, i64, 31);
    define_fallback_type!(Q8F7, i16, i32, 7);
    define_fallback_type!(Q2F14, i16, i32, 14);
    define_fallback_type!(I16F16, i32, i64, 16);
}

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
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable))]
pub struct Complex<T> {
    pub real: T,
    pub imag: T,
}

#[cfg(feature = "bytemuck")]
#[allow(unsafe_code)]
unsafe impl<T: bytemuck::Pod> bytemuck::Pod for Complex<T> {}

impl<T> Complex<T> {
    #[inline(always)]
    pub const fn new(real: T, imag: T) -> Self {
        Self { real, imag }
    }
}

impl<T: Copy> Complex<T> {
    /// Returns the real component.
    #[inline(always)]
    pub const fn re(&self) -> T {
        self.real
    }

    /// Returns the imaginary component.
    #[inline(always)]
    pub const fn im(&self) -> T {
        self.imag
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

/// Helper function for saturating division in Q15 format.
#[inline(always)]
fn saturating_div_q15(a: q15, b: q15) -> q15 {
    if b == q15::ZERO {
        if a >= q15::ZERO { q15::MAX } else { q15::MIN }
    } else {
        match a.checked_div(b) {
            Some(v) => v,
            None => {
                if (a >= q15::ZERO) == (b >= q15::ZERO) {
                    q15::MAX
                } else {
                    q15::MIN
                }
            }
        }
    }
}

/// Helper function for saturating division in Q31 format.
#[inline(always)]
fn saturating_div_q31(a: q31, b: q31) -> q31 {
    if b == q31::ZERO {
        if a >= q31::ZERO { q31::MAX } else { q31::MIN }
    } else {
        match a.checked_div(b) {
            Some(v) => v,
            None => {
                if (a >= q31::ZERO) == (b >= q31::ZERO) {
                    q31::MAX
                } else {
                    q31::MIN
                }
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
        saturating_div_q15(self, rhs)
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
        saturating_div_q31(self, rhs)
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

// --- Brain Floating Point (BFloat16) ---

/// 16-bit Brain Floating Point (`bfloat16`) format.
///
/// Composed of 1 sign bit, 8 exponent bits, and 7 fraction bits (matching the upper 16 bits of IEEE-754 `f32`).
/// Provides the full dynamic range of single-precision `f32` with a 50% SRAM footprint, ideal for
/// microcontroller circular delay lines, reverb buffers, and Edge AI inference feature stores.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct BFloat16(pub u16);

impl core::fmt::Debug for BFloat16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "BFloat16({:?})", self.to_f32())
    }
}

impl core::fmt::Display for BFloat16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl BFloat16 {
    pub const ZERO: Self = Self(0);
    pub const NEG_ZERO: Self = Self(0x8000);
    pub const ONE: Self = Self(0x3F80);
    pub const NEG_ONE: Self = Self(0xBF80);
    pub const NAN: Self = Self(0x7FC0);
    pub const INFINITY: Self = Self(0x7F80);
    pub const NEG_INFINITY: Self = Self(0xFF80);
    pub const MAX: Self = Self(0x7F7F);
    pub const MIN: Self = Self(0xFF7F);
    pub const MIN_POSITIVE: Self = Self(0x0080);

    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts an `f32` into `bfloat16` using Round-to-Nearest-Even (RNE) with magic rounding bias.
    /// Preserves quiet NaNs and handles subnormals correctly.
    #[inline]
    pub fn from_f32(f: f32) -> Self {
        let bits = f.to_bits();
        // NaN check: exp = 0xFF and mantissa != 0
        if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
            // Force a quiet NaN bit (bit 6) in the upper 16 bits
            return Self(((bits >> 16) | 0x0040) as u16);
        }
        // Round-to-nearest, ties to even (RNE):
        let lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(0x7FFF + lsb);
        Self((rounded >> 16) as u16)
    }

    /// Unpacks `bfloat16` into standard IEEE-754 `f32` in a single shift.
    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7F80) == 0x7F80 && (self.0 & 0x007F) != 0
    }

    #[inline]
    pub fn is_infinite(self) -> bool {
        (self.0 & 0x7FFF) == 0x7F80
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        (self.0 & 0x7F80) != 0x7F80
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        (self.0 & 0x7FFF) == 0
    }

    #[inline]
    pub fn is_sign_positive(self) -> bool {
        (self.0 & 0x8000) == 0
    }

    #[inline]
    pub fn is_sign_negative(self) -> bool {
        (self.0 & 0x8000) != 0
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self(self.0 & 0x7FFF)
    }
}

impl From<f32> for BFloat16 {
    #[inline]
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl From<BFloat16> for f32 {
    #[inline]
    fn from(v: BFloat16) -> Self {
        v.to_f32()
    }
}

// --- Double-Single (FloatFloat) Extended Precision ---

/// Double-Single extended precision number represented as an unevaluated sum `hi + lo`.
///
/// Implements Bailey's QD / Universal's `dd` algorithms scaled to single precision.
/// Delivers ~48 bits of effective precision (approaching IEEE-754 `f64`'s 53 bits)
/// while executing purely on hardware single-precision `f32` FPUs without software `f64` emulation.
/// Ideal for high-Q resonant IIR biquads, Kalman filters, and integrator loops prone to numerical instability.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct FloatFloat {
    pub hi: f32,
    pub lo: f32,
}

impl core::fmt::Debug for FloatFloat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FloatFloat({:?} + {:?})", self.hi, self.lo)
    }
}

impl core::fmt::Display for FloatFloat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f64())
    }
}

impl FloatFloat {
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    #[inline]
    pub const fn new(hi: f32, lo: f32) -> Self {
        Self { hi, lo }
    }

    #[inline]
    pub fn from_f32(val: f32) -> Self {
        Self { hi: val, lo: 0.0 }
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        self.hi + self.lo
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        (self.hi as f64) + (self.lo as f64)
    }

    #[inline]
    pub fn abs(self) -> Self {
        if self.hi < 0.0 {
            Self {
                hi: -self.hi,
                lo: -self.lo,
            }
        } else {
            self
        }
    }

    /// Addition of two FloatFloat values using TwoSum error-free transformations.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, rhs: Self) -> Self {
        let s = self.hi + rhs.hi;
        let v = s - self.hi;
        let e = (self.hi - (s - v)) + (rhs.hi - v);

        let e2 = e + self.lo + rhs.lo;
        let sum_hi = s + e2;
        let sum_lo = e2 - (sum_hi - s);
        Self {
            hi: sum_hi,
            lo: sum_lo,
        }
    }

    /// Subtraction of two FloatFloat values.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, rhs: Self) -> Self {
        self.add(Self {
            hi: -rhs.hi,
            lo: -rhs.lo,
        })
    }

    /// Multiplication of two FloatFloat values using TwoProd and TwoSum.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Self) -> Self {
        let p = self.hi * rhs.hi;
        let c = 4097.0f32 * self.hi;
        let a_hi = c - (c - self.hi);
        let a_lo = self.hi - a_hi;
        let c = 4097.0f32 * rhs.hi;
        let b_hi = c - (c - rhs.hi);
        let b_lo = rhs.hi - b_hi;
        let err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;

        let err2 = err + (self.hi * rhs.lo + self.lo * rhs.hi);
        let prod_hi = p + err2;
        let prod_lo = err2 - (prod_hi - p);
        Self {
            hi: prod_hi,
            lo: prod_lo,
        }
    }

    /// Division of two FloatFloat values.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, rhs: Self) -> Self {
        let q1 = self.hi / rhs.hi;
        let r = self.sub(rhs.mul(Self::from_f32(q1)));
        let q2 = r.hi / rhs.hi;
        let div_hi = q1 + q2;
        let div_lo = q2 - (div_hi - q1);
        Self {
            hi: div_hi,
            lo: div_lo,
        }
    }
}

impl core::ops::Add for FloatFloat {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl core::ops::Sub for FloatFloat {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.sub(rhs)
    }
}

impl core::ops::Neg for FloatFloat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl core::ops::Mul for FloatFloat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul(rhs)
    }
}

impl core::ops::Div for FloatFloat {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.div(rhs)
    }
}
