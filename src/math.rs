//! Unified math abstraction providing floating-point mathematical operations using `libm` in `no_std` environments or `std` math when available.

/// Floating-point math trait providing `sin`, `cos`, `sqrt`, `ln`, `exp`, `atan2`, `powf`, `tanh`, `abs`.
pub trait FloatMath: Sized {
    fn abs(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn tanh(self) -> Self;
}

impl FloatMath for f32 {
    #[inline(always)]
    fn abs(self) -> f32 {
        #[cfg(feature = "std")]
        {
            self.abs()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::fabsf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            f32::from_bits(self.to_bits() & 0x7FFF_FFFF)
        }
    }

    #[inline(always)]
    fn sin(self) -> f32 {
        #[cfg(feature = "std")]
        {
            self.sin()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::sinf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!("Either feature 'std' or feature 'libm' must be enabled for floating-point trig functions.");
        }
    }

    #[inline(always)]
    fn cos(self) -> f32 {
        #[cfg(feature = "std")]
        {
            self.cos()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::cosf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!("Either feature 'std' or feature 'libm' must be enabled for floating-point trig functions.");
        }
    }

    #[inline(always)]
    fn sqrt(self) -> f32 {
        #[cfg(feature = "std")]
        {
            self.sqrt()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::sqrtf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point sqrt."
            );
        }
    }

    #[inline(always)]
    fn ln(self) -> f32 {
        #[cfg(feature = "std")]
        {
            self.ln()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::logf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point log."
            );
        }
    }

    #[inline(always)]
    fn exp(self) -> f32 {
        #[cfg(feature = "std")]
        {
            self.exp()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::expf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point exp."
            );
        }
    }

    #[inline(always)]
    fn atan2(self, x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            self.atan2(x)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::atan2f(self, x)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point atan2."
            );
        }
    }

    #[inline(always)]
    fn powf(self, n: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            self.powf(n)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::powf(self, n)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point powf."
            );
        }
    }

    #[inline(always)]
    fn tanh(self) -> f32 {
        #[cfg(feature = "std")]
        {
            self.tanh()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::tanhf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point tanh."
            );
        }
    }
}

impl FloatMath for f64 {
    #[inline(always)]
    fn abs(self) -> f64 {
        #[cfg(feature = "std")]
        {
            self.abs()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::fabs(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            f64::from_bits(self.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
        }
    }

    #[inline(always)]
    fn sin(self) -> f64 {
        #[cfg(feature = "std")]
        {
            self.sin()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::sin(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!("Either feature 'std' or feature 'libm' must be enabled for floating-point trig functions.");
        }
    }

    #[inline(always)]
    fn cos(self) -> f64 {
        #[cfg(feature = "std")]
        {
            self.cos()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::cos(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!("Either feature 'std' or feature 'libm' must be enabled for floating-point trig functions.");
        }
    }

    #[inline(always)]
    fn sqrt(self) -> f64 {
        #[cfg(feature = "std")]
        {
            self.sqrt()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::sqrt(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point sqrt."
            );
        }
    }

    #[inline(always)]
    fn ln(self) -> f64 {
        #[cfg(feature = "std")]
        {
            self.ln()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::log(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point log."
            );
        }
    }

    #[inline(always)]
    fn exp(self) -> f64 {
        #[cfg(feature = "std")]
        {
            self.exp()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::exp(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point exp."
            );
        }
    }

    #[inline(always)]
    fn atan2(self, x: f64) -> f64 {
        #[cfg(feature = "std")]
        {
            self.atan2(x)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::atan2(self, x)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point atan2."
            );
        }
    }

    #[inline(always)]
    fn powf(self, n: f64) -> f64 {
        #[cfg(feature = "std")]
        {
            self.powf(n)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::pow(self, n)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point powf."
            );
        }
    }

    #[inline(always)]
    fn tanh(self) -> f64 {
        #[cfg(feature = "std")]
        {
            self.tanh()
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::tanh(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm")))]
        {
            core::compile_error!(
                "Either feature 'std' or feature 'libm' must be enabled for floating-point tanh."
            );
        }
    }
}
