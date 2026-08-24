#![no_std]
#![allow(missing_docs)]

//! # embedded-dsp
//!
//! A `#![no_std]` Rust digital signal processing library for microcontrollers, embedded systems, and real-time signal processing applications.
//!
//! Algorithm modules are Cargo-feature gated (all enabled by the `full` feature, which is
//! in `default`). `types` and `math` are always available. See the crate's `[features]`
//! table for the per-module flags.
//!
//! ## Overview
//!
//! `embedded-dsp` provides zero-allocation digital signal processing algorithms:
//!
//! - **Basic Math**: Elementwise addition, subtraction, multiplication, dot product, scale, shift, clip, logic ops.
//! - **Complex Math**: Complex vector addition, multiplication, magnitude, magnitude squared, conjugate, dot product.
//! - **Fast Math**: Trigonometric (sin, cos, atan2), square root, division, log, exp.
//! - **Fixed-Point (Q16.16)**: Saturating Q16.16 arithmetic and scanline interpolation (`fixed-point` feature).
//! - **LUT trigonometry**: Compile-time 256-entry sin/cos tables (`lut` feature).
//! - **Statistics**: Mean, variance, standard deviation, RMS, power, min/max, entropy, Kullback-Leibler, LogSumExp.
//! - **Support**: Vector copy, fill, type conversions (Q7, Q15, Q31, F32), sort, barycenter, weighted sum.
//! - **Matrix**: Matrix addition, subtraction, multiplication, scale, transpose, Gauss-Jordan inverse.
//! - **Filtering**: FIR, Biquad IIR Direct Form I, LMS adaptive filters, Convolution, Correlation, single-pole recursive filters, and O(1) recursive moving average.
//! - **Filter Design**: Biquad Lowpass, Highpass, Bandpass, Notch, Peaking EQ, Allpass, Butterworth, Chebyshev, and arbitrary-response (frequency-sampling) design.
//! - **Filter Analysis**: Frequency response (DTFT) evaluation for FIR/biquad filters, FIR group delay, and pole-based IIR stability checks.
//! - **Resampling & Multi-rate**: CIC Decimator & Interpolator, linear fractional resampler.
//! - **Kalman Filtering**: 1D/2D helpers, const-generic linear `KalmanFilter<N, M>`, and trait-based `ExtendedKalmanFilter` (EKF).
//! - **Const Generics**: Compile-time fixed-size `FirFilter<N>`, `BiquadCascade<N>`, and `Matrix<R, C>`.
//! - **Transform**: In-place Complex FFT (CFFT), Real FFT (RFFT), DCT-IV, Bit reversal, Fixed-point FFT (Q15/Q31), Haar transform, Hartley transform, and a generalized wavelet transform (Daubechies-4).
//! - **Companding**: µ-law and A-law audio companding (ITU-T G.711 family).
//! - **Audio**: Goertzel single-frequency detector, peak/RMS envelope followers, Mel filterbank, and MFCC feature extraction.
//! - **Spectral Analysis & PSD**: Welch's method power spectral density estimation (averaged periodograms), single-segment periodograms in linear and dB scale.
//! - **Spatial & 2D Signal Processing**: 2D DCT/IDCT, 2D Convolution, 2D Non-linear Filtering (Min/Max/Median), Sobel edge detection, 2D Histogram, MSE, PSNR.
//! - **Controller**: PID motor controller, Clarke transform, Park transform, Inverse Clarke/Park.
//! - **Interpolation**: Linear, Bilinear, Cubic spline interpolation.
//! - **Quaternion**: Norm, normalization, product, conjugate, inverse, rotation matrix conversion.
//! - **Window**: Hanning, Hamming, Blackman, Blackman-Harris, Bartlett, Welch, Flat-top window generators.
//! - **Distance**: Euclidean, Cosine, Chebyshev, Manhattan, Minkowski, Jaccard, Hamming, Canberra, Bray-Curtis.

#[cfg(feature = "std")]
extern crate std;

macro_rules! gated_mod {
    ($feature:literal, $module:ident) => {
        #[cfg(feature = $feature)]
        pub mod $module;
        #[cfg(feature = $feature)]
        pub use $module::*;
    };
    (math $feature:literal, $module:ident) => {
        #[cfg(all(feature = $feature, any(feature = "std", feature = "libm")))]
        pub mod $module;
        #[cfg(all(feature = $feature, any(feature = "std", feature = "libm")))]
        pub use $module::*;
    };
}

gated_mod!(math "audio", audio);
gated_mod!("basic-math", basic_math);
gated_mod!("companding", companding);
gated_mod!("complex-math", complex_math);
gated_mod!("const-generics", const_generics);
gated_mod!("controller", controller);
gated_mod!("distance", distance);
gated_mod!("fast-math", fast_math);
gated_mod!(math "filter-analysis", filter_analysis);
gated_mod!(math "filter-design", filter_design);
gated_mod!("filtering", filtering);
gated_mod!("fixed-point", fixed_point);
gated_mod!("interpolation", interpolation);
gated_mod!("kalman", kalman);
gated_mod!("lut", lut);
pub mod math;
gated_mod!("matrix", matrix);
gated_mod!("psd", psd);
gated_mod!("quaternion", quaternion);
gated_mod!("resampling", resampling);
gated_mod!("spatial", spatial);
gated_mod!("statistics", statistics);
gated_mod!("support", support);
gated_mod!("transform", transform);
pub mod types;
gated_mod!("window", window);

pub use math::*;
pub use types::*;
