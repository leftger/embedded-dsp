#![no_std]
#![allow(missing_docs)]

//! # embedded-dsp
//!
//! A `#![no_std]` Rust digital signal processing library for microcontrollers, embedded systems, and real-time signal processing applications.
//!
//! ## Overview
//!
//! `embedded-dsp` provides zero-allocation digital signal processing algorithms:
//!
//! - **Basic Math**: Elementwise addition, subtraction, multiplication, dot product, scale, shift, clip, logic ops.
//! - **Complex Math**: Complex vector addition, multiplication, magnitude, magnitude squared, conjugate, dot product.
//! - **Fast Math**: Trigonometric (sin, cos, atan2), square root, division, log, exp.
//! - **Statistics**: Mean, variance, standard deviation, RMS, power, min/max, entropy, Kullback-Leibler, LogSumExp.
//! - **Support**: Vector copy, fill, type conversions (Q7, Q15, Q31, F32), sort, barycenter, weighted sum.
//! - **Matrix**: Matrix addition, subtraction, multiplication, scale, transpose, Gauss-Jordan inverse.
//! - **Filtering**: FIR, Biquad IIR Direct Form I, LMS adaptive filters, Convolution, Correlation, single-pole recursive filters, and O(1) recursive moving average.
//! - **Filter Design**: Biquad Lowpass, Highpass, Bandpass, Notch, Peaking EQ, Allpass, Butterworth, and Chebyshev design.
//! - **Filter Analysis**: Frequency response (DTFT) evaluation for FIR/biquad filters, FIR group delay, and pole-based IIR stability checks.
//! - **Resampling & Multi-rate**: CIC Decimator & Interpolator, linear fractional resampler.
//! - **Kalman Filtering**: 1D/2D helpers, const-generic linear `KalmanFilter<N, M>`, and trait-based `ExtendedKalmanFilter` (EKF).
//! - **Const Generics**: Compile-time fixed-size `FirFilter<N>`, `BiquadCascade<N>`, and `Matrix<R, C>`.
//! - **Transform**: In-place Complex FFT (CFFT), Real FFT (RFFT), DCT-IV, Bit reversal, Fixed-point FFT (Q15/Q31).
//! - **Spectral Analysis & PSD**: Welch's method power spectral density estimation (averaged periodograms), single-segment periodograms in linear and dB scale.
//! - **Spatial & 2D Signal Processing**: 2D DCT/IDCT, 2D Convolution, 2D Non-linear Filtering (Min/Max/Median), Sobel edge detection, 2D Histogram, MSE, PSNR.
//! - **Controller**: PID motor controller, Clarke transform, Park transform, Inverse Clarke/Park.
//! - **Interpolation**: Linear, Bilinear, Cubic spline interpolation.
//! - **Quaternion**: Norm, normalization, product, conjugate, inverse, rotation matrix conversion.
//! - **Window**: Hanning, Hamming, Blackman, Blackman-Harris, Bartlett, Welch, Flat-top window generators.
//! - **Distance**: Euclidean, Cosine, Chebyshev, Manhattan, Minkowski, Jaccard, Hamming, Canberra, Bray-Curtis.

#[cfg(feature = "std")]
extern crate std;

pub mod basic_math;
pub mod complex_math;
pub mod const_generics;
pub mod controller;
pub mod distance;
pub mod fast_math;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod filter_analysis;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod filter_design;
pub mod filtering;
pub mod interpolation;
pub mod kalman;
pub mod math;
pub mod matrix;
pub mod psd;
pub mod quaternion;
pub mod resampling;
pub mod spatial;
pub mod statistics;
pub mod support;
pub mod transform;
pub mod types;
pub mod window;

pub use basic_math::*;
pub use complex_math::*;
pub use const_generics::*;
pub use controller::*;
pub use distance::*;
pub use fast_math::*;
#[cfg(any(feature = "std", feature = "libm"))]
pub use filter_analysis::*;
#[cfg(any(feature = "std", feature = "libm"))]
pub use filter_design::*;
pub use filtering::*;
pub use interpolation::*;
pub use kalman::*;
pub use math::*;
pub use matrix::*;
pub use psd::*;
pub use quaternion::*;
pub use resampling::*;
pub use spatial::*;
pub use statistics::*;
pub use support::*;
pub use transform::*;
pub use types::*;
pub use window::*;
