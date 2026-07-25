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
//! - **Filtering**: FIR, Biquad IIR Direct Form I, LMS adaptive filters, Convolution, Correlation.
//! - **Transform**: In-place Complex FFT (CFFT), Real FFT (RFFT), DCT-IV, Bit reversal.
//! - **Controller**: PID motor controller, Clarke transform, Park transform, Inverse Clarke/Park.
//! - **Interpolation**: Linear, Bilinear, Cubic spline interpolation.
//! - **Quaternion**: Norm, normalization, product, conjugate, inverse, rotation matrix conversion.
//! - **Window**: Hanning, Hamming, Blackman, Bartlett, Welch, Flat-top window generators.
//! - **Distance**: Euclidean, Cosine, Chebyshev, Manhattan, Minkowski, Jaccard, Hamming, Canberra, Bray-Curtis.
//! - **Bayes**: Gaussian Naive Bayes classifier.
//! - **SVM**: Support Vector Machine classifier with Linear, Polynomial, RBF, and Sigmoid kernels.

#[cfg(feature = "std")]
extern crate std;

pub mod basic_math;
pub mod bayes;
pub mod complex_math;
pub mod controller;
pub mod distance;
pub mod fast_math;
pub mod filtering;
pub mod interpolation;
pub mod math;
pub mod matrix;
pub mod quaternion;
pub mod statistics;
pub mod support;
pub mod svm;
pub mod transform;
pub mod types;
pub mod window;

pub use basic_math::*;
pub use bayes::*;
pub use complex_math::*;
pub use controller::*;
pub use distance::*;
pub use fast_math::*;
pub use filtering::*;
pub use interpolation::*;
pub use math::*;
pub use matrix::*;
pub use quaternion::*;
pub use statistics::*;
pub use support::*;
pub use svm::*;
pub use transform::*;
pub use types::*;
pub use window::*;
