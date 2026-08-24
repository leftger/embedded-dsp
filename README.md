# embedded-dsp

<p align="center">
  <img src="assets/aztec_rustacean.png" alt="embedded-dsp" width="100%">
</p>

[![crates.io](https://img.shields.io/crates/v/embedded-dsp.svg)](https://crates.io/crates/embedded-dsp)
[![docs.rs](https://img.shields.io/docsrs/embedded-dsp)](https://docs.rs/embedded-dsp)
[![CI](https://github.com/leftger/embedded-dsp/actions/workflows/ci.yml/badge.svg)](https://github.com/leftger/embedded-dsp/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

A **`#![no_std]` Rust Digital Signal Processing library** designed for microcontrollers (Cortex-M, RISC-V, AVR, Xtensa), embedded systems, and real-time signal processing. Neural-net and classical ML inference live in [`embedded-nn`](https://github.com/leftger/embedded-nn).

---

## Features

- **`#![no_std]` First**: Pure `core` compatibility for bare-metal targets with zero dynamic allocation required.
- **`libm`, `defmt`, & `serde` Integrations**: Optional formatting logs via `defmt`, model serialization via `serde`, and floating-point math routines in `#![no_std]` environments via `libm`.
- **Per-module Cargo features**: Every algorithm module is optional. `full` (in `default`) enables them all; `default-features = false` plus the modules you want keeps firmware images small.
- **Fixed-Point & Floating-Point**: CMSIS-style `f32`, `f64`, `q31`, `q15`, `q7`, and `q63` saturating arithmetic, plus Q16.16 (`fixed-point`) and a 256-entry sin/cos LUT (`lut`).
- **22 Core DSP Modules**:
  1. **Basic Math**: Elementwise `add`, `sub`, `mult`, `negate`, `offset`, `scale`, `shift`, `dot_prod`, `clip`, bitwise operations.
  2. **Complex Math**: Complex vector addition, multiplication, magnitude, conjugate, dot product.
  3. **Fast Math**: Trigonometric `sin`, `cos`, `tan`, `sin_cos`, `sqrt`, `vsqrt`, `divide`, `log`, `log10`, `exp`, `atan2`.
  4. **Filtering**: FIR filters, Biquad IIR cascade (DF1 and transposed DF-II, f32/q15/q31), LMS / leaky LMS / NLMS (`f32`/`q15`), 1D convolution & correlation, FFT fast convolution, 1D conditional/thresholded median filters (`f32`, `q15`, `q31`), single-pole recursive low/high-pass filters (`SinglePoleFilter` / `SinglePoleFilterQ15`), Q15 DC blocker (`DcBlockerQ15`), O(1) recursive moving average (`RecursiveMovingAverage<N>` / `RecursiveMovingAverageQ15<N>`), and const-generic real-time `CircularBuffer<T, N>`.
  5. **Filter Design**: Biquad Low-Pass, High-Pass, Band-Pass, Notch, Peaking EQ, All-Pass, multi-stage Butterworth design, multi-stage Chebyshev Low-Pass/High-Pass design, continuous-to-discrete Bilinear Transform with cutoff frequency pre-warping, Windowed-Sinc FIR design (Low-pass, High-pass, Band-pass, Band-stop), and arbitrary-response FIR design via frequency sampling (`fir_custom_frequency_sampling`).
  6. **Audio**: Goertzel single-frequency detector (`GoertzelDetector` / `GoertzelDetectorQ15`), peak/RMS envelope followers (`PeakEnvelopeFollower`/`RmsEnvelopeFollower` and Q15), Mel filterbank (`mel_filterbank_f32`), and MFCC feature extraction (`mfcc_f32`).
  7. **Spectral Analysis & PSD**: Welch's method power spectral density estimation (averaged periodograms), single-segment periodograms in linear and dB scale.
  8. **Spatial & 2D Signal Processing**: 2D DCT-II / IDCT-II, 2D spatial convolution with normalization, 2D non-linear filtering (Min/Max/Median), Sobel edge detection, 2D histogram binning, MSE, and PSNR.
  9. **Resampling & Multi-rate**: Cascaded Integrator-Comb (CIC) Decimator & Interpolator, linear fractional resampler, spectral 2:1 sinc zero-padding interpolation.
  10. **Kalman Filtering**: 1D/2D helpers, const-generic linear `KalmanFilter<N, M>`, and trait-based Extended Kalman Filter (`EkfModel`), with `_with_input` variants for models driven by an exogenous input outside the state.
  11. **Const Generics**: Compile-time fixed-size `FirFilter<N>`, `FirFilterQ15<N>`, `BiquadCascade<COEFFS, STATE>`, `BiquadCascadeQ15<COEFFS, STATE>`, and `Matrix<R, C, N>`.
  12. **Transforms**: In-place Complex FFT (`cfft`), Real FFT (`rfft`, packed `rfft_q15`/`rfft_q31` and `irfft_q15`/`irfft_q31`), Discrete Cosine Transform (`dct4`), Fast Walsh-Hadamard Transform (`fwht_f32`/`fwht_i32`), Fixed-Point FFT (`cfft_q15`/`cfft_q31`), Haar Transform (`haar_transform_f32`/`haar_transform_i32`), self-inverse Hartley Transform (`hartley_transform_f32`), and a generalized wavelet transform (`wavelet_transform_f32` with the built-in Daubechies-4 filter).
  13. **Matrix & Regression**: Matrix addition, subtraction, multiplication, scaling, transpose, Gauss-Jordan inversion, and weighted polynomial least-squares curve fitting.
  14. **Controller**: PID motor controller (`f32`/`q15`/`q31`), Clarke and Park transforms (`f32`/`q15`).
  15. **Statistics**: Mean, variance, standard deviation, RMS, power, min/max, entropy, KL divergence, logsumexp.
  16. **Support, PRNG & Noise**: Array copy/fill, zero-allocation sorting (`sort_f32`), format conversions (`q15` ↔ `f32` ↔ `q31`), rounded FIR tap quantizer (`fir_taps_f32_to_q15`), XorShift64 PRNG, uniform and Box-Muller Gaussian noise generators.
  17. **Interpolation**: Linear, Bilinear, and Cubic Spline interpolation.
  18. **Quaternions**: Norm, normalization, quaternion product, conjugate, inverse, rotation matrix conversion.
  19. **Window Functions**: Hanning, Hamming, Blackman, 4-term Blackman-Harris, Bartlett, Welch, Flat-top generators (`f32`), plus Q15 Hanning/Hamming/Blackman/Bartlett and `apply_window_q15`.
  20. **Distance Metrics**: Euclidean, Cosine, Chebyshev, Manhattan, Minkowski, Jaccard, Hamming, Canberra, Bray-Curtis.
  21. **Filter Analysis**: FIR/biquad-cascade frequency response (DTFT) evaluation, magnitude/phase/dB helpers, FIR group delay, and pole-based IIR stability checks.
  22. **Companding**: µ-law and A-law curves (`mu_law_compress_f32` / `a_law_*`) and ITU-T G.711 bytes (`linear_to_ulaw` / `ulaw_to_linear`, `linear_to_alaw` / `alaw_to_linear`).

---

## Quick Start

Add `embedded-dsp` to your `Cargo.toml`:

```toml
[dependencies]
# For standard std environments (all modules)
embedded-dsp = "0.4.0"

# For bare-metal #![no_std] with libm and every algorithm module
embedded-dsp = { version = "0.4.0", default-features = false, features = ["libm", "full"] }

# For bare-metal, only the pieces you use (example: FIR/biquad + Q15 math)
embedded-dsp = { version = "0.4.0", default-features = false, features = ["libm", "filtering", "basic-math"] }
```

`types` and `math` (`FloatMath`) are always compiled. Other modules map 1:1 to Cargo features (`filtering`, `transform`, `kalman`, `fixed-point`, `lut`, …). Enabling `kalman` also pulls `matrix` (and thus `basic-math`); enabling `audio` or `psd` also pulls `transform`. FFT-backed helpers (`fast_convolve_f32`, `fir_custom_frequency_sampling`, `spectral_interpolate_2x_f32`) need `transform` as well.

Minimum supported Rust is **1.88** (edition 2024).

### Migrating from 0.3.0

- Naive Bayes and SVM live in [`embedded-nn`](https://github.com/leftger/embedded-nn), not this crate.
- `default-features = false` no longer compiles every module. Use `features = ["libm", "full"]` or list modules.
- `cfft_q15` / `rfft_q15` (and q31) are integer FFTs with about `1/n` scale versus the `f32` transforms.

### Basic Example

```rust
use embedded_dsp::*;

fn main() {
    // 1. Vector Operations
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let mut vec_out = [0.0f32; 4];
    add_f32(&a, &b, &mut vec_out);

    // 2. Q15 Fixed-Point Saturating Addition
    let q15_a = [20000i16, 25000];
    let q15_b = [15000i16, 10000];
    let mut q15_out = [0i16; 2];
    add_q15(&q15_a, &q15_b, &mut q15_out); // Output: [32767, 32767] (clamped at i16::MAX)

    // 3. Filter Design & Biquad Execution
    let coeffs = biquad_lowpass_coeffs(1000.0, 48000.0, 0.7071);
    let mut biquad = BiquadCascade::<5, 4>::new(coeffs);
    let mut dst = [0.0f32; 4];
    biquad.process(&a, &mut dst);

    // 4. Kalman Sensor Filtering (1D helper or generic N×M)
    let mut kf = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
    kf.predict(0.0);
    let _filtered_reading = kf.update(10.2);

    let mut kf2 = KalmanFilter::<2, 1>::from_variances([0.0, 0.0], 1.0, 0.01, 0.1);
    kf2.predict(&[[1.0, 0.1], [0.0, 1.0]]);
    let _ = kf2.update(&[[1.0, 0.0]], &[1.0]);

    // 5. 64-Point Complex FFT
    let mut fft_data = [0.0f32; 128]; // 64 complex pairs [re, im, ...]
    cfft_f32(&mut fft_data, 64, 0, 1);
}
```

---

## Running Included Examples

```bash
# Run basic usage example
cargo run --example basic_usage

# Run performance comparison benchmark (libm vs embedded-dsp)
cargo run --release --example perf_comparison
```

---

## License

The contents of this repository are dual-licensed under the _MIT OR Apache 2.0_
License. That means you can choose either the MIT license or the Apache 2.0
license when you re-use this code. See [`LICENSE`](./LICENSE), [`LICENSE-MIT`](./LICENSE-MIT), or
[`LICENSE-APACHE`](./LICENSE-APACHE) for more information on each specific
license.
