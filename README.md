# embedded-dsp

[![Crates.io](https://img.shields.io/crates/v/embedded-dsp.svg)](https://crates.io/crates/embedded-dsp)
[![Documentation](https://docs.rs/embedded-dsp/badge.svg)](https://docs.rs/embedded-dsp)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

A **`#![no_std]` Rust Digital Signal Processing library** designed for microcontrollers (Cortex-M, RISC-V, AVR, Xtensa), embedded systems, and real-time digital signal processing applications.

---

## Features

- **`#![no_std]` First**: Pure `core` compatibility for bare-metal targets with zero dynamic allocation required.
- **`libm` Integration**: Unified `FloatMath` trait providing floating-point math routines in `#![no_std]` environments via `libm`.
- **Fixed-Point & Floating-Point**: Complete support for `f32`, `f64`, `q31`, `q15`, `q7`, and `q63` saturating arithmetic.
- **16 Core DSP Modules**:
  1. **Basic Math**: Elementwise `add`, `sub`, `mult`, `negate`, `offset`, `scale`, `shift`, `dot_prod`, `clip`, bitwise operations.
  2. **Complex Math**: Complex vector addition, multiplication, magnitude, conjugate, dot product.
  3. **Fast Math**: Trigonometric `sin`, `cos`, `sin_cos`, `sqrt`, `vsqrt`, `divide`, `log`, `exp`, `atan2`.
  4. **Filtering**: FIR filters, Biquad IIR cascade, LMS adaptive filters, 1D convolution & correlation.
  5. **Transforms**: In-place Complex FFT (`cfft`), Real FFT (`rfft`), Discrete Cosine Transform (`dct4`).
  6. **Matrix Operations**: Matrix addition, subtraction, multiplication, scaling, transpose, Gauss-Jordan inversion.
  7. **Controller**: PID motor controller, Clarke and Park transforms.
  8. **Statistics**: Mean, variance, standard deviation, RMS, power, min/max, entropy, KL divergence, logsumexp.
  9. **Support & Conversions**: Array copy/fill, zero-allocation sorting (`sort_f32`), format conversions (`q15` ↔ `f32` ↔ `q31`).
  10. **Interpolation**: Linear, Bilinear, and Cubic Spline interpolation.
  11. **Quaternions**: Norm, normalization, quaternion product, conjugate, inverse, rotation matrix conversion.
  12. **Window Functions**: Hanning, Hamming, Blackman, Bartlett, Welch, Flat-top generators.
  13. **Distance Metrics**: Euclidean, Cosine, Chebyshev, Manhattan, Minkowski, Jaccard, Hamming, Canberra, Bray-Curtis.
  14. **Machine Learning**: Support Vector Machine (`SvmInstanceF32`) and Gaussian Naive Bayes (`GaussianNaiveBayesInstanceF32`).

---

## Quick Start

Add `embedded-dsp` to your `Cargo.toml`:

```toml
[dependencies]
# For bare-metal #![no_std] environments with libm
embedded-dsp = { version = "0.1", default-features = false, features = ["libm"] }

# For standard std environments
embedded-dsp = "0.1"
```

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

    // 3. 64-Point Complex FFT
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

Dual-licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))
