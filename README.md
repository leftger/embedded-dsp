# embedded-dsp

<p align="center">
  <img src="assets/aztec_rustacean.png" alt="embedded-dsp" width="100%">
</p>

[![crates.io](https://img.shields.io/crates/v/embedded-dsp.svg)](https://crates.io/crates/embedded-dsp)
[![docs.rs](https://img.shields.io/docsrs/embedded-dsp)](https://docs.rs/embedded-dsp)
[![CI](https://github.com/leftger/embedded-dsp/actions/workflows/ci.yml/badge.svg)](https://github.com/leftger/embedded-dsp/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

A high-performance **`#![no_std]` Rust Digital Signal Processing library** designed for microcontrollers (Cortex-M, RISC-V, AVR, Xtensa), bare-metal DSP, and real-time audio/sensor pipelines.

---

## Highlights

- **`#![no_std]` First**: Pure `core` compatibility with zero heap allocations.
- **Fixed & Float Parity**: CMSIS-style `f32`, `f64`, `q7`, `q15`, `q31`, strongly-typed `Q15`/`Q31` newtypes, and the polymorphic `DspSample` trait.
- **Hardware Acceleration**: ARM Cortex-M assembly intrinsics (`smlad`, `smlald`, `ssat`, `qadd16`) via `cortex-m-dsp`, with portable SWAR vector fallbacks.
- **Pure-Integer CORDIC Engine**: Shift-and-add `sin`, `cos`, `atan2`, polar conversion, and `sqrt` requiring no hardware multipliers.
- **Streaming Pipelines**: Zero-allocation [`DspNode`](src/pipeline.rs) composable processing chains (`Chain`, `Gain`, `Limiter`).
- **Production Tested**: Continuous integration across 6 bare-metal architectures (`thumbv6m`, `thumbv7em`, `thumbv7em-hf`, `riscv32imc`, `wasm32`, `x86_64`).

---

## Module Overview

| Category | Key Algorithms & Structs |
| :--- | :--- |
| **Filtering & Design** | FIR, Biquad IIR (DF-I & Transposed DF-II), LMS/NLMS, Butterworth/Chebyshev design, Windowed-Sinc, $L_\infty/L_2$ SOS Quantization & SQNR analysis, DC Blocker. |
| **Spectral & Transforms** | CFFT, RFFT (packed), Block Floating-Point FFT (`cfft_bfp_q15/q31`), Real Cepstrum, DCT-IV, FWHT, Haar, Hartley, Daubechies-4 DWT, Welch & Burg AR PSD. |
| **Audio & Voice** | Goertzel tone detector, Mel & Generalized filterbanks, MFCC, Q15 VAD, Dynamics Compressor with soft knee, Noise Gate. |
| **Control & Power** | FOC current/speed PID, Clarke & Park transforms, SOGI-PLL (grid synchronization/resolvers), Costas Loop carrier recovery. |
| **Sensor Fusion & Spatial** | Square-Root Kalman Filter (`SquareRootKalmanFilter`), EKF, 2D Spatial/Vision (Sobel, Median, DCT-II), Delay-and-Sum Beamformer, GCC-PHAT TDoA locator. |
| **Multi-rate & Resampling** | CIC Decimator/Interpolator with bit-growth normalization, Polyphase Decimation & Interpolation (Float & Q15), fractional linear resampler. |
| **Math, CORDIC & Windows** | CORDIC engine, Complex math, Fast math, Quaternions, 9 Window functions (Kaiser-Bessel with $I_0(\beta)$, Flat-top), G.711 $\mu$-law/A-law companding. |

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
# Standard std environment (all modules enabled)
embedded-dsp = "0.4.0"

# Bare-metal #![no_std] with libm
embedded-dsp = { version = "0.4.0", default-features = false, features = ["libm", "full"] }

# Minimal firmware footprint (only FIR/Biquad filtering + basic math)
embedded-dsp = { version = "0.4.0", default-features = false, features = ["libm", "filtering", "basic-math"] }
```

### Basic Example

```rust
use embedded_dsp::*;

fn main() {
    // 1. Fixed-Point Saturating Addition
    let a = [20000i16, 25000];
    let b = [15000i16, 10000];
    let mut out = [0i16; 2];
    add_q15(&a, &b, &mut out); // [32767, 32767] (clamped at i16::MAX)

    // 2. Biquad Filter Cascade
    let coeffs = biquad_lowpass_coeffs(1000.0, 48000.0, core::f32::consts::FRAC_1_SQRT_2);
    let mut filter = BiquadCascade::<5, 4>::new(coeffs);
    let input = [1.0f32, 0.5, -0.2, 0.1];
    let mut filtered = [0.0f32; 4];
    filter.process(&input, &mut filtered);

    // 3. Robust Square-Root Kalman Sensor Filter
    let mut kf = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
    kf.predict(0.0);
    let _est = kf.update(10.2);

    // 4. In-Place FFT
    let mut fft_buf = [0.0f32; 128]; // 64 complex pairs [re, im, ...]
    cfft_f32(&mut fft_buf, 64, 0, 1);
}
```

---

## Cookbook & Examples

Need copy-paste code for real-world projects? Check the **[embedded-dsp Cookbook](COOKBOOK.md)**:
- **Motor Control**: Sensorless Field-Oriented Control (FOC) with Clarke/Park and Space-Vector PWM.
- **Real-Time Audio DMA**: DC-Blocker + Biquad Peaking EQ + Peak Limiter streaming pipeline.
- **Machine Health**: Vibration spectrum analysis and bearing fault detection using Burg AR PSD.
- **Multi-rate ADC**: High-speed Cascaded Integrator-Comb (CIC) decimation.
- **Acoustic Edge AI**: Voice Activity Detection (VAD) & MFCC feature extraction.
- **Streaming Pipeline**: Composing modular `DspNode` signal processing chains.

Run any included example directly with cargo:
```bash
cargo run --example basic_usage
cargo run --example motor_control_foc
cargo run --example audio_speech_pipeline
cargo run --example sensor_fusion_navigation
cargo run --example filter_workbench_and_analysis
cargo run --example spectral_radar_transforms
cargo run --example spatial_vision_processing
cargo run --release --example perf_comparison
```

---

## License

Dual-licensed under either **MIT** or **Apache-2.0** at your option. See [`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE).
