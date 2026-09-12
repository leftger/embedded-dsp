# embedded-dsp

<p align="center">
  <img src="assets/aztec_rustacean.png" alt="embedded-dsp" width="100%">
</p>

[![crates.io](https://img.shields.io/crates/v/embedded-dsp.svg)](https://crates.io/crates/embedded-dsp)
[![docs.rs](https://img.shields.io/docsrs/embedded-dsp)](https://docs.rs/embedded-dsp)
[![CI](https://github.com/leftger/embedded-dsp/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/leftger/embedded-dsp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/leftger/embedded-dsp/branch/master/graph/badge.svg)](https://codecov.io/gh/leftger/embedded-dsp)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

A high-performance **`#![no_std]` Rust Digital Signal Processing library** designed for microcontrollers (Cortex-M, RISC-V, AVR, Xtensa), bare-metal DSP, and real-time audio/sensor pipelines.

---

## Highlights

- **`#![no_std]` First**: Pure `core` compatibility with zero heap allocations.
- **Fixed & Float Parity**: CMSIS-style `f32`, `f64`, and `q7`/`q15`/`q31` — interoperable with the [`fixed`](https://crates.io/crates/fixed) crate (optional `fixed` feature, enabled by default) with zero-dependency fallback newtypes — plus the polymorphic `DspSample` trait.
- **Hardware Acceleration**: ARM Cortex-M assembly intrinsics (`smlad`, `smlald`, `ssat`, `qadd16`) via `cortex-m-dsp`, with portable SWAR vector fallbacks.
- **Pure-Integer CORDIC Engine**: Shift-and-add `sin`, `cos`, `atan2`, polar conversion, and `sqrt` requiring no hardware multipliers.
- **Streaming Pipelines**: Zero-allocation [`DspNode`](src/pipeline.rs) composable processing chains (`Chain`, `Gain`, `Limiter`).
- **Production Tested**: Continuous integration across 7 bare-metal architectures (`thumbv6m`, `thumbv7em`, `thumbv7em-hf`, `thumbv8m-main-hf`, `riscv32imc`, `wasm32`, `x86_64`).

---

## ⚡ embedded-dsp Studio (Interactive DSP Workbench)

> **[🌐 Launch Live WebAssembly Studio](https://leftger.github.io/embedded-dsp/)** — Run real-time filter design, spectral forensics, and micro-benchmarks directly in your browser.

An interactive desktop & browser testbench inspired by `DSP-Testbench` and `embedded-nn`:

```bash
# Run native desktop GUI
cargo run -p embedded-dsp-studio
```

- **Dual Signal Lab**: Anti-aliased PolyBLEP oscillators (Saw/Square/Triangle/Sine), chirp sweeps (linear/exponential), pink noise ($1/f$), white noise, and Dirac impulse.
- **Filter & Pipeline Rack**: Real-time sweepable Biquad IIR filters, Andrew Simper 2x oversampled State Variable Filters (SVF), Audio EQ Cookbook shelves, and Q15 quantization floor.
- **Forensics & Analyzer**: 4096-point FFT magnitude, Welch PSD, oscilloscope, Lissajous phase goniometer, and true RMS/peak meters.
- **Impulse Response Analyzer**: 4096-sample freeze buffer with settling time, peak gain, and energy measurements.
- **WAV & CSV Export/Import**: Export forensic snapshots and waveforms directly to 16-bit PCM `.wav` or `.csv` files.
- **Zero-Allocation MCU Codegen**: Generates instant C (CMSIS-DSP) and `#![no_std]` Rust code snippets tuned in the GUI.

---

## Comparison with `idsp`

`embedded-dsp` is a superset of the well-established [`idsp`](https://crates.io/crates/idsp) crate for the
algorithms the two share, and adds transforms, audio/vision, sensor-fusion, control, and tooling on top.
The fixed-point/integer algorithms were ported and re-verified against `idsp`'s own test vectors where
available. Honest differences are marked, including the few places where `idsp` still has more to offer.

| Feature | `embedded-dsp` | `idsp` |
| :--- | :---: | :---: |
| `#![no_std]`, zero-allocation | ✅ | ✅ |
| Fixed point `q7`/`q15`/`q31` + `fixed` interop | ✅ | ✅ (`i8`/`i16`/`i32`/`i64`) |
| `cossin` LUT (i32) | ✅ ~5e-6 RMS | ✅ ~4e-6 RMS |
| `atan2` (i32) | ✅ ~1.3e-6 rad | ✅ ~1.3e-6 rad |
| Integer `PLL` / reciprocal `RPLL` | ✅ | ✅ |
| Integer lowpass, CORDIC, unwrap, `saturating_scale` | ✅ | ✅ |
| Biquad `f32`/`f64` DF1 + DF2T | ✅ | ✅ |
| Biquad `i32` clamping / anti-windup / guard bits | ✅ | ✅ |
| Biquad fixed-point noise shaping | ✅ | ✅ |
| Biquad generic integer `i8`/`i16`/`i32`/`i64` | ✅ `BiquadInt<T>` | ✅ |
| Biquad DF1 wide (`Q32.32`) / dither actions | ✅ | ✅ |
| Control-plane settings via `miniconf` | ✅ `config::BiquadSettings` | ✅ |
| Normal-form IIR | ✅ arbitrary numerator | ⚠️ forced `p.im·z⁻¹` factor |
| Wave digital allpass filters | ✅ | ✅ |
| PI²D² controller builder (per-action limits) | ✅ | ✅ |
| Half-band Type I–IV linear-phase FIR | ✅ | ✅ |
| Half-band cascades with known-good taps | ✅ 140 dB + 98 dB | ✅ 140 dB |
| CIC decimator/interpolator | ✅ | ✅ |
| General FIR, LMS/NLMS | ✅ | ➖ |
| FFT (CFFT/RFFT/BFP Q15/Q31), DCT, DWT, Hartley, Hilbert | ✅ | ❌ |
| Goertzel, Mel/MFCC, VAD, compressor/gate | ✅ | ❌ |
| Welch/Burg PSD analysis | ✅ | ➖ |
| Kalman (const-generic, EKF, square-root) | ✅ | ✅ (composable models) |
| 2D vision, beamforming, GCC-PHAT, quaternions, matrices | ✅ | ❌ |
| Lock-in amplifier | ✅ | ✅ |
| Dither + MASH delta-sigma | ✅ | ✅ |
| Resampling (polyphase, fractional) | ✅ | ➖ |
| Companding (G.711 µ/A-law) | ✅ | ❌ |
| In-repo micro-benchmarks | ✅ | ✅ (`tests/embedded`) |
| Python bindings | ❌ | ✅ |
| Interactive WebAssembly studio | ✅ | ❌ |

Legend: ✅ full support · ➖ partial/alternative coverage · ⚠️ quirk · ❌ not provided.

---

## Module Overview

| Category | Key Algorithms & Structs |
| :--- | :--- |
| **Filtering & Design** | FIR, Biquad IIR (DF-I & Transposed DF-II), LMS/NLMS, Butterworth/Chebyshev design, Windowed-Sinc, $L_\infty/L_2$ SOS Quantization & SQNR analysis, DC Blocker. |
| **Spectral & Transforms** | CFFT, RFFT (packed), Block Floating-Point FFT (`cfft_bfp_q15/q31`), Hilbert Transform FIR & Analytic Signal (`HilbertTransformF32/Q15`), Real Cepstrum, DCT-IV, FWHT, Haar, Hartley, Daubechies-4 DWT, Welch & Burg AR PSD. |
| **Audio & Voice** | Goertzel tone detector, Mel & Generalized filterbanks, MFCC, Q15 VAD, Dynamics Compressor with soft knee, Noise Gate. |
| **Control & Power** | FOC current/speed PID, Clarke & Park transforms, SOGI-PLL (grid synchronization/resolvers), Costas Loop carrier recovery. |
| **Sensor Fusion & Spatial** | Square-Root Kalman Filter (`SquareRootKalmanFilter`), EKF, 2D Spatial/Vision (Sobel, Median, DCT-II), Delay-and-Sum Beamformer, GCC-PHAT TDoA locator. |
| **Multi-rate & Resampling** | CIC Decimator/Interpolator with bit-growth normalization, Polyphase Decimation & Interpolation (Float & Q15), fractional linear resampler. |
| **Math, CORDIC & Numerics** | `BFloat16` (50% SRAM buffer reduction), `FloatFloat` (~48-bit double-single extended precision on `f32` FPU), Fast Bit-Manip Log/Pow/dB (`fast_log2`, `fast_pow2`, `fast_gain_to_db`), EFT (`two_sum`, `two_prod`, `two_div`), Horner polynomials & roots, strided dot products, CORDIC engine, Complex math, Quaternions (`nalgebra` interop), 9 Windows, G.711 $\mu$/A-law companding. |

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
# Standard std environment (all modules enabled)
embedded-dsp = "0.5.0"

# Bare-metal #![no_std] with libm
embedded-dsp = { version = "0.5.0", default-features = false, features = ["libm", "full"] }

# Minimal firmware footprint (only FIR/Biquad filtering + basic math)
embedded-dsp = { version = "0.5.0", default-features = false, features = ["libm", "filtering", "basic-math"] }
```

### Basic Example

```rust
use embedded_dsp::*;

fn main() {
    // 1. Fixed-Point Saturating Addition
    let a = [q15::from_bits(20000), q15::from_bits(25000)];
    let b = [q15::from_bits(15000), q15::from_bits(10000)];
    let mut out = [q15::ZERO; 2];
    add_q15(&a, &b, &mut out); // [I1F15::MAX, I1F15::MAX] (saturated)

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

### Benchmarks

Native micro-benchmarks (run in CI, results attached to the job summary):

```bash
cargo bench -p embedded-dsp --bench dsp_benchmarks
```

Bare-metal cycle counts on Cortex-M are in [`tests/embedded`](tests/embedded) and include the
integer primitives shared with `idsp` (`cossin`, `atan2`, `IntPll`).

---

## License

Dual-licensed under either **MIT** or **Apache-2.0** at your option. See [`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE).
