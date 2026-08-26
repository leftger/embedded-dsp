# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- **Hardware DSP Intrinsics**: ARM Cortex-M assembly intrinsics (`smlad`, `smlald`, `qadd16`, `qsub16`, `ssat`) via `cortex-m-dsp`, with portable SWAR vector fallbacks.
- **Strongly-Typed Fixed-Point & Polymorphism**: Strongly-typed `Q15` and `Q31` newtypes with operator overloading and `DspSample` polymorphic trait.
- **Block Floating-Point (BFP) FFT**: Headroom-preserving `cfft_bfp_q15` and `cfft_bfp_q31` for 30–40 dB higher dynamic range without bit loss.
- **Homomorphic Deconvolution**: `real_cepstrum_f32` for echo, seismic, sonar, and speech formant separation.
- **Pure-Integer CORDIC Engine**: `cordic_sin_cos_q15/q31`, `cordic_cartesian_to_polar_q15`, `cordic_atan2_q15`, `cordic_sqrt_q15` using 16-iteration shift-and-add arithmetic without hardware multipliers.
- **Zero-Allocation Streaming Pipelines**: Composable `DspNode<T>` processing chains with `Chain`, `Gain`, `Limiter`, `SinglePoleFilter`, `DcBlockerQ15`, and `PidInstance`.
- **Generalized Filterbanks & Fixed-Point VAD**: `generalized_triangular_filterbank` (Linear, Octave, Mel, Bark), integer `fast_log2_q15`, and energy + zero-crossing `VadDetectorQ15`.
- **Filter Quantization & SQNR Analysis**: Automated $L_\infty$ and $L_2$ SOS biquad quantization with scaling headroom prediction and DTFT SQNR analysis.
- **Second-Order Generalized Integrator PLL (SOGI-PLL) & Costas Loop**: `SogiPll` for single-phase grid synchronization (solar inverters/UPS) and `CostasLoop` for BPSK/QPSK carrier recovery.
- **Dynamics Processor & Noise Gate**: `DynamicsCompressor` with soft-knee logarithmic curves and `NoiseGate` downward expander implementing `DspNode<f32>`.
- **Square-Root Covariance Kalman Filter (SRKF)**: `SquareRootKalmanFilter<N, M>` Cholesky-factor state estimator guaranteeing positive semi-definiteness without filter divergence.
- **Autoregressive Burg PSD Estimation**: `ar_burg_f32` and `ar_psd_f32` for super-resolution spectral peaks on short data buffers.
- **Kaiser-Bessel & Flat-Top Windows**: `kaiser_f32` with zero-order modified Bessel $I_0(\beta)$ evaluation.
- **Acoustic Localization & Beamforming**: `DelayAndSumBeamformer` fractional delay array processor and `gcc_phat_tdoa_f32` Time Difference of Arrival (TDoA) locator.
- **Production DSP Cookbook**: `COOKBOOK.md` with 6 real-world copy-paste recipes (FOC motor control, I2S audio DMA, vibration diagnostics, CIC decimation, VAD/MFCC, and streaming chains).
- **Benchmark Suite**: `benches/dsp_benchmarks.rs` tracking throughput across SIMD, FFT, CORDIC, and filtering.
- **Multi-Target GitHub CI Matrix**: Automated testing across 6 architectures (`x86_64`, `thumbv6m`, `thumbv7em`, `thumbv7em-hf`, `riscv32imc`, `wasm32`).

## [0.4.0] - 2026-08-23

This is a **breaking** release relative to crates.io `0.3.0`. New DSP APIs are
additive, but module features, removed classifiers, and honest integer kernels
change what `default-features = false` compiles and how several `q15`/`q31`
entry points behave.

### Added

- Restored Q16.16 arithmetic (`fixed-point`) and compile-time sin/cos lookup
  tables (`lut`) from the 0.2.0 tree.
- Integer (no-FPU) Q15/Q31 kernels: radix-2 `cfft`/`rfft` (per-stage `>>1`),
  Newton `sqrt_q15`/`sqrt_q31`, CORDIC `sin_cos_q31`/`atan2_q15`/`atan2_q31`,
  integer `cmplx_mag` / `std` / `rms`, DF1 `biquad_cascade_df1_q15`/`q31`,
  SOS quantizers `biquad_coeffs_f32_to_q15`/`q31`, and LUT `sin_q16`/`cos_q16`.
- Q15 single-pole IIR (`SinglePoleFilterQ15`) and transposed DF-II biquad
  cascades (`biquad_cascade_df2t_f32`/`q15`/`q31`).
- Packed real FFT for `rfft_q15`/`rfft_q31` (N/2 complex FFT + unpack) and
  matching `irfft_q15`/`irfft_q31`. Combined scale is about `1/n` versus `f32`
  (`irfft(rfft(x)) ≈ x / n`).
- Q15 windows (`hanning_q15`/`hamming_q15`/`blackman_q15`/`bartlett_q15`) and
  `apply_window_q15`.
- Q15 Clarke/Park (`clarke_q15`/`park_q15` and inverses). Park takes Q15
  `sin`/`cos` of θ.
- Integer G.711 μ-law/A-law (`linear_to_ulaw`/`ulaw_to_linear`,
  `linear_to_alaw`/`alaw_to_linear`).
- NLMS and leaky LMS (`nlms_f32`/`nlms_q15`, `lms_leaky_f32`/`lms_leaky_q15`,
  plus `lms_q15`).
- Q15 envelope followers, recursive moving average (`RecursiveMovingAverageQ15`),
  and DC blocker (`DcBlockerQ15`).
- Const-generic `FirFilterQ15` and `BiquadCascadeQ15`.
- Rounded FIR tap quantizer `fir_taps_f32_to_q15`.
- Q15 Goertzel detector (`GoertzelDetectorQ15`).
- Per-module Cargo features (plus a `full` meta-feature on by default), matching
  the 0.2.0 "compile only what you use" model. `types` and `math` stay always-on.
  FFT-backed helpers (`fast_convolve_f32`, `fir_custom_frequency_sampling`,
  `spectral_interpolate_2x_f32`) additionally require `transform`.

### Changed

- `default-features = false` no longer compiles every DSP module. Bare-metal
  installs that want the whole crate should use
  `features = ["libm", "full"]` (or pick individual modules).
- Published `q15`/`q31` FFT, sqrt, trig, complex-magnitude, and RMS/std
  kernels are integer arithmetic (they previously rounded through `f32`).

### Removed

- Gaussian Naive Bayes and SVM classifiers. Classical ML inference belongs in
  [`embedded-nn`](https://github.com/leftger/embedded-nn); this crate keeps the
  DSP front-end (Mel filterbank, MFCC, Goertzel, envelopes) that those models
  consume.

### Migration from 0.3.0

- Replace `use embedded_dsp::{GaussianNb, ...}` with [`embedded-nn`](https://github.com/leftger/embedded-nn).
- Bare-metal: `default-features = false, features = ["libm", "full"]` (or a
  module list). `features = ["libm"]` alone is no longer the whole crate.
- Integer FFT: expect ~`1/n` scaling versus `cfft_f32` / `rfft_f32`.

## [0.3.0] - 2026-08-23

### Added

- **Filter analysis**: DTFT frequency-response evaluation for FIR taps, single biquad
  sections, and biquad cascades (`fir_frequency_response`, `biquad_frequency_response`,
  `biquad_cascade_frequency_response`), magnitude/phase/dB helpers, FIR group delay
  (`fir_group_delay`), and pole-based IIR stability checks (`biquad_pole_radius`,
  `biquad_is_stable`, `biquad_cascade_is_stable`).
- **Chebyshev filter design**: `chebyshev_biquad_stage`, `chebyshev_lowpass_biquads`,
  `chebyshev_highpass_biquads` — a multi-stage recursive Chebyshev low/high-pass design,
  parameterized by passband ripple and pole count.
- **Single-pole recursive filters**: `SinglePoleFilter` (low-pass/high-pass) and decay-factor
  helpers (`single_pole_decay_from_cutoff`, `single_pole_decay_from_time_constant`) — the
  cheapest possible IIR smoothing/DC-blocking filter.
- **Recursive moving average**: `RecursiveMovingAverage<N>`, an O(1)-per-sample const-generic
  moving-average filter (add/subtract only, no convolution).
- **Haar transform**: `haar_transform_f32` / `inverse_haar_transform_f32` (in-place,
  orthogonal) and `haar_transform_i32` (non-normalized, integer, forward-only).
- **Hartley transform**: `hartley_transform_f32`, a real, self-inverse transform
  (`H[H[a]] = a`).
- **Generalized wavelet transform**: `wavelet_step_f32` / `inverse_wavelet_step_f32` /
  `wavelet_transform_f32` / `inverse_wavelet_transform_f32` for arbitrary orthogonal wavelet
  filter taps, with the Daubechies-4 filter provided (`DAUBECHIES_4`).
- **Custom FIR filter design**: `fir_custom_frequency_sampling` — designs an FIR kernel
  matching an arbitrary desired frequency response via the frequency-sampling method (inverse
  FFT, circular shift, truncate, Hamming window).
- **Audio companding**: `mu_law_compress_f32` / `mu_law_expand_f32` and
  `a_law_compress_f32` / `a_law_expand_f32` (ITU-T G.711-family nonlinear compression curves).
- **Welch PSD & spectral analysis, 2D spatial DSP, bilinear transform filter design, and
  extended matrix/regression math** (Welch's method PSD and periodograms; 2D DCT/IDCT, 2D
  convolution, non-linear filtering, Sobel edge detection, 2D histogram, MSE/PSNR; the
  bilinear transform with cutoff pre-warping; weighted polynomial least-squares fitting).

### Changed

- Bumped `edition` to `2024` in `Cargo.toml` (raises the minimum supported Rust version
  needed to build this crate).

## [0.2.2] - 2026-08-08

### Added

- Exogenous-input support for `ExtendedKalmanFilter`: `_with_input` variants of `predict`/
  `update` for models driven by an input signal outside the state vector.

## [0.2.1] - 2026-08-08

Initial crates.io release.
