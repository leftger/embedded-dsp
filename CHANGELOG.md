# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Restored Q16.16 arithmetic (`fixed-point`) and compile-time sin/cos lookup
  tables (`lut`) from the 0.2.0 tree.
- Integer (no-FPU) Q15/Q31 kernels: radix-2 `cfft`/`rfft` (per-stage `>>1`),
  Newton `sqrt_q15`/`sqrt_q31`, CORDIC `sin_cos_q31`/`atan2_q15`/`atan2_q31`,
  integer `cmplx_mag` / `std` / `rms`, DF1 `biquad_cascade_df1_q15`/`q31`,
  SOS quantizers `biquad_coeffs_f32_to_q15`/`q31`, and LUT `sin_q16`/`cos_q16`.
- Per-module Cargo features (plus a `full` meta-feature on by default), matching
  the 0.2.0 "compile only what you use" model. `types` and `math` stay always-on.
  FFT-backed helpers (`fast_convolve_f32`, `fir_custom_frequency_sampling`,
  `spectral_interpolate_2x_f32`) additionally require `transform`.

### Changed

- `default-features = false` no longer compiles every DSP module. Bare-metal
  installs that want the whole crate should use
  `features = ["libm", "full"]` (or pick individual modules).

### Removed

- Gaussian Naive Bayes and SVM classifiers. Classical ML inference belongs in
  `embedded-nn`; this crate keeps the DSP front-end (Mel filterbank, MFCC,
  Goertzel, envelopes) that those models consume.

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
