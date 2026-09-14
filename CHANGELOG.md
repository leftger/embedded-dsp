# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.6.0] - 2026-09-13

### Added

- **C ABI** (`embedded-dsp-ffi`): a standalone `staticlib`/`cdylib` exposing FIR, Direct Form I biquad cascades, the complex FFT, and the audio-EQ designer to C and C++, with a hand-maintained header and a C smoke test that compiles, links, and runs against it in CI.
- **Python bindings** (`embedded-dsp-py`): a PyO3 stable-ABI (`abi3-py39`) extension module (`embedded_dsp`) exposing the audio-EQ designer, FIR, and biquad cascades, with maturin wheel metadata.
- **Verification CI**: a Miri undefined-behaviour job, bounded `cargo-fuzz` differential targets (`fuzz/`, `f32` against in-process `f64`) backed by an always-on randomized differential test, a scheduled sharded `cargo-mutants` run over the always-compiled core, `cargo-hack` per-feature builds, and `cargo-machete`.
- **Unified audio-EQ builder, `IHo`, and WebAudio export** (`filter_design`): `EqFilter` + `EqShape` + `BiquadType` design every RBJ Audio EQ Cookbook response from one fluent, validating builder — including the `IHo` (integrator-over-harmonic-oscillator) section `idsp` had and this crate lacked — and `WebAudioFilter` mirrors a `BiquadFilterNode`'s `type`/`frequency`/`detune`/`Q`/`gain`. Coefficients are bit-checked against `idsp`'s `iir::coefficients` across all nine response types.

- **RBJ EQ parameter conversions**: `biquad_q_from_bw` and `biquad_q_from_shelf_slope` turn an octave bandwidth or a shelf slope into the `q` the biquad designers take. The bandwidth relation is bilinear-transform corrected, so a band keeps its octave width as the centre approaches Nyquist. Also corrected the `biquad_bandpass_coeffs` doc, which named the wrong variant.
- **Swept-sine inverse filter**: `Sweep::inverse_filter` deconvolves an exponential sweep in one complex multiply per bin, recovering an impulse response at `t = 0`. Matches `idsp` 0.22.1.
- **Hyperbolic CORDIC**: `cordic_sqrt_atanh2_q31` and `cordic_atanh_q31` add the hyperbolic vectoring mode, within ~`6e-9` of `f64` across the representable domain. The `cosh`/`sinh` rotation and the linear modes are deliberately not ported; see the README note.
- **Composable Kalman models** (`kalman_compose`): prediction and measurement are pluggable phases over a shared `Estimate` — vector measurements, control input, and EKF through `Dynamics` — generic over `DspSample`. A superset of the composition `idsp` offers.
- **`idsp` as a test-only oracle**: a dev-dependency, never a runtime one, cross-checking the Kalman models step by step.
- **liquid-dsp ports** (no C vendoring, no heap): RRC/RC/GMSK-TX pulse design, Kaiser-windowed sinc FIR, 2nd-order elliptic biquad, `AgcF32`, CRC/Hamming(7,4), `MSequence`, `PolyphaseResampF32`, `GardnerSymbolSync`, analog FM/DSB-AM/SSB.
- **`examples/sr_spike.rs`**: measures `SquareRootKalmanFilter` against `KalmanFilter` and an `f64` reference, over two-state regimes and a state-dimension sweep at `N = 4…12`.
- **`CostasLoop::reset()`**: restores phase, frequency estimate and arm-filter state, keeping the configured centre frequency and coefficients.
- **Typed layout views and combinators** (`pipeline`): `View`/`ViewMut` with `FrameMajor`/`LaneMajor` markers and `as_layout` re-read the same storage under either layout with no copy and no `unsafe`, driven by `SplitViewProcess`/`SplitViewInplace` and the `ViewProcess`/`ViewInplace` forms on `Split`. Added `Parallel` (per-lane branches over tuples or arrays), `ByLane` (per-lane configuration), `Butterfly`, `Unsplit`, and `Split::stateless`/`stateful` — the `dsp-process` design, natively and with no new dependency. The comparison table's view-framework row now reads at parity.
- **Chunk bridges** (`pipeline`): `FnSplitProcess` adapts a closure into a `SplitProcess`; `ChunkInOut<P, Q, R>` runs a chunk stage (`[X; Q] → [Y; R]`) over a block, with the chunk counts checked at compile time; `PerFrame` plus `Split::process_frames`/`inplace_frames` run the same stage once per frame of a frame-major view. The remaining `dsp-process` gaps are now the gap-tolerant `Buffer` and the scratch-buffer `Major`.
- **`Buffer`** (`pipeline`): one fixed-size buffer doing three jobs by which `Process` implementation is in play — a delay line (`X → X`), a chunk collector (`X → Option<[X; N]>`), or a chunk streamer (`Option<[X; N]> → X`), with `block` and `inplace` forms that cross a gapped stream in runs instead of sample by sample. `Major`'s stage-major traversal is now the only `dsp-process` piece still unported.
- **Sample-generic stages (`DspSample`)**: the trait gained `Accum`/`Coeff` associated types and `madd`/`from_accum`/`coeff_from_f32`. Fixed-point widths express their widening multiply-accumulate through the trait — full Q30/Q62 product in `i64`, one saturating narrow in `from_accum` — which is what lets a stage be written once instead of as a hand-written twin per width. The existing arithmetic (`sat_*`, `abs_val`, `to_f32`, `from_f32`) is unchanged.
- **Shift-aware and high-product accumulation (`DspSample`)**: `FRAC`, `mul_high`, `from_accum_shifted`, and `accum_from_shifted` cover the other two fixed-point accumulation schemes the crate already used — per-term high-multiply (FIR), `FRAC - post_shift` narrowing (biquad cascades), and pre-shifted transposed state (DF-II transposed) — again without changing per-width arithmetic.
- **`AdaptiveSample` and `average_accum`**: `AdaptiveSample` (`: DspSample<Coeff = Self>`) carries the LMS/NLMS scalar algebra that genuinely differs between float and fixed point; `DspSample::average_accum` covers the moving average's accumulator division.

### Fixed

- **`SquareRootKalmanFilter` did not do square-root filtering.** Both steps formed `P` and re-factorized it, and a `.max(1e-12)` Cholesky floor pinned the reported covariance at ~`1e-12`. Prediction now takes a Householder QR and the correction a whitened Potter update, so `P` is never formed: the covariance is **7×/152×/205× closer to `f64` at `N = 8/10/12`**. `N` and `M` are bounded at 16.
- **`linear_to_alaw` panicked for inputs `-7..=-1`** (a negative shift from a negative segment index). It now uses the G.711 13-bit form, bit-identical to the reference across all 65 536 inputs.
- **`CORDIC_ATAN_Q31[0]` was ~6% too large**, because the series was evaluated at `x = 1` where it reaches 0.835. `atan_taylor` now folds at 0.5 with two more terms; entry 0 is exactly `0x2000_0000` and every entry is within 4 LSBs.
- **Fixed-point `atan2` saturated `i32` on large inputs**, corrupting the angle by up to **9.9°**. The first-rotation pre-scale now normalises into `[2^28, 2^29)`; worst error is `5.6e-9`.
- **`CostasLoop` carrier tracking** mixed its quadrature arm with `-sin(theta)` instead of `+sin(theta)` and used an unnormalised detector, so it collapsed to ~0 Hz at the 50 Hz loop bandwidth this crate's own tests use. It now mixes with `+sin`, filters both arms, and detects with `2·i·q/(i² + q²)`, giving an amplitude-independent estimate across the pull-in range.

### Changed

- **`CostasLoop::process_sample` returns the low-pass filtered I/Q arms** (demodulated baseband) rather than the raw mixer outputs.
- **`CostasLoop`'s effective loop bandwidth is capped at `center_freq_hz / 20`**: `loop_bandwidth_hz` is an upper bound, since the arm filter's lag destabilises larger values. Documented on the type.
- **`SinglePoleFilter` is generic over `DspSample`**: `SinglePoleFilter<T>` now serves both the `f32` and `q15` recurrences and `SinglePoleFilterQ15` is removed. Constructors are per-width (`SinglePoleFilter::<f32>::lowpass`, `SinglePoleFilter::<q15>::lowpass`/`lowpass_from_f32`), with a coefficient-taking `SinglePoleFilter::<T>::new` for other widths; the `DspNode` bridge moved out of `pipeline` and next to the type. The `q15` path is bit-identical to the old twin and unchanged in speed (266 MSamples/s before and after).
- **FIR and the biquad cascades are generic over `DspSample`**: `FirInstance<'a, T>`, `BiquadCascadeInstance<'a, T>`, and `BiquadCascadeDf2tInstance<'a, T>` replace nine `*F32`/`*Q15`/`*Q31` twins. The old names remain as type aliases and `fir_f32`/`fir_q15`/`biquad_cascade_df1_q15`/… as thin wrappers, so existing call sites are unchanged; the fixed-point cascades gained `with_post_shift` (`init` is the zero-headroom form). Each generic loop is bit-for-bit identical to the kernel it replaces.
- **LMS/NLMS and the recursive moving average are generic over `DspSample`**: `LmsInstance`/`NlmsInstance` (built on `AdaptiveSample`) and `RecursiveMovingAverage<T, N>` replace four width twins, again keeping the old names as aliases/wrappers. The q15 paths are bit-exact with the kernels they replace; `RecursiveMovingAverage::<N>` becomes `RecursiveMovingAverage::<f32, N>`.
- **One composition vocabulary (`pipeline`)**: `Process`/`Inplace` are now blanket-derived from `SplitProcess`/`SplitInplace` — a stateless stage implements `SplitProcess<X, Y, ()>` once and inherits `Process` (and, through the second blanket, `DspNode`) — so `Split`, `Chain`, `Gain`, `Limiter`, `Offset`, `Identity`, `Buffer`, `SinglePoleFilter<T>`, `PidInstanceF32/Q15`, and `DcBlockerQ15` lost their hand-written bridges. The split vocabulary's receivers became `&mut self` and its methods were renamed to keep method resolution unambiguous: `SplitProcess::process`/`block` → `process_with_state`/`block_with_state`, `SplitInplace::inplace` → `inplace_with_state`, `SplitViewProcess::process_view` → `process_view_with_state`, `SplitViewInplace::inplace_view` → `inplace_view_with_state`. `DspNode::process_block`'s shorter-buffer clamp and the specialised `Buffer`/`ChunkInOut` block/in-place paths are preserved.

### Removed

- **The duplicate `filtering::Dsm` and `filtering::XorShift32` are gone.** They shadowed the dedicated `dsm::Dsm` / `dither::XorShift32` modules, so only one of each could be re-exported at the crate root. The dedicated modules are now the single implementation, re-exported at the crate root as before, and `filtering` re-exports neither name. `dsm::Dsm` is `Default` + `process()` + `reset()` — the historical `new()`/`process_sample()` names are deliberately **not** carried over. `dither::XorShift32` gains the `next_f32()` / `tpdf_dither_f32()` helpers that only the removed duplicate had.

- **The sample-genericization compatibility layer is removed.** The width-suffixed instance aliases (`FirInstanceF32/Q31/Q15`, `BiquadCascadeInstanceF32/Q15/Q31`, `BiquadCascadeDf2tInstanceF32/Q15/Q31`, `LmsInstanceF32/Q15`, `NlmsInstanceF32/Q15`, `PidInstanceF32/Q31/Q15`, `HilbertTransformF32/Q15`, `RecursiveMovingAverageQ15`) and their thin width wrappers (`fir_f32/q31/q15`, `biquad_cascade_df1_*`, `biquad_cascade_df2t_*`, `lms_*`, `lms_leaky_*`, `nlms_*`, `pid_f32/q31/q15`) are gone. Call the generic API directly instead: `FirInstance<f32>`, `BiquadCascadeInstance<q15>`, `LmsInstance<f32>`, `HilbertTransform<'_, f32>`, `RecursiveMovingAverage<q15, N>`, and the free functions `fir` / `biquad_cascade_df1` / `biquad_cascade_df2t` / `lms` / `lms_leaky` / `nlms`; PID updates are `PidInstance::<T>::process`.

### Changed

- **`filtering` is split into family submodules** (`fir`, `biquad`, `convolution`, `adaptive`, `recursive`, `lockin`, `int_filters`, `normal_form`, `wdf`) behind a re-exporting facade; `embedded_dsp::filtering::*` and the crate-root glob are unchanged. The integration-test suite is likewise consolidated from 46 files into 36 domain-named files, with every test and its `required-features` preserved.
- **docs.rs shows feature-gate badges** for every module (`#[cfg_attr(docsrs, doc(cfg(...)))]`).

## [0.5.1] - 2026-09-06

### Changed

- **The `fixed` dependency is now optional**, pulled in only by the fixed-point features; default builds no longer compile it.
- **Dependency bumps**: `defmt` 0.3 → 1.1, plus Dependabot action updates.
- **CI**: Codecov coverage, a `cargo-deny` audit, Dependabot, and a release workflow.

## [0.5.0] - 2026-09-01

### Added

- **`nalgebra` interop**: New optional `nalgebra` feature (implies `quaternion`) adding `quaternion_to_nalgebra`, `quaternion_from_nalgebra`, `quaternion_to_unit_nalgebra`, and `quaternion_from_unit_nalgebra` to bridge the `[w, x, y, z]` quaternion representation used by `quaternion` with `nalgebra`'s `Quaternion`/`UnitQuaternion`, for pipelines that pair `embedded-dsp` with a `nalgebra`-based crate downstream.

### Removed

- **Breaking**: Removed the `Q15`/`Q31` strongly-typed newtypes added in 0.4.1. They were never adopted internally, and would have been a redundant wrapper now that `q7`/`q15`/`q31` are themselves real `fixed` crate types (see below).

### Changed

- Fixed-point internals (`fixed_point::Q16` arithmetic and the shared `q7_mult`/`q15_mult`/`q31_mult` saturating-multiply helpers) now build on the `fixed` crate instead of hand-rolled bit-shifting. Public function names, signatures, and documented behavior are unchanged. Raises MSRV to 1.93.
- **Breaking**: `q7`, `q15`, and `q31` are no longer plain `i8`/`i16`/`i32` aliases — they're now real [`fixed`](https://crates.io/crates/fixed) crate types (`fixed::types::I1F7`/`I1F15`/`I1F31`), so they interoperate directly with the rest of the Rust fixed-point ecosystem instead of just being CMSIS-flavored raw integers. Construct a value from a raw CMSIS-style bit pattern with `q15::from_bits(20000)`, get it back with `.to_bits()`, and use `q15::ZERO` / `q15::saturating_from_num(0.5)` in place of `0i16` / manual float-to-fixed shifting. Bare arithmetic operators now panic-on-overflow in debug builds like any other primitive-backed type; existing call sites already used the explicit `wrapping_*`/`saturating_*`/`checked_*` methods and are unaffected. `q63` is unchanged — it remains a plain `i64` wide-accumulator alias, not a fixed-point value.
- **Breaking**: `fast_log2_q15`'s return type changed from `q15` to a new `Q8F7` type (`fixed::FixedI16<U7>`), matching the Q8.7 range its packed integer result actually occupies (previously mistyped as Q1.15, which would silently saturate legitimate outputs now that `q15` enforces the `[-1, 1)` range). Similarly, `GoertzelDetectorQ15`'s `coeff` field is now the new `Q2F14` type (`fixed::FixedI16<U14>`) instead of `q15`, since Goertzel coefficients (`2*cos(ω)`) range up to ±2.0.
- Adds `impl DspSample for q15` and `impl DspSample for q31`, restoring generic-programming support over the real fixed-point types (previously only implemented for the now-removed `Q15`/`Q31` newtypes).

## [0.4.1] - 2026-08-26

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
