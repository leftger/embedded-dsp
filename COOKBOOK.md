# embedded-dsp Cookbook & Production Recipes

A practical guide to implementing real-time signal processing pipelines on constrained microcontrollers (`#![no_std]`, zero heap allocation) using `embedded-dsp`.

---

## Table of Contents
1. [Motor Control: Sensorless Field-Oriented Control (FOC)](#1-motor-control-sensorless-field-oriented-control-foc)
2. [Audio Streaming DMA Pipeline: DC-Block + Headroom-Scaled Biquad + Limiter](#2-audio-streaming-dma-pipeline)
3. [Condition Monitoring: Vibration Spectral Analysis & Welch PSD](#3-condition-monitoring-vibration-spectral-analysis)
4. [Multi-rate ADC: High-Rate CIC Decimation & Polyphase FIR](#4-multi-rate-adc-cic-decimation)
5. [Acoustic Feature Extraction & Voice Activity Detection (VAD)](#5-acoustic-feature-extraction--vad)
6. [Zero-Allocation Composable Processing Chains (`DspNode`)](#6-zero-allocation-composable-processing-chains)

---

## 1. Motor Control: Sensorless Field-Oriented Control (FOC)

Execute high-frequency (20–50 kHz) current loop control in pure Q15 or floating-point arithmetic.

```rust
use embedded_dsp::*;

// Setup controller states
let mut id_pid = PidInstanceQ15::new(8000, 2000, 0);   // D-axis flux PID
let mut iq_pid = PidInstanceQ15::new(12000, 3000, 0);  // Q-axis torque PID

// ADC current measurements (Phase A, B, C) in Q15 format
let i_a: q15 = 12000;
let i_b: q15 = -6000;

// 1. Forward Clarke Transform (3-phase abc -> 2-phase stationary αβ)
let mut i_alpha: q15 = 0;
let mut i_beta: q15 = 0;
clarke_q15(i_a, i_b, &mut i_alpha, &mut i_beta);

// 2. Rotor Angle via CORDIC or LUT
let rotor_angle_q15: q15 = 8192; // 45 degrees
let (sin_theta, cos_theta) = cordic_sin_cos_q15(rotor_angle_q15);

// 3. Forward Park Transform (stationary αβ -> rotating dq reference frame)
let mut i_d: q15 = 0;
let mut i_q: q15 = 0;
park_q15(i_alpha, i_beta, sin_theta, cos_theta, &mut i_d, &mut i_q);

// 4. Current Loop PID Regulators
let target_i_d: q15 = 0;      // Zero d-axis current (maximum torque per ampere)
let target_i_q: q15 = 15000;  // Commanded torque
let v_d = pid_q15(&mut id_pid, target_i_d.saturating_sub(i_d));
let v_q = pid_q15(&mut iq_pid, target_i_q.saturating_sub(i_q));

// 5. Inverse Park Transform (rotating dq -> stationary αβ voltage commands)
let mut v_alpha: q15 = 0;
let mut v_beta: q15 = 0;
inv_park_q15(v_d, v_q, sin_theta, cos_theta, &mut v_alpha, &mut v_beta);
```

---

## 2. Audio Streaming DMA Pipeline

Process incoming I2S DMA buffers in-place: remove DC bias, apply parametric EQ without internal register overflow, and clamp outputs.

```rust
use embedded_dsp::*;

// 1. Design Butterworth filter in floating-point
let biquad_float = biquad_lowpass_coeffs(2000.0, 48000.0, core::f32::consts::FRAC_1_SQRT_2);

// 2. Quantize & scale to Q15 with automatic L-infinity peak headroom protection
let mut q15_coeffs = [0i16; 5];
let post_shift = biquad_quantize_and_scale_q15(
    &biquad_float,
    &mut q15_coeffs,
    ScalingStrategy::LInfNorm,
).expect("quantization failed");

// 3. Initialise DMA block processor
let mut biquad_state = [0i16; 4];
let mut biquad = BiquadCascadeInstanceQ15::init(1, &q15_coeffs, &mut biquad_state, post_shift);
let mut dc_blocker = DcBlockerQ15::new(32112); // R ≈ 0.98

// In DMA callback / audio loop
fn process_dma_buffer(
    dma_rx: &[q15],
    dma_tx: &mut [q15],
    dc_blocker: &mut DcBlockerQ15,
    biquad: &mut BiquadCascadeInstanceQ15,
) {
    let mut temp = [0i16; 64];
    let len = dma_rx.len().min(temp.len());

    // Step A: DC Blocking
    for i in 0..len {
        temp[i] = dc_blocker.process_sample(dma_rx[i]);
    }

    // Step B: Biquad Filtering
    biquad_cascade_df1_q15(biquad, &temp[..len], &mut dma_tx[..len]);
}
```

---

## 3. Condition Monitoring: Vibration Spectral Analysis

Extract power spectral density from accelerometer data to detect mechanical bearing or motor imbalance defects.

```rust
use embedded_dsp::*;

const N: usize = 256;
let mut raw_accel = [1000i16; N];
let mut windowed = [0i16; N];
let mut fft_buffer = [0i16; 2 * N];

// 1. Apply Hanning window in Q15
let mut window = [0i16; N];
hanning_window_q15(&mut window);
mult_q15(&raw_accel, &window, &mut windowed);

// 2. Prepare complex buffer
for i in 0..N {
    fft_buffer[2 * i] = windowed[i];
    fft_buffer[2 * i + 1] = 0;
}

// 3. Block Floating-Point FFT (dynamically preserves 30+ dB higher SNR)
let scale_exponent = cfft_bfp_q15(&mut fft_buffer, N, 0, 1);

// 4. Compute power spectrum and locate peak vibration harmonic
let mut peak_bin = 0;
let mut peak_power = 0i64;
for k in 0..(N / 2) {
    let re = fft_buffer[2 * k] as i64;
    let im = fft_buffer[2 * k + 1] as i64;
    let power = re * re + im * im;
    if power > peak_power {
        peak_power = power;
        peak_bin = k;
    }
}
println!("Peak defect harmonic at bin {}, Block scale exponent: {}", peak_bin, scale_exponent);
```

---

## 4. Multi-rate ADC: CIC Decimation

Downsample a high-frequency oversampled 1-bit or 16-bit PDM/ADC input without floating-point arithmetic or hardware multiplications.

```rust
use embedded_dsp::*;

// 3-stage CIC decimator with downsampling factor R = 8
let mut cic = CicDecimator::<3>::new(8);
assert_eq!(cic.gain_bits(), 9); // Automatic bit-growth: 8^3 = 512 => 9 bits

let high_rate_input = [15000i32; 64];
let mut baseband_output = [0i32; 8];
let mut out_idx = 0;

for &sample in &high_rate_input {
    // Normalizes bit-growth automatically to prevent accumulator overflow
    if let Some(decimated) = cic.process_sample_scaled(sample) {
        if out_idx < baseband_output.len() {
            baseband_output[out_idx] = decimated;
            out_idx += 1;
        }
    }
}
```

---

## 5. Acoustic Feature Extraction & VAD

Extract Voice Activity and acoustic features for low-power edge wake-word / anomaly detection.

```rust
use embedded_dsp::*;

let vad = VadDetectorQ15::new(500, 4); // Energy & zero-crossing thresholds
let frame = [2000i16; 128];

if vad.is_active(&frame) {
    let mut mel_energies = [0.0f32; 16];
    let mut mfcc = [0.0f32; 12];
    let frame_f32: [f32; 128] = [0.1; 128];

    let status = mfcc_f32(
        &frame_f32,
        16000.0,
        300.0,
        8000.0,
        &mut mel_energies,
        &mut mfcc,
    );
    assert_eq!(status, Status::Success);
}
```

---

## 6. Zero-Allocation Composable Processing Chains

Chain multiple DSP nodes into a single streaming processor with zero heap allocations.

```rust
use embedded_dsp::pipeline::*;
use embedded_dsp::*;

// Create individual components
let lowpass = SinglePoleFilter::lowpass(0.05);
let limiter = Limiter::new(-0.8f32, 0.8f32);
let gain = Gain::new(1.5f32);

// Compose into a single sequential pipeline: Lowpass -> Gain -> Limiter
let mut dsp_chain = lowpass.then(gain).then(limiter);

// In-place block stream processing (e.g. DMA buffer)
let mut audio_buffer = [0.1f32, 0.5, 0.9, -1.2, 0.3];
dsp_chain.process_in_place(&mut audio_buffer);
```
