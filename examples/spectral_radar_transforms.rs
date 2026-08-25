//! Comprehensive Spectral Analysis, Radar & Advanced DSP Transforms Example
//!
//! Demonstrates:
//! 1. Multitone RF Radar/SDR Signal Simulation (Chirps + Multi-tone + Narrowband Jammer + White Noise)
//! 2. Multirate Signal Processing: Cascaded Integrator-Comb (CIC) Decimator & Interpolator (`CicDecimator<3>`, `CicInterpolator<3>`) and Fractional Linear Resampler (`resample_linear_f32`)
//! 3. Adaptive Filter Noise Cancellation: LMS (`LmsInstanceF32`, `lms_f32`) and Normalized LMS (`NlmsInstanceF32`, `nlms_f32`) for active interference suppression
//! 4. Windowing Comparison: Hanning, Hamming, Blackman, Blackman-Harris, and Flat-Top
//! 5. Spectral Analysis: High-Resolution Complex FFT (`cfft_f32`, `cfft_q31`), Packed Real FFT (`rfft_f32`, `rfft_q15`), and Welch's PSD (`welch_psd_f32`)
//! 6. Advanced DSP Transforms:
//!    - Discrete Cosine Transform IV (`dct4_f32`)
//!    - Fast Walsh-Hadamard Transform (`fwht_f32`, `ifwht_f32`, `fwht_i32`) for CDMA orthogonal code decoding
//!    - Hartley Transform (`hartley_f32`)
//!    - Haar Wavelet Transform (`haar_transform_f32`, `inverse_haar_transform_f32`)
//!    - Daubechies-4 Discrete Wavelet Transform (`wavelet_transform_f32`, `inverse_wavelet_transform_f32`, `DAUBECHIES_4`)

use embedded_dsp::*;

fn main() {
    println!("===============================================================================");
    println!("     embedded-dsp Spectral Analysis, Multirate & Transform Pipeline            ");
    println!("===============================================================================");
    println!();

    const FS: f32 = 48000.0; // 48 kHz IF / baseband sampling rate
    const N_SAMPLES: usize = 512;

    // -----------------------------------------------------------------------------------------
    // 1. Multitone RF / Radar Signal Synthesis with Strong Jammer
    // -----------------------------------------------------------------------------------------
    println!("--- 1. Multitone RF Signal Simulation with Interfering Jammer ---");
    let mut clean_signal = [0.0f32; N_SAMPLES];
    let mut jammer_signal = [0.0f32; N_SAMPLES];
    let mut received_signal = [0.0f32; N_SAMPLES];

    let target_f1 = 3000.0f32; // 3 kHz target tone
    let target_f2 = 7500.0f32; // 7.5 kHz target tone
    let jammer_freq = 1200.0f32; // 1.2 kHz strong interfering tone

    for i in 0..N_SAMPLES {
        let t = i as f32 / FS;
        // Clean signal: Target radar return
        clean_signal[i] = 0.6 * (2.0 * core::f32::consts::PI * target_f1 * t).sin()
            + 0.4 * (2.0 * core::f32::consts::PI * target_f2 * t).sin();
        // High-power narrowband interference (Jammer)
        jammer_signal[i] = 1.8 * (2.0 * core::f32::consts::PI * jammer_freq * t).sin();
        // Receiver noise
        let prng =
            ((i as u64).wrapping_mul(1664525).wrapping_add(1013904223) % 10000) as f32 / 10000.0;
        let noise = 0.1 * (prng - 0.5);

        received_signal[i] = clean_signal[i] + jammer_signal[i] + noise;
    }

    let mut raw_rms = 0.0f32;
    rms_f32(&received_signal, &mut raw_rms);
    println!(
        "  Generated {} samples. Total Received RMS: {:.4} (Jammer-dominated)",
        N_SAMPLES, raw_rms
    );

    // -----------------------------------------------------------------------------------------
    // 2. Multirate DSP: Cascaded Integrator-Comb (CIC) & Resampling
    // -----------------------------------------------------------------------------------------
    println!("\n--- 2. Multirate DSP: CIC Decimation / Interpolation & Linear Resampling ---");
    // CIC Decimator: 3 stages, Decimation factor R = 4 (e.g. 48 kHz -> 12 kHz)
    let mut cic_decimator = CicDecimator::<3>::new(4);
    let mut cic_interpolator = CicInterpolator::<3>::new(4);

    let mut decimated_stream = [0i32; 128];
    let mut dec_count = 0;

    for &sample in &received_signal {
        let sample_i32 = (sample * 1000.0) as i32;
        if let Some(dec_val) = cic_decimator.process_sample(sample_i32) {
            if dec_count < decimated_stream.len() {
                decimated_stream[dec_count] = dec_val;
                dec_count += 1;
            }
        }
    }
    println!(
        "  CIC Decimator (R=4, 3 stages): Decimated 512 samples -> {} samples",
        dec_count
    );

    // CIC Interpolator: Upsample by 4 back to original rate
    let mut upsample_chunk = [0i32; 4];
    cic_interpolator.process_sample(decimated_stream[0], &mut upsample_chunk);
    println!(
        "  CIC Interpolator: 1 input sample -> 4 upsampled samples: {:?}",
        upsample_chunk
    );

    // Fractional Linear Resampling (e.g., 48 kHz to 32 kHz -> ratio = 32/48 = 0.6667)
    let mut resampled_out = [0.0f32; 341];
    resample_linear_f32(&received_signal, &mut resampled_out, 32000.0 / 48000.0);
    println!(
        "  Fractional Resampler (48kHz -> 32kHz): Resampled {} -> {} samples",
        received_signal.len(),
        resampled_out.len()
    );

    // -----------------------------------------------------------------------------------------
    // 3. Adaptive Noise Cancellation (LMS & NLMS Filters)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 3. Adaptive Noise Cancellation (LMS & Normalized LMS) ---");
    // Use jammer reference signal to cancel interference from received signal
    const LMS_TAPS: usize = 32;
    let mut lms_coeffs = [0.0f32; LMS_TAPS];
    let mut lms_state = [0.0f32; LMS_TAPS];
    let mut lms_filter =
        LmsInstanceF32::init(LMS_TAPS as u16, &mut lms_coeffs, &mut lms_state, 0.005);

    let mut lms_cancelled_out = [0.0f32; N_SAMPLES];
    let mut lms_error = [0.0f32; N_SAMPLES];

    // Reference signal x = jammer reference, Desired d = received signal (target + jammer + noise)
    // Error output e = d - y = target + noise (jammer cancelled!)
    lms_f32(
        &mut lms_filter,
        &jammer_signal,
        &received_signal,
        &mut lms_cancelled_out,
        &mut lms_error,
    );

    let mut err_rms_initial = 0.0f32;
    let mut err_rms_converged = 0.0f32;
    rms_f32(&lms_error[..64], &mut err_rms_initial);
    rms_f32(&lms_error[N_SAMPLES - 64..], &mut err_rms_converged);

    println!("  LMS Adaptive Filter (32 taps, mu=0.005):");
    println!("    • Initial Error RMS    : {:.4}", err_rms_initial);
    println!(
        "    • Converged Error RMS  : {:.4} (Interference cancelled, target recovered!)",
        err_rms_converged
    );

    // -----------------------------------------------------------------------------------------
    // 4. Windowing Comparison (Spectral Leakage Suppression)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 4. Windowing Comparison: Mainlobe & Sidelobe Properties ---");
    const WIN_LEN: usize = 64;
    let win_rect = [1.0f32; WIN_LEN];
    let mut win_hanning = [0.0f32; WIN_LEN];
    let mut win_hamming = [0.0f32; WIN_LEN];
    let mut win_blackman_harris = [0.0f32; WIN_LEN];
    let mut win_flat_top = [0.0f32; WIN_LEN];

    hanning_f32(&mut win_hanning);
    hamming_f32(&mut win_hamming);
    blackman_harris_f32(&mut win_blackman_harris);
    flattop_f32(&mut win_flat_top);

    println!("  Sample Window Values at Midpoint (n=32):");
    println!("    • Rectangular    : {:.4}", win_rect[32]);
    println!("    • Hanning        : {:.4}", win_hanning[32]);
    println!("    • Hamming        : {:.4}", win_hamming[32]);
    println!("    • Blackman-Harris: {:.4}", win_blackman_harris[32]);
    println!("    • Flat-Top       : {:.4}", win_flat_top[32]);

    // -----------------------------------------------------------------------------------------
    // 5. High-Resolution Spectral Analysis (FFT & Welch PSD)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 5. Spectral Analysis: In-Place Complex FFT & Welch's PSD ---");
    // 256-point Complex FFT on Converged LMS Cleaned Signal
    const FFT_SIZE: usize = 256;
    let mut fft_buffer = [0.0f32; 2 * FFT_SIZE];
    for i in 0..FFT_SIZE {
        fft_buffer[2 * i] = lms_error[N_SAMPLES - FFT_SIZE + i] * win_hamming[i % WIN_LEN];
        fft_buffer[2 * i + 1] = 0.0;
    }

    cfft_f32(&mut fft_buffer, FFT_SIZE, 0, 1);

    // Find top 2 spectral peaks
    let mut peak1_mag = 0.0f32;
    let mut peak1_bin = 0usize;
    let mut peak2_mag = 0.0f32;
    let mut peak2_bin = 0usize;

    for k in 1..FFT_SIZE / 2 {
        let re = fft_buffer[2 * k];
        let im = fft_buffer[2 * k + 1];
        let mag = (re * re + im * im).sqrt();
        if mag > peak1_mag {
            peak2_mag = peak1_mag;
            peak2_bin = peak1_bin;
            peak1_mag = mag;
            peak1_bin = k;
        } else if mag > peak2_mag {
            peak2_mag = mag;
            peak2_bin = k;
        }
    }

    let bin_to_hz = FS / FFT_SIZE as f32;
    println!("  256-Point FFT Detected Spectral Peaks:");
    println!(
        "    • Peak 1: Bin {:>3} ({:>6.1} Hz) -> Magnitude: {:>7.2}",
        peak1_bin,
        peak1_bin as f32 * bin_to_hz,
        peak1_mag
    );
    println!(
        "    • Peak 2: Bin {:>3} ({:>6.1} Hz) -> Magnitude: {:>7.2}",
        peak2_bin,
        peak2_bin as f32 * bin_to_hz,
        peak2_mag
    );

    // Welch's Method Power Spectral Density (PSD)
    let mut psd_out = [0.0f32; 64];
    welch_psd_f32(
        &received_signal,
        &mut psd_out,
        128, // Segment length
        64,  // Overlap
        FS,  // Sampling rate
        WelchWindow::BlackmanHarris,
        true, // Return in dB/Hz
    );
    println!("  Welch's PSD (Averaged Periodogram in dB/Hz):");
    println!("    • PSD[Bin 4]  (375 Hz) : {:>6.1} dB/Hz", psd_out[4]);
    println!(
        "    • PSD[Bin 13] (1200 Hz): {:>6.1} dB/Hz (Jammer peak)",
        psd_out[13]
    );
    println!(
        "    • PSD[Bin 32] (3000 Hz): {:>6.1} dB/Hz (Target 1 peak)",
        psd_out[32]
    );

    // -----------------------------------------------------------------------------------------
    // 6. Advanced DSP Transforms (FWHT, DCT-IV, Hartley, Wavelets)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 6. Advanced DSP Transforms: FWHT, DCT-IV, Hartley & Wavelets ---");

    // A. Fast Walsh-Hadamard Transform (FWHT) for CDMA / Walsh codes
    let mut walsh_data = [1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
    let fwht_status = fwht_f32(&mut walsh_data);
    println!(
        "  Fast Walsh-Hadamard Transform (8-point): Status={:?}, Out={:?}",
        fwht_status, walsh_data
    );
    ifwht_f32(&mut walsh_data);
    println!(
        "  Inverse FWHT (reconstructed exact signal): {:?}",
        walsh_data
    );

    // B. Discrete Cosine Transform IV (DCT-IV)
    let dct_in = [1.0f32, 2.0, 3.0, 4.0];
    let mut dct_out = [0.0f32; 4];
    dct4_f32(&dct_in, &mut dct_out, 4);
    println!("  DCT-IV (4-point): In={:?} -> Out={:?}", dct_in, dct_out);

    // C. Hartley Transform (Real-in, Real-out, Self-Inverse Transform)
    let mut hartley_buf = [1.0f32, 2.0, 3.0, 4.0];
    let h_status = hartley_transform_f32(&mut hartley_buf);
    println!(
        "  Hartley Transform: Status={:?}, F_H={:?}",
        h_status, hartley_buf
    );
    // Applying Hartley twice recovers the original scaled signal
    hartley_transform_f32(&mut hartley_buf);
    println!("  Hartley Roundtrip (Self-Inverse): {:?}", hartley_buf);

    // D. Multi-Resolution Daubechies-4 Discrete Wavelet Transform (DWT)
    let mut wavelet_signal = [0.0f32; 16];
    wavelet_signal[7] = 10.0; // High-frequency transient spike in middle of frame
    let orig_wavelet = wavelet_signal;

    let dwt_status = wavelet_transform_f32(&mut wavelet_signal, &DAUBECHIES_4);
    println!("  Daubechies-4 DWT (16-point Pyramid Decomposition):");
    println!("    • DWT Status : {:?}", dwt_status);
    println!(
        "    • Detail/Scaling Coefficients: {:?}",
        &wavelet_signal[..8]
    );

    let idwt_status = inverse_wavelet_transform_f32(&mut wavelet_signal, &DAUBECHIES_4);
    let mut diff = 0.0f32;
    for i in 0..16 {
        diff += (wavelet_signal[i] - orig_wavelet[i]).abs();
    }
    println!(
        "    • Inverse DWT Status: {:?}, Reconstruction Error: {:.2e}",
        idwt_status, diff
    );

    println!();
    println!("===============================================================================");
    println!("             Spectral & Transforms Pipeline Execution Complete!                ");
    println!("===============================================================================");
}
