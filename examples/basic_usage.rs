//! Basic usage example for `embedded-dsp` showing vector math, FIR filtering, PID control, and FFT.

use embedded_dsp::*;

fn main() {
    println!("=== embedded-dsp Basic Usage Example ===");

    // 1. Vector Operations
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let mut vec_out = [0.0f32; 4];
    add_f32(&a, &b, &mut vec_out);
    println!("Vector Add: {:?}", vec_out);

    let dot = dot_prod_f32(&a, &b);
    println!("Vector Dot Product: {}", dot);

    // 2. Q15 Fixed-Point Saturating Math
    let q15_a = [q15::from_bits(20000), q15::from_bits(25000)];
    let q15_b = [q15::from_bits(15000), q15::from_bits(10000)];
    let mut q15_out = [q15::ZERO; 2];
    add_q15(&q15_a, &q15_b, &mut q15_out);
    println!("Q15 Saturating Add (clamped at 32767): {:?}", q15_out);

    // 3. FIR Filtering
    let coeffs = [0.25f32, 0.5, 0.25]; // 3-tap moving average filter
    let mut state = [0.0f32; 3 + 4 - 1];
    let mut fir = FirInstanceF32::init(3, &coeffs, &mut state);

    let input_signal = [1.0f32, 2.0, 3.0, 4.0];
    let mut filtered_signal = [0.0f32; 4];
    fir_f32(&mut fir, &input_signal, &mut filtered_signal);
    println!("FIR Filter Output: {:?}", filtered_signal);

    // 4. PID Motor Controller
    let mut pid = PidInstanceF32::new(2.0, 0.1, 0.05);
    let control_output = pid.process(10.0);
    println!("PID Control Signal: {}", control_output);

    // 5. 64-Point Complex FFT
    let mut fft_data = [0.0f32; 128]; // 64 complex pairs [re, im, ...]
    for i in 0..64 {
        fft_data[2 * i] = (i as f32 * 0.1).sin();
    }
    cfft_f32(&mut fft_data, 64, 0, 1);
    println!("64-Point Complex FFT processed successfully!");

    // 6. 1D Conditional Median Filtering (Impulse / Spike Rejection)
    let spiky_signal = [1.0f32, 1.1, 1.0, 100.0, 1.2, 1.1, 1.0];
    let mut clean_signal = [0.0f32; 7];
    median_filter_1d_f32(&spiky_signal, &mut clean_signal, 3, 5.0);
    println!("Conditional Median Filter Out: {:?}", clean_signal);

    // 7. Welch's Method Power Spectral Density (PSD)
    let mut psd_out = [0.0f32; 32];
    let mut sine_wave = [0.0f32; 128];
    for (i, val) in sine_wave.iter_mut().enumerate() {
        *val = (2.0 * core::f32::consts::PI * 100.0 * (i as f32) / 1000.0).sin();
    }
    welch_psd_f32(
        &sine_wave,
        &mut psd_out,
        64,
        32,
        1000.0,
        WelchWindow::Hamming,
        true,
    );
    println!("Welch PSD (dB) at bin 6: {:.2} dB", psd_out[6]);

    // 8. 2D Spatial Processing (2D DCT & Sobel Edge Detection)
    let img_4x4 = [
        0.0f32, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0,
    ];
    let mut edges = [0.0f32; 16];
    sobel_edge_detection_f32(&img_4x4, &mut edges, 4, 4, 15.0);
    println!("2D Sobel Edge Output (4x4): {:?}", edges);

    // 9. Weighted Polynomial Least-Squares Sensor Calibration
    let x_cal = [0.0f32, 1.0, 2.0, 3.0, 4.0];
    let y_cal = [2.0f32, 5.0, 8.0, 11.0, 14.0]; // y = 2 + 3x
    let mut cal_coeffs = [0.0f32; 2];
    polynomial_least_squares_fit(&x_cal, &y_cal, None, 1, &mut cal_coeffs);
    println!(
        "Fitted Sensor Calibration: y = {:.2} + {:.2}*x",
        cal_coeffs[0], cal_coeffs[1]
    );
}
