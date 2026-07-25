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
    let q15_a = [20000i16, 25000];
    let q15_b = [15000i16, 10000];
    let mut q15_out = [0i16; 2];
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
}
