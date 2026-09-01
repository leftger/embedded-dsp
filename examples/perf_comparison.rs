use embedded_dsp::*;
use std::time::Instant;

fn main() {
    println!(
        "========================================================================================="
    );
    println!(
        "                  embedded-dsp vs libm Performance Benchmark Comparison                  "
    );
    println!(
        "========================================================================================="
    );
    println!();

    const ITERATIONS: usize = 1_000_000;
    const VECTOR_SIZE: usize = 256;

    // -----------------------------------------------------------------------------------------
    // 1. Scalar Trigonometric Sine Performance (1,000,000 operations)
    // -----------------------------------------------------------------------------------------
    println!(
        "--- 1. Scalar Math Operations ({} iterations) ---",
        ITERATIONS
    );

    // libm::sinf
    let start = Instant::now();
    let mut sum_libm = 0.0f32;
    for i in 0..ITERATIONS {
        let x = (i % 360) as f32 * (std::f32::consts::PI / 180.0);
        sum_libm += libm::sinf(x);
    }
    let duration_libm_sin = start.elapsed();

    // embedded-dsp f32 sin
    let start = Instant::now();
    let mut sum_dsp_f32 = 0.0f32;
    for i in 0..ITERATIONS {
        let x = (i % 360) as f32 * (std::f32::consts::PI / 180.0);
        sum_dsp_f32 += sin_f32(x);
    }
    let duration_dsp_f32_sin = start.elapsed();

    // embedded-dsp q31 sin (Fixed Point Integer Math)
    let start = Instant::now();
    let mut sum_dsp_q31 = q31::ZERO;
    for i in 0..ITERATIONS {
        let theta_q31 = q31::from_bits(((i % 360) as i64 * 2147483648 / 360) as i32);
        sum_dsp_q31 = sum_dsp_q31.wrapping_add(sin_q31(theta_q31));
    }
    let duration_dsp_q31_sin = start.elapsed();

    println!(
        "  • libm::sinf         : {:>9.2?}  (sum check: {:.2})",
        duration_libm_sin, sum_libm
    );
    println!(
        "  • embedded-dsp f32   : {:>9.2?}  (sum check: {:.2})",
        duration_dsp_f32_sin, sum_dsp_f32
    );
    println!(
        "  • embedded-dsp q31   : {:>9.2?}  (sum check: {})",
        duration_dsp_q31_sin, sum_dsp_q31
    );
    println!();

    // -----------------------------------------------------------------------------------------
    // 2. Square Root Performance (1,000,000 operations)
    // -----------------------------------------------------------------------------------------
    println!(
        "--- 2. Square Root Operations ({} iterations) ---",
        ITERATIONS
    );

    // libm::sqrtf
    let start = Instant::now();
    let mut sum_libm_sqrt = 0.0f32;
    for i in 1..=ITERATIONS {
        sum_libm_sqrt += libm::sqrtf(i as f32);
    }
    let duration_libm_sqrt = start.elapsed();

    // embedded-dsp sqrt_f32
    let start = Instant::now();
    let mut sum_dsp_sqrt_f32 = 0.0f32;
    let mut out_f32 = 0.0f32;
    for i in 1..=ITERATIONS {
        sqrt_f32(i as f32, &mut out_f32);
        sum_dsp_sqrt_f32 += out_f32;
    }
    let duration_dsp_sqrt_f32 = start.elapsed();

    // embedded-dsp sqrt_q31 (Fixed-point)
    let start = Instant::now();
    let mut sum_dsp_sqrt_q31 = q31::ZERO;
    let mut out_q31 = q31::ZERO;
    for i in 1..=ITERATIONS {
        let val_q31 = q31::from_bits((i as i64 * 2147483647 / ITERATIONS as i64) as i32);
        sqrt_q31(val_q31, &mut out_q31);
        sum_dsp_sqrt_q31 = sum_dsp_sqrt_q31.wrapping_add(out_q31);
    }
    let duration_dsp_sqrt_q31 = start.elapsed();

    println!(
        "  • libm::sqrtf        : {:>9.2?}  (sum check: {:.2})",
        duration_libm_sqrt, sum_libm_sqrt
    );
    println!(
        "  • embedded-dsp f32   : {:>9.2?}  (sum check: {:.2})",
        duration_dsp_sqrt_f32, sum_dsp_sqrt_f32
    );
    println!(
        "  • embedded-dsp q31   : {:>9.2?}  (sum check: {})",
        duration_dsp_sqrt_q31, sum_dsp_sqrt_q31
    );
    println!();

    // -----------------------------------------------------------------------------------------
    // 3. DSP Vector Dot Product (Vector size: 256, 100,000 iterations)
    // -----------------------------------------------------------------------------------------
    const VECTOR_ITERATIONS: usize = 100_000;
    println!(
        "--- 3. Vector Dot Product (Size: {}, {} iterations) ---",
        VECTOR_SIZE, VECTOR_ITERATIONS
    );

    let src_a_f32 = vec![1.5f32; VECTOR_SIZE];
    let src_b_f32 = vec![2.5f32; VECTOR_SIZE];

    let src_a_q31 = vec![q31::from_bits(1000000); VECTOR_SIZE];
    let src_b_q31 = vec![q31::from_bits(2000000); VECTOR_SIZE];

    let src_a_q15 = vec![q15::from_bits(1000); VECTOR_SIZE];
    let src_b_q15 = vec![q15::from_bits(2000); VECTOR_SIZE];

    // f32 Dot Product
    let start = Instant::now();
    let mut dot_f32_sum = 0.0f32;
    for _ in 0..VECTOR_ITERATIONS {
        dot_f32_sum += dot_prod_f32(&src_a_f32, &src_b_f32);
    }
    let duration_dot_f32 = start.elapsed();

    // q31 Dot Product
    let start = Instant::now();
    let mut dot_q31_sum = 0i64;
    for _ in 0..VECTOR_ITERATIONS {
        dot_q31_sum = dot_q31_sum.wrapping_add(dot_prod_q31(&src_a_q31, &src_b_q31));
    }
    let duration_dot_q31 = start.elapsed();

    // q15 Dot Product
    let start = Instant::now();
    let mut dot_q15_sum = 0i64;
    for _ in 0..VECTOR_ITERATIONS {
        dot_q15_sum = dot_q15_sum.wrapping_add(dot_prod_q15(&src_a_q15, &src_b_q15));
    }
    let duration_dot_q15 = start.elapsed();

    println!(
        "  • dot_prod_f32       : {:>9.2?}  (sum check: {:.2})",
        duration_dot_f32, dot_f32_sum
    );
    println!(
        "  • dot_prod_q31       : {:>9.2?}  (sum check: {}, speedup vs f32: {:.2}x)",
        duration_dot_q31,
        dot_q31_sum,
        duration_dot_f32.as_secs_f64() / duration_dot_q31.as_secs_f64()
    );
    println!(
        "  • dot_prod_q15       : {:>9.2?}  (sum check: {}, speedup vs f32: {:.2}x)",
        duration_dot_q15,
        dot_q15_sum,
        duration_dot_f32.as_secs_f64() / duration_dot_q15.as_secs_f64()
    );
    println!();

    // -----------------------------------------------------------------------------------------
    // 4. 256-Point Complex FFT (10,000 iterations)
    // -----------------------------------------------------------------------------------------
    const FFT_ITERATIONS: usize = 10_000;
    println!(
        "--- 4. 256-Point Complex FFT ({} iterations) ---",
        FFT_ITERATIONS
    );

    let mut fft_data_f32 = vec![0.0f32; 512]; // 256 complex pairs
    let mut fft_data_q31 = vec![q31::ZERO; 512];
    let mut fft_data_q15 = vec![q15::ZERO; 512];

    for i in 0..256 {
        fft_data_f32[2 * i] = (i as f32).sin();
        fft_data_q31[2 * i] = q31::from_bits((i * 1000) as i32);
        fft_data_q15[2 * i] = q15::from_bits((i * 100) as i16);
    }

    // cfft_f32
    let start = Instant::now();
    for _ in 0..FFT_ITERATIONS {
        cfft_f32(&mut fft_data_f32, 256, 0, 1);
    }
    let duration_cfft_f32 = start.elapsed();

    // cfft_q31
    let start = Instant::now();
    for _ in 0..FFT_ITERATIONS {
        cfft_q31(&mut fft_data_q31, 256, 0, 1);
    }
    let duration_cfft_q31 = start.elapsed();

    // cfft_q15
    let start = Instant::now();
    for _ in 0..FFT_ITERATIONS {
        cfft_q15(&mut fft_data_q15, 256, 0, 1);
    }
    let duration_cfft_q15 = start.elapsed();

    println!("  • cfft_f32 (256-pt)  : {:>9.2?}", duration_cfft_f32);
    println!("  • cfft_q31 (256-pt)  : {:>9.2?}", duration_cfft_q31);
    println!("  • cfft_q15 (256-pt)  : {:>9.2?}", duration_cfft_q15);
    println!();
    println!(
        "========================================================================================="
    );
}
