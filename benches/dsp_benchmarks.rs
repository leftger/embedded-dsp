//! Embedded DSP performance benchmarks for filter kernels, transforms, and SIMD operations.

use embedded_dsp::*;
use std::time::Instant;

fn bench_dot_prod_q15() {
    let a = [1234i16; 1024];
    let b = [5678i16; 1024];

    let start = Instant::now();
    let iterations = 10_000;
    let mut sum: q63 = 0;
    for _ in 0..iterations {
        sum = sum.wrapping_add(dot_prod_q15(&a, &b));
    }
    let elapsed = start.elapsed();
    let throughput_mops = (iterations as f64 * 1024.0) / elapsed.as_secs_f64() / 1e6;
    println!(
        "dot_prod_q15 (SIMD/SWAR): {:.2} MOps/s ({:?} for {} iterations, sum={})",
        throughput_mops, elapsed, iterations, sum
    );
}

fn bench_cfft() {
    const N: usize = 256;
    let mut data_f32 = [0.5f32; 2 * N];
    let mut data_q15 = [16384i16; 2 * N];
    let mut data_bfp = [16384i16; 2 * N];

    let iterations = 1_000;

    let start_f = Instant::now();
    for _ in 0..iterations {
        cfft_f32(&mut data_f32, N, 0, 1);
    }
    let elapsed_f = start_f.elapsed();

    let start_q = Instant::now();
    for _ in 0..iterations {
        cfft_q15(&mut data_q15, N, 0, 1);
    }
    let elapsed_q = start_q.elapsed();

    let start_bfp = Instant::now();
    let mut total_exp = 0;
    for _ in 0..iterations {
        total_exp += cfft_bfp_q15(&mut data_bfp, N, 0, 1);
    }
    let elapsed_bfp = start_bfp.elapsed();

    println!(
        "256-pt CFFT f32:      {:.2} us/transform ({:?})",
        elapsed_f.as_secs_f64() * 1e6 / iterations as f64,
        elapsed_f
    );
    println!(
        "256-pt CFFT Q15:      {:.2} us/transform ({:?})",
        elapsed_q.as_secs_f64() * 1e6 / iterations as f64,
        elapsed_q
    );
    println!(
        "256-pt CFFT BFP Q15:  {:.2} us/transform ({:?}, scale={})",
        elapsed_bfp.as_secs_f64() * 1e6 / iterations as f64,
        elapsed_bfp,
        total_exp
    );
}

fn bench_fir_q15() {
    const TAPS: usize = 32;
    const SAMPLES: usize = 512;
    let coeffs = [1000i16; TAPS];
    let mut state = [0i16; TAPS];
    let src = [2000i16; SAMPLES];
    let mut dst = [0i16; SAMPLES];

    let mut fir = FirInstanceQ15::init(TAPS as u16, &coeffs, &mut state);

    let iterations = 2_000;
    let start = Instant::now();
    for _ in 0..iterations {
        fir_q15(&mut fir, &src, &mut dst);
    }
    let elapsed = start.elapsed();
    let samples_per_sec = (iterations as f64 * SAMPLES as f64) / elapsed.as_secs_f64();
    println!(
        "FIR 32-tap Q15:       {:.2} MSamples/sec ({:?})",
        samples_per_sec / 1e6,
        elapsed
    );
}

fn bench_cordic_vs_lut() {
    let angles = [1000i16, 5000, 15000, 25000, -10000, -20000];
    let iterations = 10_000;

    let start_cordic = Instant::now();
    let mut sum_c = 0i32;
    for _ in 0..iterations {
        for &a in &angles {
            let (s, c) = cordic_sin_cos_q15(a);
            sum_c = sum_c.wrapping_add(s as i32 + c as i32);
        }
    }
    let elapsed_cordic = start_cordic.elapsed();

    let start_lut = Instant::now();
    let mut sum_l = 0i32;
    for _ in 0..iterations {
        for &a in &angles {
            let rad = a as f32 / 32768.0;
            let s = fast_sin_i16(rad);
            let c = fast_cos_i16(rad);
            sum_l = sum_l.wrapping_add(s as i32 + c as i32);
        }
    }
    let elapsed_lut = start_lut.elapsed();

    println!(
        "CORDIC sin/cos Q15:   {:.2} MCalls/s ({:?}, sum={})",
        (iterations * angles.len()) as f64 / elapsed_cordic.as_secs_f64() / 1e6,
        elapsed_cordic,
        sum_c
    );
    println!(
        "LUT sin/cos Q15:      {:.2} MCalls/s ({:?}, sum={})",
        (iterations * angles.len()) as f64 / elapsed_lut.as_secs_f64() / 1e6,
        elapsed_lut,
        sum_l
    );
}

fn bench_mult_q31() {
    let a = [1_234_567_890i32; 1024];
    let b = [-987_654_321i32; 1024];

    let iterations = 20_000;
    let start = Instant::now();
    let mut sum: q31 = 0;
    for _ in 0..iterations {
        for i in 0..a.len() {
            sum = sum.wrapping_add(q31_mult(a[i], b[i]));
        }
    }
    let elapsed = start.elapsed();
    let throughput_mops = (iterations as f64 * a.len() as f64) / elapsed.as_secs_f64() / 1e6;
    println!(
        "q31_mult:             {:.2} MOps/s ({:?} for {} iterations, sum={})",
        throughput_mops, elapsed, iterations, sum
    );
}

fn bench_pid_q31() {
    let mut pid = PidInstanceQ31::new(i32::MAX / 4, i32::MAX / 20, i32::MAX / 100);

    let iterations = 200_000;
    let start = Instant::now();
    let mut sum: i64 = 0;
    for i in 0..iterations {
        let in_val = ((i % 1000) as i32).wrapping_mul(1_000_000);
        sum = sum.wrapping_add(pid.process(in_val) as i64);
    }
    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "PidInstanceQ31::process: {:.2} MOps/s ({:?} for {} iterations, sum={})",
        ops_per_sec / 1e6,
        elapsed,
        iterations,
        sum
    );
}

fn main() {
    println!("=== embedded-dsp Performance Benchmarks ===\n");
    bench_dot_prod_q15();
    bench_fir_q15();
    bench_cfft();
    bench_cordic_vs_lut();
    bench_mult_q31();
    bench_pid_q31();
    println!("\n=== Benchmark Complete ===");
}
