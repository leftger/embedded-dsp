//! Embedded DSP performance benchmarks for filter kernels, transforms, and SIMD operations.
//!
//! Every measurement is recorded as a *throughput* metric (higher is better), so
//! results can be checked against a committed baseline for regressions:
//!
//! ```text
//! cargo bench -p embedded-dsp --bench dsp_benchmarks -- --write-baseline baseline.txt
//! cargo bench -p embedded-dsp --bench dsp_benchmarks -- --check-baseline baseline.txt
//! ```
//!
//! `--check-baseline` exits non-zero if any metric is more than `--max-regression`
//! (default `2.0`) times slower than the baseline.

use embedded_dsp::*;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

/// Collects benchmark results keyed by a stable name.
#[derive(Default)]
struct Recorder {
    values: BTreeMap<&'static str, f64>,
}

impl Recorder {
    fn record(&mut self, name: &'static str, throughput: f64) {
        self.values.insert(name, throughput);
    }

    fn write_baseline(&self, path: &str) -> std::io::Result<()> {
        let mut out =
            String::from("# embedded-dsp benchmark baseline (throughput, higher is better)\n");
        out.push_str("# name\tvalue\n");
        for (name, value) in &self.values {
            out.push_str(&format!("{name}\t{value:.4}\n"));
        }
        std::fs::write(path, out)
    }

    /// Returns `true` if every metric is within `max_regression` of the baseline.
    fn check_baseline(&self, path: &str, max_regression: f64) -> std::io::Result<bool> {
        let text = std::fs::read_to_string(path)?;
        let mut baseline = BTreeMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((name, value)) = line.split_once('\t')
                && let Ok(value) = value.trim().parse::<f64>()
            {
                baseline.insert(name.to_owned(), value);
            }
        }

        let mut ok = true;
        for (name, current) in &self.values {
            match baseline.get(*name) {
                Some(&reference) if reference > 0.0 => {
                    let ratio = current / reference;
                    if ratio < 1.0 / max_regression {
                        ok = false;
                        println!(
                            "REGRESSION {name}: {current:.2} vs baseline {reference:.2} ({:.2}x slower)",
                            1.0 / ratio
                        );
                    } else {
                        println!("ok {name}: {current:.2} ({ratio:.2}x baseline)");
                    }
                }
                Some(_) => {}
                None => println!("no baseline for {name} (new benchmark)"),
            }
        }
        for name in baseline.keys() {
            if !self.values.contains_key(name.as_str()) {
                println!("baseline has {name} but it was not measured");
            }
        }
        Ok(ok)
    }
}

fn bench_dot_prod_q15(rec: &mut Recorder) {
    let a = [q15::from_bits(1234); 1024];
    let b = [q15::from_bits(5678); 1024];

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
    rec.record("dot_prod_q15", throughput_mops);
}

fn bench_cfft(rec: &mut Recorder) {
    const N: usize = 256;
    let mut data_f32 = [0.5f32; 2 * N];
    let mut data_q15 = [q15::from_bits(16384); 2 * N];
    let mut data_bfp = [q15::from_bits(16384); 2 * N];

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
    rec.record("cfft_f32", iterations as f64 / elapsed_f.as_secs_f64());
    rec.record("cfft_q15", iterations as f64 / elapsed_q.as_secs_f64());
    rec.record(
        "cfft_bfp_q15",
        iterations as f64 / elapsed_bfp.as_secs_f64(),
    );
}

fn bench_fir_q15(rec: &mut Recorder) {
    const TAPS: usize = 32;
    const SAMPLES: usize = 512;
    let coeffs = [q15::from_bits(1000); TAPS];
    let mut state = [q15::ZERO; TAPS];
    let src = [q15::from_bits(2000); SAMPLES];
    let mut dst = [q15::ZERO; SAMPLES];

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
    rec.record("fir_q15", samples_per_sec);
}

fn bench_cordic_vs_lut(rec: &mut Recorder) {
    let angles = [
        q15::from_bits(1000),
        q15::from_bits(5000),
        q15::from_bits(15000),
        q15::from_bits(25000),
        q15::from_bits(-10000),
        q15::from_bits(-20000),
    ];
    let iterations = 10_000;

    let start_cordic = Instant::now();
    let mut sum_c = 0i32;
    for _ in 0..iterations {
        for &a in &angles {
            let (s, c) = cordic_sin_cos_q15(black_box(a));
            sum_c = sum_c.wrapping_add(s.to_bits() as i32 + c.to_bits() as i32);
        }
    }
    let elapsed_cordic = start_cordic.elapsed();

    let start_lut = Instant::now();
    let mut sum_l = 0i32;
    for _ in 0..iterations {
        for &a in &angles {
            let rad = black_box(a.to_bits() as f32 / 32768.0);
            let s = fast_sin_i16(rad);
            let c = fast_cos_i16(rad);
            sum_l = sum_l.wrapping_add(s as i32 + c as i32);
        }
    }
    let elapsed_lut = start_lut.elapsed();

    let calls = (iterations * angles.len()) as f64;
    println!(
        "CORDIC sin/cos Q15:   {:.2} MCalls/s ({:?}, sum={})",
        calls / elapsed_cordic.as_secs_f64() / 1e6,
        elapsed_cordic,
        sum_c
    );
    println!(
        "LUT sin/cos Q15:      {:.2} MCalls/s ({:?}, sum={})",
        calls / elapsed_lut.as_secs_f64() / 1e6,
        elapsed_lut,
        sum_l
    );
    rec.record("cordic_sin_cos_q15", calls / elapsed_cordic.as_secs_f64());
    rec.record("lut_sin_cos_q15", calls / elapsed_lut.as_secs_f64());
}

fn bench_mult_q31(rec: &mut Recorder) {
    let a = [q31::from_bits(1_234_567_890); 1024];
    let b = [q31::from_bits(-987_654_321); 1024];

    let iterations = 20_000;
    let start = Instant::now();
    let mut sum: q31 = q31::ZERO;
    for _ in 0..iterations {
        for i in 0..a.len() {
            sum = sum.wrapping_add(q31_mult(black_box(a[i]), black_box(b[i])));
        }
    }
    let elapsed = start.elapsed();
    let throughput_mops = (iterations as f64 * a.len() as f64) / elapsed.as_secs_f64() / 1e6;
    println!(
        "q31_mult:             {:.2} MOps/s ({:?} for {} iterations, sum={})",
        throughput_mops, elapsed, iterations, sum
    );
    rec.record("q31_mult", throughput_mops);
}

fn bench_pid_q31(rec: &mut Recorder) {
    let mut pid = PidInstanceQ31::new(
        q31::from_bits(i32::MAX / 4),
        q31::from_bits(i32::MAX / 20),
        q31::from_bits(i32::MAX / 100),
    );

    let iterations = 200_000;
    let start = Instant::now();
    let mut sum: i64 = 0;
    for i in 0..iterations {
        let in_val = q31::from_bits((i % 1000i32).wrapping_mul(1_000_000));
        sum = sum.wrapping_add(pid.process(in_val).to_bits() as i64);
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
    rec.record("pid_q31", ops_per_sec);
}

fn bench_idsp_parity_ops(rec: &mut Recorder) {
    use embedded_dsp::fast_math::atan2_i32;
    use embedded_dsp::filtering::{BiquadFixed, DirectForm1NoiseShaped};
    use embedded_dsp::pll::{IntPll, IntPllState};

    let iterations: usize = 200_000;

    // cossin (128-entry midpoint LUT, comparable to idsp's ~23.5 cycles on M7)
    let mut phase = 0i32;
    let start = Instant::now();
    let mut sum = 0i64;
    for _ in 0..iterations {
        phase = phase.wrapping_add(0x0100_0000);
        let (c, s) = cossin(phase);
        sum = sum.wrapping_add(c as i64 + s as i64);
    }
    let elapsed = start.elapsed();
    let cossin_rate = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "cossin i32:            {:.2} MCalls/s ({:?}, sum={})",
        cossin_rate / 1e6,
        elapsed,
        sum
    );
    rec.record("cossin_i32", cossin_rate);

    // atan2_i32 (comparable to idsp's ~52 cycles on M7)
    let start = Instant::now();
    let mut sum = 0i64;
    for i in 1..=iterations {
        let a = (i as i32).wrapping_mul(31).rotate_left(7);
        let b = (i as i32).wrapping_mul(17).rotate_right(3);
        sum = sum.wrapping_add(atan2_i32(a, b) as i64);
    }
    let elapsed = start.elapsed();
    let atan2_rate = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "atan2_i32:             {:.2} MCalls/s ({:?}, sum={})",
        atan2_rate / 1e6,
        elapsed,
        sum
    );
    rec.record("atan2_i32", atan2_rate);

    // IntPll (type-2 order-3 integer PLL)
    let pll = IntPll::from_bandwidth(1e-3, 4.0);
    let mut pll_state = IntPllState::default();
    let start = Instant::now();
    let mut acc = 0i32;
    let mut sum = 0i64;
    for _ in 0..iterations {
        acc = acc.wrapping_add(0x0010_0000);
        sum = sum.wrapping_add(pll.process(&mut pll_state, acc) as i64);
    }
    let elapsed = start.elapsed();
    let pll_rate = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "IntPll::process:       {:.2} MCalls/s ({:?}, sum={})",
        pll_rate / 1e6,
        elapsed,
        sum
    );
    rec.record("int_pll", pll_rate);

    // Fixed-point biquad with noise shaping (idsp's i32 biquad does ~8.5 cyc)
    let bq = BiquadFixed::<30>::new(
        [1_000_000, 2_000_000, 1_000_000, 1_500_000_000, -500_000_000],
        -1 << 30,
        1 << 30,
        0,
    );
    let mut bq_state = DirectForm1NoiseShaped::new();
    let start = Instant::now();
    let mut sum = 0i64;
    let mut x = 0i32;
    for _ in 0..iterations {
        x = x.wrapping_add(10_000);
        sum = sum.wrapping_add(bq.process_noise_shaped(&mut bq_state, x) as i64);
    }
    let elapsed = start.elapsed();
    let bq_rate = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "BiquadFixed process:   {:.2} MCalls/s ({:?}, sum={})",
        bq_rate / 1e6,
        elapsed,
        sum
    );
    rec.record("biquad_fixed_noise_shaped", bq_rate);
}

fn main() {
    let mut write_baseline: Option<String> = None;
    let mut check_baseline: Option<String> = None;
    let mut max_regression = 2.0f64;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--write-baseline" if i + 1 < args.len() => {
                i += 1;
                write_baseline = Some(args[i].clone());
            }
            "--check-baseline" if i + 1 < args.len() => {
                i += 1;
                check_baseline = Some(args[i].clone());
            }
            "--max-regression" if i + 1 < args.len() => {
                i += 1;
                max_regression = args[i].parse().unwrap_or(2.0);
            }
            other => eprintln!("ignoring unknown argument: {other}"),
        }
        i += 1;
    }

    println!("=== embedded-dsp Performance Benchmarks ===\n");
    let mut rec = Recorder::default();
    bench_dot_prod_q15(&mut rec);
    bench_fir_q15(&mut rec);
    bench_cfft(&mut rec);
    bench_cordic_vs_lut(&mut rec);
    bench_mult_q31(&mut rec);
    bench_pid_q31(&mut rec);
    bench_idsp_parity_ops(&mut rec);
    println!("\n=== Benchmark Complete ===");

    if let Some(path) = &write_baseline {
        match rec.write_baseline(path) {
            Ok(()) => println!("\nwrote baseline to {path}"),
            Err(err) => eprintln!("\nfailed to write baseline {path}: {err}"),
        }
    }

    if let Some(path) = &check_baseline {
        println!("\n=== Baseline check ({path}, max regression {max_regression}x) ===");
        match rec.check_baseline(path, max_regression) {
            Ok(true) => println!("baseline check passed"),
            Ok(false) => {
                eprintln!("baseline check FAILED");
                std::process::exit(1);
            }
            Err(err) => eprintln!("failed to read baseline {path}: {err}"),
        }
    }
}
