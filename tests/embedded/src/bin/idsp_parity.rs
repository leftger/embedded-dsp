#![no_std]
#![no_main]

//! Bare-metal cycle-count benchmarks for the integer DSP primitives shared with
//! `idsp`: `cossin`, `atan2`, the type-2 order-3 integer PLL, and the
//! fixed-point biquad noise-shaper.
//!
//! Run on a Cortex-M7 (e.g. STM32H7) for numbers directly comparable to
//! `idsp`'s published `tests/embedded` results (cossin ~23.5 cycles,
//! atan2 ~52 cycles).

use core::hint::black_box;

use defmt::*;
use embedded_dsp::fast_math::{atan2_i32, cossin};
use embedded_dsp::filtering::{BiquadFixed, DirectForm1NoiseShaped, NormalForm, NormalFormState};
use embedded_dsp::pll::{IntPll, IntPllState};
use embedded_dsp::resampling::HbfDecCascade;
use embedded_dsp_bench::timeit;

#[cortex_m_rt::entry]
fn main() -> ! {
    info!("Setup hardware cycle counter...");
    let mut c = unwrap!(cortex_m::Peripherals::take());

    c.DCB.enable_trace();
    c.DWT.enable_cycle_counter();

    info!("Starting idsp-parity integer cycle benchmarking...");

    // cossin: i32 phase -> (cos, sin)
    let phase = black_box(0x1234_5678i32);
    let cycles = timeit(|| {
        black_box(cossin(black_box(phase)));
    });
    info!("cossin i32: {} cycles/call", cycles);

    // atan2: (y, x) -> i32 phase
    let y = black_box(0x1234_5678i32);
    let x = black_box(0x2345_6789i32);
    let cycles = timeit(|| {
        black_box(atan2_i32(black_box(y), black_box(x)));
    });
    info!("atan2_i32: {} cycles/call", cycles);

    // Type-2 order-3 integer PLL
    let pll = IntPll::from_bandwidth(1e-3, 4.0);
    let mut pll_state = IntPllState::default();
    let mut input = black_box(0x0010_0000i32);
    let cycles = timeit(|| {
        input = input.wrapping_add(black_box(0x0010_0000));
        black_box(pll.process(black_box(&mut pll_state), black_box(input)));
    });
    info!("IntPll::process: {} cycles/call", cycles);

    // Fixed-point biquad with first-order noise shaping
    let biquad = BiquadFixed::<30>::new(
        [1 << 28, 1 << 27, 1 << 28, (1.5 * (1i64 << 30) as f32) as i32, -(1 << 29)],
        -(1 << 30),
        1 << 30,
        0,
    );
    let mut biquad_state = DirectForm1NoiseShaped::new();
    let mut sample = black_box(0i32);
    let cycles = timeit(|| {
        sample = sample.wrapping_add(black_box(10_000));
        black_box(biquad.process_noise_shaped(black_box(&mut biquad_state), black_box(sample)));
    });
    info!("BiquadFixed noise-shaped: {} cycles/call", cycles);

    // Normal-form quadrature oscillator
    let nco = NormalForm::oscillator(0.1);
    let mut nco_state = NormalFormState::default();
    let cycles = timeit(|| {
        black_box(nco.process_quadrature(black_box(&mut nco_state), black_box(0.0)));
    });
    info!("NormalForm oscillator: {} cycles/call", cycles);

    // Half-band decimation cascade (rate 8)
    let mut hbf = HbfDecCascade::<3>::new();
    let src = black_box([1.0f32; 8]);
    let mut dst = [0.0f32; 1];
    let cycles = timeit(|| {
        hbf.process(black_box(&src), black_box(&mut dst));
    });
    info!("HbfDecCascade<3> (8 samples -> 1): {} cycles/8 samples", cycles);

    loop {
        cortex_m::asm::wfi();
    }
}
