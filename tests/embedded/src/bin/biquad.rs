#![no_std]
#![no_main]

use defmt::*;
use embedded_dsp::filtering::{
    Biquad, BiquadClamp, BiquadFixed, DirectForm1, DirectForm1NoiseShaped, DirectForm2Transposed,
};
use embedded_dsp::pipeline::Split;
use embedded_dsp_bench::*;

#[cortex_m_rt::entry]
fn main() -> ! {
    info!("Setup hardware cycle counter...");
    let mut c = unwrap!(cortex_m::Peripherals::take());

    c.DCB.enable_trace();
    c.DWT.enable_cycle_counter();

    info!("Starting biquad cycle benchmarking...");
    CyclesResults::header();

    let coeff = Biquad::<f32>::new(0.2, 0.1, 0.05, 0.5, -0.1);
    let clamp = BiquadClamp::new(coeff, -1.0f32, 1.0f32, 0.0f32);

    let mut df1 = Split::new(clamp, DirectForm1::<f32>::new());
    bench_process(&mut df1).show("biquad clamp df1 f32");

    let mut df2t = Split::new(clamp, DirectForm2Transposed::<f32>::new());
    bench_process(&mut df2t).show("biquad clamp df2t f32");

    let ba_fixed = [1 << 30, 0, 0, 0, 0];
    let fixed_filter = BiquadFixed::<30>::new(ba_fixed, -1000, 1000, 0);
    let mut fixed_node = Split::new(fixed_filter, DirectForm1NoiseShaped::new());
    bench_process(&mut fixed_node).show("biquad fixed noise shaped");

    loop {
        cortex_m::asm::wfi();
    }
}
