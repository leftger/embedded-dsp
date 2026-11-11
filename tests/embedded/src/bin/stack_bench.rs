#![no_std]
#![no_main]

use core::hint::black_box;
use defmt::*;
use embedded_dsp::filtering::{
    Biquad, Deadband, DirectForm2Transposed, Hysteresis, RunningMedian, SavitzkyGolay, SlewRateLimiter,
};
use embedded_dsp::kalman::{UkfModel, UnscentedKalmanFilter};
use embedded_dsp::quaternion::{MadgwickFilter, MahonyFilter};
use embedded_dsp_bench::*;

struct TestUkfModel;
impl UkfModel<2, 1> for TestUkfModel {
    fn f(&self, x: &[f32; 2], dt: f32) -> [f32; 2] {
        [x[0] + x[1] * dt, x[1]]
    }
    fn h(&self, x: &[f32; 2]) -> [f32; 1] {
        [x[0]]
    }
}

#[cortex_m_rt::entry]
fn main() -> ! {
    info!("Starting Embedded-DSP Stack High-Water Mark (HWM) Profiler...");

    const GUARD: usize = 64;
    const WINDOW: usize = 4096;

    // 1. Biquad DF2T steady-state settling and filtering
    let (_, biquad_stack) = run_with_stack_hwm(GUARD, WINDOW, || {
        let biquad = Biquad::<f32>::new(0.05, 0.1, 0.05, 0.8, -0.2);
        let mut state = DirectForm2Transposed::<f32>::new();
        let _ = biquad.settle_to_df2t(&mut state, 1.0);
        let mut y = 0.0f32;
        for i in 0..100 {
            y += biquad.process_df2t(&mut state, black_box(i as f32 * 0.1));
        }
        black_box(y)
    });
    info!("STACK_HWM_BYTES_biquad_df2t = {=usize}", biquad_stack);

    // 2. Signal Conditioning: Deadband, Hysteresis, SlewRateLimiter
    let (_, cond_stack) = run_with_stack_hwm(GUARD, WINDOW, || {
        let deadband = unwrap!(Deadband::<f32>::recentered(0.2f32).ok());
        let mut hyst = unwrap!(Hysteresis::<f32>::new(0.3f32, 0.7f32, false).ok());
        let mut slew = unwrap!(SlewRateLimiter::<f32>::symmetric(10.0f32, 1000.0f32).ok());
        let mut sum = 0.0f32;
        for i in 0..50 {
            let s = deadband.apply(i as f32 * 0.05);
            let h = if hyst.update(s) { 1.0 } else { 0.0 };
            sum += slew.filter(h);
        }
        black_box(sum)
    });
    info!("STACK_HWM_BYTES_conditioning = {=usize}", cond_stack);

    // 3. RunningMedian & SavitzkyGolay
    let (_, sg_median_stack) = run_with_stack_hwm(GUARD, WINDOW, || {
        let mut median = unwrap!(RunningMedian::<5, f32>::new().ok());
        let mut sg = unwrap!(SavitzkyGolay::<7, 3, f32>::latest(0.001f32).ok());
        let mut acc = 0.0f32;
        for i in 0..20 {
            let m = median.filter(black_box(i as f32));
            let s = sg.filter(m);
            acc += s + sg.first_derivative();
        }
        black_box(acc)
    });
    info!("STACK_HWM_BYTES_savitzky_median = {=usize}", sg_median_stack);

    // 4. Attitude Estimation: Mahony & Madgwick
    let (_, attitude_stack) = run_with_stack_hwm(GUARD, WINDOW, || {
        let mut mahony = MahonyFilter::default();
        let mut madgwick = MadgwickFilter::default();
        let gyro = [0.01f32, -0.02, 0.0];
        let accel = [0.0f32, 0.0, 9.81];
        let dt = 0.001f32;
        let mut q_sum = 0.0f32;
        for _ in 0..20 {
            let q1 = mahony.update_6dof(black_box(gyro), black_box(accel), dt);
            let q2 = madgwick.update_6dof(black_box(gyro), black_box(accel), dt);
            q_sum += q1[0] + q2[0];
        }
        black_box(q_sum)
    });
    info!("STACK_HWM_BYTES_attitude = {=usize}", attitude_stack);

    // 5. Unscented Kalman Filter (UKF)
    let (_, ukf_stack) = run_with_stack_hwm(GUARD, WINDOW, || {
        let model = TestUkfModel;
        let mut ukf = UnscentedKalmanFilter::<2, 1>::new(
            [0.0, 1.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.01, 0.0], [0.0, 0.01]],
            [[0.1]],
        );
        for _ in 0..10 {
            let _ = ukf.predict(&model, 0.01);
            let _ = ukf.update(&model, &[1.0]);
        }
        black_box(ukf.x[0])
    });
    info!("STACK_HWM_BYTES_ukf = {=usize}", ukf_stack);

    info!("All stack benchmarks complete.");
    loop {
        cortex_m::asm::wfi();
    }
}
