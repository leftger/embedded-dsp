#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use embedded_dsp::filter_design::{biquad_lowpass_coeffs, harmonic_notch_coeffs};
use embedded_dsp::filtering::{
    Biquad, BiquadCascadeDf2tInstance, BiquadCascadeInstance, DeadbandF32, DeadbandF64,
    DirectForm1, DirectForm2Transposed, HysteresisF32, HysteresisF64, RunningMedianF32,
    SavitzkyGolayF32, SlewRateLimiterF32, SlewRateLimiterF64, biquad_cascade_df1_checked,
    biquad_cascade_df1_settle_f32, biquad_cascade_df2t_checked, biquad_cascade_df2t_settle_f32,
};
use embedded_dsp::kalman::{UkfModel, UnscentedKalmanFilter};
use embedded_dsp::quaternion::{MadgwickFilter, MahonyFilter};
use embedded_dsp::types::Status;

#[test]
fn test_biquad_settle_to_df2t() {
    let coeffs = biquad_lowpass_coeffs(100.0, 1000.0, core::f32::consts::FRAC_1_SQRT_2);
    let biquad = Biquad::<f32>::new(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]);
    let mut state = DirectForm2Transposed::<f32>::new();

    // DC steady state for 5.0
    let steady = biquad.settle_to_df2t(&mut state, 5.0).unwrap();
    assert!((steady - 5.0).abs() < 1e-4);

    // After settle, feeding 5.0 must immediately return 5.0 with zero transient jump
    let y0 = biquad.process_df2t(&mut state, 5.0);
    assert!((y0 - 5.0).abs() < 1e-4);

    // DirectForm2Transposed convenience method
    let mut state2 = DirectForm2Transposed::<f32>::new();
    let s2 = state2.settle(&biquad, 2.5).unwrap();
    assert!((s2 - 2.5).abs() < 1e-4);

    // Error on NaN/Inf
    assert_eq!(
        biquad.settle_to_df2t(&mut state, f32::NAN),
        Err(Status::NanInf)
    );
    assert_eq!(
        biquad.settle_to_df2t(&mut state, f32::INFINITY),
        Err(Status::NanInf)
    );
}

#[test]
fn test_biquad_settle_to_df1() {
    let coeffs = biquad_lowpass_coeffs(100.0, 1000.0, core::f32::consts::FRAC_1_SQRT_2);
    let biquad = Biquad::<f32>::new(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]);
    let mut state = DirectForm1::<f32>::new();

    let steady = biquad.settle_to_df1(&mut state, 3.0).unwrap();
    assert!((steady - 3.0).abs() < 1e-4);

    let y0 = biquad.process_df1(&mut state, 3.0);
    assert!((y0 - 3.0).abs() < 1e-4);

    let mut state2 = DirectForm1::<f32>::new();
    let s2 = state2.settle(&biquad, 1.5).unwrap();
    assert!((s2 - 1.5).abs() < 1e-4);

    assert_eq!(
        biquad.settle_to_df1(&mut state, f32::NAN),
        Err(Status::NanInf)
    );
}

#[test]
fn test_biquad_f64_settle() {
    let biquad = Biquad::<f64>::new(0.1, 0.2, 0.1, 0.5, -0.1);
    let mut df2t = DirectForm2Transposed::<f64>::new();
    let mut df1 = DirectForm1::<f64>::new();

    let s_df2t = biquad.settle_to_df2t(&mut df2t, 2.0).unwrap();
    let s_df1 = biquad.settle_to_df1(&mut df1, 2.0).unwrap();
    assert!((s_df2t - s_df1).abs() < 1e-9);

    assert_eq!(
        biquad.settle_to_df2t(&mut df2t, f64::NAN),
        Err(Status::NanInf)
    );
    assert_eq!(
        biquad.settle_to_df1(&mut df1, f64::INFINITY),
        Err(Status::NanInf)
    );
}

#[test]
fn test_biquad_process_checked() {
    let coeffs = biquad_lowpass_coeffs(100.0, 1000.0, core::f32::consts::FRAC_1_SQRT_2);
    let biquad = Biquad::<f32>::new(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]);
    let mut df1 = DirectForm1::<f32>::new();
    let mut df2t = DirectForm2Transposed::<f32>::new();

    assert!(biquad.process_df1_checked(&mut df1, 1.0).is_ok());
    assert!(biquad.process_df2t_checked(&mut df2t, 1.0).is_ok());

    let state_before_df1 = df1;
    let state_before_df2t = df2t;

    // Checked methods must return NanInf and leave states unmodified
    assert_eq!(
        biquad.process_df1_checked(&mut df1, f32::NAN),
        Err(Status::NanInf)
    );
    assert_eq!(df1, state_before_df1);

    assert_eq!(
        biquad.process_df2t_checked(&mut df2t, f32::INFINITY),
        Err(Status::NanInf)
    );
    assert_eq!(df2t, state_before_df2t);

    // f64 variants
    let b64 = Biquad::<f64>::new(0.1, 0.2, 0.1, 0.5, -0.1);
    let mut df1_64 = DirectForm1::<f64>::new();
    let mut df2t_64 = DirectForm2Transposed::<f64>::new();
    assert_eq!(
        b64.process_df1_checked(&mut df1_64, f64::NAN),
        Err(Status::NanInf)
    );
    assert_eq!(
        b64.process_df2t_checked(&mut df2t_64, f64::INFINITY),
        Err(Status::NanInf)
    );
}

#[test]
fn test_biquad_cascade_settle_and_checked() {
    let lp = biquad_lowpass_coeffs(200.0, 1000.0, core::f32::consts::FRAC_1_SQRT_2);
    let mut coeffs = [0.0f32; 10];
    coeffs[..5].copy_from_slice(&lp);
    coeffs[5..10].copy_from_slice(&lp);

    let mut state_df1 = [0.0f32; 8];
    let mut cascade_df1 = BiquadCascadeInstance::init(2, &coeffs, &mut state_df1);
    let out_df1 = biquad_cascade_df1_settle_f32(&mut cascade_df1, 4.0).unwrap();
    assert!((out_df1 - 4.0).abs() < 1e-3);

    let mut state_df2t = [0.0f32; 4];
    let mut cascade_df2t = BiquadCascadeDf2tInstance::init(2, &coeffs, &mut state_df2t);
    let out_df2t = biquad_cascade_df2t_settle_f32(&mut cascade_df2t, 4.0).unwrap();
    assert!((out_df2t - 4.0).abs() < 1e-3);

    let src = [1.0, 2.0, 3.0, 4.0];
    let mut dst = [0.0; 4];
    assert!(biquad_cascade_df1_checked(&mut cascade_df1, &src, &mut dst).is_ok());
    assert!(biquad_cascade_df2t_checked(&mut cascade_df2t, &src, &mut dst).is_ok());

    let bad_src = [1.0, f32::NAN, 3.0];
    assert_eq!(
        biquad_cascade_df1_checked(&mut cascade_df1, &bad_src, &mut dst),
        Err(Status::NanInf)
    );
    assert_eq!(
        biquad_cascade_df2t_checked(&mut cascade_df2t, &bad_src, &mut dst),
        Err(Status::NanInf)
    );
}

#[test]
fn test_harmonic_notch_coeffs() {
    let (_notches, count) = harmonic_notch_coeffs::<4>(50.0, 1000.0, 5.0);
    assert_eq!(count, 4); // 50, 100, 150, 200 Hz are all below 500 Hz Nyquist

    // If fundamental is high, harmonics exceeding Nyquist should be bounded
    let (notches_high, count_high) = harmonic_notch_coeffs::<4>(200.0, 1000.0, 5.0);
    assert_eq!(count_high, 2); // 200, 400 Hz < 500 Hz; 600, 800 Hz exceed Nyquist
    assert_eq!(notches_high[2], [1.0, 0.0, 0.0, 0.0, 0.0]); // pass-through
}

#[test]
fn test_deadband() {
    let plain = DeadbandF32::plain(0.2).unwrap();
    assert_eq!(plain.apply(0.1), 0.0);
    assert_eq!(plain.apply(-0.15), 0.0);
    assert_eq!(plain.apply(0.5), 0.5);
    assert_eq!(plain.apply(-0.5), -0.5);

    let recentered = DeadbandF32::recentered(0.2).unwrap();
    assert_eq!(recentered.apply(0.1), 0.0);
    assert!((recentered.apply(0.5) - 0.3).abs() < 1e-6);
    assert!((recentered.apply(-0.5) - -0.3).abs() < 1e-6);

    assert_eq!(recentered.apply_checked(f32::NAN), Err(Status::NanInf));
    assert!(recentered.apply_checked(0.5).is_ok());

    // Error on negative threshold
    assert_eq!(DeadbandF32::plain(-1.0), Err(Status::ArgumentError));

    // f64
    let plain64 = DeadbandF64::plain(0.5).unwrap();
    assert_eq!(plain64.apply(0.3), 0.0);
    assert_eq!(plain64.apply(1.0), 1.0);
    assert_eq!(DeadbandF64::plain(-0.1), Err(Status::ArgumentError));
}

#[test]
fn test_hysteresis() {
    let mut hyst = HysteresisF32::new(0.2, 0.8, false).unwrap();
    assert!(!hyst.state());

    // Below high threshold -> remains false
    assert!(!hyst.update(0.7));
    // Above high threshold -> transitions to true
    assert!(hyst.update(0.85));
    // In deadband between low and high -> remains true
    assert!(hyst.update(0.5));
    // Above low threshold -> remains true
    assert!(hyst.update(0.25));
    // Below low threshold -> transitions to false
    assert!(!hyst.update(0.15));

    // NaN ignores update and holds state
    assert!(!hyst.update(f32::NAN));
    hyst.reset(true);
    assert!(hyst.state());
    assert!(hyst.update(f32::NAN));

    assert_eq!(hyst.update_checked(f32::NAN), Err(Status::NanInf));

    // Invalid bounds
    assert_eq!(
        HysteresisF32::new(0.8, 0.2, false),
        Err(Status::ArgumentError)
    );

    // f64
    let mut h64 = HysteresisF64::new(1.0, 2.0, false).unwrap();
    assert!(h64.update(2.5));
    assert!(h64.update(1.5));
    assert!(!h64.update(0.5));
}

#[test]
fn test_slew_rate_limiter() {
    // 10 units/sec at 1000 Hz = 0.01 per sample max step
    let mut slew = SlewRateLimiterF32::symmetric(10.0, 1000.0).unwrap();

    // First sample seeds immediately
    assert_eq!(slew.filter(5.0), 5.0);
    assert_eq!(slew.value(), 5.0);

    // Target jumps to 10.0 -> clamped to 5.0 + 0.01 = 5.01
    assert!((slew.filter(10.0) - 5.01).abs() < 1e-6);

    // Target drops to 0.0 -> clamped to 5.01 - 0.01 = 5.00
    assert!((slew.filter(0.0) - 5.00).abs() < 1e-6);

    // Small step below rate limit reaches target directly
    assert!((slew.filter(5.005) - 5.005).abs() < 1e-6);

    // Reset clears state
    slew.reset();
    assert_eq!(slew.filter(-2.0), -2.0);

    assert_eq!(slew.filter_checked(f32::NAN), Err(Status::NanInf));

    // f64 asymmetric
    let mut s64 = SlewRateLimiterF64::asymmetric(20.0, 10.0, 1000.0).unwrap();
    assert_eq!(s64.filter(0.0), 0.0);
    assert!((s64.filter(1.0) - 0.02).abs() < 1e-9);
    assert!((s64.filter(0.0) - 0.01).abs() < 1e-9);
}

#[test]
fn test_running_median() {
    // Even window length rejected
    assert_eq!(RunningMedianF32::<4>::new(), Err(Status::ArgumentError));

    let mut median = RunningMedianF32::<5>::new().unwrap();

    // Seeded with first sample
    assert_eq!(median.filter(10.0), 10.0);

    // Input sequence: [10, 10, 10, 10, 10] initially
    // Feed 12, 11, 100 (outlier spike), 13
    median.filter(12.0);
    median.filter(11.0);
    let m_outlier = median.filter(1000.0); // outlier spike
    // Window contents: [11, 12, 1000, 10, 10] -> sorted: [10, 10, 11, 12, 1000] -> median is 11
    assert_eq!(m_outlier, 11.0);

    // NaN input is rejected and median holds
    assert_eq!(median.filter(f32::NAN), 11.0);
    assert_eq!(median.filter_checked(f32::NAN), Err(Status::NanInf));

    median.reset();
    assert_eq!(median.filter(3.0), 3.0);
}

#[test]
fn test_savitzky_golay() {
    // Test polynomial reconstruction: y(t) = 3*t^2 + 2*t + 1
    // At t, y' = 6*t + 2, y'' = 6
    let dt = 0.01f32;
    let mut sg = SavitzkyGolayF32::<7, 3>::latest(dt).unwrap();
    assert_eq!(sg.delay(), 0.0);

    let mut time = 0.0f32;
    for step in 0..50 {
        time = (step as f32) * dt;
        let y = 3.0 * time * time + 2.0 * time + 1.0;
        sg.filter(y);
    }

    let expected_val = 3.0 * time * time + 2.0 * time + 1.0;
    let expected_d1 = 6.0 * time + 2.0;
    let expected_d2 = 6.0;

    assert!((sg.value() - expected_val).abs() < 1e-2);
    assert!((sg.first_derivative() - expected_d1).abs() < 1e-2);
    assert!((sg.second_derivative() - expected_d2).abs() < 5e-2);

    // Centered mode
    let mut sg_centered = SavitzkyGolayF32::<7, 3>::centered(dt).unwrap();
    assert!((sg_centered.delay() - 3.0 * dt).abs() < 1e-6);

    for step in 0..50 {
        let t = (step as f32) * dt;
        let y = 3.0 * t * t + 2.0 * t + 1.0;
        sg_centered.filter(y);
    }
    // Centered evaluates at time - 3*dt
    let t_eval = time - 3.0 * dt;
    let expected_center_val = 3.0 * t_eval * t_eval + 2.0 * t_eval + 1.0;
    assert!((sg_centered.value() - expected_center_val).abs() < 1e-2);

    assert_eq!(sg.filter_checked(f32::NAN), Err(Status::NanInf));

    // Even window length rejected in centered mode
    assert_eq!(
        SavitzkyGolayF32::<6, 3>::centered(dt),
        Err(Status::ArgumentError)
    );
}

#[test]
fn test_mahony_filter() {
    let mut mahony = MahonyFilter::default();
    assert_eq!(mahony.q, [1.0, 0.0, 0.0, 0.0]);

    // Stationary on Earth: gravity = [0, 0, 9.81], gyro = [0, 0, 0]
    let accel = [0.0, 0.0, 9.81];
    let gyro = [0.0, 0.0, 0.0];
    let dt = 0.01;

    for _ in 0..50 {
        mahony.update_6dof(gyro, accel, dt);
    }

    // Should remain identity quaternion [1, 0, 0, 0]
    assert!((mahony.q[0] - 1.0).abs() < 1e-3);
    assert!(mahony.q[1].abs() < 1e-3);
    assert!(mahony.q[2].abs() < 1e-3);
    assert!(mahony.q[3].abs() < 1e-3);

    // 9-DoF update with magnetometer pointing north [1, 0, 0]
    let mag = [1.0, 0.0, 0.0];
    for _ in 0..50 {
        mahony.update_9dof(gyro, accel, mag, dt);
    }
    assert!((mahony.q[0] - 1.0).abs() < 1e-3);

    mahony.reset();
    assert_eq!(mahony.q, [1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_madgwick_filter() {
    let mut madgwick = MadgwickFilter::default();
    assert_eq!(madgwick.q, [1.0, 0.0, 0.0, 0.0]);

    let accel = [0.0, 0.0, 9.81];
    let gyro = [0.0, 0.0, 0.0];
    let dt = 0.01;

    for _ in 0..50 {
        madgwick.update_6dof(gyro, accel, dt);
    }

    assert!((madgwick.q[0] - 1.0).abs() < 1e-3);
    assert!(madgwick.q[1].abs() < 1e-3);
    assert!(madgwick.q[2].abs() < 1e-3);
    assert!(madgwick.q[3].abs() < 1e-3);

    // 9-DoF update
    let mag = [1.0, 0.0, 0.0];
    for _ in 0..50 {
        madgwick.update_9dof(gyro, accel, mag, dt);
    }
    assert!((madgwick.q[0] - 1.0).abs() < 1e-3);

    madgwick.reset();
    assert_eq!(madgwick.q, [1.0, 0.0, 0.0, 0.0]);
}

struct UkfConstantVelocityModel;
impl UkfModel<2, 1> for UkfConstantVelocityModel {
    fn f(&self, x: &[f32; 2], dt: f32) -> [f32; 2] {
        // [pos + vel*dt, vel]
        [x[0] + x[1] * dt, x[1]]
    }
    fn h(&self, x: &[f32; 2]) -> [f32; 1] {
        // Measure position only
        [x[0]]
    }
}

#[test]
fn test_unscented_kalman_filter() {
    let model = UkfConstantVelocityModel;
    let mut ukf = UnscentedKalmanFilter::<2, 1>::with_parameters(
        [0.0, 0.0],                 // initial guess: pos=0, vel=0
        [[10.0, 0.0], [0.0, 10.0]], // initial covariance
        [[0.01, 0.0], [0.0, 0.01]], // process noise
        [[0.1]],                    // measurement noise
        1.0,                        // alpha = 1.0 for f32 numerical stability
        0.0,                        // beta
        0.0,                        // kappa
    );

    let dt = 0.1f32;
    let true_vel = 2.0f32;
    let mut true_pos = 0.0f32;

    // Simulate 40 steps of motion with noisy position measurements
    for _step in 1..=40 {
        true_pos += true_vel * dt;
        let meas = [true_pos];

        assert_eq!(ukf.predict(&model, dt), Status::Success);
        assert_eq!(ukf.update(&model, &meas), Status::Success);
    }

    #[cfg(feature = "std")]
    std::println!(
        "ukf.x = {:?}, true_pos = {}, true_vel = {}",
        ukf.x,
        true_pos,
        true_vel
    );

    // UKF should successfully track velocity and position
    assert!((ukf.x[0] - true_pos).abs() < 0.5);
    assert!((ukf.x[1] - true_vel).abs() < 0.2);

    // Covariance should have contracted significantly
    assert!(ukf.p[0][0] < 1.0);
    assert!(ukf.p[1][1] < 1.0);
}
