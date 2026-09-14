//! Streaming automatic-gain-control tests.

use embedded_dsp::dynamics::{AgcF32, NoiseGate, SafetyLimiter};
use embedded_dsp::pipeline::DspNode;

#[test]
fn agc_drives_unit_power_and_lock() {
    let mut agc = AgcF32::new(AgcF32::DEFAULT_BANDWIDTH);
    assert!((agc.bandwidth() - 0.01).abs() < 1e-9);
    agc.set_scale(1.0);
    let mut y = 0.0;
    for _ in 0..8000 {
        y = agc.process(0.1);
    }
    assert!((y.abs() - 1.0).abs() < 0.05);
    assert!(agc.rssi_db().abs() < 1.0);

    agc.lock();
    assert!(agc.is_locked());
    let g = agc.gain();
    agc.process(10.0);
    assert!((agc.gain() - g).abs() < 1e-6);

    agc.unlock();
    agc.reset();
    assert!(!agc.is_locked());
    assert!((agc.gain() - 1.0).abs() < 1e-6);

    let mut node = AgcF32::new(0.2);
    let _ = node.process_sample(0.5);
    node.set_bandwidth(2.0);
    assert!((node.bandwidth() - 1.0).abs() < 1e-6);
}

#[test]
fn agc_gain_clamps_at_the_upper_rail_for_near_silent_input() {
    // A tiny, non-zero input drives an unlocked AGC's gain up toward the 1e6 ceiling.
    let mut agc = AgcF32::new(0.5);
    let mut g = 0.0f32;
    for _ in 0..200_000 {
        agc.process(1e-9);
        g = agc.gain();
    }
    assert!(g <= 1e6, "gain must clamp at the 1e6 ceiling, got {g}");
}

#[test]
fn noise_gate_reports_the_noise_floor_for_near_silent_input() {
    // Below the 1e-6 amplitude threshold, `process` uses the -120 dB floor rather than log10(0).
    let mut gate = NoiseGate::new(-45.0, -40.0, 0.002, 0.05, 48_000.0);
    for _ in 0..10_000 {
        gate.process(0.0);
    }
    // Fully closed: output attenuated by ~40 dB relative to a (silent) input of 0.
    assert_eq!(gate.process(0.0), 0.0);
}

#[test]
fn safety_limiter_recovers_toward_unity_gain_for_near_silent_input() {
    // Below the 1e-6 amplitude threshold, `process` still runs the release recovery branch.
    let mut limiter = SafetyLimiter::new(0.95, 0.05, 48_000.0);
    let mut y = 0.0f32;
    for _ in 0..10_000 {
        y = limiter.process(0.0);
    }
    assert_eq!(y, 0.0);
}
