//! Streaming automatic-gain-control tests.

use embedded_dsp::dynamics::AgcF32;
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
