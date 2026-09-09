use embedded_dsp::*;

#[test]
fn test_snapshot_buffer_push_and_reset() {
    let mut snap = SnapshotBuffer::<4>::new();
    let empty: &[f32] = &[];
    assert_eq!(snap.samples(), empty);
    assert!(!snap.is_full());

    assert!(snap.push(1.0));
    assert!(snap.push(2.0));
    assert!(snap.push(3.0));
    assert!(snap.push(4.0));
    assert!(snap.is_full());
    assert!(!snap.push(5.0)); // Overflow rejected

    assert_eq!(snap.samples(), &[1.0, 2.0, 3.0, 4.0]);

    snap.reset();
    assert_eq!(snap.samples(), empty);
    assert!(!snap.is_full());
}

#[test]
fn test_impulse_response_analyzer() {
    let response = [1.0f32, 0.5, 0.25, 0.1, 0.02, 0.005, 0.0001];
    let info = analyze_impulse_response(&response, 0.01);

    assert_eq!(info.peak_gain, 1.0);
    assert!(info.is_stable);
    assert!(info.settling_time_samples <= 6);
    assert!(info.total_energy > 0.0);

    let empty_info = analyze_impulse_response(&[], 0.01);
    assert_eq!(empty_info.peak_gain, 0.0);
    assert!(empty_info.is_stable);
}
