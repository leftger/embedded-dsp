use embedded_dsp::*;

#[test]
fn test_differential_evaluation_exact_match() {
    let reference = [1.0f32, -0.5, 0.25, -0.125, 0.0];
    let target = [1.0f32, -0.5, 0.25, -0.125, 0.0];

    let metrics = evaluate_differential(&reference, &target);
    assert!(metrics.is_exact_match);
    assert_eq!(metrics.peak_absolute_error, 0.0);
    assert_eq!(metrics.rms_error, 0.0);
    assert!(metrics.sqnr_db >= 139.0);

    let mut diff = [0.0f32; 5];
    phase_invert_sum(&reference, &target, &mut diff);
    for &d in &diff {
        assert_eq!(d, 0.0);
    }
}

#[test]
fn test_differential_evaluation_with_noise() {
    let reference = [1.0f32, 0.8, 0.6, 0.4, 0.2];
    let target = [1.01f32, 0.79, 0.61, 0.39, 0.21];

    let metrics = evaluate_differential(&reference, &target);
    assert!(!metrics.is_exact_match);
    assert!(metrics.peak_absolute_error > 0.0);
    assert!(metrics.rms_error > 0.0);
    assert!(metrics.sqnr_db > 20.0);

    let empty_metrics = evaluate_differential(&[], &[]);
    assert!(empty_metrics.is_exact_match);
}
