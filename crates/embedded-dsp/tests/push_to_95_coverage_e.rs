use embedded_dsp::*;

#[test]
fn test_square_root_kalman_filter_exhaustive() {
    let x0 = [0.0f32, 0.0];
    let s0 = [[1.0f32, 0.0], [0.0, 1.0]];
    let f = [[1.0f32, 1.0], [0.0, 1.0]];
    let s_q = [[0.1f32, 0.0], [0.0, 0.1]];
    let h = [[1.0f32, 0.0]];
    let s_r = [[0.5f32]];

    let mut sr_kf = SquareRootKalmanFilter::<2, 1>::new(x0, s0, f, s_q, h, s_r);
    sr_kf.predict();
    let status = sr_kf.update(&[1.0f32]);
    assert_eq!(status, Status::Success);

    let cov = sr_kf.covariance();
    assert!(cov[0][0] > 0.0);
}

#[test]
fn test_psd_error_branches_and_db() {
    let src = [1.0f32; 128];
    let mut dst = [0.0f32; 32];

    // Invalid arguments for welch_psd_f32
    assert_eq!(
        welch_psd_f32(
            &src,
            &mut dst,
            3,
            0,
            1000.0,
            WelchWindow::Rectangular,
            false
        ),
        Status::ArgumentError
    );
    assert_eq!(
        welch_psd_f32(
            &src,
            &mut dst,
            64,
            64,
            1000.0,
            WelchWindow::Rectangular,
            false
        ),
        Status::ArgumentError
    );
    assert_eq!(
        welch_psd_f32(
            &src,
            &mut dst,
            64,
            16,
            -100.0,
            WelchWindow::Rectangular,
            false
        ),
        Status::ArgumentError
    );
    assert_eq!(
        welch_psd_f32(
            &src[..10],
            &mut dst,
            64,
            16,
            1000.0,
            WelchWindow::Rectangular,
            false
        ),
        Status::LengthError
    );

    // ar_burg_f32 invalid arguments
    let mut ar_coeffs = [0.0f32; 4];
    assert_eq!(
        ar_burg_f32(&src[..4], 4, &mut ar_coeffs),
        Err(Status::ArgumentError)
    );
    assert_eq!(
        ar_burg_f32(&src, 0, &mut ar_coeffs),
        Err(Status::ArgumentError)
    );

    // ar_psd_f32 db and linear
    if let Ok(noise_var) = ar_burg_f32(&src[..32], 4, &mut ar_coeffs) {
        assert_eq!(
            ar_psd_f32(&ar_coeffs, noise_var, 32, &mut dst, true),
            Status::Success
        );
        assert_eq!(
            ar_psd_f32(&ar_coeffs, noise_var, 32, &mut dst, false),
            Status::Success
        );
    }
}

#[test]
fn test_quantization_and_scaling_strategies() {
    let sos_f32 = [0.1f32, 0.2, 0.3, 0.4, 0.5];
    let mut q15_out = [q15::ZERO; 5];
    let mut q31_out = [q31::ZERO; 5];

    assert!(
        biquad_quantize_and_scale_q15(&sos_f32, &mut q15_out, ScalingStrategy::LInfNorm).is_ok()
    );
    assert!(biquad_quantize_and_scale_q15(&sos_f32, &mut q15_out, ScalingStrategy::L2Norm).is_ok());
    assert!(biquad_quantize_and_scale_q15(&sos_f32, &mut q15_out, ScalingStrategy::Direct).is_ok());

    assert!(
        biquad_quantize_and_scale_q31(&sos_f32, &mut q31_out, ScalingStrategy::LInfNorm).is_ok()
    );
    assert!(biquad_quantize_and_scale_q31(&sos_f32, &mut q31_out, ScalingStrategy::L2Norm).is_ok());
    assert!(biquad_quantize_and_scale_q31(&sos_f32, &mut q31_out, ScalingStrategy::Direct).is_ok());

    let taps_f32 = [0.1f32, 0.2, 0.3, 0.4];
    let mut taps_q15 = [q15::ZERO; 4];
    assert!(fir_quantize_q15(&taps_f32, &mut taps_q15).is_ok());

    // Error length tests
    let mut bad_q15 = [q15::ZERO; 4];
    assert_eq!(
        biquad_quantize_and_scale_q15(&sos_f32, &mut bad_q15, ScalingStrategy::Direct),
        Err(Status::LengthError)
    );
    assert_eq!(
        fir_quantize_q15(&taps_f32, &mut bad_q15[..2]),
        Err(Status::LengthError)
    );
}
