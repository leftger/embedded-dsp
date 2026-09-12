use embedded_dsp::*;

#[test]
fn test_complex_math_q31_q15_all() {
    let a_q31 = [q31::from_bits(10000), q31::from_bits(20000)];
    let b_q31 = [q31::from_bits(5000), q31::from_bits(10000)];
    let mut out_q31 = [q31::ZERO; 2];

    cmplx_add_q31(&a_q31, &b_q31, &mut out_q31);
    cmplx_sub_q31(&a_q31, &b_q31, &mut out_q31);
    cmplx_mult_cmplx_q31(&a_q31, &b_q31, &mut out_q31);
    cmplx_mult_real_q31(&a_q31, &b_q31, &mut out_q31);
    cmplx_conj_q31(&a_q31, &mut out_q31);

    let mut mag_q31 = [q31::ZERO; 1];
    cmplx_mag_q31(&a_q31, &mut mag_q31);
    cmplx_mag_squared_q31(&a_q31, &mut mag_q31);
    let _dot_q31 = cmplx_dot_prod_q31(&a_q31, &b_q31);

    let a_q15 = [q15::from_bits(1000), q15::from_bits(2000)];
    let b_q15 = [q15::from_bits(500), q15::from_bits(1000)];
    let mut out_q15 = [q15::ZERO; 2];

    cmplx_add_q15(&a_q15, &b_q15, &mut out_q15);
    cmplx_sub_q15(&a_q15, &b_q15, &mut out_q15);
    cmplx_mult_cmplx_q15(&a_q15, &b_q15, &mut out_q15);
    cmplx_mult_real_q15(&a_q15, &b_q15, &mut out_q15);
    cmplx_conj_q15(&a_q15, &mut out_q15);

    let mut mag_q15 = [q15::ZERO; 1];
    cmplx_mag_q15(&a_q15, &mut mag_q15);
    cmplx_mag_squared_q15(&a_q15, &mut mag_q15);
    let _dot_q15 = cmplx_dot_prod_q15(&a_q15, &b_q15);
}

#[test]
fn test_transforms_q31_q15_and_wavelets() {
    let mut q31_buf = [q31::from_bits(1000); 32];
    let mut out_q31 = [q31::ZERO; 32];
    cfft_q31(&mut q31_buf[..32], 16, 0, 1);
    cfft_q31(&mut q31_buf[..32], 16, 1, 1);
    let _scale_q31 = cfft_bfp_q31(&mut q31_buf[..32], 16, 0, 1);
    rfft_q31(&q31_buf[..16], &mut out_q31[..32], 16, 0);
    rfft_q31(&q31_buf[..16], &mut out_q31[..32], 16, 1);
    irfft_q31(&out_q31[..32], &mut q31_buf[..16], 16);

    let mut q15_buf = [q15::from_bits(100); 32];
    let mut out_q15 = [q15::ZERO; 32];
    cfft_q15(&mut q15_buf[..32], 16, 0, 1);
    cfft_q15(&mut q15_buf[..32], 16, 1, 1);
    let _scale_q15 = cfft_bfp_q15(&mut q15_buf[..32], 16, 0, 1);
    rfft_q15(&q15_buf[..16], &mut out_q15[..32], 16, 0);
    rfft_q15(&q15_buf[..16], &mut out_q15[..32], 16, 1);
    irfft_q15(&out_q15[..32], &mut q15_buf[..16], 16);

    let f32_buf = [1.0f32; 16];
    let mut out_f32 = [0.0f32; 32];
    rfft_f32(&f32_buf, &mut out_f32, 16, 0);
    rfft_f32(&f32_buf, &mut out_f32, 16, 1);

    let mut cep_out = [0.0f32; 16];
    assert_eq!(real_cepstrum_f32(&f32_buf, &mut cep_out), Status::Success);

    // FWHT and Haar
    let mut fwht_buf = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(fwht_f32(&mut fwht_buf), Status::Success);
    assert_eq!(ifwht_f32(&mut fwht_buf), Status::Success);

    let mut fwht_i32_buf = [10i32, 20, 30, 40];
    assert_eq!(fwht_i32(&mut fwht_i32_buf), Status::Success);

    let mut haar_i32_buf = [10i32, 20, 30, 40];
    assert_eq!(haar_transform_i32(&mut haar_i32_buf), Status::Success);

    let mut haar_f32_buf = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(
        inverse_haar_transform_f32(&mut haar_f32_buf),
        Status::Success
    );

    // Wavelets
    let daub4 = [0.482_962_9, 0.836_516_3, 0.224_143_86, -0.129_409_52];
    let mut wave_buf = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(wavelet_step_f32(&mut wave_buf, 4, &daub4), Status::Success);
    assert_eq!(
        inverse_wavelet_step_f32(&mut wave_buf, 4, &daub4),
        Status::Success
    );
    assert_eq!(
        wavelet_transform_f32(&mut wave_buf, &daub4),
        Status::Success
    );
    assert_eq!(
        inverse_wavelet_transform_f32(&mut wave_buf, &daub4),
        Status::Success
    );
}

#[test]
fn test_filter_design_all_windowed_sinc() {
    let mut hp_taps = [0.0f32; 15];
    assert_eq!(
        fir_windowed_sinc_highpass(0.2, &mut hp_taps),
        Status::Success
    );

    let mut bp_taps = [0.0f32; 15];
    assert_eq!(
        fir_windowed_sinc_bandpass(0.1, 0.3, &mut bp_taps),
        Status::Success
    );

    let mut bs_taps = [0.0f32; 15];
    assert_eq!(
        fir_windowed_sinc_bandstop(0.1, 0.3, &mut bs_taps),
        Status::Success
    );

    let mut biquad_q31 = [q31::ZERO; 5];
    assert!(
        biquad_quantize_and_scale_q31(
            &[1.0, -0.5, 0.25, 0.1, -0.05],
            &mut biquad_q31,
            ScalingStrategy::Direct
        )
        .is_ok()
    );

    let mut biquad_q15 = [q15::ZERO; 5];
    assert!(
        biquad_quantize_and_scale_q15(
            &[1.0, -0.5, 0.25, 0.1, -0.05],
            &mut biquad_q15,
            ScalingStrategy::Direct
        )
        .is_ok()
    );

    let mut fir_q15_taps = [q15::ZERO; 15];
    assert!(fir_quantize_q15(&hp_taps, &mut fir_q15_taps).is_ok());

    assert_eq!(
        single_pole_decay_from_time_constant(10.0),
        (-1.0f32 / 10.0).exp()
    );
    assert_eq!(
        single_pole_decay_from_cutoff(0.1),
        (-2.0f32 * core::f32::consts::PI * 0.1).exp()
    );

    assert!(biquad_lowpass_coeffs(1000.0, 48000.0, 0.707)[0].is_finite());
    assert!(biquad_highpass_coeffs(1000.0, 48000.0, 0.707)[0].is_finite());
    assert!(biquad_bandpass_coeffs(1000.0, 48000.0, 0.707)[0].is_finite());
    assert!(biquad_notch_coeffs(1000.0, 48000.0, 0.707)[0].is_finite());
    assert!(biquad_peaking_coeffs(1000.0, 48000.0, 0.707, 3.0)[0].is_finite());
    assert!(biquad_allpass_coeffs(1000.0, 48000.0, 0.707)[0].is_finite());

    let mut biquads_bw = [0.0f32; 10];
    butterworth_lowpass_biquads(1000.0, 48000.0, 4, &mut biquads_bw);

    let mut biquads_cheb = [0.0f32; 10];
    chebyshev_lowpass_biquads(0.1, 0.5, 4, &mut biquads_cheb);
    chebyshev_highpass_biquads(0.1, 0.5, 4, &mut biquads_cheb);

    let pw = prewarp_cutoff_f32(1000.0, 48000.0);
    let biquad_out = bilinear_transform_biquad(pw, 0.0, 0.0, 1.0, 0.0, 0.0, 48000.0);
    assert_eq!(biquad_out.len(), 5);
}

#[test]
fn test_kalman_all_methods() {
    let mut kf = KalmanFilter::<2, 1>::from_variances([0.0, 0.0], 1.0, 0.01, 0.1);
    let f = [[1.0, 1.0], [0.0, 1.0]];
    kf.predict(&f);
    let b = [[0.1], [0.05]];
    let u = [1.0];
    kf.predict_with_control(&f, &b, &u);
    assert_eq!(kf.update(&[[1.0, 0.0]], &[1.0]), Status::Success);

    let mut sr_kf = SquareRootKalmanFilter::<2, 1>::new(
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.1, 0.0], [0.0, 0.1]],
        [[1.0, 0.0]],
        [[0.1]],
    );
    sr_kf.predict();
    assert_eq!(sr_kf.update(&[1.0]), Status::Success);
    assert_eq!(sr_kf.covariance().len(), 2);
}

#[test]
fn test_math_f64_full_coverage() {
    let f: f64 = 0.5;
    assert!((FloatMath::abs(f) - 0.5).abs() < 1e-6);
    assert!(FloatMath::sin(f) > 0.0);
    assert!(FloatMath::cos(f) > 0.0);
    assert!(FloatMath::tan(f) > 0.0);
    assert!(FloatMath::sqrt(f) > 0.0);
    assert!(FloatMath::ln(f) < 0.0);
    assert!(FloatMath::log10(f) < 0.0);
    assert!(FloatMath::exp(f) > 1.0);
    assert!(FloatMath::atan2(f, 1.0) > 0.0);
    assert!((FloatMath::powf(f, 2.0) - 0.25).abs() < 1e-6);
    assert!(FloatMath::tanh(f) > 0.0);

    assert_eq!(isqrt_u32(0), 0);
    assert_eq!(isqrt_u32(1), 1);
    assert_eq!(isqrt_u32(16), 4);
    assert_eq!(isqrt_u32(100), 10);

    assert_eq!(isqrt_u64(0), 0);
    assert_eq!(isqrt_u64(1), 1);
    assert_eq!(isqrt_u64(144), 12);
}
