use embedded_dsp::*;

#[test]
fn test_transforms_inverse_and_flags() {
    let mut data_q31 = [q31::from_bits(10000); 16];
    cfft_q31(&mut data_q31, 8, 1, 0);
    cfft_q31(&mut data_q31, 8, 1, 1);
    cfft_q31(&mut data_q31, 8, 0, 0);

    let mut data_q15 = [q15::from_bits(1000); 16];
    cfft_q15(&mut data_q15, 8, 1, 0);
    cfft_q15(&mut data_q15, 8, 1, 1);
    cfft_q15(&mut data_q15, 8, 0, 0);

    let mut bfp_q31 = [q31::from_bits(10000); 16];
    let _s_q31_1 = cfft_bfp_q31(&mut bfp_q31, 8, 1, 0);
    let _s_q31_2 = cfft_bfp_q31(&mut bfp_q31, 8, 1, 1);

    let mut bfp_q15 = [q15::from_bits(1000); 16];
    let _s_q15_1 = cfft_bfp_q15(&mut bfp_q15, 8, 1, 0);
    let _s_q15_2 = cfft_bfp_q15(&mut bfp_q15, 8, 1, 1);

    let src_q31 = [q31::from_bits(5000); 8];
    let mut dst_q31_a = [q31::ZERO; 16];
    let mut dst_q31_b = [q31::ZERO; 8];
    rfft_q31(&src_q31, &mut dst_q31_a, 8, 0); // packed_rfft_q31_forward
    irfft_q31(&dst_q31_a, &mut dst_q31_b, 8); // packed_irfft_q31
    rfft_q31(&src_q31, &mut dst_q31_a, 8, 1); // fallback unpack branch

    let src_q15 = [q15::from_bits(500); 8];
    let mut dst_q15_a = [q15::ZERO; 16];
    let mut dst_q15_b = [q15::ZERO; 8];
    rfft_q15(&src_q15, &mut dst_q15_a, 8, 0); // packed_rfft_q15_forward
    irfft_q15(&dst_q15_a, &mut dst_q15_b, 8); // packed_irfft_q15
    rfft_q15(&src_q15, &mut dst_q15_a, 8, 1); // fallback unpack branch

    let mut wht_f32 = [1.0f32; 8];
    assert_eq!(ifwht_f32(&mut wht_f32), Status::Success);

    let mut wht_i32 = [10i32; 8];
    assert_eq!(fwht_i32(&mut wht_i32), Status::Success);

    let mut haar_f32 = [1.0f32; 8];
    assert_eq!(haar_transform_f32(&mut haar_f32), Status::Success);
    assert_eq!(inverse_haar_transform_f32(&mut haar_f32), Status::Success);

    let mut haar_i32 = [10i32; 8];
    assert_eq!(haar_transform_i32(&mut haar_i32), Status::Success);

    let mut hartley = [1.0f32; 8];
    assert_eq!(hartley_transform_f32(&mut hartley), Status::Success);

    let db4_h = [
        0.4829629131445341,
        0.8365163037378079,
        0.2241438680420134,
        -0.1294095225512604,
    ];
    let mut wav_data = [1.0f32; 8];
    assert_eq!(wavelet_step_f32(&mut wav_data, 8, &db4_h), Status::Success);
    assert_eq!(
        inverse_wavelet_step_f32(&mut wav_data, 8, &db4_h),
        Status::Success
    );
    assert_eq!(
        wavelet_transform_f32(&mut wav_data, &db4_h),
        Status::Success
    );
    assert_eq!(
        inverse_wavelet_transform_f32(&mut wav_data, &db4_h),
        Status::Success
    );
}

#[test]
fn test_transforms_error_branches() {
    let mut bad_buf = [0.0f32; 3];
    assert_ne!(fwht_f32(&mut bad_buf), Status::Success);
    assert_ne!(ifwht_f32(&mut bad_buf), Status::Success);
    assert_ne!(haar_transform_f32(&mut bad_buf), Status::Success);
    assert_ne!(inverse_haar_transform_f32(&mut bad_buf), Status::Success);
    assert_ne!(hartley_transform_f32(&mut bad_buf), Status::Success);

    let mut bad_buf_i32 = [0i32; 3];
    assert_ne!(fwht_i32(&mut bad_buf_i32), Status::Success);
    assert_ne!(haar_transform_i32(&mut bad_buf_i32), Status::Success);

    let db4_h = [
        0.4829629131445341,
        0.8365163037378079,
        0.2241438680420134,
        -0.1294095225512604,
    ];
    assert_ne!(wavelet_step_f32(&mut bad_buf, 3, &db4_h), Status::Success);
    assert_ne!(
        inverse_wavelet_step_f32(&mut bad_buf, 3, &db4_h),
        Status::Success
    );
    assert_ne!(wavelet_transform_f32(&mut bad_buf, &db4_h), Status::Success);
    assert_ne!(
        inverse_wavelet_transform_f32(&mut bad_buf, &db4_h),
        Status::Success
    );

    let mut cep_out = [0.0f32; 2];
    assert_ne!(real_cepstrum_f32(&bad_buf, &mut cep_out), Status::Success);

    let src_short = [q31::ZERO; 2];
    let mut dst_short = [q31::ZERO; 2];
    irfft_q31(&src_short, &mut dst_short, 8);
    let src_short_q15 = [q15::ZERO; 2];
    let mut dst_short_q15 = [q15::ZERO; 2];
    irfft_q15(&src_short_q15, &mut dst_short_q15, 8);
}

#[test]
fn test_types_fixed_and_enums() {
    let q = q15::from_bits(1000);
    assert_eq!(q15::ZERO.to_bits(), 0);
    assert_eq!(q15::MIN.to_bits(), i16::MIN);
    assert_eq!(q15::MAX.to_bits(), i16::MAX);
    assert_eq!(q.to_bits(), 1000);

    let _q_sat_add = q.saturating_add(q15::from_bits(500));
    let _q_sat_sub = q.saturating_sub(q15::from_bits(500));
    let _q_sat_mul = q.saturating_mul(q15::from_bits(500));
    let _q_sat_neg = q.saturating_neg();
    let _q_sat_abs = q.saturating_abs();
    let _q_abs = q.abs();
    let _q_wrap_add = q.wrapping_add(q15::from_bits(500));
    let _q_wrap_sub = q.wrapping_sub(q15::from_bits(500));
    let _q_wrap_neg = q.wrapping_neg();
    let _q_wrap_mul = q.wrapping_mul(q15::from_bits(500));
    let _q_wrap_mul_int = q.wrapping_mul_int(2);
    let _q_chk_div = q.checked_div(q15::from_bits(500));
    let _q_chk_div_zero = q.checked_div(q15::ZERO);

    // Status enum variants
    assert_eq!(Status::Success as i8, 0);
    assert_eq!(Status::ArgumentError as i8, -1);
    assert_eq!(Status::LengthError as i8, -2);
    assert_eq!(Status::SizeMismatch as i8, -3);
    assert_eq!(Status::NanInf as i8, -4);
    assert_eq!(Status::Singular as i8, -5);
    assert_eq!(Status::TestFailure as i8, -6);
    assert_eq!(Status::DecompositionFailure as i8, -7);
}

#[test]
fn test_windows_and_statistics_extra() {
    let mut win_buf = [0.0f32; 16];
    hanning_f32(&mut win_buf);
    hamming_f32(&mut win_buf);
    blackman_f32(&mut win_buf);
    blackman_harris_f32(&mut win_buf);
    bartlett_f32(&mut win_buf);
    welch_f32(&mut win_buf);
    flattop_f32(&mut win_buf);
    kaiser_f32(&mut win_buf, 5.0);
    apply_window_f32(&mut win_buf, &[1.0f32; 16]);

    let mut q15_win = [q15::ZERO; 16];
    hanning_q15(&mut q15_win);
    hamming_q15(&mut q15_win);
    blackman_q15(&mut q15_win);
    bartlett_q15(&mut q15_win);
    apply_window_q15(&mut q15_win, &[q15::from_bits(1000); 16]);

    let src_q7 = [q7::from_bits(10); 8];
    let mut q7_res = q7::ZERO;
    let mut idx = 0usize;
    assert_eq!(mean_q7(&src_q7, &mut q7_res), Status::Success);
    assert_eq!(var_q7(&src_q7, &mut q7_res), Status::Success);
    assert_eq!(std_q7(&src_q7, &mut q7_res), Status::Success);
    assert_eq!(min_q7(&src_q7, &mut q7_res, &mut idx), Status::Success);
    assert_eq!(max_q7(&src_q7, &mut q7_res, &mut idx), Status::Success);

    let data = [0.1f32, 0.2, 0.3, 0.4];
    assert!(entropy_f32(&data).is_finite());
    assert!(kullback_leibler_f32(&data, &data).is_finite());
    assert!(logsumexp_f32(&data).is_finite());

    let mut f_res = 0.0f32;
    assert_eq!(absmax_f32(&data, &mut f_res, &mut idx), Status::Success);
    assert_eq!(absmin_f32(&data, &mut f_res, &mut idx), Status::Success);
}

#[test]
fn test_matrix_extra() {
    let mut data_a = [1.0f32, 2.0, 3.0, 4.0];
    let mut data_b = [5.0f32, 6.0, 7.0, 8.0];
    let mut data_out = [0.0f32; 4];

    let mat_a = MatrixInstance::new(2, 2, &mut data_a);
    let mat_b = MatrixInstance::new(2, 2, &mut data_b);
    let mut mat_out = MatrixInstanceMut::new(2, 2, &mut data_out);

    assert_eq!(mat_add_f32(&mat_a, &mat_b, &mut mat_out), Status::Success);
    assert_eq!(mat_sub_f32(&mat_a, &mat_b, &mut mat_out), Status::Success);
    assert_eq!(mat_scale_f32(&mat_a, 2.0, &mut mat_out), Status::Success);
    assert_eq!(mat_mult_f32(&mat_a, &mat_b, &mut mat_out), Status::Success);
    assert_eq!(mat_trans_f32(&mat_a, &mut mat_out), Status::Success);
    assert_eq!(mat_inverse_f32(&mat_a, &mut mat_out), Status::Success);
}
