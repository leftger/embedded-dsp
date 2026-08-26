use embedded_dsp::*;

// =========================================================================================
// 1. BASIC MATH & BITWISE TESTS
// =========================================================================================

#[test]
fn test_basic_math_float() {
    let a = [1.0f32, 2.0, -3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut out = [0.0f32; 4];

    add_f32(&a, &b, &mut out);
    assert_eq!(out, [6.0, 8.0, 4.0, 12.0]);

    sub_f32(&b, &a, &mut out);
    assert_eq!(out, [4.0, 4.0, 10.0, 4.0]);

    mult_f32(&a, &b, &mut out);
    assert_eq!(out, [5.0, 12.0, -21.0, 32.0]);

    negate_f32(&a, &mut out);
    assert_eq!(out, [-1.0, -2.0, 3.0, -4.0]);

    abs_f32(&a, &mut out);
    assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);

    offset_f32(&a, 10.0, &mut out);
    assert_eq!(out, [11.0, 12.0, 7.0, 14.0]);

    scale_f32(&a, 2.0, &mut out);
    assert_eq!(out, [2.0, 4.0, -6.0, 8.0]);

    clip_f32(&a, 0.0, 3.0, &mut out);
    assert_eq!(out, [1.0, 2.0, 0.0, 3.0]);

    let dot = dot_prod_f32(&a, &b);
    assert_eq!(dot, 5.0 + 12.0 - 21.0 + 32.0);
}

#[test]
fn test_fixed_point_basic_math_saturating() {
    let a = [1000i16, 2000, 3000, 30000];
    let b = [5000i16, 6000, 7000, 10000];
    let mut out = [0i16; 4];

    add_q15(&a, &b, &mut out);
    assert_eq!(out[0], 6000);
    assert_eq!(out[3], 32767); // Saturating max i16

    let a_neg = [-30000i16];
    let b_neg = [-10000i16];
    let mut out_neg = [0i16; 1];
    add_q15(&a_neg, &b_neg, &mut out_neg);
    assert_eq!(out_neg[0], -32768); // Saturating min i16

    let a_q31 = [2000000000i32];
    let b_q31 = [1000000000i32];
    let mut out_q31 = [0i32; 1];
    add_q31(&a_q31, &b_q31, &mut out_q31);
    assert_eq!(out_q31[0], 2147483647); // Saturating max i32
}

#[test]
fn test_bitwise_operations() {
    let a = [0b1100u32, 0b1010];
    let b = [0b1010u32, 0b0110];
    let mut out = [0u32; 2];

    and_u32(&a, &b, &mut out);
    assert_eq!(out, [0b1000, 0b0010]);

    or_u32(&a, &b, &mut out);
    assert_eq!(out, [0b1110, 0b1110]);

    xor_u32(&a, &b, &mut out);
    assert_eq!(out, [0b0110, 0b1100]);

    not_u32(&a, &mut out);
    assert_eq!(out, [!0b1100u32, !0b1010u32]);
}

// =========================================================================================
// 2. COMPLEX MATH TESTS
// =========================================================================================

#[test]
fn test_complex_math_comprehensive() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut out = [0.0f32; 4];

    cmplx_add_f32(&a, &b, &mut out);
    assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);

    cmplx_sub_f32(&b, &a, &mut out);
    assert_eq!(out, [4.0, 4.0, 4.0, 4.0]);

    cmplx_mult_cmplx_f32(&a, &b, &mut out);
    assert_eq!(out, [-7.0, 16.0, -11.0, 52.0]);

    cmplx_mult_real_f32(&a, &[2.0, 3.0], &mut out);
    assert_eq!(out, [2.0, 4.0, 9.0, 12.0]);

    let mut mag = [0.0f32; 2];
    cmplx_mag_f32(&a, &mut mag);
    assert!((mag[0] - (1.0f32 + 4.0).sqrt()).abs() < 1e-4);
    assert!((mag[1] - (9.0f32 + 16.0).sqrt()).abs() < 1e-4);

    let mut mag_sq = [0.0f32; 2];
    cmplx_mag_squared_f32(&a, &mut mag_sq);
    assert_eq!(mag_sq, [5.0, 25.0]);

    cmplx_conj_f32(&a, &mut out);
    assert_eq!(out, [1.0, -2.0, 3.0, -4.0]);

    let dot = cmplx_dot_prod_f32(&a, &b);
    assert_eq!(dot.real, -18.0);
    assert_eq!(dot.imag, 68.0);
}

// =========================================================================================
// 3. FAST MATH & TRIGONOMETRY TESTS
// =========================================================================================

#[test]
fn test_fast_trig_and_roots() {
    let pi = core::f32::consts::PI;

    assert!((sin_f32(0.0) - 0.0).abs() < 1e-5);
    assert!((sin_f32(pi / 2.0) - 1.0).abs() < 1e-4);
    assert!((cos_f32(0.0) - 1.0).abs() < 1e-5);
    assert!((cos_f32(pi) - (-1.0)).abs() < 1e-4);

    let mut s = 0.0f32;
    let mut c = 0.0f32;
    sin_cos_f32(45.0, &mut s, &mut c);
    assert!((s - core::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4);
    assert!((c - core::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4);

    let mut res = 0.0f32;
    assert_eq!(sqrt_f32(16.0, &mut res), Status::Success);
    assert_eq!(res, 4.0);

    let src = [4.0f32, 9.0, 16.0, 25.0];
    let mut dst = [0.0f32; 4];
    vsqrt_f32(&src, &mut dst);
    assert_eq!(dst, [2.0, 3.0, 4.0, 5.0]);

    assert!((log_f32(1.0) - 0.0).abs() < 1e-5);
    assert!((exp_f32(0.0) - 1.0).abs() < 1e-5);

    let mut atan_res = 0.0f32;
    assert_eq!(atan2_f32(1.0, 1.0, &mut atan_res), Status::Success);
    assert!((atan_res - (pi / 4.0)).abs() < 1e-4);

    let mut out_q31 = 0i32;
    assert_eq!(sqrt_q31(1073741824, &mut out_q31), Status::Success);
    assert!((out_q31 - 1518500249).abs() < 1000);
}

// =========================================================================================
// 4. FILTERING & CONVOLUTION TESTS
// =========================================================================================

#[test]
fn test_fir_filter_impulse_response() {
    let coeffs = [0.25f32, 0.5, 0.25];
    let mut state = [0.0f32; 3];
    let mut fir = FirInstanceF32::init(3, &coeffs, &mut state);

    let src = [1.0f32, 0.0, 0.0, 0.0];
    let mut dst = [0.0f32; 4];
    fir_f32(&mut fir, &src, &mut dst);

    assert_eq!(dst, [0.25, 0.5, 0.25, 0.0]);
}

#[test]
fn test_biquad_cascade_iir() {
    let coeffs = [1.0f32, 0.0, 0.0, 0.0, 0.0];
    let mut state = [0.0f32; 4];
    let mut iir = BiquadCascadeInstanceF32::init(1, &coeffs, &mut state);

    let src = [1.0f32, 2.0, 3.0, 4.0];
    let mut dst = [0.0f32; 4];
    biquad_cascade_df1_f32(&mut iir, &src, &mut dst);

    assert_eq!(dst, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_lms_adaptive_filter() {
    let mut coeffs = [0.0f32; 2];
    let mut state = [0.0f32; 2];
    let mut lms = LmsInstanceF32::init(2, &mut coeffs, &mut state, 0.01);

    let src = [1.0f32, 2.0];
    let ref_sig = [1.0f32, 2.0];
    let mut out = [0.0f32; 2];
    let mut err = [0.0f32; 2];

    lms_f32(&mut lms, &src, &ref_sig, &mut out, &mut err);
    assert_eq!(out.len(), 2);

    let mut ncoeffs = [0.0f32; 4];
    let mut nstate = [0.0f32; 4];
    let mut nlms = NlmsInstanceF32::init(4, &mut ncoeffs, &mut nstate, 0.5, 1e-6);
    let mut nout = [0.0f32; 64];
    let mut nerr = [0.0f32; 64];
    let mut nsrc = [0.0f32; 64];
    let mut nref = [0.0f32; 64];
    for i in 0..64 {
        nsrc[i] = ((i * 17) % 10) as f32 / 10.0 - 0.45;
        nref[i] = 0.5 * nsrc[i];
    }
    nlms_f32(&mut nlms, &nsrc, &nref, &mut nout, &mut nerr);
    let last_err = nerr[63].abs();
    assert!(last_err < 0.15, "nlms err {last_err}");

    let mut qcoeffs = [0i16; 4];
    let mut qstate = [0i16; 4];
    let mut qlms = LmsInstanceQ15::init(4, &mut qcoeffs, &mut qstate, 1024);
    let mut qsrc = [0i16; 64];
    let mut qref = [0i16; 64];
    for i in 0..64 {
        qsrc[i] = (nsrc[i] * 16000.0) as i16;
        qref[i] = (nref[i] * 16000.0) as i16;
    }
    let mut qout = [0i16; 64];
    let mut qerr = [0i16; 64];
    lms_q15(&mut qlms, &qsrc, &qref, &mut qout, &mut qerr);
    lms_leaky_q15(&mut qlms, &qsrc, &qref, &mut qout, &mut qerr, 32);

    let mut nqcoeffs = [0i16; 4];
    let mut nqstate = [0i16; 4];
    let mut qnlms = NlmsInstanceQ15::init(4, &mut nqcoeffs, &mut nqstate, 16384, 8);
    nlms_q15(&mut qnlms, &qsrc, &qref, &mut qout, &mut qerr);
}

#[test]
fn test_convolution_and_correlation() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0];
    let mut conv_out = [0.0f32; 4];

    conv_f32(&a, &b, &mut conv_out);
    assert_eq!(conv_out, [4.0, 13.0, 22.0, 15.0]);

    let mut corr_out = [0.0f32; 4];
    correlate_f32(&a, &b, &mut corr_out);
}

// =========================================================================================
// 5. TRANSFORMS & SPECTRAL ANALYSIS TESTS
// =========================================================================================

#[test]
fn test_cfft_and_ifft() {
    let mut data = [1.0f32, 0.0, 1.0f32, 0.0, 1.0f32, 0.0, 1.0f32, 0.0];
    cfft_f32(&mut data, 4, 0, 1);
    assert!((data[0] - 4.0).abs() < 1e-4);
    assert!((data[1] - 0.0).abs() < 1e-4);

    cfft_f32(&mut data, 4, 1, 1);
    assert!((data[0] - 1.0).abs() < 1e-4);
    assert!((data[2] - 1.0).abs() < 1e-4);
}

#[test]
fn test_rfft_and_dct4() {
    let src = [1.0f32, 2.0, 3.0, 4.0];
    let mut dst = [0.0f32; 4];

    rfft_f32(&src, &mut dst, 4, 0);
    dct4_f32(&src, &mut dst, 4);
}

// =========================================================================================
// 6. MATRIX OPERATIONS TESTS
// =========================================================================================

#[test]
fn test_matrix_operations_comprehensive() {
    let a_data = [1.0f32, 2.0, 3.0, 4.0];
    let b_data = [5.0f32, 6.0, 7.0, 8.0];
    let mut out_data = [0.0f32; 4];

    let mat_a = MatrixInstance::new(2, 2, &a_data);
    let mat_b = MatrixInstance::new(2, 2, &b_data);
    let mut mat_out = MatrixInstanceMut::new(2, 2, &mut out_data);

    assert_eq!(mat_add_f32(&mat_a, &mat_b, &mut mat_out), Status::Success);
    assert_eq!(mat_out.data, &[6.0, 8.0, 10.0, 12.0]);

    assert_eq!(mat_sub_f32(&mat_b, &mat_a, &mut mat_out), Status::Success);
    assert_eq!(mat_out.data, &[4.0, 4.0, 4.0, 4.0]);

    assert_eq!(mat_scale_f32(&mat_a, 2.0, &mut mat_out), Status::Success);
    assert_eq!(mat_out.data, &[2.0, 4.0, 6.0, 8.0]);

    assert_eq!(mat_mult_f32(&mat_a, &mat_b, &mut mat_out), Status::Success);
    assert_eq!(mat_out.data, &[19.0, 22.0, 43.0, 50.0]);

    assert_eq!(mat_trans_f32(&mat_a, &mut mat_out), Status::Success);
    assert_eq!(mat_out.data, &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_matrix_inverse() {
    let a_data = [4.0f32, 7.0, 2.0, 6.0];
    let mut inv_data = [0.0f32; 4];

    let mat_a = MatrixInstance::new(2, 2, &a_data);
    let mut mat_inv = MatrixInstanceMut::new(2, 2, &mut inv_data);

    assert_eq!(mat_inverse_f32(&mat_a, &mut mat_inv), Status::Success);

    let mut res_data = [0.0f32; 4];
    let mut mat_res = MatrixInstanceMut::new(2, 2, &mut res_data);
    mat_mult_f32(
        &mat_a,
        &MatrixInstance::new(2, 2, mat_inv.data),
        &mut mat_res,
    );

    assert!((mat_res.data[0] - 1.0).abs() < 1e-4);
    assert!((mat_res.data[1] - 0.0).abs() < 1e-4);
    assert!((mat_res.data[2] - 0.0).abs() < 1e-4);
    assert!((mat_res.data[3] - 1.0).abs() < 1e-4);
}

// =========================================================================================
// 7. CONTROLLER & MOTOR TRANSFORMS TESTS
// =========================================================================================

#[test]
fn test_pid_and_clarke_park() {
    let mut pid = PidInstanceF32::new(1.0, 0.1, 0.01);
    let out1 = pid.process(10.0);
    assert!((out1 - 11.1).abs() < 1e-3);

    let mut alpha = 0.0f32;
    let mut beta = 0.0f32;
    clarke_f32(1.0, -0.5, &mut alpha, &mut beta);
    assert_eq!(alpha, 1.0);

    let mut d = 0.0f32;
    let mut q = 0.0f32;
    park_f32(alpha, beta, 0.0, &mut d, &mut q);
    assert_eq!(d, 1.0);
    assert_eq!(q, 0.0);

    let ia = 16384i16;
    let ib = -8192i16;
    let mut a_q = 0i16;
    let mut b_q = 0i16;
    clarke_q15(ia, ib, &mut a_q, &mut b_q);
    assert_eq!(a_q, ia);
    let mut ia2 = 0i16;
    let mut ib2 = 0i16;
    inv_clarke_q15(a_q, b_q, &mut ia2, &mut ib2);
    assert!((ia2 as i32 - ia as i32).abs() < 8);
    assert!((ib2 as i32 - ib as i32).abs() < 16);

    let mut d_q = 0i16;
    let mut q_q = 0i16;
    park_q15(a_q, b_q, 0, 32767, &mut d_q, &mut q_q);
    assert!((d_q as i32 - a_q as i32).abs() < 4);
    assert!(q_q.abs() < 4);
    let mut ar = 0i16;
    let mut br = 0i16;
    inv_park_q15(d_q, q_q, 0, 32767, &mut ar, &mut br);
    assert!((ar as i32 - a_q as i32).abs() < 8);
}

// =========================================================================================
// 8. STATISTICS & INFORMATION THEORY TESTS
// =========================================================================================

#[test]
fn test_statistics_comprehensive() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mut m = 0.0f32;
    mean_f32(&data, &mut m);
    assert_eq!(m, 3.0);

    let mut v = 0.0f32;
    var_f32(&data, &mut v);
    assert_eq!(v, 2.5);

    let mut s = 0.0f32;
    std_f32(&data, &mut s);
    assert!((s - 1.5811388).abs() < 1e-4);

    let mut rms_val = 0.0f32;
    rms_f32(&data, &mut rms_val);
    assert!((rms_val - (55.0f32 / 5.0).sqrt()).abs() < 1e-4);

    let mut pwr_val = 0.0f32;
    power_f32(&data, &mut pwr_val);
    assert_eq!(pwr_val, 55.0);

    let mut min_val = 0.0f32;
    let mut min_idx = 0;
    min_f32(&data, &mut min_val, &mut min_idx);
    assert_eq!(min_val, 1.0);
    assert_eq!(min_idx, 0);

    let mut max_val = 0.0f32;
    let mut max_idx = 0;
    max_f32(&data, &mut max_val, &mut max_idx);
    assert_eq!(max_val, 5.0);
    assert_eq!(max_idx, 4);

    let prob = [0.5f32, 0.5];
    let ent = entropy_f32(&prob);
    assert!((ent - core::f32::consts::LN_2).abs() < 1e-4);

    let kl = kullback_leibler_f32(&prob, &prob);
    assert!((kl - 0.0).abs() < 1e-5);

    let lse = logsumexp_f32(&[1.0, 2.0]);
    assert!((lse - (1.0f32.exp() + 2.0f32.exp()).ln()).abs() < 1e-4);
}

// =========================================================================================
// 9. SUPPORT & CONVERSIONS TESTS
// =========================================================================================

#[test]
fn test_conversions_and_sorting() {
    let src_q15 = [16384i16, -16384];
    let mut dst_f32 = [0.0f32; 2];
    q15_to_f32(&src_q15, &mut dst_f32);
    assert!((dst_f32[0] - 0.5).abs() < 1e-4);
    assert!((dst_f32[1] - (-0.5)).abs() < 1e-4);

    let mut dst_q15 = [0i16; 2];
    f32_to_q15(&dst_f32, &mut dst_q15);
    assert_eq!(dst_q15[0], 16384);
    assert_eq!(dst_q15[1], -16384);

    let src = [5.0f32, 1.0, 4.0, 2.0, 3.0];
    let mut dst = [0.0f32; 5];
    sort_f32(&src, &mut dst, true); // Ascending
    assert_eq!(dst, [1.0, 2.0, 3.0, 4.0, 5.0]);

    sort_f32(&src, &mut dst, false); // Descending
    assert_eq!(dst, [5.0, 4.0, 3.0, 2.0, 1.0]);
}

// =========================================================================================
// 10. INTERPOLATION TESTS
// =========================================================================================

#[test]
fn test_interpolation() {
    let y_table = [0.0f32, 10.0, 20.0, 30.0];
    let val = linear_interp_f32(&y_table, 1.5, 1.0);
    assert_eq!(val, 15.0);
}

// =========================================================================================
// 11. QUATERNIONS TESTS
// =========================================================================================

#[test]
fn test_quaternion_comprehensive() {
    let q = [1.0f32, 0.0, 0.0, 0.0];
    let norm = quaternion_norm_f32(&q);
    assert_eq!(norm, 1.0);

    let mut rot = [0.0f32; 9];
    quaternion_to_rotmat_f32(&q, &mut rot);
    assert_eq!(rot, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    let q1 = [1.0f32, 0.0, 0.0, 0.0];
    let q2 = [0.0f32, 1.0, 0.0, 0.0];
    let mut prod = [0.0f32; 4];
    quaternion_product_f32(&q1, &q2, &mut prod);
    assert_eq!(prod, [0.0, 1.0, 0.0, 0.0]);
}

// =========================================================================================
// 12. WINDOW FUNCTIONS TESTS
// =========================================================================================

#[test]
fn test_windows() {
    let mut w = [0.0f32; 5];
    hanning_f32(&mut w);
    assert_eq!(w[0], 0.0);
    assert_eq!(w[2], 1.0);
    assert_eq!(w[4], 0.0);

    hamming_f32(&mut w);
    assert!((w[0] - 0.08).abs() < 1e-4);
    assert!((w[2] - 1.0).abs() < 1e-4);

    let mut sig = [2.0f32; 5];
    apply_window_f32(&mut sig, &w);
    assert!((sig[2] - 2.0).abs() < 1e-4);

    let mut wq = [0i16; 5];
    hanning_q15(&mut wq);
    assert_eq!(wq[0], 0);
    assert!((wq[2] - 32767).abs() < 2);
    assert_eq!(wq[4], 0);
    let mut sigq = [32767i16; 5];
    apply_window_q15(&mut sigq, &wq);
    assert!((sigq[2] - 32766).abs() < 4);
}

// =========================================================================================
// 13. DISTANCE METRICS TESTS
// =========================================================================================

#[test]
fn test_distance_metrics_exhaustive() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];

    assert!((euclidean_distance_f32(&a, &b) - 5.196152).abs() < 1e-4);
    assert!((chebyshev_distance_f32(&a, &b) - 3.0).abs() < 1e-4);
    assert!((manhattan_distance_f32(&a, &b) - 9.0).abs() < 1e-4);
    assert!((minkowski_distance_f32(&a, &b, 1.0) - 9.0).abs() < 1e-4);
    assert!((canberra_distance_f32(&a, &b) - (3.0 / 5.0 + 3.0 / 7.0 + 3.0 / 9.0)).abs() < 1e-4);
}

// =========================================================================================
// 14. MACHINE LEARNING CLASSIFIERS TESTS
// =========================================================================================

// =========================================================================================
// 15. FILTER DESIGN TESTS
// =========================================================================================

#[test]
fn test_filter_design_coeffs() {
    let lp = biquad_lowpass_coeffs(1000.0, 48000.0, core::f32::consts::FRAC_1_SQRT_2);
    assert_eq!(lp.len(), 5);
    assert!(lp[0] > 0.0);

    let hp = biquad_highpass_coeffs(1000.0, 48000.0, core::f32::consts::FRAC_1_SQRT_2);
    assert_eq!(hp.len(), 5);

    let notch = biquad_notch_coeffs(60.0, 1000.0, 10.0);
    assert_eq!(notch.len(), 5);

    let mut butter = [0.0f32; 10];
    butterworth_lowpass_biquads(1000.0, 48000.0, 4, &mut butter);
    assert_ne!(butter[0], 0.0);
}

// =========================================================================================
// 17. RESAMPLING & MULTI-RATE TESTS
// =========================================================================================

#[test]
fn test_cic_and_resampling() {
    let mut decimator = CicDecimator::<3>::new(4);
    let mut decimated = None;
    for i in 1..=4 {
        decimated = decimator.process_sample(i * 10);
    }
    assert!(decimated.is_some());

    let mut interpolator = CicInterpolator::<2>::new(4);
    let mut out_buf = [0i32; 4];
    interpolator.process_sample(10, &mut out_buf);
    assert_eq!(out_buf.len(), 4);

    let src = [1.0f32, 2.0, 3.0, 4.0];
    let mut dst = [0.0f32; 8];
    resample_linear_f32(&src, &mut dst, 0.5);
    assert!((dst[0] - 1.0).abs() < 1e-4);
}

// =========================================================================================
// 18. KALMAN FILTERING TESTS
// =========================================================================================

#[test]
fn test_kalman_1d_and_2d() {
    let mut kf1d = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
    kf1d.predict(0.0);
    let est = kf1d.update(10.0);
    assert!(est > 0.0);

    let mut kf2d = KalmanFilter2D::new(0.0, 0.0, 0.01, 0.1);
    kf2d.predict(0.1);
    let state = kf2d.update(1.0);
    assert!(state[0] > 0.0);
}

#[test]
fn test_kalman_generic_1x1_smoother() {
    let mut kf = KalmanFilter::<1, 1>::from_variances([0.0], 1.0, 0.01, 0.1);
    let f = [[1.0f32]];
    let h = [[1.0f32]];
    kf.predict(&f);
    assert_eq!(kf.update(&h, &[10.0]), Status::Success);
    assert!(kf.x[0] > 0.0);
    assert!(kf.x[0] < 10.0);
}

#[test]
fn test_kalman_generic_2x1_constant_velocity() {
    let mut kf = KalmanFilter::<2, 1>::from_variances([0.0, 0.0], 1.0, 0.01, 0.1);
    let dt = 0.1f32;
    let f = [[1.0, dt], [0.0, 1.0]];
    let h = [[1.0, 0.0]];

    // True trajectory: position = t, velocity = 1
    for step in 1..=20 {
        kf.predict(&f);
        let z = [step as f32 * dt];
        assert_eq!(kf.update(&h, &z), Status::Success);
    }
    assert!((kf.x[0] - 2.0).abs() < 0.5);
    assert!((kf.x[1] - 1.0).abs() < 0.5);
}

/// Range-only measurement model: state = [x, y], z = sqrt(x² + y²).
#[derive(Debug, Clone, Copy)]
struct RangeOnlyModel;

impl EkfModel<2, 1> for RangeOnlyModel {
    fn f(&self, x: &[f32; 2], _dt: f32, out: &mut [f32; 2]) {
        *out = *x;
    }

    fn h(&self, x: &[f32; 2], out: &mut [f32; 1]) {
        out[0] = (x[0] * x[0] + x[1] * x[1]).sqrt();
    }

    fn jacobian_f(&self, _x: &[f32; 2], _dt: f32, out: &mut [[f32; 2]; 2]) {
        *out = [[1.0, 0.0], [0.0, 1.0]];
    }

    fn jacobian_h(&self, x: &[f32; 2], out: &mut [[f32; 2]; 1]) {
        let r = (x[0] * x[0] + x[1] * x[1]).sqrt().max(1e-6);
        out[0] = [x[0] / r, x[1] / r];
    }
}

#[test]
fn test_ekf_range_measurement() {
    let truth = [3.0f32, 4.0]; // range = 5
    let mut ekf = ExtendedKalmanFilter::<2, 1, _>::from_variances(
        [1.0, 1.0],
        1.0,
        0.001,
        0.05,
        RangeOnlyModel,
    );

    let initial_err = {
        let dx = ekf.x[0] - truth[0];
        let dy = ekf.x[1] - truth[1];
        (dx * dx + dy * dy).sqrt()
    };

    for _ in 0..40 {
        ekf.predict(0.0);
        let z = [(truth[0] * truth[0] + truth[1] * truth[1]).sqrt()];
        assert_eq!(ekf.update(&z), Status::Success);
    }

    let final_err = {
        let dx = ekf.x[0] - truth[0];
        let dy = ekf.x[1] - truth[1];
        (dx * dx + dy * dy).sqrt()
    };
    assert!(final_err < initial_err);
    assert!(((ekf.x[0] * ekf.x[0] + ekf.x[1] * ekf.x[1]).sqrt() - 5.0).abs() < 0.5);
}

/// Constant-acceleration model driven by a commanded acceleration `u = [accel]` that isn't
/// part of the state, and measured through a position sensor with a known offset `u = [bias]`
/// that also isn't part of the state. Exercises `EkfModel::f_with_input`/`h_with_input` and
/// `ExtendedKalmanFilter::predict_with_input`/`update_with_input`.
#[derive(Debug, Clone, Copy)]
struct ControlledPositionModel;

impl EkfModel<2, 1> for ControlledPositionModel {
    fn f(&self, x: &[f32; 2], dt: f32, out: &mut [f32; 2]) {
        out[0] = x[0] + dt * x[1];
        out[1] = x[1];
    }

    fn h(&self, x: &[f32; 2], out: &mut [f32; 1]) {
        out[0] = x[0];
    }

    fn jacobian_f(&self, _x: &[f32; 2], dt: f32, out: &mut [[f32; 2]; 2]) {
        *out = [[1.0, dt], [0.0, 1.0]];
    }

    fn jacobian_h(&self, _x: &[f32; 2], out: &mut [[f32; 2]; 1]) {
        *out = [[1.0, 0.0]];
    }

    fn f_with_input<const U: usize>(
        &self,
        x: &[f32; 2],
        u: &[f32; U],
        dt: f32,
        out: &mut [f32; 2],
    ) {
        let accel = u[0];
        out[0] = x[0] + dt * x[1] + 0.5 * dt * dt * accel;
        out[1] = x[1] + dt * accel;
    }

    // Jacobian w.r.t. x is unchanged by u: accel enters f affinely, so the default
    // `jacobian_f_with_input` (which defers to `jacobian_f`) is already exact here. Implemented
    // explicitly anyway so the test exercises the override path, not just the default.
    fn jacobian_f_with_input<const U: usize>(
        &self,
        x: &[f32; 2],
        _u: &[f32; U],
        dt: f32,
        out: &mut [[f32; 2]; 2],
    ) {
        self.jacobian_f(x, dt, out)
    }

    fn h_with_input<const U: usize>(&self, x: &[f32; 2], u: &[f32; U], out: &mut [f32; 1]) {
        out[0] = x[0] + u[0];
    }

    fn jacobian_h_with_input<const U: usize>(
        &self,
        x: &[f32; 2],
        _u: &[f32; U],
        out: &mut [[f32; 2]; 1],
    ) {
        self.jacobian_h(x, out)
    }
}

#[test]
fn test_ekf_predict_with_input_matches_manual_integration() {
    let mut ekf = ExtendedKalmanFilter::<2, 1, _>::from_variances(
        [0.0, 0.0],
        1.0,
        1e-4,
        0.01,
        ControlledPositionModel,
    );

    let accel = 2.0f32;
    let dt = 0.5f32;
    for _ in 0..10 {
        ekf.predict_with_input(dt, &[accel]);
    }

    let t = 10.0 * dt;
    let expected_velocity = accel * t;
    let expected_position = 0.5 * accel * t * t;
    assert!((ekf.x[1] - expected_velocity).abs() < 1e-3);
    assert!((ekf.x[0] - expected_position).abs() < 1e-2);
}

#[test]
fn test_ekf_update_with_input_compensates_known_bias() {
    let mut ekf = ExtendedKalmanFilter::<2, 1, _>::from_variances(
        [0.0, 0.0],
        4.0,
        0.0,
        0.01,
        ControlledPositionModel,
    );

    let true_position = 10.0f32;
    let sensor_bias = 3.0f32;
    // The raw sensor reading is offset by `sensor_bias`; feeding it through plain `update`
    // (which ignores the bias) would converge to the biased reading instead of the truth.
    let biased_reading = true_position + sensor_bias;

    for _ in 0..20 {
        assert_eq!(
            ekf.update_with_input(&[biased_reading], &[sensor_bias]),
            Status::Success
        );
    }

    assert!((ekf.x[0] - true_position).abs() < 0.5);
}

#[test]
fn test_kalman_update_singular_leaves_state() {
    let mut kf = KalmanFilter::<1, 1>::new([1.0], [[0.0]], [[0.0]], [[0.0]]);
    let x_before = kf.x;
    let p_before = kf.p;
    let status = kf.update(&[[1.0]], &[2.0]);
    assert_eq!(status, Status::Singular);
    assert_eq!(kf.x, x_before);
    assert_eq!(kf.p, p_before);
}

// =========================================================================================
// 19. CONST GENERICS TESTS
// =========================================================================================

#[test]
fn test_const_generics_wrappers() {
    let mut fir = FirFilter::<3>::new([0.25, 0.5, 0.25]);
    let mut dst = [0.0f32; 4];
    fir.process(&[1.0, 0.0, 0.0, 0.0], &mut dst);
    assert_eq!(dst, [0.25, 0.5, 0.25, 0.0]);

    let mut biquad = BiquadCascade::<5, 4>::new([1.0, 0.0, 0.0, 0.0, 0.0]);
    biquad.process(&[1.0, 2.0, 3.0, 4.0], &mut dst);
    assert_eq!(dst, [1.0, 2.0, 3.0, 4.0]);

    let m1 = Matrix::<2, 2, 4>::new([1.0, 2.0, 3.0, 4.0]);
    let m2 = Matrix::<2, 2, 4>::new([5.0, 6.0, 7.0, 8.0]);
    let m3 = m1.add(&m2);
    assert_eq!(m3.data, [6.0, 8.0, 10.0, 12.0]);

    let m_mul: Matrix<2, 2, 4> = m1.mul_mat(&m2);
    assert_eq!(m_mul.data, [19.0, 22.0, 43.0, 50.0]);

    let mut fir_q = FirFilterQ15::<3>::new([8192, 16384, 8192]);
    let mut dst_q = [0i16; 4];
    fir_q.process(&[32767, 0, 0, 0], &mut dst_q);
    assert!((dst_q[0] as i32 - 8191).abs() < 4);
    assert!((dst_q[1] as i32 - 16383).abs() < 4);

    let mut bq = BiquadCascadeQ15::<5, 4>::new([32767, 0, 0, 0, 0], 0);
    bq.process(&[1000, 2000, 3000, 4000], &mut dst_q);
    assert!((dst_q[0] as i32 - 1000).abs() < 3);
    assert!((dst_q[3] as i32 - 4000).abs() < 3);
}

// =========================================================================================
// 20. NON-LINEAR & CONDITIONAL MEDIAN FILTER TESTS
// =========================================================================================

#[test]
fn test_median_filter_1d_conditional() {
    // Signal with an impulse spike at index 3
    let src = [1.0f32, 1.1, 1.0, 100.0, 1.2, 1.1, 1.0];
    let mut dst = [0.0f32; 7];

    // Standard median (threshold = 0.0) -> replaces spike with median ~1.1
    let status = median_filter_1d_f32(&src, &mut dst, 3, 0.0);
    assert_eq!(status, Status::Success);
    assert!((dst[3] - 1.1).abs() < 0.15);

    // Conditional median (threshold = 5.0) -> spike (|100 - 1.1| > 5.0) is filtered, normal small variations preserved
    let mut dst_cond = [0.0f32; 7];
    let status_cond = median_filter_1d_f32(&src, &mut dst_cond, 3, 5.0);
    assert_eq!(status_cond, Status::Success);
    assert!((dst_cond[3] - 1.1).abs() < 0.15);
    assert_eq!(dst_cond[0], src[0]); // Small noise preserved

    // Q15 conditional median
    let src_q15 = [1000i16, 1100, 1050, 30000, 1150, 1100, 1000];
    let mut dst_q15 = [0i16; 7];
    let status_q15 = median_filter_1d_q15(&src_q15, &mut dst_q15, 3, 5000);
    assert_eq!(status_q15, Status::Success);
    assert!(dst_q15[3] < 2000);

    // Q31 conditional median
    let src_q31 = [
        100000i32, 110000, 105000, 2000000000, 115000, 110000, 100000,
    ];
    let mut dst_q31 = [0i32; 7];
    let status_q31 = median_filter_1d_q31(&src_q31, &mut dst_q31, 3, 1000000);
    assert_eq!(status_q31, Status::Success);
    assert!(dst_q31[3] < 200000);
}

// =========================================================================================
// 21. FAST CONVOLUTION & SPECTRAL INTERPOLATION TESTS
// =========================================================================================

#[test]
fn test_fast_convolution_and_spectral_interpolation() {
    let sig = [1.0f32, 2.0, 3.0, 4.0];
    let ker = [0.5f32, 0.5];
    let mut dst_fast = [0.0f32; 5];
    let mut dst_direct = [0.0f32; 5];

    conv_f32(&sig, &ker, &mut dst_direct);
    let status = fast_convolve_f32(&sig, &ker, &mut dst_fast);
    assert_eq!(status, Status::Success);

    for i in 0..5 {
        assert!((dst_fast[i] - dst_direct[i]).abs() < 1e-4);
    }

    // Spectral 2:1 Sinc Interpolation of a sine wave
    let mut sine_16 = [0.0f32; 16];
    for i in 0..16 {
        sine_16[i] = (2.0 * core::f32::consts::PI * (i as f32) / 16.0).sin();
    }
    let mut interp_32 = [0.0f32; 32];
    let status_interp = spectral_interpolate_2x_f32(&sine_16, &mut interp_32);
    assert_eq!(status_interp, Status::Success);

    // Check that original points are preserved at even indices
    for i in 0..16 {
        assert!((interp_32[2 * i] - sine_16[i]).abs() < 1e-3);
    }
}

// =========================================================================================
// 22. BILINEAR TRANSFORM & PRE-WARPING TESTS
// =========================================================================================

#[test]
fn test_bilinear_transform_and_prewarping() {
    let fc = 1000.0f32;
    let fs = 48000.0f32;
    let wp = prewarp_cutoff_f32(fc, fs);
    assert!(wp > 2.0 * core::f32::consts::PI * fc); // Tan(x) > x for x in (0, pi/2)

    // Standard 2nd order analog low-pass prototype: H(s) = 1 / ( (s/wp)^2 + sqrt(2)*(s/wp) + 1 )
    let q = core::f32::consts::FRAC_1_SQRT_2;
    let a0 = wp * wp;
    let a1 = 0.0f32;
    let a2 = 0.0f32;
    let b0 = wp * wp;
    let b1 = wp / q;
    let b2 = 1.0f32;

    let biquad_coeffs = bilinear_transform_biquad(a0, a1, a2, b0, b1, b2, fs);
    assert!(biquad_coeffs[0] > 0.0); // b0
    assert!(biquad_coeffs[1] > 0.0); // b1
    assert!(biquad_coeffs[2] > 0.0); // b2
}

// =========================================================================================
// 23. WEIGHTED POLYNOMIAL LEAST SQUARES REGRESSION TESTS
// =========================================================================================

#[test]
fn test_polynomial_least_squares_fit() {
    // True line: y = 3.0 + 2.0 * x
    let x = [0.0f32, 1.0, 2.0, 3.0, 4.0];
    let y = [3.0f32, 5.0, 7.0, 9.0, 11.0];
    let mut coeffs = [0.0f32; 2];

    let status = polynomial_least_squares_fit(&x, &y, None, 1, &mut coeffs);
    assert_eq!(status, Status::Success);
    assert!((coeffs[0] - 3.0).abs() < 1e-4);
    assert!((coeffs[1] - 2.0).abs() < 1e-4);

    let val = polynomial_eval_f32(&coeffs, 2.5);
    assert!((val - 8.0).abs() < 1e-4);

    // Quadratic fit: y = 1.0 - 0.5 * x + 2.0 * x^2
    let mut y_quad = [0.0f32; 5];
    for i in 0..5 {
        y_quad[i] = 1.0 - 0.5 * x[i] + 2.0 * x[i] * x[i];
    }
    let mut quad_coeffs = [0.0f32; 3];
    let status_quad = polynomial_least_squares_fit(&x, &y_quad, None, 2, &mut quad_coeffs);
    assert_eq!(status_quad, Status::Success);
    assert!((quad_coeffs[0] - 1.0).abs() < 1e-3);
    assert!((quad_coeffs[1] - (-0.5)).abs() < 1e-3);
    assert!((quad_coeffs[2] - 2.0).abs() < 1e-3);
}

// =========================================================================================
// 24. 2D SPATIAL & IMAGE PROCESSING TESTS
// =========================================================================================

#[test]
fn test_spatial_and_image_processing() {
    // 4x4 matrix
    let src = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let mut dct_out = [0.0f32; 16];
    let mut idct_out = [0.0f32; 16];

    let status_dct = dct2d_f32(&src, &mut dct_out, 4, 4);
    assert_eq!(status_dct, Status::Success);

    let status_idct = idct2d_f32(&dct_out, &mut idct_out, 4, 4);
    assert_eq!(status_idct, Status::Success);

    // Verify reconstruction through 2D IDCT(2D DCT)
    for i in 0..16 {
        assert!((idct_out[i] - src[i]).abs() < 1e-3);
    }

    // 2D Convolution with 3x3 Box Blur
    let kernel_box = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let mut conv_out = [0.0f32; 16];
    let status_conv = convolve2d_f32(&src, &mut conv_out, 4, 4, &kernel_box, 3, 3, true);
    assert_eq!(status_conv, Status::Success);
    assert!(conv_out[5] > 0.0);

    // 2D Non-linear Filter (Median, Min, Max)
    let mut nonlin_out = [0.0f32; 16];
    let status_nonlin =
        nonlin2d_filter_f32(&src, &mut nonlin_out, 4, 4, 3, NonlinFilterType::Median);
    assert_eq!(status_nonlin, Status::Success);
    assert!(nonlin_out[5] > 0.0);

    // Sobel Edge Detection
    let mut edges = [0.0f32; 16];
    let status_sobel = sobel_edge_detection_f32(&src, &mut edges, 4, 4, 2.0);
    assert_eq!(status_sobel, Status::Success);

    // 2D Histogram
    let mut bins = [0usize; 4];
    let status_hist = histogram_2d_f32(&src, &mut bins, 1.0, 16.0);
    assert_eq!(status_hist, Status::Success);
    let total_hist: usize = bins.iter().sum();
    assert_eq!(total_hist, 16);

    // MSE and PSNR
    let mse = mse_2d_f32(&src, &idct_out);
    assert!(mse < 1e-4);
    let psnr = psnr_2d_f32(&src, &idct_out, 16.0);
    assert!(psnr > 40.0);
}

// =========================================================================================
// 25. SPECTRAL ESTIMATION & WELCH PSD TESTS
// =========================================================================================

#[test]
fn test_welch_psd_and_periodogram() {
    let fs = 1000.0f32;
    let freq = 100.0f32;
    let n_samples = 256;
    let mut signal = [0.0f32; 256];
    for i in 0..n_samples {
        signal[i] = (2.0 * core::f32::consts::PI * freq * (i as f32) / fs).sin();
    }

    let fft_len = 64;
    let mut psd_linear = [0.0f32; 32];
    let status = welch_psd_f32(
        &signal,
        &mut psd_linear,
        fft_len,
        32,
        fs,
        WelchWindow::Hamming,
        false,
    );
    assert_eq!(status, Status::Success);

    // Frequency bin spacing: fs / fft_len = 1000 / 64 = 15.625 Hz
    // Peak expected around index 100 / 15.625 ~ 6 or 7
    let mut max_idx = 0;
    let mut max_val = 0.0f32;
    for (idx, &val) in psd_linear.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    assert!(max_idx == 6 || max_idx == 7);

    // Periodogram in dB
    let mut psd_db = [0.0f32; 32];
    let status_db = periodogram_f32(
        &signal[..64],
        &mut psd_db,
        fft_len,
        fs,
        WelchWindow::Hanning,
        true,
    );
    assert_eq!(status_db, Status::Success);
    assert!(psd_db[max_idx] > psd_db[0]);
}

// =========================================================================================
// 26. NOISE GENERATION & 4-TERM BLACKMAN-HARRIS TESTS
// =========================================================================================

#[test]
fn test_noise_generation_and_blackman_harris() {
    let mut win_bh = [0.0f32; 64];
    blackman_harris_f32(&mut win_bh);
    assert!((win_bh[0] - 0.00006).abs() < 0.01);
    assert!((win_bh[32] - 1.0).abs() < 0.05);

    let mut seed = 123456789u64;
    let mut uniform_buf = [0.0f32; 100];
    uniform_noise_f32(&mut uniform_buf, -1.0, 1.0, &mut seed);
    for &v in &uniform_buf {
        assert!((-1.0..=1.0).contains(&v));
    }

    let mut gaus_buf = [0.0f32; 200];
    gaussian_noise_f32(&mut gaus_buf, 0.0, 1.0, &mut seed);
    let mut mean = 0.0f32;
    mean_f32(&gaus_buf, &mut mean);
    assert!(mean.abs() < 0.3); // Sample mean close to 0
}

// =========================================================================================
// 27. CIRCULAR BUFFER & DELAY LINE TESTS
// =========================================================================================

#[test]
fn test_circular_buffer_operations() {
    let mut cb = CircularBuffer::<f32, 4>::new(0.0);
    assert!(cb.is_empty());
    assert!(!cb.is_full());
    assert_eq!(cb.len(), 0);

    cb.push(10.0);
    assert_eq!(cb.len(), 1);
    assert_eq!(cb.latest(), Some(10.0));
    assert_eq!(cb.get(0), Some(10.0));
    assert_eq!(cb.get(1), None);

    cb.push(20.0);
    cb.push(30.0);
    cb.push(40.0);
    assert!(cb.is_full());
    assert_eq!(cb.len(), 4);
    assert_eq!(cb.latest(), Some(40.0));
    assert_eq!(cb.oldest(), Some(10.0));
    assert_eq!(cb.get(0), Some(40.0)); // x[n]
    assert_eq!(cb.get(1), Some(30.0)); // x[n-1]
    assert_eq!(cb.get(2), Some(20.0)); // x[n-2]
    assert_eq!(cb.get(3), Some(10.0)); // x[n-3]

    // Push past capacity (overwrite oldest)
    cb.push(50.0);
    assert_eq!(cb.len(), 4);
    assert_eq!(cb.latest(), Some(50.0));
    assert_eq!(cb.oldest(), Some(20.0)); // 10.0 was overwritten
    assert_eq!(cb.get(0), Some(50.0));
    assert_eq!(cb.get(3), Some(20.0));

    cb.clear(0.0);
    assert!(cb.is_empty());
}

// =========================================================================================
// 28. WINDOWED-SINC FIR FILTER DESIGN TESTS
// =========================================================================================

#[test]
fn test_windowed_sinc_fir_design() {
    let mut taps = [0.0f32; 31];

    // Low-pass design
    let status_lp = fir_windowed_sinc_lowpass(0.1, &mut taps);
    assert_eq!(status_lp, Status::Success);
    let sum_lp: f32 = taps.iter().sum();
    assert!((sum_lp - 1.0).abs() < 1e-4); // Unity DC gain

    // High-pass design
    let mut hp_taps = [0.0f32; 31];
    let status_hp = fir_windowed_sinc_highpass(0.2, &mut hp_taps);
    assert_eq!(status_hp, Status::Success);
    let sum_hp: f32 = hp_taps.iter().sum();
    assert!(sum_hp.abs() < 1e-3); // DC gain ~ 0 for high-pass

    // Band-pass design
    let mut bp_taps = [0.0f32; 31];
    let status_bp = fir_windowed_sinc_bandpass(0.1, 0.3, &mut bp_taps);
    assert_eq!(status_bp, Status::Success);

    // Band-stop design
    let mut bs_taps = [0.0f32; 31];
    let status_bs = fir_windowed_sinc_bandstop(0.1, 0.3, &mut bs_taps);
    assert_eq!(status_bs, Status::Success);
    let sum_bs: f32 = bs_taps.iter().sum();
    assert!((sum_bs - 1.0).abs() < 1e-3); // DC gain ~ 1.0 for notch/bandstop
}

#[test]
fn test_windowed_sinc_q15_stopband_matches_float() {
    const M: usize = 51;
    let mut taps = [0.0f32; M];
    assert_eq!(fir_windowed_sinc_lowpass(0.1, &mut taps), Status::Success);

    let mut q_taps = [0i16; M];
    assert_eq!(fir_taps_f32_to_q15(&taps, &mut q_taps), Status::Success);

    let mut q_as_f = [0.0f32; M];
    for i in 0..M {
        q_as_f[i] = q_taps[i] as f32 / 32768.0;
    }

    let h_dc_f = response_magnitude(fir_frequency_response(&taps, 0.0));
    let h_dc_q = response_magnitude(fir_frequency_response(&q_as_f, 0.0));
    assert!((h_dc_f - 1.0).abs() < 1e-3);
    assert!((h_dc_q - 1.0).abs() < 0.01);

    let mut max_stop_q = 0.0f32;
    let mut max_q_err = 0.0f32;
    for k in 0..16 {
        let f = 0.22 + 0.28 * k as f32 / 15.0;
        let hf = response_magnitude(fir_frequency_response(&taps, f));
        let hq = response_magnitude(fir_frequency_response(&q_as_f, f));
        max_stop_q = max_stop_q.max(hq);
        max_q_err = max_q_err.max((hf - hq).abs());
    }
    assert!(max_stop_q < 0.05, "quantized stopband mag {max_stop_q}");
    assert!(
        max_q_err < 0.02,
        "float vs Q15 stopband abs err {max_q_err}"
    );
}

// =========================================================================================
// 29. FILTER ANALYSIS TESTS (FREQUENCY RESPONSE, GROUP DELAY, STABILITY)
// =========================================================================================

#[test]
fn test_fir_frequency_response_and_group_delay() {
    // Symmetric linear-phase averaging FIR: constant group delay of (M-1)/2 = 1.0 sample.
    let taps = [0.25f32, 0.5, 0.25];

    // DC gain should be 1.0 (matches the sum of taps).
    let h_dc = fir_frequency_response(&taps, 0.0);
    assert!((response_magnitude(h_dc) - 1.0).abs() < 1e-4);
    assert!(response_phase(h_dc).abs() < 1e-4);

    // Nyquist (freq_norm = 0.5): H = 0.25 - 0.5 + 0.25 = 0.0, a null.
    let h_nyquist = fir_frequency_response(&taps, 0.5);
    assert!(response_magnitude(h_nyquist) < 1e-4);

    for &f in &[0.05f32, 0.15, 0.25, 0.4] {
        let delay = fir_group_delay(&taps, f);
        assert!((delay - 1.0).abs() < 1e-3);
    }
}

#[test]
fn test_biquad_frequency_response_and_stability() {
    let lp = biquad_lowpass_coeffs(1000.0, 48000.0, core::f32::consts::FRAC_1_SQRT_2);

    // A well-formed lowpass biquad must be stable (poles inside the unit circle).
    assert!(biquad_is_stable(&lp));
    assert!(biquad_pole_radius(&lp) < 1.0);

    // DC gain of a properly normalized lowpass biquad should be unity.
    let h_dc = biquad_frequency_response(&lp, 0.0);
    assert!((response_magnitude(h_dc) - 1.0).abs() < 1e-3);

    // An explicitly unstable section (pole outside the unit circle) must be flagged.
    let unstable = [1.0f32, 0.0, 0.0, 1.5, 0.0];
    assert!(!biquad_is_stable(&unstable));
    assert!(biquad_pole_radius(&unstable) > 1.0);

    let mut butter = [0.0f32; 10];
    butterworth_lowpass_biquads(1000.0, 48000.0, 4, &mut butter);
    assert!(biquad_cascade_is_stable(&butter));

    let h_cascade_dc = biquad_cascade_frequency_response(&butter, 0.0);
    assert!((response_magnitude(h_cascade_dc) - 1.0).abs() < 1e-3);
    assert!(response_magnitude_db(h_cascade_dc).abs() < 0.1);
}

// =========================================================================================
// 30. FAST WALSH-HADAMARD TRANSFORM (FWHT) TESTS
// =========================================================================================

#[test]
fn test_fast_walsh_hadamard_transform() {
    let mut data_f32 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let original = data_f32;

    let status_fwht = fwht_f32(&mut data_f32);
    assert_eq!(status_fwht, Status::Success);

    // Hadamard sum property: first element is sum of all elements
    let sum_orig: f32 = original.iter().sum();
    assert_eq!(data_f32[0], sum_orig);

    // Inverse FWHT
    let status_ifwht = ifwht_f32(&mut data_f32);
    assert_eq!(status_ifwht, Status::Success);
    for i in 0..8 {
        assert!((data_f32[i] - original[i]).abs() < 1e-4);
    }

    // 32-bit Integer FWHT
    let mut data_i32 = [1i32, 2, 3, 4];
    let status_i32 = fwht_i32(&mut data_i32);
    assert_eq!(status_i32, Status::Success);
    assert_eq!(data_i32[0], 10); // 1 + 2 + 3 + 4 = 10
}

// =========================================================================================
// 31. CHEBYSHEV, SINGLE-POLE, & RECURSIVE MOVING AVERAGE FILTER TESTS
// =========================================================================================

#[test]
fn test_chebyshev_biquad_stage_matches_book_debug_values() {
    // Data Set 1 from Steven W. Smith, "The Scientist and Engineer's Guide to DSP", Table 20-6
    // (low-pass, no ripple, 4-pole filter, pole-pair 1).
    let stage1 = chebyshev_biquad_stage(0.1, false, 0.0, 4, 1);
    let expected1 = [0.061885f32, 0.123770, 0.061885, 1.048600, -0.296140];
    for (got, want) in stage1.iter().zip(expected1.iter()) {
        assert!(
            (got - want).abs() < 1e-4,
            "got {stage1:?} want {expected1:?}"
        );
    }

    // Data Set 2 from Table 20-6 (high-pass, 10% ripple, 4-pole filter, pole-pair 2).
    let stage2 = chebyshev_biquad_stage(0.1, true, 10.0, 4, 2);
    let expected2 = [0.922920f32, -1.845840, 0.922920, 1.446913, -0.836654];
    for (got, want) in stage2.iter().zip(expected2.iter()) {
        assert!(
            (got - want).abs() < 1e-3,
            "got {stage2:?} want {expected2:?}"
        );
    }
}

#[test]
fn test_chebyshev_lowpass_and_highpass_cascade_design() {
    let mut lp = [0.0f32; 10];
    chebyshev_lowpass_biquads(0.1, 0.5, 4, &mut lp);
    assert!(biquad_cascade_is_stable(&lp));
    let h_dc = biquad_cascade_frequency_response(&lp, 0.0);
    assert!((response_magnitude(h_dc) - 1.0).abs() < 1e-3);

    let mut hp = [0.0f32; 10];
    chebyshev_highpass_biquads(0.1, 0.5, 4, &mut hp);
    assert!(biquad_cascade_is_stable(&hp));
    let h_nyquist = biquad_cascade_frequency_response(&hp, 0.5);
    assert!((response_magnitude(h_nyquist) - 1.0).abs() < 1e-3);
}

#[test]
fn test_single_pole_filters() {
    // A step input through a low-pass single-pole filter should settle to unity gain.
    let decay = single_pole_decay_from_cutoff(0.05);
    let mut lp = SinglePoleFilter::lowpass(decay);
    let mut y = 0.0;
    for _ in 0..500 {
        y = lp.process(1.0);
    }
    assert!((y - 1.0).abs() < 1e-3);

    // A step (DC) input through a high-pass single-pole filter should decay to zero.
    let mut hp = SinglePoleFilter::highpass(decay);
    let mut y_hp = 0.0;
    for _ in 0..500 {
        y_hp = hp.process(1.0);
    }
    assert!(y_hp.abs() < 1e-3);

    let mut lp_q = SinglePoleFilterQ15::lowpass_from_f32(decay);
    let mut hp_q = SinglePoleFilterQ15::highpass_from_f32(decay);
    let mut yq = 0i16;
    let mut yq_hp = 0i16;
    for _ in 0..500 {
        yq = lp_q.process(32767);
        yq_hp = hp_q.process(32767);
    }
    assert!(
        (yq as i32 - 32767).abs() < 400,
        "q15 LP step settled at {yq}"
    );
    assert!(yq_hp.abs() < 400, "q15 HP step settled at {yq_hp}");
}

#[test]
fn test_recursive_moving_average_matches_naive_average() {
    let mut rma = RecursiveMovingAverage::<4>::new();
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut outputs = [0.0f32; 6];
    for (i, &x) in input.iter().enumerate() {
        outputs[i] = rma.process(x);
    }

    // Growing window until full, then a proper 4-point moving average thereafter.
    let expected = [1.0f32, 1.5, 2.0, 2.5, 3.5, 4.5];
    for (got, want) in outputs.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-4,
            "got {outputs:?} want {expected:?}"
        );
    }

    let mut rma_q = RecursiveMovingAverageQ15::<4>::new();
    let input_q = [1000i16, 2000, 3000, 4000, 5000, 6000];
    let expected_q = [1000i16, 1500, 2000, 2500, 3500, 4500];
    for (i, &x) in input_q.iter().enumerate() {
        assert_eq!(rma_q.process(x), expected_q[i]);
    }

    let decay = single_pole_decay_from_cutoff(0.05);
    let mut dc = DcBlockerQ15::from_f32_decay(decay);
    let mut y_dc = 0i16;
    for _ in 0..500 {
        y_dc = dc.process(32767);
    }
    assert!(y_dc.abs() < 400, "dc blocker settled at {y_dc}");
}

// =========================================================================================
// 32. HAAR & HARTLEY TRANSFORM TESTS
// =========================================================================================

#[test]
fn test_haar_transform_f32_matches_hand_computation_and_round_trips() {
    // Hand-computed via Jörg Arndt, "Matters Computational", Ch. 24, `haar_inplace`.
    let mut data = [1.0f32, 0.0, 0.0, 0.0];
    let status = haar_transform_f32(&mut data);
    assert_eq!(status, Status::Success);
    let expected = [0.5f32, core::f32::consts::FRAC_1_SQRT_2, 0.5, 0.0];
    for (got, want) in data.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-4, "got {data:?} want {expected:?}");
    }

    // Energy (Parseval) is preserved: the transform is orthogonal.
    let energy_out: f32 = data.iter().map(|v| v * v).sum();
    assert!((energy_out - 1.0).abs() < 1e-4);

    // Forward + inverse round-trips to the original signal.
    let original = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut round_trip = original;
    assert_eq!(haar_transform_f32(&mut round_trip), Status::Success);
    assert_eq!(inverse_haar_transform_f32(&mut round_trip), Status::Success);
    for (got, want) in round_trip.iter().zip(original.iter()) {
        assert!(
            (got - want).abs() < 1e-3,
            "got {round_trip:?} want {original:?}"
        );
    }
}

#[test]
fn test_haar_transform_i32_is_non_normalized_and_forward_only() {
    let mut data = [1i32, 2, 3, 4];
    let status = haar_transform_i32(&mut data);
    assert_eq!(status, Status::Success);
    // DC term after the non-normalized transform is the exact sum of the inputs.
    assert_eq!(data[0], 10);
    assert_eq!(data, [10, -1, -4, -1]);
}

#[test]
fn test_hartley_transform_f32_is_self_inverse() {
    // A unit impulse's Hartley transform is a constant 1/sqrt(n).
    let mut impulse = [1.0f32, 0.0, 0.0, 0.0];
    assert_eq!(hartley_transform_f32(&mut impulse), Status::Success);
    for &v in impulse.iter() {
        assert!((v - 0.5).abs() < 1e-4, "impulse response: {impulse:?}");
    }

    // H[H[a]] == a: applying the transform twice recovers the original signal.
    let original = [1.0f32, 2.0, 3.0, 4.0];
    let mut data = original;
    assert_eq!(hartley_transform_f32(&mut data), Status::Success);
    let expected_once = [5.0f32, -2.0, -1.0, 0.0];
    for (got, want) in data.iter().zip(expected_once.iter()) {
        assert!(
            (got - want).abs() < 1e-3,
            "got {data:?} want {expected_once:?}"
        );
    }
    assert_eq!(hartley_transform_f32(&mut data), Status::Success);
    for (got, want) in data.iter().zip(original.iter()) {
        assert!((got - want).abs() < 1e-3, "got {data:?} want {original:?}");
    }
}

// =========================================================================================
// 33. GENERALIZED WAVELET TRANSFORM TESTS
// =========================================================================================

#[test]
fn test_wavelet_transform_daubechies4_round_trips_and_preserves_energy() {
    let original = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut data = original;

    assert_eq!(
        wavelet_transform_f32(&mut data, &DAUBECHIES_4),
        Status::Success
    );

    // Orthogonal transform: energy (sum of squares) is preserved.
    let energy_in: f32 = original.iter().map(|v| v * v).sum();
    let energy_out: f32 = data.iter().map(|v| v * v).sum();
    assert!((energy_in - energy_out).abs() < 1e-3);

    assert_eq!(
        inverse_wavelet_transform_f32(&mut data, &DAUBECHIES_4),
        Status::Success
    );
    for (got, want) in data.iter().zip(original.iter()) {
        assert!((got - want).abs() < 1e-3, "got {data:?} want {original:?}");
    }
}

#[test]
fn test_wavelet_step_rejects_bad_arguments() {
    let mut data = [1.0f32, 2.0, 3.0];
    // Not a power of 2.
    assert_eq!(
        wavelet_step_f32(&mut data, 3, &DAUBECHIES_4),
        Status::ArgumentError
    );

    let mut data8 = [0.0f32; 8];
    // Odd-length filter is invalid.
    assert_eq!(
        wavelet_step_f32(&mut data8, 8, &[1.0, 2.0, 3.0]),
        Status::ArgumentError
    );
}

// =========================================================================================
// 34. AUDIO COMPANDING TESTS
// =========================================================================================

#[test]
fn test_mu_law_and_a_law_companding_round_trip_and_expand_small_signals() {
    for &x in &[-0.9f32, -0.3, -0.05, 0.0, 0.05, 0.3, 0.9] {
        let mu_compressed = mu_law_compress_f32(x);
        assert!((-1.0..=1.0).contains(&mu_compressed));
        let mu_round_trip = mu_law_expand_f32(mu_compressed);
        assert!(
            (mu_round_trip - x).abs() < 1e-4,
            "mu-law round trip failed for x={x}: got {mu_round_trip}"
        );

        let a_compressed = a_law_compress_f32(x);
        assert!((-1.0..=1.0).contains(&a_compressed));
        let a_round_trip = a_law_expand_f32(a_compressed);
        assert!(
            (a_round_trip - x).abs() < 1e-4,
            "A-law round trip failed for x={x}: got {a_round_trip}"
        );
    }

    for &x in &[0i16, 16, -16, 256, -256, 1024, -1024, 8000, -8000, 16000] {
        let u = linear_to_ulaw(x);
        let back = ulaw_to_linear(u);
        assert!(
            (back as i32 - x as i32).abs() < 260,
            "ulaw {x} -> {u} -> {back}"
        );
        let a = linear_to_alaw(x);
        let back_a = alaw_to_linear(a);
        assert!(
            (back_a as i32 - x as i32).abs() < 260,
            "alaw {x} -> {a} -> {back_a}"
        );
    }

    // Companding expands resolution for small signals: a small input should map to a
    // proportionally larger compressed magnitude (the whole point of the nonlinearity).
    let small = 0.01f32;
    assert!(mu_law_compress_f32(small) > small);
    assert!(a_law_compress_f32(small) > small);
}

// =========================================================================================
// 35. CUSTOM FIR FILTER DESIGN (FREQUENCY SAMPLING) TESTS
// =========================================================================================

#[test]
fn test_fir_custom_frequency_sampling_matches_impulse_case() {
    // A flat, zero-phase desired response (magnitude 1 at every bin) corresponds to an
    // impulse in the time domain; after centering and windowing, only the center tap should
    // be nonzero (the Hamming window is exactly 1.0 at its own center).
    let fft_len = 8;
    let half_spec = fft_len / 2 + 1;
    let desired_real = [1.0f32; 5];
    let desired_imag = [0.0f32; 5];
    assert_eq!(desired_real.len(), half_spec);

    let mut taps = [0.0f32; 5];
    let status = fir_custom_frequency_sampling(&desired_real, &desired_imag, fft_len, &mut taps);
    assert_eq!(status, Status::Success);

    let expected = [0.0f32, 0.0, 1.0, 0.0, 0.0];
    for (got, want) in taps.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-4, "got {taps:?} want {expected:?}");
    }
}

#[test]
fn test_fir_custom_frequency_sampling_approximates_lowpass() {
    // Build a desired lowpass magnitude response (1 below cutoff, 0 above), zero phase.
    let fft_len = 64;
    let half_spec = fft_len / 2 + 1;
    let cutoff_bin = half_spec / 4;
    let mut desired_real = [0.0f32; 33];
    let desired_imag = [0.0f32; 33];
    for (k, v) in desired_real.iter_mut().enumerate().take(half_spec) {
        *v = if k <= cutoff_bin { 1.0 } else { 0.0 };
    }

    let mut taps = [0.0f32; 31];
    let status = fir_custom_frequency_sampling(
        &desired_real[..half_spec],
        &desired_imag[..half_spec],
        fft_len,
        &mut taps,
    );
    assert_eq!(status, Status::Success);

    // Passband (DC) gain should be near 1, stopband (near Nyquist) should be attenuated.
    let h_dc = fir_frequency_response(&taps, 0.0);
    let h_stop = fir_frequency_response(&taps, 0.45);
    assert!((response_magnitude(h_dc) - 1.0).abs() < 0.1);
    assert!(response_magnitude(h_stop) < 0.3);
}

// =========================================================================================
// 36. AUDIO & TINYML TESTS (GOERTZEL, ENVELOPE FOLLOWERS, MEL FILTERBANK, MFCC)
// =========================================================================================

#[test]
fn test_goertzel_detects_target_frequency_and_rejects_others() {
    let fs = 8000.0f32;
    let target = 1000.0f32;
    let amplitude = 0.8f32;
    let n = 64;

    let mut on_target = GoertzelDetector::new(target, fs);
    let mut off_target = GoertzelDetector::new(2500.0, fs);
    for i in 0..n {
        let x = amplitude * (2.0 * core::f32::consts::PI * target * i as f32 / fs).sin();
        on_target.process_sample(x);
        off_target.process_sample(x);
    }

    assert!((on_target.magnitude() - amplitude).abs() < 1e-3);
    assert!(off_target.magnitude() < 1e-3);

    let mut on_q = GoertzelDetectorQ15::new(target, fs);
    let mut off_q = GoertzelDetectorQ15::new(2500.0, fs);
    for i in 0..n {
        let x = amplitude * (2.0 * core::f32::consts::PI * target * i as f32 / fs).sin();
        let xq = (x * 32767.0) as i16;
        on_q.process_sample(xq);
        off_q.process_sample(xq);
    }
    let on_mag = on_q.magnitude() as f32 / 32767.0;
    let off_mag = off_q.magnitude() as f32 / 32767.0;
    assert!(
        (on_mag - amplitude).abs() < 0.08,
        "q15 on-target mag {on_mag}"
    );
    assert!(off_mag < 0.05, "q15 off-target mag {off_mag}");
}

#[test]
fn test_peak_and_rms_envelope_followers_converge_to_constant_input() {
    let mut peak = PeakEnvelopeFollower::new(5.0, 50.0);
    let mut rms = RmsEnvelopeFollower::new(20.0);

    let mut peak_env = 0.0;
    let mut rms_env = 0.0;
    for _ in 0..2000 {
        peak_env = peak.process(0.5);
        rms_env = rms.process(0.5);
    }

    assert!((peak_env - 0.5).abs() < 1e-3);
    assert!((rms_env - 0.5).abs() < 1e-3);

    peak.reset();
    rms.reset();
    assert_eq!(peak.process(0.0), 0.0);
    assert_eq!(rms.process(0.0), 0.0);

    let mut peak_q = PeakEnvelopeFollowerQ15::new(5.0, 50.0);
    let mut rms_q = RmsEnvelopeFollowerQ15::new(20.0);
    let xq = (0.5 * 32767.0) as i16;
    let mut peak_env_q = 0i16;
    let mut rms_env_q = 0i16;
    for _ in 0..2000 {
        peak_env_q = peak_q.process(xq);
        rms_env_q = rms_q.process(xq);
    }
    assert!((peak_env_q as i32 - xq as i32).abs() < 400);
    assert!((rms_env_q as i32 - xq as i32).abs() < 800);
}

#[test]
fn test_mel_scale_round_trip() {
    for &hz in &[100.0f32, 440.0, 1000.0, 4000.0] {
        let round_trip = mel_to_hz(hz_to_mel(hz));
        assert!(
            (round_trip - hz).abs() < 1e-2,
            "hz={hz} round_trip={round_trip}"
        );
    }
}

#[test]
fn test_mel_filterbank_impulse_activates_expected_band() {
    let fft_size = 64;
    let sample_rate = 8000.0;
    let num_bins = fft_size / 2 + 1;

    let mut power_spectrum = [0.0f32; 33];
    power_spectrum[10] = 1.0;

    let mut mel_energies = [0.0f32; 8];
    let status = mel_filterbank_f32(
        &power_spectrum[..num_bins],
        fft_size,
        sample_rate,
        0.0,
        sample_rate / 2.0,
        &mut mel_energies,
    );
    assert_eq!(status, Status::Success);

    // Verified against a from-scratch reference implementation: bin 10 sits exactly at the
    // peak of filter index 4, which should receive the full impulse energy.
    let expected = [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    for (got, want) in mel_energies.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-4,
            "got {mel_energies:?} want {expected:?}"
        );
    }
}

#[test]
fn test_mfcc_f32_produces_finite_distinguishable_coefficients() {
    let fs = 8000.0f32;
    let fft_size = 256;

    let make_frame = |freq: f32| {
        let mut frame = [0.0f32; 256];
        for (i, x) in frame.iter_mut().enumerate() {
            *x = (2.0 * core::f32::consts::PI * freq * i as f32 / fs).sin();
        }
        let mut window = [0.0f32; 256];
        hamming_f32(&mut window);
        apply_window_f32(&mut frame, &window);
        frame
    };

    let frame_low = make_frame(300.0);
    let frame_high = make_frame(2500.0);

    let mut mel_scratch_low = [0.0f32; 26];
    let mut mfcc_low = [0.0f32; 13];
    let status_low = mfcc_f32(
        &frame_low,
        fs,
        0.0,
        fs / 2.0,
        &mut mel_scratch_low,
        &mut mfcc_low,
    );
    assert_eq!(status_low, Status::Success);
    assert_eq!(fft_size, frame_low.len());

    let mut mel_scratch_high = [0.0f32; 26];
    let mut mfcc_high = [0.0f32; 13];
    let status_high = mfcc_f32(
        &frame_high,
        fs,
        0.0,
        fs / 2.0,
        &mut mel_scratch_high,
        &mut mfcc_high,
    );
    assert_eq!(status_high, Status::Success);

    for &v in mfcc_low.iter().chain(mfcc_high.iter()) {
        assert!(v.is_finite());
    }

    // Different input frequencies must produce distinguishable feature vectors.
    let diff: f32 = mfcc_low
        .iter()
        .zip(mfcc_high.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.1, "MFCC vectors too similar: diff={diff}");
}

#[test]
fn test_q16_and_lut_trig() {
    let product = from_q16(mul_q16(to_q16(1.5), to_q16(2.0)));
    assert!((product - 3.0).abs() < 0.01);

    let s = fast_sin_i16(core::f32::consts::PI / 2.0) as f32 / 32768.0;
    assert!((s - 1.0).abs() < 0.02);

    let s16 = from_q16(sin_q16(angle_to_q16(90.0)));
    assert!((s16 - 1.0).abs() < 0.05);
}

#[test]
fn test_integer_cfft_q15_tone_and_roundtrip_scale() {
    const N: usize = 32;
    let mut q = [0i16; 64];
    let mut f = [0.0f32; 64];
    for i in 0..N {
        let x = (2.0 * core::f32::consts::PI * 4.0 * i as f32 / N as f32).sin();
        f[2 * i] = x;
        q[2 * i] = (x * 32767.0) as i16;
    }
    let orig = q;
    cfft_f32(&mut f, N, 0, 1);
    cfft_q15(&mut q, N, 0, 1);

    let mut peak_f = 0usize;
    let mut peak_q = 0usize;
    let mut mag_f = 0.0f32;
    let mut mag_q = 0i32;
    for k in 0..N {
        let mf = f[2 * k] * f[2 * k] + f[2 * k + 1] * f[2 * k + 1];
        if mf > mag_f {
            mag_f = mf;
            peak_f = k;
        }
        let mq = (q[2 * k] as i32).pow(2) + (q[2 * k + 1] as i32).pow(2);
        if mq > mag_q {
            mag_q = mq;
            peak_q = k;
        }
    }
    assert_eq!(peak_f, peak_q);
    assert_eq!(peak_f, 4);

    cfft_q15(&mut q, N, 1, 1);
    let mut err = 0i32;
    for i in 0..N {
        let got = (q[2 * i] as i32) * (N as i32);
        err += (got - orig[2 * i] as i32).abs();
    }
    let mean = err / (N as i32);
    assert!(
        mean < 4000,
        "round-trip abs err sum/N = {mean}; first got*N={} orig={}",
        (q[0] as i32) * N as i32,
        orig[0]
    );
}

#[test]
fn test_sqrt_q15_and_atan2_q15_quadrants() {
    let mut s = 0i16;
    assert_eq!(sqrt_q15(16384, &mut s), Status::Success);
    // 0.5 in Q15 → sqrt ≈ 0.707 → ~23170
    assert!((s as i32 - 23170).abs() < 200);

    let mut a = 0i16;
    assert_eq!(atan2_q15(32767, 32767, &mut a), Status::Success);
    assert!(
        a > 7000 && a < 10000,
        "π/4 as Q15/π expected ~8192, got {a}"
    );
    assert_eq!(atan2_q15(32767, 0, &mut a), Status::Success);
    assert!(a > 14000, "π/2 expected ~16384, got {a}");
    assert_eq!(atan2_q15(0, -32767, &mut a), Status::Success);
    assert!(a > 14000 || a < -14000, "±π expected, got {a}");
}

#[test]
fn test_biquad_q15_matches_f32_lowpass() {
    let coeffs = biquad_lowpass_coeffs(800.0, 8000.0, core::f32::consts::FRAC_1_SQRT_2);
    let post_shift = 1u8;
    let mut qcoeffs = [0i16; 5];
    assert_eq!(
        biquad_coeffs_f32_to_q15(&coeffs, &mut qcoeffs, post_shift),
        Status::Success
    );

    let mut state_f = [0.0f32; 4];
    let mut state_q = [0i16; 4];
    let mut bq_f = BiquadCascadeInstanceF32::init(1, &coeffs, &mut state_f);
    let mut bq_q = BiquadCascadeInstanceQ15::init(1, &qcoeffs, &mut state_q, post_shift);

    let mut max_abs_err = 0i32;
    for n in 0..128 {
        let x = (2.0 * core::f32::consts::PI * 200.0 * n as f32 / 8000.0).sin() * 0.5;
        let mut yf = [0.0f32; 1];
        let mut yq = [0i16; 1];
        let xq = [(x * 32767.0) as i16];
        biquad_cascade_df1_f32(&mut bq_f, &[x], &mut yf);
        biquad_cascade_df1_q15(&mut bq_q, &xq, &mut yq);
        let expected_q = (yf[0] * 32767.0) as i32;
        let err = (yq[0] as i32 - expected_q).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
    }
    assert!(
        max_abs_err < 1500,
        "max absolute error {max_abs_err} exceeded threshold (Q15 tracking of f32 biquad)"
    );
}

#[test]
fn test_biquad_df2t_q15_matches_df1() {
    let coeffs = biquad_lowpass_coeffs(800.0, 8000.0, core::f32::consts::FRAC_1_SQRT_2);
    let post_shift = 1u8;
    let mut qcoeffs = [0i16; 5];
    assert_eq!(
        biquad_coeffs_f32_to_q15(&coeffs, &mut qcoeffs, post_shift),
        Status::Success
    );

    let mut state_df1 = [0i16; 4];
    let mut df1 = BiquadCascadeInstanceQ15::init(1, &qcoeffs, &mut state_df1, post_shift);
    let mut state_df2t = [0i16; 2];
    let mut df2t = BiquadCascadeDf2tInstanceQ15::init(1, &qcoeffs, &mut state_df2t, post_shift);

    let mut max_err = 0i32;
    for n in 0..64 {
        let x = (2.0 * core::f32::consts::PI * 200.0 * n as f32 / 8000.0).sin() * 0.5;
        let xq = [(x * 32767.0) as i16];
        let mut y1 = [0i16; 1];
        let mut y2 = [0i16; 1];
        biquad_cascade_df1_q15(&mut df1, &xq, &mut y1);
        biquad_cascade_df2t_q15(&mut df2t, &xq, &mut y2);
        max_err = max_err.max((y1[0] as i32 - y2[0] as i32).abs());
    }
    assert!(max_err < 2500, "DF1 vs DF2T max abs {max_err}");

    let coeffs = biquad_lowpass_coeffs(800.0, 8000.0, core::f32::consts::FRAC_1_SQRT_2);
    let mut state_f = [0.0f32; 4];
    let mut df1f = BiquadCascadeInstanceF32::init(1, &coeffs, &mut state_f);
    let mut state_t = [0.0f32; 2];
    let mut df2tf = BiquadCascadeDf2tInstanceF32::init(1, &coeffs, &mut state_t);
    let mut max_f = 0.0f32;
    for n in 0..64 {
        let x = (2.0 * core::f32::consts::PI * 200.0 * n as f32 / 8000.0).sin() * 0.5;
        let mut y1 = [0.0f32; 1];
        let mut y2 = [0.0f32; 1];
        biquad_cascade_df1_f32(&mut df1f, &[x], &mut y1);
        biquad_cascade_df2t_f32(&mut df2tf, &[x], &mut y2);
        max_f = max_f.max((y1[0] - y2[0]).abs());
    }
    assert!(max_f < 1e-4, "f32 DF1 vs DF2T {max_f}");
}

#[test]
fn test_packed_rfft_q15_tone_bin() {
    const N: usize = 32;
    let mut src_f = [0.0f32; N];
    let mut src_q = [0i16; N];
    for i in 0..N {
        let x = (2.0 * core::f32::consts::PI * 4.0 * i as f32 / N as f32).sin();
        src_f[i] = x;
        src_q[i] = (x * 32767.0) as i16;
    }
    let mut dst_f = [0.0f32; 2 * N];
    let mut dst_q = [0i16; 2 * N];
    rfft_f32(&src_f, &mut dst_f, N, 0);
    rfft_q15(&src_q, &mut dst_q, N, 0);

    let mut peak_f = 0usize;
    let mut peak_q = 0usize;
    let mut mag_f = 0.0f32;
    let mut mag_q = 0i32;
    for k in 0..N {
        let mf = dst_f[2 * k] * dst_f[2 * k] + dst_f[2 * k + 1] * dst_f[2 * k + 1];
        if mf > mag_f {
            mag_f = mf;
            peak_f = k;
        }
        let mq = (dst_q[2 * k] as i32).pow(2) + (dst_q[2 * k + 1] as i32).pow(2);
        if mq > mag_q {
            mag_q = mq;
            peak_q = k;
        }
    }
    assert_eq!(peak_f, 4);
    assert_eq!(peak_q, peak_f);

    let q_peak =
        ((dst_q[2 * peak_q] as f32).hypot(dst_q[2 * peak_q + 1] as f32) / 32767.0) * N as f32;
    let f_peak = dst_f[2 * peak_f].hypot(dst_f[2 * peak_f + 1]);
    assert!(
        (q_peak - f_peak).abs() / f_peak.max(1e-6) < 0.25,
        "packed rfft scale q={q_peak} f={f_peak}"
    );

    let orig = src_q;
    let mut time = [0i16; N];
    irfft_q15(&dst_q, &mut time, N);
    let mut err = 0i32;
    for i in 0..N {
        let got = (time[i] as i32) * (N as i32);
        err += (got - orig[i] as i32).abs();
    }
    let mean = err / N as i32;
    assert!(
        mean < 5000,
        "irfft round-trip mean abs {mean}; got*N={} orig={}",
        time[4] as i32 * N as i32,
        orig[4]
    );
}

#[test]
fn test_strongly_typed_q15_q31_and_dsp_sample() {
    // Q15 arithmetic
    let a = Q15::from_f32(0.5);
    let b = Q15::from_f32(0.25);
    let c = a + b;
    assert!((c.to_f32() - 0.75).abs() < 1e-4);

    let d = a * b;
    assert!((d.to_f32() - 0.125).abs() < 1e-3);

    let e = a - b;
    assert!((e.to_f32() - 0.25).abs() < 1e-4);

    let f = b / a;
    assert!((f.to_f32() - 0.5).abs() < 1e-3);

    let neg = -a;
    assert!((neg.to_f32() - (-0.5)).abs() < 1e-4);

    // Q15 saturation
    let sat_max = Q15::from_f32(0.8) + Q15::from_f32(0.8);
    assert_eq!(sat_max, Q15::MAX);

    // Q31 arithmetic
    let q31_a = Q31::from_f32(0.5);
    let q31_b = Q31::from_f32(0.25);
    let q31_c = q31_a * q31_b;
    assert!((q31_c.to_f32() - 0.125).abs() < 1e-5);

    // Generic DspSample function
    fn generic_lerp<T: DspSample>(x0: T, x1: T, t: T) -> T {
        x0 + (x1 - x0) * t
    }

    let f_lerp = generic_lerp(0.0f32, 10.0f32, 0.5f32);
    assert!((f_lerp - 5.0).abs() < 1e-5);

    let q_lerp = generic_lerp(Q15::from_f32(0.0), Q15::from_f32(0.8), Q15::from_f32(0.5));
    assert!((q_lerp.to_f32() - 0.4).abs() < 1e-3);

    // Complex operator overloads
    let c1 = Complex::new(0.5f32, 1.0f32);
    let c2 = Complex::new(2.0f32, 3.0f32);
    let c_sum = c1 + c2;
    assert_eq!(c_sum.real, 2.5);
    assert_eq!(c_sum.imag, 4.0);

    let c_mul = c1 * c2;
    // (0.5 + 1i)(2 + 3i) = (1 - 3) + (1.5 + 2)i = -2.0 + 3.5i
    assert!((c_mul.real - (-2.0)).abs() < 1e-5);
    assert!((c_mul.imag - 3.5).abs() < 1e-5);
}

#[test]
fn test_simd_dsp_intrinsics() {
    let a = [1000i16, 2000, -3000, 4000, 500, -600, 700, 800];
    let b = [2000i16, -1000, 4000, 3000, 200, 300, -400, 100];

    // Dot product
    let simd_dot = simd_dot_prod_q15(&a, &b);
    let mut expected_dot: q63 = 0;
    for i in 0..a.len() {
        expected_dot += (a[i] as i32 * b[i] as i32) as q63;
    }
    assert_eq!(simd_dot, expected_dot);

    // Add and Sub
    let mut dst_add = [0i16; 8];
    let mut dst_sub = [0i16; 8];
    simd_add_q15(&a, &b, &mut dst_add);
    simd_sub_q15(&a, &b, &mut dst_sub);

    for i in 0..8 {
        assert_eq!(dst_add[i], a[i].saturating_add(b[i]));
        assert_eq!(dst_sub[i], a[i].saturating_sub(b[i]));
    }

    // Mult
    let mut dst_mult = [0i16; 8];
    simd_mult_q15(&a, &b, &mut dst_mult);
    for i in 0..8 {
        assert_eq!(dst_mult[i], q15_mult(a[i], b[i]));
    }

    // Dual saturating primitives
    let p_a = 0x7FFF_7FFFu32; // [32767, 32767]
    let p_b = 0x0001_0001u32; // [1, 1]
    let p_sat = dual_saturating_add_q15(p_a, p_b);
    assert_eq!(p_sat, 0x7FFF_7FFFu32);
}

#[test]
fn test_fixed_point_distance_metrics() {
    let a = [10000i16, 20000, -15000, 5000];
    let b = [10000i16, 20000, -15000, 5000];
    assert_eq!(euclidean_distance_q15(&a, &b), 0);
    assert_eq!(chebyshev_distance_q15(&a, &b), 0);
    assert_eq!(manhattan_distance_q15(&a, &b), 0);
    assert_eq!(hamming_distance_q15(&a, &b), 0);

    let c = [12000i16, 18000, -10000, 4000];
    let euc = euclidean_distance_q15(&a, &c);
    assert!(euc > 0);

    let cheb = chebyshev_distance_q15(&a, &c);
    assert_eq!(cheb, 5000); // |-15000 - (-10000)| = 5000

    let manh = manhattan_distance_q15(&a, &c);
    assert_eq!(manh, 2000 + 2000 + 5000 + 1000);

    let ham = hamming_distance_q15(&a, &c);
    assert_eq!(ham, 4);

    let canb = canberra_distance_q15(&a, &c);
    assert!(canb > 0);

    let bc = bray_curtis_distance_q15(&a, &c);
    assert!(bc > 0);
}

#[test]
fn test_cic_gain_and_polyphase_q15() {
    let mut cic = CicDecimator::<3>::new(4);
    assert_eq!(cic.gain(), 64); // 4^3 = 64
    assert_eq!(cic.gain_bits(), 6); // ceil(log2(64)) = 6

    let in_samples = [1000i32; 16];
    let mut decimated_scaled = None;
    for &s in &in_samples {
        if let Some(d) = cic.process_sample_scaled(s) {
            decimated_scaled = Some(d);
        }
    }
    assert!(decimated_scaled.is_some());

    // Polyphase FIR decimation
    let src = [1000i16, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
    let coeffs = [16384i16, 16384]; // 2-tap moving average / lowpass in Q15
    let mut dst_dec = [0i16; 4];
    let written = polyphase_decimate_q15(&src, &coeffs, 2, &mut dst_dec);
    assert!(written > 0);

    // Polyphase FIR interpolation
    let mut dst_interp = [0i16; 8];
    let written_interp = polyphase_interpolate_q15(&src[..4], &coeffs, 2, &mut dst_interp);
    assert_eq!(written_interp, 8);

    // Linear fractional resampling in Q15
    let mut dst_resampled = [0i16; 8];
    resample_linear_q15(&src, &mut dst_resampled, 65536); // 1.0 ratio
    assert_eq!(dst_resampled[0], src[0]);
}

#[test]
fn test_filter_quantization_and_sqnr_analysis() {
    let lp = biquad_lowpass_coeffs(1000.0, 48000.0, core::f32::consts::FRAC_1_SQRT_2);
    let (headroom, peak) = estimate_biquad_headroom_bits(&lp);
    assert!(peak > 0.0);
    assert!(headroom <= 2);

    let mut q15_coeffs = [0i16; 5];
    let post_shift =
        biquad_quantize_and_scale_q15(&lp, &mut q15_coeffs, ScalingStrategy::LInfNorm).unwrap();

    let sqnr = biquad_quantization_snr_db(&lp, &q15_coeffs, post_shift, 64);
    assert!(sqnr > 40.0, "Expected SQNR > 40 dB, got {sqnr} dB");

    let mut q31_coeffs = [0i32; 5];
    let post_shift_q31 =
        biquad_quantize_and_scale_q31(&lp, &mut q31_coeffs, ScalingStrategy::Direct).unwrap();
    assert!(post_shift_q31 <= 2);

    let mut fir_f32 = [0.0f32; 15];
    fir_windowed_sinc_lowpass(0.2, &mut fir_f32);
    let mut fir_q15 = [0i16; 15];
    fir_quantize_q15(&fir_f32, &mut fir_q15).unwrap();

    let fir_sqnr = fir_quantization_snr_db(&fir_f32, &fir_q15, 64);
    assert!(
        fir_sqnr > 60.0,
        "Expected FIR SQNR > 60 dB, got {fir_sqnr} dB"
    );
}

#[test]
fn test_bfp_fft_and_real_cepstrum() {
    const N: usize = 32;
    let mut data_q15 = [0i16; 2 * N];
    for i in 0..N {
        let x = (2.0 * core::f32::consts::PI * 4.0 * i as f32 / N as f32).sin();
        data_q15[2 * i] = (x * 30000.0) as i16;
    }

    let scale_count = cfft_bfp_q15(&mut data_q15, N, 0, 1);
    assert!(scale_count <= 5);

    // Peak at bin 4
    let mut max_mag = 0i64;
    let mut max_bin = 0;
    for k in 0..N {
        let re = data_q15[2 * k] as i64;
        let im = data_q15[2 * k + 1] as i64;
        let mag = re * re + im * im;
        if mag > max_mag {
            max_mag = mag;
            max_bin = k;
        }
    }
    assert_eq!(max_bin, 4);

    // Q31 BFP FFT
    let mut data_q31 = [0i32; 2 * N];
    for i in 0..N {
        data_q31[2 * i] = (data_q15[2 * i] as i32) << 16;
    }
    let scale_q31 = cfft_bfp_q31(&mut data_q31, N, 0, 1);
    assert!(scale_q31 <= 5);

    // Real Cepstrum
    let mut sig = [0.0f32; 32];
    for (i, val) in sig.iter_mut().enumerate() {
        *val = (2.0 * core::f32::consts::PI * 2.0 * i as f32 / 32.0).cos();
    }
    let mut cep = [0.0f32; 32];
    let status = real_cepstrum_f32(&sig, &mut cep);
    assert_eq!(status, Status::Success);
    assert!(cep[0].is_finite());
}

#[test]
fn test_cordic_engine() {
    // Rotation: 0 rad -> sin ≈ 0 (within 2 LSB), cos ≈ 1
    let (s0, c0) = cordic_sin_cos_q15(0);
    assert!(s0.abs() <= 2);
    assert!((c0 as i32 - 32767).abs() < 100);

    // Rotation: pi/4 rad (25736 in Q15) -> sin ≈ 0.7071, cos ≈ 0.7071
    let (s_pi4, c_pi4) = cordic_sin_cos_q15(25736);
    let expected = (core::f32::consts::FRAC_1_SQRT_2 * 32768.0) as i32;
    assert!((s_pi4 as i32 - expected).abs() < 150);
    assert!((c_pi4 as i32 - expected).abs() < 150);

    // Vectoring: (1.0, 1.0) in Q15 -> mag ≈ 1.414 (scaled), angle ≈ pi/4
    let (mag, angle) = cordic_cartesian_to_polar_q15(10000, 10000);
    assert!((angle as i32 - 25736).abs() < 150);
    assert!((mag as i32 - 14142).abs() < 200);

    // Atan2
    let atan_val = cordic_atan2_q15(10000, 10000);
    assert!((atan_val as i32 - 25736).abs() < 150);

    // Sqrt
    let root = cordic_sqrt_q15(16384); // sqrt(0.5) in Q15
    assert!((root as i32 - expected).abs() < 250);
}

#[test]
fn test_dsp_pipeline_and_streaming() {
    use crate::pipeline::*;

    let lowpass = SinglePoleFilter::lowpass(0.1);
    let gain = Gain::new(2.0f32);
    let limiter = Limiter::new(-1.0f32, 1.0f32);

    let mut chain = lowpass.then(gain).then(limiter);

    let mut buffer = [0.0f32, 0.5, 1.0, 2.0, -3.0];
    chain.process_in_place(&mut buffer);

    for &s in &buffer {
        assert!((-1.0..=1.0).contains(&s));
    }

    // Q15 DC blocker in pipeline
    let dc = DcBlockerQ15::new(32000);
    let q_gain = Gain::new(16384i16);
    let mut q_chain = dc.then(q_gain);

    let mut q_buf = [10000i16; 8];
    q_chain.process_in_place(&mut q_buf);
    assert!(q_buf[7].abs() < q_buf[0].abs());
}

#[test]
fn test_generalized_filterbank_and_vad() {
    let power_spec = [1.0f32; 16];
    let left = [0, 2, 4];
    let center = [2, 4, 6];
    let right = [4, 6, 8];
    let mut energies = [0.0f32; 3];

    let status =
        generalized_triangular_filterbank(&power_spec, &left, &center, &right, &mut energies);
    assert_eq!(status, Status::Success);
    for &e in &energies {
        assert!(e > 0.0);
    }

    // Fast log2
    let l2 = fast_log2_q15(16384); // 0.5 -> log2 is -1.0
    assert!(l2 < 0);

    // VAD detector
    let vad = VadDetectorQ15::new(10, 2);
    let silence = [0i16; 32];
    assert!(!vad.is_active(&silence));

    let speech = [
        10000i16, -10000, 20000, -20000, 15000, -15000, 10000, -10000,
    ];
    assert!(vad.is_active(&speech));
}

#[test]
fn test_sogi_pll_and_costas_loop() {
    let mut pll = SogiPll::new(50.0, 10000.0, 1.414, 60.0, 1400.0);
    // Simulate 50 Hz sinusoid for 2000 samples (0.2s)
    let mut last_freq = 0.0f32;
    for n in 0..2000 {
        let t = n as f32 / 10000.0;
        let v = (2.0 * core::f32::consts::PI * 50.0 * t).sin();
        let _ = pll.process(v);
        last_freq = pll.frequency_hz();
    }
    assert!(
        (last_freq - 50.0).abs() < 1.0,
        "PLL tracked freq={last_freq}"
    );
    let (v_a, v_b) = pll.orthogonal_components();
    assert!(v_a.is_finite() && v_b.is_finite());

    // Costas Loop
    let mut costas = CostasLoop::new(1000.0, 10000.0, 50.0, 0.707);
    for n in 0..1000 {
        let t = n as f32 / 10000.0;
        let s = (2.0 * core::f32::consts::PI * 1000.0 * t).cos();
        let (i_arm, _q_arm) = costas.process_sample(s);
        assert!(i_arm.is_finite());
    }
}

#[test]
fn test_dynamics_compressor_and_noise_gate() {
    let mut comp = DynamicsCompressor::new(-20.0, 4.0, 6.0, 0.005, 0.05, 0.0, 48000.0);
    // Loud signal (0 dBFS = 1.0)
    let mut out_loud = 0.0f32;
    for _ in 0..500 {
        out_loud = comp.process(1.0);
    }
    // High compression ratio: 1.0 should be compressed below 0.6
    assert!(out_loud < 0.6 && out_loud > 0.1);

    // Noise gate: threshold -40 dB (approx 0.01), loud signal passes, quiet attenuated
    let mut gate = NoiseGate::new(-40.0, -40.0, 0.002, 0.02, 48000.0);
    let mut out_loud_gate = 0.0f32;
    for _ in 0..500 {
        out_loud_gate = gate.process(0.5);
    }
    assert!(out_loud_gate > 0.45);

    let mut out_quiet = 0.0f32;
    for _ in 0..1000 {
        out_quiet = gate.process(0.0001);
    }
    assert!(out_quiet < 0.00005);
}

#[test]
fn test_square_root_kalman_filter() {
    // 2-state constant velocity system
    let x0 = [0.0f32, 1.0];
    let s0 = [[1.0f32, 0.0], [0.0, 1.0]]; // S0 = diag(1, 1) => P0 = diag(1, 1)
    let dt = 0.1f32;
    let f = [[1.0f32, dt], [0.0, 1.0]];
    let s_q = [[0.1f32, 0.0], [0.0, 0.1]];
    let h = [[1.0f32, 0.0]]; // Measure position
    let s_r = [[0.5f32]];

    let mut srkf = SquareRootKalmanFilter::new(x0, s0, f, s_q, h, s_r);

    for k in 1..=20 {
        srkf.predict();
        let true_pos = k as f32 * dt;
        let meas = [true_pos + 0.05];
        let status = srkf.update(&meas);
        assert_eq!(status, Status::Success);
    }

    let p = srkf.covariance();
    assert!(
        p[0][0] > 0.0 && p[1][1] > 0.0,
        "P must be positive definite"
    );
    assert!((srkf.x[1] - 1.0).abs() < 0.3, "Velocity estimate tracked");
}

#[test]
fn test_burg_ar_psd_and_kaiser_window() {
    let mut sig = [0.0f32; 64];
    for (i, val) in sig.iter_mut().enumerate() {
        *val = (2.0 * core::f32::consts::PI * 8.0 * i as f32 / 64.0).sin();
    }

    let mut ar_coeffs = [0.0f32; 4];
    let var_res = ar_burg_f32(&sig, 4, &mut ar_coeffs);
    assert!(var_res.is_ok());
    let var = var_res.unwrap();
    assert!(var > 0.0);

    let mut psd = [0.0f32; 32];
    let status = ar_psd_f32(&ar_coeffs, var, 32, &mut psd, false);
    assert_eq!(status, Status::Success);
    // Peak near bin 4 (8 cycles / 64 = bin 4 of 32)
    let mut max_p = 0.0f32;
    let mut max_idx = 0;
    for (i, &p) in psd.iter().enumerate() {
        if p > max_p {
            max_p = p;
            max_idx = i;
        }
    }
    assert!((max_idx as i32 - 8).abs() <= 1, "Peak bin: {max_idx}");

    // Kaiser window
    let mut k_win = [0.0f32; 32];
    kaiser_f32(&mut k_win, 5.0);
    assert!(k_win[16] > k_win[0]); // Peak at center
}

#[test]
fn test_delay_and_sum_beamformer_and_gcc_phat() {
    let mut bf: DelayAndSumBeamformer<2, 32> = DelayAndSumBeamformer::new();
    bf.set_delays(&[0.0, 2.0]);

    // Feed pulse to ch0 at t=2 and ch1 at t=0
    let mut output = [0.0f32; 8];
    for (t, out) in output.iter_mut().enumerate() {
        let ch0 = if t == 2 { 1.0 } else { 0.0 };
        let ch1 = if t == 0 { 1.0 } else { 0.0 };
        *out = bf.process_sample(&[ch0, ch1]);
    }
    // Both signals align at t=2
    assert!(output[2] > 0.4);

    // GCC-PHAT TDoA
    let mut sig1 = [0.0f32; 64];
    let mut sig2 = [0.0f32; 64];
    // Pulse at index 10 for sig1, index 14 for sig2 (delay = 4)
    sig1[10] = 1.0;
    sig1[11] = 0.5;
    sig2[14] = 1.0;
    sig2[15] = 0.5;

    let delay_res = gcc_phat_tdoa_f32(&sig1, &sig2, 16);
    assert!(delay_res.is_ok());
    let delay = delay_res.unwrap();
    assert!((delay.abs() - 4.0).abs() < 0.5, "Estimated delay: {delay}");
}
