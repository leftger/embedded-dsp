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
    assert!((s - 0.70710678).abs() < 1e-4);
    assert!((c - 0.70710678).abs() < 1e-4);

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
    assert!((ent - 0.693147).abs() < 1e-4);

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

#[test]
fn test_bayes_and_svm_classifiers() {
    let theta = [0.0f32, 0.0, 5.0, 5.0];
    let sigma = [1.0f32, 1.0, 1.0, 1.0];
    let priors = [0.5f32, 0.5];

    let gnb = GaussianNaiveBayesInstanceF32 {
        num_classes: 2,
        num_features: 2,
        theta: &theta,
        sigma: &sigma,
        class_prior: &priors,
        epsilon: 1e-9,
    };

    let mut probs = [0.0f32; 2];
    let cls = gnb.predict(&[4.8, 5.1], &mut probs);
    assert_eq!(cls, 1);

    let sv = [0.0f32, 0.0, 2.0, 2.0];
    let dual_coefs = [-1.0f32, 1.0];

    let svm = SvmInstanceF32 {
        num_vector_dim: 2,
        num_support_vectors: 2,
        intercept: 0.0,
        dual_coefs: &dual_coefs,
        support_vectors: &sv,
        kernel_type: SvmKernelType::Linear,
        gamma: 1.0,
        coef0: 0.0,
        degree: 1,
    };

    let mut res = 0;
    assert_eq!(svm.predict(&[3.0, 3.0], &mut res), Status::Success);
    assert_eq!(res, 1);
}
