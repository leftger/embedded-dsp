use embedded_dsp::pipeline::*;
use embedded_dsp::*;

#[test]
fn test_filter_analysis_exhaustive() {
    let coeffs = [0.1f32, 0.2, 0.3, 0.4, 0.5];
    let freq = 0.1f32;
    let resp = biquad_frequency_response(&coeffs, freq);
    assert!(response_magnitude(resp).is_finite());
    assert!(response_magnitude_db(resp).is_finite());
    assert!(response_phase(resp).is_finite());

    let fir_taps = [0.1f32, 0.2, 0.3, 0.4];
    let fir_resp = fir_frequency_response(&fir_taps, freq);
    assert!(response_magnitude(fir_resp).is_finite());

    let cascade_coeffs = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5];
    let cascade_resp = biquad_cascade_frequency_response(&cascade_coeffs, freq);
    assert!(response_magnitude(cascade_resp).is_finite());

    assert!(fir_group_delay(&fir_taps, freq).is_finite());
    assert!(biquad_pole_radius(&coeffs).is_finite());
    assert!(biquad_is_stable(&coeffs));
    assert!(biquad_cascade_is_stable(&cascade_coeffs));

    assert!(biquad_peak_gain(&coeffs, 32).is_finite());
    assert!(biquad_l2_norm(&coeffs, 32).is_finite());

    let (headroom, gain) = estimate_biquad_headroom_bits(&coeffs);
    assert!(gain.is_finite());
    assert!(headroom <= 32);

    let biquad_q15_resp = biquad_q15_frequency_response(&[q15::from_bits(1000); 5], 0, freq);
    assert!(response_magnitude(biquad_q15_resp).is_finite());

    let snr_biquad = biquad_quantization_snr_db(&coeffs, &[q15::from_bits(1000); 5], 0, 32);
    assert!(snr_biquad.is_finite());

    let fir_taps_q15 = [q15::from_bits(1000); 4];
    let snr_fir = fir_quantization_snr_db(&fir_taps, &fir_taps_q15, 32);
    assert!(snr_fir.is_finite());
}

#[test]
fn test_const_generics_exhaustive() {
    let fir_taps = [0.1f32, 0.2, 0.3, 0.4];
    let mut fir = FirFilter::<4>::new(fir_taps);
    let src = [1.0f32; 8];
    let mut dst = [0.0f32; 8];
    fir.process(&src, &mut dst);
    fir.reset();

    let biquad_coeffs = [0.1f32, 0.2, 0.3, 0.4, 0.5];
    let mut biquad = BiquadCascade::<5, 4>::new(biquad_coeffs);
    biquad.process(&src, &mut dst);
    biquad.reset();

    let fir_taps_q15 = [q15::from_bits(1000); 4];
    let mut fir_q15 = FirFilterQ15::<4>::new(fir_taps_q15);
    let src_q15 = [q15::from_bits(1000); 8];
    let mut dst_q15 = [q15::ZERO; 8];
    fir_q15.process(&src_q15, &mut dst_q15);
    fir_q15.reset();

    let biquad_coeffs_q15 = [q15::from_bits(1000); 5];
    let mut biquad_q15 = BiquadCascadeQ15::<5, 4>::new(biquad_coeffs_q15, 0);
    biquad_q15.process(&src_q15, &mut dst_q15);
    biquad_q15.reset();

    let m1 = Matrix::<2, 2, 4>::new([1.0, 2.0, 3.0, 4.0]);
    let m2 = Matrix::<2, 2, 4>::new([5.0, 6.0, 7.0, 8.0]);
    let _m_add = m1.add(&m2);
    let _m_sub = m1.sub(&m2);
    let _m_scale = m1.scale(2.0);
    let _m_trans = m1.transpose();
    let _m_mul = m1.mul_mat::<2, 4, 4>(&m2);
}

#[test]
fn test_quaternion_exhaustive() {
    let mut q = [1.0f32, 2.0, 3.0, 4.0];
    assert!(quaternion_norm_f32(&q) > 0.0);
    assert_eq!(quaternion_normalize_f32(&mut q), Status::Success);

    let q1 = [1.0f32, 0.0, 0.0, 0.0];
    let q2 = [0.0f32, 1.0, 0.0, 0.0];
    let mut out = [0.0f32; 4];
    quaternion_product_f32(&q1, &q2, &mut out);
    quaternion_conjugate_f32(&q1, &mut out);
    assert_eq!(quaternion_inverse_f32(&q1, &mut out), Status::Success);

    let mut rot_mat = [0.0f32; 9];
    quaternion_to_rotmat_f32(&q, &mut rot_mat);

    // Error branches
    let mut q_zero = [0.0f32; 4];
    assert_eq!(quaternion_normalize_f32(&mut q_zero), Status::ArgumentError);
    assert_eq!(
        quaternion_inverse_f32(&q_zero, &mut out),
        Status::ArgumentError
    );
}

#[test]
fn test_pipeline_nodes_exhaustive() {
    let mut gain_i16 = Gain::<i16>::new(16384);
    assert_eq!(gain_i16.process_sample(1000i16), 500i16);

    let mut gain_i32 = Gain::<i32>::new(1073741824);
    let _g_i32 = gain_i32.process_sample(1000i32);

    let mut limiter = Limiter::new(-1.0f32, 1.0f32);
    let mut block_in = [0.5f32, 1.5, -2.0];
    let mut block_out = [0.0f32; 3];
    limiter.process_block(&block_in, &mut block_out);
    limiter.process_in_place(&mut block_in);

    let mut pid_f32 = PidInstanceF32::new(1.0, 0.1, 0.01);
    assert!(pid_f32.process_sample(1.0).is_finite());

    let mut pid_q15 = PidInstanceQ15::new(
        q15::from_bits(1000),
        q15::from_bits(100),
        q15::from_bits(10),
    );
    let _p_q15 = pid_q15.process_sample(q15::from_bits(500));

    let mut filter_f32 = SinglePoleFilter::lowpass(0.1);
    assert!(filter_f32.process_sample(1.0).is_finite());

    let mut filter_q15 = SinglePoleFilterQ15::lowpass(q15::from_bits(3000));
    let _f_q15 = filter_q15.process_sample(q15::from_bits(1000));

    let mut dc_blocker = DcBlockerQ15::new(q15::from_bits(32000));
    let _dc_out = dc_blocker.process_sample(q15::from_bits(1000));
}
