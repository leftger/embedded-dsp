use embedded_dsp::*;

#[test]
fn test_pipeline_thorough_coverage() {
    let gain_f32 = Gain::new(2.0f32);
    let limiter_f32 = Limiter::new(-1.0f32, 1.0f32);
    let mut chain = gain_f32.then(limiter_f32);

    assert_eq!(chain.process_sample(0.25), 0.5);
    assert_eq!(chain.process_sample(1.0), 1.0); // Limiter clamped
    assert_eq!(chain.process_sample(-1.0), -1.0); // Limiter clamped

    let input_block = [0.1f32, 0.6, -0.8];
    let mut output_block = [0.0f32; 3];
    chain.process_block(&input_block, &mut output_block);
    assert_eq!(output_block[0], 0.2);
    assert_eq!(output_block[1], 1.0);

    let mut in_place = [0.1f32, 0.6, -0.8];
    chain.process_in_place(&mut in_place);
    assert_eq!(in_place[0], 0.2);
    assert_eq!(in_place[1], 1.0);

    // Gain i16 and i32
    let mut gain_i16 = Gain::new(16384i16); // 0.5 in Q15
    assert_eq!(gain_i16.process_sample(10000i16), 5000i16);

    let mut gain_i32 = Gain::new(1073741824i32); // 0.5 in Q31
    assert_eq!(gain_i32.process_sample(100000i32), 50000i32);

    // Limiter i16, i32, f32
    let mut lim_i16 = Limiter::new(-100i16, 100i16);
    assert_eq!(lim_i16.process_sample(150i16), 100i16);
    assert_eq!(lim_i16.process_sample(-150i16), -100i16);
    assert_eq!(lim_i16.process_sample(50i16), 50i16);

    let mut lim_i32 = Limiter::new(-100i32, 100i32);
    assert_eq!(lim_i32.process_sample(150i32), 100i32);
}

#[test]
fn test_quaternion_coverage() {
    let q = [1.0f32, 0.0, 0.0, 0.0];
    assert_eq!(quaternion_norm_f32(&q), 1.0);

    let mut zero_q = [0.0f32; 4];
    assert_eq!(quaternion_normalize_f32(&mut zero_q), Status::ArgumentError);
    assert_eq!(
        quaternion_inverse_f32(&zero_q, &mut [0.0; 4]),
        Status::ArgumentError
    );

    let q1 = [0.7071f32, 0.7071, 0.0, 0.0];
    let q2 = [0.7071f32, 0.0, 0.7071, 0.0];
    let mut q_prod = [0.0f32; 4];
    quaternion_product_f32(&q1, &q2, &mut q_prod);
    assert!(q_prod[0].is_finite());

    let mut q_conj = [0.0f32; 4];
    quaternion_conjugate_f32(&q1, &mut q_conj);
    assert_eq!(q_conj[0], q1[0]);
    assert_eq!(q_conj[1], -q1[1]);

    let mut q_inv = [0.0f32; 4];
    assert_eq!(quaternion_inverse_f32(&q1, &mut q_inv), Status::Success);

    let mut rot_mat = [0.0f32; 9];
    quaternion_to_rotmat_f32(&q1, &mut rot_mat);
    assert_eq!(rot_mat.len(), 9);
}

#[test]
fn test_fast_math_and_approximations() {
    assert!((fast_tanh_f32(0.0)).abs() < 1e-4);
    assert!((fast_tanh_f32(5.0) - 1.0).abs() < 1e-4);
    assert!((fast_tanh_f32(-5.0) + 1.0).abs() < 1e-4);

    assert!((fast_exp_f32(0.0) - 1.0).abs() < 1e-3);
    assert_eq!(fast_exp_f32(-15.0), 0.0);
    assert_eq!(fast_exp_f32(15.0), 22026.465);

    let mut res_atan = 0.0f32;
    assert_eq!(atan2_f32(1.0, 1.0, &mut res_atan), Status::Success);
    assert!((res_atan - core::f32::consts::FRAC_PI_4).abs() < 1e-3);

    let mut q31_out = q31::ZERO;
    assert_eq!(
        atan2_q31(q31::from_bits(10000), q31::from_bits(10000), &mut q31_out),
        Status::Success
    );

    let mut q15_out = q15::ZERO;
    assert_eq!(
        atan2_q15(q15::from_bits(1000), q15::from_bits(1000), &mut q15_out),
        Status::Success
    );

    let mut div_out_q15 = q15::ZERO;
    let mut shift_q15 = 0i16;
    assert_eq!(
        divide_q15(
            q15::from_bits(100),
            q15::ZERO,
            &mut div_out_q15,
            &mut shift_q15
        ),
        Status::ArgumentError
    );
    assert_eq!(
        divide_q15(
            q15::from_bits(100),
            q15::from_bits(200),
            &mut div_out_q15,
            &mut shift_q15
        ),
        Status::Success
    );
}

#[test]
fn test_safety_limiter_and_dynamics() {
    let mut limiter = SafetyLimiter::new(0.9, 0.05, 48000.0);
    assert_eq!(limiter.current_gain(), 1.0);

    let limited = limiter.process(1.5);
    assert!(limited <= 0.9 && limited >= -0.9);
    assert!(limiter.current_gain() < 1.0);

    limiter.reset();
    assert_eq!(limiter.current_gain(), 1.0);

    let mut comp = DynamicsCompressor::new(-20.0, 4.0, 6.0, 0.005, 0.1, 4.0, 48000.0);
    let _out = comp.process(0.8);
    comp.reset();

    let mut gate = NoiseGate::new(-40.0, -30.0, 0.002, 0.05, 48000.0);
    let _g_out = gate.process(0.0001);
    gate.reset();
}

#[test]
fn test_math_trait_floats() {
    use FloatMath;
    let x: f32 = 0.5;
    assert!(x.abs() > 0.0);
    assert!(x.sin() > 0.0);
    assert!(x.cos() > 0.0);
    assert!(x.tan() > 0.0);
    assert!(x.sqrt() > 0.0);
    assert!(x.ln() < 0.0);
    assert!(x.log10() < 0.0);
    assert!(x.exp() > 1.0);
    assert!(x.atan2(1.0) > 0.0);
    assert!(x.powf(2.0) == 0.25);
    assert!(x.tanh() > 0.0);

    let y: f64 = 0.5;
    assert!(y.abs() > 0.0);
    assert!(y.sin() > 0.0);
    assert!(y.cos() > 0.0);
    assert!(y.tan() > 0.0);
    assert!(y.sqrt() > 0.0);
    assert!(y.ln() < 0.0);
    assert!(y.log10() < 0.0);
    assert!(y.exp() > 1.0);
    assert!(y.atan2(1.0) > 0.0);
    assert!(y.powf(2.0) == 0.25);
    assert!(y.tanh() > 0.0);
}
