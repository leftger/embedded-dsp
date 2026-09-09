use embedded_dsp::*;

#[test]
fn test_dsp_sample_all_primitives() {
    let a_f32: f32 = 0.5;
    let b_f32: f32 = 0.5;
    assert_eq!(a_f32.sat_add(b_f32), 1.0);
    assert_eq!(a_f32.sat_sub(b_f32), 0.0);
    assert_eq!(a_f32.sat_mul(b_f32), 0.25);

    let a_f64: f64 = 0.5;
    let b_f64: f64 = 0.5;
    assert_eq!(a_f64.sat_add(b_f64), 1.0);
    assert_eq!(a_f64.sat_sub(b_f64), 0.0);
    assert_eq!(a_f64.sat_mul(b_f64), 0.25);

    let c1 = Complex::new(1.0f32, 2.0f32);
    let c2 = Complex::new(3.0f32, 4.0f32);
    assert_eq!(c1.real + c2.real, 4.0);
    assert_eq!(c1.imag + c2.imag, 6.0);
}

#[test]
fn test_const_generics_and_cordic() {
    let mut fir = FirFilter::<4>::new([0.25f32, 0.25, 0.25, 0.25]);
    let in_buf = [1.0f32, 0.5, 0.2, 0.1];
    let mut out_buf = [0.0f32; 4];
    fir.process(&in_buf, &mut out_buf);
    fir.reset();

    let mut fir_q15 = FirFilterQ15::<4>::new([q15::from_bits(1000); 4]);
    let in_q15 = [q15::from_bits(2000); 4];
    let mut out_q15 = [q15::ZERO; 4];
    fir_q15.process(&in_q15, &mut out_q15);
    fir_q15.reset();

    let mut biquad = BiquadCascade::<5, 4>::new([1.0, 0.0, 0.0, 1.0, 0.0]);
    biquad.process(&in_buf, &mut out_buf);
    biquad.reset();

    let mut biquad_q15 = BiquadCascadeQ15::<5, 4>::new([q15::from_bits(1000); 5], 0);
    biquad_q15.process(&in_q15, &mut out_q15);
    biquad_q15.reset();

    let mat = Matrix::<2, 2, 4>::new([1.0, 2.0, 3.0, 4.0]);
    let t = mat.transpose();
    assert_eq!(t.data[1], 3.0);

    // CORDIC engine
    let (s, c) = cordic_sin_cos_q31(q31::from_bits(1000000));
    assert!(s.to_bits() != 0);
    assert!(c.to_bits() != 0);

    let atan_q15 = cordic_atan2_q15(q15::from_bits(1000), q15::from_bits(1000));
    assert!(atan_q15.to_bits() > 0);

    let sqrt_q15 = cordic_sqrt_q15(q15::from_bits(10000));
    assert!(sqrt_q15.to_bits() > 0);
}

#[test]
fn test_complex_math_extended() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [2.0f32, 1.0, 1.0, 2.0];
    let mut out = [0.0f32; 4];

    cmplx_add_f32(&a, &b, &mut out);
    cmplx_sub_f32(&a, &b, &mut out);
    cmplx_mult_cmplx_f32(&a, &b, &mut out);
    cmplx_mult_real_f32(&a, &b, &mut out);
    cmplx_conj_f32(&a, &mut out);

    let mut mag = [0.0f32; 2];
    cmplx_mag_f32(&a, &mut mag);
    cmplx_mag_squared_f32(&a, &mut mag);

    let dot = cmplx_dot_prod_f32(&a, &b);
    assert!(dot.real.is_finite());
}

#[test]
fn test_status_codes() {
    let s1 = Status::Success;
    let s2 = Status::ArgumentError;
    let s3 = Status::LengthError;
    assert_ne!(s1, s2);
    assert_ne!(s2, s3);
}

#[test]
fn test_transforms_extended() {
    let mut data = [1.0f32, 2.0, 3.0, 4.0];
    let mut out = [0.0f32; 4];
    assert_eq!(haar_transform_f32(&mut data), Status::Success);
    hartley_transform_f32(&mut data);
    dct4_f32(&data, &mut out, 4);
}
