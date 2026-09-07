//! Coverage for matrix, support, window, and fast-math DSP helpers.

use embedded_dsp::fast_math::{
    atan2_f32, atan2_q15, atan2_q31, cos_f32, cos_q31, divide_q15, divide_q31, exp_f32, log_f32,
    sin_cos_f32, sin_cos_q31, sin_f32, sin_q31, sqrt_f32, sqrt_q15, sqrt_q31, vsqrt_f32,
};
use embedded_dsp::matrix::{
    MatrixInstance, MatrixInstanceMut, mat_add_f32, mat_add_q15, mat_add_q31, mat_inverse_f32,
    mat_mult_f32, mat_mult_q15, mat_mult_q31, mat_scale_f32, mat_scale_q15, mat_scale_q31,
    mat_sub_f32, mat_sub_q15, mat_sub_q31, mat_trans_f32, mat_trans_q15, mat_trans_q31,
    polynomial_eval_f32,
};
use embedded_dsp::support::{
    XorShift64, barycenter_f32, biquad_coeffs_f32_to_q15, biquad_coeffs_f32_to_q31, copy_f32,
    copy_q7, copy_q15, copy_q31, f32_to_q7, f32_to_q15, f32_to_q31, fill_f32, fill_q7, fill_q15,
    fill_q31, fir_taps_f32_to_q15, gaussian_noise_f32, q7_to_f32, q7_to_q15, q7_to_q31, q15_to_f32,
    q15_to_q7, q15_to_q31, q31_to_f32, q31_to_q7, q31_to_q15, sort_f32, uniform_noise_f32,
    weighted_sum_f32,
};
use embedded_dsp::types::{Status, q7, q15, q31};
use embedded_dsp::window::{
    apply_window_f32, apply_window_q15, bartlett_f32, bartlett_q15, bessel_i0, blackman_f32,
    blackman_harris_f32, blackman_q15, flattop_f32, hamming_f32, hamming_q15, hanning_f32,
    hanning_q15, kaiser_f32, welch_f32,
};

#[test]
fn matrix_float_and_fixed_operations() {
    let a = MatrixInstance::new(2, 2, &[1.0f32, 2.0, 3.0, 4.0]);
    let b = MatrixInstance::new(2, 2, &[5.0f32, 6.0, 7.0, 8.0]);

    let mut out = [0.0f32; 4];
    {
        let mut dst = MatrixInstanceMut::new(2, 2, &mut out);
        assert_eq!(mat_add_f32(&a, &b, &mut dst), Status::Success);
    }
    assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
    {
        let mut dst = MatrixInstanceMut::new(2, 2, &mut out);
        assert_eq!(mat_sub_f32(&a, &b, &mut dst), Status::Success);
    }
    assert_eq!(out, [-4.0, -4.0, -4.0, -4.0]);
    {
        let mut dst = MatrixInstanceMut::new(2, 2, &mut out);
        assert_eq!(mat_scale_f32(&a, 2.0, &mut dst), Status::Success);
    }
    assert_eq!(out, [2.0, 4.0, 6.0, 8.0]);

    let mut mult = [0.0f32; 4];
    {
        let mut mdst = MatrixInstanceMut::new(2, 2, &mut mult);
        assert_eq!(mat_mult_f32(&a, &b, &mut mdst), Status::Success);
    }
    assert_eq!(mult, [19.0, 22.0, 43.0, 50.0]);

    let mut trans = [0.0f32; 4];
    {
        let mut tdst = MatrixInstanceMut::new(2, 2, &mut trans);
        assert_eq!(mat_trans_f32(&a, &mut tdst), Status::Success);
    }
    assert_eq!(trans, [1.0, 3.0, 2.0, 4.0]);

    let inv_in = MatrixInstance::new(2, 2, &[4.0f32, 7.0, 2.0, 6.0]);
    let mut inv_out = [0.0f32; 4];
    {
        let mut idst = MatrixInstanceMut::new(2, 2, &mut inv_out);
        assert_eq!(mat_inverse_f32(&inv_in, &mut idst), Status::Success);
    }

    let qa = [
        q31::from_bits(1000),
        q31::from_bits(2000),
        q31::from_bits(3000),
        q31::from_bits(4000),
    ];
    let qb = [
        q31::from_bits(1),
        q31::from_bits(2),
        q31::from_bits(3),
        q31::from_bits(4),
    ];
    let ma = MatrixInstance::new(2, 2, &qa);
    let mb = MatrixInstance::new(2, 2, &qb);
    let mut qout = [q31::ZERO; 4];
    {
        let mut qdst = MatrixInstanceMut::new(2, 2, &mut qout);
        assert_eq!(mat_add_q31(&ma, &mb, &mut qdst), Status::Success);
        assert_eq!(mat_sub_q31(&ma, &mb, &mut qdst), Status::Success);
        assert_eq!(
            mat_scale_q31(&ma, q31::from_bits(2), 0, &mut qdst),
            Status::Success
        );
        assert_eq!(mat_trans_q31(&ma, &mut qdst), Status::Success);
        assert_eq!(mat_mult_q31(&ma, &mb, &mut qdst), Status::Success);
    }

    let q15a = [
        q15::from_bits(100),
        q15::from_bits(200),
        q15::from_bits(300),
        q15::from_bits(400),
    ];
    let q15b = [
        q15::from_bits(1),
        q15::from_bits(2),
        q15::from_bits(3),
        q15::from_bits(4),
    ];
    let m15a = MatrixInstance::new(2, 2, &q15a);
    let m15b = MatrixInstance::new(2, 2, &q15b);
    let mut out15 = [q15::ZERO; 4];
    {
        let mut dst15 = MatrixInstanceMut::new(2, 2, &mut out15);
        assert_eq!(mat_add_q15(&m15a, &m15b, &mut dst15), Status::Success);
        assert_eq!(mat_sub_q15(&m15a, &m15b, &mut dst15), Status::Success);
        assert_eq!(
            mat_scale_q15(&m15a, q15::from_bits(2), 0, &mut dst15),
            Status::Success
        );
        assert_eq!(mat_trans_q15(&m15a, &mut dst15), Status::Success);
        assert_eq!(mat_mult_q15(&m15a, &m15b, &mut dst15), Status::Success);
    }

    assert_eq!(polynomial_eval_f32(&[1.0, 2.0, 3.0], 2.0), 17.0);
}

#[test]
fn support_copy_fill_conversion_sort_and_noise() {
    let src = [1.0f32, 2.0, 3.0];
    let mut f32buf = [0.0f32; 3];
    copy_f32(&src, &mut f32buf);
    assert_eq!(f32buf, src);

    let q7_src = [q7::from_bits(10), q7::from_bits(20), q7::from_bits(30)];
    let q15_src = [
        q15::from_bits(100),
        q15::from_bits(200),
        q15::from_bits(300),
    ];
    let q31_src = [
        q31::from_bits(1000),
        q31::from_bits(2000),
        q31::from_bits(3000),
    ];

    let mut q7buf = [q7::ZERO; 3];
    fill_q7(q7::from_bits(7), &mut q7buf);
    copy_q7(&q7_src, &mut q7buf);
    let mut q15buf = [q15::ZERO; 3];
    fill_q15(q15::from_bits(7), &mut q15buf);
    copy_q15(&q15_src, &mut q15buf);
    let mut q31buf = [q31::ZERO; 3];
    fill_q31(q31::from_bits(7), &mut q31buf);
    copy_q31(&q31_src, &mut q31buf);

    let mut f32buf2 = [0.0f32; 3];
    fill_f32(1.5, &mut f32buf2);

    let mut conv15 = [q15::ZERO; 3];
    q7_to_q15(&q7_src, &mut conv15);
    let mut conv31 = [q31::ZERO; 3];
    q7_to_q31(&q7_src, &mut conv31);
    let mut convf = [0.0f32; 3];
    q7_to_f32(&q7_src, &mut convf);

    let mut conv7 = [q7::ZERO; 3];
    q15_to_q7(&q15_src, &mut conv7);
    let mut conv31b = [q31::ZERO; 3];
    q15_to_q31(&q15_src, &mut conv31b);
    let mut convfb = [0.0f32; 3];
    q15_to_f32(&q15_src, &mut convfb);

    let mut conv7c = [q7::ZERO; 3];
    q31_to_q7(&q31_src, &mut conv7c);
    let mut conv15c = [q15::ZERO; 3];
    q31_to_q15(&q31_src, &mut conv15c);
    let mut convfc = [0.0f32; 3];
    q31_to_f32(&q31_src, &mut convfc);

    let floats = [0.1f32, -0.5, 0.9];
    let mut fq7 = [q7::ZERO; 3];
    f32_to_q7(&floats, &mut fq7);
    let mut fq15 = [q15::ZERO; 3];
    f32_to_q15(&floats, &mut fq15);
    let mut fq31 = [q31::ZERO; 3];
    f32_to_q31(&floats, &mut fq31);

    let mut taps = [q15::ZERO; 4];
    assert_eq!(
        fir_taps_f32_to_q15(&[0.1f32, 0.2, 0.3, 0.4], &mut taps),
        Status::Success
    );

    let mut sorted = [0.0f32; 4];
    sort_f32(&[3.0, 1.0, 4.0, 1.5], &mut sorted, true);
    assert_eq!(sorted, [1.0, 1.5, 3.0, 4.0]);
    sort_f32(&[3.0, 1.0, 4.0, 1.5], &mut sorted, false);
    assert_eq!(sorted, [4.0, 3.0, 1.5, 1.0]);

    let mut center = [0.0f32; 2];
    assert_eq!(
        barycenter_f32(&[0.0, 0.0, 2.0, 2.0], &[1.0, 1.0], &mut center, 2, 2),
        Status::Success
    );
    assert_eq!(center, [1.0, 1.0]);
    assert_eq!(weighted_sum_f32(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]), 2.0);

    let mut seed = 1u64;
    let mut noise = [0.0f32; 8];
    uniform_noise_f32(&mut noise, -1.0, 1.0, &mut seed);
    gaussian_noise_f32(&mut noise, 0.0, 1.0, &mut seed);
    let mut rng = XorShift64::new(0);
    let _ = rng.next_u64();
    let _ = rng.next_f32();

    let mut bq15 = [q15::ZERO; 5];
    let mut bq31 = [q31::ZERO; 5];
    assert_eq!(
        biquad_coeffs_f32_to_q15(&[0.5, 0.2, -0.1, 1.0, -0.3], &mut bq15, 0),
        Status::Success
    );
    assert_eq!(
        biquad_coeffs_f32_to_q31(&[0.5, 0.2, -0.1, 1.0, -0.3], &mut bq31, 0),
        Status::Success
    );
}

#[test]
fn window_functions_all_types() {
    let mut w = [0.0f32; 8];
    hanning_f32(&mut w);
    hamming_f32(&mut w);
    blackman_f32(&mut w);
    blackman_harris_f32(&mut w);
    bartlett_f32(&mut w);
    welch_f32(&mut w);
    flattop_f32(&mut w);
    let beta = bessel_i0(1.5);
    assert!(beta > 0.0);
    kaiser_f32(&mut w, 1.5);
    apply_window_f32(&mut w, &[1.0; 8]);

    let mut q = [q15::ZERO; 8];
    hanning_q15(&mut q);
    hamming_q15(&mut q);
    blackman_q15(&mut q);
    bartlett_q15(&mut q);
    apply_window_q15(&mut q, &[q15::from_bits(1000); 8]);
}

#[test]
fn fast_math_float_and_fixed() {
    let mut s = 0.0f32;
    let mut c = 0.0f32;
    sin_cos_f32(1.0, &mut s, &mut c);
    assert!(sin_f32(0.0).abs() < 1e-6);
    assert!((cos_f32(0.0) - 1.0).abs() < 1e-6);

    let mut qs = q31::ZERO;
    let mut qc = q31::ZERO;
    sin_cos_q31(q31::from_bits(0), &mut qs, &mut qc);
    let _ = sin_q31(q31::from_bits(1));
    let _ = cos_q31(q31::from_bits(1));

    let mut r = 0.0f32;
    assert_eq!(sqrt_f32(4.0, &mut r), Status::Success);
    assert_eq!(r, 2.0);
    let mut qr31 = q31::ZERO;
    assert_eq!(sqrt_q31(q31::from_bits(4 << 8), &mut qr31), Status::Success);
    let mut qr15 = q15::ZERO;
    assert_eq!(sqrt_q15(q15::from_bits(4 << 8), &mut qr15), Status::Success);
    let mut roots = [0.0f32; 3];
    vsqrt_f32(&[1.0, 4.0, 9.0], &mut roots);

    let mut quot = q31::ZERO;
    let mut shift = 0i16;
    assert_eq!(
        divide_q31(
            q31::from_bits(1 << 20),
            q31::from_bits(1 << 10),
            &mut quot,
            &mut shift
        ),
        Status::Success
    );
    let mut quot15 = q15::ZERO;
    assert_eq!(
        divide_q15(
            q15::from_bits(1 << 10),
            q15::from_bits(1 << 5),
            &mut quot15,
            &mut shift
        ),
        Status::Success
    );

    let _ = log_f32(2.0);
    let _ = exp_f32(1.0);
    let mut at = 0.0f32;
    assert_eq!(atan2_f32(1.0, 1.0, &mut at), Status::Success);
    let mut at31 = q31::ZERO;
    assert_eq!(
        atan2_q31(q31::from_bits(1000), q31::from_bits(1000), &mut at31),
        Status::Success
    );
    let mut at15 = q15::ZERO;
    assert_eq!(
        atan2_q15(q15::from_bits(100), q15::from_bits(100), &mut at15),
        Status::Success
    );
}
