//! Coverage for fixed-point type methods, fixed-point basic math, and transforms.

use embedded_dsp::basic_math::{
    abs_q7, abs_q15, abs_q31, add_q7, add_q15, add_q31, clip_q7, clip_q15, clip_q31, dot_prod_q7,
    dot_prod_q15, dot_prod_q31, mult_q7, mult_q15, mult_q31, negate_q7, negate_q15, negate_q31,
    offset_q7, offset_q15, offset_q31, scale_q7, scale_q15, scale_q31, sub_q7, sub_q15, sub_q31,
};
use embedded_dsp::transform::{
    DAUBECHIES_4, cfft_f32, cfft_q15, cfft_q31, dct4_f32, fwht_f32, fwht_i32, haar_transform_f32,
    haar_transform_i32, hartley_transform_f32, ifwht_f32, inverse_haar_transform_f32,
    inverse_wavelet_transform_f32, rfft_f32, wavelet_transform_f32,
};
use embedded_dsp::types::{q7, q15, q31, q63};

#[test]
fn fixed_point_type_methods_are_covered() {
    let mut a = q31::from_num(0.25f32);
    let b = q31::from_num(0.125f32);
    assert!((a.to_num::<f32>() - 0.25).abs() < 1e-6);
    assert!(a.to_bits() > 0);
    a = q31::saturating_from_num(0.5);
    assert!((a.saturating_add(b).to_num::<f32>() - 0.625).abs() < 1e-6);
    assert!((a.saturating_sub(b).to_num::<f32>() - 0.375).abs() < 1e-6);
    assert!(a.saturating_mul(b).to_num::<f32>() > 0.0);
    assert!((a.saturating_neg().to_num::<f32>() + 0.5).abs() < 1e-6);
    assert!((a.saturating_abs().to_num::<f32>() - 0.5).abs() < 1e-6);
    assert!((a.wrapping_add(b).to_num::<f32>() - 0.625).abs() < 1e-6);
    assert!((a.wrapping_sub(b).to_num::<f32>() - 0.375).abs() < 1e-6);
    assert!((a.wrapping_neg().to_num::<f32>() + 0.5).abs() < 1e-6);
    let _ = a.wrapping_mul(b);
    let _ = a.wrapping_mul_int(2);
    let _ = a.wrapping_div(b);
    let _ = a.wrapping_div_int(2);
    let _ = a.checked_div(b);

    let q = q7::from_num(0.25f32);
    let qa = q15::from_num(0.25f32);
    let qb = q15::from_num(0.125f32);
    assert!((q.to_num::<f32>() - 0.25).abs() < 1e-3);
    assert!(q.saturating_add(q).to_num::<f32>() > 0.4);
    assert!(qa.wrapping_add(qb).to_num::<f32>() > 0.3);
    let _: q63 = 0;
}

#[test]
fn fixed_point_basic_math_families() {
    let q31_a = [
        q31::from_bits(1000),
        q31::from_bits(-2000),
        q31::from_bits(3000),
        q31::from_bits(-4000),
    ];
    let q31_b = [
        q31::from_bits(1),
        q31::from_bits(2),
        q31::from_bits(3),
        q31::from_bits(4),
    ];
    let mut out = [q31::ZERO; 4];
    abs_q31(&q31_a, &mut out);
    add_q31(&q31_a, &q31_b, &mut out);
    sub_q31(&q31_a, &q31_b, &mut out);
    mult_q31(&q31_a, &q31_b, &mut out);
    negate_q31(&q31_a, &mut out);
    offset_q31(&q31_a, q31::from_bits(100), &mut out);
    scale_q31(&q31_a, q31::from_bits(2), 0, &mut out);
    clip_q31(&q31_a, q31::from_bits(-100), q31::from_bits(100), &mut out);
    let _ = dot_prod_q31(&q31_a, &q31_b);

    let q15_a = [
        q15::from_bits(100),
        q15::from_bits(-200),
        q15::from_bits(300),
        q15::from_bits(-400),
    ];
    let q15_b = [
        q15::from_bits(1),
        q15::from_bits(2),
        q15::from_bits(3),
        q15::from_bits(4),
    ];
    let mut out15 = [q15::ZERO; 4];
    abs_q15(&q15_a, &mut out15);
    add_q15(&q15_a, &q15_b, &mut out15);
    sub_q15(&q15_a, &q15_b, &mut out15);
    mult_q15(&q15_a, &q15_b, &mut out15);
    negate_q15(&q15_a, &mut out15);
    offset_q15(&q15_a, q15::from_bits(10), &mut out15);
    scale_q15(&q15_a, q15::from_bits(2), 0, &mut out15);
    clip_q15(&q15_a, q15::from_bits(-10), q15::from_bits(10), &mut out15);
    let _ = dot_prod_q15(&q15_a, &q15_b);

    let q7_a = [
        q7::from_bits(10),
        q7::from_bits(-20),
        q7::from_bits(30),
        q7::from_bits(-40),
    ];
    let q7_b = [
        q7::from_bits(1),
        q7::from_bits(2),
        q7::from_bits(3),
        q7::from_bits(4),
    ];
    let mut out7 = [q7::ZERO; 4];
    abs_q7(&q7_a, &mut out7);
    add_q7(&q7_a, &q7_b, &mut out7);
    sub_q7(&q7_a, &q7_b, &mut out7);
    mult_q7(&q7_a, &q7_b, &mut out7);
    negate_q7(&q7_a, &mut out7);
    offset_q7(&q7_a, q7::from_bits(2), &mut out7);
    scale_q7(&q7_a, q7::from_bits(2), 0, &mut out7);
    clip_q7(&q7_a, q7::from_bits(-5), q7::from_bits(5), &mut out7);
    let _ = dot_prod_q7(&q7_a, &q7_b);
}

#[test]
fn transform_families_execute() {
    let mut fft = [0.0f32; 16];
    fft[0] = 1.0;
    cfft_f32(&mut fft, 8, 0, 1);
    cfft_f32(&mut fft, 8, 1, 0);

    let mut q31fft = [q31::ZERO; 16];
    q31fft[0] = q31::from_bits(1 << 20);
    cfft_q31(&mut q31fft, 8, 0, 1);

    let mut q15fft = [q15::ZERO; 16];
    q15fft[0] = q15::from_bits(1 << 10);
    cfft_q15(&mut q15fft, 8, 0, 1);

    let real = [1.0f32, 0.5, -0.2, 0.7, 0.1, -0.4, 0.3, 0.9];
    rfft_f32(&real, &mut [0.0f32; 16], 8, 0);

    let mut dct = [0.0f32; 8];
    dct4_f32(&[1.0f32; 8], &mut dct, 8);

    let mut fwht = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(fwht_f32(&mut fwht), embedded_dsp::types::Status::Success);
    assert_eq!(ifwht_f32(&mut fwht), embedded_dsp::types::Status::Success);

    let mut fwhti = [1i32, 2, 3, 4];
    assert_eq!(fwht_i32(&mut fwhti), embedded_dsp::types::Status::Success);

    let mut haar = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(
        haar_transform_f32(&mut haar),
        embedded_dsp::types::Status::Success
    );
    assert_eq!(
        inverse_haar_transform_f32(&mut haar),
        embedded_dsp::types::Status::Success
    );

    let mut haari = [1i32, 2, 3, 4];
    assert_eq!(
        haar_transform_i32(&mut haari),
        embedded_dsp::types::Status::Success
    );

    let mut hart = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(
        hartley_transform_f32(&mut hart),
        embedded_dsp::types::Status::Success
    );

    let mut wave = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(
        wavelet_transform_f32(&mut wave, &DAUBECHIES_4),
        embedded_dsp::types::Status::Success
    );
    assert_eq!(
        inverse_wavelet_transform_f32(&mut wave, &DAUBECHIES_4),
        embedded_dsp::types::Status::Success
    );
}
