//! Coverage-gap tests for DSP statistics, interpolation, distance, and complex math.

use embedded_dsp::complex_math::{
    cmplx_add_f32, cmplx_add_q15, cmplx_add_q31, cmplx_conj_f32, cmplx_conj_q15, cmplx_conj_q31,
    cmplx_dot_prod_f32, cmplx_mag_f32, cmplx_mag_q15, cmplx_mag_q31, cmplx_mag_squared_f32,
    cmplx_mult_cmplx_f32, cmplx_mult_cmplx_q15, cmplx_mult_cmplx_q31, cmplx_mult_real_f32,
    cmplx_sub_f32, cmplx_sub_q15, cmplx_sub_q31,
};
use embedded_dsp::distance::{
    bray_curtis_distance_f32, bray_curtis_distance_q15, canberra_distance_f32,
    canberra_distance_q15, chebyshev_distance_f32, chebyshev_distance_q15, chebyshev_distance_q31,
    cosine_distance_f32, cosine_distance_q15, cosine_distance_q31, euclidean_distance_f32,
    euclidean_distance_q15, euclidean_distance_q31, hamming_distance_f32, hamming_distance_q15,
    hamming_distance_q31, jaccard_distance_f32, manhattan_distance_f32, manhattan_distance_q15,
    manhattan_distance_q31, minkowski_distance_f32,
};
use embedded_dsp::interpolation::{
    SplineInstanceF32, bilinear_interp_f32, linear_interp_f32, linear_interp_q15, linear_interp_q31,
};
use embedded_dsp::statistics::{
    absmax_f32, absmin_f32, entropy_f32, kullback_leibler_f32, logsumexp_f32, max_f32, max_q7,
    max_q15, max_q31, mean_f32, mean_f64, mean_q7, mean_q15, mean_q31, min_f32, min_q7, min_q15,
    min_q31, power_f32, power_q7, power_q15, power_q31, rms_f32, rms_q15, rms_q31, std_f32,
    std_f64, std_q7, std_q15, std_q31, var_f32, var_f64, var_q7, var_q15, var_q31,
};
use embedded_dsp::types::{Complex, Status, q7, q15, q31, q63};

#[test]
fn statistics_float_families() {
    let src = [1.0f32, 2.0, 3.0, 4.0];
    let mut result = 0.0f32;
    assert_eq!(mean_f32(&src, &mut result), Status::Success);
    assert_eq!(var_f32(&src, &mut result), Status::Success);
    assert_eq!(std_f32(&src, &mut result), Status::Success);
    assert_eq!(rms_f32(&src, &mut result), Status::Success);
    assert_eq!(power_f32(&src, &mut result), Status::Success);

    let mut min_v = 0.0f32;
    let mut min_i = 0;
    let mut max_v = 0.0f32;
    let mut max_i = 0;
    assert_eq!(min_f32(&src, &mut min_v, &mut min_i), Status::Success);
    assert_eq!(max_f32(&src, &mut max_v, &mut max_i), Status::Success);
    assert_eq!(absmin_f32(&src, &mut min_v, &mut min_i), Status::Success);
    assert_eq!(absmax_f32(&src, &mut max_v, &mut max_i), Status::Success);

    let src64 = [1.0f64, 2.0, 3.0, 4.0];
    let mut r64 = 0.0f64;
    assert_eq!(mean_f64(&src64, &mut r64), Status::Success);
    assert_eq!(var_f64(&src64, &mut r64), Status::Success);
    assert_eq!(std_f64(&src64, &mut r64), Status::Success);

    let prob = [0.1f32, 0.2, 0.3, 0.4];
    let _ = entropy_f32(&prob);
    let _ = kullback_leibler_f32(&prob, &[0.25; 4]);
    let _ = logsumexp_f32(&[0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn statistics_fixed_point_families() {
    let q31_src = [
        q31::from_bits(1000),
        q31::from_bits(2000),
        q31::from_bits(3000),
    ];
    let mut q31_r = q31::ZERO;
    assert_eq!(mean_q31(&q31_src, &mut q31_r), Status::Success);
    assert_eq!(var_q31(&q31_src, &mut q31_r), Status::Success);
    assert_eq!(std_q31(&q31_src, &mut q31_r), Status::Success);
    assert_eq!(rms_q31(&q31_src, &mut q31_r), Status::Success);
    let mut q63_r = 0i64;
    assert_eq!(power_q31(&q31_src, &mut q63_r), Status::Success);
    let mut idx = 0;
    let mut val = q31::ZERO;
    assert_eq!(min_q31(&q31_src, &mut val, &mut idx), Status::Success);
    assert_eq!(max_q31(&q31_src, &mut val, &mut idx), Status::Success);

    let q15_src = [
        q15::from_bits(100),
        q15::from_bits(200),
        q15::from_bits(300),
    ];
    let mut q15_r = q15::ZERO;
    assert_eq!(mean_q15(&q15_src, &mut q15_r), Status::Success);
    assert_eq!(var_q15(&q15_src, &mut q15_r), Status::Success);
    assert_eq!(std_q15(&q15_src, &mut q15_r), Status::Success);
    assert_eq!(rms_q15(&q15_src, &mut q15_r), Status::Success);
    assert_eq!(power_q15(&q15_src, &mut q63_r), Status::Success);
    let mut q15_v = q15::ZERO;
    assert_eq!(min_q15(&q15_src, &mut q15_v, &mut idx), Status::Success);
    assert_eq!(max_q15(&q15_src, &mut q15_v, &mut idx), Status::Success);

    let q7_src = [q7::from_bits(10), q7::from_bits(20), q7::from_bits(30)];
    let mut q7_r = q7::ZERO;
    assert_eq!(mean_q7(&q7_src, &mut q7_r), Status::Success);
    assert_eq!(var_q7(&q7_src, &mut q7_r), Status::Success);
    assert_eq!(std_q7(&q7_src, &mut q7_r), Status::Success);
    let mut q31_from_power = q31::ZERO;
    assert_eq!(power_q7(&q7_src, &mut q31_from_power), Status::Success);
    let mut q7_v = q7::ZERO;
    assert_eq!(min_q7(&q7_src, &mut q7_v, &mut idx), Status::Success);
    assert_eq!(max_q7(&q7_src, &mut q7_v, &mut idx), Status::Success);
}

#[test]
fn interpolation_coverage() {
    let table = [0.0f32, 2.0, 4.0, 8.0];
    assert_eq!(linear_interp_f32(&table, 1.0, 1.0), 2.0);
    assert_eq!(linear_interp_f32(&table, 10.0, 1.0), 8.0);
    assert_eq!(linear_interp_f32(&[], 1.0, 1.0), 0.0);

    let q15_table = [q15::from_bits(0), q15::from_bits(100), q15::from_bits(200)];
    let q31_table = [q31::from_bits(0), q31::from_bits(100), q31::from_bits(200)];
    let _ = linear_interp_q15(&q15_table, q15::from_bits(10));
    let _ = linear_interp_q31(&q31_table, q31::from_bits(10));
    let _ = linear_interp_q15(&[], q15::ZERO);
    let _ = linear_interp_q31(&[], q31::ZERO);

    let grid = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let _ = bilinear_interp_f32(&grid, 2, 3, 0.5, 0.5);
    let _ = bilinear_interp_f32(&grid, 2, 3, 9.0, 9.0);

    let x = [0.0f32, 1.0, 2.0];
    let y = [1.0f32, 3.0, 8.0];
    let coeffs = [1.0f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0];
    let spline = SplineInstanceF32 {
        x: &x,
        y: &y,
        coeffs: &coeffs,
    };
    assert_eq!(spline.interpolate(-1.0), 1.0);
    assert_eq!(spline.interpolate(5.0), 8.0);
    let _ = spline.interpolate(1.5);
}

#[test]
fn distance_metrics_f32_and_fixed() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 6.0, 8.0];
    let _ = euclidean_distance_f32(&a, &b);
    let _ = cosine_distance_f32(&a, &b);
    let _ = chebyshev_distance_f32(&a, &b);
    let _ = manhattan_distance_f32(&a, &b);
    let _ = minkowski_distance_f32(&a, &b, 3.0);
    let _ = jaccard_distance_f32(&[1.0, 0.0, 1.0], &[1.0, 1.0, 0.0]);
    let _ = hamming_distance_f32(&a, &b);
    let _ = canberra_distance_f32(&a, &b);
    let _ = bray_curtis_distance_f32(&a, &b);

    let q15_a = [q15::from_bits(100), q15::from_bits(200)];
    let q15_b = [q15::from_bits(300), q15::from_bits(400)];
    let _ = euclidean_distance_q15(&q15_a, &q15_b);
    let _ = chebyshev_distance_q15(&q15_a, &q15_b);
    let _ = manhattan_distance_q15(&q15_a, &q15_b);
    let _ = cosine_distance_q15(&q15_a, &q15_b);
    let _ = hamming_distance_q15(&q15_a, &q15_b);
    let _ = canberra_distance_q15(&q15_a, &q15_b);
    let _ = bray_curtis_distance_q15(&q15_a, &q15_b);

    let q31_a = [q31::from_bits(100), q31::from_bits(200)];
    let q31_b = [q31::from_bits(300), q31::from_bits(400)];
    let _ = euclidean_distance_q31(&q31_a, &q31_b);
    let _ = chebyshev_distance_q31(&q31_a, &q31_b);
    let _ = manhattan_distance_q31(&q31_a, &q31_b);
    let _ = cosine_distance_q31(&q31_a, &q31_b);
    let _ = hamming_distance_q31(&q31_a, &q31_b);
}

#[test]
fn complex_math_f32_and_fixed_point() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut out = [0.0f32; 4];
    cmplx_add_f32(&a, &b, &mut out);
    cmplx_sub_f32(&b, &a, &mut out);
    cmplx_mult_cmplx_f32(&a, &b, &mut out);
    cmplx_mult_real_f32(&a, &[2.0, 3.0], &mut out);
    cmplx_mag_f32(&a, &mut [0.0; 2]);
    cmplx_mag_squared_f32(&a, &mut [0.0; 2]);
    cmplx_conj_f32(&a, &mut out);
    let dot = cmplx_dot_prod_f32(&a, &b);
    let _ = Complex::new(dot.real, dot.imag);

    let qa = [
        q31::from_bits(100),
        q31::from_bits(200),
        q31::from_bits(300),
        q31::from_bits(400),
    ];
    let qb = [
        q31::from_bits(10),
        q31::from_bits(20),
        q31::from_bits(30),
        q31::from_bits(40),
    ];
    let mut qout = [q31::ZERO; 4];
    cmplx_add_q31(&qa, &qb, &mut qout);
    cmplx_sub_q31(&qa, &qb, &mut qout);
    cmplx_mult_cmplx_q31(&qa, &qb, &mut qout);
    cmplx_mag_q31(&qa, &mut [q31::ZERO; 2]);
    cmplx_conj_q31(&qa, &mut qout);

    let q15a = [
        q15::from_bits(100),
        q15::from_bits(200),
        q15::from_bits(300),
        q15::from_bits(400),
    ];
    let q15b = [
        q15::from_bits(10),
        q15::from_bits(20),
        q15::from_bits(30),
        q15::from_bits(40),
    ];
    let mut q15out = [q15::ZERO; 4];
    cmplx_add_q15(&q15a, &q15b, &mut q15out);
    cmplx_sub_q15(&q15a, &q15b, &mut q15out);
    cmplx_mult_cmplx_q15(&q15a, &q15b, &mut q15out);
    cmplx_mag_q15(&q15a, &mut [q15::ZERO; 2]);
    cmplx_conj_q15(&q15a, &mut q15out);
}

#[allow(dead_code)]
fn touch_fixed_aliases() {
    let _: q63 = 0;
    let _ = Status::Success;
}
