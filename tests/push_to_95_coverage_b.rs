use embedded_dsp::pipeline::DspNode;
use embedded_dsp::*;

#[test]
fn test_math_exhaustive() {
    assert_eq!(isqrt_u32(0), 0);
    assert_eq!(isqrt_u32(1), 1);
    assert_eq!(isqrt_u32(2), 1);
    assert_eq!(isqrt_u32(3), 1);
    assert_eq!(isqrt_u32(4), 2);
    assert_eq!(isqrt_u32(15), 3);
    assert_eq!(isqrt_u32(16), 4);
    assert_eq!(isqrt_u32(100), 10);
    assert_eq!(isqrt_u32(100000), 316);

    assert_eq!(isqrt_u64(0), 0);
    assert_eq!(isqrt_u64(1), 1);
    assert_eq!(isqrt_u64(2), 1);
    assert_eq!(isqrt_u64(3), 1);
    assert_eq!(isqrt_u64(4), 2);
    assert_eq!(isqrt_u64(15), 3);
    assert_eq!(isqrt_u64(16), 4);
    assert_eq!(isqrt_u64(100), 10);
    assert_eq!(isqrt_u64(1_000_000_000), 31622);

    let x32 = 0.5f32;
    assert!(FloatMath::abs(x32).is_finite());
    assert!(FloatMath::sin(x32).is_finite());
    assert!(FloatMath::cos(x32).is_finite());
    assert!(FloatMath::tan(x32).is_finite());
    assert!(FloatMath::sqrt(x32).is_finite());
    assert!(FloatMath::ln(x32).is_finite());
    assert!(FloatMath::log10(x32).is_finite());
    assert!(FloatMath::exp(x32).is_finite());
    assert!(FloatMath::atan2(x32, 1.0f32).is_finite());
    assert!(FloatMath::powf(x32, 2.0f32).is_finite());
    assert!(FloatMath::tanh(x32).is_finite());

    let x64 = 0.5f64;
    assert!(FloatMath::abs(x64).is_finite());
    assert!(FloatMath::sin(x64).is_finite());
    assert!(FloatMath::cos(x64).is_finite());
    assert!(FloatMath::tan(x64).is_finite());
    assert!(FloatMath::sqrt(x64).is_finite());
    assert!(FloatMath::ln(x64).is_finite());
    assert!(FloatMath::log10(x64).is_finite());
    assert!(FloatMath::exp(x64).is_finite());
    assert!(FloatMath::atan2(x64, 1.0f64).is_finite());
    assert!(FloatMath::powf(x64, 2.0f64).is_finite());
    assert!(FloatMath::tanh(x64).is_finite());
}

#[test]
fn test_types_and_dspsample_exhaustive() {
    assert_eq!(
        q15_mult(q15::from_bits(1000), q15::from_bits(2000)).to_bits(),
        q15::from_bits(1000)
            .saturating_mul(q15::from_bits(2000))
            .to_bits()
    );
    assert_eq!(
        q31_mult(q31::from_bits(10000), q31::from_bits(20000)).to_bits(),
        q31::from_bits(10000)
            .saturating_mul(q31::from_bits(20000))
            .to_bits()
    );
    assert_eq!(
        q7_mult(q7::from_bits(10), q7::from_bits(20)).to_bits(),
        q7::from_bits(10)
            .saturating_mul(q7::from_bits(20))
            .to_bits()
    );

    // DspSample for f32
    assert_eq!(f32::ZERO, 0.0);
    assert_eq!(f32::ONE, 1.0);
    assert_eq!(DspSample::sat_add(1.0f32, 2.0f32), 3.0f32);
    assert_eq!(DspSample::sat_sub(3.0f32, 1.0f32), 2.0f32);
    assert_eq!(DspSample::sat_mul(2.0f32, 3.0f32), 6.0f32);
    assert_eq!(DspSample::sat_div(6.0f32, 2.0f32), 3.0f32);
    assert_eq!(DspSample::abs_val(-5.0f32), 5.0f32);
    assert_eq!(DspSample::abs_val(5.0f32), 5.0f32);
    assert_eq!(DspSample::to_f32(4.5f32), 4.5f32);
    assert_eq!(<f32 as DspSample>::from_f32(4.5f32), 4.5f32);

    // DspSample for f64
    assert_eq!(f64::ZERO, 0.0);
    assert_eq!(f64::ONE, 1.0);
    assert_eq!(DspSample::sat_add(1.0f64, 2.0f64), 3.0f64);
    assert_eq!(DspSample::sat_sub(3.0f64, 1.0f64), 2.0f64);
    assert_eq!(DspSample::sat_mul(2.0f64, 3.0f64), 6.0f64);
    assert_eq!(DspSample::sat_div(6.0f64, 2.0f64), 3.0f64);
    assert_eq!(DspSample::abs_val(-5.0f64), 5.0f64);
    assert_eq!(DspSample::abs_val(5.0f64), 5.0f64);
    assert_eq!(DspSample::to_f32(4.5f64), 4.5f32);
    assert_eq!(<f64 as DspSample>::from_f32(4.5f32), 4.5f64);

    // DspSample for q15
    let a_q15 = q15::from_bits(1000);
    let b_q15 = q15::from_bits(500);
    assert_eq!(
        DspSample::sat_add(a_q15, b_q15),
        a_q15.saturating_add(b_q15)
    );
    assert_eq!(
        DspSample::sat_sub(a_q15, b_q15),
        a_q15.saturating_sub(b_q15)
    );
    assert_eq!(
        DspSample::sat_mul(a_q15, b_q15),
        a_q15.saturating_mul(b_q15)
    );
    let _div_q15 = DspSample::sat_div(a_q15, b_q15);
    let _div_zero15 = DspSample::sat_div(a_q15, q15::ZERO);
    let _div_neg_zero15 = DspSample::sat_div(-a_q15, q15::ZERO);
    assert_eq!(DspSample::abs_val(-a_q15), a_q15);
    let _f_q15 = DspSample::to_f32(a_q15);
    let _q15_from_f = <q15 as DspSample>::from_f32(0.5);

    // DspSample for q31
    let a_q31 = q31::from_bits(100000);
    let b_q31 = q31::from_bits(50000);
    assert_eq!(
        DspSample::sat_add(a_q31, b_q31),
        a_q31.saturating_add(b_q31)
    );
    assert_eq!(
        DspSample::sat_sub(a_q31, b_q31),
        a_q31.saturating_sub(b_q31)
    );
    assert_eq!(
        DspSample::sat_mul(a_q31, b_q31),
        a_q31.saturating_mul(b_q31)
    );
    let _div_q31 = DspSample::sat_div(a_q31, b_q31);
    let _div_zero31 = DspSample::sat_div(a_q31, q31::ZERO);
    let _div_neg_zero31 = DspSample::sat_div(-a_q31, q31::ZERO);
    assert_eq!(DspSample::abs_val(-a_q31), a_q31);
    let _f_q31 = DspSample::to_f32(a_q31);
    let _q31_from_f = <q31 as DspSample>::from_f32(0.5);

    // Complex operations
    let c1 = Complex::new(1.0f32, 2.0f32);
    let c2 = Complex::new(3.0f32, 4.0f32);
    let c_add = c1 + c2;
    let c_sub = c1 - c2;
    let c_mul = c1 * c2;
    let c_scale = c1 * 2.0f32;
    let c_neg = -c1;
    assert_eq!(c_add.real, 4.0);
    assert_eq!(c_sub.real, -2.0);
    assert!(c_mul.real.is_finite());
    assert_eq!(c_scale.real, 2.0);
    assert_eq!(c_neg.real, -1.0);
}

#[test]
fn test_resampling_exhaustive() {
    let mut cic_dec = CicDecimator::<3>::new(4);
    assert!(cic_dec.gain() > 0);
    assert!(cic_dec.gain_bits() > 0);
    for i in 0..10 {
        let _out = cic_dec.process_sample(i * 10);
        let _out_s = cic_dec.process_sample_scaled(i * 10);
    }

    let mut cic_interp = CicInterpolator::<3>::new(4);
    assert!(cic_interp.gain() > 0);
    assert!(cic_interp.gain_bits() > 0);
    let mut interp_buf = [0i32; 4];
    for i in 0..5 {
        cic_interp.process_sample(i * 10, &mut interp_buf);
    }

    let coeffs_q15 = [q15::from_bits(4096); 8];
    let src_q15 = [q15::from_bits(1000); 16];
    let mut dst_dec = [q15::ZERO; 8];
    let count_dec = polyphase_decimate_q15(&src_q15, &coeffs_q15, 2, &mut dst_dec);
    assert!(count_dec > 0);

    let mut dst_interp = [q15::ZERO; 32];
    let count_interp = polyphase_interpolate_q15(&src_q15, &coeffs_q15, 2, &mut dst_interp);
    assert!(count_interp > 0);

    let mut dst_lin_q15 = [q15::ZERO; 32];
    resample_linear_q15(&src_q15, &mut dst_lin_q15, 0x00008000); // 0.5 ratio

    let src_f32 = [1.0f32; 16];
    let mut dst_lin_f32 = [0.0f32; 32];
    resample_linear_f32(&src_f32, &mut dst_lin_f32, 0.5);

    let mut dst_spec = [0.0f32; 32];
    let status_spec = spectral_interpolate_2x_f32(&src_f32, &mut dst_spec);
    assert_eq!(status_spec, Status::Success);
}

#[test]
fn test_spatial_exhaustive() {
    let src_img = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut dst_img = [0.0f32; 9];
    assert_eq!(dct2d_f32(&src_img, &mut dst_img, 3, 3), Status::Success);
    let mut recovered = [0.0f32; 9];
    assert_eq!(idct2d_f32(&dst_img, &mut recovered, 3, 3), Status::Success);

    let kernel = [0.0f32, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    let mut convolved = [0.0f32; 9];
    assert_eq!(
        convolve2d_f32(&src_img, &mut convolved, 3, 3, &kernel, 3, 3, true),
        Status::Success
    );

    let mut nonlin_out = [0.0f32; 9];
    assert_eq!(
        nonlin2d_filter_f32(&src_img, &mut nonlin_out, 3, 3, 3, NonlinFilterType::Min),
        Status::Success
    );
    assert_eq!(
        nonlin2d_filter_f32(&src_img, &mut nonlin_out, 3, 3, 3, NonlinFilterType::Max),
        Status::Success
    );
    assert_eq!(
        nonlin2d_filter_f32(&src_img, &mut nonlin_out, 3, 3, 3, NonlinFilterType::Median),
        Status::Success
    );

    let mut edges = [0.0f32; 9];
    assert_eq!(
        sobel_edge_detection_f32(&src_img, &mut edges, 3, 3, 2.0),
        Status::Success
    );

    let mut hist_bins = [0usize; 5];
    assert_eq!(
        histogram_2d_f32(&src_img, &mut hist_bins, 0.0, 10.0),
        Status::Success
    );

    let mse = mse_2d_f32(&src_img, &recovered);
    assert!(mse.is_finite());

    let psnr = psnr_2d_f32(&src_img, &recovered, 9.0);
    assert!(psnr.is_finite());
}

#[test]
fn test_fast_math_and_dynamics_exhaustive() {
    assert!(fast_exp_f32(1.0).is_finite());
    assert!(fast_tanh_f32(1.0).is_finite());
    assert!(log_f32(2.0).is_finite());
    assert!(exp_f32(1.0).is_finite());

    let mut s_f32 = 0.0f32;
    let mut c_f32 = 0.0f32;
    fast_math::sin_cos_f32(45.0, &mut s_f32, &mut c_f32);
    assert_ne!(fast_math::sin_f32(1.0), 0.0);
    assert_ne!(fast_math::cos_f32(1.0), 0.0);

    let mut s_q31 = q31::ZERO;
    let mut c_q31 = q31::ZERO;
    fast_math::sin_cos_q31(q31::from_bits(10000), &mut s_q31, &mut c_q31);
    assert_ne!(fast_math::sin_q31(q31::from_bits(10000)).to_bits(), 0);
    assert_ne!(fast_math::cos_q31(q31::from_bits(10000)).to_bits(), 0);

    // LUT negative inputs
    assert_ne!(lut::fast_sin_i16(-1.0), 0);
    assert_ne!(lut::fast_cos_i16(-1.0), 0);
    assert_ne!(lut::sin_q16(-10000), 0);
    assert_ne!(lut::cos_q16(-10000), 0);

    // CORDIC / Atan2 all 4 quadrants
    let mut res_f32 = 0.0f32;
    assert_eq!(atan2_f32(1.0, 1.0, &mut res_f32), Status::Success);

    let mut res_q31 = q31::ZERO;
    assert_eq!(
        atan2_q31(q31::from_bits(10000), q31::from_bits(10000), &mut res_q31),
        Status::Success
    );
    assert_eq!(
        atan2_q31(q31::from_bits(10000), q31::from_bits(-10000), &mut res_q31),
        Status::Success
    );
    assert_eq!(
        atan2_q31(q31::from_bits(-10000), q31::from_bits(10000), &mut res_q31),
        Status::Success
    );
    assert_eq!(
        atan2_q31(q31::from_bits(-10000), q31::from_bits(-10000), &mut res_q31),
        Status::Success
    );
    assert_eq!(
        atan2_q31(q31::ZERO, q31::ZERO, &mut res_q31),
        Status::Success
    );
    assert_eq!(
        atan2_q31(q31::from_bits(10000), q31::ZERO, &mut res_q31),
        Status::Success
    );

    let mut res_q15 = q15::ZERO;
    assert_eq!(
        atan2_q15(q15::from_bits(1000), q15::from_bits(1000), &mut res_q15),
        Status::Success
    );
    assert_eq!(
        atan2_q15(q15::from_bits(1000), q15::from_bits(-1000), &mut res_q15),
        Status::Success
    );
    assert_eq!(
        atan2_q15(q15::from_bits(-1000), q15::from_bits(1000), &mut res_q15),
        Status::Success
    );
    assert_eq!(
        atan2_q15(q15::from_bits(-1000), q15::from_bits(-1000), &mut res_q15),
        Status::Success
    );

    let mut out_f32 = 0.0f32;
    assert_eq!(sqrt_f32(4.0, &mut out_f32), Status::Success);
    assert_eq!(sqrt_f32(-1.0, &mut out_f32), Status::ArgumentError);

    let mut out_q31 = q31::ZERO;
    assert_eq!(
        sqrt_q31(q31::from_bits(1000000), &mut out_q31),
        Status::Success
    );
    assert_eq!(
        sqrt_q31(q31::from_bits(-100), &mut out_q31),
        Status::ArgumentError
    );

    let mut out_q15 = q15::ZERO;
    assert_eq!(
        sqrt_q15(q15::from_bits(10000), &mut out_q15),
        Status::Success
    );
    assert_eq!(
        sqrt_q15(q15::from_bits(-100), &mut out_q15),
        Status::ArgumentError
    );

    let src_v = [1.0f32, 4.0, 9.0, 16.0];
    let mut dst_v = [0.0f32; 4];
    vsqrt_f32(&src_v, &mut dst_v);

    let mut quot_q31 = q31::ZERO;
    let mut shift_q31 = 0i16;
    assert_eq!(
        divide_q31(
            q31::from_bits(5000),
            q31::from_bits(10000),
            &mut quot_q31,
            &mut shift_q31
        ),
        Status::Success
    );
    assert_eq!(
        divide_q31(
            q31::from_bits(5000),
            q31::ZERO,
            &mut quot_q31,
            &mut shift_q31
        ),
        Status::ArgumentError
    );

    let mut quot_q15 = q15::ZERO;
    let mut shift_q15 = 0i16;
    assert_eq!(
        divide_q15(
            q15::from_bits(500),
            q15::from_bits(1000),
            &mut quot_q15,
            &mut shift_q15
        ),
        Status::Success
    );
    assert_eq!(
        divide_q15(
            q15::from_bits(500),
            q15::ZERO,
            &mut quot_q15,
            &mut shift_q15
        ),
        Status::ArgumentError
    );

    // SIMD odd lengths (remainder loops)
    let src_odd_a = [q15::from_bits(100); 5];
    let src_odd_b = [q15::from_bits(200); 5];
    let mut dst_odd = [q15::ZERO; 5];
    let _dot_odd = intrinsics::simd_dot_prod_q15(&src_odd_a, &src_odd_b);
    intrinsics::simd_add_q15(&src_odd_a, &src_odd_b, &mut dst_odd);
    intrinsics::simd_sub_q15(&src_odd_a, &src_odd_b, &mut dst_odd);
    intrinsics::simd_mult_q15(&src_odd_a, &src_odd_b, &mut dst_odd);

    // SafetyLimiter
    let mut limiter = SafetyLimiter::new(0.9, 0.01, 16000.0);
    assert!(limiter.process(0.5).is_finite());
    assert!(limiter.process(1.5).is_finite());
    assert!(limiter.process_sample(0.2).is_finite());
    assert!(limiter.current_gain() <= 1.0);
    limiter.reset();

    // DynamicsCompressor
    let mut comp = DynamicsCompressor::new(-10.0, 4.0, 6.0, 0.001, 0.05, 3.0, 16000.0);
    assert!(comp.process(0.1).is_finite());
    assert!(comp.process(0.8).is_finite());
    assert!(comp.process_sample(0.8).is_finite());
    comp.reset();

    // NoiseGate
    let mut gate = NoiseGate::new(-40.0, -30.0, 0.001, 0.05, 16000.0);
    assert!(gate.process(0.001).is_finite());
    assert!(gate.process(0.5).is_finite());
    assert!(gate.process_sample(0.5).is_finite());
    gate.reset();
}
