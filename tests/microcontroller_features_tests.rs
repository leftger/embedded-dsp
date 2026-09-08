use embedded_dsp::*;

#[test]
fn test_eft_two_sum_and_diff() {
    // 1. two_sum_f32
    let a = 1.0f32;
    let b = 1.0e-7f32;
    let (s, r) = two_sum_f32(a, b);
    assert_eq!(s, a + b);
    assert!(r.abs() > 0.0);
    assert!(((s as f64 + r as f64) - (a as f64 + b as f64)).abs() < 1e-15);

    // 2. two_sum_f64
    let a64 = 1.0f64;
    let b64 = 1.0e-16f64;
    let (s64, r64) = two_sum_f64(a64, b64);
    assert_eq!(s64, a64 + b64);
    assert!(r64.abs() > 0.0);

    // 3. quick_two_sum_f32 (|a| >= |b|)
    let qa = 2.0f32;
    let qb = 1.0e-6f32;
    let (qs, qr) = quick_two_sum_f32(qa, qb);
    assert_eq!(qs, qa + qb);
    assert!(((qs as f64 + qr as f64) - (qa as f64 + qb as f64)).abs() < 1e-15);

    // 4. two_diff_f32
    let (d, dr) = two_diff_f32(a, b);
    assert_eq!(d, a - b);
    assert!(((d as f64 + dr as f64) - (a as f64 - b as f64)).abs() < 1e-15);
}

#[test]
fn test_compensated_sum_and_dot_prod() {
    // Test empty slices
    assert_eq!(sum_f32_compensated(&[]), 0.0);
    assert_eq!(dot_prod_f32_compensated(&[], &[]), 0.0);
    assert_eq!(dot_prod_f32_compensated(&[1.0], &[]), 0.0);

    // Single element
    assert_eq!(sum_f32_compensated(&[42.0]), 42.0);

    // Neumaier handles catastrophic cancellation and high dynamic range
    // where naive f32 addition fails
    let samples = [1.0e7f32, 1.0, -1.0e7, 2.0, -1.0e7, 1.0e7];
    let comp_sum = sum_f32_compensated(&samples);
    // Exact mathematical sum is 3.0
    assert!((comp_sum - 3.0).abs() < 1e-5);
    // Also test branch where next term is larger than running sum
    let reverse_samples = [1.0f32, 1.0e7, -1.0e7, 2.0];
    let comp_reverse = sum_f32_compensated(&reverse_samples);
    assert!((comp_reverse - 3.0).abs() < 1e-5);

    // Compensated dot product
    let vec_a = [1.0e5f32, 1.0, -1.0e5, 2.0];
    let vec_b = [1.0e2f32, 1.0, 1.0e2, 2.0];
    // Products: 1e7, 1, -1e7, 4. Sum should be 5.
    let comp_dot = dot_prod_f32_compensated(&vec_a, &vec_b);
    assert!((comp_dot - 5.0).abs() < 1e-4);

    // Compensated mean
    let mut mean_val = 0.0f32;
    assert_eq!(
        mean_f32_compensated(&[], &mut mean_val),
        Status::LengthError
    );
    assert_eq!(
        mean_f32_compensated(&samples, &mut mean_val),
        Status::Success
    );
    assert!((mean_val - (3.0 / 6.0)).abs() < 1e-5);
}

#[test]
fn test_adc_normalizer_all_resolutions_and_methods() {
    // 1. 12-bit ADC (STM32 typical, 0..4095 centered at 2048)
    let adc12 = AdcNormalizer::new(12, 2048);
    assert_eq!(adc12.shift_q15, 4);
    assert_eq!(adc12.shift_q31, 20);

    // Center point -> 0
    assert_eq!(adc12.to_q15_sample(2048), q15::ZERO);
    assert_eq!(adc12.to_q31_sample(2048), q31::ZERO);
    assert_eq!(adc12.to_f32_sample(2048), 0.0);

    // Min point -> negative saturation / -1.0
    assert_eq!(adc12.to_q15_sample(0), q15::MIN);
    assert_eq!(adc12.to_q31_sample(0), q31::MIN);
    assert_eq!(adc12.to_f32_sample(0), -1.0);

    // Max point (4095) -> near +1.0
    let q15_max = adc12.to_q15_sample(4095);
    assert!(q15_max.to_bits() > 32700);
    let q31_max = adc12.to_q31_sample(4095);
    assert!(q31_max.to_bits() > 2140000000);
    let f32_max = adc12.to_f32_sample(4095);
    assert!(f32_max > 0.999 && f32_max <= 1.0);

    // 2. 16-bit ADC (shift_q15 = 0, shift_q31 = 16)
    let adc16 = AdcNormalizer::new(16, 32768);
    assert_eq!(adc16.shift_q15, 0);
    assert_eq!(adc16.shift_q31, 16);
    assert_eq!(adc16.to_q15_sample(32768), q15::ZERO);
    assert_eq!(adc16.to_q31_sample(32768), q31::ZERO);
    assert_eq!(adc16.to_f32_sample(32768), 0.0);
    assert_eq!(adc16.to_q15_sample(0), q15::MIN);

    // 3. 8-bit ADC (shift_q15 = 8, shift_q31 = 24)
    let adc8 = AdcNormalizer::new(8, 128);
    assert_eq!(adc8.shift_q15, 8);
    assert_eq!(adc8.shift_q31, 24);
    assert_eq!(adc8.to_q15_sample(128), q15::ZERO);
    assert_eq!(adc8.to_q15_sample(0), q15::MIN);

    // 4. Over-voltage / clamp beyond normal range
    let adc10 = AdcNormalizer::new(10, 512);
    // Raw value above 10-bit range (e.g. 2000)
    assert_eq!(adc10.to_q15_sample(2000), q15::MAX);
    assert_eq!(adc10.to_q31_sample(2000), q31::MAX);
    assert_eq!(adc10.to_f32_sample(2000), 1.0);

    // 5. Slice normalization
    let raw_buf = [2048u16, 0, 4095, 3072];
    let mut dst_q15 = [q15::ZERO; 4];
    let mut dst_q31 = [q31::ZERO; 4];
    let mut dst_f32 = [0.0f32; 4];

    adc12.normalize_to_q15(&raw_buf, &mut dst_q15);
    adc12.normalize_to_q31(&raw_buf, &mut dst_q31);
    adc12.normalize_to_f32(&raw_buf, &mut dst_f32);

    assert_eq!(dst_q15[0], q15::ZERO);
    assert_eq!(dst_q15[1], q15::MIN);
    assert_eq!(dst_q31[0], q31::ZERO);
    assert_eq!(dst_q31[1], q31::MIN);
    assert_eq!(dst_f32[0], 0.0);
    assert_eq!(dst_f32[1], -1.0);

    // Mismatched slice length
    let mut small_q15 = [q15::ZERO; 2];
    adc12.normalize_to_q15(&raw_buf, &mut small_q15);
    assert_eq!(small_q15[0], q15::ZERO);
    assert_eq!(small_q15[1], q15::MIN);

    let mut small_q31 = [q31::ZERO; 2];
    adc12.normalize_to_q31(&raw_buf, &mut small_q31);
    assert_eq!(small_q31[0], q31::ZERO);

    let mut small_f32 = [0.0f32; 2];
    adc12.normalize_to_f32(&raw_buf, &mut small_f32);
    assert_eq!(small_f32[0], 0.0);
}

#[test]
fn test_wide_and_saturated_fixed_point_dot_products() {
    // Empty slices
    assert_eq!(dot_prod_q31_wide(&[], &[]), 0);
    assert_eq!(dot_prod_q31_saturated(&[], &[]), q31::ZERO);
    assert_eq!(dot_prod_q15_wide(&[], &[]), 0);
    assert_eq!(dot_prod_q15_saturated(&[], &[]), q15::ZERO);

    // Q31 wide and saturated
    let a_q31 = [q31::from_num(0.5), q31::from_num(0.5), q31::from_num(-0.25)];
    let b_q31 = [q31::from_num(0.5), q31::from_num(0.5), q31::from_num(0.5)];
    // Expected math: 0.25 + 0.25 - 0.125 = 0.375
    let wide_res = dot_prod_q31_wide(&a_q31, &b_q31);
    assert!(wide_res > 0);
    let sat_res = dot_prod_q31_saturated(&a_q31, &b_q31);
    let diff = (sat_res.to_num::<f32>() - 0.375f32).abs();
    assert!(diff < 1e-4);

    // Q31 saturation clamping on positive overflow
    let large_a_q31 = [q31::MAX, q31::MAX, q31::MAX];
    let large_b_q31 = [q31::MAX, q31::MAX, q31::MAX];
    let sat_overflow = dot_prod_q31_saturated(&large_a_q31, &large_b_q31);
    assert_eq!(sat_overflow, q31::MAX);

    // Q31 saturation clamping on negative overflow
    let neg_a_q31 = [q31::MIN, q31::MIN];
    let pos_b_q31 = [q31::MAX, q31::MAX];
    let sat_underflow = dot_prod_q31_saturated(&neg_a_q31, &pos_b_q31);
    assert_eq!(sat_underflow, q31::MIN);

    // Q15 wide and saturated
    let a_q15 = [q15::from_num(0.5), q15::from_num(0.5), q15::from_num(-0.25)];
    let b_q15 = [q15::from_num(0.5), q15::from_num(0.5), q15::from_num(0.5)];
    let wide_q15 = dot_prod_q15_wide(&a_q15, &b_q15);
    assert!(wide_q15 > 0);
    let sat_q15 = dot_prod_q15_saturated(&a_q15, &b_q15);
    let diff_q15 = (sat_q15.to_num::<f32>() - 0.375f32).abs();
    assert!(diff_q15 < 1e-3);

    // Q15 saturation clamping
    let large_a_q15 = [q15::MAX, q15::MAX, q15::MAX];
    let large_b_q15 = [q15::MAX, q15::MAX, q15::MAX];
    assert_eq!(dot_prod_q15_saturated(&large_a_q15, &large_b_q15), q15::MAX);

    let neg_a_q15 = [q15::MIN, q15::MIN];
    let pos_b_q15 = [q15::MAX, q15::MAX];
    assert_eq!(dot_prod_q15_saturated(&neg_a_q15, &pos_b_q15), q15::MIN);
}

#[test]
fn test_variance_based_qsnr_and_metrics() {
    // Empty inputs
    assert_eq!(root_mean_square_error(&[], &[]), 0.0);
    assert_eq!(signal_to_noise_ratio(&[], &[]), 140.0);
    assert_eq!(quantization_snr(&[], &[]), 140.0);

    // Exact match
    let signal = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(root_mean_square_error(&signal, &signal), 0.0);
    assert_eq!(signal_to_noise_ratio(&signal, &signal), 140.0);
    assert_eq!(quantization_snr(&signal, &signal), 140.0);

    // Signal with constant DC offset vs AC variation
    // Here ref has huge DC offset (100.0) + small AC sine (0.1, -0.1...)
    let ref_with_dc = [100.1f32, 99.9, 100.1, 99.9];
    let noisy_target = [100.15f32, 99.85, 100.15, 99.85]; // 0.05 noise

    let snr = signal_to_noise_ratio(&ref_with_dc, &noisy_target);
    let qsnr = quantization_snr(&ref_with_dc, &noisy_target);

    // Raw SNR is inflated by the 100.0 DC offset!
    assert!(snr > 60.0);
    // QSNR properly measures only the AC variance (0.1^2) against the noise (0.05^2)
    // 10 * log10(0.01 / 0.0025) = 10 * log10(4) ~= 6.02 dB
    assert!((qsnr - 6.02).abs() < 0.2);

    // Flat DC signal has 0 AC variance -> returns 0.0 dB
    let flat_dc = [5.0f32, 5.0, 5.0, 5.0];
    let flat_noisy = [5.1f32, 5.1, 5.1, 5.1];
    assert_eq!(quantization_snr(&flat_dc, &flat_noisy), 0.0);

    // Zero signal
    let zero_sig = [0.0f32, 0.0, 0.0];
    let noise_only = [0.1f32, 0.1, 0.1];
    assert_eq!(signal_to_noise_ratio(&zero_sig, &noise_only), 0.0);
    assert_eq!(quantization_snr(&zero_sig, &noise_only), 0.0);

    // Differential metrics integration
    let diff_metrics = evaluate_differential(&ref_with_dc, &noisy_target);
    assert!((diff_metrics.qsnr_db - qsnr).abs() < 1e-4);
    assert!((diff_metrics.sqnr_db - snr).abs() < 1e-4);
    assert!((diff_metrics.peak_absolute_error - 0.05).abs() < 1e-4);

    // Mismatched slice lengths
    let shorter_target = [100.15f32, 99.85];
    assert!(root_mean_square_error(&ref_with_dc, &shorter_target) > 0.0);
    assert!(signal_to_noise_ratio(&ref_with_dc, &shorter_target) > 0.0);
    assert!(quantization_snr(&ref_with_dc, &shorter_target) > 0.0);
    let diff_mismatch = evaluate_differential(&ref_with_dc, &shorter_target);
    assert!(!diff_mismatch.is_exact_match);
}

#[test]
fn test_microcontroller_edge_cases() {
    // AdcNormalizer clamping and traits
    let adc_clamped_low = AdcNormalizer::new(0, 100);
    assert_eq!(adc_clamped_low.shift_q15, 15);
    assert_eq!(adc_clamped_low.shift_q31, 31);

    let adc_clamped_high = AdcNormalizer::new(20, 100);
    assert_eq!(adc_clamped_high.shift_q15, 0);
    assert_eq!(adc_clamped_high.shift_q31, 16);

    let adc_copy = adc_clamped_low;
    assert_eq!(adc_clamped_low, adc_copy);
    let debug_str = format!("{:?}", adc_clamped_low);
    assert!(debug_str.contains("AdcNormalizer"));

    // Mismatched lengths for dot products
    let a_q31 = [q31::from_num(0.5), q31::from_num(0.5)];
    let b_q31 = [q31::from_num(0.5)];
    assert!(dot_prod_q31_wide(&a_q31, &b_q31) > 0);
    assert!(dot_prod_q31_saturated(&a_q31, &b_q31) > q31::ZERO);

    let a_q15 = [q15::from_num(0.5), q15::from_num(0.5)];
    let b_q15 = [q15::from_num(0.5)];
    assert!(dot_prod_q15_wide(&a_q15, &b_q15) > 0);
    assert!(dot_prod_q15_saturated(&a_q15, &b_q15) > q15::ZERO);

    // Compensated dot product mismatched length
    let a_f32 = [1.0f32, 2.0];
    let b_f32 = [3.0f32];
    assert_eq!(dot_prod_f32_compensated(&a_f32, &b_f32), 3.0);
}

#[test]
fn test_bfloat16_operations_and_edge_cases() {
    use std::collections::HashSet;

    // Constants
    assert_eq!(BFloat16::ZERO.to_f32(), 0.0);
    assert_eq!(BFloat16::NEG_ZERO.to_f32(), -0.0);
    assert_eq!(BFloat16::ONE.to_f32(), 1.0);
    assert_eq!(BFloat16::NEG_ONE.to_f32(), -1.0);
    assert!(BFloat16::NAN.is_nan());
    assert!(BFloat16::INFINITY.is_infinite());
    assert!(BFloat16::NEG_INFINITY.is_infinite());
    assert!(BFloat16::MAX.to_f32() > 3.3e38);
    assert!(BFloat16::MIN.to_f32() < -3.3e38);
    assert!(BFloat16::MIN_POSITIVE.to_f32() > 0.0);

    // Queries
    assert!(BFloat16::ZERO.is_zero());
    assert!(BFloat16::NEG_ZERO.is_zero());
    assert!(!BFloat16::ONE.is_zero());

    assert!(BFloat16::ONE.is_sign_positive());
    assert!(!BFloat16::NEG_ONE.is_sign_positive());
    assert!(BFloat16::NEG_ONE.is_sign_negative());
    assert!(!BFloat16::ONE.is_sign_negative());

    assert!(BFloat16::ONE.is_finite());
    assert!(!BFloat16::INFINITY.is_finite());
    assert!(!BFloat16::NAN.is_finite());

    // abs
    assert_eq!(BFloat16::NEG_ONE.abs(), BFloat16::ONE);
    assert_eq!(BFloat16::ONE.abs(), BFloat16::ONE);

    // from_bits and to_bits
    let bf = BFloat16::from_bits(0x3F80);
    assert_eq!(bf.to_bits(), 0x3F80);
    assert_eq!(bf, BFloat16::ONE);

    // from_f32 and to_f32
    let bf_pi = BFloat16::from_f32(core::f32::consts::PI);
    let f_pi = bf_pi.to_f32();
    assert!((f_pi - core::f32::consts::PI).abs() < 0.02);

    // NaN handling preserves NaN and sets quiet bit
    let nan_f32 = f32::from_bits(0x7F80_0001); // signaling NaN in f32
    let bf_nan = BFloat16::from_f32(nan_f32);
    assert!(bf_nan.is_nan());

    // Conversions via From / Into traits
    let bf_conv: BFloat16 = 2.5f32.into();
    let f_conv: f32 = bf_conv.into();
    assert_eq!(f_conv, 2.5f32);

    // Display and Debug
    let disp = format!("{}", BFloat16::ONE);
    assert_eq!(disp, "1");
    let dbg = format!("{:?}", BFloat16::ONE);
    assert!(dbg.contains("BFloat16(1.0)"));

    // Derive traits (Default, Ord, Hash)
    assert_eq!(BFloat16::default(), BFloat16::ZERO);
    assert!(BFloat16::ONE > BFloat16::ZERO);
    let mut set = HashSet::new();
    set.insert(BFloat16::ONE);
    assert!(set.contains(&BFloat16::ONE));

    // Support slice conversions
    let raw_floats = [0.0f32, 1.0, -1.0, core::f32::consts::PI, 100.0];
    let mut bf_buf = [BFloat16::ZERO; 5];
    let mut f_buf = [0.0f32; 5];

    f32_to_bfloat16(&raw_floats, &mut bf_buf);
    bfloat16_to_f32(&bf_buf, &mut f_buf);

    for i in 0..5 {
        assert!((raw_floats[i] - f_buf[i]).abs() < (raw_floats[i].abs() * 0.01).max(1e-5));
    }

    // Mismatched length slice conversions
    let mut short_bf = [BFloat16::ZERO; 2];
    f32_to_bfloat16(&raw_floats, &mut short_bf);
    assert_eq!(short_bf[1], BFloat16::ONE);

    let mut short_f = [0.0f32; 2];
    bfloat16_to_f32(&bf_buf, &mut short_f);
    assert_eq!(short_f[1], 1.0);
}

#[test]
fn test_floatfloat_extended_precision() {
    // Constants and constructors
    assert_eq!(FloatFloat::ZERO.to_f32(), 0.0);
    assert_eq!(FloatFloat::ONE.to_f32(), 1.0);

    let ff = FloatFloat::new(1.0, 1e-7);
    assert_eq!(ff.hi, 1.0);
    assert_eq!(ff.lo, 1e-7);
    assert_eq!(FloatFloat::from_f32(3.5).to_f32(), 3.5);

    // to_f64
    let val_f64 = ff.to_f64();
    assert!((val_f64 - 1.0000001).abs() < 1e-12);

    // abs
    let neg_ff = -ff;
    assert_eq!(neg_ff.abs(), ff);
    assert_eq!(ff.abs(), ff);

    // Addition and Subtraction
    let a = FloatFloat::from_f32(1.0);
    let b = FloatFloat::new(1e-8, 0.0);
    let sum = a + b;
    let diff = sum - b;
    assert!((diff.to_f64() - 1.0).abs() < 1e-14);

    // Multiplication with extended precision that standard f32 loses
    let x = FloatFloat::new(1.0, 1e-8);
    let y = FloatFloat::from_f32(2.0);
    let prod = x * y;
    let expected = (1.0f64 + 1e-8f64) * 2.0f64;
    assert!((prod.to_f64() - expected).abs() < 1e-14);

    // Standard f32 completely loses the 1e-8 component
    assert_eq!((1.0f32 + 1e-8f32) * 2.0f32, 2.0f32);
    // While FloatFloat preserved it!
    assert!(prod.to_f64() > 2.0);

    // Division
    let dividend = FloatFloat::from_f32(1.0);
    let divisor = FloatFloat::from_f32(3.0);
    let quotient = dividend / divisor;
    let expected = 1.0 / 3.0;
    assert!((quotient.to_f64() - expected).abs() < 1e-12);

    // Display, Debug, Default, Clone, Copy
    let dbg = format!("{:?}", ff);
    assert!(dbg.contains("FloatFloat"));
    let disp = format!("{}", ff);
    assert!(disp.contains("1.0000001"));
    assert_eq!(FloatFloat::default(), FloatFloat::ZERO);
}

#[test]
fn test_strided_dot_products() {
    // 1. f32 strided dot product
    // Interleaved stereo buffer: [L0, R0, L1, R1, L2, R2]
    let interleaved_stereo_a = [1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0];
    let interleaved_stereo_b = [4.0f32, 40.0, 5.0, 50.0, 6.0, 60.0];

    // Left channel dot product: L0*L0 + L1*L1 + L2*L2 = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    let dot_l = dot_prod_f32_strided(&interleaved_stereo_a, 2, &interleaved_stereo_b, 2, 3);
    assert_eq!(dot_l, 32.0);

    // Right channel dot product: 10*40 + 20*50 + 30*60 = 400 + 1000 + 1800 = 3200
    let dot_r = dot_prod_f32_strided(
        &interleaved_stereo_a[1..],
        2,
        &interleaved_stereo_b[1..],
        2,
        3,
    );
    assert_eq!(dot_r, 3200.0);

    // Edge cases for f32: count 0, stride 0, truncation past slice length
    assert_eq!(
        dot_prod_f32_strided(&interleaved_stereo_a, 0, &interleaved_stereo_b, 2, 3),
        0.0
    );
    assert_eq!(
        dot_prod_f32_strided(&interleaved_stereo_a, 2, &interleaved_stereo_b, 0, 3),
        0.0
    );
    assert_eq!(
        dot_prod_f32_strided(&interleaved_stereo_a, 2, &interleaved_stereo_b, 2, 0),
        0.0
    );
    // Requesting count 10 when only 3 elements available
    assert_eq!(
        dot_prod_f32_strided(&interleaved_stereo_a, 2, &interleaved_stereo_b, 2, 10),
        32.0
    );

    // 2. f64 strided dot product
    let a64 = [1.0f64, 10.0, 2.0, 20.0, 3.0, 30.0];
    let b64 = [4.0f64, 40.0, 5.0, 50.0, 6.0, 60.0];
    let dot_l64 = dot_prod_f64_strided(&a64, 2, &b64, 2, 3);
    assert_eq!(dot_l64, 32.0);
    assert_eq!(dot_prod_f64_strided(&a64, 0, &b64, 2, 3), 0.0);
    assert_eq!(dot_prod_f64_strided(&a64, 2, &b64, 0, 3), 0.0);
    assert_eq!(dot_prod_f64_strided(&a64, 2, &b64, 2, 0), 0.0);
    assert_eq!(dot_prod_f64_strided(&a64, 2, &b64, 2, 10), 32.0);

    // 3. q31 strided dot product
    let a_q31 = [
        q31::from_num(0.25),
        q31::from_num(0.0),
        q31::from_num(0.5),
        q31::from_num(0.0),
    ];
    let b_q31 = [
        q31::from_num(0.25),
        q31::from_num(0.0),
        q31::from_num(0.5),
        q31::from_num(0.0),
    ];
    let dot_q31 = dot_prod_q31_strided(&a_q31, 2, &b_q31, 2, 2);
    assert!(dot_q31 > 0);
    assert_eq!(dot_prod_q31_strided(&a_q31, 0, &b_q31, 2, 2), 0);
    assert_eq!(dot_prod_q31_strided(&a_q31, 2, &b_q31, 0, 2), 0);
    assert_eq!(dot_prod_q31_strided(&a_q31, 2, &b_q31, 2, 0), 0);
    assert_eq!(dot_prod_q31_strided(&a_q31, 2, &b_q31, 2, 10), dot_q31);

    // 4. q15 strided dot product
    let a_q15 = [
        q15::from_num(0.25),
        q15::from_num(0.0),
        q15::from_num(0.5),
        q15::from_num(0.0),
    ];
    let b_q15 = [
        q15::from_num(0.25),
        q15::from_num(0.0),
        q15::from_num(0.5),
        q15::from_num(0.0),
    ];
    let dot_q15 = dot_prod_q15_strided(&a_q15, 2, &b_q15, 2, 2);
    assert!(dot_q15 > 0);
    assert_eq!(dot_prod_q15_strided(&a_q15, 0, &b_q15, 2, 2), 0);
    assert_eq!(dot_prod_q15_strided(&a_q15, 2, &b_q15, 0, 2), 0);
    assert_eq!(dot_prod_q15_strided(&a_q15, 2, &b_q15, 2, 0), 0);
    assert_eq!(dot_prod_q15_strided(&a_q15, 2, &b_q15, 2, 10), dot_q15);
}

#[test]
fn test_eft_two_prod_and_two_div() {
    // 1. two_prod_f32
    let a = 1.0000001f32;
    let b = 1.0000002f32;
    let (p, r) = two_prod_f32(a, b);
    assert_eq!(p, a * b);
    let exact_prod = (a as f64) * (b as f64);
    let eft_prod = (p as f64) + (r as f64);
    assert!((exact_prod - eft_prod).abs() < 1e-15);

    // 2. two_prod_f64
    let a64 = 1.0000000000000002f64;
    let b64 = 1.0000000000000004f64;
    let (p64, r64) = two_prod_f64(a64, b64);
    assert_eq!(p64, a64 * b64);
    assert!(r64 != 0.0);

    // 3. two_div_f32
    let div_a = 1.0f32;
    let div_b = 3.0f32;
    let (q, r_div) = two_div_f32(div_a, div_b);
    assert_eq!(q, div_a / div_b);
    let exact_quot = (div_a as f64) / (div_b as f64);
    let eft_quot = (q as f64) + (r_div as f64);
    assert!((exact_quot - eft_quot).abs() < 1e-14);
}

#[test]
fn test_polynomial_eval_and_root_finding() {
    // 1. poly_eval_f32: P(x) = 1 + 2x + 3x^2
    let coeffs_f32 = [1.0f32, 2.0, 3.0];
    assert_eq!(poly_eval_f32(&[], 2.0), 0.0);
    assert_eq!(poly_eval_f32(&[5.0], 2.0), 5.0);
    // P(2) = 1 + 2*2 + 3*4 = 17
    assert_eq!(poly_eval_f32(&coeffs_f32, 2.0), 17.0);

    // 2. poly_eval_f64
    let coeffs_f64 = [1.0f64, 2.0, 3.0];
    assert_eq!(poly_eval_f64(&[], 2.0), 0.0);
    assert_eq!(poly_eval_f64(&[5.0], 2.0), 5.0);
    assert_eq!(poly_eval_f64(&coeffs_f64, 2.0), 17.0);

    // 3. poly_eval_q15: P(x) = 0.125 + 0.25x
    let coeffs_q15 = [q15::from_num(0.125), q15::from_num(0.25)];
    assert_eq!(poly_eval_q15(&[], q15::from_num(0.5)), q15::ZERO);
    assert_eq!(
        poly_eval_q15(&[q15::from_num(0.5)], q15::from_num(0.5)),
        q15::from_num(0.5)
    );
    let r_q15 = poly_eval_q15(&coeffs_q15, q15::from_num(0.5));
    // 0.125 + 0.25 * 0.5 = 0.25
    assert_eq!(r_q15, q15::from_num(0.25));

    // 4. poly_eval_q31: P(x) = 0.125 + 0.25x
    let coeffs_q31 = [q31::from_num(0.125), q31::from_num(0.25)];
    assert_eq!(poly_eval_q31(&[], q31::from_num(0.5)), q31::ZERO);
    assert_eq!(
        poly_eval_q31(&[q31::from_num(0.5)], q31::from_num(0.5)),
        q31::from_num(0.5)
    );
    let r_q31 = poly_eval_q31(&coeffs_q31, q31::from_num(0.5));
    assert_eq!(r_q31, q31::from_num(0.25));

    // 5. poly_eval_with_deriv_f32: P(x) = 1 + 2x + 3x^2, P'(x) = 2 + 6x
    assert_eq!(poly_eval_with_deriv_f32(&[], 2.0), (0.0, 0.0));
    let (p, d) = poly_eval_with_deriv_f32(&coeffs_f32, 2.0);
    assert_eq!(p, 17.0);
    assert_eq!(d, 14.0);

    // 6. poly_root_f32
    // Length error
    assert_eq!(poly_root_f32(&[], 1.0, 20, 1e-5), Err(Status::LengthError));
    assert_eq!(
        poly_root_f32(&[1.0], 1.0, 20, 1e-5),
        Err(Status::LengthError)
    );

    // Linear root: P(x) = -6 + 2x => root at x = 3.0
    let linear_coeffs = [-6.0f32, 2.0];
    let root_lin = poly_root_f32(&linear_coeffs, 0.0, 20, 1e-6).unwrap();
    assert!((root_lin - 3.0).abs() < 1e-5);

    // Quadratic root: P(x) = -4 + x^2 => roots at +-2.0
    let quad_coeffs = [-4.0f32, 0.0, 1.0];
    let root_pos = poly_root_f32(&quad_coeffs, 1.0, 20, 1e-6).unwrap();
    assert!((root_pos - 2.0).abs() < 1e-5);
    let root_neg = poly_root_f32(&quad_coeffs, -1.0, 20, 1e-6).unwrap();
    assert!((root_neg - (-2.0)).abs() < 1e-5);

    // Initial guess is already an exact root
    let root_exact = poly_root_f32(&quad_coeffs, 2.0, 20, 1e-6).unwrap();
    assert_eq!(root_exact, 2.0);

    // Flat derivative MathError: P(x) = 1 + x^2 at x = 0 has derivative 0
    let flat_coeffs = [1.0f32, 0.0, 1.0];
    assert_eq!(
        poly_root_f32(&flat_coeffs, 0.0, 20, 1e-6),
        Err(Status::Singular)
    );

    // Non-converging TestFailure: max_iter exhausted before reaching tolerance
    assert_eq!(
        poly_root_f32(&quad_coeffs, 10.0, 1, 1e-6),
        Err(Status::TestFailure)
    );

    // Default tolerance when tol <= 0.0
    let root_def_tol = poly_root_f32(&linear_coeffs, 0.0, 20, -1.0).unwrap();
    assert!((root_def_tol - 3.0).abs() < 1e-5);
}

#[test]
fn test_fast_bit_manipulation_log_and_pow() {
    // 1. Non-positive inputs for logarithms
    assert_eq!(fast_log2_f32(0.0), f32::NEG_INFINITY);
    assert_eq!(fast_log2_f32(-5.0), f32::NEG_INFINITY);
    assert_eq!(fast_ln_f32(0.0), f32::NEG_INFINITY);
    assert_eq!(fast_log10_f32(0.0), f32::NEG_INFINITY);
    assert_eq!(fast_gain_to_db_f32(0.0), f32::NEG_INFINITY);

    // 2. Base-2 logarithm accuracy
    assert!((fast_log2_f32(1.0) - 0.0).abs() < 0.01);
    assert!((fast_log2_f32(2.0) - 1.0).abs() < 0.01);
    assert!((fast_log2_f32(4.0) - 2.0).abs() < 0.01);
    assert!((fast_log2_f32(8.0) - 3.0).abs() < 0.01);
    assert!((fast_log2_f32(0.5) - (-1.0)).abs() < 0.01);
    assert!((fast_log2_f32(0.25) - (-2.0)).abs() < 0.01);

    // 3. Natural logarithm accuracy
    assert!((fast_ln_f32(core::f32::consts::E) - 1.0).abs() < 0.01);
    assert!((fast_ln_f32(1.0) - 0.0).abs() < 0.01);

    // 4. Base-10 logarithm accuracy
    assert!((fast_log10_f32(1.0) - 0.0).abs() < 0.01);
    assert!((fast_log10_f32(10.0) - 1.0).abs() < 0.01);
    assert!((fast_log10_f32(100.0) - 2.0).abs() < 0.02);

    // 5. Base-2 exponential accuracy and limits
    assert_eq!(fast_pow2_f32(-130.0), 0.0);
    assert_eq!(fast_pow2_f32(130.0), f32::INFINITY);
    assert!((fast_pow2_f32(0.0) - 1.0).abs() < 0.01);
    assert!((fast_pow2_f32(1.0) - 2.0).abs() < 0.02);
    assert!((fast_pow2_f32(3.0) - 8.0).abs() < 0.05);
    assert!((fast_pow2_f32(-1.0) - 0.5).abs() < 0.01);

    // 6. Base-10 exponential
    assert!((fast_pow10_f32(0.0) - 1.0).abs() < 0.01);
    assert!((fast_pow10_f32(1.0) - 10.0).abs() < 0.1);
    assert!((fast_pow10_f32(2.0) - 100.0).abs() < 1.0);

    // 7. Decibel conversions
    // 0 dB is unity gain
    assert!((fast_gain_to_db_f32(1.0) - 0.0).abs() < 0.1);
    assert!((fast_db_to_gain_f32(0.0) - 1.0).abs() < 0.01);

    // +6.02 dB is ~2x gain
    assert!((fast_gain_to_db_f32(2.0) - 6.02).abs() < 0.2);
    assert!((fast_db_to_gain_f32(6.02) - 2.0).abs() < 0.05);

    // -20 dB is 0.1x gain
    assert!((fast_gain_to_db_f32(0.1) - (-20.0)).abs() < 0.2);
    assert!((fast_db_to_gain_f32(-20.0) - 0.1).abs() < 0.01);
}

#[test]
fn test_hilbert_transform_and_analytic_signal() {
    // 1. Constants verification
    assert_eq!(HILBERT_COEFFS_35.len(), 35);
    assert_eq!(HILBERT_COEFFS_35[17], 0.0); // Center tap is 0
    assert_eq!(HILBERT_COEFFS_35_Q15.len(), 35);
    assert_eq!(HILBERT_COEFFS_35_Q15[17], q15::ZERO);

    // 2. hilbert_fir_design_f32
    let mut bad_coeffs_even = [0.0f32; 4];
    assert_eq!(
        hilbert_fir_design_f32(&mut bad_coeffs_even),
        Status::ArgumentError
    );
    let mut bad_coeffs_short = [0.0f32; 1];
    assert_eq!(
        hilbert_fir_design_f32(&mut bad_coeffs_short),
        Status::ArgumentError
    );

    let mut designed_15 = [0.0f32; 15];
    assert_eq!(hilbert_fir_design_f32(&mut designed_15), Status::Success);
    assert_eq!(designed_15[7], 0.0); // Center tap M=7 is 0
    // Odd taps should be anti-symmetric: h[7+k] == -h[7-k]
    for k in 1..=7 {
        assert!((designed_15[7 + k] + designed_15[7 - k]).abs() < 1e-6);
    }
    // Even k should be 0
    assert_eq!(designed_15[7 + 2], 0.0);
    assert_eq!(designed_15[7 - 2], 0.0);

    // 3. HilbertTransformF32 construction & errors
    let mut state_35 = [0.0f32; 35];
    let mut short_state = [0.0f32; 30];
    assert!(HilbertTransformF32::new(&HILBERT_COEFFS_35, &mut short_state).is_err());
    assert!(HilbertTransformF32::new(&bad_coeffs_even, &mut state_35[..4]).is_err());

    let mut hilbert = HilbertTransformF32::new(&HILBERT_COEFFS_35, &mut state_35).unwrap();
    assert_eq!(hilbert.group_delay(), 17);

    // 4. Cosine wave analytic signal:
    // x[n] = cos(omega * n).
    // For n > group_delay, I[n] = cos(omega * (n - 17)), Q[n] = sin(omega * (n - 17)).
    // Envelope sqrt(I^2 + Q^2) should settle to 1.0!
    let freq = 0.1f32;
    let pi2 = 2.0 * core::f32::consts::PI;
    let mut cos_signal = [0.0f32; 80];
    for (i, val) in cos_signal.iter_mut().enumerate() {
        *val = (pi2 * freq * i as f32).cos();
    }

    let mut analytic_out = [Complex::<f32>::default(); 80];
    assert_eq!(
        hilbert.process_analytic_block(&cos_signal, &mut analytic_out),
        Status::Success
    );

    // After 25 samples (well past 17-sample delay settling), check envelope and 90 deg phase
    for sample in analytic_out.iter().take(75).skip(25) {
        let i_sample = sample.real;
        let q_sample = sample.imag;
        let env = (i_sample * i_sample + q_sample * q_sample).sqrt();
        assert!((env - 1.0).abs() < 0.08); // FIR bandpass ripple is within ±0.5 dB (~6%)
    }

    // Single sample processing and reset
    hilbert.reset();
    let (i_val, q_val) = hilbert.process_sample(1.0);
    assert_eq!(i_val, 0.0); // Filter delay is 17, so delayed sample is 0
    assert_eq!(q_val, HILBERT_COEFFS_35[0]); // State[0] * coeffs[0]

    let mut quad_buf = [0.0f32; 10];
    assert_eq!(
        hilbert.process_block(&cos_signal[..10], &mut quad_buf),
        Status::Success
    );

    // 5. HilbertTransformQ15
    let mut state_q15 = [q15::ZERO; 35];
    let mut short_state_q15 = [q15::ZERO; 10];
    assert!(HilbertTransformQ15::new(&HILBERT_COEFFS_35_Q15, &mut short_state_q15).is_err());

    let mut hilbert_q15 = HilbertTransformQ15::new(&HILBERT_COEFFS_35_Q15, &mut state_q15).unwrap();
    assert_eq!(hilbert_q15.group_delay(), 17);
    hilbert_q15.reset();

    let mut cos_q15 = [q15::ZERO; 80];
    for (i, val) in cos_q15.iter_mut().enumerate() {
        *val = q15::from_num(cos_signal[i] * 0.9); // scale to avoid saturation
    }
    let mut q15_analytic = [Complex::<q15>::default(); 80];
    assert_eq!(
        hilbert_q15.process_analytic_block(&cos_q15, &mut q15_analytic),
        Status::Success
    );

    let mut q15_quad = [q15::ZERO; 10];
    assert_eq!(
        hilbert_q15.process_block(&cos_q15[..10], &mut q15_quad),
        Status::Success
    );

    hilbert_q15.reset();
    let (q15_i, q15_q) = hilbert_q15.process_sample(q15::from_num(0.5));
    hilbert_q15.reset();
    let cmx_sample = hilbert_q15.process_analytic_sample(q15::from_num(0.5));
    assert_eq!(cmx_sample.real, q15_i);
    assert_eq!(cmx_sample.imag, q15_q);

    // 6. analytic_envelope_f32 and analytic_phase_f32
    let test_cmx = [
        Complex {
            real: 3.0f32,
            imag: 4.0,
        },
        Complex {
            real: 0.0f32,
            imag: 1.0,
        },
        Complex {
            real: -1.0f32,
            imag: 0.0,
        },
    ];
    let mut env_buf = [0.0f32; 3];
    let mut phase_buf = [0.0f32; 3];

    analytic_envelope_f32(&test_cmx, &mut env_buf);
    analytic_phase_f32(&test_cmx, &mut phase_buf);

    assert_eq!(env_buf[0], 5.0);
    assert_eq!(env_buf[1], 1.0);
    assert_eq!(env_buf[2], 1.0);

    assert!((phase_buf[0] - 4.0f32.atan2(3.0)).abs() < 1e-6);
    assert!((phase_buf[1] - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((phase_buf[2] - core::f32::consts::PI).abs() < 1e-6);

    // Mismatched lengths
    let mut short_env = [0.0f32; 1];
    analytic_envelope_f32(&test_cmx, &mut short_env);
    assert_eq!(short_env[0], 5.0);

    let mut short_phase = [0.0f32; 1];
    analytic_phase_f32(&test_cmx, &mut short_phase);
    assert!((short_phase[0] - 4.0f32.atan2(3.0)).abs() < 1e-6);
}

#[test]
fn test_additional_coverage_branches() {
    // 1. fast_pow2_f32 underflow saturation and overflow
    assert_eq!(fast_pow2_f32(-130.0), 0.0);
    assert_eq!(fast_pow2_f32(130.0), f32::INFINITY);

    // 2. poly_root_f32 step.abs() < tol_pos branch (large derivative, small step)
    // P(x) = 1000 * x - 1000, root at 1.0. Start at 1.0001 with tol = 1e-3.
    // p = 0.1 >= 1e-3, d = 1000, step = 0.0001 < 1e-3 => step < tol triggers!
    let steep_poly = [-1000.0f32, 1000.0];
    let root_step = poly_root_f32(&steep_poly, 1.0001, 10, 1e-3).unwrap();
    assert!((root_step - 1.0).abs() < 1e-3);

    // 3. poly_root_f32 loop exhausted but p.abs() < tol * 10.0 branch
    // P(x) = x^2 - 2, root at sqrt(2) ~ 1.41421356. Start at 1.0 with max_iter = 2 and tol = 1e-4.
    // Iter 0: x = 1.0, p = -1, d = 2, step = -0.5, x = 1.5.
    // Iter 1: x = 1.5, p = 0.25, d = 3, step = 0.0833, x = 1.41666.
    // Iter 2: p(1.41666) = 0.00694. If tol = 1e-3: p < tol * 10 (0.01) triggers Ok after loop!
    let quad = [-2.0f32, 0.0, 1.0];
    let root_near = poly_root_f32(&quad, 1.0, 2, 1e-3).unwrap();
    assert!((root_near - 1.4142).abs() < 0.01);

    // 4. Wavelet length limits (m > 1024)
    let h_daub = [0.4829629f32, 0.8365163, 0.22414386, -0.12940952];
    let mut large_wavelet = [0.0f32; 2048];
    assert_eq!(
        wavelet_step_f32(&mut large_wavelet, 2048, &h_daub),
        Status::LengthError
    );
    assert_eq!(
        inverse_wavelet_step_f32(&mut large_wavelet, 2048, &h_daub),
        Status::LengthError
    );

    // 5. Window boundary cases (0 and 1 length)
    let mut empty_kaiser = [0.0f32; 0];
    kaiser_f32(&mut empty_kaiser, 5.0);
    let mut single_kaiser = [0.0f32; 1];
    kaiser_f32(&mut single_kaiser, 5.0);
    assert_eq!(single_kaiser[0], 1.0);

    let mut empty_q15 = [q15::ZERO; 0];
    hanning_q15(&mut empty_q15);
    let mut single_q15 = [q15::ZERO; 1];
    hanning_q15(&mut single_q15);
    assert_eq!(single_q15[0], q15::MAX);

    // 6. Fast division edge cases (denominator = 0)
    let mut q_res = q31::ZERO;
    let mut q_shift = 0i16;
    assert_eq!(
        divide_q31(q31::from_num(0.5), q31::ZERO, &mut q_res, &mut q_shift),
        Status::ArgumentError
    );
    let mut q15_res = q15::ZERO;
    let mut q15_shift = 0i16;
    assert_eq!(
        divide_q15(q15::from_num(0.5), q15::ZERO, &mut q15_res, &mut q15_shift),
        Status::ArgumentError
    );

    // 7. Distance empty inputs
    assert_eq!(euclidean_distance_f32(&[], &[]), 0.0);
    assert_eq!(chebyshev_distance_f32(&[], &[]), 0.0);
    assert_eq!(cosine_distance_f32(&[], &[]), 1.0);
}
