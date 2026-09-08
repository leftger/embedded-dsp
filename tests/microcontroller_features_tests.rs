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
