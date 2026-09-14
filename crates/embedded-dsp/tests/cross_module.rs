//! Consolidated tests: push_to_95_coverage_a, push_to_95_coverage_b, push_to_95_coverage_c, push_to_95_coverage_d, push_to_95_coverage_e, coverage_boost_tests, more_coverage_boost, final_90_plus_coverage_boost.

use embedded_dsp::pipeline::DspNode;
use embedded_dsp::pipeline::*;
use embedded_dsp::*;

// ─── from push_to_95_coverage_a.rs ────────────────────────────────────────
#[test]
fn test_audio_exhaustive() {
    let mut detector = GoertzelDetector::new(1000.0, 16000.0);
    let sample_block = [0.1f32; 160];
    for &s in &sample_block {
        detector.process_sample(s);
    }
    assert!(detector.magnitude().is_finite());
    detector.reset();

    let mut q15_detector = GoertzelDetectorQ15::new(1000.0, 16000.0);
    let q15_block = [q15::from_bits(1000); 160];
    for &s in &q15_block {
        q15_detector.process_sample(s);
    }
    assert!(q15_detector.magnitude().to_bits() >= 0);
    q15_detector.reset();

    let mut peak_env = PeakEnvelopeFollower::new(10.0, 100.0);
    assert!(peak_env.process(0.5).is_finite());
    peak_env.reset();

    let mut rms_env = RmsEnvelopeFollower::new(100.0);
    assert!(rms_env.process(0.5).is_finite());
    rms_env.reset();

    let mut peak_env_q15 = PeakEnvelopeFollowerQ15::new(10.0, 100.0);
    assert!(peak_env_q15.process(q15::from_bits(10000)).to_bits() >= 0);
    peak_env_q15.reset();

    let mut rms_env_q15 = RmsEnvelopeFollowerQ15::new(100.0);
    assert!(rms_env_q15.process(q15::from_bits(10000)).to_bits() >= 0);
    rms_env_q15.reset();

    assert!(hz_to_mel(1000.0).is_finite());
    assert!(mel_to_hz(1000.0).is_finite());

    let fft_mag = [1.0f32; 64];
    let mut filterbank_energies = [0.0f32; 10];
    let status = mel_filterbank_f32(
        &fft_mag,
        64,
        16000.0,
        100.0,
        8000.0,
        &mut filterbank_energies,
    );
    assert_eq!(status, Status::Success);

    let frame = [0.1f32; 64];
    let mut mel_scratch = [0.0f32; 16];
    let mut mfcc_coeffs = [0.0f32; 10];
    let status = mfcc_f32(
        &frame,
        16000.0,
        100.0,
        8000.0,
        &mut mel_scratch,
        &mut mfcc_coeffs,
    );
    assert_eq!(status, Status::Success);

    let left = [0usize, 1, 2];
    let center = [1usize, 2, 3];
    let right = [2usize, 3, 4];
    let mut tri_energies = [0.0f32; 3];
    let status =
        generalized_triangular_filterbank(&fft_mag, &left, &center, &right, &mut tri_energies);
    assert_eq!(status, Status::Success);

    let q15_val = fast_log2_q15(q15::from_bits(4096));
    assert!(q15_val.to_bits() != 0);

    let vad = VadDetectorQ15::new(100, 2);
    let q15_frame = [q15::from_bits(1000); 16];
    let _ = vad.is_active(&q15_frame);
}

#[test]
fn test_beamforming_exhaustive() {
    let mut bf = DelayAndSumBeamformer::<4, 64>::new();
    bf.set_delays(&[0.0, 1.0, 2.0, 3.0]);
    bf.set_weights(&[0.25, 0.25, 0.25, 0.25]);
    let mic_sample = [1.0, 1.0, 1.0, 1.0];
    let out = bf.process_sample(&mic_sample);
    assert!(out.is_finite());
    bf.reset();

    let sig_a = [1.0f32; 64];
    let sig_b = [1.0f32; 64];
    let result = gcc_phat_tdoa_f32(&sig_a, &sig_b, 10);
    assert!(result.is_ok());
}

#[test]
fn test_controller_exhaustive() {
    let mut inst_f32 = PidInstance::<f32>::new(1.0, 0.1, 0.01);
    assert!(inst_f32.process(1.0).is_finite());
    assert!(PidInstance::process(&mut inst_f32, 1.0).is_finite());
    inst_f32.reset();

    let mut inst_q31 = PidInstance::<q31>::new(
        q31::from_bits(1_000_000_000),
        q31::from_bits(100_000_000),
        q31::from_bits(10_000_000),
    );
    let _res_q31_1 = inst_q31.process(q31::from_bits(1_000_000_000));
    let _res_q31_2 = PidInstance::process(&mut inst_q31, q31::from_bits(1_000_000_000));
    inst_q31.reset();

    let mut inst_q15 = PidInstance::<q15>::new(
        q15::from_bits(10000),
        q15::from_bits(1000),
        q15::from_bits(100),
    );
    let _res_q15_1 = inst_q15.process(q15::from_bits(10000));
    let _res_q15_2 = PidInstance::process(&mut inst_q15, q15::from_bits(10000));
    inst_q15.reset();

    let (mut alpha, mut beta) = (0.0f32, 0.0f32);
    clarke_f32(1.0, 0.0, &mut alpha, &mut beta);
    let (mut d, mut q) = (0.0f32, 0.0f32);
    park_f32(alpha, beta, 0.5, &mut d, &mut q);
    let (mut ia, mut ib) = (0.0f32, 0.0f32);
    inv_park_f32(d, q, 0.5, &mut alpha, &mut beta);
    inv_clarke_f32(alpha, beta, &mut ia, &mut ib);
    assert!(ia.is_finite() && ib.is_finite());

    let (mut alpha_q, mut beta_q) = (q15::ZERO, q15::ZERO);
    clarke_q15(q15::from_bits(1000), q15::ZERO, &mut alpha_q, &mut beta_q);
    let (mut d_q, mut q_q) = (q15::ZERO, q15::ZERO);
    park_q15(
        alpha_q,
        beta_q,
        q15::from_bits(500),
        q15::from_bits(1000),
        &mut d_q,
        &mut q_q,
    );
    inv_park_q15(
        d_q,
        q_q,
        q15::from_bits(500),
        q15::from_bits(1000),
        &mut alpha_q,
        &mut beta_q,
    );
    let (mut ia_q, mut ib_q) = (q15::ZERO, q15::ZERO);
    inv_clarke_q15(alpha_q, beta_q, &mut ia_q, &mut ib_q);
}

#[test]
fn test_intrinsics_lut_pll_psd() {
    let _d1 = intrinsics::dual_mac_q15(0x00010002, 0x00030004, 0);
    let _d2 = intrinsics::dual_mac_q63(0x00010002, 0x00030004, 0);
    let _a1 = intrinsics::dual_saturating_add_q15(0x00010002, 0x00030004);
    let _s1 = intrinsics::dual_saturating_sub_q15(0x00030004, 0x00010002);
    let _sq = intrinsics::saturate_q15(40000);
    let _sq31 = intrinsics::saturate_q31(3000000000);

    let src_a = [q15::from_bits(100); 4];
    let src_b = [q15::from_bits(200); 4];
    let mut dst = [q15::ZERO; 4];
    let _dot = intrinsics::simd_dot_prod_q15(&src_a, &src_b);
    intrinsics::simd_add_q15(&src_a, &src_b, &mut dst);
    intrinsics::simd_sub_q15(&src_a, &src_b, &mut dst);
    intrinsics::simd_mult_q15(&src_a, &src_b, &mut dst);

    assert_ne!(lut::fast_sin_i16(1.0), 0);
    assert_ne!(lut::fast_cos_i16(1.0), 0);
    assert_ne!(lut::sin_q16(10000), 0);
    assert_ne!(lut::cos_q16(10000), 0);

    let mut pll = SogiPll::new(50.0, 1000.0, 1.414, 60.0, 1400.0);
    assert!(pll.process(1.0).is_finite());
    assert!(pll.frequency_hz().is_finite());
    assert!(pll.phase().is_finite());
    let _ortho = pll.orthogonal_components();
    pll.reset();

    let mut costas = CostasLoop::new(1000.0, 10000.0, 10.0, 0.707);
    let (i_out, q_out) = costas.process_sample(1.0);
    assert!(i_out.is_finite() && q_out.is_finite());
    assert!(costas.frequency_hz().is_finite());
    assert!(costas.center_frequency_hz().is_finite());

    let psd_data = [1.0f32; 128];
    let mut psd_out = [0.0f32; 32];
    assert_eq!(
        welch_psd_f32(
            &psd_data,
            &mut psd_out,
            64,
            32,
            1000.0,
            WelchWindow::Hamming,
            true
        ),
        Status::Success
    );

    let mut pgram_out = [0.0f32; 32];
    assert_eq!(
        periodogram_f32(
            &psd_data[..64],
            &mut pgram_out,
            64,
            1000.0,
            WelchWindow::Rectangular,
            false
        ),
        Status::Success
    );

    let mut ar_coeffs = [0.0f32; 4];
    if let Ok(noise_var) = ar_burg_f32(&psd_data[..32], 4, &mut ar_coeffs) {
        let _ = ar_psd_f32(&ar_coeffs, noise_var, 32, &mut psd_out, false);
    }
}

// ─── from push_to_95_coverage_b.rs ────────────────────────────────────────
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
    let _: <f32 as DspSample>::Accum = 0.0f32;
    let _: <f32 as DspSample>::Coeff = 0.0f32;
    assert_eq!(<f32 as DspSample>::madd(0.5, 2.0, 3.0), 6.5);
    assert_eq!(<f32 as DspSample>::from_accum(6.5), 6.5);
    assert_eq!(<f32 as DspSample>::coeff_from_f32(0.25), 0.25);

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
    let _: <f64 as DspSample>::Accum = 0.0f64;
    let _: <f64 as DspSample>::Coeff = 0.0f64;
    assert_eq!(<f64 as DspSample>::madd(0.5, 2.0, 3.0), 6.5);
    assert_eq!(<f64 as DspSample>::from_accum(6.5), 6.5);
    assert_eq!(<f64 as DspSample>::coeff_from_f32(0.25), 0.25);

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

    // The Q15 accumulator is wide enough to hold several Q30 products, then narrows once.
    let _: <q15 as DspSample>::Accum = 0i64;
    let _: <q15 as DspSample>::Coeff = q15::ZERO;
    assert_eq!(
        <q15 as DspSample>::madd(0, q15::from_bits(1000), q15::from_bits(2000)),
        1000i64 * 2000
    );
    let wide_q15 = <q15 as DspSample>::madd(
        <q15 as DspSample>::madd(
            <q15 as DspSample>::madd(0, q15::MAX, q15::MAX),
            q15::MAX,
            q15::MAX,
        ),
        q15::MAX,
        q15::MAX,
    );
    assert_eq!(wide_q15, 3 * (32767i64 * 32767));
    assert_eq!(<q15 as DspSample>::from_accum(wide_q15), q15::MAX);
    assert_eq!(<q15 as DspSample>::from_accum(i64::MIN), q15::MIN);
    assert_eq!(<q15 as DspSample>::from_accum(1 << 15), q15::from_bits(1));
    assert_eq!(
        <q15 as DspSample>::coeff_from_f32(0.5),
        q15::saturating_from_num(0.5)
    );
    assert_eq!(<q15 as DspSample>::coeff_from_f32(2.0), q15::MAX);
    assert_eq!(<q15 as DspSample>::coeff_from_f32(-2.0), q15::MIN);

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

    let _: <q31 as DspSample>::Accum = 0i64;
    let _: <q31 as DspSample>::Coeff = q31::ZERO;
    assert_eq!(
        <q31 as DspSample>::madd(0, a_q31, a_q31),
        a_q31.to_bits() as i64 * a_q31.to_bits() as i64
    );
    assert_eq!(<q31 as DspSample>::from_accum(i64::MAX), q31::MAX);
    assert_eq!(<q31 as DspSample>::from_accum(i64::MIN), q31::MIN);
    assert_eq!(<q31 as DspSample>::from_accum(1 << 31), q31::from_bits(1));
    assert_eq!(
        <q31 as DspSample>::coeff_from_f32(0.5),
        q31::saturating_from_num(0.5)
    );
    assert_eq!(<q31 as DspSample>::coeff_from_f32(2.0), q31::MAX);
    assert_eq!(<q31 as DspSample>::coeff_from_f32(-2.0), q31::MIN);

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
fn test_dspsample_stage6_primitives_exhaustive() {
    // `sat_neg`: plain negation for floats (preserving signed zero), saturating for fixed widths.
    assert_eq!(<f32 as DspSample>::sat_neg(1.5), -1.5);
    assert!(<f32 as DspSample>::sat_neg(0.0).is_sign_negative());
    assert_eq!(<f64 as DspSample>::sat_neg(1.5), -1.5);
    assert!(<f64 as DspSample>::sat_neg(0.0).is_sign_negative());
    assert_eq!(
        <q15 as DspSample>::sat_neg(q15::from_bits(i16::MIN)),
        q15::from_bits(i16::MAX)
    );
    assert_eq!(
        <q15 as DspSample>::sat_neg(q15::from_bits(100)),
        q15::from_bits(-100)
    );
    assert_eq!(
        <q31 as DspSample>::sat_neg(q31::from_bits(i32::MIN)),
        q31::from_bits(i32::MAX)
    );
    assert_eq!(
        <q31 as DspSample>::sat_neg(q31::from_bits(100)),
        q31::from_bits(-100)
    );

    // `wrapping_madd`: wrap the fixed-point product at native width before widening; identical to
    // `madd` for floats since there's nothing to wrap.
    assert_eq!(<f32 as DspSample>::wrapping_madd(0.5, 2.0, 3.0), 6.5);
    assert_eq!(<f64 as DspSample>::wrapping_madd(0.5, 2.0, 3.0), 6.5);
    assert_eq!(
        <q15 as DspSample>::wrapping_madd(0, q15::from_bits(i16::MIN), q15::from_bits(i16::MIN)),
        i16::MIN as i64,
        "MIN*MIN must wrap, not saturate"
    );
    assert_eq!(
        <q31 as DspSample>::wrapping_madd(0, q31::from_bits(i32::MIN), q31::from_bits(i32::MIN)),
        i32::MIN as i64,
        "MIN*MIN must wrap, not saturate"
    );

    // `mul_shifted`: `mul_high` generalized to an explicit shift instead of the fixed `FRAC`.
    assert_eq!(<f32 as DspSample>::mul_shifted(2.0, 3.0, 5), 6.0);
    assert_eq!(<f64 as DspSample>::mul_shifted(2.0, 3.0, 5), 6.0);
    assert_eq!(
        <q15 as DspSample>::mul_shifted(q15::from_bits(1000), q15::from_bits(2000), 17),
        (1000i64 * 2000) >> 17
    );
    assert_eq!(
        <q31 as DspSample>::mul_shifted(q31::from_bits(1000), q31::from_bits(2000), 33),
        (1000i64 * 2000) >> 33
    );

    // `accum_shift`: shift a value already in the accumulator domain, staying there.
    assert_eq!(<f32 as DspSample>::accum_shift(6.5, 3), 6.5);
    assert_eq!(<f64 as DspSample>::accum_shift(6.5, 3), 6.5);
    assert_eq!(<q15 as DspSample>::accum_shift(1000i64, 3), 1000i64 >> 3);
    assert_eq!(<q31 as DspSample>::accum_shift(1000i64, 3), 1000i64 >> 3);

    // `coeff_from_q15_bits`: promote a shared Q15-precision twiddle-table entry to this sample's
    // native coefficient width.
    assert_eq!(<f32 as DspSample>::coeff_from_q15_bits(16384), 0.5);
    assert_eq!(<f64 as DspSample>::coeff_from_q15_bits(16384), 0.5);
    assert_eq!(
        <q15 as DspSample>::coeff_from_q15_bits(1000),
        q15::from_bits(1000)
    );
    assert_eq!(
        <q31 as DspSample>::coeff_from_q15_bits(1000),
        q31::from_bits(1000 << 16)
    );

    // `f64`'s remaining `DspSample` methods (pre-existing since Stage 3, but never directly
    // exercised anywhere else in the suite).
    assert_eq!(<f64 as DspSample>::average_accum(6.0, 3), 2.0);
    assert_eq!(<f64 as DspSample>::mul_high(2.0, 3.0), 6.0);
    assert_eq!(<f64 as DspSample>::from_accum_shifted(6.5, 3), 6.5);
    assert_eq!(<f64 as DspSample>::accum_from_shifted(6.5, 3), 6.5);
    assert_eq!(<f64 as DspSample>::abs_val(-5.0), 5.0);
    assert_eq!(<f64 as DspSample>::abs_val(5.0), 5.0);
}

#[test]
fn test_pid_instance_derived_trait_impls() {
    // `PidInstance<T>`'s manual Clone/Copy/Debug/PartialEq/Default (the derive macros can't add
    // bounds on associated types like `T::Coeff`, so these are hand-written).
    let a = PidInstance::<f32>::new(1.0, 0.1, 0.01);
    let b = a;
    #[allow(clippy::clone_on_copy)]
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert!(format!("{a:?}").contains("PidInstance"));

    let mut d = PidInstance::<f32>::default();
    assert_ne!(a, d);
    d.kp = a.kp;
    d.ki = a.ki;
    d.kd = a.kd;
    d.init(1);
    assert_eq!(a, d);

    let e = PidInstance::<q15>::new(q15::from_bits(100), q15::from_bits(10), q15::from_bits(1));
    let f = e;
    assert_eq!(e, f);
    assert!(format!("{e:?}").contains("PidInstance"));
    assert_eq!(PidInstance::<q15>::default(), PidInstance::<q15>::default());

    let g = PidInstance::<q31>::new(q31::from_bits(100), q31::from_bits(10), q31::from_bits(1));
    let h = g;
    assert_eq!(g, h);
    assert!(format!("{g:?}").contains("PidInstance"));
    assert_eq!(PidInstance::<q31>::default(), PidInstance::<q31>::default());
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
    // `normalize = false` skips the kernel-weight-sum normalization branch.
    assert_eq!(
        convolve2d_f32(&src_img, &mut convolved, 3, 3, &kernel, 3, 3, false),
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
    // Descending image reorders where the extrema land relative to the kernel tap-visitation
    // order, exercising the "found a new min/max after the first tap" branches.
    let src_img_desc = [9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    assert_eq!(
        nonlin2d_filter_f32(
            &src_img_desc,
            &mut nonlin_out,
            3,
            3,
            3,
            NonlinFilterType::Min
        ),
        Status::Success
    );
    assert_eq!(
        nonlin2d_filter_f32(
            &src_img_desc,
            &mut nonlin_out,
            3,
            3,
            3,
            NonlinFilterType::Max
        ),
        Status::Success
    );
    // ArgumentError: even k_size.
    assert_eq!(
        nonlin2d_filter_f32(&src_img, &mut nonlin_out, 3, 3, 2, NonlinFilterType::Min),
        Status::ArgumentError
    );
    // LengthError: declared dims exceed what the buffers hold.
    assert_eq!(
        nonlin2d_filter_f32(
            &src_img[..4],
            &mut nonlin_out,
            3,
            3,
            3,
            NonlinFilterType::Min
        ),
        Status::LengthError
    );

    let mut edges = [0.0f32; 9];
    assert_eq!(
        sobel_edge_detection_f32(&src_img, &mut edges, 3, 3, 2.0),
        Status::Success
    );
    // LengthError: declared dims exceed what the buffers hold.
    assert_eq!(
        sobel_edge_detection_f32(&src_img[..4], &mut edges, 3, 3, 2.0),
        Status::LengthError
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

// ─── from push_to_95_coverage_c.rs ────────────────────────────────────────
#[test]
fn test_transforms_inverse_and_flags() {
    let mut data_q31 = [q31::from_bits(10000); 16];
    cfft_q31(&mut data_q31, 8, 1, 0);
    cfft_q31(&mut data_q31, 8, 1, 1);
    cfft_q31(&mut data_q31, 8, 0, 0);

    let mut data_q15 = [q15::from_bits(1000); 16];
    cfft_q15(&mut data_q15, 8, 1, 0);
    cfft_q15(&mut data_q15, 8, 1, 1);
    cfft_q15(&mut data_q15, 8, 0, 0);

    let mut bfp_q31 = [q31::from_bits(10000); 16];
    let _s_q31_1 = cfft_bfp_q31(&mut bfp_q31, 8, 1, 0);
    let _s_q31_2 = cfft_bfp_q31(&mut bfp_q31, 8, 1, 1);

    let mut bfp_q15 = [q15::from_bits(1000); 16];
    let _s_q15_1 = cfft_bfp_q15(&mut bfp_q15, 8, 1, 0);
    let _s_q15_2 = cfft_bfp_q15(&mut bfp_q15, 8, 1, 1);

    let src_q31 = [q31::from_bits(5000); 8];
    let mut dst_q31_a = [q31::ZERO; 16];
    let mut dst_q31_b = [q31::ZERO; 8];
    rfft_q31(&src_q31, &mut dst_q31_a, 8, 0); // packed_rfft_q31_forward
    irfft_q31(&dst_q31_a, &mut dst_q31_b, 8); // packed_irfft_q31
    rfft_q31(&src_q31, &mut dst_q31_a, 8, 1); // fallback unpack branch

    let src_q15 = [q15::from_bits(500); 8];
    let mut dst_q15_a = [q15::ZERO; 16];
    let mut dst_q15_b = [q15::ZERO; 8];
    rfft_q15(&src_q15, &mut dst_q15_a, 8, 0); // packed_rfft_q15_forward
    irfft_q15(&dst_q15_a, &mut dst_q15_b, 8); // packed_irfft_q15
    rfft_q15(&src_q15, &mut dst_q15_a, 8, 1); // fallback unpack branch

    let mut wht_f32 = [1.0f32; 8];
    assert_eq!(ifwht_f32(&mut wht_f32), Status::Success);

    let mut wht_i32 = [10i32; 8];
    assert_eq!(fwht_i32(&mut wht_i32), Status::Success);

    let mut haar_f32 = [1.0f32; 8];
    assert_eq!(haar_transform_f32(&mut haar_f32), Status::Success);
    assert_eq!(inverse_haar_transform_f32(&mut haar_f32), Status::Success);

    let mut haar_i32 = [10i32; 8];
    assert_eq!(haar_transform_i32(&mut haar_i32), Status::Success);

    let mut hartley = [1.0f32; 8];
    assert_eq!(hartley_transform_f32(&mut hartley), Status::Success);

    let db4_h = [0.482_962_9, 0.836_516_3, 0.224_143_86, -0.129_409_52];
    let mut wav_data = [1.0f32; 8];
    assert_eq!(wavelet_step_f32(&mut wav_data, 8, &db4_h), Status::Success);
    assert_eq!(
        inverse_wavelet_step_f32(&mut wav_data, 8, &db4_h),
        Status::Success
    );
    assert_eq!(
        wavelet_transform_f32(&mut wav_data, &db4_h),
        Status::Success
    );
    assert_eq!(
        inverse_wavelet_transform_f32(&mut wav_data, &db4_h),
        Status::Success
    );
}

#[test]
fn test_transforms_error_branches() {
    let mut bad_buf = [0.0f32; 3];
    assert_ne!(fwht_f32(&mut bad_buf), Status::Success);
    assert_ne!(ifwht_f32(&mut bad_buf), Status::Success);
    assert_ne!(haar_transform_f32(&mut bad_buf), Status::Success);
    assert_ne!(inverse_haar_transform_f32(&mut bad_buf), Status::Success);
    assert_ne!(hartley_transform_f32(&mut bad_buf), Status::Success);

    let mut bad_buf_i32 = [0i32; 3];
    assert_ne!(fwht_i32(&mut bad_buf_i32), Status::Success);
    assert_ne!(haar_transform_i32(&mut bad_buf_i32), Status::Success);

    let db4_h = [0.482_962_9, 0.836_516_3, 0.224_143_86, -0.129_409_52];
    assert_ne!(wavelet_step_f32(&mut bad_buf, 3, &db4_h), Status::Success);
    assert_ne!(
        inverse_wavelet_step_f32(&mut bad_buf, 3, &db4_h),
        Status::Success
    );
    assert_ne!(wavelet_transform_f32(&mut bad_buf, &db4_h), Status::Success);
    assert_ne!(
        inverse_wavelet_transform_f32(&mut bad_buf, &db4_h),
        Status::Success
    );

    let mut cep_out = [0.0f32; 2];
    assert_ne!(real_cepstrum_f32(&bad_buf, &mut cep_out), Status::Success);

    let src_short = [q31::ZERO; 2];
    let mut dst_short = [q31::ZERO; 2];
    irfft_q31(&src_short, &mut dst_short, 8);
    let src_short_q15 = [q15::ZERO; 2];
    let mut dst_short_q15 = [q15::ZERO; 2];
    irfft_q15(&src_short_q15, &mut dst_short_q15, 8);
}

#[test]
fn test_types_fixed_and_enums() {
    let q = q15::from_bits(1000);
    assert_eq!(q15::ZERO.to_bits(), 0);
    assert_eq!(q15::MIN.to_bits(), i16::MIN);
    assert_eq!(q15::MAX.to_bits(), i16::MAX);
    assert_eq!(q.to_bits(), 1000);

    let _q_sat_add = q.saturating_add(q15::from_bits(500));
    let _q_sat_sub = q.saturating_sub(q15::from_bits(500));
    let _q_sat_mul = q.saturating_mul(q15::from_bits(500));
    let _q_sat_neg = q.saturating_neg();
    let _q_sat_abs = q.saturating_abs();
    let _q_abs = q.abs();
    let _q_wrap_add = q.wrapping_add(q15::from_bits(500));
    let _q_wrap_sub = q.wrapping_sub(q15::from_bits(500));
    let _q_wrap_neg = q.wrapping_neg();
    let _q_wrap_mul = q.wrapping_mul(q15::from_bits(500));
    let _q_wrap_mul_int = q.wrapping_mul_int(2);
    let _q_chk_div = q.checked_div(q15::from_bits(500));
    let _q_chk_div_zero = q.checked_div(q15::ZERO);

    // Status enum variants
    assert_eq!(Status::Success as i8, 0);
    assert_eq!(Status::ArgumentError as i8, -1);
    assert_eq!(Status::LengthError as i8, -2);
    assert_eq!(Status::SizeMismatch as i8, -3);
    assert_eq!(Status::NanInf as i8, -4);
    assert_eq!(Status::Singular as i8, -5);
    assert_eq!(Status::TestFailure as i8, -6);
    assert_eq!(Status::DecompositionFailure as i8, -7);
}

#[test]
fn test_windows_and_statistics_extra() {
    let mut win_buf = [0.0f32; 16];
    hanning_f32(&mut win_buf);
    hamming_f32(&mut win_buf);
    blackman_f32(&mut win_buf);
    blackman_harris_f32(&mut win_buf);
    bartlett_f32(&mut win_buf);
    welch_f32(&mut win_buf);
    flattop_f32(&mut win_buf);
    kaiser_f32(&mut win_buf, 5.0);
    apply_window_f32(&mut win_buf, &[1.0f32; 16]);

    let mut q15_win = [q15::ZERO; 16];
    hanning_q15(&mut q15_win);
    hamming_q15(&mut q15_win);
    blackman_q15(&mut q15_win);
    bartlett_q15(&mut q15_win);
    apply_window_q15(&mut q15_win, &[q15::from_bits(1000); 16]);

    let src_q7 = [q7::from_bits(10); 8];
    let mut q7_res = q7::ZERO;
    let mut idx = 0usize;
    assert_eq!(mean_q7(&src_q7, &mut q7_res), Status::Success);
    assert_eq!(var_q7(&src_q7, &mut q7_res), Status::Success);
    assert_eq!(std_q7(&src_q7, &mut q7_res), Status::Success);
    assert_eq!(min_q7(&src_q7, &mut q7_res, &mut idx), Status::Success);
    assert_eq!(max_q7(&src_q7, &mut q7_res, &mut idx), Status::Success);

    let data = [0.1f32, 0.2, 0.3, 0.4];
    assert!(entropy_f32(&data).is_finite());
    assert!(kullback_leibler_f32(&data, &data).is_finite());
    assert!(logsumexp_f32(&data).is_finite());

    let mut f_res = 0.0f32;
    assert_eq!(absmax_f32(&data, &mut f_res, &mut idx), Status::Success);
    assert_eq!(absmin_f32(&data, &mut f_res, &mut idx), Status::Success);
}

#[test]
fn test_matrix_extra() {
    let data_a = [1.0f32, 2.0, 3.0, 4.0];
    let data_b = [5.0f32, 6.0, 7.0, 8.0];
    let mut data_out = [0.0f32; 4];

    let mat_a = MatrixInstance::new(2, 2, &data_a);
    let mat_b = MatrixInstance::new(2, 2, &data_b);
    let mut mat_out = MatrixInstanceMut::new(2, 2, &mut data_out);

    assert_eq!(mat_add_f32(&mat_a, &mat_b, &mut mat_out), Status::Success);
    assert_eq!(mat_sub_f32(&mat_a, &mat_b, &mut mat_out), Status::Success);
    assert_eq!(mat_scale_f32(&mat_a, 2.0, &mut mat_out), Status::Success);
    assert_eq!(mat_mult_f32(&mat_a, &mat_b, &mut mat_out), Status::Success);
    assert_eq!(mat_trans_f32(&mat_a, &mut mat_out), Status::Success);
    assert_eq!(mat_inverse_f32(&mat_a, &mut mat_out), Status::Success);
}

// ─── from push_to_95_coverage_d.rs ────────────────────────────────────────
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

    let mut pid_f32 = PidInstance::<f32>::new(1.0, 0.1, 0.01);
    assert!(pid_f32.process_sample(1.0).is_finite());

    let mut pid_q15 = PidInstance::<q15>::new(
        q15::from_bits(1000),
        q15::from_bits(100),
        q15::from_bits(10),
    );
    let _p_q15 = pid_q15.process_sample(q15::from_bits(500));

    let mut filter_f32 = SinglePoleFilter::<f32>::lowpass(0.1);
    assert!(filter_f32.process_sample(1.0).is_finite());

    let mut filter_q15 = SinglePoleFilter::<q15>::lowpass(q15::from_bits(3000));
    let _f_q15 = filter_q15.process_sample(q15::from_bits(1000));

    let mut dc_blocker = DcBlockerQ15::new(q15::from_bits(32000));
    let _dc_out = dc_blocker.process_sample(q15::from_bits(1000));
}

// ─── from push_to_95_coverage_e.rs ────────────────────────────────────────
#[test]
fn test_square_root_kalman_filter_exhaustive() {
    let x0 = [0.0f32, 0.0];
    let s0 = [[1.0f32, 0.0], [0.0, 1.0]];
    let f = [[1.0f32, 1.0], [0.0, 1.0]];
    let s_q = [[0.1f32, 0.0], [0.0, 0.1]];
    let h = [[1.0f32, 0.0]];
    let s_r = [[0.5f32]];

    let mut sr_kf = SquareRootKalmanFilter::<2, 1>::new(x0, s0, f, s_q, h, s_r);
    sr_kf.predict();
    let status = sr_kf.update(&[1.0f32]);
    assert_eq!(status, Status::Success);

    let cov = sr_kf.covariance();
    assert!(cov[0][0] > 0.0);
}

#[test]
fn test_psd_error_branches_and_db() {
    let src = [1.0f32; 128];
    let mut dst = [0.0f32; 32];

    // Invalid arguments for welch_psd_f32
    assert_eq!(
        welch_psd_f32(
            &src,
            &mut dst,
            3,
            0,
            1000.0,
            WelchWindow::Rectangular,
            false
        ),
        Status::ArgumentError
    );
    assert_eq!(
        welch_psd_f32(
            &src,
            &mut dst,
            64,
            64,
            1000.0,
            WelchWindow::Rectangular,
            false
        ),
        Status::ArgumentError
    );
    assert_eq!(
        welch_psd_f32(
            &src,
            &mut dst,
            64,
            16,
            -100.0,
            WelchWindow::Rectangular,
            false
        ),
        Status::ArgumentError
    );
    assert_eq!(
        welch_psd_f32(
            &src[..10],
            &mut dst,
            64,
            16,
            1000.0,
            WelchWindow::Rectangular,
            false
        ),
        Status::LengthError
    );

    // ar_burg_f32 invalid arguments
    let mut ar_coeffs = [0.0f32; 4];
    assert_eq!(
        ar_burg_f32(&src[..4], 4, &mut ar_coeffs),
        Err(Status::ArgumentError)
    );
    assert_eq!(
        ar_burg_f32(&src, 0, &mut ar_coeffs),
        Err(Status::ArgumentError)
    );

    // ar_psd_f32 db and linear
    if let Ok(noise_var) = ar_burg_f32(&src[..32], 4, &mut ar_coeffs) {
        assert_eq!(
            ar_psd_f32(&ar_coeffs, noise_var, 32, &mut dst, true),
            Status::Success
        );
        assert_eq!(
            ar_psd_f32(&ar_coeffs, noise_var, 32, &mut dst, false),
            Status::Success
        );
    }
}

#[test]
fn test_quantization_and_scaling_strategies() {
    let sos_f32 = [0.1f32, 0.2, 0.3, 0.4, 0.5];
    let mut q15_out = [q15::ZERO; 5];
    let mut q31_out = [q31::ZERO; 5];

    assert!(
        biquad_quantize_and_scale_q15(&sos_f32, &mut q15_out, ScalingStrategy::LInfNorm).is_ok()
    );
    assert!(biquad_quantize_and_scale_q15(&sos_f32, &mut q15_out, ScalingStrategy::L2Norm).is_ok());
    assert!(biquad_quantize_and_scale_q15(&sos_f32, &mut q15_out, ScalingStrategy::Direct).is_ok());

    assert!(
        biquad_quantize_and_scale_q31(&sos_f32, &mut q31_out, ScalingStrategy::LInfNorm).is_ok()
    );
    assert!(biquad_quantize_and_scale_q31(&sos_f32, &mut q31_out, ScalingStrategy::L2Norm).is_ok());
    assert!(biquad_quantize_and_scale_q31(&sos_f32, &mut q31_out, ScalingStrategy::Direct).is_ok());

    let taps_f32 = [0.1f32, 0.2, 0.3, 0.4];
    let mut taps_q15 = [q15::ZERO; 4];
    assert!(fir_quantize_q15(&taps_f32, &mut taps_q15).is_ok());

    // Error length tests
    let mut bad_q15 = [q15::ZERO; 4];
    assert_eq!(
        biquad_quantize_and_scale_q15(&sos_f32, &mut bad_q15, ScalingStrategy::Direct),
        Err(Status::LengthError)
    );
    assert_eq!(
        fir_quantize_q15(&taps_f32, &mut bad_q15[..2]),
        Err(Status::LengthError)
    );

    // More than 26 biquad stages (130 elements) overflows the internal 128-element scratch
    // buffer, an ArgumentError distinct from the length-mismatch case above.
    let big_sos = [0.1f32; 130];
    let mut big_q15 = [q15::ZERO; 130];
    let mut big_q31 = [q31::ZERO; 130];
    assert_eq!(
        biquad_quantize_and_scale_q15(&big_sos, &mut big_q15, ScalingStrategy::Direct),
        Err(Status::ArgumentError)
    );
    assert_eq!(
        biquad_quantize_and_scale_q31(&big_sos, &mut big_q31, ScalingStrategy::Direct),
        Err(Status::ArgumentError)
    );
}

// ─── from coverage_boost_tests.rs ────────────────────────────────────────
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

    let q1 = [
        core::f32::consts::FRAC_1_SQRT_2,
        core::f32::consts::FRAC_1_SQRT_2,
        0.0,
        0.0,
    ];
    let q2 = [
        core::f32::consts::FRAC_1_SQRT_2,
        0.0,
        core::f32::consts::FRAC_1_SQRT_2,
        0.0,
    ];
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
    assert!((-0.9..=0.9).contains(&limited));
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
    let x: f32 = 0.5;
    assert!(FloatMath::abs(x) > 0.0);
    assert!(FloatMath::sin(x) > 0.0);
    assert!(FloatMath::cos(x) > 0.0);
    assert!(FloatMath::tan(x) > 0.0);
    assert!(FloatMath::sqrt(x) > 0.0);
    assert!(FloatMath::ln(x) < 0.0);
    assert!(FloatMath::log10(x) < 0.0);
    assert!(FloatMath::exp(x) > 1.0);
    assert!(FloatMath::atan2(x, 1.0) > 0.0);
    assert!(FloatMath::powf(x, 2.0) == 0.25);
    assert!(FloatMath::tanh(x) > 0.0);

    let y: f64 = 0.5;
    assert!(FloatMath::abs(y) > 0.0);
    assert!(FloatMath::sin(y) > 0.0);
    assert!(FloatMath::cos(y) > 0.0);
    assert!(FloatMath::tan(y) > 0.0);
    assert!(FloatMath::sqrt(y) > 0.0);
    assert!(FloatMath::ln(y) < 0.0);
    assert!(FloatMath::log10(y) < 0.0);
    assert!(FloatMath::exp(y) > 1.0);
    assert!(FloatMath::atan2(y, 1.0) > 0.0);
    assert!(FloatMath::powf(y, 2.0) == 0.25);
    assert!(FloatMath::tanh(y) > 0.0);
}

// ─── from more_coverage_boost.rs ────────────────────────────────────────
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

// ─── from final_90_plus_coverage_boost.rs ────────────────────────────────────────
#[test]
fn test_complex_math_q31_q15_all() {
    let a_q31 = [q31::from_bits(10000), q31::from_bits(20000)];
    let b_q31 = [q31::from_bits(5000), q31::from_bits(10000)];
    let mut out_q31 = [q31::ZERO; 2];

    cmplx_add_q31(&a_q31, &b_q31, &mut out_q31);
    cmplx_sub_q31(&a_q31, &b_q31, &mut out_q31);
    cmplx_mult_cmplx_q31(&a_q31, &b_q31, &mut out_q31);
    cmplx_mult_real_q31(&a_q31, &b_q31, &mut out_q31);
    cmplx_conj_q31(&a_q31, &mut out_q31);

    let mut mag_q31 = [q31::ZERO; 1];
    cmplx_mag_q31(&a_q31, &mut mag_q31);
    cmplx_mag_squared_q31(&a_q31, &mut mag_q31);
    let _dot_q31 = cmplx_dot_prod_q31(&a_q31, &b_q31);

    let a_q15 = [q15::from_bits(1000), q15::from_bits(2000)];
    let b_q15 = [q15::from_bits(500), q15::from_bits(1000)];
    let mut out_q15 = [q15::ZERO; 2];

    cmplx_add_q15(&a_q15, &b_q15, &mut out_q15);
    cmplx_sub_q15(&a_q15, &b_q15, &mut out_q15);
    cmplx_mult_cmplx_q15(&a_q15, &b_q15, &mut out_q15);
    cmplx_mult_real_q15(&a_q15, &b_q15, &mut out_q15);
    cmplx_conj_q15(&a_q15, &mut out_q15);

    let mut mag_q15 = [q15::ZERO; 1];
    cmplx_mag_q15(&a_q15, &mut mag_q15);
    cmplx_mag_squared_q15(&a_q15, &mut mag_q15);
    let _dot_q15 = cmplx_dot_prod_q15(&a_q15, &b_q15);
}

#[test]
fn test_transforms_q31_q15_and_wavelets() {
    let mut q31_buf = [q31::from_bits(1000); 32];
    let mut out_q31 = [q31::ZERO; 32];
    cfft_q31(&mut q31_buf[..32], 16, 0, 1);
    cfft_q31(&mut q31_buf[..32], 16, 1, 1);
    let _scale_q31 = cfft_bfp_q31(&mut q31_buf[..32], 16, 0, 1);
    rfft_q31(&q31_buf[..16], &mut out_q31[..32], 16, 0);
    rfft_q31(&q31_buf[..16], &mut out_q31[..32], 16, 1);
    irfft_q31(&out_q31[..32], &mut q31_buf[..16], 16);

    let mut q15_buf = [q15::from_bits(100); 32];
    let mut out_q15 = [q15::ZERO; 32];
    cfft_q15(&mut q15_buf[..32], 16, 0, 1);
    cfft_q15(&mut q15_buf[..32], 16, 1, 1);
    let _scale_q15 = cfft_bfp_q15(&mut q15_buf[..32], 16, 0, 1);
    rfft_q15(&q15_buf[..16], &mut out_q15[..32], 16, 0);
    rfft_q15(&q15_buf[..16], &mut out_q15[..32], 16, 1);
    irfft_q15(&out_q15[..32], &mut q15_buf[..16], 16);

    let f32_buf = [1.0f32; 16];
    let mut out_f32 = [0.0f32; 32];
    rfft_f32(&f32_buf, &mut out_f32, 16, 0);
    rfft_f32(&f32_buf, &mut out_f32, 16, 1);

    let mut cep_out = [0.0f32; 16];
    assert_eq!(real_cepstrum_f32(&f32_buf, &mut cep_out), Status::Success);

    // FWHT and Haar
    let mut fwht_buf = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(fwht_f32(&mut fwht_buf), Status::Success);
    assert_eq!(ifwht_f32(&mut fwht_buf), Status::Success);

    let mut fwht_i32_buf = [10i32, 20, 30, 40];
    assert_eq!(fwht_i32(&mut fwht_i32_buf), Status::Success);

    let mut haar_i32_buf = [10i32, 20, 30, 40];
    assert_eq!(haar_transform_i32(&mut haar_i32_buf), Status::Success);

    let mut haar_f32_buf = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(
        inverse_haar_transform_f32(&mut haar_f32_buf),
        Status::Success
    );

    // Wavelets
    let daub4 = [0.482_962_9, 0.836_516_3, 0.224_143_86, -0.129_409_52];
    let mut wave_buf = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(wavelet_step_f32(&mut wave_buf, 4, &daub4), Status::Success);
    assert_eq!(
        inverse_wavelet_step_f32(&mut wave_buf, 4, &daub4),
        Status::Success
    );
    assert_eq!(
        wavelet_transform_f32(&mut wave_buf, &daub4),
        Status::Success
    );
    assert_eq!(
        inverse_wavelet_transform_f32(&mut wave_buf, &daub4),
        Status::Success
    );
}

#[test]
fn test_filter_design_all_windowed_sinc() {
    let mut hp_taps = [0.0f32; 15];
    assert_eq!(
        fir_windowed_sinc_highpass(0.2, &mut hp_taps),
        Status::Success
    );

    let mut bp_taps = [0.0f32; 15];
    assert_eq!(
        fir_windowed_sinc_bandpass(0.1, 0.3, &mut bp_taps),
        Status::Success
    );

    let mut bs_taps = [0.0f32; 15];
    assert_eq!(
        fir_windowed_sinc_bandstop(0.1, 0.3, &mut bs_taps),
        Status::Success
    );

    // Invalid tap-length/cutoff arguments propagate as ArgumentError, including through
    // highpass/bandstop's delegation into lowpass/bandpass.
    let mut even_taps = [0.0f32; 4];
    assert_eq!(
        fir_windowed_sinc_highpass(0.2, &mut even_taps),
        Status::ArgumentError
    );
    assert_eq!(
        fir_windowed_sinc_bandpass(0.3, 0.1, &mut bp_taps),
        Status::ArgumentError
    );
    assert_eq!(
        fir_windowed_sinc_bandstop(0.3, 0.1, &mut bs_taps),
        Status::ArgumentError
    );
    assert_eq!(
        fir_windowed_sinc_bandstop(0.1, 0.3, &mut even_taps),
        Status::ArgumentError
    );

    let mut biquad_q31 = [q31::ZERO; 5];
    assert!(
        biquad_quantize_and_scale_q31(
            &[1.0, -0.5, 0.25, 0.1, -0.05],
            &mut biquad_q31,
            ScalingStrategy::Direct
        )
        .is_ok()
    );

    let mut biquad_q15 = [q15::ZERO; 5];
    assert!(
        biquad_quantize_and_scale_q15(
            &[1.0, -0.5, 0.25, 0.1, -0.05],
            &mut biquad_q15,
            ScalingStrategy::Direct
        )
        .is_ok()
    );

    let mut fir_q15_taps = [q15::ZERO; 15];
    assert!(fir_quantize_q15(&hp_taps, &mut fir_q15_taps).is_ok());

    assert_eq!(
        single_pole_decay_from_time_constant(10.0),
        (-1.0f32 / 10.0).exp()
    );
    assert_eq!(
        single_pole_decay_from_cutoff(0.1),
        (-2.0f32 * core::f32::consts::PI * 0.1).exp()
    );

    assert!(biquad_lowpass_coeffs(1000.0, 48000.0, 0.707)[0].is_finite());
    assert!(biquad_highpass_coeffs(1000.0, 48000.0, 0.707)[0].is_finite());
    assert!(biquad_bandpass_coeffs(1000.0, 48000.0, 0.707)[0].is_finite());
    assert!(biquad_notch_coeffs(1000.0, 48000.0, 0.707)[0].is_finite());
    assert!(biquad_peaking_coeffs(1000.0, 48000.0, 0.707, 3.0)[0].is_finite());
    assert!(biquad_allpass_coeffs(1000.0, 48000.0, 0.707)[0].is_finite());

    let mut biquads_bw = [0.0f32; 10];
    butterworth_lowpass_biquads(1000.0, 48000.0, 4, &mut biquads_bw);

    let mut biquads_cheb = [0.0f32; 10];
    chebyshev_lowpass_biquads(0.1, 0.5, 4, &mut biquads_cheb);
    chebyshev_highpass_biquads(0.1, 0.5, 4, &mut biquads_cheb);

    let pw = prewarp_cutoff_f32(1000.0, 48000.0);
    let biquad_out = bilinear_transform_biquad(pw, 0.0, 0.0, 1.0, 0.0, 0.0, 48000.0);
    assert_eq!(biquad_out.len(), 5);
}

#[test]
fn test_kalman_all_methods() {
    let mut kf = KalmanFilter::<2, 1>::from_variances([0.0, 0.0], 1.0, 0.01, 0.1);
    let f = [[1.0, 1.0], [0.0, 1.0]];
    kf.predict(&f);
    let b = [[0.1], [0.05]];
    let u = [1.0];
    kf.predict_with_control(&f, &b, &u);
    assert_eq!(kf.update(&[[1.0, 0.0]], &[1.0]), Status::Success);

    let mut sr_kf = SquareRootKalmanFilter::<2, 1>::new(
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.1, 0.0], [0.0, 0.1]],
        [[1.0, 0.0]],
        [[0.1]],
    );
    sr_kf.predict();
    assert_eq!(sr_kf.update(&[1.0]), Status::Success);
    assert_eq!(sr_kf.covariance().len(), 2);
}

#[test]
fn test_math_f64_full_coverage() {
    let f: f64 = 0.5;
    assert!((FloatMath::abs(f) - 0.5).abs() < 1e-6);
    assert!(FloatMath::sin(f) > 0.0);
    assert!(FloatMath::cos(f) > 0.0);
    assert!(FloatMath::tan(f) > 0.0);
    assert!(FloatMath::sqrt(f) > 0.0);
    assert!(FloatMath::ln(f) < 0.0);
    assert!(FloatMath::log10(f) < 0.0);
    assert!(FloatMath::exp(f) > 1.0);
    assert!(FloatMath::atan2(f, 1.0) > 0.0);
    assert!((FloatMath::powf(f, 2.0) - 0.25).abs() < 1e-6);
    assert!(FloatMath::tanh(f) > 0.0);

    assert_eq!(isqrt_u32(0), 0);
    assert_eq!(isqrt_u32(1), 1);
    assert_eq!(isqrt_u32(16), 4);
    assert_eq!(isqrt_u32(100), 10);

    assert_eq!(isqrt_u64(0), 0);
    assert_eq!(isqrt_u64(1), 1);
    assert_eq!(isqrt_u64(144), 12);
}
