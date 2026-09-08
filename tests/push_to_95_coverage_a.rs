use embedded_dsp::*;

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
    let mut inst_f32 = PidInstanceF32::new(1.0, 0.1, 0.01);
    assert!(inst_f32.process(1.0).is_finite());
    assert!(pid_f32(&mut inst_f32, 1.0).is_finite());
    inst_f32.reset();

    let mut inst_q31 = PidInstanceQ31::new(
        q31::from_bits(1_000_000_000),
        q31::from_bits(100_000_000),
        q31::from_bits(10_000_000),
    );
    let _res_q31_1 = inst_q31.process(q31::from_bits(1_000_000_000));
    let _res_q31_2 = pid_q31(&mut inst_q31, q31::from_bits(1_000_000_000));
    inst_q31.reset();

    let mut inst_q15 = PidInstanceQ15::new(
        q15::from_bits(10000),
        q15::from_bits(1000),
        q15::from_bits(100),
    );
    let _res_q15_1 = inst_q15.process(q15::from_bits(10000));
    let _res_q15_2 = pid_q15(&mut inst_q15, q15::from_bits(10000));
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
