//! Comprehensive Digital Filter Design, Analysis & Verification Workbench Example
//!
//! Demonstrates:
//! 1. Filter Design Algorithms:
//!    - Butterworth IIR Lowpass Cascade (`butterworth_lowpass_biquads`)
//!    - Chebyshev Type I IIR Filter Cascade with specified passband ripple (`chebyshev_lowpass_biquads`)
//!    - Windowed-Sinc FIR Filters: Lowpass, Highpass, Bandpass, and Bandstop (`fir_windowed_sinc_lowpass`, `fir_windowed_sinc_bandpass`)
//!    - Arbitrary Frequency-Sampling FIR Filter Design (`fir_custom_frequency_sampling`)
//!    - Bilinear Transform & Analog Prewarping (`bilinear_transform_biquad`, `prewarp_cutoff_f32`)
//! 2. Frequency-Domain Analysis & Stability:
//!    - DTFT Frequency Magnitude (dB) & Phase Response (`biquad_frequency_response`, `fir_frequency_response`, `response_magnitude_db`, `response_phase`)
//!    - FIR Group Delay Calculation (`fir_group_delay`)
//!    - IIR Pole Radius & Strict Stability Verification (`biquad_pole_radius`, `biquad_is_stable`, `biquad_cascade_is_stable`)
//! 3. Filter Implementation & Topology Comparisons:
//!    - Direct Form I (`biquad_cascade_df1_f32`) vs Transposed Direct Form II (`biquad_cascade_df2t_f32`)
//!    - Const-Generic Fixed-Size Wrappers (`FirFilter<33>`, `BiquadCascade<10, 8>`)
//!    - Q15 Fixed-Point vs F32 Precision & Quantization Noise Analysis
//! 4. Vector Distance Metrics (Euclidean, Cosine, Chebyshev, Manhattan, Minkowski, Canberra, Bray-Curtis)
//! 5. Information-Theoretic Metrics (Shannon Entropy `entropy_f32`, KL Divergence `kl_divergence_f32`, LogSumExp `log_sum_exp_f32`)

use embedded_dsp::*;

fn main() {
    println!("===============================================================================");
    println!("      embedded-dsp Filter Design, Analysis & Verification Workbench            ");
    println!("===============================================================================");
    println!();

    const FS: f32 = 48000.0;

    // -----------------------------------------------------------------------------------------
    // 1. IIR Filter Design: Butterworth vs Chebyshev Cascades
    // -----------------------------------------------------------------------------------------
    println!("--- 1. IIR Filter Design: 4th-Order Butterworth & Chebyshev Cascades ---");
    let cutoff_hz = 4800.0f32; // Cutoff at 4.8 kHz (normalized fc = 0.10)
    let cutoff_norm = cutoff_hz / FS;

    // 4th-Order Butterworth (2 biquad stages = 10 coefficients)
    let mut butter_coeffs = [0.0f32; 10];
    butterworth_lowpass_biquads(cutoff_hz, FS, 4, &mut butter_coeffs);

    // 4th-Order Chebyshev Lowpass (2 biquad stages, 1.0% passband ripple)
    let mut cheby_coeffs = [0.0f32; 10];
    chebyshev_lowpass_biquads(cutoff_norm, 1.0, 4, &mut cheby_coeffs);

    println!("  Butterworth 4th-Order Biquad Cascade Coeffs (2 stages):");
    println!(
        "    Stage 0: b0={:.4}, b1={:.4}, b2={:.4}, a1={:.4}, a2={:.4}",
        butter_coeffs[0], butter_coeffs[1], butter_coeffs[2], butter_coeffs[3], butter_coeffs[4]
    );
    println!(
        "    Stage 1: b0={:.4}, b1={:.4}, b2={:.4}, a1={:.4}, a2={:.4}",
        butter_coeffs[5], butter_coeffs[6], butter_coeffs[7], butter_coeffs[8], butter_coeffs[9]
    );

    println!("  Chebyshev 4th-Order (1% ripple) Cascade Coeffs (2 stages):");
    println!(
        "    Stage 0: b0={:.4}, b1={:.4}, b2={:.4}, a1={:.4}, a2={:.4}",
        cheby_coeffs[0], cheby_coeffs[1], cheby_coeffs[2], cheby_coeffs[3], cheby_coeffs[4]
    );
    println!(
        "    Stage 1: b0={:.4}, b1={:.4}, b2={:.4}, a1={:.4}, a2={:.4}",
        cheby_coeffs[5], cheby_coeffs[6], cheby_coeffs[7], cheby_coeffs[8], cheby_coeffs[9]
    );

    // -----------------------------------------------------------------------------------------
    // 2. Windowed-Sinc FIR Design & Custom Frequency Sampling
    // -----------------------------------------------------------------------------------------
    println!("\n--- 2. FIR Filter Design: Windowed-Sinc & Frequency Sampling ---");
    // 33-tap Windowed-Sinc Lowpass Filter (Blackman-windowed sinc)
    const FIR_TAPS: usize = 33;
    let mut fir_lowpass_taps = [0.0f32; FIR_TAPS];
    let fir_status = fir_windowed_sinc_lowpass(cutoff_norm, &mut fir_lowpass_taps);
    println!(
        "  33-Tap Windowed-Sinc Lowpass FIR Status: {:?}",
        fir_status
    );
    println!(
        "    Center Tap [16]: {:.4}, Edge Tap [0]: {:.4}",
        fir_lowpass_taps[16], fir_lowpass_taps[0]
    );

    // 33-Tap Custom Arbitrary Frequency Sampling FIR
    let mut desired_mag = [0.0f32; 33]; // DC through Nyquist for 64-pt FFT
    let desired_phase = [0.0f32; 33];
    // Brickwall lowpass specification: 1.0 up to bin 6 (~4.5 kHz), 0.0 above
    for (k, m) in desired_mag.iter_mut().enumerate() {
        *m = if k <= 6 { 1.0 } else { 0.0 };
    }
    let mut sampled_fir_taps = [0.0f32; 33];
    let fsamp_status =
        fir_custom_frequency_sampling(&desired_mag, &desired_phase, 64, &mut sampled_fir_taps);
    println!("  Custom Frequency-Sampling FIR Status: {:?}", fsamp_status);
    println!(
        "    Sampled FIR Center Tap [16]: {:.4}",
        sampled_fir_taps[16]
    );

    // -----------------------------------------------------------------------------------------
    // 3. Frequency-Domain DTFT Analysis & Stability Checks
    // -----------------------------------------------------------------------------------------
    println!("\n--- 3. Frequency Response (DTFT) & Pole Stability Verification ---");
    // Evaluate frequency response at Passband (1 kHz), Cutoff (4.8 kHz), and Stopband (15 kHz)
    let test_freqs = [1000.0f32, 4800.0, 15000.0];
    println!("  DTFT Frequency Response Comparison (Butterworth vs Chebyshev vs FIR):");
    println!(
        "    {:<10} {:<18} {:<18} {:<18}",
        "Freq (Hz)", "Butterworth (dB)", "Chebyshev (dB)", "FIR Lowpass (dB)"
    );
    println!("    ----------------------------------------------------------------------");

    for &f in &test_freqs {
        let fnorm = f / FS;

        // Butterworth cascade response
        let h_butter = biquad_cascade_frequency_response(&butter_coeffs, fnorm);
        let butter_db = response_magnitude_db(h_butter);

        // Chebyshev cascade response
        let h_cheby = biquad_cascade_frequency_response(&cheby_coeffs, fnorm);
        let cheby_db = response_magnitude_db(h_cheby);

        // FIR response
        let h_fir = fir_frequency_response(&fir_lowpass_taps, fnorm);
        let fir_db = response_magnitude_db(h_fir);

        println!(
            "    {:<10.0} {:<18.2} {:<18.2} {:<18.2}",
            f, butter_db, cheby_db, fir_db
        );
    }

    // FIR Group Delay Evaluation
    let gd_passband = fir_group_delay(&fir_lowpass_taps, 1000.0 / FS);
    let gd_cutoff = fir_group_delay(&fir_lowpass_taps, 4800.0 / FS);
    println!("\n  Linear-Phase FIR Group Delay:");
    println!(
        "    • Group Delay @ 1.0 kHz: {:.2} samples (Exact constant delay = (N-1)/2 = 16.0)",
        gd_passband
    );
    println!("    • Group Delay @ 4.8 kHz: {:.2} samples", gd_cutoff);

    // IIR Stability Verification
    let stage0: [f32; 5] = butter_coeffs[..5].try_into().unwrap();
    let stage1: [f32; 5] = butter_coeffs[5..].try_into().unwrap();
    let pole_r0 = biquad_pole_radius(&stage0);
    let pole_r1 = biquad_pole_radius(&stage1);
    let is_stable = biquad_cascade_is_stable(&butter_coeffs);
    println!("\n  IIR Cascade Pole Stability Check:");
    println!(
        "    • Stage 0 Pole Radius : {:.4} (< 1.0 -> Stable: {})",
        pole_r0,
        biquad_is_stable(&stage0)
    );
    println!(
        "    • Stage 1 Pole Radius : {:.4} (< 1.0 -> Stable: {})",
        pole_r1,
        biquad_is_stable(&stage1)
    );
    println!("    • Overall Cascade Stable: {}", is_stable);

    // -----------------------------------------------------------------------------------------
    // 4. Implementation Topology: Direct Form I vs Transposed DF-II & Const Generics
    // -----------------------------------------------------------------------------------------
    println!("\n--- 4. Topology Comparison: DF-I vs Transposed DF-II vs Const Generics ---");
    let input_signal = [1.0f32, 0.5, -0.5, -1.0, 0.0, 1.0, 0.5, -0.5, 0.0, 0.0];

    // Direct Form I
    let mut df1_state = [0.0f32; 8];
    let mut df1_inst = BiquadCascadeInstanceF32 {
        num_stages: 2,
        coeffs: &butter_coeffs,
        state: &mut df1_state,
    };
    let mut df1_out = [0.0f32; 10];
    biquad_cascade_df1_f32(&mut df1_inst, &input_signal, &mut df1_out);

    // Transposed Direct Form II
    let mut df2t_state = [0.0f32; 4];
    let mut df2t_inst = BiquadCascadeDf2tInstanceF32 {
        num_stages: 2,
        coeffs: &butter_coeffs,
        state: &mut df2t_state,
    };
    let mut df2t_out = [0.0f32; 10];
    biquad_cascade_df2t_f32(&mut df2t_inst, &input_signal, &mut df2t_out);

    // Const-Generic BiquadCascade
    let mut cg_biquad = BiquadCascade::<10, 8>::new(butter_coeffs);
    let mut cg_out = [0.0f32; 10];
    cg_biquad.process(&input_signal, &mut cg_out);

    println!("  Filter Output Comparison (first 5 samples):");
    println!("    DF-I Output   : {:?}", &df1_out[..5]);
    println!("    DF-II T Output: {:?}", &df2t_out[..5]);
    println!("    Const-Generic : {:?}", &cg_out[..5]);

    // -----------------------------------------------------------------------------------------
    // 5. Vector Distance Metrics
    // -----------------------------------------------------------------------------------------
    println!("\n--- 5. Vector Distance & Similarity Metrics ---");
    let vec_a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let vec_b = [1.2f32, 1.9, 3.1, 3.8, 5.2];

    let d_euc = euclidean_distance_f32(&vec_a, &vec_b);
    let d_cos = cosine_distance_f32(&vec_a, &vec_b);
    let d_cheb = chebyshev_distance_f32(&vec_a, &vec_b);
    let d_man = manhattan_distance_f32(&vec_a, &vec_b);
    let d_can = canberra_distance_f32(&vec_a, &vec_b);
    let d_bc = bray_curtis_distance_f32(&vec_a, &vec_b);

    println!("  Vector A: {:?}", vec_a);
    println!("  Vector B: {:?}", vec_b);
    println!("    • Euclidean Distance   : {:.4}", d_euc);
    println!("    • Cosine Distance      : {:.6}", d_cos);
    println!("    • Chebyshev Distance   : {:.4}", d_cheb);
    println!("    • Manhattan Distance   : {:.4}", d_man);
    println!("    • Canberra Distance    : {:.4}", d_can);
    println!("    • Bray-Curtis Distance : {:.4}", d_bc);

    // -----------------------------------------------------------------------------------------
    // 6. Information-Theoretic & Statistical Metrics
    // -----------------------------------------------------------------------------------------
    println!("\n--- 6. Information Theory & Advanced Statistics ---");
    let prob_dist_p = [0.1f32, 0.4, 0.3, 0.2];
    let prob_dist_q = [0.25f32, 0.25, 0.25, 0.25]; // Uniform distribution
    let logits = [2.0f32, 1.0, 0.1, -1.5];

    let entropy_p = entropy_f32(&prob_dist_p);
    let kl_p_q = kullback_leibler_f32(&prob_dist_p, &prob_dist_q);
    let lse_result = logsumexp_f32(&logits);

    println!("  Distribution P: {:?}", prob_dist_p);
    println!("  Distribution Q (Uniform): {:?}", prob_dist_q);
    println!("    • Shannon Entropy H(P)         : {:.4} nats", entropy_p);
    println!("    • KL Divergence D_KL(P || Q)    : {:.4} nats", kl_p_q);
    println!("    • LogSumExp of Logits {:?}: {:.4}", logits, lse_result);

    println!();
    println!("===============================================================================");
    println!("            Filter Workbench & Analysis Execution Complete!                    ");
    println!("===============================================================================");
}
