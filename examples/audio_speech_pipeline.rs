//! Comprehensive Audio and Speech DSP Pipeline Example
//!
//! Demonstrates:
//! 1. Audio Signal Generation (voice fundamental + harmonics + 60 Hz hum + DC offset + noise)
//! 2. DC Offset Blocking (f32 SinglePoleFilter & Q15 DcBlocker)
//! 3. Biquad Equalizer Cascade (60 Hz Notch filter & Peaking EQ)
//! 4. Peak and RMS Dynamics Envelope Tracking (f32 & Q15)
//! 5. ITU-T G.711 Telecom Companding (μ-law and A-law 8-bit quantization & reconstruction)
//! 6. Goertzel Dual-Tone Multi-Frequency (DTMF) Detection (f32 & Q15)
//! 7. Audio Frontend: Windowed Mel-Filterbank & MFCC Feature Extraction for Keyword Spotting

use embedded_dsp::*;

fn main() {
    println!("===============================================================================");
    println!("              embedded-dsp Audio & Speech Processing Pipeline                  ");
    println!("===============================================================================");
    println!();

    const FS: f32 = 16000.0; // 16 kHz sampling rate (standard for speech / voice AI)
    const NUM_SAMPLES: usize = 1024;

    // -----------------------------------------------------------------------------------------
    // 1. Synthetic Audio Frame Generation
    // -----------------------------------------------------------------------------------------
    println!("--- 1. Generating Audio Signal with Mains Hum and DC Offset ---");
    let mut raw_audio = [0.0f32; NUM_SAMPLES];
    let dc_bias = 0.35f32;
    let hum_freq = 60.0f32;
    let voice_f0 = 440.0f32; // Pitch A4
    let voice_f1 = 880.0f32; // Harmonic

    for (i, sample) in raw_audio.iter_mut().enumerate() {
        let t = i as f32 / FS;
        // Signal: DC bias + 60Hz hum + Voice tones + small noise
        let hum = 0.25 * (2.0 * core::f32::consts::PI * hum_freq * t).sin();
        let voice = 0.5 * (2.0 * core::f32::consts::PI * voice_f0 * t).sin()
            + 0.25 * (2.0 * core::f32::consts::PI * voice_f1 * t).sin();
        let prng = ((i as u64).wrapping_mul(1103515245).wrapping_add(12345) % 2147483648) as f32
            / 2147483648.0;
        let noise = 0.05 * (prng - 0.5);
        *sample = dc_bias + hum + voice + noise;
    }

    let mut mean_raw = 0.0f32;
    let mut rms_raw = 0.0f32;
    mean_f32(&raw_audio, &mut mean_raw);
    rms_f32(&raw_audio, &mut rms_raw);
    println!(
        "  Raw Signal: Mean (DC) = {:.4}, RMS = {:.4}",
        mean_raw, rms_raw
    );

    // -----------------------------------------------------------------------------------------
    // 2. DC Offset Removal (Highpass Single-Pole / DC Blocker)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 2. DC Offset Removal (f32 & Q15 DC Blockers) ---");
    let mut dc_blocked_f32 = [0.0f32; NUM_SAMPLES];
    // Highpass filter with pole near 1.0 (decay = 0.995)
    let mut dc_blocker = SinglePoleFilter::highpass(0.995);
    for (i, &s) in raw_audio.iter().enumerate() {
        dc_blocked_f32[i] = dc_blocker.process(s);
    }

    let mut mean_blocked = 0.0f32;
    // Inspect after settling period (e.g. last 512 samples)
    mean_f32(&dc_blocked_f32[512..], &mut mean_blocked);
    println!(
        "  f32 Filter: Settled Mean after DC blocker = {:.6}",
        mean_blocked
    );

    // Q15 fixed-point DC blocker verification
    let mut dc_blocker_q15 = DcBlockerQ15::from_f32_decay(0.995);
    let mut q15_raw = [0i16; NUM_SAMPLES];
    f32_to_q15(&raw_audio, &mut q15_raw);
    let mut q15_blocked = [0i16; NUM_SAMPLES];
    for (i, &s) in q15_raw.iter().enumerate() {
        q15_blocked[i] = dc_blocker_q15.process(s);
    }
    let mut mean_q15_out = 0i16;
    mean_q15(&q15_blocked[512..], &mut mean_q15_out);
    println!(
        "  Q15 Filter: Settled Mean in Q15 = {} (expected ~ 0)",
        mean_q15_out
    );

    // -----------------------------------------------------------------------------------------
    // 3. Parametric Equalizer Cascade (60 Hz Notch + 440 Hz Peaking Boost)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 3. Biquad Equalizer Cascade: 60 Hz Hum Rejection & 440 Hz Peaking ---");
    // Stage 0: 60 Hz Notch Filter (Q = 10.0)
    let notch_coeffs = biquad_notch_coeffs(60.0, FS, 10.0);
    // Stage 1: 440 Hz Peaking EQ (+6 dB boost, Q = 2.0)
    let peaking_coeffs = biquad_peaking_coeffs(440.0, FS, 2.0, 6.0);

    let mut cascade_coeffs = [0.0f32; 10];
    cascade_coeffs[..5].copy_from_slice(&notch_coeffs);
    cascade_coeffs[5..].copy_from_slice(&peaking_coeffs);

    let mut eq_state = [0.0f32; 4 * 2]; // 4 state variables per biquad stage
    let mut eq_cascade = BiquadCascadeInstanceF32 {
        num_stages: 2,
        coeffs: &cascade_coeffs,
        state: &mut eq_state,
    };

    let mut equalized_audio = [0.0f32; NUM_SAMPLES];
    biquad_cascade_df1_f32(&mut eq_cascade, &dc_blocked_f32, &mut equalized_audio);

    println!(
        "  Notch Filter Coeffs: b0={:.4}, b1={:.4}, b2={:.4}, a1={:.4}, a2={:.4}",
        notch_coeffs[0], notch_coeffs[1], notch_coeffs[2], notch_coeffs[3], notch_coeffs[4]
    );
    println!(
        "  Peaking Filter Coeffs: b0={:.4}, b1={:.4}, b2={:.4}, a1={:.4}, a2={:.4}",
        peaking_coeffs[0],
        peaking_coeffs[1],
        peaking_coeffs[2],
        peaking_coeffs[3],
        peaking_coeffs[4]
    );
    println!(
        "  EQ Cascade filtered {} samples successfully.",
        equalized_audio.len()
    );

    // -----------------------------------------------------------------------------------------
    // 4. Dynamics Envelope Followers (Peak & RMS)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 4. Dynamics Envelope Tracking (Peak and RMS Followers) ---");
    // Attack = 5 ms (80 samples @ 16kHz), Release = 50 ms (800 samples)
    let mut peak_follower = PeakEnvelopeFollower::new(80.0, 800.0);
    let mut rms_follower = RmsEnvelopeFollower::new(160.0);

    let mut peak_env = [0.0f32; NUM_SAMPLES];
    let mut rms_env = [0.0f32; NUM_SAMPLES];

    for i in 0..NUM_SAMPLES {
        peak_env[i] = peak_follower.process(equalized_audio[i]);
        rms_env[i] = rms_follower.process(equalized_audio[i]);
    }

    println!("  Settled Peak Envelope: {:.4}", peak_env[NUM_SAMPLES - 1]);
    println!("  Settled RMS Envelope : {:.4}", rms_env[NUM_SAMPLES - 1]);

    // Fixed-Point Q15 Followers
    let mut peak_follower_q15 = PeakEnvelopeFollowerQ15::new(80.0, 800.0);
    let mut q15_eq = [0i16; NUM_SAMPLES];
    f32_to_q15(&equalized_audio, &mut q15_eq);
    let mut last_q15_peak = 0i16;
    for &sample in &q15_eq {
        last_q15_peak = peak_follower_q15.process(sample);
    }
    println!("  Q15 Peak Envelope Level: {} (Q15)", last_q15_peak);

    // -----------------------------------------------------------------------------------------
    // 5. ITU-T G.711 Telecom Companding (μ-law and A-law 8-bit Codecs)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 5. ITU-T G.711 Telecom Companding Codec (16-bit PCM -> 8-bit Byte) ---");
    let test_samples: [i16; 6] = [0, 100, -500, 4096, -16384, 30000];
    println!(
        "  {:<12} {:<12} {:<12} {:<12} {:<12}",
        "Original PCM", "μ-law Byte", "μ-law Recv", "A-law Byte", "A-law Recv"
    );
    println!("  ------------------------------------------------------------------");
    for &orig in &test_samples {
        let u_byte = linear_to_ulaw(orig);
        let u_dec = ulaw_to_linear(u_byte);
        let a_byte = linear_to_alaw(orig);
        let a_dec = alaw_to_linear(a_byte);
        println!(
            "  {:<12} 0x{:02X} ({:<5})  {:<12} 0x{:02X} ({:<5})  {:<12}",
            orig, u_byte, u_byte, u_dec, a_byte, a_byte, a_dec
        );
    }

    // Continuous non-linear curve verification
    let x_f32 = 0.15f32;
    let u_comp = mu_law_compress_f32(x_f32);
    let u_exp = mu_law_expand_f32(u_comp);
    let a_comp = a_law_compress_f32(x_f32);
    let a_exp = a_law_expand_f32(a_comp);
    println!(
        "  Floating-point μ-law roundtrip: x = {:.4} -> comp = {:.4} -> expand = {:.4}",
        x_f32, u_comp, u_exp
    );
    println!(
        "  Floating-point A-law roundtrip: x = {:.4} -> comp = {:.4} -> expand = {:.4}",
        x_f32, a_comp, a_exp
    );

    // -----------------------------------------------------------------------------------------
    // 6. Goertzel Dual-Tone Multi-Frequency (DTMF) Detection
    // -----------------------------------------------------------------------------------------
    println!("\n--- 6. Goertzel Dual-Tone Multi-Frequency (DTMF) Detection ---");
    // Synthesize DTMF Digit '9': Low group 852 Hz + High group 1477 Hz
    let dtmf_low_freq = 852.0f32;
    let dtmf_high_freq = 1477.0f32;
    let mut dtmf_signal = [0.0f32; 256];

    for (i, s) in dtmf_signal.iter_mut().enumerate() {
        let t = i as f32 / FS;
        *s = 0.5 * (2.0 * core::f32::consts::PI * dtmf_low_freq * t).sin()
            + 0.5 * (2.0 * core::f32::consts::PI * dtmf_high_freq * t).sin();
    }

    // Evaluate against candidate DTMF frequencies
    let candidate_freqs = [
        697.0f32, 770.0, 852.0, 941.0, 1209.0, 1336.0, 1477.0, 1633.0,
    ];
    println!("  Scanning 8 DTMF frequency bins on 256-sample frame:");

    for &freq in &candidate_freqs {
        let mut detector = GoertzelDetector::new(freq, FS);
        for &sample in &dtmf_signal {
            detector.process_sample(sample);
        }
        let mag = detector.magnitude();
        let is_detected = mag > 0.35;
        let star = if is_detected {
            " <== DETECTED TONE"
        } else {
            ""
        };
        println!(
            "    Frequency {:>6.1} Hz: Magnitude = {:.4}{}",
            freq, mag, star
        );
    }

    // -----------------------------------------------------------------------------------------
    // 7. MFCC & Mel Filterbank Speech Feature Extraction
    // -----------------------------------------------------------------------------------------
    println!("\n--- 7. MFCC Speech Feature Extraction (Acoustic Frontend) ---");
    // 256-point speech frame (16 ms @ 16 kHz)
    const FRAME_SIZE: usize = 256;
    let mut speech_frame = [0.0f32; FRAME_SIZE];
    for (i, val) in speech_frame.iter_mut().enumerate() {
        let t = i as f32 / FS;
        // Formant simulation: 500 Hz + 1500 Hz + 2500 Hz
        *val = 0.6 * (2.0 * core::f32::consts::PI * 500.0 * t).sin()
            + 0.3 * (2.0 * core::f32::consts::PI * 1500.0 * t).sin()
            + 0.1 * (2.0 * core::f32::consts::PI * 2500.0 * t).sin();
    }

    // Apply Hamming window to reduce spectral leakage
    let mut window = [0.0f32; FRAME_SIZE];
    hamming_f32(&mut window);
    apply_window_f32(&mut speech_frame, &window);

    // Compute MFCCs: 26 Mel filter channels -> 13 Cepstral Coefficients
    let mut mel_scratch = [0.0f32; 26];
    let mut mfcc_coeffs = [0.0f32; 13];

    let status = mfcc_f32(
        &speech_frame,
        FS,
        100.0,  // Low freq: 100 Hz
        7000.0, // High freq: 7000 Hz
        &mut mel_scratch,
        &mut mfcc_coeffs,
    );

    if status == Status::Success {
        println!("  Extracted 13 MFCC Coefficients for Wake-Word / Speech AI:");
        for (idx, coeff) in mfcc_coeffs.iter().enumerate() {
            println!("    MFCC[{:>2}]: {:>9.4}", idx, coeff);
        }
    } else {
        println!("  MFCC extraction failed with status: {:?}", status);
    }

    println!();
    println!("===============================================================================");
    println!("                 Audio & Speech Pipeline Execution Complete!                   ");
    println!("===============================================================================");
}
