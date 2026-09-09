#[cfg(test)]
mod test_suite {
    use crate::state::{ProcessorMode, StudioPreset, StudioState, WaveformType};
    use crate::views::codegen::{CodegenView, TargetLanguage};

    #[test]
    fn test_studio_state_default_and_recompute() {
        let mut state = StudioState::default();
        assert_eq!(state.sample_rate, 48000.0);
        assert_eq!(state.block_size, 2048);
        assert_eq!(state.time_a.len(), 2048);
        assert_eq!(state.time_b.len(), 2048);
        assert_eq!(state.fft_mag_a.len(), 1024);

        // Modify parameters and recompute
        state.source_a.frequency_hz = 1000.0;
        state.proc_a.cutoff_hz = 2000.0;
        state.recompute_all();

        assert!(state.vu_peak_a > 0.0);
        assert!(state.vu_rms_a > 0.0);
    }

    #[test]
    fn test_all_studio_presets() {
        let mut state = StudioState::default();

        for &preset in StudioPreset::ALL {
            state.load_preset(preset);
            assert!(!state.time_a.is_empty());
            assert!(!state.time_b.is_empty());
            assert!(!state.fft_mag_a.is_empty());
            assert!(!state.fft_mag_b.is_empty());
        }
    }

    #[test]
    fn test_null_cancellation_preset_achieves_perfect_null() {
        let mut state = StudioState::default();
        state.load_preset(StudioPreset::NullCancellationTest);

        // In null test preset, Slot A and Slot B have identical inputs and filter configurations
        let max_diff = state
            .time_null
            .iter()
            .fold(0.0f32, |acc, &x| acc.max(x.abs()));

        // Should cancel out with essentially machine precision
        assert!(
            max_diff < 1e-5,
            "A/B Null cancellation residual too high: {}",
            max_diff
        );
    }

    #[test]
    fn test_all_waveform_generators() {
        let mut state = StudioState::default();
        state.proc_a.mode = ProcessorMode::Bypass;

        for &waveform in WaveformType::ALL {
            state.source_a.waveform = waveform;
            state.source_a.frequency_hz = 440.0;
            state.source_a.amplitude = 0.8;
            state.source_a.mute = false;
            state.source_a.invert = false;
            state.recompute_all();

            for (idx, &s) in state.time_a.iter().enumerate() {
                assert!(
                    s.is_finite(),
                    "Waveform {:?} produced non-finite sample at idx {}: {}",
                    waveform,
                    idx,
                    s
                );
                assert!(
                    (-1.05..=1.05).contains(&s),
                    "Waveform {:?} out of bounds at idx {}: {}",
                    waveform,
                    idx,
                    s
                );
            }
        }
    }

    #[test]
    fn test_source_mute_and_invert() {
        let mut state = StudioState::default();
        state.source_a.waveform = WaveformType::Sine;
        state.source_a.amplitude = 0.9;
        state.source_a.mute = true;
        state.recompute_all();

        // When muted, signal should be all zeros
        for &s in &state.time_a {
            assert_eq!(s, 0.0);
        }

        // When inverted, polarity is reversed
        state.source_a.mute = false;
        state.source_a.invert = false;
        state.proc_a.mode = ProcessorMode::Bypass;
        state.recompute_all();
        let normal_first = state.time_a[1];

        state.source_a.invert = true;
        state.recompute_all();
        let inverted_first = state.time_a[1];

        assert!((normal_first + inverted_first).abs() < 1e-6);
    }

    #[test]
    fn test_all_processor_modes() {
        let mut state = StudioState::default();
        state.source_a.waveform = WaveformType::Sine;
        state.source_a.frequency_hz = 440.0;
        state.source_a.amplitude = 0.7;

        for &mode in ProcessorMode::ALL {
            state.proc_a.mode = mode;
            state.proc_a.cutoff_hz = 1000.0;
            state.proc_a.q = 1.0;
            state.proc_a.gain_db = 3.0;
            state.proc_a.drive = 0.8;
            state.recompute_all();

            for &s in &state.time_a {
                assert!(s.is_finite(), "Diverged in mode {:?}", mode);
            }
        }
    }

    #[test]
    fn test_q15_quantization_floor() {
        let mut state = StudioState::default();
        state.load_preset(StudioPreset::FloatVsFixedQ15);

        // Verify that Q15 quantization introduces bounded quantization noise (small difference)
        let max_diff = state
            .time_null
            .iter()
            .fold(0.0f32, |acc, &x| acc.max(x.abs()));

        assert!(
            max_diff > 0.0 && max_diff < 0.001,
            "Expected Q15 quantization noise, got {}",
            max_diff
        );
    }

    #[test]
    fn test_forensic_snapshot_analysis() {
        let mut state = StudioState::default();
        state.proc_a.mode = ProcessorMode::BiquadLowpass;
        state.proc_a.cutoff_hz = 1000.0;
        state.proc_a.q = 2.0;

        state.trigger_snapshot(0.01);
        assert!(state.snapshot_captured);
        assert_eq!(state.snapshot_buffer.samples().len(), 4096);

        let info = state
            .snapshot_response_info
            .expect("Should have analysis info");
        assert!(info.is_stable);
        assert!(info.peak_gain > 0.0);
        assert!(info.total_energy > 0.0);
        assert!(info.settling_time_samples > 0);
    }

    #[test]
    fn test_micro_benchmarks_execution() {
        let mut state = StudioState::default();
        state.run_benchmarks();

        assert!(
            state
                .last_bench_report
                .contains("embedded-dsp Micro-Benchmark Results")
        );
        assert!(state.last_bench_report.contains("Biquad Cascade DF1 F32"));
        assert!(state.last_bench_report.contains("State Variable Filter"));
        assert!(state.last_bench_report.contains("PolyBLEP Saw Oscillator"));
        assert!(state.last_bench_report.contains("CFFT F32"));
        assert!(state.last_bench_report.contains("fast_ln_f32() vs std::ln"));
    }

    #[test]
    fn test_mcu_codegen() {
        let state = StudioState::default();
        let mut codegen = CodegenView::new();

        codegen.target_language = TargetLanguage::RustNoStd;
        assert_eq!(codegen.target_language, TargetLanguage::RustNoStd);
        let rust_code = crate::views::codegen::generate_rust_code(&state.proc_a, state.sample_rate);
        assert!(rust_code.contains("#![no_std]"));
        assert!(rust_code.contains("pub static BIQUAD_COEFFS: [f32; 5]"));
        assert!(rust_code.contains("biquad_cascade_df1_f32"));

        codegen.target_language = TargetLanguage::CmsisDspC;
        assert_eq!(codegen.target_language, TargetLanguage::CmsisDspC);
        let c_code = crate::views::codegen::generate_c_code(&state.proc_a, state.sample_rate);
        assert!(c_code.contains("#include \"arm_math.h\""));
        assert!(c_code.contains("static const float32_t biquad_coeffs[5]"));
        assert!(c_code.contains("arm_biquad_cascade_df1_f32"));
    }

    #[test]
    fn test_wav_and_csv_export_and_import() {
        use crate::export::{export_csv, export_wav_16bit, parse_csv, parse_wav_16bit};

        let sample_rate = 48000;
        let original_samples: Vec<f32> = (0..1024)
            .map(|i| {
                (2.0 * core::f32::consts::PI * 440.0 * (i as f32) / (sample_rate as f32)).sin()
            })
            .collect();

        // WAV round-trip
        let wav_bytes = export_wav_16bit(&original_samples, sample_rate);
        assert!(wav_bytes.len() >= 44 + 1024 * 2);
        assert_eq!(&wav_bytes[0..4], b"RIFF");
        assert_eq!(&wav_bytes[8..12], b"WAVE");

        let (parsed_samples, parsed_rate) =
            parse_wav_16bit(&wav_bytes).expect("Failed to parse WAV");
        assert_eq!(parsed_rate, sample_rate);
        assert_eq!(parsed_samples.len(), original_samples.len());
        for (orig, parsed) in original_samples.iter().zip(parsed_samples.iter()) {
            assert!((orig - parsed).abs() < 1e-3);
        }

        // CSV round-trip
        let csv_text = export_csv(&original_samples, sample_rate as f32);
        assert!(csv_text.starts_with("sample_idx,time_secs,amplitude\n"));
        let parsed_csv = parse_csv(&csv_text).expect("Failed to parse CSV");
        assert_eq!(parsed_csv.len(), original_samples.len());
        for (orig, parsed) in original_samples.iter().zip(parsed_csv.iter()) {
            assert!((orig - parsed).abs() < 1e-4);
        }
    }
}
