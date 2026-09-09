//! Global State and Audio Engine simulation for embedded-dsp-studio.

use core::f32::consts::TAU;
use embedded_dsp::filter_design::{
    biquad_bandpass_coeffs, biquad_highpass_coeffs, biquad_lowpass_coeffs, biquad_notch_coeffs,
    biquad_peaking_coeffs,
};
use embedded_dsp::filtering::{BiquadCascadeInstanceF32, biquad_cascade_df1_f32};
use embedded_dsp::snapshot::{ImpulseResponseInfo, SnapshotBuffer, analyze_impulse_response};
use embedded_dsp::svf::StateVariableFilter;
use embedded_dsp::synthesis::{
    ChirpSweep, KellettPinkNoise, PolyBlepOscillator, PolyBlepWaveform, WhiteNoise,
};
use embedded_dsp::types::q15;
use embedded_dsp::window::hanning_f32;
use serde::{Deserialize, Serialize};

pub const DEFAULT_SAMPLE_RATE: f32 = 48000.0;
pub const DEFAULT_BLOCK_SIZE: usize = 2048;
pub const SNAPSHOT_SIZE: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveformType {
    Sine,
    Sawtooth,
    Square,
    Triangle,
    ChirpLinear,
    ChirpExponential,
    PinkNoise,
    WhiteNoise,
    Impulse,
    Step,
}

impl WaveformType {
    pub const ALL: &'static [WaveformType] = &[
        WaveformType::Sine,
        WaveformType::Sawtooth,
        WaveformType::Square,
        WaveformType::Triangle,
        WaveformType::ChirpLinear,
        WaveformType::ChirpExponential,
        WaveformType::PinkNoise,
        WaveformType::WhiteNoise,
        WaveformType::Impulse,
        WaveformType::Step,
    ];

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Sine => "Sine (PolyBLEP)",
            Self::Sawtooth => "Sawtooth (PolyBLEP)",
            Self::Square => "Square (PolyBLEP)",
            Self::Triangle => "Triangle (PolyBLEP)",
            Self::ChirpLinear => "Linear Chirp Sweep",
            Self::ChirpExponential => "Exponential Chirp",
            Self::PinkNoise => "Paul Kellett Pink Noise (1/f)",
            Self::WhiteNoise => "Xorshift White Noise",
            Self::Impulse => "Unit Impulse [δ(n)]",
            Self::Step => "Unit Step [u(n)]",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceRouting {
    SourceA,
    SourceB,
    SumBoth,
}

impl SourceRouting {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::SourceA => "Source A",
            Self::SourceB => "Source B",
            Self::SumBoth => "Source A + B (Sum)",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalSourceConfig {
    pub enabled: bool,
    pub waveform: WaveformType,
    pub frequency_hz: f32,
    pub amplitude: f32,
    pub phase_deg: f32,
    pub mute: bool,
    pub invert: bool,
}

impl Default for SignalSourceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            waveform: WaveformType::Sine,
            frequency_hz: 440.0,
            amplitude: 0.707,
            phase_deg: 0.0,
            mute: false,
            invert: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessorMode {
    Bypass,
    BiquadLowpass,
    BiquadHighpass,
    BiquadBandpass,
    BiquadNotch,
    BiquadPeaking,
    SvfLowpass,
    SvfHighpass,
    SvfBandpass,
    SvfNotch,
    SvfPeak,
    FixedPointQ15,
}

impl ProcessorMode {
    pub const ALL: &'static [ProcessorMode] = &[
        ProcessorMode::Bypass,
        ProcessorMode::BiquadLowpass,
        ProcessorMode::BiquadHighpass,
        ProcessorMode::BiquadBandpass,
        ProcessorMode::BiquadNotch,
        ProcessorMode::BiquadPeaking,
        ProcessorMode::SvfLowpass,
        ProcessorMode::SvfHighpass,
        ProcessorMode::SvfBandpass,
        ProcessorMode::SvfNotch,
        ProcessorMode::SvfPeak,
        ProcessorMode::FixedPointQ15,
    ];

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Bypass => "Bypass (Thru)",
            Self::BiquadLowpass => "Biquad IIR Lowpass",
            Self::BiquadHighpass => "Biquad IIR Highpass",
            Self::BiquadBandpass => "Biquad IIR Bandpass",
            Self::BiquadNotch => "Biquad IIR Notch",
            Self::BiquadPeaking => "Biquad IIR Peaking EQ",
            Self::SvfLowpass => "State Variable Lowpass (Simper)",
            Self::SvfHighpass => "State Variable Highpass (Simper)",
            Self::SvfBandpass => "State Variable Bandpass (Simper)",
            Self::SvfNotch => "State Variable Notch (Simper)",
            Self::SvfPeak => "State Variable Peak (Simper)",
            Self::FixedPointQ15 => "Q15 Fixed-Point Quantizer",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    pub mode: ProcessorMode,
    pub cutoff_hz: f32,
    pub q: f32,
    pub gain_db: f32,
    pub drive: f32,
    pub mute: bool,
    pub invert: bool,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            mode: ProcessorMode::BiquadLowpass,
            cutoff_hz: 1000.0,
            q: 0.7071,
            gain_db: 0.0,
            drive: 0.5,
            mute: false,
            invert: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudioPreset {
    BiquadVsSvf,
    FloatVsFixedQ15,
    NullCancellationTest,
    PolyBlepHarmonics,
    PinkNoiseAnalysis,
    ImpulseResponseForensic,
}

impl StudioPreset {
    pub const ALL: &'static [StudioPreset] = &[
        StudioPreset::BiquadVsSvf,
        StudioPreset::FloatVsFixedQ15,
        StudioPreset::NullCancellationTest,
        StudioPreset::PolyBlepHarmonics,
        StudioPreset::PinkNoiseAnalysis,
        StudioPreset::ImpulseResponseForensic,
    ];

    pub fn title(&self) -> &'static str {
        match self {
            Self::BiquadVsSvf => "⚡ Biquad IIR vs State Variable Filter",
            Self::FloatVsFixedQ15 => "🔬 Float32 vs Q15 Fixed-Point Quantization Noise",
            Self::NullCancellationTest => "🎯 A/B Null Cancellation Test (Exact Zero Error)",
            Self::PolyBlepHarmonics => "🌊 PolyBLEP Anti-Aliasing & Harmonics",
            Self::PinkNoiseAnalysis => "📊 Pink Noise 1/f Spectral Slope (-3 dB/oct)",
            Self::ImpulseResponseForensic => "🔍 Forensic Impulse Response & Settling Time",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::BiquadVsSvf => {
                "Compare classic Biquad Direct Form I with Andrew Simper's double-sampled State Variable Filter under 1 kHz resonance."
            }
            Self::FloatVsFixedQ15 => {
                "Measure quantization distortion, SNR and spectral degradation introduced by CMSIS-style Q15 arithmetic."
            }
            Self::NullCancellationTest => {
                "Route identical signals through both processors and invert one to verify bit-exact null cancellation."
            }
            Self::PolyBlepHarmonics => {
                "Inspect harmonic suppression of band-limited PolyBLEP Sawtooth and Square waveforms without aliasing foldover."
            }
            Self::PinkNoiseAnalysis => {
                "Evaluate Paul Kellett's multi-pole IIR pink noise generator across the 20 Hz - 20 kHz log frequency spectrum."
            }
            Self::ImpulseResponseForensic => {
                "Inject a Dirac unit impulse into resonant filters to evaluate ringing, damping, settling time and stability."
            }
        }
    }
}

pub struct StudioState {
    pub sample_rate: f32,
    pub block_size: usize,

    // Signal Sources
    pub source_a: SignalSourceConfig,
    pub source_b: SignalSourceConfig,
    pub routing_a: SourceRouting,
    pub routing_b: SourceRouting,

    // Processors
    pub proc_a: ProcessorConfig,
    pub proc_b: ProcessorConfig,
    pub null_test_mode: bool,

    // Master
    pub master_gain: f32,
    pub master_mute: bool,
    pub limiter: bool,

    // Computed Time-Domain Waveform Buffers
    pub time_a: Vec<f32>,
    pub time_b: Vec<f32>,
    pub time_null: Vec<f32>,
    pub time_master: Vec<f32>,

    // Computed Frequency-Domain Magnitudes in dB
    pub fft_mag_a: Vec<f32>,
    pub fft_mag_b: Vec<f32>,
    pub fft_mag_null: Vec<f32>,
    pub fft_frequencies: Vec<f32>,

    // Metering
    pub vu_peak_a: f32,
    pub vu_rms_a: f32,
    pub vu_peak_b: f32,
    pub vu_rms_b: f32,
    pub vu_peak_master: f32,
    pub vu_rms_master: f32,

    // Forensic Snapshot Buffer
    pub snapshot_buffer: SnapshotBuffer<SNAPSHOT_SIZE>,
    pub snapshot_response_info: Option<ImpulseResponseInfo>,
    pub snapshot_captured: bool,

    // Benchmarking Metrics
    pub last_bench_report: String,
}

impl Default for StudioState {
    fn default() -> Self {
        let mut state = Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            block_size: DEFAULT_BLOCK_SIZE,

            source_a: SignalSourceConfig {
                enabled: true,
                waveform: WaveformType::Sine,
                frequency_hz: 440.0,
                amplitude: 0.8,
                phase_deg: 0.0,
                mute: false,
                invert: false,
            },
            source_b: SignalSourceConfig {
                enabled: true,
                waveform: WaveformType::Sawtooth,
                frequency_hz: 880.0,
                amplitude: 0.6,
                phase_deg: 0.0,
                mute: false,
                invert: false,
            },
            routing_a: SourceRouting::SourceA,
            routing_b: SourceRouting::SourceB,

            proc_a: ProcessorConfig {
                mode: ProcessorMode::BiquadLowpass,
                cutoff_hz: 1200.0,
                q: 1.414,
                gain_db: 0.0,
                drive: 0.5,
                mute: false,
                invert: false,
            },
            proc_b: ProcessorConfig {
                mode: ProcessorMode::SvfLowpass,
                cutoff_hz: 1200.0,
                q: 1.414,
                gain_db: 0.0,
                drive: 0.5,
                mute: false,
                invert: false,
            },
            null_test_mode: false,

            master_gain: 1.0,
            master_mute: false,
            limiter: true,

            time_a: vec![0.0; DEFAULT_BLOCK_SIZE],
            time_b: vec![0.0; DEFAULT_BLOCK_SIZE],
            time_null: vec![0.0; DEFAULT_BLOCK_SIZE],
            time_master: vec![0.0; DEFAULT_BLOCK_SIZE],

            fft_mag_a: vec![-120.0; DEFAULT_BLOCK_SIZE / 2],
            fft_mag_b: vec![-120.0; DEFAULT_BLOCK_SIZE / 2],
            fft_mag_null: vec![-120.0; DEFAULT_BLOCK_SIZE / 2],
            fft_frequencies: vec![0.0; DEFAULT_BLOCK_SIZE / 2],

            vu_peak_a: 0.0,
            vu_rms_a: 0.0,
            vu_peak_b: 0.0,
            vu_rms_b: 0.0,
            vu_peak_master: 0.0,
            vu_rms_master: 0.0,

            snapshot_buffer: SnapshotBuffer::new(),
            snapshot_response_info: None,
            snapshot_captured: false,

            last_bench_report: "No benchmark run yet. Click 'Run DSP Benchmarks'.".to_string(),
        };

        state.recompute_all();
        state
    }
}

impl StudioState {
    pub fn load_preset(&mut self, preset: StudioPreset) {
        match preset {
            StudioPreset::BiquadVsSvf => {
                self.source_a.waveform = WaveformType::Sawtooth;
                self.source_a.frequency_hz = 220.0;
                self.source_a.amplitude = 0.8;
                self.source_a.mute = false;

                self.routing_a = SourceRouting::SourceA;
                self.routing_b = SourceRouting::SourceA;

                self.proc_a.mode = ProcessorMode::BiquadLowpass;
                self.proc_a.cutoff_hz = 1200.0;
                self.proc_a.q = 2.5;

                self.proc_b.mode = ProcessorMode::SvfLowpass;
                self.proc_b.cutoff_hz = 1200.0;
                self.proc_b.q = 2.5;
                self.null_test_mode = false;
            }
            StudioPreset::FloatVsFixedQ15 => {
                self.source_a.waveform = WaveformType::Sine;
                self.source_a.frequency_hz = 1000.0;
                self.source_a.amplitude = 0.85;

                self.routing_a = SourceRouting::SourceA;
                self.routing_b = SourceRouting::SourceA;

                self.proc_a.mode = ProcessorMode::Bypass;
                self.proc_b.mode = ProcessorMode::FixedPointQ15;
                self.null_test_mode = true; // Observe quantization residual
            }
            StudioPreset::NullCancellationTest => {
                self.source_a.waveform = WaveformType::Sine;
                self.source_a.frequency_hz = 440.0;
                self.source_a.amplitude = 0.8;

                self.routing_a = SourceRouting::SourceA;
                self.routing_b = SourceRouting::SourceA;

                self.proc_a.mode = ProcessorMode::BiquadLowpass;
                self.proc_a.cutoff_hz = 1500.0;
                self.proc_a.q = 0.7071;

                self.proc_b.mode = ProcessorMode::BiquadLowpass;
                self.proc_b.cutoff_hz = 1500.0;
                self.proc_b.q = 0.7071;
                self.null_test_mode = true; // Expect zero cancellation
            }
            StudioPreset::PolyBlepHarmonics => {
                self.source_a.waveform = WaveformType::Square;
                self.source_a.frequency_hz = 880.0;
                self.source_a.amplitude = 0.7;

                self.source_b.waveform = WaveformType::Triangle;
                self.source_b.frequency_hz = 880.0;
                self.source_b.amplitude = 0.7;

                self.routing_a = SourceRouting::SourceA;
                self.routing_b = SourceRouting::SourceB;

                self.proc_a.mode = ProcessorMode::Bypass;
                self.proc_b.mode = ProcessorMode::Bypass;
                self.null_test_mode = false;
            }
            StudioPreset::PinkNoiseAnalysis => {
                self.source_a.waveform = WaveformType::PinkNoise;
                self.source_a.amplitude = 0.9;

                self.routing_a = SourceRouting::SourceA;
                self.routing_b = SourceRouting::SourceA;

                self.proc_a.mode = ProcessorMode::Bypass;
                self.proc_b.mode = ProcessorMode::BiquadLowpass;
                self.proc_b.cutoff_hz = 2500.0;
                self.proc_b.q = 0.7071;
                self.null_test_mode = false;
            }
            StudioPreset::ImpulseResponseForensic => {
                self.source_a.waveform = WaveformType::Impulse;
                self.source_a.amplitude = 1.0;

                self.routing_a = SourceRouting::SourceA;
                self.routing_b = SourceRouting::SourceA;

                self.proc_a.mode = ProcessorMode::BiquadLowpass;
                self.proc_a.cutoff_hz = 800.0;
                self.proc_a.q = 4.0; // Underdamped ringing

                self.proc_b.mode = ProcessorMode::SvfLowpass;
                self.proc_b.cutoff_hz = 800.0;
                self.proc_b.q = 4.0;
                self.null_test_mode = false;

                self.trigger_snapshot(0.01);
            }
        }

        self.recompute_all();
    }

    /// Recomputes the entire DSP block: generates sources, runs processors, calculates FFTs and levels.
    pub fn recompute_all(&mut self) {
        let n = self.block_size;
        let mut raw_a = vec![0.0f32; n];
        let mut raw_b = vec![0.0f32; n];

        // 1. Generate Signal Source A
        generate_source_signal(&self.source_a, self.sample_rate, &mut raw_a);

        // 2. Generate Signal Source B
        generate_source_signal(&self.source_b, self.sample_rate, &mut raw_b);

        // 3. Routing to Processor Inputs
        let mut in_a = vec![0.0f32; n];
        let mut in_b = vec![0.0f32; n];
        match self.routing_a {
            SourceRouting::SourceA => in_a.copy_from_slice(&raw_a),
            SourceRouting::SourceB => in_a.copy_from_slice(&raw_b),
            SourceRouting::SumBoth => {
                for i in 0..n {
                    in_a[i] = 0.5 * (raw_a[i] + raw_b[i]);
                }
            }
        }
        match self.routing_b {
            SourceRouting::SourceA => in_b.copy_from_slice(&raw_a),
            SourceRouting::SourceB => in_b.copy_from_slice(&raw_b),
            SourceRouting::SumBoth => {
                for i in 0..n {
                    in_b[i] = 0.5 * (raw_a[i] + raw_b[i]);
                }
            }
        }

        // 4. Run Processor A
        self.time_a.resize(n, 0.0);
        run_dsp_processor(&self.proc_a, self.sample_rate, &in_a, &mut self.time_a);

        // 5. Run Processor B
        self.time_b.resize(n, 0.0);
        run_dsp_processor(&self.proc_b, self.sample_rate, &in_b, &mut self.time_b);

        // 6. Compute Null Difference and Master Output
        self.time_null.resize(n, 0.0);
        self.time_master.resize(n, 0.0);

        let master_mult = if self.master_mute {
            0.0
        } else {
            self.master_gain
        };
        for i in 0..n {
            let diff = self.time_a[i] - self.time_b[i];
            self.time_null[i] = diff;

            let mixed = if self.null_test_mode {
                diff
            } else {
                0.5 * (self.time_a[i] + self.time_b[i])
            };
            let mut out = mixed * master_mult;
            if self.limiter {
                out = out.clamp(-1.0, 1.0);
            }
            self.time_master[i] = out;
        }

        // 7. Calculate FFTs
        let fft_len = n / 2;
        self.fft_mag_a.resize(fft_len, -120.0);
        self.fft_mag_b.resize(fft_len, -120.0);
        self.fft_mag_null.resize(fft_len, -120.0);
        self.fft_frequencies.resize(fft_len, 0.0);

        compute_power_spectrum_db(&self.time_a, &mut self.fft_mag_a);
        compute_power_spectrum_db(&self.time_b, &mut self.fft_mag_b);
        compute_power_spectrum_db(&self.time_null, &mut self.fft_mag_null);

        let bin_hz = (self.sample_rate * 0.5) / (fft_len as f32);
        for i in 0..fft_len {
            self.fft_frequencies[i] = (i as f32) * bin_hz;
        }

        // 8. Metering: Peak and RMS
        let (pa, ra) = compute_peak_and_rms(&self.time_a);
        self.vu_peak_a = pa;
        self.vu_rms_a = ra;

        let (pb, rb) = compute_peak_and_rms(&self.time_b);
        self.vu_peak_b = pb;
        self.vu_rms_b = rb;

        let (pm, rm) = compute_peak_and_rms(&self.time_master);
        self.vu_peak_master = pm;
        self.vu_rms_master = rm;
    }

    /// Captures a deterministic 4096-sample snapshot and analyzes its impulse response metrics.
    pub fn trigger_snapshot(&mut self, threshold: f32) {
        let mut buf = SnapshotBuffer::<SNAPSHOT_SIZE>::new();
        // Generate an impulse and run processor A through it for forensic analysis
        let mut impulse_in = vec![0.0f32; SNAPSHOT_SIZE];
        impulse_in[0] = 1.0;

        let mut out = vec![0.0f32; SNAPSHOT_SIZE];
        run_dsp_processor(&self.proc_a, self.sample_rate, &impulse_in, &mut out);

        for &sample in &out {
            buf.push(sample);
        }

        let info = analyze_impulse_response(buf.samples(), threshold);
        self.snapshot_buffer = buf;
        self.snapshot_response_info = Some(info);
        self.snapshot_captured = true;
    }

    /// Runs real-time microbenchmarks comparing DSP routines.
    pub fn run_benchmarks(&mut self) {
        use web_time::Instant;

        let mut out = String::new();
        let iterations = 100;
        let n = self.block_size;
        let in_buf = vec![0.5f32; n];
        let mut out_buf = vec![0.0f32; n];

        out.push_str("--- embedded-dsp Micro-Benchmark Results ---\n");
        out.push_str(&format!(
            "Configuration: Block Size = {}, Sample Rate = {:.0} Hz, Iterations = {}\n\n",
            n, self.sample_rate, iterations
        ));

        // 1. Biquad Direct Form I
        let coeffs = biquad_lowpass_coeffs(1000.0, self.sample_rate, 0.7071);
        let mut state = [0.0f32; 4];
        let start = Instant::now();
        for _ in 0..iterations {
            let mut inst = BiquadCascadeInstanceF32::init(1, &coeffs, &mut state);
            biquad_cascade_df1_f32(&mut inst, &in_buf, &mut out_buf);
        }
        let dur_biquad = start.elapsed();
        let biquad_per_block_us = dur_biquad.as_secs_f64() * 1e6 / (iterations as f64);
        let biquad_msamples_per_sec = ((n * iterations) as f64) / dur_biquad.as_secs_f64() / 1e6;
        out.push_str(&format!(
            "• Biquad Cascade DF1 F32:  {:>7.2} µs/block  ({:.2} M samples/sec)\n",
            biquad_per_block_us, biquad_msamples_per_sec
        ));

        // 2. State Variable Filter (SVF)
        let mut svf = StateVariableFilter::new(self.sample_rate);
        svf.set_cutoff(1000.0);
        svf.set_resonance(0.7071);
        let start = Instant::now();
        for _ in 0..iterations {
            for i in 0..n {
                svf.process(in_buf[i]);
                out_buf[i] = svf.low();
            }
        }
        let dur_svf = start.elapsed();
        let svf_per_block_us = dur_svf.as_secs_f64() * 1e6 / (iterations as f64);
        let svf_msamples_per_sec = ((n * iterations) as f64) / dur_svf.as_secs_f64() / 1e6;
        out.push_str(&format!(
            "• State Variable Filter:   {:>7.2} µs/block  ({:.2} M samples/sec)\n",
            svf_per_block_us, svf_msamples_per_sec
        ));

        // 3. PolyBLEP Oscillator
        let mut osc = PolyBlepOscillator::new(self.sample_rate, 440.0, PolyBlepWaveform::Sawtooth);
        let start = Instant::now();
        for _ in 0..iterations {
            for i in 0..n {
                out_buf[i] = osc.next_sample();
            }
        }
        let dur_osc = start.elapsed();
        let osc_per_block_us = dur_osc.as_secs_f64() * 1e6 / (iterations as f64);
        let osc_msamples_per_sec = ((n * iterations) as f64) / dur_osc.as_secs_f64() / 1e6;
        out.push_str(&format!(
            "• PolyBLEP Saw Oscillator: {:>7.2} µs/block  ({:.2} M samples/sec)\n",
            osc_per_block_us, osc_msamples_per_sec
        ));

        // 4. CFFT F32
        let mut complex_buf = vec![0.0f32; 2 * 1024];
        let start = Instant::now();
        for _ in 0..iterations {
            embedded_dsp::transform::cfft_f32(&mut complex_buf, 1024, 0, 1);
        }
        let dur_fft = start.elapsed();
        let fft_per_block_us = dur_fft.as_secs_f64() * 1e6 / (iterations as f64);
        out.push_str(&format!(
            "• 1024-point CFFT F32:     {:>7.2} µs/transform\n",
            fft_per_block_us
        ));

        // 5. Fast Math vs Standard Math (fast_ln_f32 vs std::ln)
        let mut sum_std = 0.0f32;
        let start = Instant::now();
        for _ in 0..iterations {
            for i in 0..n {
                sum_std += ((i as f32 + 1.0) * 0.01).ln();
            }
        }
        let dur_std = start.elapsed();

        let mut sum_fast = 0.0f32;
        let start = Instant::now();
        for _ in 0..iterations {
            for i in 0..n {
                sum_fast += embedded_dsp::fast_math::fast_ln_f32((i as f32 + 1.0) * 0.01);
            }
        }
        let dur_fast = start.elapsed();
        let speedup = dur_std.as_secs_f64() / dur_fast.as_secs_f64().max(1e-9);
        out.push_str(&format!(
            "• fast_ln_f32() vs std::ln: {:>7.2} µs ({:.2}x speedup, check: {:.1}/{:.1})\n",
            dur_fast.as_secs_f64() * 1e6 / (iterations as f64),
            speedup,
            sum_std,
            sum_fast
        ));

        self.last_bench_report = out;
    }
}

fn generate_source_signal(config: &SignalSourceConfig, sample_rate: f32, dst: &mut [f32]) {
    if !config.enabled || config.mute {
        dst.fill(0.0);
        return;
    }

    let n = dst.len();
    let amp = config.amplitude;
    let phase_offset = config.phase_deg / 360.0;

    match config.waveform {
        WaveformType::Sine => {
            let freq = config.frequency_hz;
            let phase_step = freq / sample_rate;
            for i in 0..n {
                let phase = (phase_offset + (i as f32) * phase_step) % 1.0;
                dst[i] = (phase * TAU).sin() * amp;
            }
        }
        WaveformType::Sawtooth => {
            let mut osc = PolyBlepOscillator::new(
                sample_rate,
                config.frequency_hz,
                PolyBlepWaveform::Sawtooth,
            );
            for i in 0..n {
                dst[i] = osc.next_sample() * amp;
            }
        }
        WaveformType::Square => {
            let mut osc =
                PolyBlepOscillator::new(sample_rate, config.frequency_hz, PolyBlepWaveform::Square);
            for i in 0..n {
                dst[i] = osc.next_sample() * amp;
            }
        }
        WaveformType::Triangle => {
            let mut osc = PolyBlepOscillator::new(
                sample_rate,
                config.frequency_hz,
                PolyBlepWaveform::Triangle,
            );
            for i in 0..n {
                dst[i] = osc.next_sample() * amp;
            }
        }
        WaveformType::ChirpLinear => {
            let mut chirp = ChirpSweep::new(
                sample_rate,
                config.frequency_hz.max(20.0),
                (sample_rate * 0.45).min(20000.0),
                (n as f32) / sample_rate,
                false,
            );
            for i in 0..n {
                dst[i] = chirp.next_sample() * amp;
            }
        }
        WaveformType::ChirpExponential => {
            let mut chirp = ChirpSweep::new(
                sample_rate,
                config.frequency_hz.max(20.0),
                (sample_rate * 0.45).min(20000.0),
                (n as f32) / sample_rate,
                true,
            );
            for i in 0..n {
                dst[i] = chirp.next_sample() * amp;
            }
        }
        WaveformType::PinkNoise => {
            let mut pink = KellettPinkNoise::new(1337);
            for i in 0..n {
                dst[i] = pink.next_sample() * amp;
            }
        }
        WaveformType::WhiteNoise => {
            let mut white = WhiteNoise::new(9999);
            for i in 0..n {
                dst[i] = white.next_sample() * amp;
            }
        }
        WaveformType::Impulse => {
            dst.fill(0.0);
            if n > 0 {
                dst[0] = 1.0 * amp;
            }
        }
        WaveformType::Step => {
            dst.fill(1.0 * amp);
        }
    }

    if config.invert {
        for s in dst.iter_mut() {
            *s = -*s;
        }
    }
}

fn run_dsp_processor(config: &ProcessorConfig, sample_rate: f32, src: &[f32], dst: &mut [f32]) {
    if config.mute {
        dst.fill(0.0);
        return;
    }

    let n = src.len().min(dst.len());
    let cutoff = config.cutoff_hz.clamp(10.0, sample_rate * 0.49);
    let q = config.q.max(0.1);

    match config.mode {
        ProcessorMode::Bypass => {
            dst[..n].copy_from_slice(&src[..n]);
        }
        ProcessorMode::BiquadLowpass => {
            let coeffs = biquad_lowpass_coeffs(cutoff, sample_rate, q);
            let mut state = [0.0f32; 4];
            let mut inst = BiquadCascadeInstanceF32::init(1, &coeffs, &mut state);
            biquad_cascade_df1_f32(&mut inst, &src[..n], &mut dst[..n]);
        }
        ProcessorMode::BiquadHighpass => {
            let coeffs = biquad_highpass_coeffs(cutoff, sample_rate, q);
            let mut state = [0.0f32; 4];
            let mut inst = BiquadCascadeInstanceF32::init(1, &coeffs, &mut state);
            biquad_cascade_df1_f32(&mut inst, &src[..n], &mut dst[..n]);
        }
        ProcessorMode::BiquadBandpass => {
            let coeffs = biquad_bandpass_coeffs(cutoff, sample_rate, q);
            let mut state = [0.0f32; 4];
            let mut inst = BiquadCascadeInstanceF32::init(1, &coeffs, &mut state);
            biquad_cascade_df1_f32(&mut inst, &src[..n], &mut dst[..n]);
        }
        ProcessorMode::BiquadNotch => {
            let coeffs = biquad_notch_coeffs(cutoff, sample_rate, q);
            let mut state = [0.0f32; 4];
            let mut inst = BiquadCascadeInstanceF32::init(1, &coeffs, &mut state);
            biquad_cascade_df1_f32(&mut inst, &src[..n], &mut dst[..n]);
        }
        ProcessorMode::BiquadPeaking => {
            let coeffs = biquad_peaking_coeffs(cutoff, sample_rate, q, config.gain_db);
            let mut state = [0.0f32; 4];
            let mut inst = BiquadCascadeInstanceF32::init(1, &coeffs, &mut state);
            biquad_cascade_df1_f32(&mut inst, &src[..n], &mut dst[..n]);
        }
        ProcessorMode::SvfLowpass => {
            let mut svf = StateVariableFilter::new(sample_rate);
            svf.set_cutoff(cutoff);
            svf.set_resonance(q);
            svf.set_drive(config.drive);
            for i in 0..n {
                svf.process(src[i]);
                dst[i] = svf.low();
            }
        }
        ProcessorMode::SvfHighpass => {
            let mut svf = StateVariableFilter::new(sample_rate);
            svf.set_cutoff(cutoff);
            svf.set_resonance(q);
            svf.set_drive(config.drive);
            for i in 0..n {
                svf.process(src[i]);
                dst[i] = svf.high();
            }
        }
        ProcessorMode::SvfBandpass => {
            let mut svf = StateVariableFilter::new(sample_rate);
            svf.set_cutoff(cutoff);
            svf.set_resonance(q);
            svf.set_drive(config.drive);
            for i in 0..n {
                svf.process(src[i]);
                dst[i] = svf.band();
            }
        }
        ProcessorMode::SvfNotch => {
            let mut svf = StateVariableFilter::new(sample_rate);
            svf.set_cutoff(cutoff);
            svf.set_resonance(q);
            svf.set_drive(config.drive);
            for i in 0..n {
                svf.process(src[i]);
                dst[i] = svf.notch();
            }
        }
        ProcessorMode::SvfPeak => {
            let mut svf = StateVariableFilter::new(sample_rate);
            svf.set_cutoff(cutoff);
            svf.set_resonance(q);
            svf.set_drive(config.drive);
            for i in 0..n {
                svf.process(src[i]);
                dst[i] = svf.peak();
            }
        }
        ProcessorMode::FixedPointQ15 => {
            // Emulate fixed-point quantization: round to Q15 (i16) and back to f32
            for i in 0..n {
                let q_sample = q15::saturating_from_num(src[i]);
                dst[i] = q_sample.to_num::<f32>();
            }
        }
    }

    if config.invert {
        for s in dst[..n].iter_mut() {
            *s = -*s;
        }
    }
}

fn compute_power_spectrum_db(time_signal: &[f32], out_mag_db: &mut [f32]) {
    let n = time_signal.len();
    if n < 2 || (n & (n - 1)) != 0 {
        return;
    }

    let mut windowed = vec![0.0f32; n];
    windowed.copy_from_slice(time_signal);
    let mut win_coeffs = vec![0.0f32; n];
    hanning_f32(&mut win_coeffs);
    for i in 0..n {
        windowed[i] *= win_coeffs[i];
    }

    let mut c_data = vec![0.0f32; 2 * n];
    for i in 0..n {
        c_data[2 * i] = windowed[i];
        c_data[2 * i + 1] = 0.0;
    }
    embedded_dsp::transform::cfft_f32(&mut c_data, n, 0, 1);

    let num_bins = out_mag_db.len().min(n / 2);
    let norm = 2.0 / (n as f32);

    for i in 0..num_bins {
        let re = c_data[2 * i];
        let im = c_data[2 * i + 1];
        let mag = (re * re + im * im).sqrt() * norm;
        let db = if mag > 1e-6 {
            20.0 * mag.log10()
        } else {
            -120.0
        };
        out_mag_db[i] = db.clamp(-120.0, 20.0);
    }
}

fn compute_peak_and_rms(signal: &[f32]) -> (f32, f32) {
    if signal.is_empty() {
        return (0.0, 0.0);
    }

    let mut peak = 0.0f32;
    let mut sum_sq = 0.0f32;

    for &s in signal {
        let a = s.abs();
        if a > peak {
            peak = a;
        }
        sum_sq += s * s;
    }

    let rms = (sum_sq / (signal.len() as f32)).sqrt();
    (peak, rms)
}
