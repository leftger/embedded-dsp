use embedded_dsp::*;

#[test]
fn test_polyblep_oscillator_waveforms() {
    let waveforms = [
        PolyBlepWaveform::Sine,
        PolyBlepWaveform::Sawtooth,
        PolyBlepWaveform::Square,
        PolyBlepWaveform::Triangle,
    ];

    for &wf in &waveforms {
        let mut osc = PolyBlepOscillator::new(44100.0, 440.0, wf);
        assert_eq!(osc.frequency(), 440.0);

        osc.set_frequency(880.0);
        assert_eq!(osc.frequency(), 880.0);

        osc.reset_phase();
        osc.set_waveform(wf);

        let mut samples = [0.0f32; 100];
        for s in samples.iter_mut() {
            *s = osc.next_sample();
            assert!(s.is_finite());
            assert!(*s >= -1.0 && *s <= 1.0);
        }
    }
}

#[test]
fn test_chirp_sweep_exponential_with_equal_start_and_end_freq_uses_the_linear_fallback() {
    // start_freq == end_freq makes the exponential rate ~0, exercising next_sample's
    // `rate.abs() < 1e-6` fallback (a pure tone, since there's nothing to sweep).
    let mut sweep = ChirpSweep::new(44100.0, 440.0, 440.0, 0.1, true);
    for _ in 0..50 {
        let s = sweep.next_sample();
        assert!(s.is_finite());
        assert!((-1.0..=1.0).contains(&s));
    }
}

#[test]
fn test_polyblep_oscillator_zero_frequency_skips_the_blep_correction() {
    // A zero frequency gives phase_step = 0, so poly_blep's `dt <= 0.0` early return runs (no
    // discontinuity correction needed since the phase never advances).
    for wf in [
        PolyBlepWaveform::Sawtooth,
        PolyBlepWaveform::Square,
        PolyBlepWaveform::Triangle,
    ] {
        let mut osc = PolyBlepOscillator::new(44100.0, 0.0, wf);
        for _ in 0..8 {
            let s = osc.next_sample();
            assert!(s.is_finite());
        }
    }
}

#[test]
fn test_white_and_pink_noise_generators() {
    let mut white = WhiteNoise::new(12345);
    let mut pink = KellettPinkNoise::new(54321);

    for _ in 0..500 {
        let w = white.next_sample();
        let p = pink.next_sample();

        assert!(w.is_finite());
        assert!(p.is_finite());
        assert!((-1.0..=1.0).contains(&w));
        assert!((-1.0..=1.0).contains(&p));
    }
}

#[test]
fn test_chirp_sweep_linear_and_exponential() {
    let mut linear = ChirpSweep::new(44100.0, 100.0, 1000.0, 0.1, false);
    let mut exp_sweep = ChirpSweep::new(44100.0, 100.0, 1000.0, 0.1, true);

    linear.reset();
    exp_sweep.reset();

    for _ in 0..200 {
        let l = linear.next_sample();
        let e = exp_sweep.next_sample();

        assert!(l.is_finite());
        assert!(e.is_finite());
        assert!((-1.0..=1.0).contains(&l));
        assert!((-1.0..=1.0).contains(&e));
    }
}

#[test]
fn synthesis_generators_compose_through_split_process() {
    // A generator has no real input, so `()` stands in for it: each of these reaches
    // `Process`/`SplitProcess` (via the `()` input) but not `DspNode`, which requires the input and
    // output types to match. Verify the trait path agrees with `next_sample` bit-for-bit.
    let mut osc_trait = PolyBlepOscillator::new(44100.0, 440.0, PolyBlepWaveform::Sawtooth);
    let mut osc_inherent = PolyBlepOscillator::new(44100.0, 440.0, PolyBlepWaveform::Sawtooth);
    let mut white_trait = WhiteNoise::new(0xDEAD_BEEF);
    let mut white_inherent = WhiteNoise::new(0xDEAD_BEEF);
    let mut pink_trait = KellettPinkNoise::new(0xDEAD_BEEF);
    let mut pink_inherent = KellettPinkNoise::new(0xDEAD_BEEF);
    let mut chirp_trait = ChirpSweep::new(44100.0, 100.0, 1000.0, 0.1, true);
    let mut chirp_inherent = ChirpSweep::new(44100.0, 100.0, 1000.0, 0.1, true);

    for _ in 0..200 {
        assert_eq!(
            osc_trait.process_with_state(&mut (), ()),
            osc_inherent.next_sample()
        );
        assert_eq!(
            white_trait.process_with_state(&mut (), ()),
            white_inherent.next_sample()
        );
        assert_eq!(
            pink_trait.process_with_state(&mut (), ()),
            pink_inherent.next_sample()
        );
        assert_eq!(
            chirp_trait.process_with_state(&mut (), ()),
            chirp_inherent.next_sample()
        );
    }
}
