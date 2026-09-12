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
