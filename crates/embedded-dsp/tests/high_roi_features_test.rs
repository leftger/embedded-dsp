use core::f32::consts::PI;
use embedded_dsp::controller::PidBuilder;
use embedded_dsp::fast_math::{Unwrapper, atan2_i32, cossin, cossin_f32, fast_atan2_f32};
use embedded_dsp::filtering::{DirectForm1, Lockin, LockinAmplifier, NormalForm, NormalFormState};
use embedded_dsp::pipeline::SplitProcess;
use embedded_dsp::resampling::{HbfDec, HbfDecCascade, HbfInt};
use embedded_dsp::synthesis::{Accu, AccuOsc, Sweep};
use embedded_dsp::types::Complex;

#[test]
fn test_cossin_fixed_and_float() {
    // Zero phase
    let (c0, s0) = cossin(0);
    assert!(c0 > 2_140_000_000); // Close to i32::MAX (2147483647)
    assert!(s0.abs() < 100_000);

    let (cf0, sf0) = cossin_f32(0.0);
    assert!((cf0 - 1.0).abs() < 1e-3);
    assert!(sf0.abs() < 1e-3);

    // PI / 2
    let (c_pi2, s_pi2) = cossin_f32(PI / 2.0);
    assert!(c_pi2.abs() < 2e-3, "cos(pi/2) was {c_pi2}");
    assert!((s_pi2 - 1.0).abs() < 2e-3, "sin(pi/2) was {s_pi2}");

    // PI
    let (c_pi, s_pi) = cossin_f32(PI);
    assert!((c_pi - (-1.0)).abs() < 2e-3, "cos(pi) was {c_pi}");
    assert!(s_pi.abs() < 2e-3, "sin(pi) was {s_pi}");

    // Sweep 100 points across [-PI, PI] and verify against standard math
    for i in 0..=100 {
        let angle = -PI + (2.0 * PI * i as f32 / 100.0);
        let (c, s) = cossin_f32(angle);
        let expected_c = angle.cos();
        let expected_s = angle.sin();
        assert!(
            (c - expected_c).abs() < 3e-3,
            "Angle {angle}: cos error {} (got {c}, expected {expected_c})",
            (c - expected_c).abs()
        );
        assert!(
            (s - expected_s).abs() < 3e-3,
            "Angle {angle}: sin error {} (got {s}, expected {expected_s})",
            (s - expected_s).abs()
        );
    }
}

#[test]
fn test_atan2_fixed_and_float() {
    // Quadrant axes
    let a_pos_x = fast_atan2_f32(0.0, 1.0);
    assert!(a_pos_x.abs() < 2e-3);

    let a_pos_y = fast_atan2_f32(1.0, 0.0);
    assert!((a_pos_y - PI / 2.0).abs() < 2e-3);

    let a_neg_y = fast_atan2_f32(-1.0, 0.0);
    assert!((a_neg_y - (-PI / 2.0)).abs() < 2e-3);

    // Diagonals
    let a_diag1 = fast_atan2_f32(1.0, 1.0);
    assert!((a_diag1 - PI / 4.0).abs() < 2e-3);

    let a_diag2 = fast_atan2_f32(1.0, -1.0);
    assert!((a_diag2 - 3.0 * PI / 4.0).abs() < 2e-3);

    // Integer atan2_i32
    let a_int = atan2_i32(1_000_000, 1_000_000);
    // PI/4 in i32 is (1 << 29)
    let expected_q31 = 1 << 29;
    assert!((a_int - expected_q31).abs() < 500_000);
}

#[test]
fn test_phase_unwrapper() {
    let mut unwrapper = Unwrapper::new();
    assert_eq!(unwrapper.turns(), 0);

    // Step positive phase: 0.0 -> 0.4pi -> 0.8pi -> -0.8pi (wrapped over +pi boundary)
    let p0 = 0i32;
    let p1 = (0.4 * 2147483648.0) as i32;
    let p2 = (0.8 * 2147483648.0) as i32;
    let p3 = (-0.8 * 2147483648.0) as i32;

    let u0 = unwrapper.update(p0);
    let u1 = unwrapper.update(p1);
    let u2 = unwrapper.update(p2);
    let u3 = unwrapper.update(p3);

    assert!(u1 > u0);
    assert!(u2 > u1);
    assert!(
        u3 > u2,
        "Unwrapped phase should monotonically increase across boundary jump: u2={u2}, u3={u3}"
    );
    assert_eq!(unwrapper.turns(), 1);
}

#[test]
fn test_half_band_decimator_and_interpolator() {
    let coeffs_dec = [
        -0.00086943,
        0.00577837,
        -0.02201674,
        0.06357869,
        -0.16627679,
        0.61979312,
    ];
    let mut dec = HbfDec::new(coeffs_dec);

    let input = [1.0f32; 64];
    let mut output_dec = [0.0f32; 32];
    dec.process(&input, &mut output_dec);

    assert!(output_dec[20..32].iter().all(|&y| (y - 1.0).abs() < 0.05));

    // Test Interpolator
    let mut interp = HbfInt::new(coeffs_dec);
    let mut output_interp = [0.0f32; 64];
    interp.process(&output_dec, &mut output_interp);

    assert!(output_interp[40..64].iter().all(|&y| (y - 1.0).abs() < 0.1));

    // Test Cascade
    let mut cascade = HbfDecCascade::<3>::new();
    let src = [1.0f32; 256];
    let mut dst = [0.0f32; 32];
    cascade.process(&src, &mut dst);
    assert!(dst[20..32].iter().all(|&y| (y - 1.0).abs() < 0.1));
}

#[test]
fn test_pid_builder_biquad_synthesis() {
    let ts = 0.001; // 1 kHz sample rate
    let pid = PidBuilder::new()
        .kp(2.5)
        .ki(10.0)
        .output_limits(-15.0, 15.0);

    let biquad_controller = pid.build(ts);
    let mut state = DirectForm1::<f32>::new();

    // Feed a step error and verify that clamping and stability work
    let mut out = 0.0;
    for _ in 0..50 {
        out = biquad_controller.process_df1(&mut state, 10.0);
    }
    assert_eq!(
        out, 15.0,
        "Anti-windup limit should saturate at max clamp +15.0"
    );

    // Reverse error to negative step
    for _ in 0..100 {
        out = biquad_controller.process_df1(&mut state, -10.0);
    }
    assert_eq!(
        out, -15.0,
        "Anti-windup limit should saturate at min clamp -15.0"
    );

    // Test derivative gain limit roll-off
    let pid_with_d = PidBuilder::new().kp(1.0).kd(0.1).limit_d(5.0).build(ts);
    // At high frequencies (Nyquist, z = -1), gain should not exceed limit_d
    let b = pid_with_d.coeff.ba;
    let h_nyquist = (b[0] - b[1] + b[2]) / (1.0 + b[3] - b[4]);
    assert!(
        h_nyquist.abs() <= 6.01,
        "High-frequency derivative gain clamped: {h_nyquist}"
    );
}

#[test]
fn test_lockin_amplifier() {
    let sample_rate = 10_000.0;
    let carrier_hz = 500.0;
    let decay = 0.98; // Lowpass cutoff << 500 Hz

    let mut lockin = LockinAmplifier::new(carrier_hz, sample_rate, decay);

    // Generate in-phase carrier: x[n] = 2.0 * cos(2*pi*500*t)
    // Demodulation with cos gives: 2.0 * cos^2(wt) = 1.0 + cos(2wt)
    // Lowpass filter removes cos(2wt) (1000 Hz) leaving DC = 1.0 in the I channel.
    // Demodulation with sin gives: 2.0 * cos(wt) * sin(wt) = sin(2wt) -> filtered to 0.0 in Q channel.
    let mut last_iq = Complex::new(0.0f32, 0.0f32);
    for n in 0..3000 {
        let t = n as f32 / sample_rate;
        let x = 2.0 * (2.0 * PI * carrier_hz * t).cos();
        last_iq = lockin.process(x);
    }

    assert!(
        (last_iq.real - 1.0).abs() < 0.05,
        "I channel should converge near 1.0, got {}",
        last_iq.real
    );
    assert!(
        last_iq.imag.abs() < 0.05,
        "Q channel should converge near 0.0, got {}",
        last_iq.imag
    );

    // Also test SplitProcess Lockin<C>
    let biquad_lp = PidBuilder::new().kp(0.05).build(0.001);
    let lockin_filter = Lockin::new(biquad_lp);
    let mut filter_states = [DirectForm1::<f32>::new(), DirectForm1::<f32>::new()];
    let mixed = lockin_filter.process(&mut filter_states, (1.0f32, Complex::new(0.8f32, 0.6f32)));
    assert!(mixed.real > 0.0);
}

#[test]
fn test_exponential_sweep_and_osc() {
    let stop_nyquist = 0.25;
    let harmonics = 500.0;
    let cycles = 2.0;

    let sweep = Sweep::fit(stop_nyquist, harmonics, cycles).expect("Valid sweep fit");
    assert!(sweep.rate > 0);
    assert!(sweep.octave() > 0.0);
    assert!(sweep.decade() > 0.0);
    assert!((sweep.cycles() - 2.0).abs() < 0.01);

    // Test AccuOsc iteration
    let osc = AccuOsc::new(sweep);
    let samples: Vec<_> = osc.take(100).collect();
    assert_eq!(samples.len(), 100);
    for s in samples {
        assert!(s.real != 0 || s.imag != 0);
    }
}

#[test]
fn test_normal_form_oscillator_quadrature_and_period() {
    // Oscillator at 10% of the sample rate -> period of 10 samples.
    let nco = NormalForm::oscillator(0.1);
    let mut state = NormalFormState::default();

    // Kick with a unit impulse.
    let (first_re, first_im) = nco.process_quadrature(&mut state, 1.0);
    assert!((first_re - 1.0).abs() < 1e-6, "impulse starts at re=1");
    assert!(first_im.abs() < 1e-6, "impulse starts at im=0");

    let mut first_peak = None;
    let mut last_peak = None;
    let mut peak_spacing_ok = true;

    for n in 1..=100 {
        let (re, im) = nco.process_quadrature(&mut state, 0.0);
        let amp = (re * re + im * im).sqrt();
        assert!(
            (amp - 1.0).abs() < 2e-4,
            "oscillator amplitude must stay on the unit circle (n={n}, amp={amp})"
        );
        if (re - 1.0).abs() < 1e-3 {
            if let Some(prev) = last_peak {
                if n - prev != 10 {
                    peak_spacing_ok = false;
                }
            } else {
                first_peak = Some(n);
            }
            last_peak = Some(n);
        }
    }

    // Full rotation after 10 samples.
    assert_eq!(first_peak, Some(10), "cosine peaks again after one period");
    assert!(peak_spacing_ok, "peaks must repeat every 10 samples");
}

#[test]
fn test_normal_form_from_ba_pole_extraction() {
    // Denominator: (z - (0.5 + 0.5j))(z - (0.5 - 0.5j)) = z^2 - z + 0.5
    let ba = [[1.0f32, 0.0, 0.0], [1.0, -1.0, 0.5]];
    let nf = NormalForm::from_ba(&ba);
    assert!(
        (nf.p.re() - 0.5).abs() < 1e-5,
        "p.re was {p_re}",
        p_re = nf.p.re()
    );
    assert!(
        (nf.p.im() - 0.5).abs() < 1e-5,
        "p.im was {p_im}",
        p_im = nf.p.im()
    );

    // Impulse response must exactly match the biquad transfer function:
    // H(z) = 1 / (1 - z^-1 + 0.5 z^-2)
    // h[0] = 1, h[1] = 1, h[2] = 0.5, h[3] = 0, h[4] = -0.25, ...
    let mut state = NormalFormState::default();
    let h0 = nf.process(&mut state, 1.0);
    let h1 = nf.process(&mut state, 0.0);
    let h2 = nf.process(&mut state, 0.0);
    let h3 = nf.process(&mut state, 0.0);
    let h4 = nf.process(&mut state, 0.0);
    assert!((h0 - 1.0).abs() < 1e-5);
    assert!((h1 - 1.0).abs() < 1e-5);
    assert!((h2 - 0.5).abs() < 1e-5);
    assert!(h3.abs() < 1e-5);
    assert!((h4 - (-0.25)).abs() < 1e-5);
}

#[test]
fn test_normal_form_bandpass_dc_rejection() {
    // Narrow bandpass at 0.25 Nyquist-normalized with Q = 10.
    let bp = NormalForm::bandpass(0.25, 10.0);
    let mut state = NormalFormState::default();

    // Feed a DC signal: numerator (1 - z^-2) guarantees zero DC gain.
    let mut out = 0.0;
    for _ in 0..2000 {
        out = bp.process(&mut state, 1.0);
    }
    assert!(out.abs() < 1e-3, "DC must be rejected, got {out}");
}

#[test]
fn test_accu_wrapping_and_algebra() {
    use core::num::Wrapping;

    // Same behavior as the idsp doctest: 127 + 127 wraps to -2 in i8.
    let mut acc = Accu::new(Wrapping(0i8), Wrapping(127));
    assert_eq!(acc.next(), Some(Wrapping(127)));
    assert_eq!(acc.next(), Some(Wrapping(-2)));

    // Scaling: multiply state and step by the same factor.
    let acc = Accu::new(Wrapping(2i32), Wrapping(3i32));
    let scaled = acc * Wrapping(4i32);
    assert_eq!(scaled.state, Wrapping(8));
    assert_eq!(scaled.step, Wrapping(12));

    // Composition: add and subtract accumulators elementwise.
    let a = Accu::new(Wrapping(1i32), Wrapping(2i32));
    let b = Accu::new(Wrapping(3i32), Wrapping(4i32));
    let sum = a + b;
    assert_eq!((sum.state, sum.step), (Wrapping(4), Wrapping(6)));
    let diff = b - a;
    assert_eq!((diff.state, diff.step), (Wrapping(2), Wrapping(2)));
}
