use embedded_dsp::filtering::{
    Biquad, BiquadClamp, BiquadFixed, DirectForm1, DirectForm1NoiseShaped, Dsm, XorShift32,
};
use embedded_dsp::pipeline::{Lanes, Pair, Process, Split, SplitProcess};
#[cfg(feature = "bytemuck")]
use embedded_dsp::types::Complex;
use quickcheck_macros::quickcheck;

#[test]
fn test_split_process_biquad_clamp_anti_windup() {
    // Lowpass/integrator biquad coefficients
    let coeff = Biquad::<f32>::new(0.1, 0.2, 0.1, 0.9, -0.2);
    let clamp = BiquadClamp::new(coeff, -2.0f32, 2.0f32, 0.0f32);
    let mut state = DirectForm1::<f32>::new();

    // Push large inputs to force saturation
    for _ in 0..50 {
        let y = clamp.process_df1(&mut state, 1000.0f32);
        assert!(y <= 2.0 + 1e-6);
        assert!(y >= -2.0 - 1e-6);
    }

    // Output must stay clamped at upper bound
    assert!((state.xy[2] - 2.0).abs() < 1e-4);

    // Instant recovery: once input returns to 0.0, the filter recovers in 3 cycles
    // (whereas an unwound integrator would take hundreds of cycles to un-integrate).
    let _ = clamp.process_df1(&mut state, 0.0f32);
    let _ = clamp.process_df1(&mut state, 0.0f32);
    let y_rec = clamp.process_df1(&mut state, 0.0f32);
    assert!(y_rec < 2.0, "Expected y_rec < 2.0, got {}", y_rec);
}

#[test]
fn test_split_process_lanes_multichannel() {
    let coeff = Biquad::<f32>::new(0.2, 0.1, 0.05, 0.5, -0.1);
    let clamp = BiquadClamp::new(coeff, -10.0f32, 10.0f32, 0.0f32);
    let lanes = Lanes::new(clamp);

    let mut states = [DirectForm1::<f32>::new(); 3];
    let inputs = [1.0f32, 2.0f32, 3.0f32];

    let outputs = lanes.process(&mut states, inputs);

    // Verify against individual single-channel processing
    let mut single0 = DirectForm1::<f32>::new();
    let mut single1 = DirectForm1::<f32>::new();
    let mut single2 = DirectForm1::<f32>::new();

    let expected0 = clamp.process_df1(&mut single0, 1.0f32);
    let expected1 = clamp.process_df1(&mut single1, 2.0f32);
    let expected2 = clamp.process_df1(&mut single2, 3.0f32);

    assert!((outputs[0] - expected0).abs() < 1e-6);
    assert!((outputs[1] - expected1).abs() < 1e-6);
    assert!((outputs[2] - expected2).abs() < 1e-6);
}

#[test]
fn test_pair_and_split_pipeline() {
    let coeff = Biquad::<f32>::new(0.5, 0.0, 0.0, 0.0, 0.0);
    let pair = Pair::new(coeff, coeff);

    let mut states = (DirectForm1::<f32>::new(), DirectForm1::<f32>::new());
    let out = pair.process(&mut states, [4.0f32, 8.0f32]);
    assert_eq!(out, [2.0, 4.0]);

    // Split pipeline wrapper
    let mut split_pipeline = Split::new(clamp_node(), DirectForm1::<f32>::new());
    let y = split_pipeline.process(10.0f32);
    assert_eq!(y, 5.0); // clamped to 5.0
}

fn clamp_node() -> BiquadClamp<f32> {
    BiquadClamp::new(Biquad::new(1.0, 0.0, 0.0, 0.0, 0.0), -5.0, 5.0, 0.0)
}

#[test]
fn test_dsm_delta_sigma_mean() {
    let mut dsm = Dsm::<3>::new();
    // 0x4000_0000 is 1/4 of full scale (2^32)
    let x: u32 = 0x4000_0000;
    let n = 1 << 16;
    let mut sum: i64 = 0;
    for _ in 0..n {
        sum += dsm.process_sample(x) as i64;
    }
    let mean = sum as f64 / n as f64;
    // Expected average is 0.25, with high precision over 65536 samples
    assert!((mean - 0.25).abs() < 0.01, "DSM mean was {}", mean);
}

#[test]
fn test_xorshift32_and_dither() {
    let mut rng = XorShift32::new(42);
    let mut dither_sum = 0.0f32;
    let n = 10000;
    for _ in 0..n {
        let d = rng.tpdf_dither_f32();
        assert!((-1.0..=1.0).contains(&d));
        dither_sum += d;
    }
    let avg = dither_sum / (n as f32);
    assert!(avg.abs() < 0.05, "TPDF dither average was {}", avg);
}

#[test]
fn test_biquad_fixed_noise_shaped() {
    // Q30 coefficients: unity gain with b0 = 1 << 30
    let ba = [1 << 30, 0, 0, 0, 0];
    let filter = BiquadFixed::<30>::new(ba, -1000, 1000, 0);
    let mut state = DirectForm1NoiseShaped::new();

    let out = filter.process_noise_shaped(&mut state, 500);
    assert_eq!(out, 500);

    // Verify clamping at limits
    let clamped_out = filter.process_noise_shaped(&mut state, 2000);
    assert_eq!(clamped_out, 1000);
}

#[cfg(feature = "bytemuck")]
#[test]
fn test_bytemuck_zeroable_and_pod_complex() {
    let c = Complex::new(1.0f32, -2.0f32);
    let bytes: &[u8] = bytemuck::bytes_of(&c);
    assert_eq!(bytes.len(), 8);

    let decoded: &Complex<f32> = bytemuck::from_bytes(bytes);
    assert_eq!(decoded.real, 1.0f32);
    assert_eq!(decoded.imag, -2.0f32);
}

#[quickcheck]
fn prop_biquad_clamp_always_within_bounds(raw_input: i16) -> bool {
    let x = (raw_input as f32) * 0.1;
    let coeff = Biquad::<f32>::new(0.5, 0.2, 0.1, 0.4, -0.1);
    let clamp = BiquadClamp::new(coeff, -10.0f32, 10.0f32, 0.0f32);
    let mut state = DirectForm1::<f32>::new();

    let y = clamp.process_df1(&mut state, x);
    (-10.0..=10.0).contains(&y)
}

#[quickcheck]
fn prop_lanes_matches_independent_channels(x0: i16, x1: i16) -> bool {
    let in0 = x0 as f32;
    let in1 = x1 as f32;

    let coeff = Biquad::<f32>::new(0.3, 0.1, 0.0, 0.1, 0.0);
    let clamp = BiquadClamp::new(coeff, -100.0f32, 100.0f32, 0.0f32);
    let lanes = Lanes::new(clamp);

    let mut state_lanes = [DirectForm1::<f32>::new(); 2];
    let out_lanes = lanes.process(&mut state_lanes, [in0, in1]);

    let mut s0 = DirectForm1::<f32>::new();
    let mut s1 = DirectForm1::<f32>::new();
    let out0 = clamp.process_df1(&mut s0, in0);
    let out1 = clamp.process_df1(&mut s1, in1);

    (out_lanes[0] - out0).abs() < 1e-5 && (out_lanes[1] - out1).abs() < 1e-5
}
