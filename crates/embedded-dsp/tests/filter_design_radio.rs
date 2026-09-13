//! Radio-oriented FIR and IIR filter-design tests.

use embedded_dsp::filter_design::{
    elliptic_lowpass_biquad, estimate_req_filter_len, firdes_gmsk_tx, firdes_kaiser, firdes_rc,
    firdes_rrc, kaiser_beta_as, pulse_shaping_len,
};
use embedded_dsp::types::Status;

fn mag_h_biquad(c: [f32; 5], f_norm: f32) -> f32 {
    let w = 2.0 * core::f32::consts::PI * f_norm;
    let (s, co) = (w.sin(), w.cos());
    let [b0, b1, b2, a1, a2] = c;
    let zr = co;
    let zi = -s;
    let z2r = co * co - s * s;
    let z2i = -2.0 * co * s;
    let num_r = b0 + b1 * zr + b2 * z2r;
    let num_i = b1 * zi + b2 * z2i;
    let den_r = 1.0 - a1 * zr - a2 * z2r;
    let den_i = -a1 * zi - a2 * z2i;
    let den2 = den_r * den_r + den_i * den_i;
    let hr = (num_r * den_r + num_i * den_i) / den2;
    let hi = (num_i * den_r - num_r * den_i) / den2;
    (hr * hr + hi * hi).sqrt()
}

#[test]
fn pulse_shaping_len_and_errors() {
    assert_eq!(pulse_shaping_len(4, 6), 49);
    let mut h = [0.0f32; 8];
    assert_eq!(firdes_rc(0, 3, 0.35, 0.0, &mut h), Status::ArgumentError);
    assert_eq!(firdes_rrc(4, 3, 1.1, 0.0, &mut h), Status::ArgumentError);
    assert_eq!(firdes_gmsk_tx(4, 3, 0.3, 0.0, &mut h), Status::LengthError);
    assert_eq!(firdes_rrc(4, 3, 0.3, 0.0, &mut h), Status::LengthError);
    assert_eq!(
        firdes_gmsk_tx(4, 3, 0.0, 0.0, &mut [0.0; 25]),
        Status::ArgumentError
    );
}

#[test]
fn raised_cosine_is_nyquist() {
    const SPS: usize = 4;
    const SPAN: usize = 6;
    let mut h = [0.0f32; pulse_shaping_len(SPS, SPAN)];
    assert_eq!(firdes_rc(SPS, SPAN, 0.35, 0.0, &mut h), Status::Success);
    let c = (h.len() - 1) / 2;
    for k in 1..=SPAN {
        assert!(h[c + k * SPS].abs() < 2e-3);
        assert!(h[c - k * SPS].abs() < 2e-3);
    }
    assert!(h[c] > 0.5);

    let mut h0 = [0.0f32; pulse_shaping_len(SPS, SPAN)];
    assert_eq!(firdes_rc(SPS, SPAN, 0.0, 0.0, &mut h0), Status::Success);
    assert!((h0[c] - 1.0).abs() < 1e-3);

    let mut hs = [0.0f32; pulse_shaping_len(4, 3)];
    assert_eq!(firdes_rc(4, 3, 0.25, 0.0, &mut hs), Status::Success);
    assert!(hs[20].is_finite());
}

#[test]
fn rrc_special_cases_and_beta_zero() {
    const SPS: usize = 4;
    const SPAN: usize = 4;
    let mut h = [0.0f32; pulse_shaping_len(SPS, SPAN)];
    assert_eq!(firdes_rrc(SPS, SPAN, 0.5, 0.0, &mut h), Status::Success);
    let c = (h.len() - 1) / 2;
    assert!(h[c] > 0.5);
    assert!(h[18].is_finite() && h[18].abs() > 0.0);

    let mut h0 = [0.0f32; pulse_shaping_len(SPS, SPAN)];
    assert_eq!(firdes_rrc(SPS, SPAN, 0.0, 0.0, &mut h0), Status::Success);
    assert!((h0[c] - 1.0).abs() < 1e-3);

    let mut hdt = [0.0f32; pulse_shaping_len(SPS, SPAN)];
    assert_eq!(firdes_rrc(SPS, SPAN, 0.35, 0.25, &mut hdt), Status::Success);
}

#[test]
fn gmsk_tx_is_nonnegative_and_peaked() {
    const SPS: usize = 4;
    const SPAN: usize = 4;
    let mut h = [0.0f32; pulse_shaping_len(SPS, SPAN)];
    assert_eq!(firdes_gmsk_tx(SPS, SPAN, 0.3, 0.0, &mut h), Status::Success);
    let c = (h.len() - 1) / 2;
    assert!(h.iter().all(|&x| x >= -1e-5));
    assert!(h[c] > h[0] && h[c] > h[h.len() - 1]);
    assert!(h.iter().sum::<f32>() > 0.0);
}

#[test]
fn elliptic_lowpass_biquad_shape() {
    assert!(elliptic_lowpass_biquad(0.0, 1.0, 40.0).is_err());
    assert!(elliptic_lowpass_biquad(0.1, 0.0, 40.0).is_err());
    assert!(elliptic_lowpass_biquad(0.1, 5.0, 4.0).is_err());
    let c = elliptic_lowpass_biquad(0.1, 1.0, 40.0).unwrap();
    let h0 = mag_h_biquad(c, 0.0);
    let hc = mag_h_biquad(c, 0.1);
    let hs = mag_h_biquad(c, 0.35);
    let ep = (10.0f32.powf(0.1) - 1.0).sqrt();
    let g0 = 1.0 / (1.0 + ep * ep).sqrt();
    assert!((h0 - g0).abs() < 0.05);
    assert!(hc > 0.3);
    assert!(hs < 0.15);
    assert!(c.iter().all(|x| x.is_finite()));
}

#[test]
fn kaiser_fir_design() {
    assert!(estimate_req_filter_len(0.0, 60.0).is_err());
    assert!(estimate_req_filter_len(0.1, 0.0).is_err());
    let n = estimate_req_filter_len(0.1, 60.0).unwrap();
    assert_eq!(n, ((60.0 - 7.95) / (14.26 * 0.1)) as usize);
    assert!(kaiser_beta_as(20.0).abs() < 1e-6);
    let b60 = kaiser_beta_as(60.0);
    assert!((b60 - 0.1102 * (60.0 - 8.7)).abs() < 1e-5);
    assert!(kaiser_beta_as(120.0) > kaiser_beta_as(100.0));

    let mut h = [0.0f32; 41];
    assert_eq!(firdes_kaiser(0.2, 60.0, 0.0, &mut h), Status::Success);
    let c = h.len() / 2;
    assert!((h[c] - 1.0).abs() < 1e-5);
    let sum: f32 = h.iter().sum();
    assert!((sum - 1.0 / (2.0 * 0.2)).abs() < 0.15);
    assert!(h[0].abs() < h[c]);
    assert_eq!(firdes_kaiser(0.0, 60.0, 0.0, &mut h), Status::ArgumentError);
    assert_eq!(firdes_kaiser(0.2, 60.0, 1.0, &mut h), Status::ArgumentError);
    assert_eq!(firdes_kaiser(0.2, 60.0, 0.0, &mut []), Status::LengthError);

    // Mid-ripple Kaiser β, negative As (abs), df at 0.5, and As too small for a positive length.
    let b30 = kaiser_beta_as(30.0);
    assert!(b30 > 0.0 && b30 < kaiser_beta_as(60.0));
    assert!((kaiser_beta_as(-60.0) - kaiser_beta_as(60.0)).abs() < 1e-6);
    assert!(estimate_req_filter_len(0.5, 60.0).unwrap() > 0);
    assert!(estimate_req_filter_len(0.1, 1.0).is_err());

    let mut one = [0.0f32; 1];
    assert_eq!(firdes_kaiser(0.2, 10.0, 0.0, &mut one), Status::Success);
    assert!((one[0] - 1.0).abs() < 1e-6);
}
