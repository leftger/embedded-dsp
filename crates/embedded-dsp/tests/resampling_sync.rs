//! Arbitrary-rate polyphase resampling and Gardner timing-recovery tests.

use embedded_dsp::resampling::{GardnerSymbolSync, PolyphaseResampF32};

#[test]
fn polyphase_resamp_constant_and_errors() {
    assert!(PolyphaseResampF32::<4, 1>::from_taps(&[1.0; 3], 1.0).is_err());
    assert!(PolyphaseResampF32::<4, 1>::from_taps(&[1.0; 4], 0.0).is_err());
    assert!(PolyphaseResampF32::<1, 4>::from_taps(&[1.0; 4], 1.0).is_err());

    let mut r = PolyphaseResampF32::<4, 1>::from_taps(&[1.0; 4], 0.5).unwrap();
    let src = [0.7f32; 8];
    let mut dst = [0.0f32; 20];
    let n = r.process_block(&src, &mut dst);
    assert!(n >= 8);
    for sample in dst.iter().take(n.min(16)).skip(2) {
        assert!((*sample - 0.7).abs() < 1e-4);
    }

    r.reset();
    let mut out = [0.0f32; 4];
    assert!(r.push(0.5, &mut out) >= 1);
}

#[test]
fn gardner_sync_emits_symbols() {
    assert!(GardnerSymbolSync::new(1.5, 0.01).is_err());
    assert!(GardnerSymbolSync::new(4.0, -0.01).is_err());
    let mut sync = GardnerSymbolSync::new(4.0, 0.02).unwrap();
    let mut n_sym = 0;
    for n in 0..400 {
        let t = n as f32;
        let bits = if ((n / 4) % 2) == 0 { 1.0 } else { -1.0 };
        let pulse = 0.5 + 0.5 * (core::f32::consts::PI * (t - 1.7) / 4.0).cos();
        if sync.push(bits * pulse.max(0.0)).is_some() {
            n_sym += 1;
        }
    }
    assert!(n_sym > 40);
    sync.reset();
    assert!(sync.push(0.0).is_none());
}
