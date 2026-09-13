//! AM, FM, and Hilbert-based SSB modem tests.

use embedded_dsp::modem::{AmDsb, FmDemod, FmMod, SsbDemod, SsbMod, SsbSideband};
use embedded_dsp::transform::{HilbertTransformF32, hilbert_fir_design_f32};
use embedded_dsp::types::Status;

#[test]
fn fm_mod_demod_recovers_tone() {
    assert!(FmMod::new(0.0).is_err());
    assert!(FmDemod::new(-0.1).is_err());
    let kf = 0.05;
    let mut tx = FmMod::new(kf).unwrap();
    let mut rx = FmDemod::new(kf).unwrap();
    assert!((tx.kf() - kf).abs() < 1e-9);
    let mut err = 0.0f32;
    let mut n = 0;
    for i in 0..200 {
        let m = (2.0 * core::f32::consts::PI * 0.01 * i as f32).sin() * 0.4;
        let y = rx.demodulate(tx.modulate(m));
        if i > 4 {
            err += (y - m).abs();
            n += 1;
        }
    }
    assert!(err / (n as f32) < 0.03);
    tx.reset();
    rx.reset();
}

#[test]
fn am_dsb_envelope_roundtrip() {
    assert!(AmDsb::new(0.0, false).is_err());
    let am = AmDsb::new(0.5, false).unwrap();
    assert!(!am.suppressed_carrier());
    assert!((am.mod_index() - 0.5).abs() < 1e-9);
    for i in 0..32 {
        let m = (i as f32 / 16.0) - 1.0;
        let d = am.demodulate_envelope(am.modulate(m));
        assert!((d - m).abs() < 1e-5);
    }

    let suppressed = AmDsb::new(0.8, true).unwrap();
    let d = suppressed.demodulate_envelope(suppressed.modulate(-0.5));
    assert!((d - 0.5).abs() < 1e-5);
}

#[test]
fn ssb_usb_roundtrip_lsb_cancelled() {
    const N: usize = 31;
    let mut h = [0.0f32; N];
    assert_eq!(hilbert_fir_design_f32(&mut h), Status::Success);

    let delay = {
        let mut st_tx = [0.0f32; N];
        let mut st_rx = [0.0f32; N];
        let mut i_delay = [0.0f32; N];
        let ht_tx = HilbertTransformF32::new(&h, &mut st_tx).unwrap();
        let ht_rx = HilbertTransformF32::new(&h, &mut st_rx).unwrap();
        let mut tx = SsbMod::new(ht_tx, SsbSideband::Usb, 0.7, true).unwrap();
        let mut rx = SsbDemod::new(ht_rx, &mut i_delay, SsbSideband::Usb, 0.7).unwrap();
        let delay = tx.group_delay() + rx.group_delay();

        let mut input = [0.0f32; 128];
        let mut output = [0.0f32; 128];
        for i in 0..128 {
            input[i] = (2.0 * core::f32::consts::PI * 0.07 * i as f32).sin() * 0.5;
            output[i] = rx.demodulate(tx.modulate(input[i]));
        }
        let mut err = 0.0;
        let mut n = 0;
        for i in (delay + 4)..120 {
            err += (output[i] - input[i - delay]).abs();
            n += 1;
        }
        assert!(err / (n as f32) < 0.12);
        delay
    };

    {
        let mut st_tx = [0.0f32; N];
        let mut st_lsb = [0.0f32; N];
        let mut i_delay = [0.0f32; N];
        let ht_tx = HilbertTransformF32::new(&h, &mut st_tx).unwrap();
        let ht_lsb = HilbertTransformF32::new(&h, &mut st_lsb).unwrap();
        let mut tx = SsbMod::new(ht_tx, SsbSideband::Usb, 0.7, true).unwrap();
        let mut rx_lsb = SsbDemod::new(ht_lsb, &mut i_delay, SsbSideband::Lsb, 0.7).unwrap();
        let mut leak = 0.0;
        let mut n = 0;
        for i in 0..128 {
            let m = (2.0 * core::f32::consts::PI * 0.07 * i as f32).sin() * 0.5;
            let d = rx_lsb.demodulate(tx.modulate(m));
            if i > delay + 8 {
                leak += d.abs();
                n += 1;
            }
        }
        assert!(leak / (n as f32) < 0.08);
        tx.reset();
        rx_lsb.reset();
        assert_eq!(tx.group_delay(), (N - 1) / 2);
    }

    let mut st = [0.0f32; N];
    let mut short = [0.0f32; 3];
    let ht = HilbertTransformF32::new(&h, &mut st).unwrap();
    assert_eq!(
        SsbDemod::new(ht, &mut short, SsbSideband::Usb, 0.7).err(),
        Some(Status::LengthError)
    );

    let mut st2 = [0.0f32; N];
    let ht2 = HilbertTransformF32::new(&h, &mut st2).unwrap();
    assert_eq!(
        SsbMod::new(ht2, SsbSideband::Lsb, 0.0, false).err(),
        Some(Status::ArgumentError)
    );

    let mut st3 = [0.0f32; N];
    let mut i_delay = [0.0f32; N];
    let ht3 = HilbertTransformF32::new(&h, &mut st3).unwrap();
    assert_eq!(
        SsbDemod::new(ht3, &mut i_delay, SsbSideband::Usb, 0.0).err(),
        Some(Status::ArgumentError)
    );

    let mut st_lsb = [0.0f32; N];
    let ht_lsb = HilbertTransformF32::new(&h, &mut st_lsb).unwrap();
    let mut lsb = SsbMod::new(ht_lsb, SsbSideband::Lsb, 0.5, false).unwrap();
    let y = lsb.modulate(0.2);
    assert!(y.real.is_finite() && y.imag.is_finite());
    // Carrier-on DSB-style offset: real part should sit near +1 after the Hilbert delay.
    let mut last = y;
    for _ in 0..N {
        last = lsb.modulate(0.0);
    }
    assert!((last.real - 1.0).abs() < 0.25);
}
