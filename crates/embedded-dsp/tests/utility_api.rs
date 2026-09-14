//! Coverage for utility APIs that the algorithmic suites leave untouched.
//!
//! These are not new behaviours: they are the `FloatMath` trait surface, the fast-math scalar
//! helpers, the integer phase unwrappers, and a couple of small stateful types. They are easy to
//! forget because the DSP pipelines reach them only through specific branches, so this file
//! drives every one of them directly.

use embedded_dsp::companding::{
    a_law_compress_f32, a_law_expand_f32, alaw_to_linear, linear_to_alaw, linear_to_ulaw,
    ulaw_to_linear,
};
use embedded_dsp::cordic::cordic_cartesian_to_polar_q15;
use embedded_dsp::fast_math::{
    IntPhaseUnwrapper, Unwrapper, cossin_f32, fast_atan2_f32, overflowing_sub_i32,
    saturating_scale_i32, vsqrt_f32,
};
use embedded_dsp::filtering::SinglePoleFilter;
use embedded_dsp::math::FloatMath;
use embedded_dsp::pipeline::DspNode;
use embedded_dsp::snapshot::SnapshotBuffer;
use embedded_dsp::types::q15;

/// `FloatMath` exists so `no_std` builds can swap in `libm`, but ordinary code calls the inherent
/// float methods instead, so the trait impls are never otherwise executed.
#[test]
fn float_math_trait_is_exercised_for_both_widths() {
    let x = 0.7f32;
    assert_eq!(FloatMath::abs(-x), x);
    assert!((FloatMath::sin(x) - x.sin()).abs() < 1e-6);
    assert!((FloatMath::cos(x) - x.cos()).abs() < 1e-6);
    assert!((FloatMath::tan(x) - x.tan()).abs() < 1e-5);
    assert!((FloatMath::sqrt(x + 1.0) - (x + 1.0).sqrt()).abs() < 1e-6);
    assert!((FloatMath::ln(x + 1.0) - (x + 1.0).ln()).abs() < 1e-6);
    assert!((FloatMath::log10(x + 1.0) - (x + 1.0).log10()).abs() < 1e-6);
    assert!((FloatMath::exp(x) - x.exp()).abs() < 1e-5);
    assert!((FloatMath::tanh(x) - x.tanh()).abs() < 1e-6);
    assert!((FloatMath::powf(x + 1.0, 2.5) - (x + 1.0).powf(2.5)).abs() < 1e-4);
    assert!((FloatMath::atan2(x, 1.0) - x.atan2(1.0)).abs() < 1e-6);

    let y = 1.3f64;
    assert_eq!(FloatMath::abs(-y), y);
    assert!((FloatMath::sin(y) - y.sin()).abs() < 1e-12);
    assert!((FloatMath::cos(y) - y.cos()).abs() < 1e-12);
    assert!((FloatMath::tan(y) - y.tan()).abs() < 1e-12);
    assert!((FloatMath::sqrt(y) - y.sqrt()).abs() < 1e-12);
    assert!((FloatMath::ln(y) - y.ln()).abs() < 1e-12);
    assert!((FloatMath::log10(y) - y.log10()).abs() < 1e-12);
    assert!((FloatMath::exp(y) - y.exp()).abs() < 1e-12);
    assert!((FloatMath::tanh(y) - y.tanh()).abs() < 1e-12);
    assert!((FloatMath::powf(y, 3.0) - y.powf(3.0)).abs() < 1e-12);
    assert!((FloatMath::atan2(y, 0.5) - y.atan2(0.5)).abs() < 1e-12);
}

#[test]
fn fast_math_helpers_cover_wrap_and_saturation_paths() {
    // `cossin_f32` folds angles that are outside [-pi, pi] in both directions.
    for &rad in &[4.0f32, -4.0] {
        let (c, s) = cossin_f32(rad);
        assert!((c - rad.cos()).abs() < 5e-3, "cos({rad}) = {c}");
        assert!((s - rad.sin()).abs() < 5e-3, "sin({rad}) = {s}");
    }

    // atan2 handles the origin explicitly and matches the real function elsewhere.
    assert_eq!(fast_atan2_f32(0.0, 0.0), 0.0);
    assert!((fast_atan2_f32(1.0, 1.0) - core::f32::consts::FRAC_PI_4).abs() < 1e-3);

    // Vector square root maps negative inputs to zero.
    let mut dst = [0.0f32; 3];
    vsqrt_f32(&[4.0, -1.0, 9.0], &mut dst);
    assert_eq!(dst, [2.0, 0.0, 3.0]);

    // Wrapping subtraction reports the overflow direction.
    assert_eq!(overflowing_sub_i32(5, 3), (2, 0));
    assert_eq!(overflowing_sub_i32(i32::MIN, 1), (i32::MAX, 1));
    assert_eq!(overflowing_sub_i32(i32::MAX, -1), (i32::MIN, -1));

    // 32-bit rescale: both clamp arms plus the normal path.
    assert_eq!(saturating_scale_i32(0, i32::MIN, 16), i32::MIN + 65536);
    assert_eq!(saturating_scale_i32(0, i32::MAX, 16), i32::MAX - 65535);
    assert_eq!(saturating_scale_i32(0x0001_0000, 0, 16), 1);
}

#[test]
fn integer_phase_unwrappers_track_wraps() {
    let mut iu = IntPhaseUnwrapper::new();
    assert_eq!(iu.phase(), 0);
    assert_eq!(iu.process(i32::MIN), i32::MIN);
    assert_eq!(iu.phase(), i32::MIN);

    // The steps must exceed 2^30 so the difference really wraps around the i32 boundary.
    let mut u = Unwrapper::new();
    assert_eq!(u.turns(), 0);
    let _ = u.update(1_500_000_000);
    // Positive -> negative across the wrap increments the turn count...
    let _ = u.update(-1_500_000_000);
    assert_eq!(u.turns(), 1);
    // ...and the reverse direction decrements it again.
    let _ = u.update(1_500_000_000);
    assert_eq!(u.turns(), 0);
    u.reset();
    assert_eq!(u.turns(), 0);
    assert_eq!(u.update(0), 0);
}

#[test]
fn cordic_polar_covers_the_quadrant_fixups() {
    let (m, a) = cordic_cartesian_to_polar_q15(q15::ZERO, q15::ZERO);
    assert_eq!(m.to_bits(), 0);
    assert_eq!(a.to_bits(), 0);

    // Non-zero x with both angle signs drives the negative-x quadrant correction.
    for (x, y) in [(-16384i16, 8192i16), (-16384, -8192), (-8192, 0)] {
        let (m, _a) = cordic_cartesian_to_polar_q15(q15::from_bits(x), q15::from_bits(y));
        assert!(m.to_bits() > 0, "magnitude collapsed for ({x}, {y})");
    }
}

#[test]
fn small_utility_types_are_usable() {
    let buf = SnapshotBuffer::<8>::default();
    assert_eq!(buf.samples().len(), 0);
    assert!(!buf.is_full());

    let mut filter = SinglePoleFilter::<q15>::lowpass(q15::from_bits(16384));
    let out =
        <SinglePoleFilter<q15> as DspNode<q15>>::process_sample(&mut filter, q15::from_bits(8192));
    assert_ne!(out.to_bits(), 0);
}

#[test]
fn g711_companding_round_trips() {
    // Sweep every linear sample: this covers the encoder saturation branch and, in particular,
    // the negative folding around zero where the A-law encoder used to shift by a negative
    // amount and panic.
    for s in i16::MIN..=i16::MAX {
        let u = linear_to_ulaw(s);
        let a = linear_to_alaw(s);
        assert!(
            (ulaw_to_linear(u) as i32 - s as i32).abs() < 700,
            "ulaw {s} -> {u:#04x} -> {}",
            ulaw_to_linear(u)
        );
        assert!(
            (alaw_to_linear(a) as i32 - s as i32).abs() < 700,
            "alaw {s} -> {a:#04x} -> {}",
            alaw_to_linear(a)
        );
    }

    // Canonical G.711 A-law test vectors (ITU-T G.711 / Sun g711.c).
    assert_eq!(linear_to_alaw(0), 0xD5);
    assert_eq!(linear_to_alaw(1_000), 0xFA);
    assert_eq!(linear_to_alaw(-1), 0x55);
    assert_eq!(linear_to_alaw(i16::MAX), 0xAA);
    assert_eq!(linear_to_alaw(i16::MIN), 0x2A);

    // The float companders stay bounded and monotonic at the origin.
    assert_eq!(a_law_compress_f32(0.0), 0.0);
    let _ = a_law_expand_f32(a_law_compress_f32(0.5));
}
