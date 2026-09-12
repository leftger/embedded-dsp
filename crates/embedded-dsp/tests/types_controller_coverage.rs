//! Coverage for the fixed-point `DspSample` plumbing and the PID builder.
//!
//! The `fallback` fixed-point types in `types.rs` (used when the `fixed`
//! feature is off) expose arithmetic helpers and integer comparisons that no
//! test touched, and the `PidBuilder` accessors/validation paths were
//! unexercised.

use embedded_dsp::controller::{PidAction, PidBuilder, PidError};
use embedded_dsp::types::{DspSample, q15, q31};

// ─────────────────────────────────────────────────────────────────────────────
// DspSample impls (present in both `fixed` configurations)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dsp_sample_saturating_ops_for_float_types() {
    assert_eq!(<f32 as DspSample>::sat_div(1.0, 4.0), 0.25);
    assert_eq!(<f64 as DspSample>::to_f32(2.5), 2.5);
}

#[test]
fn dsp_sample_saturating_ops_for_q15_and_q31() {
    let one = q15::from_bits(32_767);
    let half = q15::from_bits(16_384);
    let _ = <q15 as DspSample>::sat_mul(half, one);
    let _ = <q15 as DspSample>::sat_div(half, one);
    assert!((<q15 as DspSample>::to_f32(half) - 0.5).abs() < 1e-3);

    let one31 = q31::from_bits(i32::MAX);
    let half31 = q31::from_bits(1 << 30);
    let _ = <q31 as DspSample>::sat_mul(half31, one31);
    let _ = <q31 as DspSample>::sat_div(half31, one31);
    assert!((<q31 as DspSample>::to_f32(half31) - 0.5).abs() < 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// PID builder
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn pid_builder_accessors_chain_into_a_valid_configuration() {
    let builder = PidBuilder::default()
        .gain(PidAction::P, 1.0)
        .limit(PidAction::I, 0.5)
        .kd2(0.25)
        .limit_d2(0.125)
        .offset(0.75)
        .output_limits(-1.0, 1.0);

    assert!(builder.validate(0.001).is_ok());
}

#[test]
fn pid_builder_validate_rejects_non_finite_period() {
    assert!(matches!(
        PidBuilder::default().validate(f32::NAN),
        Err(PidError::NonFinite("period"))
    ));
}

#[test]
fn pid_builder_validate_rejects_non_finite_limit() {
    assert!(matches!(
        PidBuilder::default()
            .limit(PidAction::I, f32::NAN)
            .validate(0.001),
        Err(PidError::NonFinite("limit"))
    ));
}

#[test]
fn pid_builder_validate_rejects_zero_limit() {
    assert!(matches!(
        PidBuilder::default()
            .limit(PidAction::D, 0.0)
            .validate(0.001),
        Err(PidError::NonPositive("limit"))
    ));
}

#[test]
fn pid_builder_validate_rejects_inverted_output_limits() {
    assert!(matches!(
        PidBuilder::default()
            .output_limits(1.0, -1.0)
            .validate(0.001),
        Err(PidError::InvertedRange("output_limits"))
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// Fallback fixed-point types (`fixed` feature off)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(feature = "fixed"))]
mod fallback_fixed_point {
    use embedded_dsp::types::{FixedNum, I16F16, q15};

    #[test]
    fn fixed_num_f64_conversion_edge_cases() {
        // NaN maps to zero.
        assert_eq!(
            <f64 as FixedNum>::to_raw_fixed(f64::NAN, 8, -1000, 1000, false),
            0
        );

        // Saturating conversion clamps to the low rail.
        assert_eq!(
            <f64 as FixedNum>::to_raw_fixed(-1.0e9, 8, -100, 100, true),
            -100
        );
        assert_eq!(
            <f64 as FixedNum>::to_raw_fixed(1.0e9, 8, -100, 100, true),
            100
        );

        // Exact .5 tie with an odd integer part rounds away from zero.
        // 1.5 / 256 scaled by 256 gives abs_int = 1 (odd) -> rounds to 2.
        let tie = 1.5f64 / 256.0;
        assert_eq!(
            <f64 as FixedNum>::to_raw_fixed(tie, 8, -1000, 1000, true),
            2
        );

        assert_eq!(<f64 as FixedNum>::from_raw_fixed(256, 8), 1.0);
    }

    #[test]
    fn fixed_num_integer_conversion_saturates() {
        assert_eq!(
            <i32 as FixedNum>::to_raw_fixed(1000, 8, -100, 100, true),
            100
        );
        assert_eq!(
            <i32 as FixedNum>::to_raw_fixed(-1000, 8, -100, 100, true),
            -100
        );
        assert_eq!(<i32 as FixedNum>::from_raw_fixed(256, 8), 1);
    }

    #[test]
    fn q15_wrapping_div_and_recip_edge_cases() {
        let half = q15::from_bits(16_384);
        let zero = q15::from_bits(0);

        // Division by zero short-circuits to zero instead of dividing.
        assert_eq!(half.wrapping_div(zero), zero);
        assert_eq!(half.wrapping_div_int(0), zero);

        // Reciprocal of zero saturates to MAX.
        assert_eq!(zero.recip(), q15::MAX);
    }

    #[test]
    fn q15_checked_div_reports_zero_and_overflow() {
        let zero = q15::from_bits(0);
        assert_eq!(q15::from_bits(16_384).checked_div(zero), None);

        // 1.0 / tiny overflows the i16 backing store.
        assert_eq!(q15::from_bits(32_767).checked_div(q15::from_bits(1)), None);

        // 0.25 / 0.5 = 0.5, which fits.
        assert_eq!(
            q15::from_bits(8_192).checked_div(q15::from_bits(16_384)),
            Some(q15::from_bits(16_384))
        );
    }

    #[test]
    fn fallback_fixed_num_scaling_saturates() {
        // q15 has 15 fractional bits; converting into a narrow window clamps.
        assert_eq!(
            q15::from_bits(32_767).to_raw_fixed(15, -100, 100, true),
            100
        );
        assert_eq!(
            q15::from_bits(-32_768).to_raw_fixed(15, -100, 100, true),
            -100
        );

        // Raw values at or below the type's own precision shift left...
        assert_eq!(<q15 as FixedNum>::from_raw_fixed(1, 15), q15::from_bits(1));
        assert_eq!(
            <q15 as FixedNum>::from_raw_fixed(128, 8),
            q15::from_bits(16_384)
        );
        // ...and coarser inputs shift right.
        assert_eq!(<q15 as FixedNum>::from_raw_fixed(32, 20), q15::from_bits(1));
    }

    #[test]
    fn fallback_arithmetic_operators() {
        let a = q15::from_bits(1_000);
        let b = q15::from_bits(200);

        assert_eq!(a + b, q15::from_bits(1_200));
        assert_eq!(a - b, q15::from_bits(800));

        // Q15 multiply: (1000 * 200) >> 15 == 6.
        assert_eq!(a * b, q15::from_bits(6));

        let mut sub = a;
        sub -= b;
        assert_eq!(sub, q15::from_bits(800));

        let mut mul = a;
        mul *= b;
        assert_eq!(mul, q15::from_bits(6));
    }

    #[test]
    fn fallback_display_shows_raw_bits() {
        assert_eq!(format!("{}", q15::from_bits(1_000)), "1000");
        assert_eq!(format!("{}", q15::from_bits(-1)), "-1");
    }

    #[test]
    fn fallback_integer_comparisons() {
        // I16F16 is backed by i32 with 16 fractional bits, so 65536 raw == 1.
        let one = I16F16::from_bits(1 << 16);

        assert!(one == 1i32);
        assert!(1i32 == one);
        assert_eq!(1i32.partial_cmp(&one), Some(core::cmp::Ordering::Equal));

        let two = I16F16::from_bits(2 << 16);
        assert!(1i32 < two);
    }
}
