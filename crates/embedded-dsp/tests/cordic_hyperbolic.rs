//! Tests for the hyperbolic CORDIC modes.
//!
//! `cordic_sqrt_atanh2_q31` / `cordic_atanh_q31` are the hyperbolic counterparts of
//! `cordic_cartesian_to_polar_q15`: they replace `(x, y)` with the hyperbolic norm and angle.
//! These tests check both against the `f64` functions they stand in for, and pin the domain
//! guards, since hyperbolic CORDIC only converges for points inside the unit hyperbola.

use embedded_dsp::cordic::{cordic_atanh_q31, cordic_sqrt_atanh2_q31};
use embedded_dsp::types::q31;

const SCALE: f64 = 2147483648.0;

/// `tanh(1) ≈ 0.7616`: past this `atanh(y/x)` exceeds 1.0 and no longer fits Q31.
const TANH_1: f64 = 0.761_594_155_955_764_9;

/// Measured worst case is ~6e-9, about a dozen Q31 LSBs; leave a little room for drift.
const TOL: f64 = 2e-8;

fn to_q31(v: f64) -> q31 {
    q31::from_bits((v * SCALE).round() as i32)
}

fn from_q31(v: q31) -> f64 {
    v.to_bits() as f64 / SCALE
}

#[test]
fn atanh_tracks_the_real_function_over_its_representable_range() {
    let mut worst = 0.0f64;
    let mut worst_at = 0.0f64;

    // `atanh(r)` has to stay below 1.0 to fit Q31, so r stays below `tanh(1)`.
    for k in 1..=100 {
        let r = TANH_1 * k as f64 / 100.0;
        let x = to_q31(0.75);
        let y = to_q31(0.75 * r);

        let got = from_q31(cordic_atanh_q31(y, x));
        let want = r.atanh();
        if (got - want).abs() > worst {
            worst = (got - want).abs();
            worst_at = r;
        }
    }

    assert!(worst < TOL, "worst atanh error {worst} at r={worst_at}");
}

#[test]
fn sqrt_atanh2_returns_the_hyperbolic_norm() {
    let mut worst_mag = 0.0f64;
    let mut worst_ang = 0.0f64;

    for k in 1..=40 {
        let r = TANH_1 * k as f64 / 40.0;
        let x = 0.95;
        let y = x * r;

        let (mag, ang) = cordic_sqrt_atanh2_q31(to_q31(x), to_q31(y));
        worst_mag = worst_mag.max((from_q31(mag) - (x * x - y * y).sqrt()).abs());
        worst_ang = worst_ang.max((from_q31(ang) - r.atanh()).abs());
    }

    assert!(worst_mag < TOL, "worst magnitude error {worst_mag}");
    assert!(worst_ang < TOL, "worst angle error {worst_ang}");
}

#[test]
fn negative_operands_are_odd_symmetric() {
    for &(x, y) in &[(0.8, 0.3), (0.9, -0.4), (0.6, 0.5)] {
        let (mx, my) = (
            cordic_sqrt_atanh2_q31(to_q31(x), to_q31(-y)),
            cordic_sqrt_atanh2_q31(to_q31(x), to_q31(y)),
        );
        // Negating y reflects the point across the x axis: same norm, negated angle.
        assert!(
            (from_q31(mx.0) - from_q31(my.0)).abs() < TOL,
            "magnitude depends on the sign of y: {} vs {}",
            from_q31(mx.0),
            from_q31(my.0)
        );
        assert!(
            (from_q31(mx.1) + from_q31(my.1)).abs() < TOL,
            "angle is not odd in y: {} vs {}",
            from_q31(mx.1),
            from_q31(my.1)
        );
    }
}

#[test]
fn outside_the_unit_hyperbola_the_result_is_zero() {
    // |y| >= x has no real atanh, and the CORDIC cannot converge there.
    for &(x, y) in &[
        (0.5, 0.5),
        (0.5, 0.75),
        (0.5, -0.9),
        (0.0, 0.0),
        (-0.5, 0.1),
        (0.0, 0.5),
    ] {
        let (mag, ang) = cordic_sqrt_atanh2_q31(to_q31(x), to_q31(y));
        assert_eq!(
            (mag.to_bits(), ang.to_bits()),
            (0, 0),
            "expected (0, 0) for x={x} y={y}"
        );
        assert_eq!(cordic_atanh_q31(to_q31(y), to_q31(x)).to_bits(), 0);
    }

    // On the axis the answer is near-exact: norm x, angle 0. The angle is not bit-exact because
    // the first microrotation overshoots and has to be walked back within the Q31 grid.
    let (mag, ang) = cordic_sqrt_atanh2_q31(to_q31(0.5), to_q31(0.0));
    assert!(
        (from_q31(mag) - 0.5).abs() < TOL,
        "magnitude {}",
        from_q31(mag)
    );
    assert!(from_q31(ang).abs() < 1e-8, "angle {}", from_q31(ang));
}
