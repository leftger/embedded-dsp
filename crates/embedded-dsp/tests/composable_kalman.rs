//! Tests for the composable Kalman models exposed by `embedded_dsp::kalman_models`.
//!
//! These deliberately exercise the *re-exported* surface rather than re-testing `idsp` itself:
//! they prove the feature wires up a usable API (the types, the `dsp_process` traits, and the
//! composition between them) and that composing the pieces gives the documented results.

use embedded_dsp::kalman_models::{
    ConstantVelocity, Direct, Estimate, Kalman, Observation, Process, RandomWalk, Transition,
    dsp_process::Split,
};

/// A scalar random walk should converge onto a constant input.
///
/// The steady-state gain of `F = H = 1` is roughly `sqrt(Q/R)` once settled, so `Q` sets how
/// fast that happens: at `Q = 1e-6` the time constant is ~1000 samples.
#[test]
fn random_walk_converges_on_a_constant_signal() {
    let mut filter = Split::new(
        RandomWalk::<f64>::new(0.01, 1.0),
        Estimate::new([0.0], [[1.0]]),
    );

    let mut last = 0.0;
    for _ in 0..200 {
        last = filter.process(5.0);
    }
    assert!((last - 5.0).abs() < 0.05, "converged to {last}");
}

/// `Direct` observes one state component; `Observation` with `H = [1, 0]` says the same thing
/// the long way round. They must agree exactly.
#[test]
fn direct_and_dense_observations_agree() {
    let prior = Estimate::new([2.0, -1.0], [[4.0, 1.5], [1.5, 3.0]]);
    let (mut direct, mut dense) = (prior, prior);

    let from_direct = Direct::<f64, 0>::new(0.75).correct(&mut direct, 3.5);
    let from_dense = Observation::new([1.0, 0.0], 0.75).correct(&mut dense, 3.5);

    assert_eq!(from_direct, from_dense);
    assert_eq!(direct, dense);
}

/// One correction step against the closed-form linear-Gaussian update.
#[test]
fn correction_matches_the_closed_form_update() {
    let observation = Observation::<f64, 2>::new([1.25, -0.5], 0.75);
    let prior = Estimate::<f64, 2>::new([2.0, -1.0], [[4.0, 1.5], [1.5, 3.0]]);
    let mut estimate = prior;

    let output = observation.correct(&mut estimate, 3.5);

    // u = P H', s = H u + R, K = u / s, x += K * innovation, P -= K u'.
    let h = observation.matrix;
    let u = [
        prior.covariance[0][0] * h[0] + prior.covariance[0][1] * h[1],
        prior.covariance[1][0] * h[0] + prior.covariance[1][1] * h[1],
    ];
    let variance = h[0] * u[0] + h[1] * u[1] + observation.noise;
    let gain = [u[0] / variance, u[1] / variance];
    let innovation = 3.5 - h[0] * prior.state[0] - h[1] * prior.state[1];

    let expected = [
        prior.state[0] + gain[0] * innovation,
        prior.state[1] + gain[1] * innovation,
    ];
    for (got, want) in estimate.state.iter().zip(expected) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    assert!((output - (h[0] * expected[0] + h[1] * expected[1])).abs() < 1e-12);

    for (i, row) in estimate.covariance.iter().enumerate() {
        for (j, got) in row.iter().enumerate() {
            let want = prior.covariance[i][j] - gain[i] * u[j];
            assert!((got - want).abs() < 1e-12, "P[{i}][{j}]: {got} vs {want}");
        }
    }
}

/// The composition that matters: a constant-velocity model with a direct position observation.
#[test]
fn constant_velocity_composition_tracks_a_ramp() {
    let predict = ConstantVelocity::<f64>::new(1.0, [[1e-8, 0.0], [0.0, 1e-8]]);
    let update = Direct::<f64, 0>::new(0.5);
    let mut filter = Split::new(
        Kalman::new(predict, update),
        Estimate::new([0.0, 0.0], [[10.0, 0.0], [0.0, 10.0]]),
    );

    let mut tracked = 0.0;
    for step in 1..=80 {
        tracked = filter.process(2.0 * step as f64);
    }

    // The ramp is at 160 by the last step; a converged tracker sits within a sample of it.
    assert!((tracked - 160.0).abs() < 1.0, "tracked {tracked}");
}

/// The same filter built from the generic pieces must behave identically to the canned ones.
#[test]
fn assembled_pieces_match_the_canned_models() {
    let noise = [[1e-8, 0.0], [0.0, 1e-8]];
    let prior = Estimate::new([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]);

    let mut canned = Split::new(
        Kalman::new(
            ConstantVelocity::<f64>::new(1.0, noise),
            Direct::<f64, 0>::new(0.5),
        ),
        prior,
    );
    let mut assembled = Split::new(
        Kalman::new(
            Transition::new([[1.0, 1.0], [0.0, 1.0]], noise),
            Observation::new([1.0, 0.0], 0.5),
        ),
        prior,
    );

    for step in 1..=40 {
        let position = 1.5 * step as f64;
        let a = canned.process(position);
        let b = assembled.process(position);
        assert!((a - b).abs() < 1e-12, "step {step}: {a} vs {b}");
    }
}
