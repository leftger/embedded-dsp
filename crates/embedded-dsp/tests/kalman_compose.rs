//! Tests for the composable Kalman models in `embedded_dsp::kalman_compose`.
//!
//! The centrepiece is an oracle cross-check: the same filter is run through this crate and
//! through upstream `idsp` (a **dev-dependency**, never a runtime one) on identical random
//! models and measurements, and the two must agree step for step. That pins the prediction and
//! measurement algebra to a reference implementation rather than to this crate's own opinion of
//! it, which matters because a covariance update that is subtly wrong still produces
//! plausible-looking numbers.
//!
//! Around that sit the properties the composition is *supposed* to add: the joint vector update
//! agreeing with sequential scalar updates, `Direct` agreeing with the equivalent dense row, an
//! absent measurement still predicting, and a state-dependent Jacobian behaving as an EKF.

use embedded_dsp::kalman_compose::{
    ConstantVelocity, ControlModel, Direct, Dynamics, Estimate, Kalman, Observation, Optional,
    Predict, Transition, Update, VectorObservation,
};
use embedded_dsp::types::Status;
use idsp::kalman as upstream;

/// Tolerance for two independent implementations of the same algebra.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + b.abs())
}

fn assert_estimate_close(
    ours: &Estimate<f64, 2>,
    theirs: &upstream::Estimate<f64, 2>,
    step: usize,
) {
    for i in 0..2 {
        assert!(
            close(ours.state[i], theirs.state[i]),
            "step {step}: state[{i}] {} vs {}",
            ours.state[i],
            theirs.state[i]
        );
        for j in 0..2 {
            assert!(
                close(ours.covariance[i][j], theirs.covariance[i][j]),
                "step {step}: P[{i}][{j}] {} vs {}",
                ours.covariance[i][j],
                theirs.covariance[i][j]
            );
        }
    }
}

/// Small deterministic generator: no dependency, no flakiness on reruns.
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as u32) as f64 / (u32::MAX as f64 / 2.0) - 1.0
    }
}

#[test]
fn matches_upstream_idsp_step_for_step() {
    let mut rng = Lcg(0x5eed);

    // A modestly contracting F keeps the random walk bounded.
    let f = [[0.98, 0.05], [-0.02, 0.97]];
    let q = [[1e-3, 1e-4], [1e-4, 2e-3]];
    let h = [1.0, 0.25];
    let r = 0.05;

    let initial_state = [0.4, -0.2];
    let initial_covariance = [[0.5, 0.1], [0.1, 0.3]];

    let mut ours = Estimate::<f64, 2>::new(initial_state, initial_covariance);
    let mut theirs = upstream::Estimate::<f64, 2>::new(initial_state, initial_covariance);

    let ours_predict = Transition::new(f, q);
    let ours_update = Observation::new(h, r);
    let theirs_predict = upstream::Transition::<f64, 2>::new(f, q);
    let theirs_update = upstream::Observation::<f64, 2>::new(h, r);

    for step in 0..32 {
        let z = 0.5 + rng.next_unit();

        ours_predict.predict(&mut ours, ());
        theirs_predict.predict(&mut theirs);
        assert!(
            close(ours.state[0], theirs.state[0])
                && close(ours.covariance[0][0], theirs.covariance[0][0]),
            "prediction diverged at step {step}"
        );

        let ours_z = ours_update.update(&mut ours, z).unwrap();
        let theirs_z = theirs_update.correct(&mut theirs, z);
        assert!(
            close(ours_z, theirs_z),
            "step {step}: Hx {ours_z} vs {theirs_z}"
        );

        assert_estimate_close(&ours, &theirs, step);
    }
}

#[test]
fn direct_agrees_with_upstream_too() {
    // Same check through the no-matrix path: `Direct` must reduce to the dense unit row.
    let mut ours = Estimate::<f64, 2>::new([0.4, -0.2], [[0.5, 0.1], [0.1, 0.3]]);
    let mut theirs = upstream::Estimate::<f64, 2>::new([0.4, -0.2], [[0.5, 0.1], [0.1, 0.3]]);

    for (i, z) in [1.0, 2.0, 0.5].into_iter().enumerate() {
        let a = Direct::<f64, 2, 0>::new(0.25).update(&mut ours, z).unwrap();
        let b = upstream::Direct::<f64, 0>::new(0.25).correct(&mut theirs, z);
        assert!(close(a, b), "step {i}");
        assert_estimate_close(&ours, &theirs, i);
    }
}

#[test]
fn scalar_update_matches_the_closed_form() {
    // x⁺ = x + K(z - Hx), P⁺ = (I - KH)P, with K = P Hᵀ / (H P Hᵀ + R).
    let prior = Estimate::<f64, 2>::new([0.3, -0.7], [[0.8, 0.2], [0.2, 0.4]]);
    let h = [1.5, -0.25];
    let r = 0.3;
    let z = 1.1;

    let mut ours = prior;
    let output = Observation::new(h, r).update(&mut ours, z).unwrap();

    let hp = [
        prior.covariance[0][0] * h[0] + prior.covariance[0][1] * h[1],
        prior.covariance[1][0] * h[0] + prior.covariance[1][1] * h[1],
    ];
    let variance = h[0] * hp[0] + h[1] * hp[1] + r;
    let gain = [hp[0] / variance, hp[1] / variance];
    let innovation = z - (h[0] * prior.state[0] + h[1] * prior.state[1]);

    for i in 0..2 {
        assert!(close(ours.state[i], prior.state[i] + gain[i] * innovation));
        for j in 0..2 {
            // (I - KH)P is a plain matrix product, so compute it directly.
            let expected = (0..2)
                .map(|t| {
                    let mut m = -gain[i] * h[t];
                    if i == t {
                        m += 1.0;
                    }
                    m * prior.covariance[t][j]
                })
                .sum::<f64>();
            assert!(
                close(ours.covariance[i][j], expected),
                "P[{i}][{j}] {} vs {expected}",
                ours.covariance[i][j]
            );
        }
    }
    assert!(close(output, h[0] * ours.state[0] + h[1] * ours.state[1]));
}

#[test]
fn joint_vector_update_matches_sequential_scalar_updates() {
    // With uncorrelated measurement noise the posterior is independent of whether the
    // components are fused jointly or one after another, so the matrix-inverse path and the
    // scalar path have to agree.
    let prior = Estimate::<f64, 2>::new([0.2, 0.9], [[0.6, 0.15], [0.15, 0.5]]);
    let h = [[1.0, 0.0], [0.5, 1.0]];
    let r = [[0.25, 0.0], [0.0, 0.5]];
    let z = [1.4, -0.3];

    let mut joint = prior;
    let out = VectorObservation::new(h, r).update(&mut joint, z).unwrap();

    let mut sequential = prior;
    Observation::new(h[0], r[0][0])
        .update(&mut sequential, z[0])
        .unwrap();
    Observation::new(h[1], r[1][1])
        .update(&mut sequential, z[1])
        .unwrap();

    for i in 0..2 {
        assert!(close(joint.state[i], sequential.state[i]), "x[{i}]");
        assert!(close(
            out[i],
            h[i][0] * joint.state[0] + h[i][1] * joint.state[1]
        ));
        for j in 0..2 {
            assert!(
                close(joint.covariance[i][j], sequential.covariance[i][j]),
                "P[{i}][{j}]"
            );
        }
    }
}

#[test]
fn optional_predicts_when_the_measurement_is_absent() {
    let f = [[1.0, 1.0], [0.0, 1.0]];
    let q = [[1e-4, 0.0], [0.0, 1e-4]];
    let h = [1.0, 0.0];
    let r = 0.5;
    let initial = Estimate::<f64, 2>::new([0.0, 2.0], [[1.0, 0.0], [0.0, 1.0]]);

    let mut gated = Kalman::new(Transition::new(f, q), Observation::new(h, r), initial).optional();
    assert!(matches!(gated.step((), None), Ok(None)));

    // The same model without the wrapper, predicted only.
    let mut plain = Kalman::new(Transition::new(f, q), Observation::new(h, r), initial);
    plain.advance(());

    for i in 0..2 {
        assert!(
            close(gated.estimate.state[i], plain.estimate.state[i]),
            "x[{i}]"
        );
        for j in 0..2 {
            assert!(close(
                gated.estimate.covariance[i][j],
                plain.estimate.covariance[i][j]
            ));
        }
    }

    // And with a measurement present it behaves exactly like the unwrapped filter.
    let mut wrapped =
        Kalman::new(Transition::new(f, q), Observation::new(h, r), initial).optional();
    wrapped.step((), Some(3.0)).unwrap();
    let mut reference = Kalman::new(Transition::new(f, q), Observation::new(h, r), initial);
    reference.step((), 3.0).unwrap();
    assert!(close(
        wrapped.estimate.state[0],
        reference.estimate.state[0]
    ));
}

#[test]
fn control_model_applies_its_control_term() {
    // x⁻ = F x + B u, so comparing against a hand-applied matrix-vector product checks both the
    // state term and the control term without needing a reference implementation.
    let f = [[1.0, 1.0], [0.0, 1.0]];
    let b = [[0.5, 0.0], [1.0, 0.25]];
    let q = [[1e-5, 0.0], [0.0, 1e-5]];
    let u = [2.0, -4.0];
    let initial = Estimate::<f64, 2>::new([1.0, 0.5], [[0.1, 0.0], [0.0, 0.1]]);

    let mut filter = Kalman::new(
        Transition::new(ControlModel::new(f, b), q),
        Observation::new([1.0, 0.0], 1.0),
        initial,
    );
    filter.advance(u);

    let expected = [
        f[0][0] * initial.state[0] + f[0][1] * initial.state[1] + b[0][0] * u[0] + b[0][1] * u[1],
        f[1][0] * initial.state[0] + f[1][1] * initial.state[1] + b[1][0] * u[0] + b[1][1] * u[1],
    ];
    for i in 0..2 {
        assert!(close(filter.estimate.state[i], expected[i]), "x[{i}]");
    }
}

#[test]
fn a_state_dependent_jacobian_behaves_like_an_ekf() {
    /// `x' = [x0 + sin(x1), x1]`.
    struct Coupled;

    impl Dynamics<f64, 2> for Coupled {
        type Input = ();

        fn advance(&self, x: &[f64; 2], _: ()) -> [f64; 2] {
            [x[0] + x[1].sin(), x[1]]
        }

        fn jacobian(&self, x: &[f64; 2], _: ()) -> [[f64; 2]; 2] {
            [[1.0, x[1].cos()], [0.0, 1.0]]
        }
    }

    // The measurement pins x0 at 1.0; the model keeps trying to add `sin(x1)`, so tracking the
    // constant means the filter must also drive x1 toward zero.
    let mut ekf = Kalman::new(
        Transition::new(Coupled, [[1e-8, 0.0], [0.0, 1e-8]]),
        Observation::new([1.0, 0.0], 0.01),
        Estimate::<f64, 2>::new([0.0, 0.5], [[1.0, 0.0], [0.0, 1.0]]),
    );

    for _ in 0..200 {
        ekf.step((), 1.0).unwrap();
    }

    assert!(
        (ekf.estimate.state[0] - 1.0).abs() < 0.05,
        "x0 = {}",
        ekf.estimate.state[0]
    );
    assert!(
        ekf.estimate.state[1].abs() < 0.2,
        "x1 should have been driven toward zero, got {}",
        ekf.estimate.state[1]
    );
}

#[test]
fn constant_velocity_helpers_track_a_ramp() {
    // The canned constructors have to be equivalent to assembling the pieces by hand.
    let q = [[1e-8, 0.0], [0.0, 1e-8]];
    let initial = Estimate::<f64, 2>::new([0.0, 0.0], [[10.0, 0.0], [0.0, 10.0]]);

    let mut canned = Kalman::constant_velocity(1.0, q, 0.5, initial);
    let mut assembled = Kalman::new(
        Transition::new(ConstantVelocity::new(1.0), q),
        Observation::new([1.0, 0.0], 0.5),
        initial,
    );

    for step in 1..=60 {
        let position = 1.5 * step as f64;
        let a = canned.step((), position).unwrap();
        let b = assembled.step((), position).unwrap();
        assert!(close(a, b), "step {step}: {a} vs {b}");
    }
    assert!((canned.estimate.state[0] - 90.0).abs() < 1.0);
}

#[test]
fn the_same_models_run_on_f32() {
    // The composition is generic over `DspSample`, so nothing here is f64-specific.
    let mut filter = Kalman::random_walk(0.01f32, 1.0, Estimate::new([0.0f32], [[1.0]]));
    let mut tracked = 0.0;
    for _ in 0..200 {
        tracked = filter.step((), 5.0).unwrap();
    }
    assert!((tracked - 5.0).abs() < 0.05, "converged to {tracked}");

    // An explicit `Optional` also has to type-check through the generic path.
    let mut gated: Kalman<_, Optional<Observation<f32, 1>>, f32, 1> = filter.optional();
    assert!(matches!(gated.step((), None), Ok(None)));
}

#[test]
fn projections_report_what_the_estimate_predicts() {
    let e = Estimate::<f64, 2>::new([0.5, -0.25], [[1.0, 0.0], [0.0, 1.0]]);

    assert!(close(Observation::new([2.0, 4.0], 1.0).project(&e), 0.0));
    assert!(close(Direct::<f64, 2, 1>::new(1.0).project(&e), -0.25));

    let v = VectorObservation::new([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]).project(&e);
    assert!(close(v[0], 0.5) && close(v[1], -0.25));

    // `Optional` projects the same measurement, just wrapped.
    let optional = Optional(Observation::new([1.0, 0.0], 1.0));
    assert_eq!(optional.project(&e).map(|x| close(x, 0.5)), Some(true));

    // And the filter forwards it.
    let filter = Kalman::new(
        Transition::identity([[0.0; 2]; 2]),
        Observation::new([1.0, 0.0], 1.0),
        e,
    );
    assert!(close(filter.project(), 0.5));
}

#[test]
fn an_identity_transition_leaves_the_state_alone() {
    let q = [[0.1, 0.0], [0.0, 0.2]];
    let initial = Estimate::<f64, 2>::new([0.5, -0.25], [[1.0, 0.1], [0.1, 1.0]]);
    let mut filter = Kalman::new(
        Transition::identity(q),
        Observation::new([1.0, 0.0], 1.0),
        initial,
    );
    filter.advance(());

    for i in 0..2 {
        assert!(close(filter.estimate.state[i], initial.state[i]), "x[{i}]");
        for j in 0..2 {
            assert!(close(
                filter.estimate.covariance[i][j],
                initial.covariance[i][j] + q[i][j]
            ));
        }
    }

    // The zero-state constructor is the other half of the ergonomics.
    let zero = Estimate::<f64, 2>::from_covariance([[2.0, 0.0], [0.0, 3.0]]);
    assert_eq!(zero.state, [0.0, 0.0]);
    assert_eq!(zero.covariance, [[2.0, 0.0], [0.0, 3.0]]);
}

#[test]
fn a_singular_innovation_covariance_is_reported_and_not_applied() {
    // A zero measurement variance on top of a zero row of P leaves nothing to divide by. The
    // estimate must come back untouched rather than as NaN.
    let prior = Estimate::<f64, 2>::new([1.0, 1.0], [[0.0, 0.0], [0.0, 1.0]]);

    let mut scalar = prior;
    assert_eq!(
        Observation::new([1.0, 0.0], 0.0).update(&mut scalar, 2.0),
        Err(Status::Singular)
    );
    assert_eq!(scalar, prior);

    let mut direct = prior;
    assert_eq!(
        Direct::<f64, 2, 0>::new(0.0).update(&mut direct, 2.0),
        Err(Status::Singular)
    );
    assert_eq!(direct, prior);

    // The vector path reaches the same verdict through its own inverse.
    let mut vector = Estimate::<f64, 2>::new([0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]);
    let before = vector;
    assert_eq!(
        VectorObservation::new([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
            .update(&mut vector, [1.0, 1.0]),
        Err(Status::Singular)
    );
    assert_eq!(vector, before);
}

#[test]
fn the_vector_path_pivots_off_a_zero_leading_entry() {
    // S = [[0, 1], [1, 2]] has a zero on the diagonal, so the Gauss-Jordan step has to swap rows
    // before it can normalise. It is invertible (det = -1), so the update must still succeed.
    let mut estimate = Estimate::<f64, 2>::new([1.0, 1.0], [[0.0, 1.0], [1.0, 2.0]]);
    let observation = VectorObservation::new([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]);

    let output = observation.update(&mut estimate, [2.0, 3.0]).unwrap();

    // S⁻¹ = [[-2, 1], [1, 0]], so K = P S⁻¹ and the posterior is finite and symmetric.
    assert!(output.iter().all(|v| v.is_finite()));
    assert!(estimate.covariance[0][1].abs() < 1e-12);
    assert!(estimate.covariance[1][0].abs() < 1e-12);
}
