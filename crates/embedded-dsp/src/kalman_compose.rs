//! Composable Kalman models: assemble a filter from phases instead of picking one.
//!
//! [`crate::kalman`] gives you filters with a fixed shape — `KalmanFilter1D`, `KalmanFilter2D`,
//! `KalmanFilter<N, M>`, an EKF, a square-root filter — and you choose the one matching your
//! model. This module inverts that: a filter is a **prediction phase** and a **measurement
//! phase** over a shared [`Estimate`], and both phases are pluggable types.
//!
//! | Piece | Role |
//! | :--- | :--- |
//! | [`Estimate`] | state vector plus its error covariance |
//! | [`Dynamics`] | how the state evolves, and its Jacobian |
//! | [`Transition`] | a [`Dynamics`] plus process noise `Q` |
//! | [`ConstantVelocity`] | the `[[1, dt], [0, 1]]` motion model |
//! | [`ControlModel`] | `F x + B u`, for systems with a control input |
//! | [`Observation`] | scalar measurement row `H` and noise `R` |
//! | [`VectorObservation`] | `M`-dimensional `H` and `R`, for multi-sensor updates |
//! | [`Direct`] | observe a single state component, with no `H` at all |
//! | [`Optional`] | skip the update when a measurement does not arrive |
//! | [`Kalman`] | a phase pair plus its estimate: the running filter |
//!
//! # Why compose
//!
//! Three things this expresses that the fixed filters cannot:
//!
//! * **[`Dynamics`] unifies the linear and nonlinear cases.** A transition matrix returns itself
//!   from `jacobian` and ignores the state; a model that computes its Jacobian from the state is
//!   an EKF. Same code path, so there is no separate EKF type to keep in sync.
//! * **Measurement shape is a choice, not a type you must find.** `M = 1` uses
//!   [`Observation`] and needs no matrix inverse at all; `M > 1` uses [`VectorObservation`].
//! * **Measurements may be absent.** [`Optional`] lets one phase pair handle a sensor that drops
//!   out, which is awkward to express when the update is a fixed method call.
//!
//! Everything is generic over [`DspSample`], so the same models run on `f32`, `f64` or the
//! crate's fixed-point types.
//!
//! # Examples
//!
//! The smallest complete filter, built from pieces:
//!
//! ```
//! use embedded_dsp::kalman_compose::{Estimate, Kalman};
//!
//! // A scalar random walk: F = H = 1.
//! let mut filter = Kalman::random_walk(0.01, 1.0, Estimate::new([0.0f32], [[1.0]]));
//! let tracked = filter.step((), 2.0).unwrap();
//! assert!(tracked > 0.0 && tracked < 2.0);
//! ```
//!
//! A position/velocity tracker, where the model is reusable configuration:
//!
//! ```
//! use embedded_dsp::kalman_compose::{
//!     ConstantVelocity, Estimate, Kalman, Observation, Transition,
//! };
//!
//! let process_noise = [[1e-4f32, 0.0], [0.0, 1e-4]];
//! let mut filter = Kalman::new(
//!     Transition::new(ConstantVelocity::new(1.0), process_noise),
//!     Observation::new([1.0, 0.0], 0.5),
//!     Estimate::new([0.0, 0.0], [[10.0, 0.0], [0.0, 10.0]]),
//! );
//!
//! let mut tracked = 0.0;
//! for step in 1..=80 {
//!     tracked = filter.step((), 2.0 * step as f32).unwrap();
//! }
//! assert!((tracked - 160.0).abs() < 1.0);
//! ```
//!
//! A nonlinear model is the same composition, with a [`Dynamics`] that linearises:
//!
//! ```
//! use embedded_dsp::kalman_compose::{Dynamics, Estimate, Kalman, Observation, Transition};
//!
//! /// `x' = [x0 + sin(x1), x1]` — the Jacobian depends on the state, so this is an EKF.
//! struct Coupled;
//!
//! impl Dynamics<f32, 2> for Coupled {
//!     type Input = ();
//!
//!     fn advance(&self, x: &[f32; 2], _: ()) -> [f32; 2] {
//!         [x[0] + x[1].sin(), x[1]]
//!     }
//!
//!     fn jacobian(&self, x: &[f32; 2], _: ()) -> [[f32; 2]; 2] {
//!         [[1.0, x[1].cos()], [0.0, 1.0]]
//!     }
//! }
//!
//! let mut ekf = Kalman::new(
//!     Transition::new(Coupled, [[1e-6, 0.0], [0.0, 1e-6]]),
//!     Observation::new([1.0, 0.0], 0.01),
//!     Estimate::new([0.0, 0.5], [[1.0, 0.0], [0.0, 1.0]]),
//! );
//! for _ in 0..200 {
//!     let _ = ekf.step((), 1.0);
//! }
//! assert!(
//!     (ekf.estimate.state[0] - 1.0).abs() < 0.25,
//!     "x0 = {}, x1 = {}",
//!     ekf.estimate.state[0],
//!     ekf.estimate.state[1],
//! );
//! ```

use crate::types::{DspSample, Status};

// ─────────────────────────────────────────────────────────────────────────────
// Small generic matrix helpers (any `DspSample`, so f32/f64/fixed all work)
// ─────────────────────────────────────────────────────────────────────────────

fn identity_mat<T: DspSample, const N: usize>() -> [[T; N]; N] {
    let mut m = [[T::ZERO; N]; N];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = T::ONE;
    }
    m
}

/// `M · v` for `M: R×C`.
fn mat_vec<T: DspSample, const R: usize, const C: usize>(m: &[[T; C]; R], v: &[T; C]) -> [T; R] {
    let mut out = [T::ZERO; R];
    for (o, row) in out.iter_mut().zip(m) {
        let mut acc = T::ZERO;
        for (a, b) in row.iter().zip(v) {
            acc = acc + *a * *b;
        }
        *o = acc;
    }
    out
}

/// `A · B` for `A: R×K` and `B: K×C`.
fn mat_mul<T: DspSample, const R: usize, const K: usize, const C: usize>(
    a: &[[T; K]; R],
    b: &[[T; C]; K],
) -> [[T; C]; R] {
    let mut out = [[T::ZERO; C]; R];
    for (o, arow) in out.iter_mut().zip(a) {
        for (c, oc) in o.iter_mut().enumerate() {
            let mut acc = T::ZERO;
            for (k, ak) in arow.iter().enumerate() {
                acc = acc + *ak * b[k][c];
            }
            *oc = acc;
        }
    }
    out
}

/// `A · Bᵀ` for `A: R×K` and `B: C×K`.
fn mat_mul_bt<T: DspSample, const R: usize, const K: usize, const C: usize>(
    a: &[[T; K]; R],
    b: &[[T; K]; C],
) -> [[T; C]; R] {
    let mut out = [[T::ZERO; C]; R];
    for (o, arow) in out.iter_mut().zip(a) {
        for (c, oc) in o.iter_mut().enumerate() {
            let mut acc = T::ZERO;
            for (k, ak) in arow.iter().enumerate() {
                acc = acc + *ak * b[c][k];
            }
            *oc = acc;
        }
    }
    out
}

fn dot<T: DspSample, const N: usize>(a: &[T; N], b: &[T; N]) -> T {
    let mut acc = T::ZERO;
    for (x, y) in a.iter().zip(b) {
        acc = acc + *x * *y;
    }
    acc
}

/// Gauss-Jordan inverse with partial pivoting. `None` when the matrix is singular.
fn invert<T: DspSample, const M: usize>(s: &[[T; M]; M]) -> Option<[[T; M]; M]> {
    let mut a = *s;
    let mut inv = identity_mat::<T, M>();

    for col in 0..M {
        let mut pivot = col;
        for r in (col + 1)..M {
            if a[r][col].abs_val() > a[pivot][col].abs_val() {
                pivot = r;
            }
        }
        if a[pivot][col].abs_val() <= T::ZERO {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
            inv.swap(pivot, col);
        }

        let d = a[col][col];
        for c in 0..M {
            a[col][c] = a[col][c].sat_div(d);
            inv[col][c] = inv[col][c].sat_div(d);
        }
        for r in 0..M {
            if r == col {
                continue;
            }
            let f = a[r][col];
            for c in 0..M {
                a[r][c] = a[r][c] - f * a[col][c];
                inv[r][c] = inv[r][c] - f * inv[col][c];
            }
        }
    }
    Some(inv)
}

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

/// A state estimate: the state vector and its error covariance.
///
/// Deliberately plain data, so it can be snapshotted, logged or handed to another phase pair
/// without going through the filter that owns it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate<T: DspSample, const N: usize> {
    /// Estimated state vector.
    pub state: [T; N],
    /// Symmetric state-error covariance `P`.
    pub covariance: [[T; N]; N],
}

impl<T: DspSample, const N: usize> Estimate<T, N> {
    /// Construct an estimate from a state and a covariance.
    #[must_use]
    pub fn new(state: [T; N], covariance: [[T; N]; N]) -> Self {
        Self { state, covariance }
    }

    /// A zero state with the given covariance.
    #[must_use]
    pub fn from_covariance(covariance: [[T; N]; N]) -> Self {
        Self {
            state: [T::ZERO; N],
            covariance,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dynamics: how the state evolves
// ─────────────────────────────────────────────────────────────────────────────

/// Model of the state evolution, `x⁻ = f(x, u)`, and its Jacobian `F = ∂f/∂x`.
///
/// One trait covers both the linear and the nonlinear case. A linear model returns its stored
/// matrix from [`jacobian`](Dynamics::jacobian) and ignores the state; a model that computes the
/// Jacobian from the state is an extended Kalman filter. Nothing else in the composition changes.
pub trait Dynamics<T: DspSample, const N: usize> {
    /// Control input consumed by this model (`()` when there is none).
    type Input: Copy;

    /// Advance the state one step: `x⁻ = f(x, u)`.
    fn advance(&self, state: &[T; N], input: Self::Input) -> [T; N];

    /// Jacobian evaluated at `state` and `input`.
    fn jacobian(&self, state: &[T; N], input: Self::Input) -> [[T; N]; N];
}

/// A transition matrix *is* a linear model: `f(x, u) = F x`.
impl<T: DspSample, const N: usize> Dynamics<T, N> for [[T; N]; N] {
    type Input = ();

    fn advance(&self, state: &[T; N], _input: ()) -> [T; N] {
        mat_vec(self, state)
    }

    fn jacobian(&self, _state: &[T; N], _input: ()) -> [[T; N]; N] {
        *self
    }
}

/// Constant-velocity motion for `[position, velocity]`: `F = [[1, dt], [0, 1]]`.
///
/// A [`Dynamics`] rather than a complete filter, so it drops straight into a [`Transition`] and
/// can be swapped for any other model without touching the rest of the composition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantVelocity<T: DspSample = f32> {
    /// Sample interval `dt`.
    pub interval: T,
}

impl<T: DspSample> ConstantVelocity<T> {
    /// Construct a constant-velocity model for the given sample interval.
    #[must_use]
    pub fn new(interval: T) -> Self {
        Self { interval }
    }
}

impl<T: DspSample> Dynamics<T, 2> for ConstantVelocity<T> {
    type Input = ();

    fn advance(&self, state: &[T; 2], _input: ()) -> [T; 2] {
        [state[0] + self.interval * state[1], state[1]]
    }

    fn jacobian(&self, _state: &[T; 2], _input: ()) -> [[T; 2]; 2] {
        [[T::ONE, self.interval], [T::ZERO, T::ONE]]
    }
}

/// Linear model with a control input: `f(x, u) = F x + B u`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlModel<const N: usize, const U: usize, T: DspSample = f32> {
    /// State-transition matrix `F`.
    pub matrix: [[T; N]; N],
    /// Control matrix `B`, `N` rows by `U` columns.
    pub control: [[T; U]; N],
}

impl<const N: usize, const U: usize, T: DspSample> ControlModel<N, U, T> {
    /// Construct a controlled linear model.
    #[must_use]
    pub fn new(matrix: [[T; N]; N], control: [[T; U]; N]) -> Self {
        Self { matrix, control }
    }
}

impl<const N: usize, const U: usize, T: DspSample> Dynamics<T, N> for ControlModel<N, U, T> {
    type Input = [T; U];

    fn advance(&self, state: &[T; N], input: [T; U]) -> [T; N] {
        let mut out = mat_vec(&self.matrix, state);
        let bu = mat_vec(&self.control, &input);
        for (o, b) in out.iter_mut().zip(bu) {
            *o = *o + b;
        }
        out
    }

    fn jacobian(&self, _state: &[T; N], _input: [T; U]) -> [[T; N]; N] {
        self.matrix
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phases
// ─────────────────────────────────────────────────────────────────────────────

/// Prediction phase: advance an estimate in time.
pub trait Predict<T: DspSample, const N: usize> {
    /// Control input consumed by this phase (`()` when there is none).
    type Input: Copy;

    /// Advance `estimate` one step in place.
    fn predict(&self, estimate: &mut Estimate<T, N>, input: Self::Input);
}

/// Measurement phase: fold one measurement into an estimate.
pub trait Update<T: DspSample, const N: usize> {
    /// Measurement consumed by this phase.
    type Measurement: Copy;

    /// The measurement the current estimate predicts.
    fn project(&self, estimate: &Estimate<T, N>) -> Self::Measurement;

    /// Correct `estimate` with `measurement`, returning the posterior measurement estimate.
    ///
    /// # Errors
    ///
    /// [`Status::Singular`] when the innovation covariance is not invertible. The estimate is
    /// left untouched in that case, so the caller can keep predicting.
    fn update(
        &self,
        estimate: &mut Estimate<T, N>,
        measurement: Self::Measurement,
    ) -> Result<Self::Measurement, Status>;
}

/// Predict with `x ← f(x, u)` and `P ← F P Fᵀ + Q`, with `f` and `F` taken at the prior state.
fn predict_with<T: DspSample, const N: usize, D: Dynamics<T, N>>(
    estimate: &mut Estimate<T, N>,
    dynamics: &D,
    input: D::Input,
    noise: &[[T; N]; N],
) {
    let f = dynamics.jacobian(&estimate.state, input);
    let next = dynamics.advance(&estimate.state, input);

    let fp = mat_mul(&f, &estimate.covariance);
    let mut p = mat_mul_bt(&fp, &f);
    for (row, qrow) in p.iter_mut().zip(noise) {
        for (c, q) in row.iter_mut().zip(qrow) {
            *c = *c + *q;
        }
    }

    estimate.state = next;
    estimate.covariance = p;
}

/// Linear transition with process noise: a [`Dynamics`] plus `Q`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transition<D, const N: usize, T: DspSample = f32> {
    /// State evolution model.
    pub dynamics: D,
    /// Symmetric process-noise covariance `Q`.
    pub noise: [[T; N]; N],
}

impl<D, const N: usize, T: DspSample> Transition<D, N, T> {
    /// Construct a transition from a model and its process noise.
    #[must_use]
    pub fn new(dynamics: D, noise: [[T; N]; N]) -> Self {
        Self { dynamics, noise }
    }
}

impl<D, const N: usize, T: DspSample> Predict<T, N> for Transition<D, N, T>
where
    D: Dynamics<T, N>,
{
    type Input = D::Input;

    fn predict(&self, estimate: &mut Estimate<T, N>, input: D::Input) {
        predict_with(estimate, &self.dynamics, input, &self.noise);
    }
}

/// Process noise equal to the identity scaled by a variance, for `Transition::identity`.
impl<const N: usize, T: DspSample> Transition<[[T; N]; N], N, T> {
    /// An identity transition (`F = I`) with the supplied process noise.
    #[must_use]
    pub fn identity(noise: [[T; N]; N]) -> Self {
        Self::new(identity_mat(), noise)
    }
}

/// `A - K (H P)`-style covariance update shared by the vector and scalar paths.
fn subtract_gain_terms<T: DspSample, const N: usize, const M: usize>(
    covariance: &[[T; N]; N],
    gain: &[[T; M]; N],
    h_p: &[[T; N]; M],
) -> [[T; N]; N] {
    let mut out = [[T::ZERO; N]; N];
    for r in 0..N {
        for c in 0..N {
            let mut acc = covariance[r][c];
            for m in 0..M {
                acc = acc - gain[r][m] * h_p[m][c];
            }
            out[r][c] = acc;
        }
    }
    out
}

/// Scalar measurement model: a row `H` and a variance `R`.
///
/// `M = 1` needs no matrix inverse — the innovation covariance is a single number — so this is
/// both the cheapest and the most numerically direct path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation<T: DspSample, const N: usize> {
    /// Measurement row `H`.
    pub matrix: [T; N],
    /// Measurement-noise variance `R`.
    pub noise: T,
}

impl<T: DspSample, const N: usize> Observation<T, N> {
    /// Construct a scalar observation model.
    #[must_use]
    pub fn new(matrix: [T; N], noise: T) -> Self {
        Self { matrix, noise }
    }
}

impl<T: DspSample, const N: usize> Update<T, N> for Observation<T, N> {
    type Measurement = T;

    fn project(&self, estimate: &Estimate<T, N>) -> T {
        dot(&self.matrix, &estimate.state)
    }

    fn update(
        &self,
        estimate: &mut Estimate<T, N>,
        measurement: T,
    ) -> Result<T, Status> {
        // P Hᵀ, then the scalar innovation covariance S = H P Hᵀ + R.
        let mut hp = [T::ZERO; N];
        for (o, row) in hp.iter_mut().zip(&estimate.covariance) {
            *o = dot(row, &self.matrix);
        }
        let variance = dot(&self.matrix, &hp) + self.noise;
        if variance.abs_val() <= T::ZERO {
            return Err(Status::Singular);
        }

        let innovation = measurement - dot(&self.matrix, &estimate.state);
        let gain: [T; N] = core::array::from_fn(|i| hp[i].sat_div(variance));

        for (x, k) in estimate.state.iter_mut().zip(gain) {
            *x = *x + k * innovation;
        }

        // P ← (I - K H) P, where (K H)[r][t] = K[r] * H[t].
        let p = estimate.covariance;
        let mut out = [[T::ZERO; N]; N];
        for (r, orow) in out.iter_mut().enumerate() {
            for (c, o) in orow.iter_mut().enumerate() {
                let mut acc = T::ZERO;
                for t in 0..N {
                    let mut ikh = T::ZERO - gain[r] * self.matrix[t];
                    if r == t {
                        ikh = ikh + T::ONE;
                    }
                    acc = acc + ikh * p[t][c];
                }
                *o = acc;
            }
        }
        estimate.covariance = out;

        Ok(dot(&self.matrix, &estimate.state))
    }
}

/// `M`-dimensional measurement model: a matrix `H` and a covariance `R`.
///
/// The measurement dimension is a const parameter, so `N`-state/multi-sensor updates stay on the
/// stack. `M = 1` is legal but [`Observation`] is cheaper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorObservation<const N: usize, const M: usize, T: DspSample = f32> {
    /// Measurement matrix `H`, `M` rows by `N` columns.
    pub matrix: [[T; N]; M],
    /// Symmetric measurement-noise covariance `R`.
    pub noise: [[T; M]; M],
}

impl<const N: usize, const M: usize, T: DspSample> VectorObservation<N, M, T> {
    /// Construct a vector observation model.
    #[must_use]
    pub fn new(matrix: [[T; N]; M], noise: [[T; M]; M]) -> Self {
        Self { matrix, noise }
    }
}

impl<const N: usize, const M: usize, T: DspSample> Update<T, N> for VectorObservation<N, M, T> {
    type Measurement = [T; M];

    fn project(&self, estimate: &Estimate<T, N>) -> [T; M] {
        mat_vec(&self.matrix, &estimate.state)
    }

    fn update(
        &self,
        estimate: &mut Estimate<T, N>,
        measurement: [T; M],
    ) -> Result<[T; M], Status> {
        // S = H P Hᵀ + R.
        let h_p = mat_mul(&self.matrix, &estimate.covariance);
        let mut s = mat_mul_bt::<T, M, N, M>(&h_p, &self.matrix);
        for (row, nrow) in s.iter_mut().zip(&self.noise) {
            for (c, n) in row.iter_mut().zip(nrow) {
                *c = *c + *n;
            }
        }
        let s_inv = invert(&s).ok_or(Status::Singular)?;

        let hx = mat_vec(&self.matrix, &estimate.state);
        let mut innovation = [T::ZERO; M];
        for (i, (z, p)) in innovation.iter_mut().zip(measurement.iter().zip(hx)) {
            *i = *z - p;
        }

        // K = P Hᵀ S⁻¹.
        let p_ht = mat_mul_bt::<T, N, N, M>(&estimate.covariance, &self.matrix);
        let gain = mat_mul(&p_ht, &s_inv);

        let delta = mat_vec(&gain, &innovation);
        for (x, d) in estimate.state.iter_mut().zip(delta) {
            *x = *x + d;
        }
        estimate.covariance = subtract_gain_terms(&estimate.covariance, &gain, &h_p);

        Ok(mat_vec(&self.matrix, &estimate.state))
    }
}

/// Direct observation of state component `I`, with measurement variance `R`.
///
/// This carries no `H` at all: every `H` product collapses to indexing, so it is both cheaper
/// and exactly symmetric with [`VectorObservation`] where `H` is the `I`-th unit row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Direct<T: DspSample, const N: usize, const I: usize> {
    /// Measurement-noise variance `R`.
    pub noise: T,
}

impl<T: DspSample, const N: usize, const I: usize> Direct<T, N, I> {
    /// Construct a direct observation of component `I`.
    #[must_use]
    pub fn new(noise: T) -> Self {
        Self { noise }
    }
}

impl<T: DspSample, const N: usize, const I: usize> Update<T, N> for Direct<T, N, I> {
    type Measurement = T;

    fn project(&self, estimate: &Estimate<T, N>) -> T {
        estimate.state[I]
    }

    fn update(
        &self,
        estimate: &mut Estimate<T, N>,
        measurement: T,
    ) -> Result<T, Status> {
        let p = estimate.covariance;
        let variance = p[I][I] + self.noise;
        if variance.abs_val() <= T::ZERO {
            return Err(Status::Singular);
        }

        // P Hᵀ is the I-th column of P, and H P is the I-th row.
        let innovation = measurement - estimate.state[I];
        for (i, x) in estimate.state.iter_mut().enumerate() {
            *x = *x + p[i][I].sat_div(variance) * innovation;
        }

        let mut out = [[T::ZERO; N]; N];
        for r in 0..N {
            let gain = p[r][I].sat_div(variance);
            for c in 0..N {
                out[r][c] = p[r][c] - gain * p[I][c];
            }
        }
        estimate.covariance = out;

        Ok(estimate.state[I])
    }
}

/// Wraps a measurement phase so that the measurement may be absent.
///
/// With this, a phase pair keeps predicting when a sensor drops out instead of stalling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Optional<U>(pub U);

impl<T: DspSample, const N: usize, U: Update<T, N>> Update<T, N> for Optional<U> {
    type Measurement = Option<U::Measurement>;

    fn project(&self, estimate: &Estimate<T, N>) -> Option<U::Measurement> {
        Some(self.0.project(estimate))
    }

    fn update(
        &self,
        estimate: &mut Estimate<T, N>,
        measurement: Option<U::Measurement>,
    ) -> Result<Option<U::Measurement>, Status> {
        match measurement {
            Some(m) => self.0.update(estimate, m).map(Some),
            None => Ok(None),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The filter
// ─────────────────────────────────────────────────────────────────────────────

/// A prediction phase and a measurement phase, plus the estimate they operate on.
///
/// The phases are plain data and are `Copy` when their models are, so one model can drive many
/// filters: build the [`Transition`]/[`Observation`] pair once and use it for every instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kalman<P, U, T: DspSample, const N: usize> {
    /// Prediction phase.
    pub predict: P,
    /// Measurement phase.
    pub update: U,
    /// The estimate being tracked.
    pub estimate: Estimate<T, N>,
}

impl<P, U, T: DspSample, const N: usize> Kalman<P, U, T, N> {
    /// Compose a filter from a prediction phase, a measurement phase and an initial estimate.
    #[must_use]
    pub fn new(predict: P, update: U, estimate: Estimate<T, N>) -> Self {
        Self {
            predict,
            update,
            estimate,
        }
    }

    /// Wrap the measurement phase so a missing measurement still predicts.
    #[must_use]
    pub fn optional(self) -> Kalman<P, Optional<U>, T, N>
    where
        U: Update<T, N>,
    {
        Kalman::new(self.predict, Optional(self.update), self.estimate)
    }
}

impl<P, U, T: DspSample, const N: usize> Kalman<P, U, T, N>
where
    P: Predict<T, N>,
    U: Update<T, N>,
{
    /// Predict, then correct with one measurement.
    ///
    /// Returns the posterior measurement estimate, or [`Status::Singular`] from the measurement
    /// phase (the prediction has still been applied in that case).
    ///
    /// # Errors
    ///
    /// Propagates the measurement phase's error; see [`Update::update`].
    pub fn step(
        &mut self,
        input: P::Input,
        measurement: U::Measurement,
    ) -> Result<U::Measurement, Status> {
        self.predict.predict(&mut self.estimate, input);
        self.update.update(&mut self.estimate, measurement)
    }

    /// Predict only, i.e. what [`step`](Kalman::step) does before the measurement arrives.
    pub fn advance(&mut self, input: P::Input) {
        self.predict.predict(&mut self.estimate, input);
    }

    /// The measurement the current estimate predicts.
    #[must_use]
    pub fn project(&self) -> U::Measurement {
        self.update.project(&self.estimate)
    }
}

/// The smallest complete filter: one state, `F = H = 1`.
impl<T: DspSample> Kalman<Transition<[[T; 1]; 1], 1, T>, Observation<T, 1>, T, 1> {
    /// Scalar random walk with the given process and measurement noise.
    #[must_use]
    pub fn random_walk(
        process_noise: T,
        measurement_noise: T,
        initial: Estimate<T, 1>,
    ) -> Self {
        Kalman::new(
            Transition::new([[T::ONE]], [[process_noise]]),
            Observation::new([T::ONE], measurement_noise),
            initial,
        )
    }
}

/// The common position/velocity tracker: [`ConstantVelocity`] observed directly at position.
impl<T: DspSample> Kalman<Transition<ConstantVelocity<T>, 2, T>, Observation<T, 2>, T, 2> {
    /// Constant-velocity tracker observing position, i.e. `H = [1, 0]`.
    #[must_use]
    pub fn constant_velocity(
        interval: T,
        process_noise: [[T; 2]; 2],
        measurement_noise: T,
        initial: Estimate<T, 2>,
    ) -> Self {
        Kalman::new(
            Transition::new(ConstantVelocity::new(interval), process_noise),
            Observation::new([T::ONE, T::ZERO], measurement_noise),
            initial,
        )
    }
}
