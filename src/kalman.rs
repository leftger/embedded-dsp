//! Kalman filtering and state estimation algorithms for zero-allocation embedded applications.
//!
//! Includes convenience 1D/2D filters, a const-generic linear [`KalmanFilter`], and an
//! [`ExtendedKalmanFilter`] driven by a user [`EkfModel`]. Measurement dimension `M` must be
//! ≤ 16 (same limit as [`crate::matrix::mat_inverse_f32`]). Covariance updates use
//! `P ← (I − KH) P`; a Joseph-form update may be added later for improved numerical stability.
//!
//! `EkfModel::f`/`h` only see the state (plus `dt` for `f`), which doesn't fit models whose
//! process or measurement equations depend on an exogenous input that isn't part of the state
//! (a commanded actuation, a measured current used for an IR-drop correction, etc). For that,
//! implement the `_with_input` trait methods and drive the filter with
//! [`ExtendedKalmanFilter::predict_with_input`] / [`ExtendedKalmanFilter::update_with_input`].
//! Their default implementations ignore `u` and defer to `f`/`h`/the Jacobians, so existing
//! [`EkfModel`] implementations keep compiling unchanged.

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::matrix::{MatrixInstance, MatrixInstanceMut, mat_inverse_f32};
use crate::types::Status;

/// Scalar (1D) Kalman filter for single-variable sensor smoothing and estimation.
#[derive(Debug, Clone, Copy)]
pub struct KalmanFilter1D {
    /// Estimated state
    pub x: f32,
    /// Estimation error covariance
    pub p: f32,
    /// Process noise covariance
    pub q: f32,
    /// Measurement noise covariance
    pub r: f32,
}

impl KalmanFilter1D {
    /// Initialises a 1D Kalman filter with initial estimate `x0`, initial covariance `p0`, process noise `q`, and measurement noise `r`.
    pub fn new(x0: f32, p0: f32, q: f32, r: f32) -> Self {
        Self { x: x0, p: p0, q, r }
    }

    /// Prediction step incorporating process input `u` (optional control input).
    pub fn predict(&mut self, u: f32) {
        self.x += u;
        self.p += self.q;
    }

    /// Measurement update step with new sensor reading `z`. Returns updated state estimate.
    pub fn update(&mut self, z: f32) -> f32 {
        let k = self.p / (self.p + self.r);
        self.x += k * (z - self.x);
        self.p = (1.0 - k) * self.p;
        self.x
    }
}

/// 2-State (Position + Velocity) linear Kalman filter for motion tracking and sensor fusion.
#[derive(Debug, Clone, Copy)]
pub struct KalmanFilter2D {
    /// State vector: `[position, velocity]`
    pub x: [f32; 2],
    /// 2x2 State covariance matrix (row-major: `[p00, p01, p10, p11]`)
    pub p: [f32; 4],
    /// Process noise variance
    pub q_var: f32,
    /// Measurement noise variance
    pub r_var: f32,
}

impl KalmanFilter2D {
    /// Initialise a 2D position/velocity Kalman filter.
    pub fn new(initial_pos: f32, initial_vel: f32, q_var: f32, r_var: f32) -> Self {
        Self {
            x: [initial_pos, initial_vel],
            p: [1.0, 0.0, 0.0, 1.0],
            q_var,
            r_var,
        }
    }

    /// Predict state forward by time delta `dt`.
    pub fn predict(&mut self, dt: f32) {
        // State transition: x_pos = x_pos + dt * x_vel
        self.x[0] += dt * self.x[1];

        // P_new = F * P * F^T + Q
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;

        let p00 =
            self.p[0] + dt * (self.p[2] + self.p[1]) + dt2 * self.p[3] + 0.25 * dt4 * self.q_var;
        let p01 = self.p[1] + dt * self.p[3] + 0.5 * dt3 * self.q_var;
        let p10 = self.p[2] + dt * self.p[3] + 0.5 * dt3 * self.q_var;
        let p11 = self.p[3] + dt2 * self.q_var;

        self.p = [p00, p01, p10, p11];
    }

    /// Update filter with position measurement `z_pos`. Returns updated position and velocity `[pos, vel]`.
    pub fn update(&mut self, z_pos: f32) -> [f32; 2] {
        // Innovation
        let y = z_pos - self.x[0];
        let s = self.p[0] + self.r_var;

        // Kalman gain K = P * H^T / S  (where H = [1, 0])
        let k0 = self.p[0] / s;
        let k1 = self.p[2] / s;

        // State update
        self.x[0] += k0 * y;
        self.x[1] += k1 * y;

        // Covariance update: P = (I - K * H) * P
        let p00 = (1.0 - k0) * self.p[0];
        let p01 = (1.0 - k0) * self.p[1];
        let p10 = self.p[2] - k1 * self.p[0];
        let p11 = self.p[3] - k1 * self.p[1];

        self.p = [p00, p01, p10, p11];
        self.x
    }
}

// --- Const-generic linear Kalman & EKF helpers ---

#[inline]
fn mat_vec_mul<const R: usize, const C: usize>(
    a: &[[f32; C]; R],
    x: &[f32; C],
    out: &mut [f32; R],
) {
    for r in 0..R {
        let mut sum = 0.0f32;
        for c in 0..C {
            sum += a[r][c] * x[c];
        }
        out[r] = sum;
    }
}

#[inline]
fn mat_mul<const R: usize, const K: usize, const C: usize>(
    a: &[[f32; K]; R],
    b: &[[f32; C]; K],
    out: &mut [[f32; C]; R],
) {
    for r in 0..R {
        for c in 0..C {
            let mut sum = 0.0f32;
            for k in 0..K {
                sum += a[r][k] * b[k][c];
            }
            out[r][c] = sum;
        }
    }
}

/// Computes `out = a * b^T` where `a` is R×K and `b` is C×K (so `b^T` is K×C).
#[inline]
fn mat_mul_bt<const R: usize, const K: usize, const C: usize>(
    a: &[[f32; K]; R],
    b: &[[f32; K]; C],
    out: &mut [[f32; C]; R],
) {
    for r in 0..R {
        for c in 0..C {
            let mut sum = 0.0f32;
            for k in 0..K {
                sum += a[r][k] * b[c][k];
            }
            out[r][c] = sum;
        }
    }
}

#[inline]
fn mat_add_inplace_nn<const N: usize>(a: &mut [[f32; N]; N], b: &[[f32; N]; N]) {
    for r in 0..N {
        for c in 0..N {
            a[r][c] += b[r][c];
        }
    }
}

#[inline]
fn mat_add_inplace_mm<const M: usize>(a: &mut [[f32; M]; M], b: &[[f32; M]; M]) {
    for r in 0..M {
        for c in 0..M {
            a[r][c] += b[r][c];
        }
    }
}

#[inline]
fn identity_n<const N: usize>() -> [[f32; N]; N] {
    let mut i = [[0.0f32; N]; N];
    for n in 0..N {
        i[n][n] = 1.0;
    }
    i
}

/// Invert an `M×M` matrix using [`mat_inverse_f32`]. Requires `M ≤ 16`.
fn invert_mxm<const M: usize>(s: &[[f32; M]; M], s_inv: &mut [[f32; M]; M]) -> Status {
    if M == 0 {
        return Status::SizeMismatch;
    }
    if M > 16 {
        return Status::ArgumentError;
    }

    let mut flat_src = [0.0f32; 16 * 16];
    let mut flat_dst = [0.0f32; 16 * 16];
    for r in 0..M {
        for c in 0..M {
            flat_src[r * M + c] = s[r][c];
        }
    }

    let src = MatrixInstance::new(M as u16, M as u16, &flat_src[..M * M]);
    let mut dst = MatrixInstanceMut::new(M as u16, M as u16, &mut flat_dst[..M * M]);
    let status = mat_inverse_f32(&src, &mut dst);
    if status != Status::Success {
        return status;
    }

    for r in 0..M {
        for c in 0..M {
            s_inv[r][c] = flat_dst[r * M + c];
        }
    }
    Status::Success
}

/// Predict: `x ← F x`, `P ← F P Fᵀ + Q`.
fn kf_predict_core<const N: usize>(
    x: &mut [f32; N],
    p: &mut [[f32; N]; N],
    q: &[[f32; N]; N],
    f: &[[f32; N]; N],
) {
    let mut x_new = [0.0f32; N];
    mat_vec_mul(f, x, &mut x_new);
    *x = x_new;

    let mut fp = [[0.0f32; N]; N];
    mat_mul(f, p, &mut fp);
    let mut p_new = [[0.0f32; N]; N];
    mat_mul_bt(&fp, f, &mut p_new);
    mat_add_inplace_nn(&mut p_new, q);
    *p = p_new;
}

/// Predict with control: `x ← F x + B u`, then same `P` update.
fn kf_predict_control_core<const N: usize, const U: usize>(
    x: &mut [f32; N],
    p: &mut [[f32; N]; N],
    q: &[[f32; N]; N],
    f: &[[f32; N]; N],
    b: &[[f32; U]; N],
    u: &[f32; U],
) {
    let mut x_new = [0.0f32; N];
    mat_vec_mul(f, x, &mut x_new);
    let mut bu = [0.0f32; N];
    mat_vec_mul(b, u, &mut bu);
    for i in 0..N {
        x_new[i] += bu[i];
    }
    *x = x_new;

    let mut fp = [[0.0f32; N]; N];
    mat_mul(f, p, &mut fp);
    let mut p_new = [[0.0f32; N]; N];
    mat_mul_bt(&fp, f, &mut p_new);
    mat_add_inplace_nn(&mut p_new, q);
    *p = p_new;
}

/// Measurement update with linear `H`. Leaves state unchanged on singular `S`.
fn kf_update_core<const N: usize, const M: usize>(
    x: &mut [f32; N],
    p: &mut [[f32; N]; N],
    r: &[[f32; M]; M],
    h: &[[f32; N]; M],
    z: &[f32; M],
) -> Status {
    if M > 16 {
        return Status::ArgumentError;
    }
    if M == 0 {
        return Status::SizeMismatch;
    }

    // y = z - H x
    let mut hx = [0.0f32; M];
    mat_vec_mul(h, x, &mut hx);
    let mut y = [0.0f32; M];
    for i in 0..M {
        y[i] = z[i] - hx[i];
    }

    // S = H P Hᵀ + R
    let mut hp = [[0.0f32; N]; M];
    mat_mul(h, p, &mut hp);
    let mut s = [[0.0f32; M]; M];
    mat_mul_bt(&hp, h, &mut s);
    mat_add_inplace_mm(&mut s, r);

    let mut s_inv = [[0.0f32; M]; M];
    let inv_status = invert_mxm(&s, &mut s_inv);
    if inv_status != Status::Success {
        return inv_status;
    }

    // P Hᵀ (N×M): rows of P times columns of Hᵀ (= rows of H)
    let mut pht = [[0.0f32; M]; N];
    for i in 0..N {
        for j in 0..M {
            let mut sum = 0.0f32;
            for k in 0..N {
                sum += p[i][k] * h[j][k];
            }
            pht[i][j] = sum;
        }
    }

    // K = (P Hᵀ) S⁻¹  (N×M)
    let mut k = [[0.0f32; M]; N];
    mat_mul(&pht, &s_inv, &mut k);

    // x ← x + K y
    let mut ky = [0.0f32; N];
    mat_vec_mul(&k, &y, &mut ky);
    let mut x_new = *x;
    for i in 0..N {
        x_new[i] += ky[i];
    }

    // P ← (I - K H) P
    let mut kh = [[0.0f32; N]; N];
    mat_mul(&k, h, &mut kh);
    let mut i_kh = identity_n::<N>();
    for r in 0..N {
        for c in 0..N {
            i_kh[r][c] -= kh[r][c];
        }
    }
    let mut p_new = [[0.0f32; N]; N];
    mat_mul(&i_kh, p, &mut p_new);

    *x = x_new;
    *p = p_new;
    Status::Success
}

/// Const-generic linear Kalman filter: `x' = F x (+ B u) + w`, `z = H x + v`.
///
/// Measurement dimension `M` must be ≤ 16 so the innovation covariance can be inverted
/// with the crate's stack-limited matrix inverse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanFilter<const N: usize, const M: usize> {
    /// State estimate
    pub x: [f32; N],
    /// State covariance `P` (`N×N`)
    pub p: [[f32; N]; N],
    /// Process noise covariance `Q` (`N×N`)
    pub q: [[f32; N]; N],
    /// Measurement noise covariance `R` (`M×M`)
    pub r: [[f32; M]; M],
}

impl<const N: usize, const M: usize> KalmanFilter<N, M> {
    /// Create a filter with initial state `x0`, covariance `p0`, and noise covariances `q` / `r`.
    pub fn new(x0: [f32; N], p0: [[f32; N]; N], q: [[f32; N]; N], r: [[f32; M]; M]) -> Self {
        Self { x: x0, p: p0, q, r }
    }

    /// Create a filter with diagonal `P`, `Q`, and `R` initialized from scalar variances.
    pub fn from_variances(x0: [f32; N], p_var: f32, q_var: f32, r_var: f32) -> Self {
        let mut p = [[0.0f32; N]; N];
        let mut q = [[0.0f32; N]; N];
        let mut r = [[0.0f32; M]; M];
        for i in 0..N {
            p[i][i] = p_var;
            q[i][i] = q_var;
        }
        for i in 0..M {
            r[i][i] = r_var;
        }
        Self::new(x0, p, q, r)
    }

    /// Prediction without control input: `x ← F x`, `P ← F P Fᵀ + Q`.
    pub fn predict(&mut self, f: &[[f32; N]; N]) {
        kf_predict_core(&mut self.x, &mut self.p, &self.q, f);
    }

    /// Prediction with control: `x ← F x + B u`, `P ← F P Fᵀ + Q`.
    pub fn predict_with_control<const U: usize>(
        &mut self,
        f: &[[f32; N]; N],
        b: &[[f32; U]; N],
        u: &[f32; U],
    ) {
        kf_predict_control_core(&mut self.x, &mut self.p, &self.q, f, b, u);
    }

    /// Measurement update with observation matrix `H` (`M×N`) and measurement `z`.
    ///
    /// On success returns [`Status::Success`] and updates `x` / `P`. If `S` is singular or
    /// `M > 16`, returns an error status and leaves the filter state unchanged.
    pub fn update(&mut self, h: &[[f32; N]; M], z: &[f32; M]) -> Status {
        kf_update_core(&mut self.x, &mut self.p, &self.r, h, z)
    }
}

/// User-supplied nonlinear process and measurement model for an EKF (static dispatch).
pub trait EkfModel<const N: usize, const M: usize> {
    /// Process model: `out = f(x, dt)`.
    fn f(&self, x: &[f32; N], dt: f32, out: &mut [f32; N]);

    /// Measurement model: `out = h(x)`.
    fn h(&self, x: &[f32; N], out: &mut [f32; M]);

    /// Process Jacobian `F = ∂f/∂x` evaluated at `x`.
    fn jacobian_f(&self, x: &[f32; N], dt: f32, out: &mut [[f32; N]; N]);

    /// Measurement Jacobian `H = ∂h/∂x` evaluated at `x` (`M×N`).
    fn jacobian_h(&self, x: &[f32; N], out: &mut [[f32; N]; M]);

    /// Process model with an explicit exogenous input `u` (a control input,
    /// measured disturbance, or anything else that drives `f` but isn't
    /// part of the state): `out = f(x, u, dt)`.
    ///
    /// Default: ignores `u` and defers to [`EkfModel::f`], so models that
    /// don't need an input compile unchanged.
    fn f_with_input<const U: usize>(
        &self,
        x: &[f32; N],
        u: &[f32; U],
        dt: f32,
        out: &mut [f32; N],
    ) {
        let _ = u;
        self.f(x, dt, out)
    }

    /// Process Jacobian for [`EkfModel::f_with_input`], `F = ∂f/∂x` evaluated at `(x, u)`.
    ///
    /// Default: defers to [`EkfModel::jacobian_f`], which is exact whenever `u` enters `f`
    /// affinely (so it doesn't change the derivative with respect to `x`).
    fn jacobian_f_with_input<const U: usize>(
        &self,
        x: &[f32; N],
        u: &[f32; U],
        dt: f32,
        out: &mut [[f32; N]; N],
    ) {
        let _ = u;
        self.jacobian_f(x, dt, out)
    }

    /// Measurement model with an explicit exogenous input `u` (e.g. a measured current used
    /// for an IR-drop correction that isn't part of the state): `out = h(x, u)`.
    ///
    /// Default: ignores `u` and defers to [`EkfModel::h`].
    fn h_with_input<const U: usize>(&self, x: &[f32; N], u: &[f32; U], out: &mut [f32; M]) {
        let _ = u;
        self.h(x, out)
    }

    /// Measurement Jacobian for [`EkfModel::h_with_input`], `H = ∂h/∂x` evaluated at `(x, u)`.
    ///
    /// Default: defers to [`EkfModel::jacobian_h`], which is exact whenever `u` enters `h`
    /// affinely.
    fn jacobian_h_with_input<const U: usize>(
        &self,
        x: &[f32; N],
        u: &[f32; U],
        out: &mut [[f32; N]; M],
    ) {
        let _ = u;
        self.jacobian_h(x, out)
    }
}

/// Extended Kalman filter with compile-time dimensions and a user [`EkfModel`].
///
/// Measurement dimension `M` must be ≤ 16. Covariance update uses `P ← (I − KH) P`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtendedKalmanFilter<const N: usize, const M: usize, Model> {
    /// State estimate
    pub x: [f32; N],
    /// State covariance `P` (`N×N`)
    pub p: [[f32; N]; N],
    /// Process noise covariance `Q` (`N×N`)
    pub q: [[f32; N]; N],
    /// Measurement noise covariance `R` (`M×M`)
    pub r: [[f32; M]; M],
    /// Nonlinear process / measurement model
    pub model: Model,
}

impl<const N: usize, const M: usize, Model: EkfModel<N, M>> ExtendedKalmanFilter<N, M, Model> {
    /// Create an EKF with initial state, covariances, and model.
    pub fn new(
        x0: [f32; N],
        p0: [[f32; N]; N],
        q: [[f32; N]; N],
        r: [[f32; M]; M],
        model: Model,
    ) -> Self {
        Self {
            x: x0,
            p: p0,
            q,
            r,
            model,
        }
    }

    /// Create an EKF with diagonal covariances from scalar variances.
    pub fn from_variances(x0: [f32; N], p_var: f32, q_var: f32, r_var: f32, model: Model) -> Self {
        let mut p = [[0.0f32; N]; N];
        let mut q = [[0.0f32; N]; N];
        let mut r = [[0.0f32; M]; M];
        for i in 0..N {
            p[i][i] = p_var;
            q[i][i] = q_var;
        }
        for i in 0..M {
            r[i][i] = r_var;
        }
        Self::new(x0, p, q, r, model)
    }

    /// EKF predict: `x ← f(x, dt)`, `P ← F P Fᵀ + Q` with `F = ∂f/∂x`.
    pub fn predict(&mut self, dt: f32) {
        let mut f_jac = [[0.0f32; N]; N];
        self.model.jacobian_f(&self.x, dt, &mut f_jac);

        let mut x_new = [0.0f32; N];
        self.model.f(&self.x, dt, &mut x_new);

        ekf_predict_apply(&mut self.x, &mut self.p, &self.q, &f_jac, x_new);
    }

    /// EKF predict with an exogenous input `u`, via [`EkfModel::f_with_input`] /
    /// [`EkfModel::jacobian_f_with_input`]. See the [module docs](self) for when this is
    /// needed instead of [`ExtendedKalmanFilter::predict`].
    pub fn predict_with_input<const U: usize>(&mut self, dt: f32, u: &[f32; U]) {
        let mut f_jac = [[0.0f32; N]; N];
        self.model.jacobian_f_with_input(&self.x, u, dt, &mut f_jac);

        let mut x_new = [0.0f32; N];
        self.model.f_with_input(&self.x, u, dt, &mut x_new);

        ekf_predict_apply(&mut self.x, &mut self.p, &self.q, &f_jac, x_new);
    }

    /// EKF update with measurement `z`. Linearizes `h` at the current estimate.
    ///
    /// On singular innovation covariance or `M > 16`, returns an error and leaves state unchanged.
    pub fn update(&mut self, z: &[f32; M]) -> Status {
        let mut h_jac = [[0.0f32; N]; M];
        self.model.jacobian_h(&self.x, &mut h_jac);

        let mut hx = [0.0f32; M];
        self.model.h(&self.x, &mut hx);

        ekf_update_apply(&mut self.x, &mut self.p, &self.r, &h_jac, &hx, z)
    }

    /// EKF update with an exogenous input `u`, via [`EkfModel::h_with_input`] /
    /// [`EkfModel::jacobian_h_with_input`]. See the [module docs](self) for when this is
    /// needed instead of [`ExtendedKalmanFilter::update`].
    ///
    /// On singular innovation covariance or `M > 16`, returns an error and leaves state unchanged.
    pub fn update_with_input<const U: usize>(&mut self, z: &[f32; M], u: &[f32; U]) -> Status {
        let mut h_jac = [[0.0f32; N]; M];
        self.model.jacobian_h_with_input(&self.x, u, &mut h_jac);

        let mut hx = [0.0f32; M];
        self.model.h_with_input(&self.x, u, &mut hx);

        ekf_update_apply(&mut self.x, &mut self.p, &self.r, &h_jac, &hx, z)
    }
}

/// Shared EKF predict math: `x ← x_new`, `P ← F P Fᵀ + Q`. Factored out so
/// [`ExtendedKalmanFilter::predict`] and [`ExtendedKalmanFilter::predict_with_input`] (which
/// differ only in how `x_new`/`f_jac` are computed) don't duplicate the covariance propagation.
fn ekf_predict_apply<const N: usize>(
    x: &mut [f32; N],
    p: &mut [[f32; N]; N],
    q: &[[f32; N]; N],
    f_jac: &[[f32; N]; N],
    x_new: [f32; N],
) {
    *x = x_new;

    let mut fp = [[0.0f32; N]; N];
    mat_mul(f_jac, p, &mut fp);
    let mut p_new = [[0.0f32; N]; N];
    mat_mul_bt(&fp, f_jac, &mut p_new);
    mat_add_inplace_nn(&mut p_new, q);
    *p = p_new;
}

/// Shared EKF update math: linearizes around `hx = h(x)` and reuses the linear-filter update
/// core. Factored out so [`ExtendedKalmanFilter::update`] and
/// [`ExtendedKalmanFilter::update_with_input`] (which differ only in how `hx`/`h_jac` are
/// computed) don't duplicate the linearization.
fn ekf_update_apply<const N: usize, const M: usize>(
    x: &mut [f32; N],
    p: &mut [[f32; N]; N],
    r: &[[f32; M]; M],
    h_jac: &[[f32; N]; M],
    hx: &[f32; M],
    z: &[f32; M],
) -> Status {
    if M > 16 {
        return Status::ArgumentError;
    }
    if M == 0 {
        return Status::SizeMismatch;
    }

    // Reuse linear update with innovation z' = z - h(x) + H x so that
    // y = z' - H x = z - h(x).
    let mut z_equiv = [0.0f32; M];
    let mut hx_lin = [0.0f32; M];
    mat_vec_mul(h_jac, x, &mut hx_lin);
    for i in 0..M {
        z_equiv[i] = z[i] - hx[i] + hx_lin[i];
    }

    kf_update_core(x, p, r, h_jac, &z_equiv)
}

// ─────────────────────────────────────────────────────────────────────────────
// Square-Root Covariance Kalman Filter (SRKF)
// ─────────────────────────────────────────────────────────────────────────────

/// Square-Root Covariance Kalman Filter (SRKF) for $N$-state, $M$-measurement linear systems.
///
/// Propagates the lower-triangular Cholesky factor $S$ of the covariance matrix ($P = S S^T$).
/// By operating directly on the square-root factors via orthogonal Givens transformations,
/// the filter **guarantees numerical positive-definiteness and never diverges** due to roundoff error.
#[derive(Debug, Clone)]
pub struct SquareRootKalmanFilter<const N: usize, const M: usize> {
    /// State estimate vector $\hat{x} \in \mathbb{R}^N$.
    pub x: [f32; N],
    /// Lower-triangular Cholesky factor of state covariance $P = S S^T$.
    pub s: [[f32; N]; N],
    /// State transition matrix $F \in \mathbb{R}^{N \times N}$.
    pub f: [[f32; N]; N],
    /// Lower-triangular Cholesky factor of process noise covariance $Q = S_Q S_Q^T$.
    pub s_q: [[f32; N]; N],
    /// Measurement matrix $H \in \mathbb{R}^{M \times N}$.
    pub h: [[f32; N]; M],
    /// Lower-triangular Cholesky factor of measurement noise $R = S_R S_R^T$.
    pub s_r: [[f32; M]; M],
}

impl<const N: usize, const M: usize> SquareRootKalmanFilter<N, M> {
    /// Initialize a new Square-Root Kalman Filter from explicit Cholesky factors.
    pub fn new(
        x0: [f32; N],
        s0: [[f32; N]; N],
        f: [[f32; N]; N],
        s_q: [[f32; N]; N],
        h: [[f32; N]; M],
        s_r: [[f32; M]; M],
    ) -> Self {
        Self {
            x: x0,
            s: s0,
            f,
            s_q,
            h,
            s_r,
        }
    }

    /// Predict step: propagates state $\hat{x}^- = F \hat{x}$ and triangularizes $[F S \quad S_Q]$.
    pub fn predict(&mut self) {
        // 1. State prediction: x = F * x
        let mut x_new = [0.0f32; N];
        mat_vec_mul(&self.f, &self.x, &mut x_new);
        self.x = x_new;

        // 2. Covariance square-root prediction: S^- via Cholesky factor of FS(FS)^T + S_Q(S_Q)^T
        let mut fs = [[0.0f32; N]; N];
        mat_mul(&self.f, &self.s, &mut fs);

        let mut s_new = [[0.0f32; N]; N];
        for i in 0..N {
            for j in 0..=i {
                let mut sum = 0.0f32;
                for k in 0..N {
                    sum += fs[i][k] * fs[j][k] + self.s_q[i][k] * self.s_q[j][k];
                }
                s_new[i][j] = sum;
            }
        }
        cholesky_inplace_lower(&mut s_new);
        self.s = s_new;
    }

    /// Update step: updates state $\hat{x}^+$ and factor $S^+$ given measurement vector $z \in \mathbb{R}^M$.
    pub fn update(&mut self, z: &[f32; M]) -> Status {
        if M == 0 || M > 16 {
            return Status::ArgumentError;
        }

        // Innovation y = z - H x
        let mut hx = [0.0f32; M];
        mat_vec_mul(&self.h, &self.x, &mut hx);
        let mut y = [0.0f32; M];
        for i in 0..M {
            y[i] = z[i] - hx[i];
        }

        // Innovation covariance S_yy = H P H^T + R = (H S) (H S)^T + S_R S_R^T
        let mut hs = [[0.0f32; N]; M];
        mat_mul(&self.h, &self.s, &mut hs);

        let mut s_yy = [[0.0f32; M]; M];
        for r in 0..M {
            for c in 0..M {
                let mut sum = 0.0f32;
                for k in 0..N {
                    sum += hs[r][k] * hs[c][k];
                }
                for k in 0..M {
                    sum += self.s_r[r][k] * self.s_r[c][k];
                }
                s_yy[r][c] = sum;
            }
        }

        // Invert S_yy
        let mut s_yy_inv = [[0.0f32; M]; M];
        let status = invert_mxm(&s_yy, &mut s_yy_inv);
        if status != Status::Success {
            return status;
        }

        // Kalman gain: K = P H^T S_yy^-1 = S S^T H^T S_yy^-1
        let mut p = [[0.0f32; N]; N];
        mat_mul_bt(&self.s, &self.s, &mut p);

        let mut pht = [[0.0f32; M]; N];
        mat_mul_bt(&p, &self.h, &mut pht);

        let mut k_gain = [[0.0f32; M]; N];
        mat_mul(&pht, &s_yy_inv, &mut k_gain);

        // Update state: x = x + K y
        let mut ky = [0.0f32; N];
        mat_vec_mul(&k_gain, &y, &mut ky);
        for i in 0..N {
            self.x[i] += ky[i];
        }

        // Update covariance: P+ = (I - K H) P (I - K H)^T + K R K^T (Joseph form)
        let mut i_kh = identity_n::<N>();
        let mut kh = [[0.0f32; N]; N];
        mat_mul(&k_gain, &self.h, &mut kh);
        for r in 0..N {
            for c in 0..N {
                i_kh[r][c] -= kh[r][c];
            }
        }

        let mut i_kh_p = [[0.0f32; N]; N];
        mat_mul(&i_kh, &p, &mut i_kh_p);
        let mut p_plus = [[0.0f32; N]; N];
        mat_mul_bt(&i_kh_p, &i_kh, &mut p_plus);

        let mut r_mat = [[0.0f32; M]; M];
        mat_mul_bt(&self.s_r, &self.s_r, &mut r_mat);
        let mut kr = [[0.0f32; M]; N];
        mat_mul(&k_gain, &r_mat, &mut kr);
        let mut krkt = [[0.0f32; N]; N];
        mat_mul_bt(&kr, &k_gain, &mut krkt);
        mat_add_inplace_nn(&mut p_plus, &krkt);

        // Factor updated P+ into lower-triangular S+
        cholesky_inplace_lower(&mut p_plus);
        self.s = p_plus;

        Status::Success
    }

    /// Reconstructs the full covariance matrix $P = S S^T$.
    pub fn covariance(&self) -> [[f32; N]; N] {
        let mut p = [[0.0f32; N]; N];
        mat_mul_bt(&self.s, &self.s, &mut p);
        p
    }
}

/// Compute lower-triangular Cholesky factor $L$ in-place such that $A = L L^T$.
fn cholesky_inplace_lower<const N: usize>(a: &mut [[f32; N]; N]) {
    for i in 0..N {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= a[i][k] * a[j][k];
            }
            if i == j {
                a[i][j] = sum.max(1e-12).sqrt();
            } else {
                let diag = a[j][j].max(1e-12);
                a[i][j] = sum / diag;
            }
        }
        for j in (i + 1)..N {
            a[i][j] = 0.0;
        }
    }
}
