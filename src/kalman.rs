//! Kalman filtering and state estimation algorithms for zero-allocation embedded applications.

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
