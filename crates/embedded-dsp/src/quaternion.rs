//! Quaternion math functions (norm, normalize, product, conjugate, inverse, rotation matrix conversion).

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::types::*;

/// Compute quaternion norm: `sqrt(w^2 + x^2 + y^2 + z^2)`.
pub fn quaternion_norm_f32(q: &[f32; 4]) -> f32 {
    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
}

/// Normalize quaternion in-place.
pub fn quaternion_normalize_f32(q: &mut [f32; 4]) -> Status {
    let norm = quaternion_norm_f32(q);
    if norm < 1e-12 {
        return Status::ArgumentError;
    }
    let inv = 1.0 / norm;
    q[0] *= inv;
    q[1] *= inv;
    q[2] *= inv;
    q[3] *= inv;
    Status::Success
}

/// Compute quaternion product: `out = q1 * q2`.
pub fn quaternion_product_f32(q1: &[f32; 4], q2: &[f32; 4], out: &mut [f32; 4]) {
    let (w1, x1, y1, z1) = (q1[0], q1[1], q1[2], q1[3]);
    let (w2, x2, y2, z2) = (q2[0], q2[1], q2[2], q2[3]);

    out[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    out[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    out[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    out[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
}

/// Compute quaternion conjugate: `[w, -x, -y, -z]`.
pub fn quaternion_conjugate_f32(q: &[f32; 4], out: &mut [f32; 4]) {
    out[0] = q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = -q[3];
}

/// Compute quaternion inverse: `q* / ||q||^2`.
pub fn quaternion_inverse_f32(q: &[f32; 4], out: &mut [f32; 4]) -> Status {
    let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if norm_sq < 1e-12 {
        return Status::ArgumentError;
    }
    let inv_sq = 1.0 / norm_sq;
    out[0] = q[0] * inv_sq;
    out[1] = -q[1] * inv_sq;
    out[2] = -q[2] * inv_sq;
    out[3] = -q[3] * inv_sq;
    Status::Success
}

/// Convert normalized quaternion `[w, x, y, z]` to a 3x3 rotation matrix (stored row-major as 9 floats).
pub fn quaternion_to_rotmat_f32(q: &[f32; 4], rot_mat: &mut [f32; 9]) {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);

    rot_mat[0] = 1.0 - 2.0 * (y * y + z * z);
    rot_mat[1] = 2.0 * (x * y - z * w);
    rot_mat[2] = 2.0 * (x * z + y * w);

    rot_mat[3] = 2.0 * (x * y + z * w);
    rot_mat[4] = 1.0 - 2.0 * (x * x + z * z);
    rot_mat[5] = 2.0 * (y * z - x * w);

    rot_mat[6] = 2.0 * (x * z - y * w);
    rot_mat[7] = 2.0 * (y * z + x * w);
    rot_mat[8] = 1.0 - 2.0 * (x * x + y * y);
}

// ─────────────────────────────────────────────────────────────────────────────
// Attitude Estimation Filters: Mahony & Madgwick
// ─────────────────────────────────────────────────────────────────────────────

/// Mahony complementary attitude estimation filter for 6-DoF IMU and 9-DoF MARG sensor fusion.
///
/// Uses proportional and integral feedback on angular velocity error to estimate attitude quaternion.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct MahonyFilter {
    /// Attitude quaternion `[w, x, y, z]`.
    pub q: [f32; 4],
    /// Integral error accumulator `[ex, ey, ez]`.
    pub integral_err: [f32; 3],
    /// Proportional gain.
    pub kp: f32,
    /// Integral gain.
    pub ki: f32,
}

impl Default for MahonyFilter {
    fn default() -> Self {
        Self::new(2.5, 0.0)
    }
}

impl MahonyFilter {
    /// Creates a new Mahony filter with given `kp` and `ki` gains.
    pub fn new(kp: f32, ki: f32) -> Self {
        Self {
            q: [1.0, 0.0, 0.0, 0.0],
            integral_err: [0.0; 3],
            kp,
            ki,
        }
    }

    /// Resets the attitude to identity `[1, 0, 0, 0]` and clears the integrator.
    pub fn reset(&mut self) {
        self.q = [1.0, 0.0, 0.0, 0.0];
        self.integral_err = [0.0; 3];
    }

    /// 6-DoF update using 3-axis gyroscope (rad/s) and 3-axis accelerometer.
    pub fn update_6dof(&mut self, mut gyro: [f32; 3], accel: [f32; 3], dt: f32) -> [f32; 4] {
        let (mut q0, mut q1, mut q2, mut q3) = (self.q[0], self.q[1], self.q[2], self.q[3]);
        let a_norm = (accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]).sqrt();

        if a_norm > 1e-6 {
            let inv_a = 1.0 / a_norm;
            let (ax, ay, az) = (accel[0] * inv_a, accel[1] * inv_a, accel[2] * inv_a);

            // Estimated gravity direction in body frame: v = R^T * [0, 0, 1]^T
            let vx = 2.0 * (q1 * q3 - q0 * q2);
            let vy = 2.0 * (q0 * q1 + q2 * q3);
            let vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

            // Error is cross product of measured gravity and estimated gravity
            let ex = ay * vz - az * vy;
            let ey = az * vx - ax * vz;
            let ez = ax * vy - ay * vx;

            if self.ki > 0.0 {
                self.integral_err[0] += ex * dt;
                self.integral_err[1] += ey * dt;
                self.integral_err[2] += ez * dt;
            }

            // Apply PI feedback
            gyro[0] += self.kp * ex + self.ki * self.integral_err[0];
            gyro[1] += self.kp * ey + self.ki * self.integral_err[1];
            gyro[2] += self.kp * ez + self.ki * self.integral_err[2];
        }

        // Integrate quaternion rate: q_dot = 0.5 * q * [0, gx, gy, gz]
        let (gx, gy, gz) = (gyro[0] * 0.5 * dt, gyro[1] * 0.5 * dt, gyro[2] * 0.5 * dt);
        q0 += -q1 * gx - q2 * gy - q3 * gz;
        q1 += q0 * gx + q2 * gz - q3 * gy;
        q2 += q0 * gy - q1 * gz + q3 * gx;
        q3 += q0 * gz + q1 * gy - q2 * gx;

        let mut q_out = [q0, q1, q2, q3];
        let _ = quaternion_normalize_f32(&mut q_out);
        self.q = q_out;
        self.q
    }

    /// 9-DoF update using 3-axis gyroscope (rad/s), 3-axis accelerometer, and 3-axis magnetometer.
    pub fn update_9dof(&mut self, mut gyro: [f32; 3], accel: [f32; 3], mag: [f32; 3], dt: f32) -> [f32; 4] {
        let (mut q0, mut q1, mut q2, mut q3) = (self.q[0], self.q[1], self.q[2], self.q[3]);
        let a_norm = (accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]).sqrt();
        let m_norm = (mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2]).sqrt();

        if a_norm > 1e-6 && m_norm > 1e-6 {
            let inv_a = 1.0 / a_norm;
            let (ax, ay, az) = (accel[0] * inv_a, accel[1] * inv_a, accel[2] * inv_a);

            let inv_m = 1.0 / m_norm;
            let (mx, my, mz) = (mag[0] * inv_m, mag[1] * inv_m, mag[2] * inv_m);

            // Rotate mag measurement to Earth reference frame
            let hx = 2.0 * (mx * (0.5 - q2 * q2 - q3 * q3) + my * (q1 * q2 - q0 * q3) + mz * (q1 * q3 + q0 * q2));
            let hy = 2.0 * (mx * (q1 * q2 + q0 * q3) + my * (0.5 - q1 * q1 - q3 * q3) + mz * (q2 * q3 - q0 * q1));
            let bx = (hx * hx + hy * hy).sqrt();
            let bz = 2.0 * (mx * (q1 * q3 - q0 * q2) + my * (q2 * q3 + q0 * q1) + mz * (0.5 - q1 * q1 - q2 * q2));

            // Estimated gravity & magnetic direction in body frame
            let vx = 2.0 * (q1 * q3 - q0 * q2);
            let vy = 2.0 * (q0 * q1 + q2 * q3);
            let vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

            let wx = 2.0 * bx * (0.5 - q2 * q2 - q3 * q3) + 2.0 * bz * (q1 * q3 - q0 * q2);
            let wy = 2.0 * bx * (q1 * q2 - q0 * q3) + 2.0 * bz * (q0 * q1 + q2 * q3);
            let wz = 2.0 * bx * (q0 * q2 + q1 * q3) + 2.0 * bz * (0.5 - q1 * q1 - q2 * q2);

            // Error is sum of cross products
            let ex = (ay * vz - az * vy) + (my * wz - mz * wy);
            let ey = (az * vx - ax * vz) + (mz * wx - mx * wz);
            let ez = (ax * vy - ay * vx) + (mx * wy - my * wx);

            if self.ki > 0.0 {
                self.integral_err[0] += ex * dt;
                self.integral_err[1] += ey * dt;
                self.integral_err[2] += ez * dt;
            }

            gyro[0] += self.kp * ex + self.ki * self.integral_err[0];
            gyro[1] += self.kp * ey + self.ki * self.integral_err[1];
            gyro[2] += self.kp * ez + self.ki * self.integral_err[2];
        } else if a_norm > 1e-6 {
            return self.update_6dof(gyro, accel, dt);
        }

        let (gx, gy, gz) = (gyro[0] * 0.5 * dt, gyro[1] * 0.5 * dt, gyro[2] * 0.5 * dt);
        q0 += -q1 * gx - q2 * gy - q3 * gz;
        q1 += q0 * gx + q2 * gz - q3 * gy;
        q2 += q0 * gy - q1 * gz + q3 * gx;
        q3 += q0 * gz + q1 * gy - q2 * gx;

        let mut q_out = [q0, q1, q2, q3];
        let _ = quaternion_normalize_f32(&mut q_out);
        self.q = q_out;
        self.q
    }
}

/// Madgwick gradient-descent attitude estimation filter.
///
/// Fast, orientation tracking filter with low computational overhead.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct MadgwickFilter {
    /// Attitude quaternion `[w, x, y, z]`.
    pub q: [f32; 4],
    /// Filter gain representing gyro measurement error divergence rate.
    pub beta: f32,
}

impl Default for MadgwickFilter {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl MadgwickFilter {
    /// Creates a new Madgwick filter with divergence rate gain `beta`.
    pub fn new(beta: f32) -> Self {
        Self {
            q: [1.0, 0.0, 0.0, 0.0],
            beta,
        }
    }

    /// Resets the attitude quaternion to identity `[1, 0, 0, 0]`.
    pub fn reset(&mut self) {
        self.q = [1.0, 0.0, 0.0, 0.0];
    }

    /// 6-DoF update using 3-axis gyroscope (rad/s) and 3-axis accelerometer.
    pub fn update_6dof(&mut self, gyro: [f32; 3], accel: [f32; 3], dt: f32) -> [f32; 4] {
        let (q0, q1, q2, q3) = (self.q[0], self.q[1], self.q[2], self.q[3]);
        let (gx, gy, gz) = (gyro[0], gyro[1], gyro[2]);

        // Rate of change of quaternion from gyroscope
        let mut q_dot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
        let mut q_dot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
        let mut q_dot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
        let mut q_dot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

        let a_norm = (accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]).sqrt();
        if a_norm > 1e-6 {
            let inv_a = 1.0 / a_norm;
            let (ax, ay, az) = (accel[0] * inv_a, accel[1] * inv_a, accel[2] * inv_a);

            // Auxiliary variables to avoid repeated arithmetic
            let _2q0 = 2.0 * q0;
            let _2q1 = 2.0 * q1;
            let _2q2 = 2.0 * q2;
            let _2q3 = 2.0 * q3;
            let _4q1 = 4.0 * q1;
            let _4q2 = 4.0 * q2;

            // Gradient descent algorithm corrective step
            let mut s0 = -_2q2 * (2.0 * (q1 * q3 - q0 * q2) - ax) + _2q1 * (2.0 * (q0 * q1 + q2 * q3) - ay);
            let mut s1 = _2q3 * (2.0 * (q1 * q3 - q0 * q2) - ax) + _2q0 * (2.0 * (q0 * q1 + q2 * q3) - ay) - _4q1 * (1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2 - az);
            let mut s2 = -_2q0 * (2.0 * (q1 * q3 - q0 * q2) - ax) + _2q3 * (2.0 * (q0 * q1 + q2 * q3) - ay) - _4q2 * (1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2 - az);
            let mut s3 = _2q1 * (2.0 * (q1 * q3 - q0 * q2) - ax) + _2q2 * (2.0 * (q0 * q1 + q2 * q3) - ay);

            let s_norm = (s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3).sqrt();
            if s_norm > 1e-6 {
                let inv_s = 1.0 / s_norm;
                s0 *= inv_s;
                s1 *= inv_s;
                s2 *= inv_s;
                s3 *= inv_s;

                // Apply feedback step
                q_dot0 -= self.beta * s0;
                q_dot1 -= self.beta * s1;
                q_dot2 -= self.beta * s2;
                q_dot3 -= self.beta * s3;
            }
        }

        // Integrate rate of change of quaternion
        let mut q_new = [
            q0 + q_dot0 * dt,
            q1 + q_dot1 * dt,
            q2 + q_dot2 * dt,
            q3 + q_dot3 * dt,
        ];
        let _ = quaternion_normalize_f32(&mut q_new);
        self.q = q_new;
        self.q
    }

    /// 9-DoF update using 3-axis gyroscope (rad/s), 3-axis accelerometer, and 3-axis magnetometer.
    pub fn update_9dof(&mut self, gyro: [f32; 3], accel: [f32; 3], mag: [f32; 3], dt: f32) -> [f32; 4] {
        let (q0, q1, q2, q3) = (self.q[0], self.q[1], self.q[2], self.q[3]);
        let (gx, gy, gz) = (gyro[0], gyro[1], gyro[2]);

        let a_norm = (accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]).sqrt();
        let m_norm = (mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2]).sqrt();

        if a_norm < 1e-6 || m_norm < 1e-6 {
            return self.update_6dof(gyro, accel, dt);
        }

        let inv_a = 1.0 / a_norm;
        let (ax, ay, az) = (accel[0] * inv_a, accel[1] * inv_a, accel[2] * inv_a);

        let inv_m = 1.0 / m_norm;
        let (mx, my, mz) = (mag[0] * inv_m, mag[1] * inv_m, mag[2] * inv_m);

        // Reference direction of Earth's magnetic field
        let _2q0mx = 2.0 * q0 * mx;
        let _2q0my = 2.0 * q0 * my;
        let _2q0mz = 2.0 * q0 * mz;
        let _2q1mx = 2.0 * q1 * mx;

        let hx = mx * (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) + 2.0 * my * (q1 * q2 - q0 * q3) + 2.0 * mz * (q1 * q3 + q0 * q2);
        let hy = 2.0 * mx * (q1 * q2 + q0 * q3) + my * (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) + 2.0 * mz * (q2 * q3 - q0 * q1);
        let _2bx = (hx * hx + hy * hy).sqrt();
        let _2bz = 2.0 * (mx * (q1 * q3 - q0 * q2) + my * (q2 * q3 + q0 * q1) + mz * (0.5 - q1 * q1 - q2 * q2));
        let _4bx = 2.0 * _2bx;
        let _4bz = 2.0 * _2bz;

        let _2q0 = 2.0 * q0;
        let _2q1 = 2.0 * q1;
        let _2q2 = 2.0 * q2;
        let _2q3 = 2.0 * q3;
        let _4q1 = 4.0 * q1;
        let _4q2 = 4.0 * q2;
        let q1q1 = q1 * q1;
        let q2q2 = q2 * q2;
        let q3q3 = q3 * q3;

        // Gradient descent algorithm corrective step
        let mut s0 = -_2q2 * (2.0 * (q1 * q3 - q0 * q2) - ax) + _2q1 * (2.0 * (q0 * q1 + q2 * q3) - ay)
            - _2bz * q2 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1 * q3 - q0 * q2) - mx)
            + (-_2bx * q3 + _2bz * q1) * (_2bx * (q1 * q2 - q0 * q3) + _2bz * (q0 * q1 + q2 * q3) - my)
            + _2bx * q2 * (_2bx * (q0 * q2 + q1 * q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);

        let mut s1 = _2q3 * (2.0 * (q1 * q3 - q0 * q2) - ax) + _2q0 * (2.0 * (q0 * q1 + q2 * q3) - ay)
            - _4q1 * (1.0 - 2.0 * q1q1 - 2.0 * q2q2 - az)
            + _2bz * q3 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1 * q3 - q0 * q2) - mx)
            + (_2bx * q2 + _2bz * q0) * (_2bx * (q1 * q2 - q0 * q3) + _2bz * (q0 * q1 + q2 * q3) - my)
            + (_2bx * q3 - _4bz * q1) * (_2bx * (q0 * q2 + q1 * q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);

        let mut s2 = -_2q0 * (2.0 * (q1 * q3 - q0 * q2) - ax) + _2q3 * (2.0 * (q0 * q1 + q2 * q3) - ay)
            - _4q2 * (1.0 - 2.0 * q1q1 - 2.0 * q2q2 - az)
            + (-_4bx * q2 - _2bz * q0) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1 * q3 - q0 * q2) - mx)
            + (_2bx * q1 + _2bz * q3) * (_2bx * (q1 * q2 - q0 * q3) + _2bz * (q0 * q1 + q2 * q3) - my)
            + (_2bx * q0 - _4bz * q2) * (_2bx * (q0 * q2 + q1 * q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);

        let mut s3 = _2q1 * (2.0 * (q1 * q3 - q0 * q2) - ax) + _2q2 * (2.0 * (q0 * q1 + q2 * q3) - ay)
            + (-_4bx * q3 + _2bz * q1) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1 * q3 - q0 * q2) - mx)
            + (-_2bx * q0 + _2bz * q2) * (_2bx * (q1 * q2 - q0 * q3) + _2bz * (q0 * q1 + q2 * q3) - my)
            + _2bx * q1 * (_2bx * (q0 * q2 + q1 * q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);

        let s_norm = (s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3).sqrt();
        if s_norm > 1e-6 {
            let inv_s = 1.0 / s_norm;
            s0 *= inv_s;
            s1 *= inv_s;
            s2 *= inv_s;
            s3 *= inv_s;
        }

        // Rate of change from gyro
        let q_dot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz) - self.beta * s0;
        let q_dot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy) - self.beta * s1;
        let q_dot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx) - self.beta * s2;
        let q_dot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx) - self.beta * s3;

        let mut q_new = [
            q0 + q_dot0 * dt,
            q1 + q_dot1 * dt,
            q2 + q_dot2 * dt,
            q3 + q_dot3 * dt,
        ];
        let _ = quaternion_normalize_f32(&mut q_new);
        self.q = q_new;
        self.q
    }
}
