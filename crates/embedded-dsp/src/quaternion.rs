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
