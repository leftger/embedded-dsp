//! Conversions between the `[w, x, y, z]` quaternion representation used by
//! [`crate::quaternion`] and `nalgebra`'s [`Quaternion`]/[`UnitQuaternion`].
//!
//! `embedded-dsp`'s quaternion functions operate on plain `[f32; 4]` arrays
//! rather than a `nalgebra` type, so there is no free `From`/`Into`
//! conversion between the two (both types are foreign to this crate, so the
//! orphan rule blocks a blanket trait impl here). These functions bridge the
//! two representations explicitly for pipelines that use `embedded-dsp` for
//! signal conditioning and a `nalgebra`-based crate (such as an AHRS filter)
//! downstream.

use nalgebra::{Quaternion, UnitQuaternion};

/// Converts embedded-dsp's `[w, x, y, z]` quaternion into a `nalgebra`
/// [`Quaternion<f32>`], without normalizing.
#[inline(always)]
pub fn quaternion_to_nalgebra(q: &[f32; 4]) -> Quaternion<f32> {
    Quaternion::new(q[0], q[1], q[2], q[3])
}

/// Converts a `nalgebra` [`Quaternion<f32>`] into embedded-dsp's
/// `[w, x, y, z]` representation.
#[inline(always)]
pub fn quaternion_from_nalgebra(q: &Quaternion<f32>) -> [f32; 4] {
    [q.w, q.i, q.j, q.k]
}

/// Converts embedded-dsp's `[w, x, y, z]` quaternion into a `nalgebra`
/// [`UnitQuaternion<f32>`], normalizing it in the process.
#[inline(always)]
pub fn quaternion_to_unit_nalgebra(q: &[f32; 4]) -> UnitQuaternion<f32> {
    UnitQuaternion::from_quaternion(quaternion_to_nalgebra(q))
}

/// Converts a `nalgebra` [`UnitQuaternion<f32>`] into embedded-dsp's
/// `[w, x, y, z]` representation.
#[inline(always)]
pub fn quaternion_from_unit_nalgebra(q: &UnitQuaternion<f32>) -> [f32; 4] {
    quaternion_from_nalgebra(q.quaternion())
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn quaternion_round_trips_through_nalgebra() {
        let q = [0.5f32, 0.5, 0.5, 0.5];
        let back = quaternion_from_nalgebra(&quaternion_to_nalgebra(&q));
        assert_eq!(back, q);
    }

    #[test]
    fn unit_quaternion_round_trips_for_already_normalized_input() {
        let q = [0.5f32, 0.5, 0.5, 0.5]; // norm == 1.0
        let back = quaternion_from_unit_nalgebra(&quaternion_to_unit_nalgebra(&q));
        for i in 0..4 {
            assert!((back[i] - q[i]).abs() < 1e-6, "component {i}: {back:?} vs {q:?}");
        }
    }

    #[test]
    fn unit_quaternion_conversion_normalizes() {
        let q = [2.0f32, 0.0, 0.0, 0.0]; // norm == 2.0, not a unit quaternion
        let uq = quaternion_to_unit_nalgebra(&q);
        assert!((uq.quaternion().norm() - 1.0).abs() < 1e-6);
        let back = quaternion_from_unit_nalgebra(&uq);
        assert!((back[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn matches_quaternion_module_normalize() {
        let mut q = [3.0f32, 0.0, 4.0, 0.0]; // norm == 5.0
        crate::quaternion::quaternion_normalize_f32(&mut q);

        let via_nalgebra = quaternion_from_unit_nalgebra(&quaternion_to_unit_nalgebra(&[
            3.0, 0.0, 4.0, 0.0,
        ]));

        for i in 0..4 {
            assert!(
                (via_nalgebra[i] - q[i]).abs() < 1e-6,
                "component {i}: {via_nalgebra:?} vs {q:?}"
            );
        }
    }
}
