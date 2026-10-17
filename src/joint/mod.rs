//! Joints (constraints) between bodies.
//!
//! Create typed joints with the `create_*_joint` helpers (distance, motor,
//! parallel, prismatic, revolute, spherical, weld, wheel, filter), each taking
//! the matching def from [`crate::types`] (`default_revolute_joint_def`, etc.).
//!
//! Port of `joint.h` / `joint.c` (lifecycle, plumbing, and per-type solve).
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

mod api;
mod distance;
mod draw;
mod lifecycle;
mod motor;
mod parallel;
mod plumbing;
mod prismatic;
mod revolute;
mod solve;
mod spherical;
mod types;
mod weld;
mod wheel;
mod wheel_api;

pub use api::*;
pub use distance::*;
pub use draw::*;
pub use lifecycle::*;
pub use motor::*;
pub use parallel::*;
pub use plumbing::*;
pub use prismatic::*;
pub use revolute::*;
pub use solve::*;
pub use spherical::*;
pub use types::*;
pub use weld::*;
pub use wheel::*;
pub use wheel_api::*;

use crate::math_functions::{abs_float, mul_sv, sub, Matrix3, Vec3, VEC3_ZERO};

/// Solve the 3x3 point constraint `K * b = rhs` without forming `K` through matrix products.
/// (joint.h: b3SolvePointConstraint)
///
/// K = [(1/m1 + 1/m2) * eye(2) - skew(r1) * invI1 * skew(r1) - skew(r2) * invI2 * skew(r2)]
#[inline(always)]
pub(crate) fn solve_point_constraint(
    m_a: f32,
    m_b: f32,
    i_a: Matrix3,
    i_b: Matrix3,
    r_a: Vec3,
    r_b: Vec3,
    rhs: Vec3,
) -> Vec3 {
    let axy = r_a.z * i_a.cy.y - r_a.y * i_a.cz.y;
    let axz = r_a.z * i_a.cy.z - r_a.y * i_a.cz.z;
    let ay = sub(mul_sv(r_a.x, i_a.cz), mul_sv(r_a.z, i_a.cx));
    let az = sub(mul_sv(r_a.y, i_a.cx), mul_sv(r_a.x, i_a.cy));

    let bxy = r_b.z * i_b.cy.y - r_b.y * i_b.cz.y;
    let bxz = r_b.z * i_b.cy.z - r_b.y * i_b.cz.z;
    let by = sub(mul_sv(r_b.x, i_b.cz), mul_sv(r_b.z, i_b.cx));
    let bz = sub(mul_sv(r_b.y, i_b.cx), mul_sv(r_b.x, i_b.cy));

    let m = m_a + m_b;
    let kxx = (axy * r_a.z - axz * r_a.y) + (bxy * r_b.z - bxz * r_b.y) + m;
    let kyy = (ay.z * r_a.x - ay.x * r_a.z) + (by.z * r_b.x - by.x * r_b.z) + m;
    let kzz = (az.x * r_a.y - az.y * r_a.x) + (bz.x * r_b.y - bz.y * r_b.x) + m;
    let kxy = (ay.y * r_a.z - ay.z * r_a.y) + (by.y * r_b.z - by.z * r_b.y);
    let kxz = (az.y * r_a.z - az.z * r_a.y) + (bz.y * r_b.z - bz.z * r_b.y);
    let kyz = (az.z * r_a.x - az.x * r_a.z) + (bz.z * r_b.x - bz.x * r_b.z);

    let cxx = kyy * kzz - kyz * kyz;
    let cyy = kxx * kzz - kxz * kxz;
    let czz = kxx * kyy - kxy * kxy;
    let cxy = kxz * kyz - kxy * kzz;
    let cxz = kxy * kyz - kyy * kxz;
    let cyz = kxy * kxz - kxx * kyz;

    let det = kxx * cxx + kxy * cxy + kxz * cxz;
    if abs_float(det) > 1000.0 * f32::MIN_POSITIVE {
        let inv_det = 1.0 / det;
        return Vec3 {
            x: inv_det * (cxx * rhs.x + cxy * rhs.y + cxz * rhs.z),
            y: inv_det * (cxy * rhs.x + cyy * rhs.y + cyz * rhs.z),
            z: inv_det * (cxz * rhs.x + cyz * rhs.y + czz * rhs.z),
        };
    }

    VEC3_ZERO
}
