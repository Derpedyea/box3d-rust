//! Wide (four lane) convex contact constraints from contact_solver.c, scalar `b3FloatW` path
//! (B3_SIMD_NONE).
//!
//! Graph-color convex contacts are packed four at a time into a [`ContactConstraintWide`] and
//! solved with the `b3FloatW` emulation from [`crate::simd`], so every lane runs exactly
//! the arithmetic of C's scalar wide path, including the decisions C makes for the whole wide
//! constraint (the wide point count, rolling resistance, restitution).
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

mod prepare;
mod restitution;
mod solve;
mod store;

pub use prepare::prepare_contacts_convex;
pub use restitution::apply_restitution_convex;
pub use solve::{push_contacts_convex, solve_contacts_convex, warm_start_contacts_convex};
pub use store::store_impulses_convex;

use crate::body::{body_flags, BodyState, IDENTITY_BODY_STATE};
use crate::constants::MAX_MANIFOLD_POINTS;
use crate::core::{NULL_INDEX, SIMD_WIDTH};
use crate::solver::states::SolverStates;
// The wide vec3 type and its helpers live in `simd.h` upstream (moved there from
// contact_solver.c in 9f998c8), so they are imported from `crate::simd`.
use crate::simd::{
    abs_w, add_w, blend_w, cross_w, div_w, dot_w, greater_than_w, lane, less_than_w, mul_svw,
    mul_w, neg_w, or_w, splat_w, sqrt_w, sub_w, zero_w, FloatW, Vec3W,
};

/// Wide vec2. (b3Vec2W)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2W {
    pub x: FloatW,
    pub y: FloatW,
}

/// Wide quaternion. (b3QuatW)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuatW {
    pub v: Vec3W,
    pub s: FloatW,
}

/// Wide symmetric matrix2. (b3SymMatrix2W)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymMatrix2W {
    pub cxx: FloatW,
    pub cxy: FloatW,
    pub cyy: FloatW,
}

/// Wide symmetric matrix3. (b3SymMatrix3W)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymMatrix3W {
    pub cxx: FloatW,
    pub cxy: FloatW,
    pub cxz: FloatW,
    pub cyy: FloatW,
    pub cyz: FloatW,
    pub czz: FloatW,
}

/// Wide matrix3. (b3Matrix3W)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3W {
    pub cx: Vec3W,
    pub cy: Vec3W,
    pub cz: Vec3W,
}

pub const VEC3W_ZERO: Vec3W = Vec3W {
    x: FloatW::ZERO,
    y: FloatW::ZERO,
    z: FloatW::ZERO,
};

pub const SYM3W_ZERO: SymMatrix3W = SymMatrix3W {
    cxx: VEC3W_ZERO.x,
    cxy: VEC3W_ZERO.x,
    cxz: VEC3W_ZERO.x,
    cyy: VEC3W_ZERO.x,
    cyz: VEC3W_ZERO.x,
    czz: VEC3W_ZERO.x,
};

/// (b3ContactConstraintPointWide)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactConstraintPointWide {
    pub anchor_as: Vec3W,
    pub anchor_bs: Vec3W,
    pub base_separations: FloatW,
    pub normal_impulses: FloatW,
    pub total_normal_impulses: FloatW,
    pub normal_masses: FloatW,
    pub lever_arms: FloatW,
    pub relative_velocities: FloatW,
    pub restitution_impulses: FloatW,
}

/// Solves four contacts. (b3ContactConstraintWide)
///
/// C keeps `b3Manifold* manifolds[B3_SIMD_WIDTH]`; the port keeps the contact id of each lane
/// (`NULL_INDEX` for an empty lane) and reaches the single manifold through the contact.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactConstraintWide {
    /// These are base 1, 0 for null
    pub index_a: [i32; SIMD_WIDTH],
    pub index_b: [i32; SIMD_WIDTH],

    pub point_counts: [i32; SIMD_WIDTH],

    pub inv_mass_a: FloatW,
    pub inv_mass_b: FloatW,
    pub inv_i_a: SymMatrix3W,
    pub inv_i_b: SymMatrix3W,
    pub normal: Vec3W,

    pub tangent1: Vec3W,
    pub tangent2: Vec3W,

    // Friction centers
    pub center_a: Vec3W,
    pub center_b: Vec3W,
    pub twist_mass: FloatW,
    pub twist_impulse: FloatW,
    pub tangent_mass: SymMatrix2W,
    pub friction_impulse: Vec2W,
    pub rolling_mass: SymMatrix3W,
    pub rolling_impulse: Vec3W,
    pub friction: FloatW,
    pub rolling_resistance: FloatW,
    pub tangent_velocity1: FloatW,
    pub tangent_velocity2: FloatW,
    pub restitution: FloatW,

    pub contact_ids: [i32; SIMD_WIDTH],

    pub points: [ContactConstraintPointWide; MAX_MANIFOLD_POINTS],
}

impl ContactConstraintWide {
    /// Wide point count: the maximum over the four lanes.
    pub fn point_count(&self) -> usize {
        let point_count1 = self.point_counts[0].max(self.point_counts[1]);
        let point_count2 = self.point_counts[2].max(self.point_counts[3]);
        let point_count = point_count1.max(point_count2);
        debug_assert!(0 < point_count && point_count as usize <= MAX_MANIFOLD_POINTS);
        point_count as usize
    }
}

/// Wide version of BodyState. (b3BodyStateW)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyStateW {
    pub v: Vec3W,
    pub w: Vec3W,
    pub dp: Vec3W,
    pub dq: QuatW,
}

// a + b
#[inline]
pub fn add_v2w(a: Vec2W, b: Vec2W) -> Vec2W {
    Vec2W {
        x: add_w(a.x, b.x),
        y: add_w(a.y, b.y),
    }
}

// m * a
#[inline]
pub fn mul_mv2w(m: SymMatrix2W, a: Vec2W) -> Vec2W {
    Vec2W {
        x: add_w(mul_w(m.cxx, a.x), mul_w(m.cxy, a.y)),
        y: add_w(mul_w(m.cxy, a.x), mul_w(m.cyy, a.y)),
    }
}

// m * a
#[inline]
pub fn mul_mvw(m: SymMatrix3W, a: Vec3W) -> Vec3W {
    Vec3W {
        x: add_w(
            mul_w(m.cxx, a.x),
            add_w(mul_w(m.cxy, a.y), mul_w(m.cxz, a.z)),
        ),
        y: add_w(
            mul_w(m.cxy, a.x),
            add_w(mul_w(m.cyy, a.y), mul_w(m.cyz, a.z)),
        ),
        z: add_w(
            mul_w(m.cxz, a.x),
            add_w(mul_w(m.cyz, a.y), mul_w(m.czz, a.z)),
        ),
    }
}

// a - m * b
#[inline]
pub fn mul_sub_mvw(a: Vec3W, m: SymMatrix3W, b: Vec3W) -> Vec3W {
    let c = mul_mvw(m, b);
    Vec3W {
        x: sub_w(a.x, c.x),
        y: sub_w(a.y, c.y),
        z: sub_w(a.z, c.z),
    }
}

// a + m * b
#[inline]
pub fn mul_add_mvw(a: Vec3W, m: SymMatrix3W, b: Vec3W) -> Vec3W {
    let c = mul_mvw(m, b);
    Vec3W {
        x: add_w(a.x, c.x),
        y: add_w(a.y, c.y),
        z: add_w(a.z, c.z),
    }
}

/// (b3MakeMatrixFromQuatW)
#[inline]
pub fn make_matrix_from_quat_w(q: QuatW) -> Matrix3W {
    let x2 = add_w(q.v.x, q.v.x);
    let y2 = add_w(q.v.y, q.v.y);
    let z2 = add_w(q.v.z, q.v.z);
    let xx2 = mul_w(q.v.x, x2);
    let yy2 = mul_w(q.v.y, y2);
    let zz2 = mul_w(q.v.z, z2);
    let xy2 = mul_w(q.v.x, y2);
    let xz2 = mul_w(q.v.x, z2);
    let yz2 = mul_w(q.v.y, z2);
    let xw2 = mul_w(q.s, x2);
    let yw2 = mul_w(q.s, y2);
    let zw2 = mul_w(q.s, z2);
    let one = splat_w(1.0);

    Matrix3W {
        cx: Vec3W {
            x: sub_w(one, add_w(yy2, zz2)),
            y: add_w(xy2, zw2),
            z: sub_w(xz2, yw2),
        },
        cy: Vec3W {
            x: sub_w(xy2, zw2),
            y: sub_w(one, add_w(xx2, zz2)),
            z: add_w(yz2, xw2),
        },
        cz: Vec3W {
            x: add_w(xz2, yw2),
            y: sub_w(yz2, xw2),
            z: sub_w(one, add_w(xx2, yy2)),
        },
    }
}

/// (b3MulM3VW)
#[inline]
pub fn mul_m3vw(m: Matrix3W, a: Vec3W) -> Vec3W {
    Vec3W {
        x: add_w(
            mul_w(m.cx.x, a.x),
            add_w(mul_w(m.cy.x, a.y), mul_w(m.cz.x, a.z)),
        ),
        y: add_w(
            mul_w(m.cx.y, a.x),
            add_w(mul_w(m.cy.y, a.y), mul_w(m.cz.y, a.z)),
        ),
        z: add_w(
            mul_w(m.cx.z, a.x),
            add_w(mul_w(m.cy.z, a.y), mul_w(m.cz.z, a.z)),
        ),
    }
}

/// (b3InvRotateVectorW)
#[inline]
pub fn inv_rotate_vector_w(q: QuatW, a: Vec3W) -> Vec3W {
    let t = cross_w(q.v, a);
    let t = Vec3W {
        x: add_w(t.x, t.x),
        y: add_w(t.y, t.y),
        z: add_w(t.z, t.z),
    };
    let u = cross_w(q.v, t);
    Vec3W {
        x: add_w(sub_w(a.x, mul_w(q.s, t.x)), u.x),
        y: add_w(sub_w(a.y, mul_w(q.s, t.y)), u.y),
        z: add_w(sub_w(a.z, mul_w(q.s, t.z)), u.z),
    }
}

/// (b3AddSymW)
#[inline]
pub fn add_sym_w(a: SymMatrix3W, b: SymMatrix3W) -> SymMatrix3W {
    SymMatrix3W {
        cxx: add_w(a.cxx, b.cxx),
        cxy: add_w(a.cxy, b.cxy),
        cxz: add_w(a.cxz, b.cxz),
        cyy: add_w(a.cyy, b.cyy),
        cyz: add_w(a.cyz, b.cyz),
        czz: add_w(a.czz, b.czz),
    }
}

/// (b3InvertSymW)
#[inline]
pub fn invert_sym_w(m: SymMatrix3W) -> SymMatrix3W {
    let cxx = sub_w(mul_w(m.cyy, m.czz), mul_w(m.cyz, m.cyz));
    let cxy = sub_w(mul_w(m.cxz, m.cyz), mul_w(m.cxy, m.czz));
    let cxz = sub_w(mul_w(m.cxy, m.cyz), mul_w(m.cxz, m.cyy));
    let cyy = sub_w(mul_w(m.cxx, m.czz), mul_w(m.cxz, m.cxz));
    let cyz = sub_w(mul_w(m.cxy, m.cxz), mul_w(m.cxx, m.cyz));
    let czz = sub_w(mul_w(m.cxx, m.cyy), mul_w(m.cxy, m.cxy));

    let det = add_w(
        mul_w(m.cxx, cxx),
        add_w(mul_w(m.cxy, cxy), mul_w(m.cxz, cxz)),
    );
    let valid = greater_than_w(abs_w(det), splat_w(1000.0 * f32::MIN_POSITIVE));
    let inv_det = blend_w(zero_w(), div_w(splat_w(1.0), det), valid);

    SymMatrix3W {
        cxx: mul_w(inv_det, cxx),
        cxy: mul_w(inv_det, cxy),
        cxz: mul_w(inv_det, cxz),
        cyy: mul_w(inv_det, cyy),
        cyz: mul_w(inv_det, cyz),
        czz: mul_w(inv_det, czz),
    }
}

/// (b3PerpW)
#[inline]
pub fn perp_w(a: Vec3W) -> Vec3W {
    let zero = zero_w();
    let half = splat_w(0.5);
    let mask = or_w(less_than_w(a.x, neg_w(half)), greater_than_w(a.x, half));

    let p = Vec3W {
        x: blend_w(zero, a.y, mask),
        y: blend_w(a.z, neg_w(a.x), mask),
        z: blend_w(neg_w(a.y), zero, mask),
    };

    let length_squared = dot_w(p, p);
    let valid = greater_than_w(length_squared, splat_w(1000.0 * f32::MIN_POSITIVE));
    let s = blend_w(zero, div_w(splat_w(1.0), sqrt_w(length_squared)), valid);
    mul_svw(s, p)
}

/// The four 16-byte rows of a body state.
#[inline]
fn state_rows(s: &BodyState) -> [FloatW; 4] {
    [
        FloatW::new([
            s.linear_velocity.x,
            s.linear_velocity.y,
            s.linear_velocity.z,
            s.padding1,
        ]),
        FloatW::new([
            s.angular_velocity.x,
            s.angular_velocity.y,
            s.angular_velocity.z,
            s.padding2,
        ]),
        FloatW::new([
            s.delta_position.x,
            s.delta_position.y,
            s.delta_position.z,
            f32::from_bits(s.flags),
        ]),
        FloatW::new([
            s.delta_rotation.v.x,
            s.delta_rotation.v.y,
            s.delta_rotation.v.z,
            s.delta_rotation.s,
        ]),
    ]
}

/// Indices are base 1 with 0 for null. (b3GatherBodies)
#[inline]
pub fn gather_bodies(states: &SolverStates, indices: &[i32; SIMD_WIDTH]) -> BodyStateW {
    let rows: [[FloatW; 4]; SIMD_WIDTH] = core::array::from_fn(|i| {
        if indices[i] == 0 {
            state_rows(&IDENTITY_BODY_STATE)
        } else {
            states.rows((indices[i] - 1) as usize)
        }
    });

    let [vx, vy, vz, _] = FloatW::transpose([rows[0][0], rows[1][0], rows[2][0], rows[3][0]]);
    let [wx, wy, wz, _] = FloatW::transpose([rows[0][1], rows[1][1], rows[2][1], rows[3][1]]);
    let [px, py, pz, _] = FloatW::transpose([rows[0][2], rows[1][2], rows[2][2], rows[3][2]]);
    let [qx, qy, qz, qs] = FloatW::transpose([rows[0][3], rows[1][3], rows[2][3], rows[3][3]]);

    BodyStateW {
        v: Vec3W {
            x: vx,
            y: vy,
            z: vz,
        },
        w: Vec3W {
            x: wx,
            y: wy,
            z: wz,
        },
        dp: Vec3W {
            x: px,
            y: py,
            z: pz,
        },
        dq: QuatW {
            v: Vec3W {
                x: qx,
                y: qy,
                z: qz,
            },
            s: qs,
        },
    }
}

/// This writes only the velocities back to the solver bodies. (b3ScatterBodies)
#[inline]
pub fn scatter_bodies(states: &SolverStates, indices: &[i32; SIMD_WIDTH], body: &BodyStateW) {
    let zero = FloatW::ZERO;
    let v = FloatW::transpose([body.v.x, body.v.y, body.v.z, zero]);
    let w = FloatW::transpose([body.w.x, body.w.y, body.w.z, zero]);
    for (i, &index) in indices.iter().enumerate() {
        // Indices are 0 for null
        if index == 0 {
            continue;
        }
        let index = (index - 1) as usize;
        if (states.flags(index) & body_flags::DYNAMIC_FLAG) == 0 {
            continue;
        }
        states.set_velocity_rows(index, v[i], w[i]);
    }
}

/// Empty lane marker for [`ContactConstraintWide::contact_ids`].
pub const EMPTY_LANE: i32 = NULL_INDEX;
