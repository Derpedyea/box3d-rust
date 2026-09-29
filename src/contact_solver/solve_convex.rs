//! Push and solve Convex contact constraints. Scalar form of `b3PushContacts_Convex` and
//! `b3SolveContacts_Convex` (B3_SIMD_NONE): same friction order, clamp formulas, and
//! SymMatrix muls as the wide path.
//!
//! The wide path packs four consecutive convex contacts of a color into one
//! `b3ContactConstraintWide`. Lanes never share a dynamic body, so the port solves each lane
//! as its own constraint. The exceptions are the decisions C makes for the whole wide
//! constraint: the point count every lane loops over (the maximum across the lanes, with the
//! shorter lanes running their zero-padded points), the deferred restitution iteration count,
//! and whether the rolling resistance block runs. The Convex kernels (and the Convex warm start)
//! therefore walk the convex constraints of a color in the same groups of four and make those
//! decisions per group.
//!
//! C stores `negRestitutionVelocities` (`restitution * vn`) for the wide path. The shared
//! scalar constraint stores the Mesh form `restitution_velocity` (`-restitution * vn`), and
//! negation is exact, so `-restitution_velocity` is bit-identical to the wide value.
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use super::ContactConstraint;
use crate::body::{body_flags, BodyState, IDENTITY_BODY_STATE};
use crate::constants::MAX_MANIFOLD_POINTS;
use crate::core::NULL_INDEX;
use crate::math_functions::{
    add, blend2, cross, dot, dot2, mul_add, mul_mv2_sym, mul_mv_sym, mul_sub, mul_sv,
    rotate_vector, sub, sub2, Vec2,
};
use crate::solver::StepContext;

/// SIMD width of the wide contact constraint. (B3_SIMD_WIDTH)
pub(super) const SIMD_WIDTH: usize = 4;

/// Wide point count of one group of four lanes: the maximum over the lanes. C loops every lane
/// over this count; the points past a lane's own count were zeroed in prepare, and running them
/// keeps the (signed zero) velocity updates bit-identical to the wide kernels.
pub(super) fn wide_point_count(group: &[ContactConstraint]) -> usize {
    let mut point_count = 0;
    for c in group.iter() {
        point_count = point_count.max(c.constraints[0].point_count as usize);
    }
    debug_assert!(0 < point_count && point_count <= MAX_MANIFOLD_POINTS);
    point_count
}

// Scalar lanes of the `b3FloatW` operations used by the wide contact kernels (B3_SIMD_NONE).

/// Scalar lane of `b3MaxW`: `a >= b ? a : b`.
#[inline]
fn max_w(a: f32, b: f32) -> f32 {
    if a >= b {
        a
    } else {
        b
    }
}

/// Scalar lane of `b3SymClampW`: clamp a to [-b, b] as `r = a <= b ? a : b; r = r <= -b ? -b : r`.
#[inline]
fn sym_clamp_w(a: f32, b: f32) -> f32 {
    let r = if a <= b { a } else { b };
    if r <= -b {
        -b
    } else {
        r
    }
}

fn gather_state(states: &[BodyState], index: i32) -> BodyState {
    if index == NULL_INDEX {
        IDENTITY_BODY_STATE
    } else {
        states[index as usize]
    }
}

/// This writes only the velocities back to the solver bodies. (b3ScatterBodies, one lane)
fn scatter_state(
    states: &mut [BodyState],
    index: i32,
    v: crate::math_functions::Vec3,
    w: crate::math_functions::Vec3,
) {
    if index != NULL_INDEX && (states[index as usize].flags & body_flags::DYNAMIC_FLAG) != 0 {
        let state = &mut states[index as usize];
        state.linear_velocity = v;
        state.angular_velocity = w;
    }
}

/// Solve the non-penetration constraints with the soft bias. No friction and no restitution.
/// (b3PushContacts_Convex)
pub fn push_contacts_convex(
    constraints: &mut [ContactConstraint],
    states: &mut [BodyState],
    context: &StepContext,
) {
    let inv_h = context.inv_h;
    let contact_speed = -context.contact_speed;

    for group in constraints.chunks_mut(SIMD_WIDTH) {
        let point_count = wide_point_count(group);
        for contact_constraint in group.iter_mut() {
            push_lane(
                contact_constraint,
                states,
                inv_h,
                contact_speed,
                point_count,
            );
        }
    }
}

/// One lane of `b3PushContacts_Convex`, looping the wide point count.
fn push_lane(
    contact_constraint: &mut ContactConstraint,
    states: &mut [BodyState],
    inv_h: f32,
    contact_speed: f32,
    point_count: usize,
) {
    {
        let index_a = contact_constraint.index_a;
        let index_b = contact_constraint.index_b;

        let m_a = contact_constraint.inv_mass_a;
        let i_a = contact_constraint.inv_i_a;
        let m_b = contact_constraint.inv_mass_b;
        let i_b = contact_constraint.inv_i_b;

        let state_a = gather_state(states, index_a);
        let state_b = gather_state(states, index_b);
        let mut v_a = state_a.linear_velocity;
        let mut w_a = state_a.angular_velocity;
        let mut v_b = state_b.linear_velocity;
        let mut w_b = state_b.angular_velocity;
        let dq_a = state_a.delta_rotation;
        let dq_b = state_b.delta_rotation;

        // Stiffer for static contacts to avoid bodies getting pushed through the ground. C
        // selects this per lane from the null body index; the prepared softness is chosen
        // from the static contact flag, which is set exactly when one body is static.
        let softness = contact_constraint.softness;
        let bias_rate = softness.mass_scale * softness.bias_rate;
        let mass_scale = softness.mass_scale;
        let impulse_scale = softness.impulse_scale;

        let dp = sub(state_b.delta_position, state_a.delta_position);

        let constraint = &mut contact_constraint.constraints[0];
        let normal = constraint.normal;

        for point_index in 0..point_count {
            let cp = &mut constraint.points[point_index];

            // Fixed anchor points for applying impulses
            let r_a = cp.r_a;
            let r_b = cp.r_b;

            // Moving anchors for current separation
            let rs_a = rotate_vector(dq_a, r_a);
            let rs_b = rotate_vector(dq_b, r_b);

            // compute current separation
            // this is subject to round-off error if the anchor is far from the body center of mass
            let ds = add(dp, sub(rs_b, rs_a));
            let s = dot(normal, ds) + cp.base_separation;

            // Apply speculative bias if separation is greater than zero, otherwise apply soft constraint bias
            let separated = s > 0.0;

            // Speculative bias - positive
            let spec_bias = s * inv_h;

            // Overlap bias - negative
            let overlap_bias = max_w(bias_rate * s, contact_speed);
            let velocity_bias = if separated { spec_bias } else { overlap_bias };

            let point_mass_scale = if separated { 1.0 } else { mass_scale };
            let point_impulse_scale = if separated { 0.0 } else { impulse_scale };

            // Relative velocity at contact
            let vr_a = add(v_a, cross(w_a, r_a));
            let vr_b = add(v_b, cross(w_b, r_b));
            let vn = dot(sub(vr_b, vr_a), normal);

            // Compute normal impulse
            let neg_impulse = cp.normal_mass * (point_mass_scale * vn + velocity_bias)
                + point_impulse_scale * cp.normal_impulse;

            // Clamp the accumulated impulse
            let new_impulse = max_w(cp.normal_impulse - neg_impulse, 0.0);
            let delta_impulse = new_impulse - cp.normal_impulse;
            cp.normal_impulse = new_impulse;

            // Apply contact impulse
            let p = mul_sv(delta_impulse, normal);
            w_a = sub(w_a, mul_mv_sym(i_a, cross(r_a, p)));
            v_a = mul_sub(v_a, m_a, p);
            w_b = add(w_b, mul_mv_sym(i_b, cross(r_b, p)));
            v_b = mul_add(v_b, m_b, p);
        }

        scatter_state(states, index_a, v_a, w_a);
        scatter_state(states, index_b, v_b, w_b);
    }
}

/// Solve the normal constraint, friction, and rolling resistance. Deferred restitution augments
/// the normal constraint. (b3SolveContacts_Convex)
///
/// `constraints` must be the full convex prefix of one graph color so the groups of four line
/// up with C's wide constraints.
pub fn solve_contacts_convex(
    constraints: &mut [ContactConstraint],
    states: &mut [BodyState],
    context: &StepContext,
) {
    for group in constraints.chunks_mut(SIMD_WIDTH) {
        // Wide point count: the maximum over the lanes. Shorter lanes run their zero-padded
        // points, like C.
        let point_count = wide_point_count(group);

        // anyRestitution / b3AllZeroW across the lanes and points of the wide constraint.
        let mut have_restitution = false;
        for c in group.iter() {
            let mc = &c.constraints[0];
            for point_index in 0..point_count {
                if -mc.points[point_index].restitution_velocity != 0.0 {
                    have_restitution = true;
                }
            }
        }

        // b3AllZeroW( c->rollingResistance ) == false
        let mut have_rolling = false;
        for c in group.iter() {
            if c.rolling_resistance != 0.0 {
                have_rolling = true;
            }
        }

        // Restitution with more than one point needs iteration to reduce spin.
        // This makes restitution have a significant cost.
        let iteration_count = if have_restitution { point_count } else { 1 };

        for contact_constraint in group.iter_mut() {
            solve_lane(
                contact_constraint,
                states,
                context,
                point_count,
                iteration_count,
                have_restitution,
                have_rolling,
            );
        }
    }
}

/// One lane of `b3SolveContacts_Convex`.
fn solve_lane(
    contact_constraint: &mut ContactConstraint,
    states: &mut [BodyState],
    context: &StepContext,
    point_count: usize,
    iteration_count: usize,
    have_restitution: bool,
    have_rolling: bool,
) {
    let inv_h = context.inv_h;
    let epsilon = f32::EPSILON;

    let index_a = contact_constraint.index_a;
    let index_b = contact_constraint.index_b;

    let m_a = contact_constraint.inv_mass_a;
    let i_a = contact_constraint.inv_i_a;
    let m_b = contact_constraint.inv_mass_b;
    let i_b = contact_constraint.inv_i_b;

    let state_a = gather_state(states, index_a);
    let state_b = gather_state(states, index_b);
    let mut v_a = state_a.linear_velocity;
    let mut w_a = state_a.angular_velocity;
    let mut v_b = state_b.linear_velocity;
    let mut w_b = state_b.angular_velocity;
    let dq_a = state_a.delta_rotation;
    let dq_b = state_b.delta_rotation;

    let friction = contact_constraint.friction;
    let rolling_resistance = contact_constraint.rolling_resistance;
    let rolling_mass = contact_constraint.rolling_mass;

    let constraint = &mut contact_constraint.constraints[0];
    let normal = constraint.normal;
    debug_assert!(constraint.point_count as usize <= point_count);

    let mut keep_restitution = false;

    let dp = sub(state_b.delta_position, state_a.delta_position);

    let mut total_normal_impulse = 0.0;
    let mut total_twist_limit = 0.0;

    let mut velocity_biases = [0.0f32; MAX_MANIFOLD_POINTS];

    for iteration in 0..iteration_count {
        let accumulate = iteration + 1 == iteration_count;

        for point_index in 0..point_count {
            let cp = &mut constraint.points[point_index];

            // Fixed anchor points for applying impulses
            let r_a = cp.r_a;
            let r_b = cp.r_b;

            let velocity_bias;
            if iteration == 0 {
                // Moving anchors for current separation
                let rs_a = rotate_vector(dq_a, r_a);
                let rs_b = rotate_vector(dq_b, r_b);

                // compute current separation
                // this is subject to round-off error if the anchor is far from the body center of mass
                let ds = add(dp, sub(rs_b, rs_a));
                let s = dot(normal, ds) + cp.base_separation;

                // Speculative bias, positive if separated and zero if overlapped
                let mut bias = max_w(0.0, s * inv_h);

                if have_restitution {
                    // A lane with no restitution keeps the speculative bias. A bouncing lane stops
                    // bouncing once every point on its manifold has separated.
                    let neg_restitution_velocity = -cp.restitution_velocity;
                    let restitution_mask = 0.0 > neg_restitution_velocity;
                    let separated = s > 0.0;
                    if restitution_mask {
                        bias = neg_restitution_velocity;
                    }
                    keep_restitution = keep_restitution || (restitution_mask && !separated);
                }

                velocity_biases[point_index] = bias;
                velocity_bias = bias;
            } else {
                velocity_bias = velocity_biases[point_index];
            }

            // Relative velocity at contact
            let vr_a = add(v_a, cross(w_a, r_a));
            let vr_b = add(v_b, cross(w_b, r_b));
            let vn = dot(sub(vr_b, vr_a), normal);

            // Compute normal impulse
            let neg_impulse = cp.normal_mass * (vn + velocity_bias);

            // Clamp the accumulated impulse
            let new_impulse = max_w(cp.normal_impulse - neg_impulse, 0.0);
            let delta_impulse = new_impulse - cp.normal_impulse;
            cp.normal_impulse = new_impulse;

            // Don't accumulate the total impulse while bouncing.
            if accumulate {
                cp.total_normal_impulse += new_impulse;
                total_normal_impulse += new_impulse;
                total_twist_limit += cp.lever_arm * new_impulse;
            }

            // Apply contact impulse
            let p = mul_sv(delta_impulse, normal);
            w_a = sub(w_a, mul_mv_sym(i_a, cross(r_a, p)));
            v_a = mul_sub(v_a, m_a, p);
            w_b = add(w_b, mul_mv_sym(i_b, cross(r_b, p)));
            v_b = mul_add(v_b, m_b, p);
        }
    }

    if have_restitution && !keep_restitution {
        for point_index in 0..point_count {
            constraint.points[point_index].restitution_velocity = 0.0;
        }
    }

    // Rolling resistance
    if have_rolling {
        // flip A/B order to negate
        let mut delta_impulse = mul_mv_sym(rolling_mass, sub(w_a, w_b));
        let old_impulse = constraint.rolling_impulse;
        constraint.rolling_impulse = add(old_impulse, delta_impulse);

        let max_impulse = rolling_resistance * total_normal_impulse;
        let length_squared = dot(constraint.rolling_impulse, constraint.rolling_impulse);

        let mask = length_squared > epsilon + max_impulse * max_impulse;

        // No approximate _mm_rsqrt_ps here to maintain cross-platform determinism
        let normalize = max_impulse / (length_squared.sqrt() + epsilon);
        let mut scale = if mask { normalize } else { 1.0 };

        // Ensure zero rolling resistance yields no impulse
        let rolling_mask = rolling_resistance > 0.0;
        scale = if rolling_mask { scale } else { 0.0 };

        constraint.rolling_impulse = mul_sv(scale, constraint.rolling_impulse);

        delta_impulse = sub(constraint.rolling_impulse, old_impulse);

        w_a = sub(w_a, mul_mv_sym(i_a, delta_impulse));
        w_b = add(w_b, mul_mv_sym(i_b, delta_impulse));
    }

    // Central twist friction
    {
        let twist_speed = dot(constraint.normal, sub(w_b, w_a));
        let max_impulse = friction * total_twist_limit;
        let mut delta_impulse = -(constraint.twist_mass * twist_speed);
        let old_impulse = constraint.twist_impulse;
        constraint.twist_impulse = sym_clamp_w(old_impulse + delta_impulse, max_impulse);
        delta_impulse = constraint.twist_impulse - old_impulse;

        let l = mul_sv(delta_impulse, constraint.normal);
        w_a = sub(w_a, mul_mv_sym(i_a, l));
        w_b = add(w_b, mul_mv_sym(i_b, l));
    }

    // Central friction
    {
        let tangent1 = constraint.tangent1;
        let tangent2 = constraint.tangent2;

        // Fixed anchor points for applying impulses
        let r_a = constraint.center_a;
        let r_b = constraint.center_b;

        // Relative tangent velocity at contact
        let vr_a = add(v_a, cross(w_a, r_a));
        let vr_b = add(v_b, cross(w_b, r_b));
        let vr = sub(vr_b, vr_a);
        let vt = Vec2 {
            x: dot(vr, tangent1) - constraint.tangent_velocity1,
            y: dot(vr, tangent2) - constraint.tangent_velocity2,
        };

        // Incremental tangent impulse
        let tm = mul_mv2_sym(constraint.tangent_mass, vt);
        let mut delta_impulse = Vec2 { x: -tm.x, y: -tm.y };
        let mut new_impulse = Vec2 {
            x: constraint.friction_impulse.x + delta_impulse.x,
            y: constraint.friction_impulse.y + delta_impulse.y,
        };

        let max_impulse = friction * total_normal_impulse;

        // Clamp the accumulated impulse
        let length_squared = dot2(new_impulse, new_impulse);

        // Max impulse can be zero
        let mask = length_squared > max_impulse * max_impulse;

        // No approximate _mm_rsqrt_ps here to maintain cross-platform determinism. Add epsilon to
        // avoid divide by zero.
        let scale = if mask {
            max_impulse / (length_squared.sqrt() + epsilon)
        } else {
            1.0
        };
        new_impulse.x *= scale;
        new_impulse.y *= scale;

        delta_impulse = sub2(new_impulse, constraint.friction_impulse);
        constraint.friction_impulse = new_impulse;

        // Apply delta impulse
        let p = blend2(delta_impulse.x, tangent1, delta_impulse.y, tangent2);
        w_a = sub(w_a, mul_mv_sym(i_a, cross(r_a, p)));
        v_a = mul_sub(v_a, m_a, p);
        w_b = add(w_b, mul_mv_sym(i_b, cross(r_b, p)));
        v_b = mul_add(v_b, m_b, p);
    }

    scatter_state(states, index_a, v_a, w_a);
    scatter_state(states, index_b, v_b, w_b);
}
