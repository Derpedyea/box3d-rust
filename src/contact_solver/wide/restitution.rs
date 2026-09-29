//! Post-solve restitution for wide convex contact constraints. (b3ApplyRestitution_Convex)
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use super::{
    add_vw, cross_w, dot_w, gather_bodies, make_matrix_from_quat_w, mul_add_mvw, mul_add_svw,
    mul_m3vw, mul_sub_mvw, mul_sub_svw, mul_svw, scatter_bodies, sub_vw, ContactConstraintWide,
};
use crate::body::BodyState;
use crate::simd_scalar::{
    add_w, all_zero_w, and_w, blend_w, greater_than_w, less_than_w, max_w, min_w, mul_w, neg_w,
    splat_w, sub_w, zero_w,
};
use crate::solver::StepContext;

/// Poisson kinetic restitution for the wide convex constraints of one color.
/// (b3ApplyRestitution_Convex)
pub fn apply_restitution_convex(
    constraints: &mut [ContactConstraintWide],
    states: &mut [BodyState],
    context: &StepContext,
    restitution_threshold: f32,
    propagate: bool,
) {
    let inv_h = splat_w(context.inv_h);
    let neg_restitution_threshold = splat_w(-restitution_threshold);
    let zero = zero_w();

    for c in constraints.iter_mut() {
        if !propagate && all_zero_w(c.restitution) {
            continue;
        }

        let point_count = c.point_count();

        let mut b_a = gather_bodies(states, &c.index_a);
        let mut b_b = gather_bodies(states, &c.index_b);

        let restitution_mask = greater_than_w(c.restitution, zero);
        let dp = sub_vw(b_b.dp, b_a.dp);
        let dq_a = make_matrix_from_quat_w(b_a.dq);
        let dq_b = make_matrix_from_quat_w(b_b.dq);

        for cp in c.points.iter_mut().take(point_count) {
            let r_a = cp.anchor_as;
            let r_b = cp.anchor_bs;

            let normal_mass = if propagate {
                cp.normal_masses
            } else {
                blend_w(zero, cp.normal_masses, restitution_mask)
            };

            let compression_impulse = sub_w(cp.total_normal_impulses, cp.restitution_impulses);
            let armed = and_w(
                and_w(
                    restitution_mask,
                    less_than_w(cp.relative_velocities, neg_restitution_threshold),
                ),
                greater_than_w(compression_impulse, zero),
            );

            let rs_a = mul_m3vw(dq_a, r_a);
            let rs_b = mul_m3vw(dq_b, r_b);
            let ds = add_vw(dp, sub_vw(rs_b, rs_a));
            let s = add_w(dot_w(c.normal, ds), cp.base_separations);

            let spec_bias = max_w(zero, mul_w(s, inv_h));
            let velocity_bias = blend_w(
                spec_bias,
                mul_w(c.restitution, cp.relative_velocities),
                armed,
            );

            let vr_a = add_vw(b_a.v, cross_w(b_a.w, r_a));
            let vr_b = add_vw(b_b.v, cross_w(b_b.w, r_b));
            let vn = dot_w(sub_vw(vr_b, vr_a), c.normal);

            let neg_impulse = mul_w(normal_mass, add_w(vn, velocity_bias));

            let new_impulse = max_w(sub_w(cp.normal_impulses, neg_impulse), zero);
            let mut impulse = sub_w(new_impulse, cp.normal_impulses);

            let approach_impulse = min_w(
                max_w(neg_w(mul_w(normal_mass, vn)), zero),
                max_w(impulse, zero),
            );
            let allowance = sub_w(
                mul_w(c.restitution, add_w(compression_impulse, approach_impulse)),
                cp.restitution_impulses,
            );
            let max_impulse = add_w(approach_impulse, max_w(allowance, zero));
            impulse = blend_w(impulse, min_w(impulse, max_impulse), armed);

            cp.normal_impulses = add_w(cp.normal_impulses, impulse);
            cp.restitution_impulses =
                add_w(cp.restitution_impulses, sub_w(impulse, approach_impulse));
            cp.total_normal_impulses = add_w(cp.total_normal_impulses, impulse);

            let p = mul_svw(impulse, c.normal);
            b_a.w = mul_sub_mvw(b_a.w, c.inv_i_a, cross_w(r_a, p));
            b_a.v = mul_sub_svw(b_a.v, c.inv_mass_a, p);
            b_b.w = mul_add_mvw(b_b.w, c.inv_i_b, cross_w(r_b, p));
            b_b.v = mul_add_svw(b_b.v, c.inv_mass_b, p);
        }

        scatter_bodies(states, &c.index_a, &b_a);
        scatter_bodies(states, &c.index_b, &b_b);
    }
}
