//! Prepare wide convex contact constraints. (b3PrepareContacts_Convex)
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use super::{
    add_sym_w, gather_bodies, invert_sym_w, lane, mul_mvw, perp_w, ContactConstraintPointWide,
    ContactConstraintWide, SymMatrix2W, SymMatrix3W, Vec2W, EMPTY_LANE, SYM3W_ZERO,
};
use crate::body::{decode_awake_index, BodySim, BodyState};
use crate::constants::{speculative_distance, MAX_MANIFOLD_POINTS, MIN_FRICTION_WEIGHT};
use crate::contact::{contact_flags, Contact};
use crate::core::SIMD_WIDTH;
use crate::manifold::{Manifold, ManifoldPoint};
use crate::simd_scalar::{
    abs_w, add_vw, add_w, all_zero_w, and_w, any_true_w, blend_w, cross_w, div_w, dot_w,
    greater_than_w, max_w, min_w, mul_add_svw, mul_svw, mul_w, neg_w, set_w, splat_w, sqrt_w,
    sub_vw, sub_w, zero_w, Vec3W,
};

macro_rules! gather_lanes {
    ($lanes:expr, |$l:ident| $field:expr) => {{
        let f = |$l: usize| $field;
        set_w(f(0), f(1), f(2), f(3))
    }};
}

/// (b3GatherInvInertiaW)
fn gather_inv_inertia_w(sims: &[&BodySim; SIMD_WIDTH]) -> SymMatrix3W {
    SymMatrix3W {
        cxx: set_w(
            sims[0].inv_inertia_world.cx.x,
            sims[1].inv_inertia_world.cx.x,
            sims[2].inv_inertia_world.cx.x,
            sims[3].inv_inertia_world.cx.x,
        ),
        cxy: set_w(
            sims[0].inv_inertia_world.cx.y,
            sims[1].inv_inertia_world.cx.y,
            sims[2].inv_inertia_world.cx.y,
            sims[3].inv_inertia_world.cx.y,
        ),
        cxz: set_w(
            sims[0].inv_inertia_world.cx.z,
            sims[1].inv_inertia_world.cx.z,
            sims[2].inv_inertia_world.cx.z,
            sims[3].inv_inertia_world.cx.z,
        ),
        cyy: set_w(
            sims[0].inv_inertia_world.cy.y,
            sims[1].inv_inertia_world.cy.y,
            sims[2].inv_inertia_world.cy.y,
            sims[3].inv_inertia_world.cy.y,
        ),
        cyz: set_w(
            sims[0].inv_inertia_world.cy.z,
            sims[1].inv_inertia_world.cy.z,
            sims[2].inv_inertia_world.cy.z,
            sims[3].inv_inertia_world.cy.z,
        ),
        czz: set_w(
            sims[0].inv_inertia_world.cz.z,
            sims[1].inv_inertia_world.cz.z,
            sims[2].inv_inertia_world.cz.z,
            sims[3].inv_inertia_world.cz.z,
        ),
    }
}

/// Prepare the wide constraints for the convex contacts of one graph color. Returns true if
/// any lane has restitution. (b3PrepareContacts_Convex, one color)
///
/// `contact_ids` is the color's convex contact array. Writes the sampled normal velocity onto
/// the manifold points of lanes that enable hit events, like C.
pub fn prepare_contacts_convex(
    constraints: &mut Vec<ContactConstraintWide>,
    contact_ids: &[i32],
    contacts: &mut [Contact],
    sims: &[BodySim],
    states: &[BodyState],
    enable_warm_starting: bool,
) -> bool {
    let zero_sim = BodySim {
        inv_mass: 0.0,
        inv_inertia_world: crate::math_functions::MAT3_ZERO,
        ..BodySim::default()
    };
    let zero_manifold = Manifold::default();

    let zero_w_ = zero_w();
    let one_w = splat_w(1.0);
    let two_w = splat_w(2.0);
    let warm_start_scale = if enable_warm_starting { one_w } else { zero_w_ };
    let inv_tau = splat_w(1.0 / speculative_distance());
    let min_friction_weight = splat_w(MIN_FRICTION_WEIGHT);
    let min_det = splat_w(1000.0 * f32::MIN_POSITIVE);
    let mut any_restitution = false;

    let color_contact_count = contact_ids.len();
    let wide_count = if color_contact_count == 0 {
        0
    } else {
        ((color_contact_count - 1) >> 2) + 1
    };

    constraints.clear();
    constraints.reserve(wide_count);

    for local_wide_index in 0..wide_count {
        let mut index_a = [0i32; SIMD_WIDTH];
        let mut index_b = [0i32; SIMD_WIDTH];
        let mut point_counts = [0i32; SIMD_WIDTH];
        let mut lane_ids = [EMPTY_LANE; SIMD_WIDTH];
        let mut hit_event_lanes = 0u32;

        let mut sim_index_a = [crate::core::NULL_INDEX; SIMD_WIDTH];
        let mut sim_index_b = [crate::core::NULL_INDEX; SIMD_WIDTH];

        for lane_index in 0..SIMD_WIDTH {
            let contact_index = SIMD_WIDTH * local_wide_index + lane_index;
            if contact_index < color_contact_count {
                let contact_id = contact_ids[contact_index];
                let contact = &contacts[contact_id as usize];
                debug_assert!(contact.manifold_count() == 1);

                let a = decode_awake_index(contact.encoded_body_sim_a);
                let b = decode_awake_index(contact.encoded_body_sim_b);

                index_a[lane_index] = a + 1;
                index_b[lane_index] = b + 1;
                point_counts[lane_index] = contact.manifolds[0].point_count;
                lane_ids[lane_index] = contact_id;
                sim_index_a[lane_index] = a;
                sim_index_b[lane_index] = b;
                if (contact.flags & contact_flags::SIM_ENABLE_HIT_EVENT) != 0 {
                    hit_event_lanes |= 1 << lane_index;
                }
            }
        }

        let sim_lanes_a: [&BodySim; SIMD_WIDTH] = core::array::from_fn(|i| {
            if sim_index_a[i] == crate::core::NULL_INDEX {
                &zero_sim
            } else {
                &sims[sim_index_a[i] as usize]
            }
        });
        let sim_lanes_b: [&BodySim; SIMD_WIDTH] = core::array::from_fn(|i| {
            if sim_index_b[i] == crate::core::NULL_INDEX {
                &zero_sim
            } else {
                &sims[sim_index_b[i] as usize]
            }
        });
        let contact_lanes: [Option<&Contact>; SIMD_WIDTH] = core::array::from_fn(|i| {
            if lane_ids[i] == EMPTY_LANE {
                None
            } else {
                Some(&contacts[lane_ids[i] as usize])
            }
        });
        let manifold_lanes: [&Manifold; SIMD_WIDTH] =
            core::array::from_fn(|i| match contact_lanes[i] {
                Some(c) => &c.manifolds[0],
                None => &zero_manifold,
            });

        let m_a = gather_lanes!(sim_lanes_a, |l| sim_lanes_a[l].inv_mass);
        let m_b = gather_lanes!(sim_lanes_b, |l| sim_lanes_b[l].inv_mass);
        let i_a = gather_inv_inertia_w(&sim_lanes_a);
        let i_b = gather_inv_inertia_w(&sim_lanes_b);

        let contact_field = |l: usize, f: fn(&Contact) -> f32| match contact_lanes[l] {
            Some(c) => f(c),
            None => 0.0,
        };
        let friction = gather_lanes!(contact_lanes, |l| contact_field(l, |c| c.friction));
        let rolling_resistance = gather_lanes!(contact_lanes, |l| contact_field(l, |c| c
            .rolling_resistance));
        let restitution = gather_lanes!(contact_lanes, |l| contact_field(l, |c| c.restitution));
        let tangent_velocity = Vec3W {
            x: gather_lanes!(contact_lanes, |l| contact_field(l, |c| c
                .tangent_velocity
                .x)),
            y: gather_lanes!(contact_lanes, |l| contact_field(l, |c| c
                .tangent_velocity
                .y)),
            z: gather_lanes!(contact_lanes, |l| contact_field(l, |c| c
                .tangent_velocity
                .z)),
        };

        let normal = Vec3W {
            x: gather_lanes!(manifold_lanes, |l| manifold_lanes[l].normal.x),
            y: gather_lanes!(manifold_lanes, |l| manifold_lanes[l].normal.y),
            z: gather_lanes!(manifold_lanes, |l| manifold_lanes[l].normal.z),
        };
        let twist_impulse = gather_lanes!(manifold_lanes, |l| manifold_lanes[l].twist_impulse);
        let friction_impulse = Vec3W {
            x: gather_lanes!(manifold_lanes, |l| manifold_lanes[l].friction_impulse.x),
            y: gather_lanes!(manifold_lanes, |l| manifold_lanes[l].friction_impulse.y),
            z: gather_lanes!(manifold_lanes, |l| manifold_lanes[l].friction_impulse.z),
        };
        let rolling_impulse = Vec3W {
            x: gather_lanes!(manifold_lanes, |l| manifold_lanes[l].rolling_impulse.x),
            y: gather_lanes!(manifold_lanes, |l| manifold_lanes[l].rolling_impulse.y),
            z: gather_lanes!(manifold_lanes, |l| manifold_lanes[l].rolling_impulse.z),
        };

        let tangent1 = perp_w(normal);
        let tangent2 = cross_w(tangent1, normal);
        let tangent_velocity1 = dot_w(tangent_velocity, tangent1);
        let tangent_velocity2 = dot_w(tangent_velocity, tangent2);

        let rolling_mask = greater_than_w(rolling_resistance, zero_w_);
        let c_twist_impulse = mul_w(warm_start_scale, twist_impulse);
        let c_rolling_impulse = Vec3W {
            x: blend_w(
                zero_w_,
                mul_w(warm_start_scale, rolling_impulse.x),
                rolling_mask,
            ),
            y: blend_w(
                zero_w_,
                mul_w(warm_start_scale, rolling_impulse.y),
                rolling_mask,
            ),
            z: blend_w(
                zero_w_,
                mul_w(warm_start_scale, rolling_impulse.z),
                rolling_mask,
            ),
        };
        let c_friction_impulse = Vec2W {
            x: mul_w(warm_start_scale, dot_w(friction_impulse, tangent1)),
            y: mul_w(warm_start_scale, dot_w(friction_impulse, tangent2)),
        };

        let point_count_w = set_w(
            point_counts[0] as f32,
            point_counts[1] as f32,
            point_counts[2] as f32,
            point_counts[3] as f32,
        );

        let mut center_a = Vec3W {
            x: zero_w_,
            y: zero_w_,
            z: zero_w_,
        };
        let mut center_b = center_a;
        let mut total_friction_weight = zero_w_;

        let empty_point = ContactConstraintPointWide {
            anchor_as: center_a,
            anchor_bs: center_a,
            base_separations: zero_w_,
            normal_impulses: zero_w_,
            total_normal_impulses: zero_w_,
            normal_masses: zero_w_,
            lever_arms: zero_w_,
            relative_velocities: zero_w_,
            restitution_impulses: zero_w_,
        };
        let mut points = [empty_point; MAX_MANIFOLD_POINTS];

        for (point_index, cp) in points.iter_mut().enumerate() {
            let point_mask = greater_than_w(point_count_w, splat_w(point_index as f32));

            let mp: [&ManifoldPoint; SIMD_WIDTH] =
                core::array::from_fn(|l| &manifold_lanes[l].points[point_index]);

            let mut r_a = Vec3W {
                x: set_w(
                    mp[0].anchor_a.x,
                    mp[1].anchor_a.x,
                    mp[2].anchor_a.x,
                    mp[3].anchor_a.x,
                ),
                y: set_w(
                    mp[0].anchor_a.y,
                    mp[1].anchor_a.y,
                    mp[2].anchor_a.y,
                    mp[3].anchor_a.y,
                ),
                z: set_w(
                    mp[0].anchor_a.z,
                    mp[1].anchor_a.z,
                    mp[2].anchor_a.z,
                    mp[3].anchor_a.z,
                ),
            };
            let mut r_b = Vec3W {
                x: set_w(
                    mp[0].anchor_b.x,
                    mp[1].anchor_b.x,
                    mp[2].anchor_b.x,
                    mp[3].anchor_b.x,
                ),
                y: set_w(
                    mp[0].anchor_b.y,
                    mp[1].anchor_b.y,
                    mp[2].anchor_b.y,
                    mp[3].anchor_b.y,
                ),
                z: set_w(
                    mp[0].anchor_b.z,
                    mp[1].anchor_b.z,
                    mp[2].anchor_b.z,
                    mp[3].anchor_b.z,
                ),
            };
            let mut separation = set_w(
                mp[0].separation,
                mp[1].separation,
                mp[2].separation,
                mp[3].separation,
            );
            let mut normal_impulse = set_w(
                mp[0].normal_impulse,
                mp[1].normal_impulse,
                mp[2].normal_impulse,
                mp[3].normal_impulse,
            );

            r_a.x = blend_w(zero_w_, r_a.x, point_mask);
            r_a.y = blend_w(zero_w_, r_a.y, point_mask);
            r_a.z = blend_w(zero_w_, r_a.z, point_mask);
            r_b.x = blend_w(zero_w_, r_b.x, point_mask);
            r_b.y = blend_w(zero_w_, r_b.y, point_mask);
            r_b.z = blend_w(zero_w_, r_b.z, point_mask);
            separation = blend_w(zero_w_, separation, point_mask);
            normal_impulse = blend_w(zero_w_, normal_impulse, point_mask);

            // C0 friction center decay. Needed to prevent spinning top drift (GyroscopicPrecession sample).
            // See details in b3PrepareContacts_Mesh. This code should stay in sync.
            let mut weight = min_w(
                max_w(
                    sub_w(two_w, mul_w(separation, inv_tau)),
                    min_friction_weight,
                ),
                one_w,
            );
            weight = blend_w(zero_w_, weight, point_mask);
            center_a = mul_add_svw(center_a, weight, r_a);
            center_b = mul_add_svw(center_b, weight, r_b);
            total_friction_weight = add_w(total_friction_weight, weight);

            cp.anchor_as = r_a;
            cp.anchor_bs = r_b;
            cp.base_separations = sub_w(separation, dot_w(sub_vw(r_b, r_a), normal));
            cp.normal_impulses = mul_w(warm_start_scale, normal_impulse);
            cp.total_normal_impulses = zero_w_;
            cp.relative_velocities = zero_w_;
            cp.restitution_impulses = zero_w_;

            let rn_a = cross_w(r_a, normal);
            let rn_b = cross_w(r_b, normal);
            let mut k_normal = add_w(m_a, m_b);
            k_normal = add_w(k_normal, dot_w(rn_a, mul_mvw(i_a, rn_a)));
            k_normal = add_w(k_normal, dot_w(rn_b, mul_mvw(i_b, rn_b)));
            let valid = and_w(greater_than_w(k_normal, zero_w_), point_mask);
            cp.normal_masses = blend_w(zero_w_, div_w(one_w, k_normal), valid);
        }

        let inv_weight = blend_w(
            zero_w_,
            div_w(one_w, total_friction_weight),
            greater_than_w(total_friction_weight, zero_w_),
        );
        center_a = mul_svw(inv_weight, center_a);
        center_b = mul_svw(inv_weight, center_b);

        for (point_index, cp) in points.iter_mut().enumerate() {
            let point_mask = greater_than_w(point_count_w, splat_w(point_index as f32));
            let d = sub_vw(cp.anchor_as, center_a);
            cp.lever_arms = blend_w(zero_w_, sqrt_w(dot_w(d, d)), point_mask);
        }

        let tangent_mass = {
            let rt_a1 = cross_w(center_a, tangent1);
            let rt_a2 = cross_w(center_a, tangent2);
            let rt_b1 = cross_w(center_b, tangent1);
            let rt_b2 = cross_w(center_b, tangent2);
            let i_a_rt_a1 = mul_mvw(i_a, rt_a1);
            let i_a_rt_a2 = mul_mvw(i_a, rt_a2);
            let i_b_rt_b1 = mul_mvw(i_b, rt_b1);
            let i_b_rt_b2 = mul_mvw(i_b, rt_b2);

            let kxx = add_w(
                add_w(add_w(m_a, m_b), dot_w(rt_a1, i_a_rt_a1)),
                dot_w(rt_b1, i_b_rt_b1),
            );
            let kyy = add_w(
                add_w(add_w(m_a, m_b), dot_w(rt_a2, i_a_rt_a2)),
                dot_w(rt_b2, i_b_rt_b2),
            );
            let kxy = add_w(dot_w(rt_a1, i_a_rt_a2), dot_w(rt_b1, i_b_rt_b2));

            let det = sub_w(mul_w(kxx, kyy), mul_w(kxy, kxy));
            let valid = greater_than_w(abs_w(det), min_det);
            let inv_det = blend_w(zero_w_, div_w(one_w, det), valid);
            SymMatrix2W {
                cxx: mul_w(inv_det, kyy),
                cxy: neg_w(mul_w(inv_det, kxy)),
                cyy: mul_w(inv_det, kxx),
            }
        };

        let inv_i_ab = add_sym_w(i_a, i_b);

        let twist_mass = {
            let k_twist = dot_w(normal, mul_mvw(inv_i_ab, normal));
            blend_w(
                zero_w_,
                div_w(one_w, k_twist),
                greater_than_w(k_twist, zero_w_),
            )
        };

        let rolling_mass = if !all_zero_w(rolling_resistance) {
            invert_sym_w(inv_i_ab)
        } else {
            SYM3W_ZERO
        };

        // Only sample contact point normal velocity if needed.
        let have_restitution = any_true_w(greater_than_w(restitution, zero_w_));
        any_restitution = any_restitution || have_restitution;

        if hit_event_lanes != 0 || have_restitution {
            let b_a = gather_bodies(states, &index_a);
            let b_b = gather_bodies(states, &index_b);

            for (point_index, cp) in points.iter_mut().enumerate() {
                let vr_a = add_vw(b_a.v, cross_w(b_a.w, cp.anchor_as));
                let vr_b = add_vw(b_b.v, cross_w(b_b.w, cp.anchor_bs));
                let vn = dot_w(normal, sub_vw(vr_b, vr_a));
                cp.relative_velocities = vn;

                if hit_event_lanes != 0 {
                    for l in 0..SIMD_WIDTH {
                        if (hit_event_lanes & (1 << l)) != 0
                            && (point_index as i32) < point_counts[l]
                        {
                            contacts[lane_ids[l] as usize].manifolds[0].points[point_index]
                                .normal_velocity = lane(vn, l);
                        }
                    }
                }
            }
        }

        constraints.push(ContactConstraintWide {
            index_a,
            index_b,
            point_counts,
            inv_mass_a: m_a,
            inv_mass_b: m_b,
            inv_i_a: i_a,
            inv_i_b: i_b,
            normal,
            tangent1,
            tangent2,
            center_a,
            center_b,
            twist_mass,
            twist_impulse: c_twist_impulse,
            tangent_mass,
            friction_impulse: c_friction_impulse,
            rolling_mass,
            rolling_impulse: c_rolling_impulse,
            friction,
            rolling_resistance,
            tangent_velocity1,
            tangent_velocity2,
            restitution,
            contact_ids: lane_ids,
            points,
        });
    }

    any_restitution
}
