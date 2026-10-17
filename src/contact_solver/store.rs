//! Store solved impulses back onto contact manifolds. (b3StoreImpulses_Mesh,
//! b3StoreImpulses_Convex)
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use super::ContactConstraint;
use crate::contact::Contact;
use crate::math_functions::blend2;

/// Write the impulses of a constraint onto its contact and return whether the contact is a
/// hit-event candidate. Also writes the normal velocity prepare sampled onto the manifold
/// points, like C's prepare. (b3StoreImpulses_Mesh, one contact)
pub fn store_impulses(
    contact_constraint: &ContactConstraint,
    contact: &mut Contact,
    neg_hit_threshold: f32,
) -> bool {
    use crate::contact::contact_flags;
    debug_assert!(contact.contact_id == contact_constraint.contact_id);
    let hit_events = (contact.flags & contact_flags::SIM_ENABLE_HIT_EVENT) != 0;

    let manifold_count = contact_constraint.manifold_count;
    debug_assert!(manifold_count == contact.manifold_count());

    for manifold_index in 0..manifold_count as usize {
        let manifold = &mut contact.manifolds[manifold_index];
        let constraint = &contact_constraint.constraints[manifold_index];
        manifold.twist_impulse = constraint.twist_impulse;
        manifold.friction_impulse = blend2(
            constraint.friction_impulse.x,
            constraint.tangent1,
            constraint.friction_impulse.y,
            constraint.tangent2,
        );
        manifold.rolling_impulse = constraint.rolling_impulse;

        let count = constraint.point_count;
        debug_assert!(count == manifold.point_count);
        for point_index in 0..count as usize {
            let cp = &constraint.points[point_index];
            let mp = &mut manifold.points[point_index];
            mp.normal_impulse = cp.normal_impulse;
            mp.total_normal_impulse = cp.total_normal_impulse;
            mp.normal_velocity = if hit_events {
                cp.relative_velocity
            } else {
                0.0
            };
        }
    }

    flag_hit_event(contact_constraint, contact, neg_hit_threshold)
}

/// Whether a stored contact is a hit-event candidate. (the hit check of
/// b3StoreImpulses_Mesh)
fn flag_hit_event(
    contact_constraint: &ContactConstraint,
    contact: &Contact,
    neg_hit_threshold: f32,
) -> bool {
    use crate::contact::contact_flags;

    if (contact.flags & contact_flags::SIM_ENABLE_HIT_EVENT) == 0 {
        return false;
    }

    let manifold_count = contact_constraint.manifold_count;
    for manifold_index in 0..manifold_count as usize {
        let manifold = &contact.manifolds[manifold_index];
        for point_index in 0..manifold.point_count as usize {
            let mp = &manifold.points[point_index];

            // Need to check total impulse because the point may be speculative and not colliding
            if mp.normal_velocity < neg_hit_threshold && mp.total_normal_impulse > 0.0 {
                return true;
            }
        }
    }
    false
}
