//! Store wide convex impulses back onto the contact manifolds. (b3StoreImpulses_Convex)
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use super::{lane, ContactConstraintWide};
use crate::contact::{contact_flags, Contact};
use crate::math_functions::Vec3;

/// Store the impulses of lane `l` of constraint `c` onto its contact, and return whether the
/// contact is a hit-event candidate. Also writes the normal velocity prepare sampled onto the
/// manifold points of contacts that enable hit events, like C's prepare.
/// (b3StoreImpulses_Convex, one lane)
pub fn store_impulses_convex(
    c: &ContactConstraintWide,
    l: usize,
    contact: &mut Contact,
    neg_hit_threshold: f32,
) -> bool {
    debug_assert!(c.contact_ids[l] == contact.contact_id);
    let hit_events = (contact.flags & contact_flags::SIM_ENABLE_HIT_EVENT) != 0;
    let m = &mut contact.manifolds[0];

    let f1 = lane(&c.friction_impulse.x, l);
    let f2 = lane(&c.friction_impulse.y, l);
    m.friction_impulse = Vec3 {
        x: f1 * lane(&c.tangent1.x, l) + f2 * lane(&c.tangent2.x, l),
        y: f1 * lane(&c.tangent1.y, l) + f2 * lane(&c.tangent2.y, l),
        z: f1 * lane(&c.tangent1.z, l) + f2 * lane(&c.tangent2.z, l),
    };
    m.twist_impulse = lane(&c.twist_impulse, l);
    m.rolling_impulse = Vec3 {
        x: lane(&c.rolling_impulse.x, l),
        y: lane(&c.rolling_impulse.y, l),
        z: lane(&c.rolling_impulse.z, l),
    };

    let point_count = m.point_count as usize;
    for (cp, mp) in c.points.iter().zip(&mut m.points[..point_count]) {
        mp.normal_impulse = lane(&cp.normal_impulses, l);
        mp.total_normal_impulse = lane(&cp.total_normal_impulses, l);
        if hit_events {
            mp.normal_velocity = lane(&cp.relative_velocities, l);
        }
    }

    // Need to check total impulse because the point may be speculative and not colliding
    hit_events
        && m.points[..point_count]
            .iter()
            .any(|mp| mp.normal_velocity < neg_hit_threshold && mp.total_normal_impulse > 0.0)
}
