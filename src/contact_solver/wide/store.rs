//! Store wide convex impulses back onto the contact manifolds. (b3StoreImpulses_Convex)
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use super::{lane, ContactConstraintWide, EMPTY_LANE, SIMD_WIDTH};
use crate::bitset::BitSet;
use crate::contact::{contact_flags, Contact};
use crate::math_functions::Vec3;

/// Store impulses by contact constraint and flag hit-event candidates.
/// (b3StoreImpulses_Convex)
pub fn store_impulses_convex(
    constraints: &[ContactConstraintWide],
    contacts: &mut [Contact],
    hit_event_bit_set: &mut BitSet,
    has_hit_events: &mut bool,
    neg_hit_threshold: f32,
) {
    for c in constraints {
        for l in 0..SIMD_WIDTH {
            let contact_id = c.contact_ids[l];
            if contact_id == EMPTY_LANE {
                break;
            }

            let contact = &mut contacts[contact_id as usize];
            let m = &mut contact.manifolds[0];

            let f1 = lane(c.friction_impulse.x, l);
            let f2 = lane(c.friction_impulse.y, l);
            m.friction_impulse = Vec3 {
                x: f1 * lane(c.tangent1.x, l) + f2 * lane(c.tangent2.x, l),
                y: f1 * lane(c.tangent1.y, l) + f2 * lane(c.tangent2.y, l),
                z: f1 * lane(c.tangent1.z, l) + f2 * lane(c.tangent2.z, l),
            };
            m.twist_impulse = lane(c.twist_impulse, l);
            m.rolling_impulse = Vec3 {
                x: lane(c.rolling_impulse.x, l),
                y: lane(c.rolling_impulse.y, l),
                z: lane(c.rolling_impulse.z, l),
            };

            let point_count = m.point_count as usize;
            for point_index in 0..point_count {
                let cp = &c.points[point_index];
                let mp = &mut m.points[point_index];
                mp.normal_impulse = lane(cp.normal_impulses, l);
                mp.total_normal_impulse = lane(cp.total_normal_impulses, l);
            }

            if (contact.flags & contact_flags::SIM_ENABLE_HIT_EVENT) != 0 {
                for k in 0..point_count {
                    let mp = &contact.manifolds[0].points[k];

                    // Need to check total impulse because the point may be speculative and not colliding
                    if mp.normal_velocity < neg_hit_threshold && mp.total_normal_impulse > 0.0 {
                        hit_event_bit_set.set_bit(contact.contact_id as u32);
                        *has_hit_events = true;
                        break;
                    }
                }
            }
        }
    }
}
