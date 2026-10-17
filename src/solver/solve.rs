//! Serial solve driver from solver.c: island split → prepare → sub-step loop (push, relax
//! with deferred restitution) → store → finalize → joint events → hit events →
//! broad-phase enlarge → island sleep.
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT
#![allow(clippy::needless_range_loop)]

use super::continuous::solve_continuous;
use super::integrate::finalize_bodies;
use super::stages::ColorConstraints;
use super::StepContext;
use crate::body::body_flags;
use crate::broad_phase::{proxy_id, proxy_type};
use crate::constants::GRAPH_COLOR_COUNT;
use crate::constraint_graph::{ConstraintGraph, OVERFLOW_INDEX};
use crate::contact::{contact_flags, Contact};
use crate::contact_solver::wide::ContactConstraintWide;
use crate::contact_solver::ContactConstraint;
use crate::core::NULL_INDEX;
use crate::core::SIMD_WIDTH;
use crate::events::{BodyMoveEvent, JointEvent};
use crate::id::{BodyId, JointId};
use crate::island::split_island;
use crate::joint::{prepare_joint, JointSim};
use crate::math_functions::{aabb_contains, WORLD_TRANSFORM_IDENTITY};
use crate::solver_set::{try_sleep_island, AWAKE_SET};
use crate::types::BodyType;
use crate::world::World;

/// The graph's contacts borrowed out of the contact array, per color: convex contacts in each
/// color's convex contact order, scalar (mesh and overflow) contacts in its spec order. Lets
/// prepare and store write the contacts of a color from many threads.
struct ColorContacts<'a> {
    lanes: Vec<&'a mut Contact>,
    /// Start of color `c`'s convex contacts at `2 * c`, of its scalar ones at `2 * c + 1`,
    /// and the end at the last entry.
    starts: Vec<usize>,
}

impl<'a> ColorContacts<'a> {
    /// Every color's (convex, scalar) contacts, as disjoint slices.
    fn split(&mut self) -> Vec<(&mut [&'a mut Contact], &mut [&'a mut Contact])> {
        let mut rest = &mut self.lanes[..];
        let mut taken = 0;
        let mut out = Vec::with_capacity(self.starts.len() / 2);
        for color in self.starts[1..].chunks(2) {
            let (convex, tail) = rest.split_at_mut(color[0] - taken);
            let (scalar, tail) = tail.split_at_mut(color[1] - color[0]);
            taken = color[1];
            rest = tail;
            out.push((convex, scalar));
        }
        out
    }
}

fn color_contacts<'a>(
    pool: Option<&crate::parallel::ThreadPool>,
    contacts: &'a mut [Contact],
    graph: &ConstraintGraph,
) -> ColorContacts<'a> {
    let mut starts = Vec::with_capacity(2 * graph.colors.len() + 1);
    let mut total = 0;
    for color in &graph.colors {
        starts.push(total);
        total += color.convex_contacts.len();
        starts.push(total);
        total += color.contacts.len();
    }
    starts.push(total);

    let lanes = crate::parallel::borrow_by_slot(pool, contacts, total, |contact| {
        let color_index = contact.color_index;
        if color_index == NULL_INDEX || contact.contact_id == NULL_INDEX {
            return None;
        }
        let color_index = color_index as usize;
        let is_scalar = (contact.flags & contact_flags::SIM_MESH_CONTACT) != 0
            || color_index == OVERFLOW_INDEX as usize;
        let slot = starts[2 * color_index + is_scalar as usize] + contact.local_index as usize;
        debug_assert!(slot < starts[2 * color_index + is_scalar as usize + 1]);
        Some(slot)
    });
    ColorContacts { lanes, starts }
}

/// Solve with graph coloring. (b3Solve — serial)
///
/// Does not increment `world.step_index` — World::step owns that.
pub fn solve(world: &mut World, context: &StepContext) {
    let awake_body_count = world.solver_sets[AWAKE_SET as usize].body_sims.len();
    if awake_body_count == 0 {
        return;
    }

    world.body_move_events.resize(
        awake_body_count,
        BodyMoveEvent {
            user_data: 0,
            transform: WORLD_TRANSFORM_IDENTITY,
            body_id: BodyId::default(),
            fell_asleep: false,
        },
    );

    let contact_id_capacity = world.contact_id_pool.id_capacity();
    let joint_id_capacity = world.joint_id_pool.id_capacity();
    {
        let task_context = &mut world.task_contexts[0];
        task_context
            .hit_event_bit_set
            .set_bit_count_and_clear(contact_id_capacity as u32);
        task_context.has_hit_events = false;
        task_context
            .joint_state_bit_set
            .set_bit_count_and_clear(joint_id_capacity as u32);
    }

    // Split an awake island. This modifies:
    // - world island array and solver set
    // - island indices on bodies, contacts, and joints
    // C runs this as a task in parallel with the constraint solve (it cannot
    // overlap FinalizeBodies); the serial port runs it here, at the point
    // where C enqueues it. (b3SplitIslandTask)
    if world.split_island_id != NULL_INDEX {
        split_island(world, world.split_island_id);
        world.split_island_id = NULL_INDEX;
    }

    let constraint_ticks = crate::timer::get_ticks();
    let mut ticks = constraint_ticks;

    // Prepare joints for every color (incl. overflow) before contacts. Each joint prepares
    // independently, so all colors' joints run as one parallel pass. mem::take avoids
    // &World + &mut joint_sims from the same World.
    {
        let mut joint_sims: Vec<Vec<JointSim>> = world
            .constraint_graph
            .colors
            .iter_mut()
            .map(|color| std::mem::take(&mut color.joint_sims))
            .collect();
        {
            let world_ref: &World = world;
            let mut all: Vec<&mut JointSim> = joint_sims.iter_mut().flatten().collect();
            crate::parallel::map_reduce_blocks(
                world_ref.thread_pool.as_deref(),
                &mut all,
                16,
                |block| {
                    for joint in block.iter_mut() {
                        prepare_joint(world_ref, joint, context);
                    }
                },
                |_, _| (),
            );
        }
        for (color, sims) in world.constraint_graph.colors.iter_mut().zip(joint_sims) {
            color.joint_sims = sims;
        }
    }

    // The solver works on a copy of the awake body states that worker threads can share.
    // See [`SolverStates`].
    let mut states = std::mem::take(&mut world.solver_states);
    states.load_from(
        world.thread_pool.as_deref(),
        &world.solver_sets[AWAKE_SET as usize].body_states,
    );

    // Prepare contact constraints for every color: wide constraints for the convex contacts
    // and scalar constraints for the mesh contacts. The overflow color only has mesh contacts.
    // (b3PrepareContacts_Convex / b3PrepareContacts_Mesh / b3PrepareContacts_Overflow)
    let mut mesh_constraints = std::mem::take(&mut world.mesh_constraints);
    let mut wide_constraints = std::mem::take(&mut world.wide_constraints);
    mesh_constraints.resize_with(GRAPH_COLOR_COUNT as usize, Vec::new);
    wide_constraints.resize_with(GRAPH_COLOR_COUNT as usize, Vec::new);
    let pool = world.thread_pool.clone();

    // The graph's contacts stay borrowed from prepare to store.
    let mut lanes = color_contacts(
        pool.as_deref(),
        &mut world.contacts,
        &world.constraint_graph,
    );
    // Size the constraint arrays for the prepare stage, which overwrites every field, so
    // last step's constraints are reused.
    let mut lanes_by_color = lanes.split();
    for ((wide, mesh), (convex_lanes, scalar_lanes)) in wide_constraints
        .iter_mut()
        .zip(mesh_constraints.iter_mut())
        .zip(lanes_by_color.iter())
    {
        let wide_count = convex_lanes.len().div_ceil(SIMD_WIDTH);
        wide.truncate(wide_count);
        wide.resize_with(wide_count, ContactConstraintWide::zeroed);
        mesh.truncate(scalar_lanes.len());
        mesh.resize_with(scalar_lanes.len(), ContactConstraint::default);
    }
    debug_assert!(wide_constraints[OVERFLOW_INDEX as usize].is_empty());

    world.profile.prepare_constraints += crate::timer::get_milliseconds_and_reset(&mut ticks);

    // The joints leave the graph for the substeps.
    let mut joints: Vec<Vec<JointSim>> = world
        .constraint_graph
        .colors
        .iter_mut()
        .map(|color| std::mem::take(&mut color.joint_sims))
        .collect();

    // Contact prepare, substeps, restitution, and store. (b3SolverTask)
    let out = {
        let (color_joints, overflow_joints) = joints.split_at_mut(OVERFLOW_INDEX as usize);
        let (color_mesh, overflow_mesh) = mesh_constraints.split_at_mut(OVERFLOW_INDEX as usize);
        let (_, overflow_lanes) = lanes_by_color.pop().expect("overflow color");
        debug_assert!(lanes_by_color.len() == OVERFLOW_INDEX as usize);
        let colors = color_joints
            .iter_mut()
            .zip(wide_constraints.iter_mut())
            .zip(color_mesh.iter_mut())
            .zip(lanes_by_color)
            .map(
                |(((joints, wide), mesh), (convex_lanes, scalar_lanes))| ColorConstraints {
                    joints,
                    wide,
                    convex_lanes,
                    mesh,
                    scalar_lanes,
                },
            )
            .collect();
        let solver_world = super::stages::SolverWorld {
            bodies: &world.bodies,
            solver_sets: &world.solver_sets,
            joint_state_bit_set: &world.task_contexts[0].joint_state_bit_set,
            gravity: world.gravity,
            pool: pool.as_deref(),
            restitution_threshold: world.restitution_threshold,
            enable_restitution_propagation: world.enable_restitution_propagation,
            restitution_iterations: world.restitution_iterations,
            neg_hit_threshold: -world.hit_event_threshold,
        };
        // A pending tree rebuild runs on one worker alongside the stages, which go on with
        // the rest. Finalize needs the trees.
        let broad_phase = &mut world.broad_phase;
        let (out, ()) = crate::parallel::join(
            pool.as_deref(),
            || {
                super::stages::solve_substeps(
                    &solver_world,
                    &states,
                    context,
                    super::stages::OverflowConstraints {
                        joints: &mut overflow_joints[0],
                        mesh: &mut overflow_mesh[0],
                        lanes: overflow_lanes,
                    },
                    colors,
                )
            },
            || broad_phase.finish_tree_update(),
        );
        out
    };
    drop(lanes);
    out.times.add_to(&mut world.profile);
    ticks = crate::timer::get_ticks();
    let flagged_joints = out.flagged_joints;

    for (color, joint_sims) in world.constraint_graph.colors.iter_mut().zip(joints) {
        color.joint_sims = joint_sims;
    }
    for &joint_id in &flagged_joints {
        world.task_contexts[0]
            .joint_state_bit_set
            .set_bit(joint_id as u32);
    }

    states.store_to(
        world.thread_pool.as_deref(),
        &mut world.solver_sets[AWAKE_SET as usize].body_states,
    );
    world.solver_states = states;

    world.profile.restitution += crate::timer::get_milliseconds_and_reset(&mut ticks);

    // Flag the hit events the store stage found.
    {
        let task_context = &mut world.task_contexts[0];
        for &contact_id in &out.hits {
            task_context.hit_event_bit_set.set_bit(contact_id as u32);
        }
        task_context.has_hit_events |= !out.hits.is_empty();
    }
    world.mesh_constraints = mesh_constraints;
    world.wide_constraints = wide_constraints;

    world.profile.constraints = crate::timer::get_milliseconds(constraint_ticks);

    // Finalize bodies (CCD for non-bullets; bullets queued)
    let mut bullet_bodies: Vec<i32> = Vec::with_capacity(awake_body_count);
    {
        let awake_island_count = world.solver_sets[AWAKE_SET as usize].island_sims.len();
        let task_context = &mut world.task_contexts[0];
        task_context.sensor_hits.clear();
        task_context
            .awake_island_bit_set
            .set_bit_count_and_clear(awake_island_count as u32);
        task_context.split_island_id = NULL_INDEX;
        task_context.split_sleep_time = 0.0;
    }

    // Finish the user tree task that was queued earlier in the time step. This must be
    // complete before touching the broad-phase. The serial port already ran the tree task
    // inline in update_broad_phase_pairs, so only the validation remains.
    world.broad_phase.validate_no_moved();

    // Finalize bodies. Must happen after the constraint solver and after island splitting.
    let transform_ticks = crate::timer::get_ticks();
    finalize_bodies(world, context, &mut bullet_bodies);
    world.profile.transforms = crate::timer::get_milliseconds(transform_ticks);

    // Report joint events (C block between finalize and hit events).
    {
        debug_assert!(world.joint_events.is_empty());
        let world_id = world.world_id;
        let word_count = world.task_contexts[0].joint_state_bit_set.block_count();
        for k in 0..word_count {
            let mut word = world.task_contexts[0].joint_state_bit_set.block(k);
            while word != 0 {
                let ctz = word.trailing_zeros();
                let joint_id = (64 * k + ctz) as i32;

                debug_assert!((joint_id as usize) < world.joints.len());

                let joint = &world.joints[joint_id as usize];
                debug_assert!(joint.set_index == AWAKE_SET);

                world.joint_events.push(JointEvent {
                    joint_id: JointId {
                        index1: joint_id + 1,
                        world0: world_id,
                        generation: joint.generation,
                    },
                    user_data: joint.user_data,
                });

                word &= word - 1;
            }
        }
    }

    // Report hit events flagged during store impulses. C runs this after the
    // joint events pass.
    {
        use crate::events::ContactHitEvent;
        use crate::id::{ContactId, ShapeId};
        use crate::math_functions::{lerp, lerp_position, offset_pos, VEC3_ZERO};

        debug_assert!(world.contact_hit_events.is_empty());

        // Fast path: if no worker flagged any hit-event candidates during
        // b2StoreImpulsesTask, skip entirely.
        if world.task_contexts[0].has_hit_events {
            let threshold = world.hit_event_threshold;
            let world_id = world.world_id;

            let word_count = world.task_contexts[0].hit_event_bit_set.block_count();
            for k in 0..word_count {
                let mut word = world.task_contexts[0].hit_event_bit_set.block(k);
                while word != 0 {
                    let ctz = word.trailing_zeros();
                    let contact_id = (64 * k + ctz) as i32;

                    let contact = &world.contacts[contact_id as usize];
                    debug_assert!(
                        contact.set_index == AWAKE_SET && contact.color_index != NULL_INDEX
                    );

                    let shape_a = &world.shapes[contact.shape_id_a as usize];
                    let shape_b = &world.shapes[contact.shape_id_b as usize];
                    let body_a = &world.bodies[shape_a.body_id as usize];
                    let body_b = &world.bodies[shape_b.body_id as usize];
                    // (b3GetBodySim)
                    let sim_a = &world.solver_sets[body_a.set_index as usize].body_sims
                        [body_a.local_index as usize];
                    let sim_b = &world.solver_sets[body_b.set_index as usize].body_sims
                        [body_b.local_index as usize];
                    let mid_center = lerp_position(sim_a.center, sim_b.center, 0.5);

                    let mut approach_speed = threshold;
                    let mut point = mid_center;
                    let mut normal = VEC3_ZERO;
                    let mut found = false;
                    let mut triangle_index = 0;
                    for manifold in &contact.manifolds {
                        for p in 0..manifold.point_count as usize {
                            let mp = &manifold.points[p];
                            let mp_approach_speed = -mp.normal_velocity;

                            // Need to check total impulse because the point may be speculative and not colliding
                            if mp_approach_speed > approach_speed && mp.total_normal_impulse > 0.0 {
                                approach_speed = mp_approach_speed;
                                point = offset_pos(mid_center, lerp(mp.anchor_a, mp.anchor_b, 0.5));
                                normal = manifold.normal;
                                triangle_index = mp.triangle_index;
                                found = true;
                            }
                        }
                    }

                    if found {
                        let event = ContactHitEvent {
                            shape_id_a: ShapeId {
                                index1: shape_a.id + 1,
                                world0: world_id,
                                generation: shape_a.generation,
                            },
                            shape_id_b: ShapeId {
                                index1: shape_b.id + 1,
                                world0: world_id,
                                generation: shape_b.generation,
                            },
                            contact_id: ContactId {
                                index1: contact.contact_id + 1,
                                world0: world_id,
                                padding: 0,
                                generation: contact.generation,
                            },
                            point,
                            normal,
                            approach_speed,
                            // shapeB is never a compound today (asserted in b3CreateContact), so the
                            // childIndex argument is irrelevant for it. shapeA carries the compound.
                            user_material_id_a: shape_a
                                .get_shape_user_material_id(contact.child_index, triangle_index),
                            user_material_id_b: shape_b
                                .get_shape_user_material_id(0, triangle_index),
                        };
                        world.contact_hit_events.push(event);
                    }

                    word &= word - 1;
                }
            }
        }
    }

    // Refit BVH
    {
        let refit_ticks = crate::timer::get_ticks();
        world.broad_phase.trees[BodyType::Kinematic as usize].refit();
        world.broad_phase.trees[BodyType::Dynamic as usize].refit();
        world.profile.refit = crate::timer::get_milliseconds(refit_ticks);

        world.broad_phase.validate();
    }

    // Fast bullet bodies. Note: a bullet body may be moving slow
    if !bullet_bodies.is_empty() {
        for &sim_index in &bullet_bodies {
            solve_continuous(world, sim_index, context.dt);
        }

        // Serially enlarge broad-phase proxies for bullet shapes
        let dynamic_tree = BodyType::Dynamic as usize;

        // This loop has non-deterministic order but it shouldn't affect the result
        for &sim_index in &bullet_bodies {
            let bullet_sim =
                &mut world.solver_sets[AWAKE_SET as usize].body_sims[sim_index as usize];
            if (bullet_sim.flags & body_flags::ENLARGE_BULLET_BOUNDS) == 0 {
                continue;
            }

            // Clear flag
            bullet_sim.flags &= !body_flags::ENLARGE_BULLET_BOUNDS;

            let body_id = bullet_sim.body_id;
            debug_assert!(0 <= body_id && (body_id as usize) < world.bodies.len());

            let mut shape_id = world.bodies[body_id as usize].head_shape_id;
            while shape_id != NULL_INDEX {
                let proxy_key = world.shapes[shape_id as usize].proxy_key;
                let proxy_id_ = proxy_id(proxy_key);
                debug_assert!(proxy_type(proxy_key) == BodyType::Dynamic);

                let tree_aabb = world.broad_phase.trees[dynamic_tree].aabb(proxy_id_);
                let fat_aabb = world.fat_aabbs[shape_id as usize];

                if !aabb_contains(tree_aabb, fat_aabb) {
                    world.broad_phase.trees[dynamic_tree].enlarge_proxy(proxy_id_, fat_aabb);
                }

                shape_id = world.shapes[shape_id as usize].next_shape_id;
            }
        }
    }

    // Report sensor hits. This may include bullet sensor hits.
    {
        let mut drained: Vec<(i32, i32, u16)> = Vec::new();
        for task_context in &world.task_contexts {
            for hit in &task_context.sensor_hits {
                let sensor_index = world.shapes[hit.sensor_id as usize].sensor_index;
                let generation = world.shapes[hit.visitor_id as usize].generation;
                drained.push((sensor_index, hit.visitor_id, generation));
            }
        }
        for (sensor_index, visitor_id, generation) in drained {
            world.sensors[sensor_index as usize]
                .hits
                .push(crate::sensor::Visitor {
                    shape_id: visitor_id,
                    generation,
                });
        }
    }

    // Island sleeping
    // This must be done last because putting islands to sleep invalidates the enlarged body bits.
    if world.enable_sleep {
        // Collect split island candidate for the next time step. No need to split if sleeping is disabled.
        debug_assert!(world.split_island_id == NULL_INDEX);
        let mut split_sleep_timer = 0.0f32;
        {
            let task_context = &world.task_contexts[0];
            if task_context.split_island_id != NULL_INDEX
                && task_context.split_sleep_time >= split_sleep_timer
            {
                debug_assert!(task_context.split_sleep_time > 0.0);

                // Tie breaking for determinism. Largest island id wins. C needs this
                // due to work stealing across workers; kept for the serial port so a
                // multi-worker build later cannot change the outcome.
                let tied_but_smaller = task_context.split_sleep_time == split_sleep_timer
                    && task_context.split_island_id < world.split_island_id;
                if !tied_but_smaller {
                    world.split_island_id = task_context.split_island_id;
                    split_sleep_timer = task_context.split_sleep_time;
                }
            }
        }
        let _ = split_sleep_timer;

        // Need to process in reverse because this moves islands to sleeping solver sets.
        let count = world.solver_sets[AWAKE_SET as usize].island_sims.len();
        for island_index in (0..count).rev() {
            if world.task_contexts[0]
                .awake_island_bit_set
                .get_bit(island_index as u32)
            {
                // this island is still awake
                continue;
            }

            let island_id =
                world.solver_sets[AWAKE_SET as usize].island_sims[island_index].island_id;

            try_sleep_island(world, island_id);
        }

        world.validate_solver_sets();
    }
}
