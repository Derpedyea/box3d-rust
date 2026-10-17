//! Body integration and finalization from solver.c.
//! Serial loops over the whole awake body range (one worker, one block).
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use super::continuous::{continuous_apply, continuous_compute};
use super::states::SolverStates;
use super::StepContext;
use crate::body::body_flags;
use crate::body::{Body, BodySim, BodyState};
use crate::constants::{speculative_distance, MAX_ROTATION, TIME_TO_SLEEP};
use crate::core::NULL_INDEX;
use crate::events::BodyMoveEvent;
use crate::id::BodyId;
use crate::math_functions::Aabb;
use crate::math_functions::{
    aabb_contains, aabb_inflate, abs, blend2, dot, integrate_rotation, inv_rotate_vector,
    is_valid_vec3, length, make_matrix_from_quat, max_float, modified_cross, mul_add, mul_mm,
    mul_mv, mul_quat, mul_sv, neg, normalize_quat, offset_pos, rotate_vector, solve3, sub,
    transpose, Matrix3, Vec3, QUAT_IDENTITY, VEC3_ZERO,
};
use crate::solver_set::AWAKE_SET;
use crate::types::BodyType;
use crate::world::World;

/// Integrate velocities, apply damping, and gyroscopic torque.
/// (b3IntegrateVelocitiesTask)
pub(super) fn integrate_velocities(
    sims: &[BodySim],
    gravity: Vec3,
    states: &SolverStates,
    context: &StepContext,
    range: std::ops::Range<usize>,
) {
    let h = context.h;

    for i in range {
        let sim = &sims[i];
        let mut state = states.get(i);

        let mut v = state.linear_velocity;
        let mut w = state.angular_velocity;

        let linear_damping = 1.0 / (1.0 + h * sim.linear_damping);
        let angular_damping = 1.0 / (1.0 + h * sim.angular_damping);

        let gravity_scale = if sim.inv_mass > 0.0 {
            sim.gravity_scale
        } else {
            0.0
        };

        let linear_velocity_delta = blend2(h * sim.inv_mass, sim.force, h * gravity_scale, gravity);
        v = mul_add(linear_velocity_delta, linear_damping, v);

        let angular_velocity_delta = mul_sv(h, mul_mv(sim.inv_inertia_world, sim.torque));
        w = mul_add(angular_velocity_delta, angular_damping, w);

        // Gyroscopic torque by solving this nonlinear equation using Newton-Raphson.
        // I * (w2 - w1) + h * cross(w2, I * w2) = 0
        // This is all done in local coordinates where the Jacobian is easier to compute.
        // This improves the simulation of long skinny bodies.
        if sim.flags & body_flags::ENABLE_GYROSCOPIC_TORQUE != 0 {
            // Get current rotation.
            let q0 = sim.transform.q;
            let q = mul_quat(state.delta_rotation, q0);

            // Compute local angular velocity
            let omega1 = inv_rotate_vector(q, w);
            let mut omega2 = omega1;

            // Symmetric inertia tensor: 6 unique entries (column-major)
            let i00 = sim.inertia.cx.x;
            let i01 = sim.inertia.cy.x;
            let i02 = sim.inertia.cz.x;
            let i11 = sim.inertia.cy.y;
            let i12 = sim.inertia.cz.y;
            let i22 = sim.inertia.cz.z;

            for _gyro_iteration in 0..crate::constants::GYROSCOPIC_ITERATIONS {
                let w1 = omega2.x;
                let w2 = omega2.y;
                let w3 = omega2.z;

                let iw1 = i00 * w1 + i01 * w2 + i02 * w3;
                let iw2 = i01 * w1 + i11 * w2 + i12 * w3;
                let iw3 = i02 * w1 + i12 * w2 + i22 * w3;

                let dw = sub(omega2, omega1);
                let b = Vec3 {
                    x: i00 * dw.x + i01 * dw.y + i02 * dw.z + h * (w2 * iw3 - w3 * iw2),
                    y: i01 * dw.x + i11 * dw.y + i12 * dw.z + h * (w3 * iw1 - w1 * iw3),
                    z: i02 * dw.x + i12 * dw.y + i22 * dw.z + h * (w1 * iw2 - w2 * iw1),
                };

                let j = Matrix3 {
                    cx: Vec3 {
                        x: i00 + h * (w2 * i02 - w3 * i01),
                        y: i01 + h * (w3 * i00 - w1 * i02 - iw3),
                        z: i02 + h * (w1 * i01 - w2 * i00 + iw2),
                    },
                    cy: Vec3 {
                        x: i01 + h * (w2 * i12 - w3 * i11 + iw3),
                        y: i11 + h * (w3 * i01 - w1 * i12),
                        z: i12 + h * (w1 * i11 - w2 * i01 - iw1),
                    },
                    cz: Vec3 {
                        x: i02 + h * (w2 * i22 - w3 * i12 - iw2),
                        y: i12 + h * (w3 * i02 - w1 * i22 + iw1),
                        z: i22 + h * (w1 * i12 - w2 * i02),
                    },
                };

                omega2 = sub(omega2, solve3(j, b));
            }

            w = rotate_vector(q, omega2);
        }

        state.linear_velocity = v;
        state.angular_velocity = w;
        states.set(i, state);
    }
}

/// (b3IntegratePositionsTask)
pub(super) fn integrate_positions(
    states: &SolverStates,
    context: &StepContext,
    range: std::ops::Range<usize>,
) {
    let h = context.h;
    let max_linear_speed = context.max_linear_velocity;
    let max_angular_speed = MAX_ROTATION * context.inv_dt;
    let max_linear_speed_squared = max_linear_speed * max_linear_speed;
    let max_angular_speed_squared = max_angular_speed * max_angular_speed;

    for i in range {
        let mut state = states.get(i);
        let mut v = state.linear_velocity;
        let mut w = state.angular_velocity;

        if (state.flags & body_flags::LOCK_LINEAR_X) != 0 {
            v.x = 0.0;
        }
        if (state.flags & body_flags::LOCK_LINEAR_Y) != 0 {
            v.y = 0.0;
        }
        if (state.flags & body_flags::LOCK_LINEAR_Z) != 0 {
            v.z = 0.0;
        }
        if (state.flags & body_flags::LOCK_ANGULAR_X) != 0 {
            w.x = 0.0;
        }
        if (state.flags & body_flags::LOCK_ANGULAR_Y) != 0 {
            w.y = 0.0;
        }
        if (state.flags & body_flags::LOCK_ANGULAR_Z) != 0 {
            w.z = 0.0;
        }

        if dot(v, v) > max_linear_speed_squared {
            let ratio = max_linear_speed / length(v);
            v = mul_sv(ratio, v);
            state.flags |= body_flags::IS_SPEED_CAPPED;
        }

        if dot(w, w) > max_angular_speed_squared
            && (state.flags & body_flags::ALLOW_FAST_ROTATION) == 0
        {
            let ratio = max_angular_speed / length(w);
            w = mul_sv(ratio, w);
            state.flags |= body_flags::IS_SPEED_CAPPED;
        }

        state.linear_velocity = v;
        state.angular_velocity = w;
        state.delta_position = mul_add(state.delta_position, h, v);
        state.delta_rotation = integrate_rotation(state.delta_rotation, mul_sv(h, w));
        states.set(i, state);
    }
}

/// What finalizing a body reads besides its own data.
struct FinalizeShared<'a> {
    islands: &'a [crate::island::Island],
    enable_sleep: bool,
    enable_continuous: bool,
    time_step: f32,
    inv_time_step: f32,
    world_id: u16,
}

/// What finalizing a block of bodies produces for the serial tail: C does these through
/// atomics and per-worker bit sets.
#[derive(Default)]
struct FinalizeOut {
    awake_islands: Vec<i32>,
    /// The (sleep time, island id) split candidate, the lexicographic max.
    split: Option<(f32, i32)>,
    /// Fast non-bullet bodies, by sim index, whose continuous collision and remaining
    /// finalization run after the parallel pass.
    fast: Vec<i32>,
    bullets: Vec<i32>,
}

impl FinalizeOut {
    fn merge(mut self, mut other: FinalizeOut) -> FinalizeOut {
        self.awake_islands.append(&mut other.awake_islands);
        self.split = max_split(self.split, other.split);
        self.fast.append(&mut other.fast);
        self.bullets.append(&mut other.bullets);
        self
    }
}

/// Larger sleep time wins, then larger island id. (the b3TaskContext split candidate)
fn max_split(a: Option<(f32, i32)>, b: Option<(f32, i32)>) -> Option<(f32, i32)> {
    match (a, b) {
        (Some(x), Some(y)) => {
            if y.0 > x.0 || (y.0 == x.0 && y.1 > x.1) {
                Some(y)
            } else {
                Some(x)
            }
        }
        (x, None) => x,
        (None, y) => y,
    }
}

/// The first part of b3FinalizeBodiesTask for one body, up to continuous collision. Returns
/// true if the body is fast and not a bullet, so continuous collision must run before
/// [`finalize_body_tail`].
#[allow(clippy::too_many_arguments)]
fn finalize_body_head(
    sim_index: usize,
    sim: &mut BodySim,
    state: &mut BodyState,
    move_event: &mut BodyMoveEvent,
    body: &mut Body,
    shared: &FinalizeShared<'_>,
    out: &mut FinalizeOut,
) -> bool {
    let v = state.linear_velocity;
    let w = state.angular_velocity;
    let local_omega = inv_rotate_vector(sim.transform.q, w);
    let local_delta_rotation = inv_rotate_vector(sim.transform.q, state.delta_rotation.v);

    debug_assert!(is_valid_vec3(v));
    debug_assert!(is_valid_vec3(w));

    sim.center = offset_pos(sim.center, state.delta_position);
    sim.transform.q = normalize_quat(mul_quat(state.delta_rotation, sim.transform.q));

    let velocity_arc = modified_cross(abs(local_omega), sim.max_extent);
    let max_velocity = length(v) + length(velocity_arc);

    let rotation_arc = modified_cross(abs(local_delta_rotation), sim.max_extent);
    let max_delta_position = length(state.delta_position) + 2.0 * length(rotation_arc);

    let position_sleep_factor = 0.5;
    let sleep_velocity = max_float(
        max_velocity,
        position_sleep_factor * shared.inv_time_step * max_delta_position,
    );

    state.delta_position = VEC3_ZERO;
    state.delta_rotation = QUAT_IDENTITY;

    sim.transform.p = offset_pos(
        sim.center,
        neg(rotate_vector(sim.transform.q, sim.local_center)),
    );

    let body_id = sim.body_id;
    body.body_move_index = sim_index as i32;
    body.sleep_velocity = sleep_velocity;
    *move_event = BodyMoveEvent {
        user_data: body.user_data,
        transform: sim.transform,
        body_id: BodyId {
            index1: body_id + 1,
            world0: shared.world_id,
            generation: body.generation,
        },
        fell_asleep: false,
    };

    sim.force = VEC3_ZERO;
    sim.torque = VEC3_ZERO;

    debug_assert!((body.flags & body_flags::DIRTY_MASS) == 0);

    // Clear the transient flags (fast, speed capped, had TOI). These flags are conditionally set
    // as part of the code below.
    body.flags &= !body_flags::BODY_TRANSIENT_FLAGS;
    sim.flags &= !(body_flags::IS_FAST | body_flags::BODY_TRANSIENT_FLAGS);

    // The body state flag knows about speed capping (used for debug draw).
    body.flags |= state.flags & body_flags::IS_SPEED_CAPPED;
    state.flags &= !body_flags::BODY_TRANSIENT_FLAGS;

    if !shared.enable_sleep
        || (body.flags & body_flags::ENABLE_SLEEP) == 0
        || sleep_velocity > body.sleep_threshold
    {
        body.sleep_time = 0.0;

        let max_motion = max_float(max_delta_position, max_velocity * shared.time_step);
        if body.type_ == BodyType::Dynamic
            && shared.enable_continuous
            && max_motion > body.safety_factor * sim.min_extent
        {
            // This flag is used for debug draw and contact recycling.
            sim.flags |= body_flags::IS_FAST;

            if (sim.flags & body_flags::IS_BULLET) != 0 {
                out.bullets.push(sim_index as i32);
            } else {
                out.fast.push(sim_index as i32);
                return true;
            }
        } else {
            sim.center0 = sim.center;
            sim.rotation0 = sim.transform.q;
        }
    } else {
        sim.center0 = sim.center;
        sim.rotation0 = sim.transform.q;
        body.sleep_time += shared.time_step;
    }
    false
}

/// The rest of b3FinalizeBodiesTask for one body.
fn finalize_body_tail(
    sim: &mut BodySim,
    body: &Body,
    shared: &FinalizeShared<'_>,
    out: &mut FinalizeOut,
) {
    let rotation_matrix = make_matrix_from_quat(sim.transform.q);
    sim.inv_inertia_world = mul_mm(
        mul_mm(rotation_matrix, sim.inv_inertia_local),
        transpose(rotation_matrix),
    );

    let sleep_time = body.sleep_time;
    let island_id = body.island_id;
    if island_id != NULL_INDEX {
        if sleep_time < TIME_TO_SLEEP {
            out.awake_islands
                .push(shared.islands[island_id as usize].local_index);
        } else if shared.islands[island_id as usize].constraint_remove_count > 0 {
            out.split = max_split(out.split, Some((sleep_time, island_id)));
        }
    }

    // The shape AABBs update in [`update_shape_aabbs`].
}

/// Update the AABBs of the shapes of awake bodies that aren't fast, in parallel over the
/// shapes, and return the proxies that left their fat AABB.
/// For fast non-bullet bodies the AABB has already been updated in b3SolveContinuous
/// For fast bullet bodies the AABB will be updated at a later stage
/// (the shape loop of b3FinalizeBodiesTask)
fn update_shape_aabbs(world: &mut World) -> Vec<(i32, Aabb)> {
    let speculative = speculative_distance();
    let World {
        shapes,
        fat_aabbs,
        bodies,
        solver_sets,
        thread_pool,
        ..
    } = &mut *world;
    debug_assert!(shapes.len() <= fat_aabbs.len());
    let sims = &solver_sets[AWAKE_SET as usize].body_sims;
    let bodies = &*bodies;
    let mut items: Vec<(&mut crate::shape::Shape, &mut Aabb)> =
        shapes.iter_mut().zip(fat_aabbs.iter_mut()).collect();
    crate::parallel::map_reduce_blocks(
        thread_pool.as_deref(),
        &mut items,
        512,
        |block| {
            let mut moved = Vec::new();
            for (shape, fat_aabb) in block.iter_mut() {
                if shape.id == NULL_INDEX {
                    continue;
                }
                let body = &bodies[shape.body_id as usize];
                if body.set_index != AWAKE_SET {
                    continue;
                }
                let sim = &sims[body.local_index as usize];
                if (sim.flags & body_flags::IS_FAST) != 0 {
                    continue;
                }
                let aabb = crate::shape::compute_fat_shape_aabb(shape, sim.transform, speculative);
                shape.aabb = aabb;
                if !aabb_contains(**fat_aabb, aabb) {
                    **fat_aabb = aabb_inflate(aabb, shape.aabb_margin);
                    moved.push((shape.proxy_key, **fat_aabb));
                }
            }
            moved
        },
        |mut a, mut b| {
            a.append(&mut b);
            a
        },
    )
}

/// Finalize awake bodies after the constraint solve, including fast-body / bullet
/// detection for continuous collision. (b3FinalizeBodiesTask)
///
/// Bodies finalize in parallel. The writes C makes through atomics (moving broad-phase
/// proxies, island bits, the split candidate) and continuous collision for fast non-bullet
/// bodies, which only reads static shapes, run after, in body order.
pub(super) fn finalize_bodies(
    world: &mut World,
    context: &StepContext,
    bullet_bodies: &mut Vec<i32>,
) {
    let awake_body_count = world.solver_sets[AWAKE_SET as usize].body_sims.len();
    debug_assert!(awake_body_count <= world.body_move_events.len());

    let pool = world.thread_pool.clone();
    let enable_sleep = world.enable_sleep;
    let enable_continuous = world.enable_continuous;
    let world_id = world.world_id;
    let out = {
        let World {
            bodies,
            islands,
            solver_sets,
            body_move_events,
            ..
        } = &mut *world;
        let shared = FinalizeShared {
            islands,
            enable_sleep,
            enable_continuous,
            time_step: context.dt,
            inv_time_step: context.inv_dt,
            world_id,
        };

        // The awake bodies, borrowed in sim order.
        let awake =
            crate::parallel::borrow_by_slot(pool.as_deref(), bodies, awake_body_count, |body| {
                (body.set_index == AWAKE_SET && body.id != NULL_INDEX)
                    .then_some(body.local_index as usize)
            });

        let set = &mut solver_sets[AWAKE_SET as usize];
        let mut items: Vec<(
            usize,
            &mut BodySim,
            &mut BodyState,
            &mut BodyMoveEvent,
            &mut Body,
        )> = set
            .body_sims
            .iter_mut()
            .zip(set.body_states.iter_mut())
            .zip(body_move_events.iter_mut())
            .zip(awake)
            .enumerate()
            .map(|(i, (((sim, state), event), body))| (i, sim, state, event, body))
            .collect();

        crate::parallel::map_reduce_blocks(
            pool.as_deref(),
            &mut items,
            256,
            |block| {
                let mut out = FinalizeOut::default();
                for (i, sim, state, event, body) in block.iter_mut() {
                    if !finalize_body_head(*i, sim, state, event, body, &shared, &mut out) {
                        finalize_body_tail(sim, body, &shared, &mut out);
                    }
                }
                out
            },
            FinalizeOut::merge,
        )
    };

    let time_step = context.dt;
    let mut split = out.split;
    let mut awake_islands = out.awake_islands;
    let world_ref: &World = world;
    let results = crate::parallel::map_items(pool.as_deref(), &out.fast, |&sim_index| {
        continuous_compute(world_ref, sim_index)
    });
    for (result, &sim_index) in results.into_iter().zip(&out.fast) {
        continuous_apply(world, result, time_step);

        let shared = FinalizeShared {
            islands: &world.islands,
            enable_sleep: world.enable_sleep,
            enable_continuous: world.enable_continuous,
            time_step,
            inv_time_step: context.inv_dt,
            world_id: world.world_id,
        };
        let mut tail = FinalizeOut::default();
        let mut sim = world.solver_sets[AWAKE_SET as usize].body_sims[sim_index as usize];
        let body = &world.bodies[sim.body_id as usize];
        finalize_body_tail(&mut sim, body, &shared, &mut tail);
        world.solver_sets[AWAKE_SET as usize].body_sims[sim_index as usize] = sim;
        awake_islands.append(&mut tail.awake_islands);
        split = max_split(split, tail.split);
    }
    bullet_bodies.extend_from_slice(&out.bullets);

    // Move the proxies of enlarged shapes. (C marks them with atomics.)
    for (proxy_key, fat_aabb) in update_shape_aabbs(world) {
        world.broad_phase.mark_proxy_moved(proxy_key, fat_aabb);
    }

    let task_context = &mut world.task_contexts[0];
    for &island_index in &awake_islands {
        task_context
            .awake_island_bit_set
            .set_bit(island_index as u32);
    }
    let initial = if task_context.split_island_id == NULL_INDEX {
        None
    } else {
        Some((task_context.split_sleep_time, task_context.split_island_id))
    };
    if let Some((sleep_time, island_id)) = max_split(initial, split) {
        task_context.split_sleep_time = sleep_time;
        task_context.split_island_id = island_id;
    }
}
