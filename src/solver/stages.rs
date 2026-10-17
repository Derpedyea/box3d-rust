//! The solver's substep loop as C runs it: a fixed sequence of stages, each an array of
//! blocks that the workers claim, with the workers spinning between stages instead of being
//! dispatched per stage. Each worker starts claiming at its own home block and sweeps the
//! rest, so a worker mostly gets the same blocks every substep and finds their constraints
//! still in its cache. (b3SolverTask, b3ExecuteMainStage, b3ExecuteStage)
//!
//! Worker 0 (the calling thread) orchestrates: it runs the overflow constraints, which the
//! graph coloring doesn't cover, serially and in order, then publishes each stage and helps
//! finish it. The other workers only claim blocks. Blocks of a stage never share a dynamic
//! body, so the result doesn't depend on which worker runs which block, or how many workers
//! there are. With one worker this is a plain serial loop.
//!
//! Each block's constraints sit behind a `Mutex` so any worker can borrow them mutably;
//! claims are unique, so the locks never contend.
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use super::integrate::{integrate_positions, integrate_velocities};
use super::states::SolverStates;
use super::StepContext;
use crate::constants::{RELAX_ITERATIONS, SOLVER_ITERATIONS};
use crate::contact::Contact;
use crate::contact_solver::wide::{
    apply_restitution_convex, prepare_contacts_convex, push_contacts_convex, solve_contacts_convex,
    store_impulses_convex, warm_start_contacts_convex, ContactConstraintWide,
};
use crate::contact_solver::{
    apply_restitution, prepare_mesh_contacts, push_contacts, solve_contacts, store_impulses,
    warm_start_contacts, ContactConstraint,
};
use crate::joint::{get_joint_reaction_in, solve_joint, warm_start_joint, JointSim};
use crate::timer::{get_milliseconds_and_reset, get_ticks};
use crate::world::Profile;

/// The world data the substeps read. A view, so the contacts can stay borrowed mutably
/// from prepare to store.
pub(super) struct SolverWorld<'a> {
    pub bodies: &'a [crate::body::Body],
    pub solver_sets: &'a [crate::solver_set::SolverSet],
    pub joint_state_bit_set: &'a crate::bitset::BitSet,
    pub gravity: crate::math_functions::Vec3,
    pub pool: Option<&'a crate::parallel::ThreadPool>,
    pub restitution_threshold: f32,
    pub enable_restitution_propagation: bool,
    pub restitution_iterations: i32,
    pub neg_hit_threshold: f32,
}
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Clone, Copy, PartialEq, Eq)]
enum StageKind {
    /// Over the contact blocks of every color.
    PrepareContacts,
    IntegrateVelocities,
    WarmStart,
    Solve,
    IntegratePositions,
    Relax,
    Restitution,
    /// Over the contact blocks of every color.
    StoreImpulses,
}

impl StageKind {
    fn on_bodies(self) -> bool {
        matches!(
            self,
            StageKind::IntegrateVelocities | StageKind::IntegratePositions
        )
    }

    fn on_contacts(self) -> bool {
        matches!(self, StageKind::PrepareContacts | StageKind::StoreImpulses)
    }
}

/// Serial overflow work the orchestrator does before a stage.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Overflow {
    None,
    Prepare,
    WarmStart,
    Solve,
    Relax,
    Restitution,
    Store,
}

#[derive(Clone, Copy)]
struct Stage {
    kind: StageKind,
    /// Index into the active colors for graph stages.
    color: usize,
    overflow: Overflow,
    /// The claim value this stage's blocks hold before it runs: the claim value (stage index
    /// plus one) of the last stage on the same blocks, or 0.
    prev: u32,
}

/// Some constraints of one color, with their contacts.
enum Unit<'a, 'c> {
    Joints(&'a mut [JointSim]),
    /// Four contacts per constraint.
    Wide(&'a mut [ContactConstraintWide], &'a mut [&'c mut Contact]),
    Mesh(&'a mut [ContactConstraint], &'a mut [&'c mut Contact]),
}

/// Per stage completion counter.
struct StageSync {
    done: AtomicUsize,
}

/// (b3ComputeBlockCount): (block size, block count) with at least `min_size` items per
/// block and at most `max_block_count` blocks.
fn block_dim(item_count: usize, min_size: usize, max_block_count: usize) -> (usize, usize) {
    if item_count == 0 {
        return (min_size, 0);
    }
    let size = if item_count <= min_size * max_block_count {
        min_size
    } else {
        item_count.div_ceil(max_block_count)
    };
    (size, item_count.div_ceil(size))
}

/// Everything a worker needs, shared by reference.
struct Shared<'a, 'u, 'c> {
    world: &'a SolverWorld<'a>,
    states: &'a SolverStates,
    context: &'a StepContext,
    stages: &'a [Stage],
    syncs: &'a [StageSync],
    units: &'a [Vec<Mutex<Unit<'u, 'c>>>],
    /// The contact units of every color, as (color, unit).
    contact_units: &'a [(usize, usize)],
    /// Per block claim values: of each color's units, of the body blocks, and of the
    /// contact units.
    unit_claims: &'a [Vec<AtomicU32>],
    body_claims: &'a [AtomicU32],
    contact_claims: &'a [AtomicU32],
    worker_count: usize,
    /// The orchestrator's pool thread index. Worker indices count from it.
    main_thread: usize,
    body_count: usize,
    body_block: usize,
    body_block_count: usize,
    restitution_threshold: f32,
    propagate: bool,
    /// Stages published so far. Stage `published - 1` is the current one.
    published: AtomicUsize,
    finished: AtomicBool,
    flagged_joints: Mutex<Vec<i32>>,
    any_restitution: AtomicBool,
    hits: Mutex<Vec<i32>>,
}

impl Shared<'_, '_, '_> {
    fn block_count(&self, stage: &Stage) -> usize {
        if stage.kind.on_bodies() {
            self.body_block_count
        } else if stage.kind.on_contacts() {
            self.contact_units.len()
        } else {
            self.units[stage.color].len()
        }
    }

    fn claims(&self, stage: &Stage) -> &[AtomicU32] {
        if stage.kind.on_bodies() {
            self.body_claims
        } else if stage.kind.on_contacts() {
            self.contact_claims
        } else {
            &self.unit_claims[stage.color]
        }
    }

    /// This thread's worker index: 0 for the orchestrator.
    fn worker_index(&self) -> usize {
        let n = self.worker_count;
        (crate::parallel::thread_index() + n - self.main_thread % n) % n
    }

    /// Claim and run blocks of `stage_index`: sweep them all once from this worker's home
    /// block, taking the unclaimed ones. (b3ExecuteStage)
    fn work(&self, stage_index: usize, worker_index: usize) {
        let stage = &self.stages[stage_index];
        let count = self.block_count(stage);
        // (GetWorkerStartIndex)
        let start = if count <= self.worker_count {
            if worker_index >= count {
                return;
            }
            worker_index
        } else {
            let per_worker = count / self.worker_count;
            let remainder = count - per_worker * self.worker_count;
            per_worker * worker_index + remainder.min(worker_index)
        };
        let claims = self.claims(stage);
        let claim = stage_index as u32 + 1;
        let mut completed = 0;
        let mut block = start;
        for _ in 0..count {
            let c = &claims[block];
            if c.load(Ordering::Relaxed) == stage.prev
                && c.compare_exchange(stage.prev, claim, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                self.run_block(stage, block);
                completed += 1;
            }
            block += 1;
            if block == count {
                block = 0;
            }
        }
        if completed > 0 {
            self.syncs[stage_index]
                .done
                .fetch_add(completed, Ordering::Release);
        }
    }

    fn run_block(&self, stage: &Stage, block: usize) {
        let states = self.states;
        let context = self.context;
        match stage.kind {
            StageKind::IntegrateVelocities | StageKind::IntegratePositions => {
                let start = block * self.body_block;
                let range = start..(start + self.body_block).min(self.body_count);
                if stage.kind == StageKind::IntegrateVelocities {
                    let sims =
                        &self.world.solver_sets[crate::solver_set::AWAKE_SET as usize].body_sims;
                    integrate_velocities(sims, self.world.gravity, states, context, range);
                } else {
                    integrate_positions(states, context, range);
                }
            }
            kind => {
                let (color, block) = if kind.on_contacts() {
                    self.contact_units[block]
                } else {
                    (stage.color, block)
                };
                let mut unit = self.units[color][block]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let sims = &self.world.solver_sets[crate::solver_set::AWAKE_SET as usize].body_sims;
                match (&mut *unit, kind) {
                    (Unit::Joints(joints), StageKind::WarmStart) => {
                        for joint in joints.iter_mut() {
                            warm_start_joint(joint, states);
                        }
                    }
                    (Unit::Joints(joints), StageKind::Solve | StageKind::Relax) => {
                        let use_bias = kind == StageKind::Solve;
                        let mut flagged = Vec::new();
                        for joint in joints.iter_mut() {
                            solve_joint(joint, context, states, use_bias);
                            if use_bias && joint_reaction_exceeded(self.world, joint, context.inv_h)
                            {
                                flagged.push(joint.joint_id);
                            }
                        }
                        if !flagged.is_empty() {
                            self.flagged_joints
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .append(&mut flagged);
                        }
                    }
                    (Unit::Joints(_), _) => {}
                    (Unit::Wide(c, lanes), StageKind::PrepareContacts) => {
                        let ews = context.enable_warm_starting;
                        if prepare_contacts_convex(c, lanes, sims, states, ews) {
                            self.any_restitution.store(true, Ordering::Relaxed);
                        }
                    }
                    (Unit::Wide(c, _), StageKind::WarmStart) => {
                        warm_start_contacts_convex(c, states)
                    }
                    (Unit::Wide(c, _), StageKind::Solve) => {
                        push_contacts_convex(c, states, context)
                    }
                    (Unit::Wide(c, _), StageKind::Relax) => {
                        solve_contacts_convex(c, states, context)
                    }
                    (Unit::Wide(c, lanes), StageKind::StoreImpulses) => {
                        let mut hits = Vec::new();
                        store_impulses_convex(c, lanes, self.world.neg_hit_threshold, &mut hits);
                        self.push_hits(hits);
                    }
                    (Unit::Wide(c, _), StageKind::Restitution) => apply_restitution_convex(
                        c,
                        states,
                        context,
                        self.restitution_threshold,
                        self.propagate,
                    ),
                    (Unit::Mesh(c, lanes), StageKind::PrepareContacts) => {
                        if prepare_mesh_contacts(c, lanes, sims, states, context) {
                            self.any_restitution.store(true, Ordering::Relaxed);
                        }
                    }
                    (Unit::Mesh(c, _), StageKind::WarmStart) => warm_start_contacts(c, states),
                    (Unit::Mesh(c, _), StageKind::Solve) => push_contacts(c, states, context),
                    (Unit::Mesh(c, _), StageKind::Relax) => solve_contacts(c, states, context),
                    (Unit::Mesh(c, lanes), StageKind::StoreImpulses) => {
                        let mut hits = Vec::new();
                        store_impulses(c, lanes, self.world.neg_hit_threshold, &mut hits);
                        self.push_hits(hits);
                    }
                    (Unit::Mesh(c, _), StageKind::Restitution) => apply_restitution(
                        c,
                        states,
                        context,
                        self.restitution_threshold,
                        self.propagate,
                    ),
                    (_, StageKind::IntegrateVelocities | StageKind::IntegratePositions) => {
                        unreachable!()
                    }
                }
            }
        }
    }

    fn push_hits(&self, mut hits: Vec<i32>) {
        if !hits.is_empty() {
            self.hits
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .append(&mut hits);
        }
    }

    /// A non-orchestrating worker: join each published stage until the orchestrator is done.
    fn worker_loop(&self) {
        let worker_index = self.worker_index();
        let mut seen = 0;
        loop {
            let mut spins = 0u32;
            let published = loop {
                let published = self.published.load(Ordering::Acquire);
                if published != seen {
                    break published;
                }
                if self.finished.load(Ordering::Acquire) {
                    return;
                }
                spins += 1;
                if spins < 2000 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            };
            seen = published;
            self.work(published - 1, worker_index);
        }
    }
}

/// Whether a joint with a force or torque threshold exceeded it this substep and isn't
/// flagged yet.
pub(super) fn joint_reaction_exceeded(
    world: &SolverWorld<'_>,
    joint: &JointSim,
    inv_h: f32,
) -> bool {
    if !(joint.force_threshold < f32::MAX || joint.torque_threshold < f32::MAX) {
        return false;
    }
    if world.joint_state_bit_set.get_bit(joint.joint_id as u32) {
        return false;
    }

    let (force, torque) = get_joint_reaction_in(world.bodies, world.solver_sets, joint, inv_h);

    // Check thresholds. A zero threshold means all awake joints get reported.
    force >= joint.force_threshold || torque >= joint.torque_threshold
}

/// The constraints of the overflow color, run serially by the orchestrator, with the mesh
/// constraints' contacts.
pub(super) struct OverflowConstraints<'a, 'c> {
    pub joints: &'a mut [JointSim],
    pub mesh: &'a mut [ContactConstraint],
    pub lanes: &'a mut [&'c mut Contact],
}

/// The constraints of one graph color, with the contacts of the contact constraints.
pub(super) struct ColorConstraints<'a, 'c> {
    pub joints: &'a mut [JointSim],
    pub wide: &'a mut [ContactConstraintWide],
    /// Four per wide constraint.
    pub convex_lanes: &'a mut [&'c mut Contact],
    pub mesh: &'a mut [ContactConstraint],
    pub scalar_lanes: &'a mut [&'c mut Contact],
}

/// What the stages leave for the rest of the step.
pub(super) struct SubstepsOut {
    /// The joints the biased solve flagged for `joint_state_bit_set`.
    pub flagged_joints: Vec<i32>,
    /// The hit-event candidate contacts.
    pub hits: Vec<i32>,
    pub times: StageTimes,
}

/// Per stage kind timings for the step profile.
#[derive(Default)]
pub(super) struct StageTimes {
    pub prepare_contacts: f32,
    pub integrate_velocities: f32,
    pub warm_start: f32,
    pub solve_impulses: f32,
    pub integrate_positions: f32,
    pub relax_impulses: f32,
    pub restitution: f32,
    pub store_impulses: f32,
}

impl StageTimes {
    pub fn add_to(&self, profile: &mut Profile) {
        profile.prepare_constraints += self.prepare_contacts;
        profile.store_impulses += self.store_impulses;
        profile.integrate_velocities += self.integrate_velocities;
        profile.warm_start += self.warm_start;
        profile.solve_impulses += self.solve_impulses;
        profile.integrate_positions += self.integrate_positions;
        profile.relax_impulses += self.relax_impulses;
        profile.restitution += self.restitution;
    }
}

/// Prepare the contact constraints, run the substeps and restitution, and store the
/// impulses. The constraint slices must already have their final lengths.
pub(super) fn solve_substeps(
    world: &SolverWorld<'_>,
    states: &SolverStates,
    context: &StepContext,
    overflow: OverflowConstraints<'_, '_>,
    colors: Vec<ColorConstraints<'_, '_>>,
) -> SubstepsOut {
    let pool = world.pool;
    let worker_count = pool.map_or(1, |p| p.current_num_threads()).max(1);
    let max_block_count = 4 * worker_count;

    // Blocks per active color: joints, then wide contacts, then mesh contacts.
    let mut units: Vec<Vec<Mutex<Unit<'_, '_>>>> = Vec::new();
    let mut contact_units = Vec::new();
    for color in colors {
        let ColorConstraints {
            joints,
            wide,
            mut convex_lanes,
            mesh,
            scalar_lanes,
        } = color;
        debug_assert!(wide.len() == convex_lanes.len().div_ceil(crate::core::SIMD_WIDTH));
        debug_assert!(mesh.len() == scalar_lanes.len());
        if joints.is_empty() && wide.is_empty() && mesh.is_empty() {
            continue;
        }
        let mut color_units = Vec::new();
        let (size, _) = block_dim(joints.len(), 4, max_block_count);
        color_units.extend(joints.chunks_mut(size).map(|c| Mutex::new(Unit::Joints(c))));
        let first_contact_unit = color_units.len();
        let (size, _) = block_dim(wide.len(), 4, max_block_count);
        for c in wide.chunks_mut(size) {
            let n = (c.len() * crate::core::SIMD_WIDTH).min(convex_lanes.len());
            let (lanes, rest) = std::mem::take(&mut convex_lanes).split_at_mut(n);
            convex_lanes = rest;
            color_units.push(Mutex::new(Unit::Wide(c, lanes)));
        }
        let (size, _) = block_dim(mesh.len(), 4, max_block_count);
        color_units.extend(
            mesh.chunks_mut(size)
                .zip(scalar_lanes.chunks_mut(size))
                .map(|(c, lanes)| Mutex::new(Unit::Mesh(c, lanes))),
        );
        let color_index = units.len();
        contact_units.extend((first_contact_unit..color_units.len()).map(|u| (color_index, u)));
        units.push(color_units);
    }
    let active_color_count = units.len();

    // The stage sequence. (b3SolverTask's main loop)
    let mut stages = vec![Stage {
        kind: StageKind::PrepareContacts,
        color: 0,
        overflow: Overflow::Prepare,
        prev: 0,
    }];
    let graph = |stages: &mut Vec<Stage>, kind: StageKind, overflow: Overflow| {
        for color in 0..active_color_count {
            let overflow = if color == 0 { overflow } else { Overflow::None };
            stages.push(Stage {
                kind,
                color,
                overflow,
                prev: 0,
            });
        }
        if active_color_count == 0 && overflow != Overflow::None {
            // Keep the overflow work with an empty stage to hang it on.
            stages.push(Stage {
                kind,
                color: usize::MAX,
                overflow,
                prev: 0,
            });
        }
    };
    for _ in 0..context.sub_step_count {
        stages.push(Stage {
            kind: StageKind::IntegrateVelocities,
            color: 0,
            overflow: Overflow::None,
            prev: 0,
        });
        graph(&mut stages, StageKind::WarmStart, Overflow::WarmStart);
        for _ in 0..SOLVER_ITERATIONS {
            graph(&mut stages, StageKind::Solve, Overflow::Solve);
        }
        stages.push(Stage {
            kind: StageKind::IntegratePositions,
            color: 0,
            overflow: Overflow::None,
            prev: 0,
        });
        for _ in 0..RELAX_ITERATIONS {
            graph(&mut stages, StageKind::Relax, Overflow::Relax);
        }
    }
    // These run only if prepare finds restitution. Nothing after them shares their blocks,
    // so skipping them leaves the claim chains intact.
    for _ in 0..world.restitution_iterations {
        graph(&mut stages, StageKind::Restitution, Overflow::Restitution);
    }
    stages.push(Stage {
        kind: StageKind::StoreImpulses,
        color: 0,
        overflow: Overflow::Store,
        prev: 0,
    });

    // Chain each stage to the last one on the same blocks.
    let mut last_body = 0;
    let mut last_contact = 0;
    let mut last_color = vec![0; active_color_count];
    for (index, stage) in stages.iter_mut().enumerate() {
        let last = match stage.kind {
            kind if kind.on_bodies() => &mut last_body,
            kind if kind.on_contacts() => &mut last_contact,
            _ if stage.color == usize::MAX => continue,
            _ => &mut last_color[stage.color],
        };
        stage.prev = *last;
        *last = index as u32 + 1;
    }
    let syncs: Vec<StageSync> = stages
        .iter()
        .map(|_| StageSync {
            done: AtomicUsize::new(0),
        })
        .collect();
    let unit_claims: Vec<Vec<AtomicU32>> = units
        .iter()
        .map(|u| u.iter().map(|_| AtomicU32::new(0)).collect())
        .collect();

    let body_count = states.len();
    let (body_block, body_block_count) = block_dim(body_count, 32, max_block_count);
    let body_claims: Vec<AtomicU32> = (0..body_block_count).map(|_| AtomicU32::new(0)).collect();
    let contact_claims: Vec<AtomicU32> = contact_units.iter().map(|_| AtomicU32::new(0)).collect();
    let shared = Shared {
        world,
        states,
        context,
        stages: &stages,
        syncs: &syncs,
        units: &units,
        contact_units: &contact_units,
        unit_claims: &unit_claims,
        body_claims: &body_claims,
        contact_claims: &contact_claims,
        worker_count,
        main_thread: crate::parallel::thread_index(),
        body_count,
        body_block,
        body_block_count,
        restitution_threshold: world.restitution_threshold,
        propagate: world.enable_restitution_propagation,
        published: AtomicUsize::new(0),
        finished: AtomicBool::new(false),
        flagged_joints: Mutex::new(Vec::new()),
        any_restitution: AtomicBool::new(false),
        hits: Mutex::new(Vec::new()),
    };

    let mut overflow = overflow;
    let mut times = StageTimes::default();
    let orchestrate = |overflow: &mut OverflowConstraints<'_, '_>, times: &mut StageTimes| {
        let mut ticks = get_ticks();
        let mut flagged = Vec::new();
        let sims = &world.solver_sets[crate::solver_set::AWAKE_SET as usize].body_sims;
        for (index, stage) in stages.iter().enumerate() {
            if stage.kind == StageKind::Restitution
                && !shared.any_restitution.load(Ordering::Relaxed)
            {
                continue;
            }
            // Overflow constraints go first and have lower priority. Typically these are
            // dynamic-vs-dynamic.
            match stage.overflow {
                Overflow::None => {}
                Overflow::Prepare => {
                    // (b3PrepareContacts_Overflow)
                    if prepare_mesh_contacts(overflow.mesh, overflow.lanes, sims, states, context) {
                        shared.any_restitution.store(true, Ordering::Relaxed);
                    }
                }
                Overflow::Store => {
                    // (b3StoreImpulses_Overflow)
                    let mut hits = Vec::new();
                    store_impulses(
                        overflow.mesh,
                        overflow.lanes,
                        world.neg_hit_threshold,
                        &mut hits,
                    );
                    shared.push_hits(hits);
                }
                Overflow::WarmStart => {
                    for joint in overflow.joints.iter_mut() {
                        warm_start_joint(joint, states);
                    }
                    warm_start_contacts(overflow.mesh, states);
                }
                Overflow::Solve | Overflow::Relax => {
                    let use_bias = stage.overflow == Overflow::Solve;
                    for joint in overflow.joints.iter_mut() {
                        solve_joint(joint, context, states, use_bias);
                        if use_bias && joint_reaction_exceeded(world, joint, context.inv_h) {
                            flagged.push(joint.joint_id);
                        }
                    }
                    if use_bias {
                        // (b3PushContacts_Overflow)
                        push_contacts(overflow.mesh, states, context);
                    } else {
                        // (b3SolveContacts_Overflow)
                        solve_contacts(overflow.mesh, states, context);
                    }
                }
                Overflow::Restitution => {
                    // (b3ApplyRestitution_Overflow)
                    apply_restitution(
                        overflow.mesh,
                        states,
                        context,
                        shared.restitution_threshold,
                        shared.propagate,
                    );
                }
            }

            if stage.color != usize::MAX {
                let count = shared.block_count(stage);
                if count > 1 {
                    shared.published.store(index + 1, Ordering::Release);
                    shared.work(index, 0);
                    let mut spins = 0u32;
                    while syncs[index].done.load(Ordering::Acquire) != count {
                        spins += 1;
                        if spins < 2000 {
                            std::hint::spin_loop();
                        } else {
                            std::thread::yield_now();
                        }
                    }
                } else if count == 1 {
                    shared.claims(stage)[0].store(index as u32 + 1, Ordering::Relaxed);
                    shared.run_block(stage, 0);
                }
            }

            // A stage's time is charged when the next stage kind starts.
            let next_kind = stages.get(index + 1).map(|s| s.kind);
            if next_kind != Some(stage.kind) {
                let ms = get_milliseconds_and_reset(&mut ticks);
                match stage.kind {
                    StageKind::PrepareContacts => times.prepare_contacts += ms,
                    StageKind::StoreImpulses => times.store_impulses += ms,
                    StageKind::IntegrateVelocities => times.integrate_velocities += ms,
                    StageKind::WarmStart => times.warm_start += ms,
                    StageKind::Solve => times.solve_impulses += ms,
                    StageKind::IntegratePositions => times.integrate_positions += ms,
                    StageKind::Relax => times.relax_impulses += ms,
                    StageKind::Restitution => times.restitution += ms,
                }
            }
        }
        shared.finished.store(true, Ordering::Release);
        flagged
    };

    let mut flagged = crate::parallel::with_workers(
        pool,
        worker_count,
        || orchestrate(&mut overflow, &mut times),
        || shared.worker_loop(),
    );
    flagged.append(
        &mut shared
            .flagged_joints
            .into_inner()
            .unwrap_or_else(|e| e.into_inner()),
    );
    SubstepsOut {
        flagged_joints: flagged,
        hits: shared.hits.into_inner().unwrap_or_else(|e| e.into_inner()),
        times,
    }
}
