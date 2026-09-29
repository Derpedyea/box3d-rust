//! Port of `test/test_determinism.c`: falling-ragdoll sleep step and world
//! transform hash must match the C scalar (`BOX3D_DISABLE_SIMD`) reference.
//!
//! The Rust port is serial, so the C worker-count loops cannot vary the schedule.
//! Both precision modes have distinct golden values.
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use crate::constants::MAX_MANIFOLD_POINTS;
use crate::determinism::{
    create_falling_ragdolls, create_mesh_drop, create_query_spawn, create_wave_pile,
    destroy_falling_ragdolls, destroy_mesh_drop, destroy_query_spawn, destroy_wave_pile,
    update_falling_ragdolls, update_mesh_drop, update_query_spawn, update_wave_pile,
    FallingRagdollData, QUERY_SPAWN_COUNT,
};
use crate::math_functions::POS_ZERO;
use crate::types::default_world_def;
use crate::world::World;

// Double precision accumulates body positions in double, so the settle/sleep
// step and the state hash differ from the float build. Both modes are
// internally deterministic. Values from test_determinism.c.
#[cfg(feature = "double-precision")]
const RAGDOLL_SLEEP_STEP: i32 = 273;
#[cfg(feature = "double-precision")]
const RAGDOLL_HASH: u32 = 0x0F4D_D1EB;
#[cfg(feature = "double-precision")]
const WAVE_PILE_SLEEP_STEP: i32 = 314;
#[cfg(feature = "double-precision")]
const WAVE_PILE_HASH: u32 = 0xD9B8_8124;
#[cfg(feature = "double-precision")]
const QUERY_SPAWN_SLEEP_STEP: i32 = 242;
#[cfg(feature = "double-precision")]
const QUERY_SPAWN_HASH: u32 = 0x9B96_A098;
#[cfg(feature = "double-precision")]
const QUERY_SPAWN_HIT_COUNT: i32 = 59;
#[cfg(feature = "double-precision")]
const QUERY_SPAWN_QUERY_HASH: u32 = 0x5B44_29DC;
#[cfg(feature = "double-precision")]
const MESH_DROP_SLEEP_STEP: i32 = 217;
#[cfg(feature = "double-precision")]
const MESH_DROP_HASH: u32 = 0xCF94_6437;

#[cfg(not(feature = "double-precision"))]
const RAGDOLL_SLEEP_STEP: i32 = 274;
#[cfg(not(feature = "double-precision"))]
const RAGDOLL_HASH: u32 = 0x773A_B8ED;
#[cfg(not(feature = "double-precision"))]
const WAVE_PILE_SLEEP_STEP: i32 = 330;
#[cfg(not(feature = "double-precision"))]
const WAVE_PILE_HASH: u32 = 0x205D_6055;
#[cfg(not(feature = "double-precision"))]
const QUERY_SPAWN_SLEEP_STEP: i32 = 242;
#[cfg(not(feature = "double-precision"))]
const QUERY_SPAWN_HASH: u32 = 0xF1ED_EF47;
#[cfg(not(feature = "double-precision"))]
const QUERY_SPAWN_HIT_COUNT: i32 = 59;
#[cfg(not(feature = "double-precision"))]
const QUERY_SPAWN_QUERY_HASH: u32 = 0xE327_1F3D;
#[cfg(not(feature = "double-precision"))]
const MESH_DROP_SLEEP_STEP: i32 = 217;
#[cfg(not(feature = "double-precision"))]
const MESH_DROP_HASH: u32 = 0x256B_C40B;

// C loops each scenario over several worker counts to prove every schedule reproduces the
// first run (EnsureRepeatable). The serial port ignores the worker count, so extra runs would
// be identical work; exactly two runs per scenario keep the useful part of the check (run 2
// in a fresh world must match run 1, which catches state leaking between worlds).
const REPEAT_RUNS: i32 = 2;

// The goldens above pin exact values for the default four point manifold. A build that
// overrides B3_MAX_MANIFOLD_POINTS produces a different contact set, so those checks drop
// out and only the run to run agreement below applies. (ENSURE_GOLDEN)
const ENSURE_GOLDEN: bool = MAX_MANIFOLD_POINTS == 4;

/// (DeterminismResult)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DeterminismResult {
    sleep_step: i32,
    hash: u32,
    query_hit_count: i32,
    query_hash: u32,
    seeded: bool,
}

/// Every worker count has to reproduce the first run exactly. Scenarios without queries leave
/// the query fields zero in both the reference and the result. (EnsureRepeatable)
fn ensure_repeatable(
    reference: &mut DeterminismResult,
    mut result: DeterminismResult,
    label: &str,
) {
    if !reference.seeded {
        result.seeded = true;
        *reference = result;
        return;
    }

    assert_eq!(
        result.sleep_step, reference.sleep_step,
        "{label}: repeat sleep step"
    );
    assert_eq!(result.hash, reference.hash, "{label}: repeat hash");
    assert_eq!(
        result.query_hit_count, reference.query_hit_count,
        "{label}: repeat query hit count"
    );
    assert_eq!(
        result.query_hash, reference.query_hash,
        "{label}: repeat query hash"
    );
}

fn ensure_ragdoll_golden(data: &FallingRagdollData, label: &str) {
    if data.sleep_step != RAGDOLL_SLEEP_STEP || data.hash != RAGDOLL_HASH {
        eprintln!(
            "  {label} sleepStep={} hash=0x{:08X} (expected sleepStep={} hash=0x{:08X})",
            data.sleep_step, data.hash, RAGDOLL_SLEEP_STEP, RAGDOLL_HASH
        );
    }

    if ENSURE_GOLDEN {
        assert_eq!(
            data.sleep_step, RAGDOLL_SLEEP_STEP,
            "{label}: sleep_step mismatch"
        );
        assert_eq!(data.hash, RAGDOLL_HASH, "{label}: hash mismatch");
    }
}

/// Serial equivalent of C `SingleMultithreadingTest`. `run` only labels the output.
fn single_multithreading_test(run: i32, reference: &mut DeterminismResult) {
    let mut world = World::new(&default_world_def());
    let mut data = create_falling_ragdolls(&mut world);

    let time_step = 1.0 / 60.0;
    let step_limit = 500;
    for _ in 0..step_limit {
        world.step(time_step, 4);
        if update_falling_ragdolls(&world, &mut data) {
            break;
        }
    }

    let label = format!("run={run}");
    ensure_ragdoll_golden(&data, &label);
    ensure_repeatable(
        reference,
        DeterminismResult {
            sleep_step: data.sleep_step,
            hash: data.hash,
            ..Default::default()
        },
        &label,
    );

    destroy_falling_ragdolls(&mut data);
}

/// Test multithreaded determinism. (MultithreadingTest)
#[test]
fn multithreading_test() {
    let mut reference = DeterminismResult::default();
    for run in 1..=REPEAT_RUNS {
        single_multithreading_test(run, &mut reference);
    }
}

/// Test cross platform determinism. (CrossPlatformTest)
#[test]
fn cross_platform_test() {
    let mut world = World::new(&default_world_def());
    let mut data = create_falling_ragdolls(&mut world);

    let time_step = 1.0 / 60.0;
    loop {
        world.step(time_step, 4);
        if update_falling_ragdolls(&world, &mut data) {
            break;
        }
    }

    ensure_ragdoll_golden(&data, "cross-platform");

    destroy_falling_ragdolls(&mut data);
}

/// Serial equivalent of C `SingleWavePileTest`.
fn single_wave_pile_test(run: i32, reference: &mut DeterminismResult) {
    let mut world = World::new(&default_world_def());
    let mut data = create_wave_pile(&mut world);

    let time_step = 1.0 / 60.0;

    // Rolling resistance must put the pile to sleep within 500 steps.
    let mut done = false;
    for _ in 0..500 {
        if done {
            break;
        }
        world.step(time_step, 4);
        done = update_wave_pile(&world, &mut data);
    }

    if data.sleep_step != WAVE_PILE_SLEEP_STEP || data.hash != WAVE_PILE_HASH {
        eprintln!(
            "  wave pile run={run} sleepStep={} hash=0x{:08X} (expected sleepStep={} hash=0x{:08X})",
            data.sleep_step, data.hash, WAVE_PILE_SLEEP_STEP, WAVE_PILE_HASH
        );
    }

    assert!(done, "wave pile did not settle within 500 steps");
    if ENSURE_GOLDEN {
        assert_eq!(
            data.sleep_step, WAVE_PILE_SLEEP_STEP,
            "wave pile sleep step"
        );
        assert_eq!(data.hash, WAVE_PILE_HASH, "wave pile hash");
    }
    ensure_repeatable(
        reference,
        DeterminismResult {
            sleep_step: data.sleep_step,
            hash: data.hash,
            ..Default::default()
        },
        "wave pile",
    );

    destroy_wave_pile(&mut data);
}

/// Test multithreaded determinism of a mixed convex pile on a wave height field.
/// (WavePileTest)
#[test]
fn wave_pile_test() {
    let mut reference = DeterminismResult::default();
    for run in 1..=REPEAT_RUNS {
        single_wave_pile_test(run, &mut reference);
    }
}

/// Serial equivalent of C `SingleQuerySpawnTest`.
fn single_query_spawn_test(run: i32, reference: &mut DeterminismResult) {
    let mut world = World::new(&default_world_def());
    let mut data = create_query_spawn(&mut world);

    let time_step = 1.0 / 60.0;

    let mut done = false;
    for _ in 0..1000 {
        if done {
            break;
        }
        world.step(time_step, 4);
        done = update_query_spawn(&mut world, &mut data);
    }

    if data.sleep_step != QUERY_SPAWN_SLEEP_STEP
        || data.hash != QUERY_SPAWN_HASH
        || data.query_hit_count != QUERY_SPAWN_HIT_COUNT
        || data.query_hash != QUERY_SPAWN_QUERY_HASH
    {
        eprintln!(
            "  query spawn run={run} sleepStep={} hash=0x{:08X} hits={} queryHash=0x{:08X}\n  \
             expected  sleepStep={} hash=0x{:08X} hits={} queryHash=0x{:08X}",
            data.sleep_step,
            data.hash,
            data.query_hit_count,
            data.query_hash,
            QUERY_SPAWN_SLEEP_STEP,
            QUERY_SPAWN_HASH,
            QUERY_SPAWN_HIT_COUNT,
            QUERY_SPAWN_QUERY_HASH
        );
    }

    assert!(done, "query spawn did not settle within 1000 steps");
    assert_eq!(
        data.spawn_count as usize, QUERY_SPAWN_COUNT,
        "query spawn count"
    );
    if ENSURE_GOLDEN {
        assert_eq!(
            data.sleep_step, QUERY_SPAWN_SLEEP_STEP,
            "query spawn sleep step"
        );
        assert_eq!(data.hash, QUERY_SPAWN_HASH, "query spawn hash");
        assert_eq!(
            data.query_hit_count, QUERY_SPAWN_HIT_COUNT,
            "query spawn hit count"
        );
        assert_eq!(
            data.query_hash, QUERY_SPAWN_QUERY_HASH,
            "query spawn query hash"
        );
    }
    ensure_repeatable(
        reference,
        DeterminismResult {
            sleep_step: data.sleep_step,
            hash: data.hash,
            query_hit_count: data.query_hit_count,
            query_hash: data.query_hash,
            seeded: false,
        },
        "query spawn",
    );

    destroy_query_spawn(&mut data);
}

/// Test determinism of world queries by feeding their results back into the simulation.
/// (QuerySpawnTest)
#[test]
fn query_spawn_test() {
    let mut reference = DeterminismResult::default();
    for run in 1..=REPEAT_RUNS {
        single_query_spawn_test(run, &mut reference);
    }
}

/// Serial equivalent of C `SingleMeshDropTest`.
fn single_mesh_drop_test(run: i32, reference: &mut DeterminismResult) {
    let mut world = World::new(&default_world_def());
    let mut data = create_mesh_drop(&mut world, POS_ZERO);

    let time_step = 1.0 / 60.0;

    let mut done = false;
    for _ in 0..400 {
        if done {
            break;
        }
        world.step(time_step, 4);
        done = update_mesh_drop(&world, &mut data);
    }

    if data.sleep_step != MESH_DROP_SLEEP_STEP || data.hash != MESH_DROP_HASH {
        eprintln!(
            "  mesh drop run={run} sleepStep={} hash=0x{:08X} (expected sleepStep={} hash=0x{:08X})",
            data.sleep_step, data.hash, MESH_DROP_SLEEP_STEP, MESH_DROP_HASH
        );
    }

    assert!(done, "mesh drop did not settle within 400 steps");
    if ENSURE_GOLDEN {
        assert_eq!(
            data.sleep_step, MESH_DROP_SLEEP_STEP,
            "mesh drop sleep step"
        );
        assert_eq!(data.hash, MESH_DROP_HASH, "mesh drop hash");
    }
    ensure_repeatable(
        reference,
        DeterminismResult {
            sleep_step: data.sleep_step,
            hash: data.hash,
            ..Default::default()
        },
        "mesh drop",
    );

    destroy_mesh_drop(&mut data);
}

/// C runs only the single threaded and widest schedules because the scene is large.
/// (MeshDropTest)
#[test]
fn mesh_drop_test() {
    let mut reference = DeterminismResult::default();
    for run in 1..=REPEAT_RUNS {
        single_mesh_drop_test(run, &mut reference);
    }
}
