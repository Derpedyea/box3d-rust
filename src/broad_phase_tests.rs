// Focused unit tests for broad_phase proxy ops and tree marks (C has no dedicated test
// file at this upstream revision).
// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

use crate::broad_phase::{proxy_id, proxy_key, proxy_type, BroadPhase};
use crate::dynamic_tree::DEFAULT_CATEGORY_BITS;
use crate::math_functions::{Aabb, Vec3};
use crate::types::{BodyType, Capacity};

fn unit_aabb(x: f32, y: f32, z: f32) -> Aabb {
    Aabb {
        lower_bound: Vec3 {
            x: x - 0.5,
            y: y - 0.5,
            z: z - 0.5,
        },
        upper_bound: Vec3 {
            x: x + 0.5,
            y: y + 0.5,
            z: z + 0.5,
        },
    }
}

#[test]
fn proxy_key_round_trip() {
    for type_ in [BodyType::Static, BodyType::Kinematic, BodyType::Dynamic] {
        for id in [0, 1, 7, 100] {
            let key = proxy_key(id, type_);
            assert_eq!(proxy_type(key), type_);
            assert_eq!(proxy_id(key), id);
        }
    }
}

#[test]
fn create_destroy_and_move_marks() {
    let capacity = Capacity {
        static_shape_count: 8,
        dynamic_shape_count: 8,
        static_body_count: 0,
        dynamic_body_count: 0,
        contact_count: 8,
    };
    let mut bp = BroadPhase::new(&capacity);
    let static_tree = BodyType::Static as usize;
    let dynamic_tree = BodyType::Dynamic as usize;

    // Static without forcePairCreation does not mark the tree.
    let static_key = bp.create_proxy(
        BodyType::Static,
        unit_aabb(0.0, 0.0, 0.0),
        DEFAULT_CATEGORY_BITS,
        10,
        false,
    );
    assert!(!bp.trees[static_tree].has_tree_moved());
    assert_eq!(bp.shape_index(static_key), 10);

    // Dynamic always marks.
    let dyn_key = bp.create_proxy(
        BodyType::Dynamic,
        unit_aabb(2.0, 0.0, 0.0),
        DEFAULT_CATEGORY_BITS,
        20,
        false,
    );
    assert!(bp.trees[dynamic_tree].has_tree_moved());
    assert_eq!(bp.shape_index(dyn_key), 20);

    // Static with forcePairCreation marks.
    let static_forced = bp.create_proxy(
        BodyType::Static,
        unit_aabb(-2.0, 0.0, 0.0),
        DEFAULT_CATEGORY_BITS,
        11,
        true,
    );
    assert!(bp.trees[static_tree].has_tree_moved());
    let mut moved = [0i32; 4];
    let moved_count = bp.trees[static_tree].gather_moved_proxies(&mut moved);
    assert_eq!(moved_count, 1);
    assert_eq!(moved[0], proxy_id(static_forced));

    // Destroying the only moved proxy re-derives the marks along its path.
    bp.destroy_proxy(dyn_key);
    assert!(!bp.trees[dynamic_tree].has_tree_moved());

    bp.trees[static_tree].clear_moved();
    bp.validate_no_moved();

    bp.destroy();
}

#[test]
fn move_and_overlap() {
    let capacity = Capacity {
        static_shape_count: 4,
        dynamic_shape_count: 4,
        ..Capacity::default()
    };
    let mut bp = BroadPhase::new(&capacity);
    let dynamic_tree = BodyType::Dynamic as usize;

    let a = bp.create_proxy(
        BodyType::Dynamic,
        unit_aabb(0.0, 0.0, 0.0),
        DEFAULT_CATEGORY_BITS,
        1,
        false,
    );
    let b = bp.create_proxy(
        BodyType::Dynamic,
        unit_aabb(0.25, 0.0, 0.0),
        DEFAULT_CATEGORY_BITS,
        2,
        false,
    );
    assert!(bp.test_overlap(a, b));

    bp.trees[dynamic_tree].rebuild(false);
    bp.validate_no_moved();

    bp.move_proxy(b, unit_aabb(10.0, 0.0, 0.0));
    assert!(!bp.test_overlap(a, b));
    let mut moved = [0i32; 4];
    assert_eq!(bp.trees[dynamic_tree].gather_moved_proxies(&mut moved), 1);
    assert_eq!(moved[0], proxy_id(b));

    // A marked proxy with a grown box is found by the moved gather and refit.
    let grown = Aabb {
        lower_bound: Vec3 {
            x: -1.0,
            y: -1.0,
            z: -1.0,
        },
        upper_bound: Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
    };
    bp.mark_proxy_moved(a, grown);
    assert_eq!(bp.trees[dynamic_tree].gather_moved_proxies(&mut moved), 2);
    bp.trees[dynamic_tree].refit();
    assert_eq!(bp.trees[dynamic_tree].aabb(proxy_id(a)), grown);

    bp.validate();
    bp.trees[dynamic_tree].rebuild(false);
    bp.validate_no_moved();
}

#[test]
fn defaults_have_secret_cookie() {
    use crate::core::SECRET_COOKIE;
    use crate::types::{default_body_def, default_filter, default_world_def};

    assert_eq!(default_body_def().internal_value, SECRET_COOKIE);
    assert_eq!(default_world_def().internal_value, SECRET_COOKIE);
    assert_eq!(default_filter().group_index, 0);
}

thread_local! {
    static FILTER_CALLS: std::cell::RefCell<Vec<(i32, i32)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn recording_filter(
    _world: &crate::world::World,
    id_a: crate::id::ShapeId,
    id_b: crate::id::ShapeId,
    _context: u64,
) -> bool {
    FILTER_CALLS.with(|calls| calls.borrow_mut().push((id_a.index1 - 1, id_b.index1 - 1)));
    true
}

/// C runs the cross pass through b3ParallelFor. With the default serial task system worker
/// 0 claims every block in order, and b3CrossPairsTask flushes its candidate batch at the
/// end of each block. With one worker and two cross seeds the block size is 1 (minRange 1,
/// 32 blocks per worker), so each seed is flushed on its own and the custom filter sees the
/// pairs in seed order. A single flush over both seeds would instead emit the compound pair
/// first, since b3FlushCandidatePairs expands compounds before filtering plain pairs.
///
/// Scene (the order follows from dynamic_tree.c b3InsertLeaf and broad_phase.c
/// b3GatherCrossSeeds): the second proxy inserted into a tree lands at pair index 2 as the
/// right sibling, so in both trees the first created shape is the left child. The seeds are
/// gathered breadth first over (dynamic child, static child) in (left, right) order:
/// seed 0 = (D1, plain static sphere), seed 1 = (D2, compound). C calls the filter for the
/// plain pair, then for the compound pair.
#[test]
fn cross_pass_flushes_per_parallel_for_block() {
    use crate::body::create_body;
    use crate::compound::{create_compound, CompoundDef, CompoundSphereDef};
    use crate::geometry::{default_surface_material, Sphere};
    use crate::math_functions::{Pos, VEC3_ZERO};
    use crate::shape::{create_baked_compound_shape, create_sphere_shape};
    use crate::types::{default_body_def, default_shape_def, default_world_def};
    use crate::world::{world_set_custom_filter_callback, World};

    FILTER_CALLS.with(|calls| calls.borrow_mut().clear());

    let mut world_def = default_world_def();
    world_def.gravity = VEC3_ZERO;
    let mut world = World::new(&world_def);
    assert_eq!(world.worker_count, 1);
    world_set_custom_filter_callback(&mut world, Some(recording_filter), 0);

    let mut shape_def = default_shape_def();
    shape_def.enable_custom_filtering = true;
    let sphere = Sphere {
        center: VEC3_ZERO,
        radius: 0.5,
    };

    let mut body_def = default_body_def();
    body_def.type_ = BodyType::Static;

    // Plain static sphere first, so it is the left child of the static tree.
    body_def.position = Pos {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let plain_body = create_body(&mut world, &body_def);
    let plain_shape = create_sphere_shape(&mut world, plain_body, &shape_def, &sphere);

    let spheres = [CompoundSphereDef {
        sphere,
        material: default_surface_material(),
    }];
    let compound = create_compound(&CompoundDef {
        spheres: &spheres,
        ..CompoundDef::default()
    })
    .expect("compound");
    body_def.position = Pos {
        x: 10.0,
        y: 0.0,
        z: 0.0,
    };
    let compound_body = create_body(&mut world, &body_def);
    let compound_shape =
        create_baked_compound_shape(&mut world, compound_body, &shape_def, &compound);

    // Dynamic spheres: D1 over the plain sphere (left child), D2 over the compound.
    body_def.type_ = BodyType::Dynamic;
    body_def.position = Pos {
        x: 0.0,
        y: 0.25,
        z: 0.0,
    };
    let d1 = create_body(&mut world, &body_def);
    let d1_shape = create_sphere_shape(&mut world, d1, &shape_def, &sphere);
    body_def.position = Pos {
        x: 10.0,
        y: 0.25,
        z: 0.0,
    };
    let d2 = create_body(&mut world, &body_def);
    let d2_shape = create_sphere_shape(&mut world, d2, &shape_def, &sphere);

    crate::broad_phase::update_broad_phase_pairs(&mut world);

    let unordered = |a: i32, b: i32| (a.min(b), a.max(b));
    let calls: Vec<(i32, i32)> = FILTER_CALLS.with(|calls| {
        calls
            .borrow()
            .iter()
            .map(|&(a, b)| unordered(a, b))
            .collect()
    });
    let plain_pair = unordered(plain_shape.index1 - 1, d1_shape.index1 - 1);
    let compound_pair = unordered(compound_shape.index1 - 1, d2_shape.index1 - 1);
    assert_eq!(calls, vec![plain_pair, compound_pair]);
}
