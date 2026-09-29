//! Broad-phase tree maintenance test from test_world.c (TestEnlargedProxyDestroyed).

use crate::body::{create_body, destroy_body};
use crate::dynamic_tree::{DynamicTree, ALLOCATED_NODE, ENLARGED_NODE};
use crate::geometry::Sphere;
use crate::math_functions::{Pos, Vec3, VEC3_ZERO};
use crate::shape::create_sphere_shape;
use crate::types::{default_body_def, default_shape_def, default_world_def, BodyType};
use crate::world::World;

/// (CountEnlargedNodes)
fn count_enlarged_nodes(tree: &DynamicTree) -> i32 {
    let mut count = 0;
    for i in 0..tree.node_capacity() {
        let node = &tree.nodes[i as usize];
        if (node.flags & ALLOCATED_NODE) != 0 && (node.flags & ENLARGED_NODE) != 0 {
            count += 1;
        }
    }

    count
}

/// A proxy can be enlarged in one step and destroyed before the next, emptying the move buffer.
/// The trees must still be rebuilt or the enlarged nodes survive into the next step. Issue #149.
/// (TestEnlargedProxyDestroyed)
#[test]
fn test_enlarged_proxy_destroyed() {
    let mut world_def = default_world_def();
    world_def.gravity = VEC3_ZERO;
    let mut world = World::new(&world_def);

    let dynamic_tree = BodyType::Dynamic as usize;

    let mut body_def = default_body_def();
    body_def.type_ = BodyType::Dynamic;

    let shape_def = default_shape_def();
    let sphere = Sphere {
        center: VEC3_ZERO,
        radius: 0.25,
    };

    // Resting bodies keep the tree deep enough that the mover has an ancestor that outlives it
    for i in 0..8 {
        body_def.position = Pos {
            x: (2.0 * i as f32) as _,
            y: 0.0,
            z: 0.0,
        };
        let resting_id = create_body(&mut world, &body_def);
        create_sphere_shape(&mut world, resting_id, &shape_def, &sphere);
    }

    // This body outruns its AABB margin every step
    body_def.position = Pos {
        x: 7.0,
        y: 1.0,
        z: 0.0,
    };
    body_def.linear_velocity = Vec3::new(6.0, 0.0, 0.0);
    let mover_id = create_body(&mut world, &body_def);
    create_sphere_shape(&mut world, mover_id, &shape_def, &sphere);

    let time_step = 1.0 / 60.0;
    world.step(time_step, 4);

    assert!(count_enlarged_nodes(&world.broad_phase.trees[dynamic_tree]) > 0);

    destroy_body(&mut world, mover_id);

    // The mover was the only proxy in the move buffer
    assert_eq!(world.broad_phase.move_array.len(), 0);
    assert!(count_enlarged_nodes(&world.broad_phase.trees[dynamic_tree]) > 0);

    world.step(time_step, 4);

    assert_eq!(
        count_enlarged_nodes(&world.broad_phase.trees[dynamic_tree]),
        0
    );
}
