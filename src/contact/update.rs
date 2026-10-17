// Narrow-phase contact update: convex manifold compute and update_contact.
// Mesh/height manifolds: mesh_contact.rs.
//
// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

use super::collide::CollideStats;
use super::mesh_contact::{apply_mesh_hit_flags, compute_mesh_manifolds};
use super::{contact_flags, Contact, ContactCache, ContactGeometry, ConvexContact};
use crate::constants::MAX_MANIFOLD_POINTS;
use crate::core::NULL_INDEX;
use crate::distance::SimplexCache;
use crate::geometry::ShapeType;
use crate::geometry::SurfaceMaterial;
use crate::geometry::{Capsule, Sphere};
use crate::hull::HullData;
use crate::id::ShapeId;
use crate::manifold::{
    collide_capsule_and_sphere, collide_capsules, collide_hull_and_capsule,
    collide_hull_and_sphere, collide_hulls, collide_spheres, make_feature_id, LocalManifold,
    Manifold, ManifoldPoint, SatCache,
};
use crate::math_functions::{
    add, inv_mul_world_transforms, make_matrix_from_quat, max_float, mul_mv, mul_world_transforms,
    neg, offset_pos, rotate_vector, sub, sub_pos, WorldTransform,
};
use crate::shape::{shape_flags, Shape, ShapeGeometry};
use crate::types::{FrictionCallback, RestitutionCallback};
use crate::world::{PreSolveFcn, World};

/// The world settings the narrow phase reads while it updates contacts. Copied out of the
/// world once per collide pass so a contact update can borrow its contact and task context
/// mutably next to the shapes it reads (C passes `b3World*` and pointers everywhere).
#[derive(Clone, Copy)]
pub(crate) struct ContactParams {
    pub friction_callback: FrictionCallback,
    pub restitution_callback: RestitutionCallback,
    pub pre_solve_fcn: Option<PreSolveFcn>,
    pub pre_solve_context: u64,
    pub world_id: u16,
}

impl ContactParams {
    pub fn new(world: &World) -> ContactParams {
        ContactParams::new_from(
            world.friction_callback,
            world.restitution_callback,
            world.pre_solve_fcn,
            world.pre_solve_context,
            world.world_id,
        )
    }

    pub fn new_from(
        friction_callback: Option<FrictionCallback>,
        restitution_callback: Option<RestitutionCallback>,
        pre_solve_fcn: Option<PreSolveFcn>,
        pre_solve_context: u64,
        world_id: u16,
    ) -> ContactParams {
        ContactParams {
            friction_callback: friction_callback.unwrap_or(crate::world::default_friction_callback),
            restitution_callback: restitution_callback
                .unwrap_or(crate::world::default_restitution_callback),
            pre_solve_fcn,
            pre_solve_context,
            world_id,
        }
    }
}

/// What one contact update may write: its contact, its block's counters, and the block's
/// local manifold scratch (see [`with_local_manifold`]).
pub(crate) struct NarrowCtx<'a> {
    pub contact: &'a mut Contact,
    pub stats: &'a mut CollideStats,
    pub params: &'a ContactParams,
    pub local: &'a mut LocalManifold,
}

/// A borrowed convex geometry: a shape's own, or a compound child's.
#[derive(Clone, Copy)]
enum ConvexGeometry<'a> {
    Sphere(&'a Sphere),
    Capsule(&'a Capsule),
    Hull(&'a HullData),
}

impl ConvexGeometry<'_> {
    fn shape_type(self) -> ShapeType {
        match self {
            ConvexGeometry::Sphere(_) => ShapeType::Sphere,
            ConvexGeometry::Capsule(_) => ShapeType::Capsule,
            ConvexGeometry::Hull(_) => ShapeType::Hull,
        }
    }
}

impl<'a> ConvexGeometry<'a> {
    fn of(geometry: &'a ShapeGeometry) -> ConvexGeometry<'a> {
        match geometry {
            ShapeGeometry::Sphere(s) => ConvexGeometry::Sphere(s),
            ShapeGeometry::Capsule(c) => ConvexGeometry::Capsule(c),
            ShapeGeometry::Hull(h) => ConvexGeometry::Hull(h),
            _ => unreachable!("convex geometry expected"),
        }
    }
}

fn convex_cache(geometry: &mut ContactGeometry) -> &mut ContactCache {
    match geometry {
        ContactGeometry::Convex(convex) => &mut convex.cache,
        ContactGeometry::Mesh(_) => {
            // Convex update always runs on convex geometry contacts.
            geometry.as_convex_cache()
        }
    }
}

impl ContactGeometry {
    fn as_convex_cache(&mut self) -> &mut ContactCache {
        *self = ContactGeometry::Convex(ConvexContact::default());
        match self {
            ContactGeometry::Convex(c) => &mut c.cache,
            ContactGeometry::Mesh(_) => unreachable!(),
        }
    }
}

fn ensure_sat_cache(cache: &mut ContactCache) -> &mut SatCache {
    if !matches!(cache, ContactCache::Sat(_)) {
        *cache = ContactCache::Sat(SatCache::default());
    }
    match cache {
        ContactCache::Sat(s) => s,
        ContactCache::Simplex(_) => unreachable!(),
    }
}

fn ensure_simplex_cache(cache: &mut ContactCache) -> &mut SimplexCache {
    if !matches!(cache, ContactCache::Simplex(_)) {
        *cache = ContactCache::Simplex(SimplexCache::default());
    }
    match cache {
        ContactCache::Simplex(s) => s,
        ContactCache::Sat(_) => unreachable!(),
    }
}

fn shape_rolling_radius(geometry: ConvexGeometry<'_>) -> f32 {
    match geometry {
        ConvexGeometry::Sphere(s) => s.radius,
        ConvexGeometry::Capsule(c) => c.radius,
        ConvexGeometry::Hull(h) => 0.25 * h.inner_radius,
    }
}

thread_local! {
    /// The convex narrow phase's local manifold, zeroed once per thread: its 32-point buffer
    /// cost more to clear per contact than many contact updates.
    static CONVEX_MANIFOLD: std::cell::RefCell<Box<LocalManifold>> =
        std::cell::RefCell::new(Box::new(LocalManifold::default()));
}

/// (b3ComputeConvexManifold)
fn compute_convex_manifold(
    nc: &mut NarrowCtx<'_>,
    geom_a: ConvexGeometry<'_>,
    xf_a: WorldTransform,
    geom_b: ConvexGeometry<'_>,
    xf_b: WorldTransform,
) -> bool {
    nc.local.reset_header();
    compute_convex_manifold_in(
        &mut *nc.contact,
        &mut *nc.stats,
        &mut *nc.local,
        geom_a,
        xf_a,
        geom_b,
        xf_b,
    )
}

/// Run `f` with this thread's local manifold scratch, borrowed once for a block of contacts.
pub(crate) fn with_local_manifold<R>(f: impl FnOnce(&mut LocalManifold) -> R) -> R {
    CONVEX_MANIFOLD.with_borrow_mut(|local| f(local))
}

#[allow(clippy::too_many_arguments)]
fn compute_convex_manifold_in(
    contact: &mut Contact,
    stats: &mut CollideStats,
    geom_manifold: &mut LocalManifold,
    geom_a: ConvexGeometry<'_>,
    xf_a: WorldTransform,
    geom_b: ConvexGeometry<'_>,
    xf_b: WorldTransform,
) -> bool {
    let type_a = geom_a.shape_type();
    let type_b = geom_b.shape_type();

    let capacity = MAX_MANIFOLD_POINTS as i32;
    let transform_b_to_a = inv_mul_world_transforms(xf_a, xf_b);

    match type_a {
        ShapeType::Sphere => {
            debug_assert!(type_b == ShapeType::Sphere);
            let ConvexGeometry::Sphere(sphere_a) = geom_a else {
                unreachable!()
            };
            let ConvexGeometry::Sphere(sphere_b) = geom_b else {
                unreachable!()
            };
            collide_spheres(
                geom_manifold,
                capacity,
                sphere_a,
                sphere_b,
                transform_b_to_a,
            );
        }
        ShapeType::Capsule => {
            let ConvexGeometry::Capsule(capsule_a) = geom_a else {
                unreachable!()
            };
            if type_b == ShapeType::Sphere {
                let ConvexGeometry::Sphere(sphere_b) = geom_b else {
                    unreachable!()
                };
                collide_capsule_and_sphere(
                    geom_manifold,
                    capacity,
                    capsule_a,
                    sphere_b,
                    transform_b_to_a,
                );
            } else {
                debug_assert!(type_b == ShapeType::Capsule);
                let ConvexGeometry::Capsule(capsule_b) = geom_b else {
                    unreachable!()
                };
                collide_capsules(
                    geom_manifold,
                    capacity,
                    capsule_a,
                    capsule_b,
                    transform_b_to_a,
                );
            }
        }
        ShapeType::Hull => {
            debug_assert!(type_a == ShapeType::Hull);
            let ConvexGeometry::Hull(hull_a) = geom_a else {
                unreachable!()
            };
            let cache = convex_cache(&mut contact.geometry);

            if type_b == ShapeType::Sphere {
                let ConvexGeometry::Sphere(sphere_b) = geom_b else {
                    unreachable!()
                };
                let simplex = ensure_simplex_cache(cache);
                collide_hull_and_sphere(
                    geom_manifold,
                    capacity,
                    hull_a,
                    sphere_b,
                    transform_b_to_a,
                    simplex,
                );
            } else if type_b == ShapeType::Capsule {
                let ConvexGeometry::Capsule(capsule_b) = geom_b else {
                    unreachable!()
                };
                let simplex = ensure_simplex_cache(cache);
                collide_hull_and_capsule(
                    geom_manifold,
                    capacity,
                    hull_a,
                    capsule_b,
                    transform_b_to_a,
                    simplex,
                );
            } else {
                debug_assert!(type_b == ShapeType::Hull);
                let ConvexGeometry::Hull(hull_b) = geom_b else {
                    unreachable!()
                };
                let sat = ensure_sat_cache(cache);
                collide_hulls(
                    geom_manifold,
                    capacity,
                    hull_a,
                    hull_b,
                    transform_b_to_a,
                    sat,
                );
                stats.sat_call_count += 1;
                stats.sat_cache_hit_count += sat.hit as i32;
            }
        }
        _ => {
            debug_assert!(false, "compute_convex_manifold expects convex types");
            return false;
        }
    }

    if geom_manifold.point_count == 0 {
        contact.manifolds.clear();
        return false;
    }

    let mut old_points: [ManifoldPoint; MAX_MANIFOLD_POINTS];
    let old_count;
    {
        let contact = &mut *contact;
        if contact.manifolds.is_empty() {
            contact.manifolds.push(Manifold::default());
            old_points = [ManifoldPoint::default(); MAX_MANIFOLD_POINTS];
            old_count = 0;
        } else {
            old_count = contact.manifolds[0].point_count;
            old_points = contact.manifolds[0].points;
        }
    }

    let matrix_a = make_matrix_from_quat(xf_a.q);
    {
        let manifold = &mut contact.manifolds[0];
        debug_assert!(
            0 < geom_manifold.point_count
                && geom_manifold.point_count as usize <= MAX_MANIFOLD_POINTS
        );
        manifold.point_count = geom_manifold.point_count;
        manifold.normal = mul_mv(matrix_a, geom_manifold.normal);

        for i in 0..geom_manifold.point_count as usize {
            let source = &geom_manifold.points[i];
            let target = &mut manifold.points[i];
            target.anchor_a = mul_mv(matrix_a, source.point);
            target.anchor_b = add(target.anchor_a, sub_pos(xf_a.p, xf_b.p));
            target.separation = source.separation;
            target.feature_id = make_feature_id(source.pair);
            target.triangle_index = NULL_INDEX;
            target.normal_velocity = 0.0;
        }

        for i in 0..geom_manifold.point_count as usize {
            let pt2_feature = manifold.points[i].feature_id;
            let mut matched_impulse = 0.0;
            let mut persisted = false;

            for old in old_points.iter_mut().take(old_count as usize) {
                if pt2_feature == old.feature_id {
                    matched_impulse = old.normal_impulse;
                    persisted = true;
                    // Claimed — prevent double match (C: featureId = UINT32_MAX).
                    old.feature_id = u32::MAX;
                    break;
                }
            }

            let pt2 = &mut manifold.points[i];
            pt2.total_normal_impulse = 0.0;
            pt2.persisted = persisted;
            pt2.normal_impulse = if persisted { matched_impulse } else { 0.0 };
        }
    }

    // Claim matched old feature ids like C (UINT32_MAX); Rust uses a local copy
    // so claiming is unnecessary beyond the first match break above.

    true
}

/// (b3UpdateConvexContact)
///
/// `material_a`/`material_b` are the materials the contact mixes: a shape's first material,
/// or a compound child's own.
fn update_convex_contact(
    nc: &mut NarrowCtx<'_>,
    shape_a: &Shape,
    material_a: &SurfaceMaterial,
    geom_a: ConvexGeometry<'_>,
    xf_a: WorldTransform,
    shape_b: &Shape,
    material_b: &SurfaceMaterial,
    geom_b: ConvexGeometry<'_>,
    xf_b: WorldTransform,
    flip: bool,
) -> bool {
    let touching = compute_convex_manifold(nc, geom_a, xf_a, geom_b, xf_b);

    if !touching {
        debug_assert!(nc.contact.manifolds.is_empty());
        return false;
    }

    debug_assert!(nc.contact.manifold_count() == 1);

    if flip {
        let manifold = &mut nc.contact.manifolds[0];
        manifold.normal = neg(manifold.normal);
        for i in 0..manifold.point_count as usize {
            let mp = &mut manifold.points[i];
            std::mem::swap(&mut mp.anchor_a, &mut mp.anchor_b);
        }
    }

    let friction_cb = nc.params.friction_callback;
    let restitution_cb = nc.params.restitution_callback;

    nc.contact.friction = friction_cb(
        material_a.friction,
        material_a.user_material_id,
        material_b.friction,
        material_b.user_material_id,
    );
    nc.contact.restitution = restitution_cb(
        material_a.restitution,
        material_a.user_material_id,
        material_b.restitution,
        material_b.user_material_id,
    );

    if material_a.rolling_resistance > 0.0 || material_b.rolling_resistance > 0.0 {
        let radius_a = shape_rolling_radius(geom_a);
        let radius_b = shape_rolling_radius(geom_b);
        let max_radius = max_float(radius_a, radius_b);
        nc.contact.rolling_resistance =
            max_float(material_a.rolling_resistance, material_b.rolling_resistance) * max_radius;
    } else {
        nc.contact.rolling_resistance = 0.0;
    }

    let tangent_velocity_a = rotate_vector(xf_a.q, material_a.tangent_velocity);
    let tangent_velocity_b = rotate_vector(xf_b.q, material_b.tangent_velocity);
    nc.contact.tangent_velocity = sub(tangent_velocity_a, tangent_velocity_b);

    if let Some(pre_solve) = nc.params.pre_solve_fcn {
        if (nc.contact.flags & contact_flags::SIM_ENABLE_PRE_SOLVE_EVENTS) != 0 {
            let ctx = nc.params.pre_solve_context;
            let world_id = nc.params.world_id;
            let shape_id_a = ShapeId {
                index1: shape_a.id + 1,
                world0: world_id,
                generation: shape_a.generation,
            };
            let shape_id_b = ShapeId {
                index1: shape_b.id + 1,
                world0: world_id,
                generation: shape_b.generation,
            };
            let point = offset_pos(xf_a.p, nc.contact.manifolds[0].points[0].anchor_a);
            let normal = nc.contact.manifolds[0].normal;
            let still_touching = pre_solve(shape_id_a, shape_id_b, point, normal, ctx);
            if !still_touching {
                nc.contact.manifolds.clear();
                return false;
            }
        }
    }

    if (shape_a.flags & shape_flags::ENABLE_HIT_EVENTS) != 0
        || (shape_b.flags & shape_flags::ENABLE_HIT_EVENTS) != 0
    {
        nc.contact.flags |= contact_flags::SIM_ENABLE_HIT_EVENT;
    } else {
        nc.contact.flags &= !contact_flags::SIM_ENABLE_HIT_EVENT;
    }

    true
}

/// Update the contact manifold and touching status. (b3UpdateContact)
pub fn update_contact(
    world: &mut World,
    worker_index: i32,
    contact_id: i32,
    shape_id_a: i32,
    local_center_a: crate::math_functions::Vec3,
    xf_a: WorldTransform,
    shape_id_b: i32,
    local_center_b: crate::math_functions::Vec3,
    xf_b: WorldTransform,
    is_fast: bool,
) -> bool {
    let params = ContactParams::new(world);
    let mut stats = CollideStats::default();
    let mut local = LocalManifold::default();
    let mut nc = NarrowCtx {
        contact: &mut world.contacts[contact_id as usize],
        stats: &mut stats,
        params: &params,
        local: &mut local,
    };
    let touching = update_contact_in(
        &mut nc,
        &world.shapes,
        shape_id_a,
        local_center_a,
        xf_a,
        shape_id_b,
        local_center_b,
        xf_b,
        is_fast,
    );
    let task = &mut world.task_contexts[worker_index as usize];
    task.sat_call_count += stats.sat_call_count;
    task.sat_cache_hit_count += stats.sat_cache_hit_count;
    touching
}

/// [`update_contact`] on split borrows: the contact and counters it writes, the shapes it
/// reads.
pub(crate) fn update_contact_in(
    nc: &mut NarrowCtx<'_>,
    shapes: &[Shape],
    shape_id_a: i32,
    local_center_a: crate::math_functions::Vec3,
    xf_a: WorldTransform,
    shape_id_b: i32,
    local_center_b: crate::math_functions::Vec3,
    xf_b: WorldTransform,
    is_fast: bool,
) -> bool {
    let shape_a = &shapes[shape_id_a as usize];
    let shape_b = &shapes[shape_id_b as usize];
    debug_assert!(shape_b.shape_type() != ShapeType::Compound);

    let type_a = shape_a.shape_type();

    let touching = if type_a == ShapeType::Compound {
        update_compound_contact(nc, shape_a, xf_a, shape_b, xf_b, is_fast)
    } else if type_a == ShapeType::Mesh || type_a == ShapeType::Height {
        let touching = compute_mesh_manifolds(nc, shape_a, None, xf_a, shape_b, xf_b, is_fast);
        apply_mesh_hit_flags(nc, shape_a, shape_b);
        debug_assert!(
            (touching && !nc.contact.manifolds.is_empty())
                || (!touching && nc.contact.manifolds.is_empty())
        );
        touching
    } else {
        update_convex_contact(
            nc,
            shape_a,
            shape_a.get_material(0),
            ConvexGeometry::of(&shape_a.geometry),
            xf_a,
            shape_b,
            shape_b.get_material(0),
            ConvexGeometry::of(&shape_b.geometry),
            xf_b,
            false,
        )
    };

    if touching {
        let center_a = rotate_vector(xf_a.q, local_center_a);
        let center_b = rotate_vector(xf_b.q, local_center_b);
        let contact = &mut *nc.contact;
        for manifold in &mut contact.manifolds {
            for j in 0..manifold.point_count as usize {
                let mp = &mut manifold.points[j];
                mp.anchor_a = sub(mp.anchor_a, center_a);
                mp.anchor_b = sub(mp.anchor_b, center_b);
            }
        }
        contact.flags |= contact_flags::SIM_TOUCHING;
    } else {
        nc.contact.flags &= !contact_flags::SIM_TOUCHING;
    }

    touching
}

fn update_compound_contact(
    nc: &mut NarrowCtx<'_>,
    shape_a: &Shape,
    xf_a: WorldTransform,
    shape_b: &Shape,
    xf_b: WorldTransform,
    is_fast: bool,
) -> bool {
    use crate::compound::{get_compound_child, ChildGeometry};

    let child_index = nc.contact.child_index;
    let geom_b = ConvexGeometry::of(&shape_b.geometry);
    let type_b = geom_b.shape_type();
    let material_b = shape_b.get_material(0);

    let ShapeGeometry::Compound(compound) = &shape_a.geometry else {
        unreachable!()
    };
    let child = get_compound_child(compound, child_index);
    let child_transform = child.transform;
    let material_indices = child.material_indices;

    // Handle child material for non-meshes. The convex kernels below read the struck
    // child's material, not entry 0 of the compound material table. Matches childShapeA
    // setup in b3UpdateContact. The mesh branch keeps the compound materials.
    let child_material = if child.shape_type != ShapeType::Mesh {
        debug_assert!((material_indices[0] as i32) < shape_a.material_count());
        &shape_a.shape_materials()[material_indices[0] as usize]
    } else {
        shape_a.get_material(0)
    };

    let touching = match child.geometry {
        ChildGeometry::Capsule(c) => {
            let child_geom = ConvexGeometry::Capsule(&c);
            let flip = type_b == ShapeType::Hull;
            if flip {
                update_convex_contact(
                    nc,
                    shape_b,
                    material_b,
                    geom_b,
                    xf_b,
                    shape_a,
                    child_material,
                    child_geom,
                    xf_a,
                    true,
                )
            } else {
                update_convex_contact(
                    nc,
                    shape_a,
                    child_material,
                    child_geom,
                    xf_a,
                    shape_b,
                    material_b,
                    geom_b,
                    xf_b,
                    false,
                )
            }
        }
        ChildGeometry::Hull(h) => {
            let xf_child = mul_world_transforms(xf_a, child_transform);
            update_convex_contact(
                nc,
                shape_a,
                child_material,
                ConvexGeometry::Hull(h),
                xf_child,
                shape_b,
                material_b,
                geom_b,
                xf_b,
                false,
            )
        }
        ChildGeometry::Sphere(s) => {
            let child_geom = ConvexGeometry::Sphere(&s);
            let flip = type_b == ShapeType::Capsule || type_b == ShapeType::Hull;
            if flip {
                update_convex_contact(
                    nc,
                    shape_b,
                    material_b,
                    geom_b,
                    xf_b,
                    shape_a,
                    child_material,
                    child_geom,
                    xf_a,
                    true,
                )
            } else {
                update_convex_contact(
                    nc,
                    shape_a,
                    child_material,
                    child_geom,
                    xf_a,
                    shape_b,
                    material_b,
                    geom_b,
                    xf_b,
                    false,
                )
            }
        }
        ChildGeometry::Mesh(mesh) => {
            // The mesh kernel reads the child mesh through a shape, so this path still
            // builds a temporary child shape. Compound mesh children are rare.
            let mut child_shape = shape_a.clone();
            child_shape.geometry = ShapeGeometry::Mesh {
                data: mesh.data.clone(),
                scale: mesh.scale,
            };
            let xf_child = mul_world_transforms(xf_a, child_transform);
            let touching = compute_mesh_manifolds(
                nc,
                &child_shape,
                Some(material_indices),
                xf_child,
                shape_b,
                xf_b,
                is_fast,
            );
            apply_mesh_hit_flags(nc, shape_a, shape_b);
            debug_assert!(
                (touching && !nc.contact.manifolds.is_empty())
                    || (!touching && nc.contact.manifolds.is_empty())
            );
            touching
        }
    };

    if touching {
        let offset = rotate_vector(xf_a.q, child_transform.p);
        for manifold in &mut nc.contact.manifolds {
            for j in 0..manifold.point_count as usize {
                manifold.points[j].anchor_a = add(manifold.points[j].anchor_a, offset);
            }
        }
    }

    touching
}
