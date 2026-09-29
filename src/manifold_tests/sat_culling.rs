//! Coverage for the inner-sphere culling added to the separating axis test by upstream
//! adec25b ("SAT optimizations"): `b3ArcCanReach`, the early return when either hull has no
//! edge candidates, and the full-SAT fallback in `b3CollideHulls` when face clipping yields
//! no points.
//!
//! Every expected value below comes from the scalar C build (`BOX3D_DISABLE_SIMD`) of the
//! reference at adec25b, printed as raw float bits. The hull pair and transform in the last
//! two tests were found by searching randomized pairs for the rare paths.
//!
//! SPDX-FileCopyrightText: 2026 Erin Catto
//! SPDX-License-Identifier: MIT

use crate::core::NULL_INDEX;
use crate::hull::{create_hull, make_box_hull};
use crate::manifold::{
    arc_can_reach, collide_hulls, compute_separating_axis, LocalManifold, SatCache,
    SeparatingFeature,
};
use crate::math_functions::{Quat, Transform, Vec3};

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 { x, y, z }
}

fn vb(x: u32, y: u32, z: u32) -> Vec3 {
    v(f32::from_bits(x), f32::from_bits(y), f32::from_bits(z))
}

fn assert_vec_bits(actual: Vec3, expected: Vec3) {
    assert_eq!(
        actual.x.to_bits(),
        expected.x.to_bits(),
        "x {actual:?} {expected:?}"
    );
    assert_eq!(
        actual.y.to_bits(),
        expected.y.to_bits(),
        "y {actual:?} {expected:?}"
    );
    assert_eq!(
        actual.z.to_bits(),
        expected.z.to_bits(),
        "z {actual:?} {expected:?}"
    );
}

// Arguments are (a, b, c, length, bound). Each case isolates one condition of the test.
#[test]
fn arc_can_reach_conditions() {
    // Endpoint: face normal u already beats the bound.
    assert_eq!(arc_can_reach(0.9, 0.1, 0.5, 1.0, 0.8), 1);
    // Endpoint: face normal v already beats the bound.
    assert_eq!(arc_can_reach(0.1, 0.9, 0.5, 1.0, 0.8), 1);
    // Interior: neither endpoint reaches, but the arc peak t / s does.
    assert_eq!(arc_can_reach(0.7071, 0.7071, 0.0, 1.0, 0.9), 1);
    // Interior peak below the bound.
    assert_eq!(arc_can_reach(0.6, 0.6, 0.0, 1.0, 0.9), 0);
    // Center distance below the bound.
    assert_eq!(arc_can_reach(0.7071, 0.7071, 0.0, 0.85, 0.9), 0);
    // Peak projection falls outside the arc: a < c * b.
    assert_eq!(arc_can_reach(0.1, 0.8, 0.5, 1.0, 0.85), 0);
    // Peak projection falls outside the arc: b < c * a.
    assert_eq!(arc_can_reach(0.8, 0.1, 0.5, 1.0, 0.85), 0);
    // Near parallel normals (s < B3_PARALLEL_TOL) pass the candidate on even though t < bound^2 s.
    assert_eq!(arc_can_reach(0.5, 0.5, 0.99999, 1.0, 0.9), 1);
    // Non-positive bound.
    assert_eq!(arc_can_reach(-0.5, -0.5, 1.0, 1.0, -0.1), 1);
    // Without early return the bound is -infinity and every edge is a candidate.
    assert_eq!(arc_can_reach(-0.5, -0.4, 0.3, 1.0, f32::NEG_INFINITY), 1);
}

// A tall pyramid rests its tip on a large rotated cube, just inside the speculative distance.
// The pyramid side normals are nearly perpendicular to the center delta, so none of its edge
// arcs can beat the cube face separation: na == 0 and the query returns before the edge phase.
// The same pair without early return does find an edge pair, which the culling skipped.
#[test]
fn separating_axis_no_edge_candidates() {
    let points = [
        v(0.5, 0.0, 0.5),
        v(-0.5, 0.0, 0.5),
        v(-0.5, 0.0, -0.5),
        v(0.5, 0.0, -0.5),
        v(0.0, 6.0, 0.0),
    ];
    let pyramid = create_hull(&points, 128).expect("pyramid");
    assert_eq!(pyramid.vertex_count, 5);
    assert_eq!(pyramid.edge_count, 16);
    assert_eq!(pyramid.face_count, 5);
    assert_eq!(pyramid.inner_radius.to_bits(), 0x3ebf5638);

    let cube = make_box_hull(2.0, 2.0, 2.0);
    let xf = Transform {
        p: v(0.0, 8.01, 0.0),
        q: Quat {
            v: v(0.0, f32::from_bits(0x3e931f43), 0.0),
            s: f32::from_bits(0x3f75341a),
        },
    };

    for early_return in [true, false] {
        let q = compute_separating_axis(&pyramid, &cube.base, xf, early_return);
        assert_eq!(q.separated_feature, SeparatingFeature::InvalidAxis);

        assert_vec_bits(q.face_a.normal, vb(0xbf7f1da0, 0x3daa13c0, 0x00000000));
        assert_eq!(q.face_a.separation.to_bits(), 0xc030a7cb);
        assert_eq!((q.face_a.index_a, q.face_a.index_b), (0, 3));
        assert_eq!(q.face_a.type_, SeparatingFeature::FaceAxisA);

        assert_vec_bits(q.face_b.normal, vb(0x80000000, 0x3f800000, 0x00000000));
        assert_eq!(q.face_b.separation.to_bits(), 0x3c23d800);
        assert_eq!((q.face_b.index_a, q.face_b.index_b), (1, 2));
        assert_eq!(q.face_b.type_, SeparatingFeature::FaceAxisB);

        if early_return {
            // No edge candidates on the pyramid: the edge phase never ran.
            assert_eq!(q.edge.index_a, NULL_INDEX);
            assert_eq!(q.edge.index_b, NULL_INDEX);
            assert_eq!(q.edge.separation, f32::NEG_INFINITY);
            assert_vec_bits(q.edge.normal, vb(0, 0, 0));
        } else {
            assert_vec_bits(q.edge.normal, vb(0x3f54508e, 0x3deade46, 0xbf0bfcd9));
            assert_eq!(q.edge.separation.to_bits(), 0xbffe2a06);
            assert_eq!((q.edge.index_a, q.edge.index_b), (12, 6));
        }
        assert_eq!(q.edge.type_, SeparatingFeature::EdgePairAxis);
    }
}

// Both face axes are inside the speculative distance but face clipping of this pair yields no
// points, and the early-return query culled every edge candidate. b3CollideHulls must rerun
// the full SAT to recover the best edge pair and build the edge contact from it.
#[test]
fn collide_hulls_full_sat_fallback_builds_edge_contact() {
    let points_a = [
        v(0.202, 0.034, -0.87100005),
        v(-0.31100002, -0.425, -0.39900002),
        v(0.025, 0.36400002, -0.15),
        v(-0.049000002, 0.76600003, 0.47200003),
        v(-0.597, 0.321, -0.55),
    ];
    let points_b = [
        v(0.42200002, 0.929, 0.47400004),
        v(-0.544, 0.78200006, -0.786),
        v(-0.29200003, 0.551, 0.43300003),
        v(-0.28500003, 0.966, 0.648),
        v(0.52900004, -0.46400002, 0.393),
        v(0.49800003, -0.016, -0.284),
        v(-0.79800004, 0.93000007, -0.439),
        v(-0.67600006, 1.0, -0.105000004),
    ];
    let hull_a = create_hull(&points_a, 128).expect("hull A");
    let hull_b = create_hull(&points_b, 128).expect("hull B");
    assert_eq!(hull_a.vertex_count, 4);
    assert_eq!(hull_b.vertex_count, 8);

    let xf = Transform {
        p: vb(0x3eb2f1aa, 0xbe400000, 0xbd6353f8),
        q: Quat {
            v: vb(0xbef06346, 0x3f252cf1, 0x3f178464),
            s: f32::from_bits(0xbde91a73),
        },
    };

    // The early-return query has no edge pair.
    let early = compute_separating_axis(&hull_a, &hull_b, xf, true);
    assert_eq!(early.separated_feature, SeparatingFeature::InvalidAxis);
    assert_eq!(early.face_a.separation.to_bits(), 0x3c9cff40);
    assert_eq!((early.face_a.index_a, early.face_a.index_b), (3, 7));
    assert_eq!(early.face_b.separation.to_bits(), 0x3c957328);
    assert_eq!((early.face_b.index_a, early.face_b.index_b), (3, 1));
    assert_eq!(early.edge.index_a, NULL_INDEX);

    // The full query finds it.
    let full = compute_separating_axis(&hull_a, &hull_b, xf, false);
    assert_vec_bits(full.edge.normal, vb(0xbf63dedf, 0xbe8866fd, 0x3ebd4f72));
    assert_eq!(full.edge.separation.to_bits(), 0xbef21d0e);
    assert_eq!((full.edge.index_a, full.edge.index_b), (6, 32));

    let mut manifold = LocalManifold::default();
    let mut cache = SatCache::default();
    collide_hulls(&mut manifold, 8, &hull_a, &hull_b, xf, &mut cache);

    assert_eq!(manifold.point_count, 1);
    assert_vec_bits(manifold.normal, vb(0xbf63dedf, 0xbe8866fd, 0x3ebd4f72));
    assert_eq!(cache.separation.to_bits(), 0xbef21d0d);
    assert_eq!(cache.type_, SeparatingFeature::EdgePairAxis as u8);
    assert_eq!((cache.index_a, cache.index_b, cache.hit), (6, 32, 0));

    let point = &manifold.points[0];
    assert_vec_bits(point.point, vb(0xbde28c54, 0xbeabe3d0, 0xbefbc5d1));
    assert_eq!(point.separation.to_bits(), 0xbef21d0d);
    assert_eq!(
        (
            point.pair.owner1,
            point.pair.index1,
            point.pair.owner2,
            point.pair.index2
        ),
        (0, 6, 1, 32)
    );
}
