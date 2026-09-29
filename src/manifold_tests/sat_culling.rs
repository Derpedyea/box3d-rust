//! Coverage for the inner-sphere culling in the separating axis test (upstream adec25b "SAT
//! optimizations", revised in 9f998c8 "SAT samples"): `b3TestEdgeCandidate`, the early return
//! when either hull has no edge candidates, and a pair that needed the adec25b full-SAT
//! fallback in `b3CollideHulls` (removed in 9f998c8), plus both outcomes of the 9f998c8
//! edge contact condition when face clipping yields no points.
//!
//! Every expected value below comes from the scalar C build (`BOX3D_DISABLE_SIMD`) of the
//! reference at 9f998c8, printed as raw float bits. The hull pairs and transforms in the
//! collide tests were found by searching randomized pairs for the rare paths.
//!
//! SPDX-FileCopyrightText: 2026 Erin Catto
//! SPDX-License-Identifier: MIT

use crate::core::NULL_INDEX;
use crate::hull::{create_hull, make_box_hull};
use crate::manifold::{
    collide_hulls, compute_separating_axis, test_edge_candidate, LocalManifold, SatCache,
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

// Arguments are (a1, a2, c, bound). Each case exercises one condition of the test.
#[test]
fn test_edge_candidate_conditions() {
    // Exterior: face normal n1 already beats the bound.
    assert_eq!(test_edge_candidate(0.9, 0.1, 0.5, 0.8), 1);
    // Exterior: face normal n2 already beats the bound.
    assert_eq!(test_edge_candidate(0.1, 0.9, 0.5, 0.8), 1);
    // Interior: neither endpoint reaches, but the arc peak t / s does.
    assert_eq!(test_edge_candidate(0.7071, 0.7071, 0.0, 0.9), 1);
    // Interior peak below the bound.
    assert_eq!(test_edge_candidate(0.6, 0.6, 0.0, 0.9), 0);
    // Peak projection falls outside the arc: a1 < c * a2.
    assert_eq!(test_edge_candidate(0.1, 0.8, 0.5, 0.85), 0);
    // Peak projection falls outside the arc: a2 < c * a1.
    assert_eq!(test_edge_candidate(0.8, 0.1, 0.5, 0.85), 0);
    // Near parallel normals (s < B3_PARALLEL_TOL) pass the candidate on even though t < bound^2 s.
    assert_eq!(test_edge_candidate(0.5, 0.5, 0.99999, 0.9), 1);
    // Non-positive bound. With the exterior test failing, the interior test can only pass when
    // c >= 1, which also makes s < tol, so this case cannot isolate the bound <= 0 term.
    assert_eq!(test_edge_candidate(-0.5, -0.5, 1.0, -0.1), 1);
    // Without early return the bound is -infinity and every edge is a candidate.
    assert_eq!(test_edge_candidate(-0.5, -0.4, 0.3, f32::NEG_INFINITY), 1);
}

// A tall pyramid rests its tip on a large rotated cube, just inside the speculative distance.
// The pyramid side normals are nearly perpendicular to the center delta, so none of its edge
// arcs can beat the cube face separation, so the early-return query culls its edges and
// reports no edge pair. The same pair without early return does find an edge pair.
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

// Both face axes are inside the speculative distance and face clipping of this pair yields no
// points. At adec25b the early-return query culled every edge candidate, so b3CollideHulls
// reran a full SAT and built a deep edge contact (separation -0.47) for hulls that are
// actually apart. 9f998c8 removes that fallback; with its edge candidate test and the
// -linearSlop^2 Gauss map tolerance the early query itself finds the separating edge axis
// beyond the speculative distance, so no contact is produced.
#[test]
fn collide_hulls_separating_edge_axis_without_fallback() {
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

    // The early-return query separates on the edge pair.
    let early = compute_separating_axis(&hull_a, &hull_b, xf, true);
    assert_eq!(early.separated_feature, SeparatingFeature::EdgePairAxis);
    assert_vec_bits(early.face_a.normal, vb(0x3f49a22d, 0xbeef2928, 0x3ecdb908));
    assert_eq!(early.face_a.separation.to_bits(), 0x3c9cff40);
    assert_eq!((early.face_a.index_a, early.face_a.index_b), (3, 7));
    assert_vec_bits(early.face_b.normal, vb(0x3f4a58e2, 0xbeef6e38, 0x3eca94f7));
    assert_eq!(early.face_b.separation.to_bits(), 0x3c957328);
    assert_eq!((early.face_b.index_a, early.face_b.index_b), (3, 1));
    assert_vec_bits(early.edge.normal, vb(0x3f498c06, 0xbef1c119, 0x3ecb033c));
    assert_eq!(early.edge.separation.to_bits(), 0x3ca5bcc0);
    assert_eq!((early.edge.index_a, early.edge.index_b), (10, 6));

    // The full query agrees on the best edge pair.
    let full = compute_separating_axis(&hull_a, &hull_b, xf, false);
    assert_vec_bits(full.edge.normal, vb(0x3f498c06, 0xbef1c119, 0x3ecb033c));
    assert_eq!(full.edge.separation.to_bits(), 0x3ca5bcc0);
    assert_eq!((full.edge.index_a, full.edge.index_b), (10, 6));

    let mut manifold = LocalManifold::default();
    let mut cache = SatCache::default();
    collide_hulls(&mut manifold, 8, &hull_a, &hull_b, xf, &mut cache);

    assert_eq!(manifold.point_count, 0);
    assert_vec_bits(manifold.normal, vb(0, 0, 0));
    assert_eq!(cache.separation.to_bits(), 0x3ca5bcc0);
    assert_eq!(cache.type_, SeparatingFeature::EdgePairAxis as u8);
    assert_eq!((cache.index_a, cache.index_b, cache.hit), (10, 6, 0));
}

/// Transform from raw bits: translation, quaternion vector, quaternion scalar.
fn transform_bits(bits: [u32; 7]) -> Transform {
    Transform {
        p: vb(bits[0], bits[1], bits[2]),
        q: Quat {
            v: vb(bits[3], bits[4], bits[5]),
            s: f32::from_bits(bits[6]),
        },
    }
}

// Face clipping yields no points. The edge separation (-0.034) beats both face separations
// (-0.685, -0.295) but not the clip separation by the linear slop, so only the first clause of
// the 9f998c8 edge contact condition builds the edge contact.
#[test]
fn collide_hulls_empty_clip_edge_beats_faces_builds_edge_contact() {
    let points_a = [
        v(0.155, -0.097, 0.675),
        v(-0.21200001, -0.08800001, 0.611),
        v(-0.92300004, -0.07, 0.57100004),
        v(0.994, -0.78300005, -0.513),
        v(0.091000006, 0.731, 0.94000006),
    ];
    let points_b = [
        v(-0.351, -0.19500001, -0.086),
        v(-0.669, -0.93500006, -0.989),
        v(0.79300004, -0.84200007, 0.739),
        v(-0.108, 0.9110001, -0.89000005),
        v(0.87700003, 0.28800002, 0.85700005),
        v(0.351, -0.384, -0.13100001),
    ];
    let hull_a = create_hull(&points_a, 128).expect("hull A");
    let hull_b = create_hull(&points_b, 128).expect("hull B");
    assert_eq!((hull_a.vertex_count, hull_a.edge_count), (4, 12));
    assert_eq!((hull_b.vertex_count, hull_b.edge_count), (6, 24));
    let xf = transform_bits([
        0x3ee4dd30, 0xbf42b022, 0xbf956042, 0x3d901380, 0xbec9f15d, 0x3eb38137, 0x3f58b469,
    ]);

    let q = compute_separating_axis(&hull_a, &hull_b, xf, true);
    assert_eq!(q.separated_feature, SeparatingFeature::InvalidAxis);
    assert_eq!(q.face_a.separation.to_bits(), 0xbf2f669d);
    assert_eq!((q.face_a.index_a, q.face_a.index_b), (3, 1));
    assert_eq!(q.face_b.separation.to_bits(), 0xbe96cfeb);
    assert_eq!((q.face_b.index_a, q.face_b.index_b), (3, 4));
    assert_vec_bits(q.edge.normal, vb(0xbf0eb895, 0xbf09ff5e, 0xbf21a098));
    assert_eq!(q.edge.separation.to_bits(), 0xbd0bb9f8);
    assert_eq!((q.edge.index_a, q.edge.index_b), (10, 12));

    let mut manifold = LocalManifold::default();
    let mut cache = SatCache::default();
    collide_hulls(&mut manifold, 8, &hull_a, &hull_b, xf, &mut cache);

    assert_eq!(manifold.point_count, 1);
    assert_vec_bits(manifold.normal, vb(0xbf0eb895, 0xbf09ff5e, 0xbf21a098));
    assert_eq!(cache.separation.to_bits(), 0xbd0bb9fd);
    assert_eq!(cache.type_, SeparatingFeature::EdgePairAxis as u8);
    assert_eq!((cache.index_a, cache.index_b, cache.hit), (10, 12, 0));

    let point = &manifold.points[0];
    assert_vec_bits(point.point, vb(0x3e0a338c, 0xbee6ca58, 0xbc35cf82));
    assert_eq!(point.separation.to_bits(), 0xbd0bb9fd);
    assert_eq!(
        (
            point.pair.owner1,
            point.pair.index1,
            point.pair.owner2,
            point.pair.index2
        ),
        (0, 10, 1, 12)
    );
}

// Face clipping yields no points and the edge separation (0.00085) does not beat the best face
// separation (0.0030), so 9f998c8 returns an empty manifold. adec25b built an edge contact
// whenever face clipping was empty. The empty face clip leaves the cache cleared.
#[test]
fn collide_hulls_empty_clip_edge_below_faces_returns_no_points() {
    let points_a = [
        v(0.76500005, -0.92300004, -0.776),
        v(-0.60300004, 0.86100006, -0.85200006),
        v(0.614, -0.64400005, 0.90500003),
        v(0.72900003, -0.555, 0.54300004),
    ];
    let points_b = [
        v(-0.157, 0.509, 0.68700004),
        v(-0.41900003, 0.16700001, -0.67),
        v(-0.41400003, -0.377, 0.115),
        v(-0.254, 0.71300006, 0.32900003),
        v(-0.583, -0.404, -0.859),
    ];
    let hull_a = create_hull(&points_a, 128).expect("hull A");
    let hull_b = create_hull(&points_b, 128).expect("hull B");
    assert_eq!((hull_a.vertex_count, hull_a.edge_count), (4, 12));
    assert_eq!((hull_b.vertex_count, hull_b.edge_count), (5, 18));
    let xf = transform_bits([
        0x3d449ba6, 0xbf8ced92, 0xbfbed917, 0x3f2f2192, 0x3e9f0d56, 0xbf18f7d8, 0x3e8f685a,
    ]);

    let q = compute_separating_axis(&hull_a, &hull_b, xf, true);
    assert_eq!(q.separated_feature, SeparatingFeature::InvalidAxis);
    assert_vec_bits(q.face_a.normal, vb(0xbf4b3725, 0xbf1b81c9, 0x3cf1c73b));
    assert_eq!(q.face_a.separation.to_bits(), 0x3b42f400);
    assert_eq!((q.face_a.index_a, q.face_a.index_b), (3, 4));
    assert_eq!(q.face_b.separation.to_bits(), 0xbe86b652);
    assert_eq!((q.face_b.index_a, q.face_b.index_b), (1, 5));
    assert_vec_bits(q.edge.normal, vb(0x3e88724f, 0x3e27d34f, 0xbf73260d));
    assert_eq!(q.edge.separation.to_bits(), 0x3a5e8000);
    assert_eq!((q.edge.index_a, q.edge.index_b), (10, 12));

    let mut manifold = LocalManifold::default();
    let mut cache = SatCache::default();
    collide_hulls(&mut manifold, 8, &hull_a, &hull_b, xf, &mut cache);

    assert_eq!(manifold.point_count, 0);
    assert_vec_bits(manifold.normal, vb(0, 0, 0));
    assert_eq!(cache.separation.to_bits(), 0);
    assert_eq!(cache.type_, SeparatingFeature::InvalidAxis as u8);
    assert_eq!((cache.index_a, cache.index_b, cache.hit), (0, 0, 0));
}
