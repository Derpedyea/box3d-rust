// The b3AABBV helpers from simd.h. Tree traversal tests use the 4-lane (SSE) path;
// the rest use the scalar (B3_SIMD_NONE) path, where `b3AABBV` is a plain `b3AABB`, `b3LoadAABBV` is the identity, and
// `b3StoreAABBV( aabb, value, condition )` is `if ( condition ) *aabb = value;`, so those
// two helpers have no Rust counterpart; call sites load/store the AABB directly.
//
// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

use super::TreeNode;
use crate::math_functions::{aabb_union, Aabb};
use crate::simd::FloatW;
use wide::CmpLe;

/// (b3OverlapAABBV, scalar path)
#[inline]
pub(crate) fn overlap_aabb_v(a: &Aabb, b: &Aabb) -> bool {
    a.lower_bound.x <= b.upper_bound.x
        && a.lower_bound.y <= b.upper_bound.y
        && a.lower_bound.z <= b.upper_bound.z
        && b.lower_bound.x <= a.upper_bound.x
        && b.lower_bound.y <= a.upper_bound.y
        && b.lower_bound.z <= a.upper_bound.z
}

// The SSE path of b3AABBV: a box's bounds as two 4-lane vectors whose last lane is
// ignored. A node's bounds load straight from memory (the last lane reads the next field).
#[inline]
fn node_lower(n: &TreeNode) -> FloatW {
    let b = &n.aabb;
    FloatW::new([
        b.lower_bound.x,
        b.lower_bound.y,
        b.lower_bound.z,
        b.upper_bound.x,
    ])
}

#[inline]
fn node_upper(n: &TreeNode) -> FloatW {
    let b = &n.aabb;
    let next = f32::from_bits(n.flag_index);
    FloatW::new([b.upper_bound.x, b.upper_bound.y, b.upper_bound.z, next])
}

/// Lanes x, y, z of `a <= b` and of `c <= d` all hold. Like the scalar `<=`, NaN fails.
#[inline]
fn all_le_xyz(a: FloatW, b: FloatW, c: FloatW, d: FloatW) -> bool {
    (a.cmp_le(b) & c.cmp_le(d)).move_mask() & 0b111 == 0b111
}

/// Do two nodes' boxes overlap? (b3OverlapV on loaded nodes)
#[inline]
pub(crate) fn overlap_nodes(a: &TreeNode, b: &TreeNode) -> bool {
    all_le_xyz(node_lower(a), node_upper(b), node_lower(b), node_upper(a))
}

/// (b3OverlapNode)
#[inline]
pub(crate) fn overlap_node(av: &Aabb, node: &TreeNode) -> bool {
    let lower = FloatW::new([av.lower_bound.x, av.lower_bound.y, av.lower_bound.z, 0.0]);
    let upper = FloatW::new([av.upper_bound.x, av.upper_bound.y, av.upper_bound.z, 0.0]);
    all_le_xyz(lower, node_upper(node), node_lower(node), upper)
}

/// (b3OverlapV, scalar path)
#[inline]
pub(crate) fn overlap_v(a: &Aabb, b: &Aabb) -> bool {
    overlap_aabb_v(a, b)
}

/// (b3UnionPairV, scalar path) — the union of the sibling pair starting at `pair`.
#[inline]
pub(crate) fn union_pair_v(nodes: &[TreeNode], pair: i32) -> Aabb {
    aabb_union(nodes[pair as usize].aabb, nodes[pair as usize + 1].aabb)
}

/// (b3UnionV, scalar path: b3UnionAABBV == b3AABB_Union)
#[inline]
pub(crate) fn union_v(a: Aabb, b: Aabb) -> Aabb {
    aabb_union(a, b)
}
