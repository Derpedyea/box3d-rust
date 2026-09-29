// AABB query and closest query from dynamic_tree.c.
//
// The C callbacks take a `void* context`; the Rust versions take closures, which capture
// their context directly.
//
// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

use super::aabbv::overlap_node;
use super::{
    get_root_pair, test_category, DynamicTree, TreeNode, TreeStats, ROOT_NODE, TREE_STACK_SIZE,
};
use crate::math_functions::{clamp, dot, sub, Aabb, Vec3};

/// Squared distance from a point to a node AABB. (b3DistanceToNodeSqr)
#[inline]
fn distance_to_node_sqr(point: Vec3, node: &TreeNode) -> f32 {
    let r = sub(
        point,
        clamp(point, node.aabb.lower_bound, node.aabb.upper_bound),
    );
    dot(r, r)
}

/// (struct b3QueryClosestItem)
#[derive(Clone, Copy)]
struct QueryClosestItem {
    node_index: i32,
    distance_to_node_sqr: f32,
}

impl DynamicTree {
    /// Query an AABB for overlapping proxies. The callback is called for each proxy that
    /// overlaps the supplied AABB and passes the mask-bits filter; return false from the
    /// callback to stop. (b3DynamicTree_Query)
    pub fn query(
        &self,
        aabb: Aabb,
        mask_bits: u64,
        require_all_bits: bool,
        mut callback: impl FnMut(i32, u64) -> bool,
    ) -> TreeStats {
        let mut result = TreeStats::default();

        if self.proxy_count == 0 {
            return result;
        }

        let nodes = &self.nodes;

        let mut stack = [0i32; TREE_STACK_SIZE];
        let mut stack_count = 0usize;
        stack[stack_count] = get_root_pair(nodes);
        stack_count += 1;

        let boxv = aabb;

        while stack_count > 0 {
            stack_count -= 1;
            let pair = stack[stack_count];
            result.node_visits += 1;

            for i in 0..2 {
                let node = &nodes[pair as usize + i];
                if !overlap_node(&boxv, node) {
                    continue;
                }

                if node.is_leaf() {
                    let proxy_id = node.proxy_id();
                    let proxy = &self.proxies[proxy_id as usize];
                    if test_category(proxy.category_bits, mask_bits, require_all_bits) {
                        let proceed = callback(proxy_id, proxy.user_data);
                        result.leaf_visits += 1;

                        if !proceed {
                            return result;
                        }
                    }
                } else if stack_count < TREE_STACK_SIZE - 1 {
                    stack[stack_count] = node.left_child();
                    stack_count += 1;
                } else {
                    debug_assert!(stack_count < TREE_STACK_SIZE - 1);
                }
            }
        }

        result
    }

    /// Query for the closest proxy to a point. The callback receives the current minimum
    /// squared distance and returns an updated distance for that proxy. `min_distance_sqr`
    /// is the initial and final minimum squared distance.
    /// (b3DynamicTree_QueryClosest)
    pub fn query_closest(
        &self,
        point: Vec3,
        mask_bits: u64,
        require_all_bits: bool,
        mut callback: impl FnMut(f32, i32, u64) -> f32,
        min_distance_sqr: &mut f32,
    ) -> TreeStats {
        let mut result = TreeStats::default();

        if self.proxy_count == 0 {
            return result;
        }

        let nodes = &self.nodes;

        let mut min_sqr = *min_distance_sqr;
        let mut stack = [QueryClosestItem {
            node_index: 0,
            distance_to_node_sqr: 0.0,
        }; TREE_STACK_SIZE];
        let mut stack_count = 0usize;

        let root_pair = get_root_pair(nodes);
        let seed1 = QueryClosestItem {
            node_index: root_pair,
            distance_to_node_sqr: distance_to_node_sqr(point, &nodes[root_pair as usize]),
        };

        if root_pair == ROOT_NODE {
            stack[stack_count] = seed1;
            stack_count += 1;
        } else {
            let seed2 = QueryClosestItem {
                node_index: root_pair + 1,
                distance_to_node_sqr: distance_to_node_sqr(point, &nodes[root_pair as usize + 1]),
            };

            // Ensure we iterate the closest child first as we pop off the stack
            if seed2.distance_to_node_sqr < seed1.distance_to_node_sqr {
                stack[stack_count] = seed1;
                stack_count += 1;
                stack[stack_count] = seed2;
                stack_count += 1;
            } else {
                stack[stack_count] = seed2;
                stack_count += 1;
                stack[stack_count] = seed1;
                stack_count += 1;
            }
        }

        while stack_count > 0 {
            stack_count -= 1;
            let item = stack[stack_count];
            let node = &nodes[item.node_index as usize];
            result.node_visits += 1;

            if item.distance_to_node_sqr >= min_sqr {
                continue;
            }

            if node.is_leaf() {
                let proxy_id = node.proxy_id();
                let proxy = &self.proxies[proxy_id as usize];
                if !test_category(proxy.category_bits, mask_bits, require_all_bits) {
                    continue;
                }

                // callback to user code with minimum distance squared so far and proxy id
                let dd = callback(min_sqr, proxy_id, proxy.user_data);

                if dd < min_sqr {
                    min_sqr = dd;
                }

                result.leaf_visits += 1;
                continue;
            }

            if stack_count + 2 > TREE_STACK_SIZE {
                debug_assert!(stack_count + 2 <= TREE_STACK_SIZE);
                continue;
            }

            let pair = node.left_child();

            // Store the distance to node in the stack instead of recomputing after pop
            let item1 = QueryClosestItem {
                node_index: pair,
                distance_to_node_sqr: distance_to_node_sqr(point, &nodes[pair as usize]),
            };

            let item2 = QueryClosestItem {
                node_index: pair + 1,
                distance_to_node_sqr: distance_to_node_sqr(point, &nodes[pair as usize + 1]),
            };

            // Ensure we iterate the closest child first as we pop off the stack
            if item2.distance_to_node_sqr < item1.distance_to_node_sqr {
                stack[stack_count] = item1;
                stack_count += 1;
                stack[stack_count] = item2;
                stack_count += 1;
            } else {
                stack[stack_count] = item2;
                stack_count += 1;
                stack[stack_count] = item1;
                stack_count += 1;
            }
        }

        *min_distance_sqr = min_sqr;

        result
    }
}
