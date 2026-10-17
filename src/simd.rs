//! Wide-float helpers from `simd.h`, on `wide::f32x4`: SSE2 on x86-64, Neon on AArch64,
//! simd128 on wasm, and plain arrays elsewhere.
//!
//! Every helper computes, lane for lane, exactly what the scalar (`B3_SIMD_NONE`) path
//! computes, so results are bit-identical to the scalar reference on every target. Where
//! an SSE2 intrinsic would differ from the scalar C on signed zeros or NaN (`_mm_min_ps`,
//! `andnot` abs, `0 - x` negation) the helper is built from compares and blends instead.
//! The only representation change is the masks: comparisons return all-ones/all-zeros
//! lanes, as the SIMD paths of C do, rather than the scalar path's `1.0`/`0.0`. Callers
//! only consume masks through `blend_w`, `and_w`, `or_w`, and `any_true_w`, which treat
//! both the same way.
//!
//! The wide vec3 type and its helpers (`Vec3W`, `dot_w`, ...) also live here, as they do in
//! upstream `simd.h`. The separating-axis face phase (`manifold::separating_axis`), the hull
//! support vertex search (`hull::find_hull_support_vertex`), and the wide convex contact
//! kernels (`contact_solver::wide`) consume these helpers.
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use wide::{f32x4, i32x4, CmpEq, CmpGe, CmpGt, CmpLe, CmpLt};

/// Wide float holding four lanes. (`b3FloatW`)
pub type FloatW = f32x4;

const SIGN: f32x4 = f32x4::new([-0.0; 4]);

/// (b3ZeroW)
#[inline]
pub fn zero_w() -> FloatW {
    f32x4::ZERO
}

/// (b3SplatW)
#[inline]
pub fn splat_w(scalar: f32) -> FloatW {
    f32x4::splat(scalar)
}

/// (b3SetW)
#[inline]
pub fn set_w(a: f32, b: f32, c: f32, d: f32) -> FloatW {
    f32x4::new([a, b, c, d])
}

/// (b3LoadW)
#[inline]
pub fn load_w(data: &[f32]) -> FloatW {
    // One bounds check, then a single 16-byte load.
    let lanes: &[f32; 4] = data[..4].try_into().expect("four lanes");
    f32x4::new(*lanes)
}

/// (b3StoreW)
#[inline]
pub fn store_w(data: &mut [f32], a: FloatW) {
    data[..4].copy_from_slice(&a.to_array());
}

/// Lane `i` of a wide float.
#[inline]
pub fn lane(a: &FloatW, i: usize) -> f32 {
    a.as_array_ref()[i]
}

/// Sign flip, like scalar `-x` (`0 - x` would turn `-0` into `+0`). (b3NegW)
#[inline]
pub fn neg_w(a: FloatW) -> FloatW {
    a ^ SIGN
}

/// (b3AddW)
#[inline]
pub fn add_w(a: FloatW, b: FloatW) -> FloatW {
    a + b
}

/// (b3SubW)
#[inline]
pub fn sub_w(a: FloatW, b: FloatW) -> FloatW {
    a - b
}

/// (b3MulW)
#[inline]
pub fn mul_w(a: FloatW, b: FloatW) -> FloatW {
    a * b
}

/// (b3DivW)
#[inline]
pub fn div_w(a: FloatW, b: FloatW) -> FloatW {
    a / b
}

/// (b3SqrtW)
#[inline]
pub fn sqrt_w(a: FloatW) -> FloatW {
    a.sqrt()
}

/// `a + b * c`. Cannot use a real FMA because it wouldn't match the non-SIMD path.
/// (b3MulAddW)
#[inline]
pub fn mul_add_w(a: FloatW, b: FloatW, c: FloatW) -> FloatW {
    a + b * c
}

/// `a <= b ? a : b` per lane, as the scalar path. (b3MinW)
#[inline]
pub fn min_w(a: FloatW, b: FloatW) -> FloatW {
    a.cmp_le(b).blend(a, b)
}

/// Lane-wise minimum for inputs that are never equal and never NaN, where every min
/// definition agrees: one instruction (minps / vminq / f32x4.pmin) instead of a compare and
/// blend. The SAT support search uses it on values with distinct embedded indices.
#[inline]
pub fn min_distinct_w(a: FloatW, b: FloatW) -> FloatW {
    a.fast_min(b)
}

/// `a >= b ? a : b` per lane, as the scalar path. (b3MaxW)
#[inline]
pub fn max_w(a: FloatW, b: FloatW) -> FloatW {
    a.cmp_ge(b).blend(a, b)
}

/// (b3AndW)
#[inline]
pub fn and_w(a: FloatW, b: FloatW) -> FloatW {
    a & b
}

/// (b3OrW)
#[inline]
pub fn or_w(a: FloatW, b: FloatW) -> FloatW {
    a | b
}

/// (b3GreaterThanW)
#[inline]
pub fn greater_than_w(a: FloatW, b: FloatW) -> FloatW {
    a.cmp_gt(b)
}

/// (b3LessThanW)
#[inline]
pub fn less_than_w(a: FloatW, b: FloatW) -> FloatW {
    a.cmp_lt(b)
}

/// (b3EqualsW)
#[inline]
// Kept for a complete simd.h API; only the untargeted wide collide path uses it.
#[allow(dead_code)]
pub fn equals_w(a: FloatW, b: FloatW) -> FloatW {
    a.cmp_eq(b)
}

/// True when every lane equals zero (`-0` included). (b3AllZeroW)
#[inline]
pub fn all_zero_w(a: FloatW) -> bool {
    a.cmp_eq(f32x4::ZERO).all()
}

/// (b3AnyTrueW)
#[inline]
pub fn any_true_w(mask: FloatW) -> bool {
    mask.any()
}

/// Component-wise `mask ? b : a`. (b3BlendW)
#[inline]
pub fn blend_w(a: FloatW, b: FloatW, mask: FloatW) -> FloatW {
    mask.blend(b, a)
}

/// Replace the low `bit_count` mantissa bits of each lane with `base_index + lane`.
/// The value must be positive so the embedded index sorts with the value, and ties
/// fall to the lower index. (b3EmbedIndexW)
#[inline]
pub fn embed_index_w(value: FloatW, base_index: i32, bit_count: i32) -> FloatW {
    let mask = (1i32 << bit_count) - 1;
    let index = i32x4::splat(base_index) + i32x4::new([0, 1, 2, 3]);
    let bits: i32x4 = bytemuck::cast(value);
    bytemuck::cast((bits & i32x4::splat(!mask)) | index)
}

/// Recover the index embedded by [`embed_index_w`] from the lane holding the minimum.
/// (b3MinIndexW)
#[inline]
pub fn min_index_w(a: FloatW, bit_count: i32) -> i32 {
    let [x, y, z, w] = a.to_array();
    let mut m = x;
    m = if y < m { y } else { m };
    m = if z < m { z } else { m };
    m = if w < m { w } else { m };

    let bits = m.to_bits();
    (bits & ((1u32 << bit_count) - 1)) as i32
}

/// `a < 0 ? -a : a` per lane, as the scalar path (`andnot` abs would clear `-0`'s sign).
/// (b3AbsW)
#[inline]
pub fn abs_w(a: FloatW) -> FloatW {
    a.cmp_lt(f32x4::ZERO).blend(neg_w(a), a)
}

/// `a & ~b`. (b3AndNotW)
#[inline]
// Kept for a complete simd.h API; the 79fc559 contact kernels no longer use it.
#[allow(dead_code)]
pub fn and_not_w(a: FloatW, b: FloatW) -> FloatW {
    a & !b
}

/// Lanes where either base-1 body index is null (0). Used to select the contact softness.
/// (b3SoftMaskW)
#[inline]
pub fn soft_mask_w(index_a: &[i32; 4], index_b: &[i32; 4]) -> FloatW {
    let a = i32x4::new(*index_a).cmp_eq(i32x4::ZERO);
    let b = i32x4::new(*index_b).cmp_eq(i32x4::ZERO);
    bytemuck::cast(a | b)
}

/// Wide vec3. (b3Vec3W)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3W {
    pub x: FloatW,
    pub y: FloatW,
    pub z: FloatW,
}

/// (b3LoadVW)
#[inline]
pub fn load_vw(x: &[f32], y: &[f32], z: &[f32]) -> Vec3W {
    Vec3W {
        x: load_w(x),
        y: load_w(y),
        z: load_w(z),
    }
}

/// (b3StoreVW)
#[inline]
pub fn store_vw(x: &mut [f32], y: &mut [f32], z: &mut [f32], v: Vec3W) {
    store_w(x, v.x);
    store_w(y, v.y);
    store_w(z, v.z);
}

/// (b3SplatVW)
#[inline]
pub fn splat_vw(v: crate::math_functions::Vec3) -> Vec3W {
    Vec3W {
        x: splat_w(v.x),
        y: splat_w(v.y),
        z: splat_w(v.z),
    }
}

/// (b3NegVW)
#[inline]
pub fn neg_vw(a: Vec3W) -> Vec3W {
    Vec3W {
        x: neg_w(a.x),
        y: neg_w(a.y),
        z: neg_w(a.z),
    }
}

/// s * a (b3MulSVW)
#[inline]
pub fn mul_svw(s: FloatW, a: Vec3W) -> Vec3W {
    Vec3W {
        x: mul_w(s, a.x),
        y: mul_w(s, a.y),
        z: mul_w(s, a.z),
    }
}

/// a - s * b (b3MulSubSVW)
#[inline]
pub fn mul_sub_svw(a: Vec3W, s: FloatW, b: Vec3W) -> Vec3W {
    Vec3W {
        x: sub_w(a.x, mul_w(s, b.x)),
        y: sub_w(a.y, mul_w(s, b.y)),
        z: sub_w(a.z, mul_w(s, b.z)),
    }
}

/// a + s * b (b3MulAddSVW)
#[inline]
pub fn mul_add_svw(a: Vec3W, s: FloatW, b: Vec3W) -> Vec3W {
    Vec3W {
        x: add_w(a.x, mul_w(s, b.x)),
        y: add_w(a.y, mul_w(s, b.y)),
        z: add_w(a.z, mul_w(s, b.z)),
    }
}

/// a - b (b3SubVW)
#[inline]
pub fn sub_vw(a: Vec3W, b: Vec3W) -> Vec3W {
    Vec3W {
        x: sub_w(a.x, b.x),
        y: sub_w(a.y, b.y),
        z: sub_w(a.z, b.z),
    }
}

/// a + b (b3AddVW)
#[inline]
pub fn add_vw(a: Vec3W, b: Vec3W) -> Vec3W {
    Vec3W {
        x: add_w(a.x, b.x),
        y: add_w(a.y, b.y),
        z: add_w(a.z, b.z),
    }
}

/// `(a.x * b.x + a.y * b.y) + a.z * b.z` per lane. (b3DotW)
#[inline]
pub fn dot_w(a: Vec3W, b: Vec3W) -> FloatW {
    add_w(add_w(mul_w(a.x, b.x), mul_w(a.y, b.y)), mul_w(a.z, b.z))
}

/// (b3CrossW)
#[inline]
pub fn cross_w(a: Vec3W, b: Vec3W) -> Vec3W {
    Vec3W {
        x: sub_w(mul_w(a.y, b.z), mul_w(a.z, b.y)),
        y: sub_w(mul_w(a.z, b.x), mul_w(a.x, b.z)),
        z: sub_w(mul_w(a.x, b.y), mul_w(a.y, b.x)),
    }
}
