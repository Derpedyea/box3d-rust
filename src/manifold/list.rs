//! A contact's manifolds. Convex contacts have one, stored inline so the narrow phase and the
//! solver's prepare and store passes don't chase a pointer per contact; mesh contacts can
//! have many, kept in a vector. Derefs to a slice either way. (C: `b3Manifold* manifolds`
//! from per-count block allocators; its source notes the same todo.)
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use super::types::Manifold;

// Large on purpose: the inline manifold is the point.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ManifoldList {
    #[default]
    Empty,
    One(Manifold),
    Many(Vec<Manifold>),
}

impl ManifoldList {
    /// Append a manifold. (Growing past one moves them to a vector, which then stays.)
    pub fn push(&mut self, manifold: Manifold) {
        match self {
            ManifoldList::Empty => *self = ManifoldList::One(manifold),
            ManifoldList::One(first) => *self = ManifoldList::Many(vec![*first, manifold]),
            ManifoldList::Many(v) => v.push(manifold),
        }
    }

    /// Remove all manifolds, keeping a vector's capacity.
    pub fn clear(&mut self) {
        match self {
            ManifoldList::Many(v) => v.clear(),
            _ => *self = ManifoldList::Empty,
        }
    }
}

impl From<Vec<Manifold>> for ManifoldList {
    fn from(v: Vec<Manifold>) -> ManifoldList {
        match v.len() {
            0 => ManifoldList::Empty,
            1 => ManifoldList::One(v[0]),
            _ => ManifoldList::Many(v),
        }
    }
}

impl std::ops::Deref for ManifoldList {
    type Target = [Manifold];

    #[inline]
    fn deref(&self) -> &[Manifold] {
        match self {
            ManifoldList::Empty => &[],
            ManifoldList::One(m) => std::slice::from_ref(m),
            ManifoldList::Many(v) => v,
        }
    }
}

impl std::ops::DerefMut for ManifoldList {
    #[inline]
    fn deref_mut(&mut self) -> &mut [Manifold] {
        match self {
            ManifoldList::Empty => &mut [],
            ManifoldList::One(m) => std::slice::from_mut(m),
            ManifoldList::Many(v) => v,
        }
    }
}

impl<'a> IntoIterator for &'a ManifoldList {
    type Item = &'a Manifold;
    type IntoIter = std::slice::Iter<'a, Manifold>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut ManifoldList {
    type Item = &'a mut Manifold;
    type IntoIter = std::slice::IterMut<'a, Manifold>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
