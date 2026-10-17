//! Body states as the constraint solver's worker threads see them.
//!
//! Inside a graph color no two constraints share a dynamic body, so C's workers write
//! `b3BodyState`s through plain pointers without races. Safe Rust can't hand out `&mut` to
//! scattered elements, so during the solve the states live in relaxed atomics instead:
//! each `get`/`set` is plain loads and stores on x86-64, AArch64, and wasm, and the
//! coloring still guarantees one writer per body per stage. The stage barriers (rayon
//! joins) order the stages.
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

use crate::body::BodyState;
use crate::math_functions::{Quat, Vec3};
use crate::simd::FloatW;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// One body state: the 16 words of [`BodyState`]'s C layout.
type Cell = [AtomicU32; 16];

#[derive(Default)]
pub struct SolverStates {
    cells: Vec<Cell>,
}

impl std::fmt::Debug for SolverStates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverStates")
            .field("len", &self.cells.len())
            .finish()
    }
}

#[inline]
fn words(s: &BodyState) -> [u32; 16] {
    [
        s.linear_velocity.x.to_bits(),
        s.linear_velocity.y.to_bits(),
        s.linear_velocity.z.to_bits(),
        s.padding1.to_bits(),
        s.angular_velocity.x.to_bits(),
        s.angular_velocity.y.to_bits(),
        s.angular_velocity.z.to_bits(),
        s.padding2.to_bits(),
        s.delta_position.x.to_bits(),
        s.delta_position.y.to_bits(),
        s.delta_position.z.to_bits(),
        s.flags,
        s.delta_rotation.v.x.to_bits(),
        s.delta_rotation.v.y.to_bits(),
        s.delta_rotation.v.z.to_bits(),
        s.delta_rotation.s.to_bits(),
    ]
}

#[inline]
fn vec3(w: &[u32]) -> Vec3 {
    Vec3 {
        x: f32::from_bits(w[0]),
        y: f32::from_bits(w[1]),
        z: f32::from_bits(w[2]),
    }
}

#[inline]
fn state(w: &[u32; 16]) -> BodyState {
    BodyState {
        linear_velocity: vec3(&w[0..3]),
        padding1: f32::from_bits(w[3]),
        angular_velocity: vec3(&w[4..7]),
        padding2: f32::from_bits(w[7]),
        delta_position: vec3(&w[8..11]),
        flags: w[11],
        delta_rotation: Quat {
            v: vec3(&w[12..15]),
            s: f32::from_bits(w[15]),
        },
    }
}

impl SolverStates {
    /// Take a copy of `states`, reusing this buffer.
    pub fn load_from(&mut self, states: &[BodyState]) {
        self.cells.truncate(states.len());
        let have = self.cells.len();
        for (cell, s) in self.cells.iter_mut().zip(states) {
            for (c, w) in cell.iter_mut().zip(words(s)) {
                *c.get_mut() = w;
            }
        }
        self.cells
            .extend(states[have..].iter().map(|s| words(s).map(AtomicU32::new)));
    }

    /// Write the states back.
    pub fn store_to(&mut self, states: &mut [BodyState]) {
        debug_assert!(states.len() == self.cells.len());
        for (s, cell) in states.iter_mut().zip(&mut self.cells) {
            *s = state(&cell.each_mut().map(|c| *c.get_mut()));
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    #[inline]
    pub fn get(&self, index: usize) -> BodyState {
        state(&self.cells[index].each_ref().map(|c| c.load(Relaxed)))
    }

    #[inline]
    pub fn set(&self, index: usize, s: BodyState) {
        for (c, w) in self.cells[index].iter().zip(words(&s)) {
            c.store(w, Relaxed);
        }
    }

    /// The four 16-byte rows of state `index`, for the wide gather.
    #[inline]
    pub fn rows(&self, index: usize) -> [FloatW; 4] {
        let c = &self.cells[index];
        let row = |r: usize| {
            FloatW::new([
                f32::from_bits(c[4 * r].load(Relaxed)),
                f32::from_bits(c[4 * r + 1].load(Relaxed)),
                f32::from_bits(c[4 * r + 2].load(Relaxed)),
                f32::from_bits(c[4 * r + 3].load(Relaxed)),
            ])
        };
        [row(0), row(1), row(2), row(3)]
    }

    /// The flags word of state `index`.
    #[inline]
    pub fn flags(&self, index: usize) -> u32 {
        self.cells[index][11].load(Relaxed)
    }

    /// Write the velocity rows (with their padding lanes) of state `index`.
    #[inline]
    pub fn set_velocity_rows(&self, index: usize, v: FloatW, w: FloatW) {
        let c = &self.cells[index];
        for (k, x) in v.to_array().into_iter().enumerate() {
            c[k].store(x.to_bits(), Relaxed);
        }
        for (k, x) in w.to_array().into_iter().enumerate() {
            c[4 + k].store(x.to_bits(), Relaxed);
        }
    }
}
