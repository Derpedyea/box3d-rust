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
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

// `rows` and `set_velocity_rows` reinterpret two words as a row of four floats.
#[cfg(target_endian = "big")]
compile_error!("SolverStates assumes little-endian 64-bit words");

/// One body state: [`BodyState`]'s C layout as eight 64-bit words, two per 16-byte row, so a
/// row loads with two moves and one shuffle.
/// Cache-line aligned, like C's 64-byte `b3BodyState`, so a state never spans two lines and
/// workers writing neighboring bodies don't share one.
#[repr(align(64))]
struct Cell([AtomicU64; 8]);

impl Cell {
    fn new(words: [u64; 8]) -> Cell {
        Cell(words.map(AtomicU64::new))
    }
}

impl std::ops::Deref for Cell {
    type Target = [AtomicU64; 8];
    #[inline]
    fn deref(&self) -> &[AtomicU64; 8] {
        &self.0
    }
}

impl std::ops::DerefMut for Cell {
    #[inline]
    fn deref_mut(&mut self) -> &mut [AtomicU64; 8] {
        &mut self.0
    }
}

/// The states, plus one identity state after them that null (0) base-1 indices read, so the
/// wide gather selects an index instead of branching per lane.
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
fn pack(w: [u32; 16]) -> [u64; 8] {
    core::array::from_fn(|i| u64::from(w[2 * i]) | (u64::from(w[2 * i + 1]) << 32))
}

#[inline]
fn unpack(q: [u64; 8]) -> [u32; 16] {
    core::array::from_fn(|i| (q[i / 2] >> (32 * (i % 2))) as u32)
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
    pub fn load_from(&mut self, pool: Option<&crate::parallel::ThreadPool>, states: &[BodyState]) {
        self.cells.truncate(states.len());
        let have = self.cells.len();
        crate::parallel::zip_each(pool, &mut self.cells[..have], &states[..have], |cell, s| {
            for (c, q) in cell.iter_mut().zip(pack(words(s))) {
                *c.get_mut() = q;
            }
        });
        self.cells
            .extend(states[have..].iter().map(|s| Cell::new(pack(words(s)))));
        self.cells
            .push(Cell::new(pack(words(&crate::body::IDENTITY_BODY_STATE))));
    }

    /// Write the states back.
    pub fn store_to(
        &mut self,
        pool: Option<&crate::parallel::ThreadPool>,
        states: &mut [BodyState],
    ) {
        debug_assert!(states.len() + 1 == self.cells.len());
        let n = states.len();
        crate::parallel::zip_each(pool, states, &self.cells[..n], |s, cell| {
            *s = state(&unpack(cell.each_ref().map(|c| c.load(Relaxed))));
        });
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cells.len().saturating_sub(1)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, index: usize) -> BodyState {
        state(&unpack(
            self.cells[index].each_ref().map(|c| c.load(Relaxed)),
        ))
    }

    #[inline]
    pub fn set(&self, index: usize, s: BodyState) {
        for (c, q) in self.cells[index].iter().zip(pack(words(&s))) {
            c.store(q, Relaxed);
        }
    }

    /// The state at `index`, or the identity state for `NULL_INDEX` (static bodies).
    #[inline]
    pub fn get_or_identity(&self, index: i32) -> BodyState {
        let last = self.cells.len() - 1;
        let index = if index < 0 { last } else { index as usize };
        state(&unpack(
            self.cells[index].each_ref().map(|c| c.load(Relaxed)),
        ))
    }

    /// The four 16-byte rows of the state at base-1 `index1`, or of the identity state for 0,
    /// for the wide gather.
    #[inline]
    pub fn rows1(&self, index1: i32) -> [FloatW; 4] {
        let last = self.cells.len() - 1;
        let index = if index1 == 0 {
            last
        } else {
            (index1 - 1) as usize
        };
        let c = &self.cells[index];
        let row = |r: usize| -> FloatW {
            bytemuck::cast([c[2 * r].load(Relaxed), c[2 * r + 1].load(Relaxed)])
        };
        [row(0), row(1), row(2), row(3)]
    }

    /// The flags word of state `index`.
    #[inline]
    pub fn flags(&self, index: usize) -> u32 {
        (self.cells[index][5].load(Relaxed) >> 32) as u32
    }

    /// Write the velocity rows (with their padding lanes) of state `index`.
    #[inline]
    pub fn set_velocity_rows(&self, index: usize, v: FloatW, w: FloatW) {
        let c = &self.cells[index];
        let [v0, v1]: [u64; 2] = bytemuck::cast(v);
        let [w0, w1]: [u64; 2] = bytemuck::cast(w);
        c[0].store(v0, Relaxed);
        c[1].store(v1, Relaxed);
        c[2].store(w0, Relaxed);
        c[3].store(w1, Relaxed);
    }
}
