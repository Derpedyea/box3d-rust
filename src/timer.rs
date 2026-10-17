//! Tick counter for the step profile. (timer.c: b3GetTicks, b3GetMilliseconds,
//! b3GetMillisecondsAndReset)
//!
//! Ticks are nanoseconds since the first call. wasm32 without WASI has no clock in `std`,
//! so there the profile reads zero.
//!
//! SPDX-FileCopyrightText: 2025 Erin Catto
//! SPDX-License-Identifier: MIT

#[cfg(not(all(target_arch = "wasm32", not(target_os = "wasi"))))]
fn now() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
fn now() -> u64 {
    0
}

/// (b3GetTicks)
#[inline]
pub fn get_ticks() -> u64 {
    now()
}

/// Milliseconds since `ticks`. (b3GetMilliseconds)
#[inline]
pub fn get_milliseconds(ticks: u64) -> f32 {
    (now().wrapping_sub(ticks) as f64 * 1.0e-6) as f32
}

/// Milliseconds since `ticks`, then moves `ticks` to now. (b3GetMillisecondsAndReset)
#[inline]
pub fn get_milliseconds_and_reset(ticks: &mut u64) -> f32 {
    let ticks_now = now();
    let ms = (ticks_now.wrapping_sub(*ticks) as f64 * 1.0e-6) as f32;
    *ticks = ticks_now;
    ms
}
