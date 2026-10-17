//! Thread pool plumbing. C hands parallel work to the user's task system
//! (`b3WorldDef::enqueueTask`); the port runs it on a rayon pool owned by the world when the
//! `parallel` feature is on and `WorldDef::worker_count` is above 1, and serially otherwise.
//!
//! Every parallel stage here merges its blocks with order-free operations (sums, unions,
//! per-item writes), so results are bit-identical for any worker count.

use crate::contact::collide::CollideStats;
use crate::contact::Contact;

#[cfg(feature = "parallel")]
pub type ThreadPool = rayon::ThreadPool;

/// Stand-in so `World` has the same fields with the feature off.
#[cfg(not(feature = "parallel"))]
#[derive(Debug)]
pub struct ThreadPool;

#[cfg(not(feature = "parallel"))]
impl ThreadPool {
    /// Mirrors `rayon::ThreadPool::current_num_threads`; the stand-in pool is never built, so
    /// any caller runs serially.
    pub fn current_num_threads(&self) -> usize {
        1
    }
}

/// A pool of `worker_count` threads, or `None` (serial) for one worker, without the
/// `parallel` feature, or if the threads can't be spawned.
pub(crate) fn make_pool(worker_count: i32) -> Option<std::sync::Arc<ThreadPool>> {
    #[cfg(feature = "parallel")]
    if worker_count > 1 {
        return rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count as usize)
            .thread_name(|i| format!("box3d-worker-{i}"))
            .build()
            .ok()
            .map(std::sync::Arc::new);
    }
    let _ = worker_count;
    None
}

/// Contacts per block. Narrow phase cost varies a lot per contact, so blocks stay small
/// enough to balance (C targets 32 blocks per worker).
const COLLIDE_BLOCK: usize = 64;

pub(crate) fn collide_contacts(
    pool: Option<&ThreadPool>,
    work: &mut [&mut Contact],
    block: impl Fn(&mut [&mut Contact]) -> CollideStats + Sync,
) -> CollideStats {
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        use rayon::prelude::*;
        return pool.install(|| {
            work.par_chunks_mut(COLLIDE_BLOCK).map(&block).reduce(
                CollideStats::default,
                |mut a, b| {
                    a.add(b);
                    a
                },
            )
        });
    }
    let _ = pool;
    let mut total = CollideStats::default();
    for chunk in work.chunks_mut(COLLIDE_BLOCK) {
        total.add(block(chunk));
    }
    total
}

/// `block(start, end, out)` over `0..item_count` in the blocks C's b3ParallelFor uses,
/// appending each block's output to `out`. Blocks may run in any order: callers must not
/// depend on the order of `out`.
pub(crate) fn for_blocks<T: Send>(
    pool: Option<&ThreadPool>,
    worker_count: i32,
    item_count: i32,
    min_range: i32,
    out: &mut Vec<T>,
    block: impl Fn(i32, i32, &mut Vec<T>) + Sync,
) {
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        use rayon::prelude::*;
        let (block_size, block_count) = block_partition(worker_count, item_count, min_range);
        let outputs: Vec<Vec<T>> = pool.install(|| {
            (0..block_count)
                .into_par_iter()
                .map(|block_index| {
                    let start = block_index * block_size;
                    let end = (start + block_size).min(item_count);
                    let mut keys = Vec::new();
                    block(start, end, &mut keys);
                    keys
                })
                .collect()
        });
        for mut keys in outputs {
            out.append(&mut keys);
        }
        return;
    }
    let _ = pool;
    crate::parallel_for::parallel_for(worker_count, item_count, min_range, |start, end, _| {
        block(start, end, out);
    });
}

/// (block size, block count) of b3ParallelFor.
#[cfg(feature = "parallel")]
fn block_partition(worker_count: i32, item_count: i32, min_range: i32) -> (i32, i32) {
    if item_count <= 0 {
        return (min_range, 0);
    }
    let max_block_count = 32 * worker_count;
    let block_size = if item_count <= min_range * max_block_count {
        min_range
    } else {
        (item_count + max_block_count - 1) / max_block_count
    };
    (block_size, (item_count + block_size - 1) / block_size)
}

/// Run `main` on this thread while `worker_count - 1` pool threads run `worker`, and wait for
/// all of them. `main` must make the workers return (they poll for it). Without a pool,
/// just `main`.
pub(crate) fn with_workers<R: Send>(
    pool: Option<&ThreadPool>,
    worker_count: usize,
    main: impl FnOnce() -> R + Send,
    worker: impl Fn() + Sync,
) -> R {
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        if worker_count > 1 {
            return pool.install(|| {
                let mut out = None;
                rayon::scope(|scope| {
                    for _ in 1..worker_count {
                        scope.spawn(|_| worker());
                    }
                    out = Some(main());
                });
                out.expect("main ran")
            });
        }
    }
    let _ = (pool, worker_count, &worker);
    main()
}

/// `f` on consecutive blocks of `items` across the pool, folding the results in block
/// order with `merge`.
pub(crate) fn map_reduce_blocks<T: Send, R: Send + Default>(
    pool: Option<&ThreadPool>,
    items: &mut [T],
    block: usize,
    f: impl Fn(&mut [T]) -> R + Sync,
    merge: impl Fn(R, R) -> R + Sync,
) -> R {
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        if items.len() > 2 * block {
            use rayon::prelude::*;
            let results: Vec<R> = pool.install(|| items.par_chunks_mut(block).map(&f).collect());
            return results.into_iter().fold(R::default(), merge);
        }
    }
    let _ = pool;
    items.chunks_mut(block).map(f).fold(R::default(), merge)
}

/// `items.iter().map(f).collect()`, in parallel across the pool, keeping the order.
pub(crate) fn map_items<T: Sync, U: Send>(
    pool: Option<&ThreadPool>,
    items: &[T],
    f: impl Fn(&T) -> U + Sync,
) -> Vec<U> {
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        if items.len() > 1 {
            use rayon::prelude::*;
            return pool.install(|| items.par_iter().map(&f).collect());
        }
    }
    let _ = pool;
    items.iter().map(f).collect()
}

/// `(a(), b())`, concurrently on the pool if there is one.
pub(crate) fn join<A: Send, B: Send>(
    pool: Option<&ThreadPool>,
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        return pool.install(|| rayon::join(a, b));
    }
    let _ = pool;
    (a(), b())
}

/// Borrow the items that have a slot into a vector indexed by slot: `out[slot(x)] = x`.
/// Every slot in `0..slot_count` must be taken exactly once. The scan over `items`, which
/// reads each item, runs across the pool; placing the borrows is a cheap serial pass.
pub(crate) fn borrow_by_slot<'a, T: Send>(
    pool: Option<&ThreadPool>,
    items: &'a mut [T],
    slot_count: usize,
    slot: impl Fn(&T) -> Option<usize> + Sync,
) -> Vec<&'a mut T> {
    let mut slots: Vec<Option<&'a mut T>> = Vec::with_capacity(slot_count);
    slots.resize_with(slot_count, || None);
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        if items.len() > 4096 {
            use rayon::prelude::*;
            let found: Vec<Vec<(usize, &'a mut T)>> = pool.install(|| {
                items
                    .par_chunks_mut(2048)
                    .map(|chunk| {
                        chunk
                            .iter_mut()
                            .filter_map(|x| slot(x).map(|s| (s, x)))
                            .collect()
                    })
                    .collect()
            });
            for (s, x) in found.into_iter().flatten() {
                slots[s] = Some(x);
            }
            return slots
                .into_iter()
                .map(|x| x.expect("every slot taken"))
                .collect();
        }
    }
    let _ = pool;
    for x in items.iter_mut() {
        if let Some(s) = slot(x) {
            slots[s] = Some(x);
        }
    }
    slots
        .into_iter()
        .map(|x| x.expect("every slot taken"))
        .collect()
}

/// `items.sort_unstable()`, across the pool for large inputs. Sorting plain keys gives the
/// same order either way.
pub(crate) fn sort_unstable<T: Ord + Send>(pool: Option<&ThreadPool>, items: &mut [T]) {
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        if items.len() > 16384 {
            use rayon::prelude::*;
            pool.install(|| items.par_sort_unstable());
            return;
        }
    }
    let _ = pool;
    items.sort_unstable();
}

/// `f(a_i, b_i)` for `a` zipped with `b`, across the pool in blocks.
pub(crate) fn zip_each<A: Send, B: Sync>(
    pool: Option<&ThreadPool>,
    a: &mut [A],
    b: &[B],
    f: impl Fn(&mut A, &B) + Sync,
) {
    debug_assert!(a.len() == b.len());
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        if a.len() > 1024 {
            use rayon::prelude::*;
            pool.install(|| {
                a.par_iter_mut()
                    .zip(b.par_iter())
                    .with_min_len(512)
                    .for_each(|(x, y)| f(x, y))
            });
            return;
        }
    }
    let _ = pool;
    for (x, y) in a.iter_mut().zip(b) {
        f(x, y);
    }
}

/// The calling thread's index in its pool, or 0 outside one.
pub(crate) fn thread_index() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_thread_index().unwrap_or(0)
    }
    #[cfg(not(feature = "parallel"))]
    {
        0
    }
}

/// Run `f` on the pool (so parallel work inside it forks from a worker), or inline.
pub(crate) fn install<R: Send>(pool: Option<&ThreadPool>, f: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        return pool.install(f);
    }
    let _ = pool;
    f()
}
