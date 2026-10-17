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

/// `out = items.chunks_mut(chunk).map(f)`, in parallel across the pool, reusing `out`'s
/// allocation.
pub(crate) fn map_chunks_into<T: Send, U: Send>(
    pool: Option<&ThreadPool>,
    items: &mut [T],
    chunk: usize,
    out: &mut Vec<U>,
    f: impl Fn(&mut [T]) -> U + Sync,
) {
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        if items.len() > 8 * chunk {
            use rayon::prelude::*;
            pool.install(|| {
                items
                    .par_chunks_mut(chunk)
                    .with_min_len(4)
                    .map(&f)
                    .collect_into_vec(out)
            });
            return;
        }
    }
    let _ = pool;
    out.clear();
    out.extend(items.chunks_mut(chunk).map(f));
}

/// `f(item, lanes chunk, out)` for `items` zipped with `lanes.chunks_mut(chunk)`, in parallel
/// across the pool, concatenating the ids the calls push. Callers use the ids as a set.
pub(crate) fn zip_chunks<A: Sync, B: Send>(
    pool: Option<&ThreadPool>,
    items: &[A],
    lanes: &mut [B],
    chunk: usize,
    f: impl Fn(&A, &mut [B], &mut Vec<i32>) + Sync,
) -> Vec<i32> {
    debug_assert!(items.len() == lanes.len().div_ceil(chunk));
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        if items.len() > 32 {
            use rayon::prelude::*;
            return pool.install(|| {
                items
                    .par_iter()
                    .zip(lanes.par_chunks_mut(chunk))
                    .with_min_len(16)
                    .fold(Vec::new, |mut out, (item, lanes)| {
                        f(item, lanes, &mut out);
                        out
                    })
                    .reduce(Vec::new, |mut a, mut b| {
                        a.append(&mut b);
                        a
                    })
            });
        }
    }
    let _ = pool;
    let mut out = Vec::new();
    for (item, lanes) in items.iter().zip(lanes.chunks_mut(chunk)) {
        f(item, lanes, &mut out);
    }
    out
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
