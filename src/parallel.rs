//! Parallel execution shim.
//!
//! Native builds fan work out through Rayon. `wasm32` builds run the exact same
//! closures sequentially, because WebAssembly threads require shared memory and
//! cross-origin isolation that a plain `wasm-bindgen` target does not provide.
//!
//! Keeping the two implementations behind one API means the numeric kernels have a
//! single source of truth on every target.

/// Applies `f` to every index in `0..len`, returning the results in index order.
#[cfg(not(target_arch = "wasm32"))]
pub fn map_range<R, F>(len: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    use rayon::prelude::*;
    (0..len).into_par_iter().map(f).collect()
}

/// Applies `f` to every index in `0..len`, returning the results in index order.
#[cfg(target_arch = "wasm32")]
pub fn map_range<R, F>(len: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    (0..len).map(f).collect()
}

/// Applies `f` to every element of `items`, preserving order.
#[cfg(not(target_arch = "wasm32"))]
pub fn map_slice<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    use rayon::prelude::*;
    items.par_iter().map(f).collect()
}

/// Applies `f` to every element of `items`, preserving order.
#[cfg(target_arch = "wasm32")]
pub fn map_slice<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    items.iter().map(f).collect()
}

/// Sorts in place. Native builds use Rayon's parallel sort.
#[cfg(not(target_arch = "wasm32"))]
pub fn sort_unstable<T: Ord + Send>(items: &mut [T]) {
    use rayon::prelude::*;
    items.par_sort_unstable();
}

/// Sorts in place.
#[cfg(target_arch = "wasm32")]
pub fn sort_unstable<T: Ord + Send>(items: &mut [T]) {
    items.sort_unstable();
}

/// Fills `dst` in parallel from `f(index)`.
#[cfg(not(target_arch = "wasm32"))]
pub fn fill_range<T, F>(dst: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    use rayon::prelude::*;
    dst.par_iter_mut()
        .enumerate()
        .for_each(|(i, slot)| *slot = f(i));
}

/// Fills `dst` from `f(index)`.
#[cfg(target_arch = "wasm32")]
pub fn fill_range<T, F>(dst: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    for (i, slot) in dst.iter_mut().enumerate() {
        *slot = f(i);
    }
}

/// Fills four slices in lock-step, chunk by chunk, with disjoint mutable access.
///
/// This is the safe replacement for the raw-pointer writes that used to scatter
/// results from inside a Rayon closure. Each chunk owns a disjoint slice of every
/// output array, so the borrow checker proves the absence of aliasing.
///
/// `chunk_sizes` are element counts per slice; `c` is the indexing unit (one
/// element per logical item), so the closure receives `chunk_index * chunk_sizes[2]`
/// as the base item index.
#[cfg(not(target_arch = "wasm32"))]
pub fn zip_fill_4<A, B, C, D, F>(
    a: &mut [A],
    b: &mut [B],
    c: &mut [C],
    d: &mut [D],
    chunk_sizes: [usize; 4],
    f: F,
) where
    A: Send,
    B: Send,
    C: Send,
    D: Send,
    F: Fn(usize, &mut [A], &mut [B], &mut [C], &mut [D]) + Sync + Send,
{
    use rayon::prelude::*;
    a.par_chunks_mut(chunk_sizes[0])
        .zip(b.par_chunks_mut(chunk_sizes[1]))
        .zip(c.par_chunks_mut(chunk_sizes[2]))
        .zip(d.par_chunks_mut(chunk_sizes[3]))
        .enumerate()
        .for_each(|(chunk_index, (((a_chunk, b_chunk), c_chunk), d_chunk))| {
            f(
                chunk_index * chunk_sizes[2],
                a_chunk,
                b_chunk,
                c_chunk,
                d_chunk,
            );
        });
}

/// Fills four slices in lock-step, chunk by chunk, with disjoint mutable access.
#[cfg(target_arch = "wasm32")]
pub fn zip_fill_4<A, B, C, D, F>(
    a: &mut [A],
    b: &mut [B],
    c: &mut [C],
    d: &mut [D],
    chunk_sizes: [usize; 4],
    f: F,
) where
    A: Send,
    B: Send,
    C: Send,
    D: Send,
    F: Fn(usize, &mut [A], &mut [B], &mut [C], &mut [D]) + Sync + Send,
{
    a.chunks_mut(chunk_sizes[0])
        .zip(b.chunks_mut(chunk_sizes[1]))
        .zip(c.chunks_mut(chunk_sizes[2]))
        .zip(d.chunks_mut(chunk_sizes[3]))
        .enumerate()
        .for_each(|(chunk_index, (((a_chunk, b_chunk), c_chunk), d_chunk))| {
            f(
                chunk_index * chunk_sizes[2],
                a_chunk,
                b_chunk,
                c_chunk,
                d_chunk,
            );
        });
}
