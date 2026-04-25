"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

###############################################################################
####################### MULTIPROCESSING PREPARE HELPERS #######################
###############################################################################

import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Protocol

from tqdm import tqdm

# Modules to preload in the forkserver process so they are imported exactly
# once and inherited by every worker via ``fork()`` instead of re-imported per
# worker (which is what makes ``spawn`` slow). ``xanesnet`` is included so all
# downstream package-level side effects (registry registration, e3nn constant
# load, cuequivariance probe) happen up-front in the forkserver.
_FORKSERVER_PRELOAD = [
    "numpy",
    "torch",
    "torch_geometric",
    "e3nn",
    "xanesnet",
]


class _MpDataset(Protocol):
    """
    Protocol for datasets that can be prepared with multiprocessing. Each
    such dataset must implement ``_process_range(start, end)`` which iterates
    over ``self.datasource[start:end]`` and writes processed samples to
    ``self.processed_dir`` using ``mp_save_path`` for the filename.
    """

    @property
    def processed_dir(self) -> str: ...

    def _process_range(self, start: int, end: int) -> None: ...


def resolve_num_workers(num_workers: int | None) -> int:
    """
    Return the number of worker processes to use.

    ``None`` or any non-positive value falls back to ``os.cpu_count()``
    (with a final fallback of 1 if that is unavailable).
    """
    if num_workers is None or num_workers <= 0:
        return os.cpu_count() or 1
    return num_workers


def split_index_ranges(total: int, num_chunks: int) -> list[tuple[int, int]]:
    """
    Split ``[0, total)`` into at most ``num_chunks`` (start, end) tuples.

    Empty ranges are dropped so the worker pool only receives chunks that
    actually have work to do.
    """
    if total <= 0:
        return []
    k = max(1, min(num_chunks, total))
    base = total // k
    rem = total % k
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def mp_save_path(processed_dir: str, global_idx: int, seq: int) -> str:
    """
    Build a temporary save path for a sample produced by a worker.

    Lexicographic sorting of the resulting filenames recovers the original
    datasource ordering during the post-processing rename pass.
    """
    return os.path.join(processed_dir, f"{global_idx:010d}_{seq:06d}.pth")


def _worker_entry(dataset: _MpDataset, start: int, end: int) -> None:
    dataset._process_range(start, end)


def run_mp_prepare(
    dataset: _MpDataset,
    total: int,
    num_workers: int | None,
) -> int:
    """
    Run ``prepare`` for ``dataset`` over ``total`` datasource items using a
    pool of worker processes.

    Each worker handles a disjoint slice and writes files using
    :func:`mp_save_path`. After all workers finish, the files are sorted
    lexicographically and renamed to the canonical ``{counter}.pth`` form so
    that the on-disk layout is identical to the sequential preparation.

    Returns the number of files written.
    """
    n_workers = resolve_num_workers(num_workers)
    ranges = split_index_ranges(total, n_workers)
    if not ranges:
        return 0

    logging.info(f"Preparing dataset with {len(ranges)} worker process(es) over {total} samples.")

    # Use the ``forkserver`` start method. It's the right trade-off between
    # ``fork`` (which deadlocks when the parent has already initialised
    # PyTorch / OpenMP / CUDA state) and ``spawn`` (which re-imports every
    # heavy module in every worker, costing several seconds per worker).
    #
    # The forkserver is a small intermediary process that imports the
    # preloaded modules once and then ``fork()``s clean workers on demand.
    # Because the forkserver itself never runs compute, no OMP / BLAS thread
    # is alive at fork time and the deadlock that motivated avoiding plain
    # ``fork`` does not apply.
    ctx = mp.get_context("forkserver")
    try:
        ctx.set_forkserver_preload(_FORKSERVER_PRELOAD)
    except (AttributeError, RuntimeError):
        # Older Pythons / platforms without forkserver support: fall back to
        # ``spawn``. Workers will pay the import cost individually.
        ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=len(ranges), mp_context=ctx) as ex:
        t0 = time.perf_counter()
        futures = [ex.submit(_worker_entry, dataset, s, e) for s, e in ranges]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            # Re-raise exceptions from workers in the main process.
            f.result()
        elapsed = time.perf_counter() - t0

    rate = total / elapsed if elapsed > 0 else float("inf")
    logging.info(f"Prepared {total} samples in {elapsed:.2f}s " f"({rate:.2f} it/s, {len(ranges)} worker(s)).")

    # Rename to the canonical {counter}.pth layout.
    files = sorted(f for f in os.listdir(dataset.processed_dir) if f.endswith(".pth"))
    # Two-pass rename: first to a temp name so we never collide with an
    # already-canonical {i}.pth file produced by some worker.
    tmp_files: list[str] = []
    for i, fn in enumerate(files):
        src = os.path.join(dataset.processed_dir, fn)
        tmp = os.path.join(dataset.processed_dir, f"__tmp_{i}.pth")
        os.rename(src, tmp)
        tmp_files.append(tmp)
    for i, tmp in enumerate(tmp_files):
        dst = os.path.join(dataset.processed_dir, f"{i}.pth")
        os.rename(tmp, dst)
    return len(tmp_files)
