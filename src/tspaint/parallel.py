"""Process-based, bit-exact parallelism for the genome E-step (paral-assess.md).

The expensive part of painting — Felsenstein pruning + edge-blocked sufficient statistics over
the marginal trees, repeated each EM iteration — is an exact map-reduce over independent trees.
We split it by **contiguous marginal-tree-index ranges** and reduce the partial
:class:`~tspaint.accumulate.SuffStats` on the parent. Because
:func:`~tspaint.accumulate.accumulate_sufficient_statistics` banks each edge **once, at its
entry tree** (and counts each tree's loglik / each root once), every per-edge / per-tree term
lands in exactly one range — so the partition reproduces the whole-genome statistics, summed.

Float addition is **not** associative, so the honest contract is:

* ``n_jobs == 1`` → one chunk → **byte-identical** to the serial
  :func:`~tspaint.accumulate.accumulate_sufficient_statistics` (regression guard).
* ``n_jobs == P`` → **byte-identical to the same chunk partition reduced serially** (depends only
  on the partition + the parent's in-order fold, not on process placement).
* vs the serial single loop, ``P > 1`` differs only by reduction order — a few ULP
  (``allclose``).
* ``exact=True`` → runs serially, i.e. byte-identical to the serial single loop. (A *parallel*
  exact mode would need per-tree IPC and an arithmetic refactor that would itself perturb the
  serial bits; the serial path already delivers the guarantee, so we stop there.)

Painting (:func:`posterior_table_parallel`) is **exactly** equal to serial for any ``P`` — each
segment's posterior comes from its own tree's pruning, independent of the chunking, so stitching
the per-range tracks and re-merging the seams reproduces the serial segmentation.

Processes (not threads) because the hot loop is Python-level tree iteration that holds the GIL.
Workers receive the tree sequence by **path** (each loads it once, cached) plus the small
``(Q, π, w, labels)`` — emissions are rebuilt in-worker.
"""
from __future__ import annotations

import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import reduce

import numpy as np
import tskit

from .accumulate import accumulate_sufficient_statistics, SuffStats
from .output import posterior_table

__all__ = ["resolve_cores", "genome_chunks", "add_suffstats", "accumulate_parallel",
           "posterior_table_parallel", "make_pool", "as_path"]

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


# --- core count -----------------------------------------------------------------------------

def _safe_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return 0


def _parse_slurm_cpus_per_node(v):
    """Sum SLURM's ``SLURM_JOB_CPUS_PER_NODE`` compact form, e.g. ``"4"``, ``"4(x2)"``, ``"2,1"``."""
    total = 0
    for part in str(v).split(","):
        m = re.match(r"^\s*(\d+)(?:\(x(\d+)\))?\s*$", part)
        if m:
            total += int(m.group(1)) * (int(m.group(2)) if m.group(2) else 1)
    return total


def resolve_cores(requested=None):
    """Resolve the worker count: explicit ``requested`` else the SLURM allocation else ``1``.

    Order: a positive ``requested`` wins; otherwise ``$SLURM_CPUS_PER_TASK`` (the clean per-task
    value), then ``$SLURM_JOB_CPUS_PER_NODE`` (parsing its ``N(xM)`` form), then ``$TSPAINT_CORES``,
    then ``1`` (serial — the safe library default). The CLI resolves this once and passes an
    explicit integer down.
    """
    if requested is not None and _safe_int(requested) > 0:
        return _safe_int(requested)
    n = _safe_int(os.environ.get("SLURM_CPUS_PER_TASK"))
    if n > 0:
        return n
    n = _parse_slurm_cpus_per_node(os.environ.get("SLURM_JOB_CPUS_PER_NODE", ""))
    if n > 0:
        return n
    n = _safe_int(os.environ.get("TSPAINT_CORES"))
    return n if n > 0 else 1


# --- chunking & reduction -------------------------------------------------------------------

def genome_chunks(ts, n_jobs):
    """Contiguous marginal-tree-index ranges ``[(lo, hi), ...]`` partitioning ``[0, num_trees)``.

    Equal tree counts per chunk (at most ``n_jobs`` non-empty ranges). Edge-count balancing is a
    possible refinement; equal counts suffice while trees are of similar size.
    """
    T = int(ts.num_trees)
    n = max(1, min(int(n_jobs), T)) if T else 1
    bounds = [round(i * T / n) for i in range(n + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(n) if bounds[i + 1] > bounds[i]]


def add_suffstats(x, y):
    """Commutative reducer: element-wise sum of two :class:`~tspaint.accumulate.SuffStats`.

    ``S_cred`` is merged per node (union of keys, summed arrays). Inputs are not mutated.
    """
    cred = {n: np.array(c, float) for n, c in x.S_cred.items()}
    for n, c in y.S_cred.items():
        cred[n] = cred[n] + c if n in cred else np.array(c, float)
    return SuffStats(x.S_dwell + y.S_dwell, x.S_jumps + y.S_jumps, x.S_root + y.S_root,
                     cred, x.loglik + y.loglik)


# --- worker side (runs in a child process) --------------------------------------------------

_TS_CACHE = {}


def _load_cached(path):
    ts = _TS_CACHE.get(path)
    if ts is None:
        ts = tskit.load(path)
        _TS_CACHE[path] = ts
    return ts


def _accumulate_range(path, lo, hi, Q, pi, w, labels, soft_refs):
    from .em import build_emissions
    ts = _load_cached(path)
    emissions = build_emissions(ts, labels, w, pi)
    return accumulate_sufficient_statistics(ts, Q, pi, emissions, labels=labels,
                                            soft_refs=soft_refs, tree_range=(lo, hi))


def _paint_range(path, lo, hi, Q, pi, w, labels, focal, merge_tol):
    from .em import build_emissions
    ts = _load_cached(path)
    emissions = build_emissions(ts, labels, w, pi)
    return posterior_table(ts, Q, pi, emissions, focal=focal, merge_tol=merge_tol,
                           tree_range=(lo, hi))


# --- pool & temp-path management (parent side) ----------------------------------------------

def _pin_blas():
    for var in _BLAS_VARS:
        os.environ[var] = "1"


def make_pool(n_jobs):
    """A :class:`~concurrent.futures.ProcessPoolExecutor` for ``n_jobs`` workers, or ``None`` if ``<= 1``.

    BLAS thread env is pinned to 1 in the **parent** before the pool is created so the children
    inherit it at ``spawn`` (the child imports numpy while importing this module, before any
    initializer could run, so parent-set is the reliable point). The parent's own numpy is
    already initialised, so this does not throttle it.
    """
    n = max(1, int(n_jobs))
    if n <= 1:
        return None
    _pin_blas()
    return ProcessPoolExecutor(max_workers=n, initializer=_pin_blas)


@contextmanager
def as_path(ts):
    """Yield a filesystem path for ``ts`` — reuse a string path, else dump the ts to a temp ``.trees``.

    tskit does not expose a loaded tree sequence's source path, so an in-memory ts is dumped once
    (the temp file is removed on exit). A caller holding the original path can pass it through to
    skip the dump.
    """
    if isinstance(ts, (str, os.PathLike)):
        yield os.fspath(ts)
        return
    fd, tmp = tempfile.mkstemp(suffix=".trees")
    os.close(fd)
    try:
        ts.dump(tmp)
        yield tmp
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# --- public parallel E-step / painter -------------------------------------------------------

def accumulate_parallel(ts, Q, pi, *, w=None, labels=None, soft_refs=None, emissions=None,
                        n_jobs=1, executor=None, exact=False):
    """Parallel genome E-step → pooled :class:`~tspaint.accumulate.SuffStats` (see module docstring).

    Workers rebuild emissions from ``(labels, w, pi)``; pass an ``executor`` to reuse a pool
    across EM iterations. ``exact=True`` forces the serial single loop (byte-identical to legacy).
    """
    serial = exact or (executor is None and resolve_to_int(n_jobs) <= 1)
    if serial:
        if emissions is None:
            from .em import build_emissions
            emissions = build_emissions(ts, labels, w, pi)
        return accumulate_sufficient_statistics(ts, Q, pi, emissions, labels=labels,
                                                soft_refs=soft_refs)

    own = executor is None
    ex = executor or make_pool(n_jobs)
    try:
        with as_path(ts) as path:
            futures = [ex.submit(_accumulate_range, path, lo, hi, Q, pi, w, labels, soft_refs)
                       for (lo, hi) in genome_chunks(ts, n_jobs)]
            partials = [f.result() for f in futures]      # blocks while the temp path is live
        return reduce(add_suffstats, partials)
    finally:
        if own:
            ex.shutdown()


def posterior_table_parallel(ts, Q, pi, *, w=None, labels=None, focal=None, merge_tol=1e-12,
                             emissions=None, n_jobs=1, executor=None):
    """Parallel :func:`~tspaint.output.posterior_table` — **exactly** equal to serial for any ``n_jobs``.

    Splits by tree-range, then stitches the per-range tracks in genome order, re-merging adjacent
    equal segments at the seams (the same merge rule serial painting uses).
    """
    if executor is None and resolve_to_int(n_jobs) <= 1:
        if emissions is None:
            from .em import build_emissions
            emissions = build_emissions(ts, labels, w, pi)
        return posterior_table(ts, Q, pi, emissions, focal=focal, merge_tol=merge_tol)

    own = executor is None
    ex = executor or make_pool(n_jobs)
    try:
        with as_path(ts) as path:
            futures = [ex.submit(_paint_range, path, lo, hi, Q, pi, w, labels, focal, merge_tol)
                       for (lo, hi) in genome_chunks(ts, n_jobs)]
            chunk_tracks = [f.result() for f in futures]
        return _stitch_tracks(chunk_tracks, merge_tol)
    finally:
        if own:
            ex.shutdown()


def _stitch_tracks(chunk_tracks, merge_tol):
    """Concatenate per-range painting tracks (genome order) and re-merge equal segments at seams."""
    out = {}
    for chunk in chunk_tracks:
        for s, segs in chunk.items():
            dst = out.setdefault(s, [])
            for seg in segs:
                if (dst and dst[-1].right == seg.left and dst[-1].status == seg.status
                        and np.allclose(dst[-1].posterior, seg.posterior, atol=merge_tol, rtol=0)):
                    dst[-1].right = seg.right
                else:
                    dst.append(seg)
    return out


def resolve_to_int(n_jobs):
    """``n_jobs`` as a worker count, treating ``None`` as serial (1) — *not* SLURM-resolved."""
    return 1 if n_jobs is None else max(1, int(n_jobs))
