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
from contextlib import contextmanager, ExitStack
from functools import reduce

import numpy as np
import tskit

from .accumulate import accumulate_sufficient_statistics, SuffStats
from .output import posterior_table

__all__ = ["resolve_cores", "genome_chunks", "add_suffstats", "accumulate_parallel",
           "posterior_table_parallel", "loo_posterior_table_parallel", "date_members_parallel",
           "foreignness_track_parallel", "depth_track_parallel", "make_pool", "as_path"]

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
    """Resolve the worker count: explicit ``requested`` else the SLURM allocation else **all CPUs**.

    Order: a positive ``requested`` wins; otherwise ``$SLURM_CPUS_PER_TASK`` (the clean per-task
    value), then ``$SLURM_JOB_CPUS_PER_NODE`` (parsing its ``N(xM)`` form), then ``$TSPAINT_CORES``
    (a manual override / test hook), then ``os.cpu_count()``. So the **default** (``requested`` /
    ``n_jobs`` left as ``None``) is the number of CPUs available, or the SLURM allocation when running
    under SLURM. Pass an explicit ``n_jobs=1`` to force serial.

    Parameters
    ----------
    requested : int or None, optional
        Explicit worker count. A **positive** value is returned as-is; ``None`` (default) or any
        non-positive / non-numeric value falls through to the SLURM / environment chain above.
        ``requested=0`` does **not** force serial (it auto-detects) — pass ``1`` for serial.

    Returns
    -------
    int
        The resolved worker count (always ``>= 1``).
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
    if n > 0:
        return n
    return os.cpu_count() or 1


# --- chunking & reduction -------------------------------------------------------------------

def genome_chunks(ts, n_jobs):
    """Contiguous marginal-tree-index ranges ``[(lo, hi), ...]`` partitioning ``[0, num_trees)``.

    Each chunk is a **half-open range of marginal-tree indices** ``(lo, hi)`` — the worker processes
    trees ``lo <= ti < hi`` — **not** a genomic ``[left, right)`` interval. Equal tree counts per
    chunk (at most ``n_jobs`` non-empty ranges). Edge-count balancing is a possible refinement;
    equal counts suffice while trees are of similar size.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence to partition (only its ``num_trees`` is read).
    n_jobs : int
        Target number of chunks, clamped to at most ``num_trees`` (and at least 1).

    Returns
    -------
    list[tuple[int, int]]
        Non-empty ``(lo, hi)`` half-open marginal-tree-index ranges tiling ``[0, num_trees)`` in
        genome order (empty list for a tree sequence with no trees).
    """
    T = int(ts.num_trees)
    n = max(1, min(int(n_jobs), T)) if T else 1
    bounds = [round(i * T / n) for i in range(n + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(n) if bounds[i + 1] > bounds[i]]


def add_suffstats(x, y):
    """Combine two :class:`~tspaint.accumulate.SuffStats` element-wise — the reduce combiner.

    The associative, commutative operation :func:`accumulate_parallel` folds the workers' partial
    sufficient statistics with, via :func:`functools.reduce` in genome-chunk order. ``S_dwell``,
    ``S_jumps``, ``S_root`` and ``loglik`` add element-wise; ``S_cred`` is merged per node (union of
    keys, arrays summed where a node appears in both). Inputs are not mutated. Float addition is not
    bit-associative, so the fold order matters at the ULP level — see the module docstring for the
    exactness contract.

    Parameters
    ----------
    x, y : tspaint.accumulate.SuffStats
        Partial sufficient statistics to combine.

    Returns
    -------
    tspaint.accumulate.SuffStats
        A new ``SuffStats`` holding the element-wise sum.
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


def _resolve_emissions(ts, emissions, labels, w, pi, mask=None):
    """The one emission-resolution rule, shared by the serial fallbacks and the workers.

    A caller-supplied ``emissions`` is honoured as given; ``mask`` overrides it (the mask must be
    applied to the *labels*, so a pre-built unmasked dict cannot express it). Using this in both
    branches is what keeps a parallel run equal to its serial counterpart — passing ``emissions``
    used to be silently ignored once the work moved into a worker.
    """
    if emissions is None or mask is not None:
        from .em import build_emissions
        return build_emissions(ts, labels, w, pi, mask)
    return emissions


def _accumulate_range(path, lo, hi, Q, pi, w, labels, soft_refs, mask=None, emissions=None):
    ts = _load_cached(path)
    emissions = _resolve_emissions(ts, emissions, labels, w, pi, mask)
    return accumulate_sufficient_statistics(ts, Q, pi, emissions, labels=labels,
                                            soft_refs=soft_refs, tree_range=(lo, hi))


def _paint_range(path, lo, hi, Q, pi, w, labels, focal, merge_tol, mask=None, emissions=None):
    ts = _load_cached(path)
    emissions = _resolve_emissions(ts, emissions, labels, w, pi, mask)
    return posterior_table(ts, Q, pi, emissions, focal=focal, merge_tol=merge_tol,
                           tree_range=(lo, hi))


def _date_member(path, labels, warm, edges, kwargs):
    from .dating import fit_rate_through_time
    ts = _load_cached(path)
    return fit_rate_through_time(ts, labels, edges, fit_result=warm, **kwargs)


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

    ``n_jobs`` is first passed through :func:`resolve_cores` (so ``None`` means all CPUs / the SLURM
    allocation); a resolved count ``<= 1`` returns ``None`` and the caller runs serially in-process.
    Otherwise a **process** pool (not threads — the GIL-bound Python tree-iteration hot loop needs
    processes) is built with ``max_workers`` = the resolved count and ``initializer=_pin_blas`` so
    each worker re-pins BLAS. No start method is forced, so the platform default is used (``spawn``
    on macOS / Windows).

    Parameters
    ----------
    n_jobs : int or None
        Requested worker count, resolved via :func:`resolve_cores`.

    Returns
    -------
    concurrent.futures.ProcessPoolExecutor or None
        A pool with the resolved number of workers, or ``None`` when that resolves to ``<= 1``.
    """
    n = resolve_cores(n_jobs)                # None -> CPU count / SLURM allocation
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

    Parameters
    ----------
    ts : tskit.TreeSequence or str or os.PathLike
        A loaded tree sequence (dumped to a temp file) or an existing filesystem path (yielded
        unchanged, no temp file).

    Yields
    ------
    str
        A path to a ``.trees`` file the worker processes can load; a temp file, if created, is
        deleted on context exit.
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
                        n_jobs=None, executor=None, exact=False):
    """Parallel genome E-step → pooled :class:`~tspaint.accumulate.SuffStats` (see module docstring).

    The map-reduce parallelisation of
    :func:`~tspaint.accumulate.accumulate_sufficient_statistics`: split the marginal trees into
    contiguous index ranges (:func:`genome_chunks`), accumulate each range in a worker, and fold the
    partials with :func:`add_suffstats`. When ``emissions`` is not given, workers rebuild them
    in-process from ``(labels, w, pi)``; pass an ``executor`` to reuse a pool across EM iterations.

    Parameters
    ----------
    ts, Q, pi, labels, soft_refs
        As for :func:`~tspaint.accumulate.accumulate_sufficient_statistics`.
    w : dict[int, float], optional
        Per-tip credibility for rebuilding the tip emissions in each worker (and on the serial
        fallback when ``emissions`` is not given); tips absent default to ``1.0``
        (:func:`tspaint.em.build_emissions`).
    emissions : dict[int, numpy.ndarray], optional
        Pre-built emissions, honoured identically on the serial and parallel branches (the dict is
        shipped to each worker). Default ``None`` — every worker rebuilds them from
        ``(labels, w, pi)``, which is what :func:`tspaint.fit` does.
    n_jobs : int, optional
        Worker count (:func:`resolve_cores`). Default ``None`` = all CPUs / the SLURM allocation;
        ``1`` runs the serial single loop.
    executor : concurrent.futures.Executor, optional
        A pool to reuse (e.g. across EM iterations); when ``None`` a pool is created and shut down
        per call. Supplying an ``executor`` also bypasses the ``n_jobs <= 1`` / single-tree serial
        shortcut (unless ``exact=True``).
    exact : bool, optional
        Force the serial single loop, byte-identical to
        :func:`~tspaint.accumulate.accumulate_sufficient_statistics`. Default ``False``.

    Returns
    -------
    tspaint.accumulate.SuffStats
        The pooled span-weighted sufficient statistics (``S_dwell``, ``S_jumps``, ``S_root``,
        ``S_cred``, ``loglik``) summed over all chunks.

    Notes
    -----
    Bit-exactness (module docstring): ``n_jobs == 1`` or ``exact=True`` is byte-identical to the
    serial single loop; ``n_jobs > 1`` is byte-identical to the *same chunk partition reduced
    serially* but differs from the serial single loop only by reduction order (a few ULP,
    ``allclose``), since float addition is not associative.
    """
    n_jobs = resolve_cores(n_jobs)
    serial = exact or (executor is None and (n_jobs <= 1 or ts.num_trees <= 1))
    if serial:
        emissions = _resolve_emissions(ts, emissions, labels, w, pi)
        return accumulate_sufficient_statistics(ts, Q, pi, emissions, labels=labels,
                                                soft_refs=soft_refs)

    own = executor is None
    ex = executor or make_pool(min(n_jobs, ts.num_trees))   # no more workers than genome chunks
    try:
        with as_path(ts) as path:
            futures = [ex.submit(_accumulate_range, path, lo, hi, Q, pi, w, labels, soft_refs,
                                 None, emissions)
                       for (lo, hi) in genome_chunks(ts, n_jobs)]
            partials = [f.result() for f in futures]      # blocks while the temp path is live
        return reduce(add_suffstats, partials)
    finally:
        if own:
            ex.shutdown()


def posterior_table_parallel(ts, Q, pi, *, w=None, labels=None, focal=None, merge_tol=1e-12,
                             emissions=None, n_jobs=None, executor=None, progress=False, mask=None):
    """Parallel :func:`~tspaint.output.posterior_table` — **exactly** equal to serial for any ``n_jobs``.

    Splits by tree-range, then stitches the per-range tracks in genome order, re-merging adjacent
    equal segments at the seams (the same merge rule serial painting uses). Each segment's posterior
    comes from its own tree's pruning, independent of the chunking, so the stitched result is
    **byte-identical** to serial (not merely ``allclose``) for any worker count.

    Parameters
    ----------
    ts, Q, pi, focal, merge_tol
        As for :func:`~tspaint.output.posterior_table`.
    w : dict[int, float], optional
        Per-tip credibility for rebuilding the tip emissions in each worker
        (:func:`tspaint.em.build_emissions`); tips absent default to ``1.0``.
    labels : dict[int, int], optional
        Per-tip labels, used with ``w`` / ``pi`` (and ``mask``) to rebuild emissions in-worker.
    emissions : dict[int, numpy.ndarray], optional
        Pre-built emissions, honoured identically on the serial and parallel branches (the dict is
        shipped to each worker). Ignored when ``mask`` is given — masking must be applied to the
        labels, so a pre-built unmasked dict cannot express it. Default ``None`` — rebuilt from
        ``(labels, w, pi, mask)``.
    mask : dict[int, list[tuple[float, float]]], optional
        Fragment masking ``{ref -> [(left, right), ...]}`` (CLAUDE.md §2.3): the flagged reference
        spans emit the query (unlabelled) emission. Threaded through to every worker.
    n_jobs : int, optional
        Worker count (:func:`resolve_cores`). Default ``None`` = all CPUs / the SLURM allocation;
        ``1`` runs serially.
    executor : concurrent.futures.Executor, optional
        A pool to reuse across calls; when ``None`` a pool is created and shut down per call.
    progress : bool, optional
        Show a :mod:`tqdm` bar. Default ``False``. On the serial fallback (``n_jobs <= 1``) it is
        per-marginal-tree; in parallel it is one tick per completed genome chunk (the per-tree loop
        runs inside worker subprocesses, so per-chunk is the finest granularity the parent sees).

    Returns
    -------
    dict[int, list[tspaint.output.Segment]]
        Per focal sample, the down-pass posterior as contiguous :class:`~tspaint.output.Segment`\\ s
        covering ``[0, L)`` — identical to :func:`~tspaint.output.posterior_table`.
    """
    n_jobs = resolve_cores(n_jobs)
    if executor is None and (n_jobs <= 1 or ts.num_trees <= 1):
        emissions = _resolve_emissions(ts, emissions, labels, w, pi, mask)
        return posterior_table(ts, Q, pi, emissions, focal=focal, merge_tol=merge_tol,
                               progress=progress)

    own = executor is None
    ex = executor or make_pool(min(n_jobs, ts.num_trees))   # no more workers than genome chunks
    try:
        with as_path(ts) as path:
            futures = [ex.submit(_paint_range, path, lo, hi, Q, pi, w, labels, focal, merge_tol,
                                 mask, emissions)
                       for (lo, hi) in genome_chunks(ts, n_jobs)]
            results = futures
            if progress:
                from tqdm.auto import tqdm
                results = tqdm(futures, desc="painting", unit="chunk")
            chunk_tracks = [f.result() for f in results]
        return _stitch_tracks(chunk_tracks, merge_tol)
    finally:
        if own:
            ex.shutdown()


def _loo_range(path, lo, hi, Q, pi, w, labels, focal, merge_tol, mask=None, emissions=None):
    from .output import loo_posterior_table
    ts = _load_cached(path)
    emissions = _resolve_emissions(ts, emissions, labels, w, pi, mask)
    return loo_posterior_table(ts, Q, pi, emissions, focal=focal, merge_tol=merge_tol,
                               tree_range=(lo, hi))


def loo_posterior_table_parallel(ts, Q, pi, *, w=None, labels=None, focal=None, merge_tol=1e-12,
                                 emissions=None, n_jobs=None, executor=None, progress=False, mask=None):
    """Parallel :func:`~tspaint.output.loo_posterior_table` — **exactly** equal to serial for any ``n_jobs``.

    The leave-one-out analogue of :func:`posterior_table_parallel`: split by tree-range (the outside
    message for each tree is independent of which chunk it lands in), then stitch the per-range tracks
    in genome order, re-merging adjacent equal segments at the seams. Byte-identical to serial (each
    segment's outside message comes from its own tree's pruning, independent of the chunking).

    Parameters
    ----------
    ts, Q, pi, focal, merge_tol
        As for :func:`~tspaint.output.loo_posterior_table`.
    w, labels, emissions, mask, n_jobs, executor, progress
        As for :func:`posterior_table_parallel` (``progress`` is per-marginal-tree on the serial
        fallback, one tick per genome chunk in parallel).

    Returns
    -------
    dict[int, list[tspaint.output.Segment]]
        Per focal sample, the leave-one-out (outside-message) posterior as contiguous
        :class:`~tspaint.output.Segment`\\ s covering ``[0, L)`` — identical to
        :func:`~tspaint.output.loo_posterior_table`.
    """
    from .output import loo_posterior_table
    n_jobs = resolve_cores(n_jobs)
    if executor is None and (n_jobs <= 1 or ts.num_trees <= 1):
        emissions = _resolve_emissions(ts, emissions, labels, w, pi, mask)
        return loo_posterior_table(ts, Q, pi, emissions, focal=focal, merge_tol=merge_tol,
                                   progress=progress)

    own = executor is None
    ex = executor or make_pool(min(n_jobs, ts.num_trees))   # no more workers than genome chunks
    try:
        with as_path(ts) as path:
            futures = [ex.submit(_loo_range, path, lo, hi, Q, pi, w, labels, focal, merge_tol,
                                 mask, emissions)
                       for (lo, hi) in genome_chunks(ts, n_jobs)]
            results = futures
            if progress:
                from tqdm.auto import tqdm
                results = tqdm(futures, desc="loo", unit="chunk")
            chunk_tracks = [f.result() for f in results]
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


def _stitch_foreignness_tracks(chunk_tracks, merge_tol):
    """Concatenate per-range ``ForeignnessSegment`` tracks (genome order), re-merging equal seams
    (same rule as :func:`tspaint.introgression.foreignness_track`)."""
    out = {}
    for chunk in chunk_tracks:
        for s, segs in chunk.items():
            dst = out.setdefault(s, [])
            for seg in segs:
                p = dst[-1] if dst else None
                same_depth = p is not None and (p.depth == seg.depth
                                                 or (np.isnan(p.depth) and np.isnan(seg.depth)))
                if (p is not None and p.right == seg.left and p.status == seg.status and same_depth
                        and np.allclose(p.loo, seg.loo, atol=merge_tol, rtol=0)):
                    p.right = seg.right
                else:
                    dst.append(seg)
    return out


def _stitch_depth_tracks(chunk_tracks):
    """Concatenate per-range ``(left, right, depth)`` tracks (genome order), re-merging equal seams
    (same rule as :func:`tspaint.archaic._depth_track`)."""
    out = {}
    for chunk in chunk_tracks:
        for s, segs in chunk.items():
            dst = out.setdefault(s, [])
            for (l, r, d) in segs:
                if dst:
                    pl, pr, pd = dst[-1]
                    if pr == l and (pd == d or (np.isnan(pd) and np.isnan(d))):
                        dst[-1] = (pl, r, d)
                        continue
                dst.append((l, r, d))
    return out


def _foreignness_range(path, lo, hi, Q, pi, w, labels, focal, merge_tol, emissions=None):
    from .introgression import foreignness_track
    ts = _load_cached(path)
    emissions = _resolve_emissions(ts, emissions, labels, w, pi)
    return foreignness_track(ts, Q, pi, emissions, labels, focal=focal, depth="time",
                             merge_tol=merge_tol, tree_range=(lo, hi))


def foreignness_track_parallel(ts, Q, pi, *, w=None, labels=None, emissions=None, focal=None,
                               depth="rank", merge_tol=1e-9, n_jobs=None, executor=None, progress=False):
    """Parallel :func:`~tspaint.introgression.foreignness_track` — exactly equal to serial.

    Splits by tree-range and stitches the per-range
    :class:`~tspaint.introgression.ForeignnessSegment` tracks in genome order (re-merging equal
    seams). Workers return the **raw** nearest-reference coalescence time; the genome-wide rank
    normalisation (``depth="rank"``) is applied on the parent over the *stitched* result, so the
    whole-genome rank is identical to serial rather than computed per chunk.

    Parameters
    ----------
    ts, Q, pi, labels, focal, depth, merge_tol
        As for :func:`~tspaint.introgression.foreignness_track`.
    w : dict[int, float], optional
        Per-tip credibility for rebuilding the tip emissions in each worker
        (:func:`tspaint.em.build_emissions`); tips absent default to ``1.0``.
    emissions : dict[int, numpy.ndarray], optional
        Pre-built emissions, honoured identically on the serial and parallel branches (the dict is
        shipped to each worker). Default ``None`` — rebuilt from ``(labels, w, pi)``.
    n_jobs, executor
        As for :func:`posterior_table_parallel`.
    progress : bool, optional
        Show a :mod:`tqdm` bar. Default ``False``. It ticks once per completed genome chunk in the
        parallel path; the serial fallback (``n_jobs <= 1``) shows no bar (the serial
        :func:`~tspaint.introgression.foreignness_track` has no progress hook).

    Returns
    -------
    dict[int, list[tspaint.introgression.ForeignnessSegment]]
        Per focal sample, the foreignness components as contiguous
        :class:`~tspaint.introgression.ForeignnessSegment`\\ s covering ``[0, L)`` — identical to
        :func:`~tspaint.introgression.foreignness_track`.
    """
    from .introgression import foreignness_track, _rank_normalise_depth
    n_jobs = resolve_cores(n_jobs)
    if executor is None and (n_jobs <= 1 or ts.num_trees <= 1):
        emissions = _resolve_emissions(ts, emissions, labels, w, pi)
        return foreignness_track(ts, Q, pi, emissions, labels, focal=focal, depth=depth,
                                 merge_tol=merge_tol)
    own = executor is None
    ex = executor or make_pool(min(n_jobs, ts.num_trees))   # no more workers than genome chunks
    try:
        with as_path(ts) as path:
            futures = [ex.submit(_foreignness_range, path, lo, hi, Q, pi, w, labels, focal,
                                 merge_tol, emissions)
                       for (lo, hi) in genome_chunks(ts, n_jobs)]
            results = futures
            if progress:
                from tqdm.auto import tqdm
                results = tqdm(futures, desc="foreignness", unit="chunk")
            chunk_tracks = [f.result() for f in results]
        stitched = _stitch_foreignness_tracks(chunk_tracks, merge_tol)
        if depth == "rank":
            _rank_normalise_depth(stitched)      # genome-wide, after stitching the raw depths
        return stitched
    finally:
        if own:
            ex.shutdown()


def _depth_track_range(path, lo, hi, modern_ids, focal):
    from .archaic import _depth_track
    ts = _load_cached(path)
    return _depth_track(ts, modern_ids, focal, tree_range=(lo, hi))


def depth_track_parallel(ts, modern_ids, samples, *, n_jobs=None, executor=None, progress=False):
    """Parallel :func:`~tspaint.archaic._depth_track` — exactly equal to serial.

    Splits the nearest-modern-reference coalescence pass by tree-range and stitches the per-range
    ``(left, right, depth)`` tracks in genome order (re-merging equal seams). ``depth`` is the
    **raw** coalescence time (``nan`` where no modern reference is reachable); any ``log`` / rank
    transform is applied downstream by the caller (:func:`tspaint.detect_ghost`), unaffected by the
    chunking.

    Parameters
    ----------
    ts, modern_ids, samples
        As for :func:`~tspaint.archaic._depth_track`.
    n_jobs, executor
        As for :func:`posterior_table_parallel`.
    progress : bool, optional
        Show a :mod:`tqdm` bar. Default ``False``. It ticks once per completed genome chunk in the
        parallel path; the serial fallback (``n_jobs <= 1``) shows no bar (the serial
        :func:`~tspaint.archaic._depth_track` has no progress hook).

    Returns
    -------
    dict[int, list[tuple[float, float, float]]]
        Per sample, contiguous ``(left, right, depth)`` segments (raw nearest-modern-ref coalescence
        time) covering ``[0, L)`` — identical to :func:`~tspaint.archaic._depth_track`.
    """
    from .archaic import _depth_track
    n_jobs = resolve_cores(n_jobs)
    if executor is None and (n_jobs <= 1 or ts.num_trees <= 1):
        return _depth_track(ts, modern_ids, samples)
    own = executor is None
    ex = executor or make_pool(min(n_jobs, ts.num_trees))   # no more workers than genome chunks
    try:
        with as_path(ts) as path:
            futures = [ex.submit(_depth_track_range, path, lo, hi, modern_ids, samples)
                       for (lo, hi) in genome_chunks(ts, n_jobs)]
            results = futures
            if progress:
                from tqdm.auto import tqdm
                results = tqdm(futures, desc="depth", unit="chunk")
            chunk_tracks = [f.result() for f in results]
        return _stitch_depth_tracks(chunk_tracks)
    finally:
        if own:
            ex.shutdown()


def _dating_estep_range(path, lo, hi, q, pi, labels, w, edges, mask=None):
    """Worker: the time-inhomogeneous dating E-step over one tree-index range ``[lo, hi)``."""
    from .dating.estep import accumulate_time_binned_tv
    from .dating.em import make_Q_of_cell
    ts = _load_cached(path)
    emissions = _resolve_emissions(ts, None, labels, w, pi, mask)
    return accumulate_time_binned_tv(ts, make_Q_of_cell(q), pi, emissions, edges,
                                     tree_range=(lo, hi))


def dating_estep_parallel(ts, q, pi, labels, w, edges, *, mask=None, path=None,
                          n_jobs=None, executor=None):
    """One time-inhomogeneous dating E-step, split across genome tree-ranges (the §3 tree-parallel
    E-step for :func:`tspaint.dating.fit_rate_through_time`).

    Splits the marginal trees into contiguous index ranges (:func:`genome_chunks`), runs
    :func:`~tspaint.dating.estep.accumulate_time_binned_tv` on each in a worker, and sums the
    per-cell dwell ``D``, directional jumps ``J`` and log-likelihood (all additive over
    tree-ranges). Bit-exact vs. the serial single loop up to the chunk reduction order (float ``+``
    is not associative — a few ULP, ``allclose``), like the painting accumulator.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence (its marginal trees are the parallel axis).
    q : (n_cells, K, K) numpy.ndarray
        The current per-cell off-diagonal rate array (shipped to every worker;
        :func:`tspaint.dating.make_Q_of_cell` rebuilds the generator in-worker).
    pi : (K,) array_like
        Root frequencies ``π``.
    labels : dict[int, int]
        Reference sample-node id -> ancestry state (workers rebuild the tip emissions).
    w : dict[int, float]
        Per-tip credibility for the emissions (:func:`tspaint.em.build_emissions`).
    edges : array_like
        Log-time grid cell edges.
    mask : dict, optional
        Fragment mask ``{ref -> [(left, right), ...]}`` applied when rebuilding emissions
        (CLAUDE.md §2.3). Default ``None``.
    path : str, optional
        A filesystem path to ``ts`` for the workers to load (avoids re-dumping it every EM
        iteration). Default ``None`` — ``ts`` is dumped to a temp file for this call.
    n_jobs : int, optional
        Worker count (:func:`resolve_cores`). Default ``None`` = all CPUs / the SLURM allocation.
    executor : concurrent.futures.Executor, optional
        A pool to reuse across EM iterations; when ``None`` a pool is created and shut down per call.

    Returns
    -------
    D : (n_cells, K) numpy.ndarray
    J : (n_cells, K, K) numpy.ndarray
    loglik : float
    """
    n_jobs = resolve_cores(n_jobs)
    if executor is None and (n_jobs <= 1 or ts.num_trees <= 1):
        from .dating.estep import accumulate_time_binned_tv
        from .dating.em import make_Q_of_cell
        emissions = _resolve_emissions(ts, None, labels, w, pi, mask)
        return accumulate_time_binned_tv(ts, make_Q_of_cell(q), pi, emissions, edges)

    own_pool = executor is None
    ex = executor or make_pool(min(n_jobs, ts.num_trees))
    with ExitStack() as stack:
        if own_pool:
            stack.callback(ex.shutdown)
        p = path if path is not None else stack.enter_context(as_path(ts))
        futures = [ex.submit(_dating_estep_range, p, lo, hi, q, pi, labels, w, edges, mask)
                   for (lo, hi) in genome_chunks(ts, n_jobs)]
        parts = [f.result() for f in futures]            # chunk order (deterministic)
    D = sum(p_[0] for p_ in parts)
    J = sum(p_[1] for p_ in parts)
    loglik = float(sum(p_[2] for p_ in parts))
    return D, J, loglik


def date_members_parallel(members, labels, warm, edges, kwargs=None, *, n_jobs=None):
    """Fit the rate-through-time of each ensemble member in parallel (one worker per member).

    The coarse axis of :meth:`tspaint.Painting.rate_through_time` for an ensemble: each member is
    dated independently on the shared warm fit and time grid, so this is a plain fan-out (no
    cross-member reduction — results are deterministic and order-preserving). ``n_jobs <= 1`` runs
    serially in-process (no pool, no temp dump).

    Parameters
    ----------
    members : list[tskit.TreeSequence]
        The ensemble members (e.g. SINGER posterior samples) to date.
    labels : dict[int, int]
        Reference sample-node id -> ancestry state, shared across members
        (:func:`tspaint.dating.fit_rate_through_time`).
    warm : tspaint.em.FitResult
        A precomputed homogeneous fit to warm-start every member from (forwarded as ``fit_result``),
        so no member refits the homogeneous model.
    edges : array_like
        Shared log-time grid edges, so the per-member profiles align on one axis and can be averaged
        (:func:`tspaint.dating.log_time_grid`).
    kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`tspaint.dating.fit_rate_through_time` for every
        member. A **plain positional parameter holding a dict** (not ``**kwargs``); ``None``
        (default) is treated as an empty dict.
    n_jobs : int, optional
        Worker count (:func:`resolve_cores`). Default ``None`` = all CPUs / the SLURM allocation;
        ``1`` (or a single member) runs serially in-process.

    Returns
    -------
    list[tspaint.dating.RateThroughTime]
        One fitted :class:`~tspaint.dating.RateThroughTime` per member, in member order.
    """
    from .dating import fit_rate_through_time
    kwargs = kwargs or {}
    n_jobs = resolve_cores(n_jobs)
    if n_jobs <= 1 or len(members) <= 1:
        return [fit_rate_through_time(g, labels, edges, fit_result=warm, **kwargs) for g in members]
    from contextlib import ExitStack
    with ExitStack() as stack:
        paths = [stack.enter_context(as_path(g)) for g in members]   # dump in-memory members once
        ex = stack.enter_context(make_pool(min(n_jobs, len(members))))   # one worker per member max
        futures = [ex.submit(_date_member, p, labels, warm, edges, kwargs) for p in paths]
        return [f.result() for f in futures]                         # member order; blocks in-context


def resolve_to_int(n_jobs):
    """``n_jobs`` as a worker count, treating ``None`` as serial (1) — *not* SLURM-resolved.

    Unlike :func:`resolve_cores`, this does **not** consult SLURM / ``os.cpu_count()`` — ``None``
    maps to ``1`` (serial) and any other value to ``max(1, int(n_jobs))``.

    Parameters
    ----------
    n_jobs : int or None
        Requested worker count; ``None`` means serial.

    Returns
    -------
    int
        ``1`` when ``n_jobs is None``, else ``max(1, int(n_jobs))``.
    """
    return 1 if n_jobs is None else max(1, int(n_jobs))
