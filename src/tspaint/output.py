"""Deliverable: per-haplotype, per-position ancestry posteriors (CLAUDE.md §2.4,
§4) — Rung 6.

For every sample, at every position, the down-pass posterior over ancestry states,
returned as span-resolved :class:`Segment`s covering the whole sequence. Isolated
spans are tagged ``MISSING_INFO`` — the tree carries no information there, the
posterior falls back to the prior ``π``, and this is **distinct from a 50-50
uncertain call** (conflating them is a real interpretive error for introgression
mapping, CLAUDE.md §4.2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pruning import prune_tree

__all__ = ["Segment", "INFORMATIVE", "MISSING_INFO", "posterior_table",
           "loo_posterior_table", "missing_info_mask", "posterior_at", "hard_segments"]

INFORMATIVE = "informative"
MISSING_INFO = "missing-info"


@dataclass
class Segment:
    """A contiguous span carrying one soft ancestry posterior.

    Attributes
    ----------
    left, right : float
        Half-open genomic interval ``[left, right)`` the segment covers.
    posterior : numpy.ndarray
        ``(K,)`` posterior over ancestry states on ``[left, right)``.
    status : str
        :data:`INFORMATIVE` or :data:`MISSING_INFO` — the latter tags an isolated
        span where the tree carries no information and ``posterior`` is the prior
        ``π`` fallback, distinct from a 50-50 uncertain call (CLAUDE.md §4.2).
    """
    left: float
    right: float
    posterior: np.ndarray   # (K,) posterior over ancestry states on [left, right)
    status: str             # INFORMATIVE or MISSING_INFO


def _paint_tracks(ts, Q, pi, emissions, focal, merge_tol, pick, tree_range=None,
                  progress=False):
    """Shared engine for the per-sample segment painters.

    Runs the down-pass per marginal tree and records, for each focal sample, the
    posterior selected by ``pick(res, sample, prior)`` — ``γ`` for
    :func:`posterior_table`, the leave-one-out outside message for
    :func:`loo_posterior_table` — merging adjacent identical segments.

    ``tree_range=(lo, hi)`` restricts painting to that half-open marginal-tree-index range
    (the rest is skipped); since each segment's posterior comes from its own tree's pruning
    — independent of which trees a chunk covers — concatenating the per-range tracks in
    genome order and re-merging at the seams reproduces the full-genome result exactly
    (:func:`tspaint.parallel.posterior_table_parallel`).

    ``progress=True`` shows a per-marginal-tree :mod:`tqdm` bar; it is honoured only for the
    full-genome pass (``tree_range is None``), since a chunked worker (``tree_range`` set) is a
    subprocess whose progress is reported per-chunk by the parent (:func:`posterior_table_parallel`).
    """
    pi = np.asarray(pi, float)
    node_time = ts.tables.nodes.time
    samples = [int(s) for s in (ts.samples() if focal is None else focal)]
    tracks = {s: [] for s in samples}

    lo, hi = (0, ts.num_trees) if tree_range is None else tree_range
    tree_iter = ts.trees()
    if progress and tree_range is None:
        from tqdm import tqdm
        tree_iter = tqdm(tree_iter, total=(hi - lo), desc="painting", unit="tree")
    for ti, tree in enumerate(tree_iter):
        if ti < lo:
            continue
        if ti >= hi:
            break
        left = tree.interval.left
        right = tree.interval.right
        res = prune_tree(tree, emissions, Q, node_time, pi)
        for s in samples:
            post = pick(res, s, pi)
            status = MISSING_INFO if s in res.missing_info else INFORMATIVE
            segs = tracks[s]
            if (segs and segs[-1].right == left and segs[-1].status == status
                    and np.allclose(segs[-1].posterior, post, atol=merge_tol, rtol=0)):
                segs[-1].right = right
            else:
                segs.append(Segment(left, right, np.array(post, float), status))
    return tracks


def posterior_table(ts, Q, pi, emissions, focal=None, merge_tol=1e-12, tree_range=None,
                    progress=False):
    """Per-sample ancestry posterior as contiguous segments covering ``[0, L)``.

    Runs the down-pass per marginal tree and records each focal sample's posterior
    and info-status; adjacent identical segments are merged.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence whose marginal trees are pruned.
    Q : (K, K) numpy.ndarray
        Ancestry CTMC generator.
    pi : (K,) array_like
        Root frequencies ``π`` (the prior fallback on uninformative spans).
    emissions : dict[int, numpy.ndarray]
        Per-sample emission vector (see :func:`tspaint.em.build_emissions`).
    focal : iterable[int], optional
        Samples to record; defaults to every sample in ``ts``.
    merge_tol : float, optional
        Absolute tolerance for merging adjacent segments with equal posterior.
        Default ``1e-12``.
    progress : bool, optional
        Show a per-marginal-tree :mod:`tqdm` progress bar (full-genome pass only).
        Default ``False``.

    Returns
    -------
    dict[int, list[Segment]]
        Per focal sample, the down-pass posterior as contiguous
        :class:`Segment`\\ s covering ``[0, L)``.
    """
    return _paint_tracks(ts, Q, pi, emissions, focal, merge_tol,
                         lambda res, s, prior: res.gamma[s], tree_range=tree_range,
                         progress=progress)


def loo_posterior_table(ts, Q, pi, emissions, focal=None, merge_tol=1e-12, tree_range=None):
    """Per-sample **leave-one-out** ancestry posterior as contiguous segments.

    Like :func:`posterior_table` but paints the *outside message* — what the rest of
    the tree says about each focal sample's ancestry, **excluding that sample's own
    emission** (``PruneResult.loo``). For a labelled reference this is the
    introgression / mislabel map: where its own genealogy dissents from its label
    (CLAUDE.md §2.3, §9). Unlike the down-pass :func:`posterior_table`, it is *not*
    suppressed by a confident (e.g. hard-clamped, one-hot) tip emission, so it surfaces a
    reference's foreign tracts even where the down-pass posterior is pinned to the label.

    Parameters
    ----------
    ts, Q, pi, emissions, focal, merge_tol
        As for :func:`posterior_table`.

    Returns
    -------
    dict[int, list[Segment]]
        Per focal sample, the leave-one-out posterior as contiguous
        :class:`Segment`\\ s covering ``[0, L)``.
    """
    return _paint_tracks(ts, Q, pi, emissions, focal, merge_tol,
                         lambda res, s, prior: res.loo.get(s, prior), tree_range=tree_range)


def missing_info_mask(ts, focal=None):
    """Per-sample spans where the tree is uninformative (isolated sample).

    Topology-only (independent of ``Q``/``π``/emissions): an isolated sample is a
    root with no children over that span (CLAUDE.md §4.2).

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence to scan.
    focal : iterable[int], optional
        Samples to report; defaults to every sample in ``ts``.

    Returns
    -------
    dict[int, list[tuple[float, float]]]
        Per sample, the ``(left, right)`` spans where it is an isolated root.
    """
    samples = [int(s) for s in (ts.samples() if focal is None else focal)]
    mask = {s: [] for s in samples}
    for tree in ts.trees():
        left, right = tree.interval.left, tree.interval.right
        roots = set(tree.roots)
        for s in samples:
            if s in roots and len(tree.children(s)) == 0:
                spans = mask[s]
                if spans and spans[-1][1] == left:
                    spans[-1] = (spans[-1][0], right)
                else:
                    spans.append((left, right))
    return mask


def posterior_at(tracks, sample, position):
    """Posterior vector for ``sample`` at a genomic ``position``.

    Parameters
    ----------
    tracks : dict[int, list[Segment]]
        Per-sample segment tracks (e.g. from :func:`posterior_table`).
    sample : int
        Sample-node id to look up.
    position : float
        Genomic position.

    Returns
    -------
    numpy.ndarray or None
        The ``(K,)`` posterior on the segment covering ``position``, or ``None``
        if no segment covers it.
    """
    for seg in tracks[int(sample)]:
        if seg.left <= position < seg.right:
            return seg.posterior
    return None


def hard_segments(track, deadband=0.0):
    """Collapse a soft posterior ``track`` into hard ancestry segments.

    Produces ``[(left, right, state), ...]`` — the object a downstream tract-length /
    admixture-pulse dating analysis consumes.

    Parameters
    ----------
    track : list[Segment]
        Soft posterior segments for one sample (e.g. from :func:`posterior_table`).
    deadband : float, optional
        Confidence dead-band on the top-two posterior **margin**
        ``max(P) - 2nd-max(P)``. A switch to a new ``argmax`` state is accepted only
        where ``margin >= deadband``; otherwise the previous state is carried forward.
        Default ``0.0`` (naive ``argmax``).

    Returns
    -------
    list[tuple[float, float, int]]
        Hard ``(left, right, state)`` ancestry segments.

    Notes
    -----
    A positive ``deadband`` suppresses the low-confidence (~P=0.5) flips that fragment
    long tracts under naive ``argmax`` (``deadband=0``). Because the posterior is
    calibrated, a modest dead-band (≈0.3–0.5) recovers the true switch density and
    tract-length distribution where naive argmax over-fragments ~3× (CLAUDE.md §9).
    This is a tunable precision/recall dial a fixed hard segmenter (e.g. RFMix's CRF)
    does not expose. ``MISSING_INFO`` spans carry the previous state forward; the first
    interval takes its ``argmax``.
    """
    segs, cur = [], None
    for seg in track:
        p = np.asarray(seg.posterior, float)
        st = int(np.argmax(p))
        if cur is not None:
            order = np.sort(p)
            margin = float(order[-1] - order[-2])
            if seg.status == MISSING_INFO or margin < deadband:
                st = cur
        cur = st
        if segs and segs[-1][2] == st and segs[-1][1] == seg.left:
            segs[-1] = (segs[-1][0], seg.right, st)
        else:
            segs.append((seg.left, seg.right, st))
    return segs
