"""Per-locus foreignness diagnostics for references and queries (CLAUDE.md §2.3, §9).

The shared engine (Plan A) behind reference QC, anonymous foreign-tract inference, and
ghost-source detection. For each focal sample, per genomic segment, three components:

* ``loo``   — the leave-one-out posterior (the outside message,
  :func:`tspaint.output.loo_posterior_table`): what the rest of the genealogy says about the
  tip *ignoring its own label*.
* ``fit``   — ``max_s loo[s]``: the genealogy's confidence in *any* panel state. Low ``fit``
  (≈ ``1/K``) means the tract fits no reference well ("fits nothing").
* ``depth`` — coalescence depth to the **nearest labelled reference**, rank-normalised
  genome-wide by default (calibration-robust; ``depth="time"`` keeps raw coalescent time).
  High ``depth`` = a deep outlier.

The separation that matters: a merely *uninformative* tract has low ``fit`` but **shallow**
``depth``; a *ghost / archaic* tract has low ``fit`` AND **high** ``depth``. That is why
``depth`` is carried alongside ``fit`` — it distinguishes "fits nothing because the local
tree can't tell" from "fits nothing because it descends from a population not in the panel".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tskit

from .pruning import prune_tree
from .output import INFORMATIVE, MISSING_INFO

__all__ = ["ForeignnessSegment", "foreignness_track"]


@dataclass
class ForeignnessSegment:
    """A contiguous span carrying the per-locus foreignness components for one sample.

    Attributes
    ----------
    left, right : float
        Half-open genomic interval ``[left, right)``.
    loo : numpy.ndarray
        ``(K,)`` leave-one-out posterior (the outside message excluding the sample's own
        emission).
    fit : float
        ``max_s loo[s]`` in ``[1/K, 1]`` — the genealogy's confidence in any panel state.
    depth : float
        Coalescence depth to the nearest labelled reference: a genome-wide rank in
        ``[0, 1]`` (``depth="rank"``, default) or raw coalescent time (``depth="time"``);
        ``nan`` where no reference is reachable.
    status : str
        :data:`tspaint.output.INFORMATIVE` or :data:`tspaint.output.MISSING_INFO`.
    """
    left: float
    right: float
    loo: np.ndarray
    fit: float
    depth: float
    status: str


def _nearest_ref_depth(tree, s, ref_ids, node_time):
    """Coalescent time to the nearest labelled reference (excluding ``s`` itself)."""
    best = np.inf
    for r in ref_ids:
        if r == s:
            continue
        m = tree.mrca(s, r)
        if m == tskit.NULL:
            continue
        t = node_time[m]
        if t < best:
            best = t
    return float(best) if np.isfinite(best) else float("nan")


def _rank_normalise_depth(tracks):
    """Replace raw nearest-ref depths with their span-weighted genome-wide quantile in ``[0, 1]``."""
    depths, spans = [], []
    for segs in tracks.values():
        for seg in segs:
            if seg.status == INFORMATIVE and np.isfinite(seg.depth):
                depths.append(seg.depth)
                spans.append(seg.right - seg.left)
    if not depths:
        for segs in tracks.values():
            for seg in segs:
                seg.depth = float("nan")
        return
    depths = np.asarray(depths, float)
    spans = np.asarray(spans, float)
    order = np.argsort(depths, kind="mergesort")
    sd = depths[order]
    csum = np.cumsum(spans[order])
    total = csum[-1]
    for segs in tracks.values():
        for seg in segs:
            if seg.status == INFORMATIVE and np.isfinite(seg.depth):
                idx = int(np.searchsorted(sd, seg.depth, side="right"))
                seg.depth = float(csum[idx - 1] / total) if idx > 0 else 0.0
            else:
                seg.depth = float("nan")


def foreignness_track(ts, Q, pi, emissions, labels, focal=None, depth="rank", merge_tol=1e-9):
    """Per-sample foreignness components as contiguous segments covering ``[0, L)``.

    Single pass over the marginal trees (one pruning per tree); for each focal sample records
    the leave-one-out posterior ``loo``, the ``fit = max_s loo[s]`` and the nearest-reference
    coalescence ``depth``. Adjacent segments with equal components are merged.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence whose marginal trees are pruned.
    Q : (K, K) numpy.ndarray
        Ancestry CTMC generator.
    pi : (K,) array_like
        Root frequencies ``π`` (prior fallback on uninformative spans).
    emissions : dict[int, numpy.ndarray]
        Per-sample emission vectors (e.g. from :func:`tspaint.em.build_emissions`).
    labels : dict[int, int] or iterable[int]
        The labelled reference sample ids (only the keys / ids are used — they define which
        tips the nearest-reference depth is measured against).
    focal : iterable[int], optional
        Samples to record; defaults to every sample in ``ts``.
    depth : {"rank", "time"}, optional
        ``"rank"`` (default) rank-normalises the nearest-reference depth genome-wide into
        ``[0, 1]`` (robust to branch-length miscalibration, CLAUDE.md §6); ``"time"`` keeps the
        raw coalescent time.
    merge_tol : float, optional
        Absolute tolerance for merging adjacent segments with equal ``loo``.

    Returns
    -------
    dict[int, list[ForeignnessSegment]]
        Per focal sample, the foreignness components as contiguous
        :class:`ForeignnessSegment`\\ s covering ``[0, L)``.
    """
    if depth not in ("rank", "time"):
        raise ValueError("depth must be 'rank' or 'time'")
    pi = np.asarray(pi, float)
    node_time = ts.tables.nodes.time
    ref_ids = [int(r) for r in labels]
    samples = [int(s) for s in (ts.samples() if focal is None else focal)]
    tracks = {s: [] for s in samples}

    for tree in ts.trees():
        left, right = tree.interval.left, tree.interval.right
        res = prune_tree(tree, emissions, Q, node_time, pi)
        for s in samples:
            loo = np.asarray(res.loo.get(s, pi), float)
            fit = float(loo.max())
            status = MISSING_INFO if s in res.missing_info else INFORMATIVE
            d = _nearest_ref_depth(tree, s, ref_ids, node_time)
            segs = tracks[s]
            same_depth = (segs and ((segs[-1].depth == d)
                                    or (np.isnan(segs[-1].depth) and np.isnan(d))))
            if (segs and segs[-1].right == left and segs[-1].status == status and same_depth
                    and np.allclose(segs[-1].loo, loo, atol=merge_tol, rtol=0)):
                segs[-1].right = right
            else:
                segs.append(ForeignnessSegment(left, right, loo, fit, d, status))

    if depth == "rank":
        _rank_normalise_depth(tracks)
    return tracks
