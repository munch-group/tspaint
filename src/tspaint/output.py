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
           "missing_info_mask", "posterior_at", "hard_segments"]

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


def posterior_table(ts, Q, pi, emissions, focal=None, merge_tol=1e-12):
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

    Returns
    -------
    dict[int, list[Segment]]
        Per focal sample, the down-pass posterior as contiguous
        :class:`Segment`\\ s covering ``[0, L)``.
    """
    pi = np.asarray(pi, float)
    node_time = ts.tables.nodes.time
    samples = [int(s) for s in (ts.samples() if focal is None else focal)]
    tracks = {s: [] for s in samples}

    for tree in ts.trees():
        left = tree.interval.left
        right = tree.interval.right
        res = prune_tree(tree, emissions, Q, node_time, pi)
        for s in samples:
            post = res.gamma[s]
            status = MISSING_INFO if s in res.missing_info else INFORMATIVE
            segs = tracks[s]
            if (segs and segs[-1].right == left and segs[-1].status == status
                    and np.allclose(segs[-1].posterior, post, atol=merge_tol, rtol=0)):
                segs[-1].right = right
            else:
                segs.append(Segment(left, right, np.array(post, float), status))
    return tracks


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
