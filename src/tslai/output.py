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
           "missing_info_mask", "posterior_at"]

INFORMATIVE = "informative"
MISSING_INFO = "missing-info"


@dataclass
class Segment:
    left: float
    right: float
    posterior: np.ndarray   # (K,) posterior over ancestry states on [left, right)
    status: str             # INFORMATIVE or MISSING_INFO


def posterior_table(ts, Q, pi, emissions, focal=None, merge_tol=1e-12):
    """Per-sample ancestry posterior as contiguous segments covering ``[0, L)``.

    Runs the down-pass per marginal tree and records each focal sample's posterior
    and info-status; adjacent identical segments are merged.

    Returns
    -------
    dict[int, list[Segment]]
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
    """Posterior vector for ``sample`` at a genomic ``position`` (or ``None``)."""
    for seg in tracks[int(sample)]:
        if seg.left <= position < seg.right:
            return seg.posterior
    return None
