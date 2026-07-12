"""Edge-blocked, span-weighted sufficient statistics (CLAUDE.md §3.3) — Rung 4.

THE correctness core. Drive the genome with ``ts.edge_diffs()`` zipped with
``ts.trees()``; run pruning (:mod:`tspaint.pruning`) per tree; **bank each edge's
contribution once, on entry (``edges_in``), weighted by its own span**. A clade
persisting across many trees is one set of wide-span edges, so summing by span
partitions the genome without double-counting (the edge-table invariant: the set
of intervals on which each node is a child is disjoint). Root-state mass is
accumulated per interval (CLAUDE.md §3.3 sketch).

The expensive Van Loan call (:func:`tspaint.branch_stats.branch_expected_stats` — **the
Phasic seam**, §3.2/§12) is thereby made **once per edge**, not once per (tree x branch).
Any memoisation on ``(Q, t)`` lives *behind* that call, inside the backend, so the seam
stays a plain ``(Q, t, xi) -> (dwell, jumps)`` function and a replacement is free to cache
however suits it. The blocked approximation (CLAUDE.md §3.5): an edge's ``ξ`` is taken from
the tree at entry and held over its whole span — exact when the topology outside the edge
does not change across the span; the residual is the breakpoint flicker measured in Rung 8.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tskit

from .branch_stats import branch_expected_stats
from .model import emissions_for
from .pruning import prune_tree, _transition_cache

__all__ = ["SuffStats", "accumulate_sufficient_statistics"]


@dataclass
class SuffStats:
    """Pooled, span-weighted sufficient statistics for the M-step (CLAUDE.md §3.3).

    Attributes
    ----------
    S_dwell : numpy.ndarray, shape (K,)
        Expected dwell per state, span-weighted.
    S_jumps : numpy.ndarray, shape (K, K)
        Expected jumps per ordered pair, span-weighted.
    S_root : numpy.ndarray, shape (K,)
        Expected root-state mass, span-weighted.
    S_cred : dict
        ``node -> array([agree, disagree])`` credibility evidence for the Beta update.
    loglik : float
        Span-integrated per-locus log-likelihood (diagnostic).
    """
    S_dwell: np.ndarray      # (K,)   expected dwell per state, span-weighted
    S_jumps: np.ndarray      # (K, K) expected jumps per ordered pair, span-weighted
    S_root: np.ndarray       # (K,)   expected root-state mass, span-weighted
    S_cred: dict             # node -> array([agree, disagree]) for the Beta update
    loglik: float            # span-integrated per-locus log-likelihood (diagnostic)


def accumulate_sufficient_statistics(ts, Q, pi, emissions, *, labels=None,
                                     soft_refs=None, tree_range=None):
    """Single E-step sweep over the tree sequence (CLAUDE.md §3.3).

    Drives the genome with ``ts.edge_diffs()`` zipped with ``ts.trees()``, prunes each
    marginal tree, and banks each edge's contribution **once on entry, weighted by its
    own span** — so a clade persisting across many trees is counted once (the
    double-counting fix and the channel for genome-scale autocorrelation). Each edge's
    dwell/jump statistics come from one call to the seam,
    :func:`tspaint.branch_stats.branch_expected_stats`; any memoisation on ``(Q, t)`` is
    the backend's business, not ours. Root-state mass is accumulated per interval, not
    per edge.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The (``--compress``ed Relate or tsinfer-native) tree sequence.
    Q : numpy.ndarray, shape (K, K)
        CTMC generator (rows sum to zero).
    pi : numpy.ndarray, shape (K,)
        Root frequencies.
    emissions : dict[int, array]
        Per-tip emission vectors (labelled refs and queries), as built by
        :mod:`tspaint.model`.
    labels : dict[int, int], optional
        Label index per labelled tip, used to accumulate credibility evidence.
    soft_refs : set[int], optional
        Restrict credibility accumulation to these tips (default: all labelled).
    tree_range : tuple[int, int], optional
        ``(lo, hi)`` half-open marginal-tree-index range to process; the rest of the
        genome is iterated but skipped. Default (``None``) processes every tree. The
        per-tree arithmetic is **unchanged** by this gate, so a partition of ``[0,
        num_trees)`` into contiguous ranges sums (per :func:`tspaint.parallel.add_suffstats`)
        to the full-genome result with the same in-range addition order — the basis of the
        bit-exact parallel E-step (:mod:`tspaint.parallel`).

    Returns
    -------
    SuffStats
        Span-weighted ``S_dwell``, ``S_jumps``, ``S_root``, per-tip credibility
        ``S_cred`` and the span-integrated ``loglik``.
    """
    pi = np.asarray(pi, float)
    K = pi.shape[0]
    node_time = ts.tables.nodes.time
    labels = labels or {}

    S_dwell = np.zeros(K)
    S_jumps = np.zeros((K, K))
    S_root = np.zeros(K)
    S_cred = {}
    loglik = 0.0
    Pget = _transition_cache(Q)   # shared across all trees: expm(Q t) once per distinct t

    lo, hi = (0, ts.num_trees) if tree_range is None else tree_range
    for ti, ((interval, _edges_out, edges_in), tree) in enumerate(zip(ts.edge_diffs(), ts.trees())):
        if ti < lo:
            continue                      # skip pruning; advance the diff/tree iterators only
        if ti >= hi:
            break                         # this worker's range is done
        left, right = interval
        span = right - left
        res = prune_tree(tree, emissions_for(emissions, left, right), Q, node_time, pi, Pget=Pget)
        loglik += span * res.loglik

        # Bank each entering edge's contribution ONCE, weighted by its own span.
        for e in edges_in:
            c, p = e.child, e.parent
            if tree.parent(c) == tskit.NULL:      # defensive: child-edges are never root branches
                continue
            t = node_time[p] - node_time[c]
            if t <= 0:                     # root branch (length 0 by convention); skip (§3.4)
                continue
            xi = res.xi[(p, c)]
            w_edge = e.right - e.left
            dwell, jumps = branch_expected_stats(Q, t, xi)   # THE SEAM (§3.2, §12)
            S_dwell += w_edge * dwell
            S_jumps += w_edge * jumps

            if c in labels and (soft_refs is None or c in soft_refs):
                # leave-one-out: judge the label against what the REST of the tree says
                # about this tip, not its own (self-confirming) posterior (§2.3)
                agree = res.loo[c][labels[c]]
                if c not in S_cred:
                    S_cred[c] = np.zeros(2)
                S_cred[c] += w_edge * np.array([agree, 1.0 - agree])

        # Root-state mass for this interval's roots (per interval, not per edge).
        for r in tree.roots:
            S_root += span * res.root_marginal[r]

    return SuffStats(S_dwell, S_jumps, S_root, S_cred, float(loglik))
