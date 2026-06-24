"""Edge-blocked, span-weighted sufficient statistics (CLAUDE.md §3.3) — Rung 4.

THE correctness core. Drive the genome with ``ts.edge_diffs()`` zipped with
``ts.trees()``; run pruning (:mod:`tslai.pruning`) per tree; **bank each edge's
contribution once, on entry (``edges_in``), weighted by its own span**. A clade
persisting across many trees is one set of wide-span edges, so summing by span
partitions the genome without double-counting (the edge-table invariant: the set
of intervals on which each node is a child is disjoint). Root-state mass is
accumulated per interval (CLAUDE.md §3.3 sketch).

The expensive Van Loan call (:func:`tslai.branch_stats.branch_expected_stats`) is
thereby made **once per edge**, not once per (tree x branch). The blocked
approximation (CLAUDE.md §3.5): an edge's ``ξ`` is taken from the tree at entry and
held over its whole span — exact when the topology outside the edge does not change
across the span; the residual is the breakpoint flicker to measure in Rung 8.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tskit

from .branch_stats import branch_kernel, stats_from_kernel
from .pruning import prune_tree, _transition_cache

__all__ = ["SuffStats", "accumulate_sufficient_statistics"]


@dataclass
class SuffStats:
    S_dwell: np.ndarray      # (K,)   expected dwell per state, span-weighted
    S_jumps: np.ndarray      # (K, K) expected jumps per ordered pair, span-weighted
    S_root: np.ndarray       # (K,)   expected root-state mass, span-weighted
    S_cred: dict             # node -> array([agree, disagree]) for the Beta update
    loglik: float            # span-integrated per-locus log-likelihood (diagnostic)


def accumulate_sufficient_statistics(ts, Q, pi, emissions, *, labels=None,
                                     soft_refs=None):
    """Single E-step sweep over the tree sequence (CLAUDE.md §3.3).

    Parameters
    ----------
    ts : tskit.TreeSequence
    Q, pi : generator and root frequencies.
    emissions : dict[int, array]
        Per-tip emission vectors (labelled refs and queries), as built by
        :mod:`tslai.model`.
    labels : dict[int, int], optional
        Label index per labelled tip, used to accumulate credibility evidence.
    soft_refs : set[int], optional
        Restrict credibility accumulation to these tips (default: all labelled).

    Returns
    -------
    SuffStats
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
    kernel_cache = {}             # Van Loan branch kernel once per distinct branch length

    def kernel_for(t):
        key = float(t)
        if key not in kernel_cache:
            kernel_cache[key] = branch_kernel(Q, t)
        return kernel_cache[key]

    for (interval, _edges_out, edges_in), tree in zip(ts.edge_diffs(), ts.trees()):
        left, right = interval
        span = right - left
        res = prune_tree(tree, emissions, Q, node_time, pi, Pget=Pget)
        loglik += span * res.loglik

        # Bank each entering edge's contribution ONCE, weighted by its own span.
        for e in edges_in:
            c, p = e.child, e.parent
            if tree.parent(c) == tskit.NULL:      # defensive: child-edges are never root branches
                continue
            t = node_time[p] - node_time[c]
            kern = kernel_for(t)
            if kern is None:               # root branch (t <= 0); skip (§3.4)
                continue
            xi = res.xi[(p, c)]
            w_edge = e.right - e.left
            dwell, jumps = stats_from_kernel(kern, xi)
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
