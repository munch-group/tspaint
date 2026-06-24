"""Front-end-agnostic premise diagnostics (CLAUDE.md §5.1, §8.2).

The node-persistence histogram is the **go/no-go on the method's central premise**:
a clade must persist across many marginal trees as a single node ID (delivered
natively by msprime/tsinfer, or by Relate ``Convert --compress``). If persistence
is spiked at a single tree, the edge-blocking captures the double-counting fix but
loses the autocorrelation benefit (CLAUDE.md §5).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

__all__ = ["edge_span_summary", "node_persistence", "persistence_summary"]


def edge_span_summary(ts):
    """Summary of the tree-sequence edge-span distribution."""
    spans = ts.tables.edges.right - ts.tables.edges.left
    return {
        "median": float(np.median(spans)),
        "mean": float(spans.mean()),
        "min": float(spans.min()),
        "max": float(spans.max()),
        "n_edges": int(spans.size),
    }


def node_persistence(ts, include_samples=False):
    """Number of distinct marginal trees in which each (internal) node appears.

    Relies on cross-tree node-ID stability: a persistent clade keeps one id and so
    accumulates a count > 1.
    """
    counts = Counter()
    for tree in ts.trees():
        for u in tree.nodes():
            if include_samples or not tree.is_sample(u):
                counts[u] += 1
    if not counts:
        return np.array([], dtype=int)
    return np.fromiter(counts.values(), dtype=int, count=len(counts))


def persistence_summary(ts):
    """Summary of internal-node persistence; ``frac_singletons`` near 1 fails the premise."""
    counts = node_persistence(ts)
    if counts.size == 0:
        return {"median": 0.0, "max": 0, "frac_singletons": float("nan"), "n_internal": 0}
    return {
        "median": float(np.median(counts)),
        "max": int(counts.max()),
        "frac_singletons": float(np.mean(counts == 1)),
        "n_internal": int(counts.size),
    }
