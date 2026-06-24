"""Order-only (ranked-topology) ancestry-model variant — the CLAUDE.md §6 / §8.4 ablation.

tslai's CTMC rides on branch lengths via ``P(t) = expm(Q t)``. §6 worried that mis-calibrated
inferred branch lengths (Relate's panmictic prior; tsinfer's frequency-scale times) could bias
the fit, and proposed an order-only variant — replace each node's absolute time with the **dense
rank** of that time (coalescence ORDER only, magnitudes discarded) — as a robustness ablation,
in the spirit of Relate's order-based selection test. Sample ids and topology are preserved, so
labels/truth transfer unchanged.

**Measured result: the order-only variant is NOT beneficial — do not use it for inference.**
On a true ARG it collapses painting from ~1.0 to ~0.5. Dense-rank compresses the timescale, EM
compensates with a much larger Q, the deep/root branches then wash out, and π becomes
unidentifiable — it drifts to a degenerate extreme and the painting goes confidently wrong (the
diagnosis that mis-calibration would hurt was right; this cure is wrong). It is kept as the
runnable ablation behind ``tslai_paint(..., ranked=True)``; the actual robustness fix for the
same π failure is ``estimate_pi=False`` (hold π fixed — see :func:`tslai.em.fit`).
"""
from __future__ import annotations

import numpy as np

__all__ = ["ranked_tree_sequence"]


def ranked_tree_sequence(ts):
    """Copy of ``ts`` with node times replaced by the dense rank of each node's time
    (samples at time 0 → rank 0; the k-th oldest distinct time → rank k). Parent ranks
    stay strictly above child ranks (distinct times), so the result is a valid tree
    sequence; only the time scale changes."""
    tables = ts.dump_tables()
    nodes = tables.nodes
    _, ranks = np.unique(nodes.time, return_inverse=True)   # dense rank; min (tips) -> 0
    nodes.set_columns(flags=nodes.flags, time=ranks.astype(float),
                      population=nodes.population, individual=nodes.individual,
                      metadata=nodes.metadata, metadata_offset=nodes.metadata_offset)
    tables.sort()
    return tables.tree_sequence()
