"""Order-only / ranked-topology variant (tslai.ranked) and the estimate_pi=False
robustness fix it surfaced (CLAUDE.md §6)."""
import numpy as np
import tskit

from tslai.em import fit
from tslai.ranked import ranked_tree_sequence


def _build_ts(parents, times, samples, L=10.0):
    t = tskit.TableCollection(sequence_length=L)
    for u in range(len(times)):
        flags = tskit.NODE_IS_SAMPLE if u in samples else 0
        t.nodes.add_row(flags=flags, time=times[u])
    for c, p in parents.items():
        if p != -1:
            t.edges.add_row(0, L, p, c)
    t.sort()
    return t.tree_sequence()


def test_ranked_preserves_samples_and_topology():
    ts = _build_ts({0: 3, 1: 3, 2: 4, 3: 4, 4: -1}, [0, 0, 0, 1.0, 2.0], {0, 1, 2})
    rts = ranked_tree_sequence(ts)
    assert list(rts.samples()) == list(ts.samples())
    assert rts.num_trees == ts.num_trees
    to, tr = ts.first(), rts.first()
    assert {u: to.parent(u) for u in to.nodes()} == {u: tr.parent(u) for u in tr.nodes()}


def test_ranked_compresses_times_to_dense_rank():
    # unevenly spaced distinct times -> contiguous integer ranks (tips at 0)
    ts = _build_ts({0: 3, 1: 3, 2: 4, 3: 4, 4: -1}, [0, 0, 0, 0.3, 17.0], {0, 1, 2})
    rts = ranked_tree_sequence(ts)
    assert sorted(set(rts.tables.nodes.time)) == [0.0, 1.0, 2.0]
    assert list(rts.tables.nodes.time) == [0.0, 0.0, 0.0, 1.0, 2.0]   # parent ranks > child


def test_estimate_pi_false_holds_pi_fixed():
    # estimate_pi=False must leave π at π0 (the robustness fix); True keeps it a distribution.
    ts = _build_ts({0: 2, 1: 2, 2: -1}, [0, 0, 1.0], {0, 1})
    labels = {0: 0, 1: 1}
    pi0 = np.array([0.7, 0.3])
    fixed = fit(ts, labels, pi0=pi0, max_iter=5, estimate_pi=False)
    assert np.allclose(fixed.pi, pi0)
    free = fit(ts, labels, pi0=pi0, max_iter=5, estimate_pi=True)
    assert np.isclose(free.pi.sum(), 1.0)
