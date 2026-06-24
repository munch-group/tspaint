"""High-level paint() / Painting API tests (CLAUDE.md §2.4)."""
import numpy as np
import pytest

import tslai
from tslai.sim import SOURCE_A, SOURCE_B, ADMIXED


def _admixture(L=5e5):
    ts = tslai.simulate_admixture(n_admix=6, n_ref=6, sequence_length=L, recombination_rate=1e-8,
                                  random_seed=1, Ne=1000, T_admix=30, T_split=5000, f_A=0.5)
    npop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in names.items() if n == SOURCE_A)
    B = next(p for p, n in names.items() if n == SOURCE_B)
    admix = next(p for p, n in names.items() if n == ADMIXED)
    sop = {A: 0, B: 1}
    labels = {int(s): sop[npop[s]] for s in ts.samples() if npop[s] in (A, B)}
    queries = [int(s) for s in ts.samples() if npop[s] == admix]
    truth = tslai.metrics.map_truth({q: tslai.local_ancestry_truth(ts)[0][q] for q in queries}, sop)
    return ts, labels, queries, truth


def test_tidy_namespaces_exposed():
    for ns in ("metrics", "compare", "io", "experiments"):
        assert hasattr(tslai, ns)
    assert callable(tslai.paint) and tslai.Painting is not None
    assert callable(tslai.metrics.balanced_accuracy)
    assert callable(tslai.compare.head_to_head)


@pytest.mark.slow
def test_paint_returns_painting_over_default_queries():
    ts, labels, queries, _ = _admixture()
    p = tslai.paint(ts, labels)                 # queries default to the non-labelled samples
    assert isinstance(p, tslai.Painting)
    assert set(p.posteriors) == set(queries)
    assert p.Q.shape == (2, 2) and p.pi.shape == (2,)
    for q in queries:
        segs = p.posteriors[q]
        assert segs[0].left == 0.0 and segs[-1].right == ts.sequence_length
        assert np.allclose(segs[0].posterior.sum(), 1.0)


@pytest.mark.slow
def test_painting_segments_deadband_and_accuracy():
    ts, labels, queries, truth = _admixture()
    p = tslai.paint(ts, labels, deadband=0.4)
    raw, db = p.segments(deadband=0.0), p.segments()     # default uses deadband=0.4

    def nsw(d):
        return sum(sum(1 for k in range(1, len(v)) if v[k][2] != v[k - 1][2]) for v in d.values())
    assert nsw(db) <= nsw(raw)                           # deadband never adds switches
    for q in queries:
        assert db[q][0][0] == 0.0 and db[q][-1][1] == ts.sequence_length
    # strong structure + recent admixture -> accurate painting
    assert tslai.metrics.balanced_accuracy(p.posteriors, truth, samples=queries) > 0.9
