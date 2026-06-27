"""High-level paint() / Painting API tests (CLAUDE.md §2.4)."""
import numpy as np
import pytest

import tspaint
from tspaint.sim import SOURCE_A, SOURCE_B, ADMIXED


def _admixture(L=5e5):
    ts = tspaint.simulate_admixture(n_admix=6, n_ref=6, sequence_length=L, recombination_rate=1e-8,
                                  random_seed=1, Ne=1000, T_admix=30, T_split=5000, f_A=0.5)
    npop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in names.items() if n == SOURCE_A)
    B = next(p for p, n in names.items() if n == SOURCE_B)
    admix = next(p for p, n in names.items() if n == ADMIXED)
    sop = {A: 0, B: 1}
    labels = {int(s): sop[npop[s]] for s in ts.samples() if npop[s] in (A, B)}
    queries = [int(s) for s in ts.samples() if npop[s] == admix]
    truth = tspaint.metrics.map_truth({q: tspaint.local_ancestry_truth(ts)[0][q] for q in queries}, sop)
    return ts, labels, queries, truth


def test_tidy_namespaces_exposed():
    for ns in ("metrics", "compare", "io", "experiments"):
        assert hasattr(tspaint, ns)
    assert callable(tspaint.paint) and tspaint.Painting is not None
    assert callable(tspaint.metrics.balanced_accuracy)
    assert callable(tspaint.compare.head_to_head)


@pytest.mark.slow
def test_paint_returns_painting_over_default_queries():
    ts, labels, queries, _ = _admixture()
    p = tspaint.paint(ts, labels)                 # queries default to the non-labelled samples
    assert isinstance(p, tspaint.Painting)
    assert set(p.posteriors) == set(queries)
    assert p.Q.shape == (2, 2) and p.pi.shape == (2,)
    for q in queries:
        segs = p.posteriors[q]
        assert segs[0].left == 0.0 and segs[-1].right == ts.sequence_length
        assert np.allclose(segs[0].posterior.sum(), 1.0)


@pytest.mark.slow
def test_painting_segments_deadband_and_accuracy():
    ts, labels, queries, truth = _admixture()
    p = tspaint.paint(ts, labels, deadband=0.4)
    raw, db = p.segments(deadband=0.0), p.segments()     # default uses deadband=0.4

    def nsw(d):
        return sum(sum(1 for k in range(1, len(v)) if v[k][2] != v[k - 1][2]) for v in d.values())
    assert nsw(db) <= nsw(raw)                           # deadband never adds switches
    for q in queries:
        assert db[q][0][0] == 0.0 and db[q][-1][1] == ts.sequence_length
    # strong structure + recent admixture -> accurate painting
    assert tspaint.metrics.balanced_accuracy(p.posteriors, truth, samples=queries) > 0.9


@pytest.mark.slow
def test_paint_smooth_option_reduces_switches():
    ts, labels, queries, _ = _admixture()
    plain = tspaint.paint(ts, labels)
    smoothed = tspaint.paint(ts, labels, smooth=True)         # horizontal BP smoother (CLAUDE.md §7)
    assert set(smoothed.posteriors) == set(queries)

    def nsw(P):
        return sum(sum(1 for k in range(1, len(v)) if v[k][2] != v[k - 1][2])
                   for v in P.segments().values())
    assert nsw(smoothed) <= nsw(plain)


# --- ensemble input: paint() accepts a list of tree sequences -------------------------------

def test_paint_empty_ensemble_raises():
    ts, labels, _, _ = _admixture(L=1e5)
    with pytest.raises(ValueError, match="empty ensemble"):
        tspaint.paint([], labels)


@pytest.mark.slow
def test_paint_ensemble_mean_matches_single():
    """A degenerate ensemble of identical members must equal the single-ts painting (the
    M-step is scale-invariant) with a zero uncertainty band."""
    ts, labels, queries, _ = _admixture(L=1e5)
    single = tspaint.paint(ts, labels, queries)
    ens = tspaint.paint([ts, ts, ts], labels, queries)        # list -> ensemble path

    assert ens.queries == single.queries
    assert isinstance(ens.ts, list) and len(ens.ts) == 3
    for q in queries:
        segs = ens.posteriors[q]
        assert hasattr(segs[0], "posterior_std")              # MergedSegment carries the band
        assert segs[0].left == 0.0 and segs[-1].right == ts.sequence_length
        assert all(np.allclose(s.posterior_std, 0.0, atol=1e-9) for s in segs)   # identical -> no spread
        for pos in np.linspace(0, ts.sequence_length, 7)[1:-1]:
            np.testing.assert_allclose(ens.posterior_at(q, pos), single.posterior_at(q, pos),
                                       atol=1e-8)


@pytest.mark.slow
def test_paint_ensemble_band_from_distinct_args():
    """Distinct ARGs over the same samples produce a non-trivial uncertainty band and a valid
    mean painting covering the genome."""
    from tspaint.ranked import ranked_tree_sequence
    ts, labels, queries, _ = _admixture(L=1e5)
    ens = tspaint.paint([ts, ranked_tree_sequence(ts)], labels, queries)
    assert any(s.posterior_std.sum() > 1e-6 for q in queries for s in ens.posteriors[q])
    for q in queries:
        segs = ens.posteriors[q]
        assert segs[0].left == 0.0 and segs[-1].right == ts.sequence_length
        assert all(np.isclose(s.posterior.sum(), 1.0) for s in segs)


@pytest.mark.slow
def test_painting_ensemble_methods():
    """introgression_map merges across the ensemble; rate_through_time is guarded."""
    ts, labels, queries, _ = _admixture(L=1e5)
    p = tspaint.paint([ts, ts], labels, queries)
    m = p.introgression_map(queries[0])
    assert m[0].left == 0.0 and m[-1].right == ts.sequence_length
    assert hasattr(m[0], "posterior_std")                     # merged leave-one-out map
    with pytest.raises(ValueError, match="ensemble"):
        p.rate_through_time()
