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


# --- refs: also paint the reference haplotypes, framing the queries --------------------------

@pytest.mark.slow
def test_paint_refs_true_frames_queries():
    ts, labels, queries, _ = _admixture(L=1e5)
    ref1 = [s for s in labels if labels[s] == 0]
    ref2 = [s for s in labels if labels[s] == 1]
    p = tspaint.paint(ts, labels, queries, refs=True)
    assert p.queries[:len(ref1)] == ref1                     # ref1 (state 0) -> first rows
    assert p.queries[-len(ref2):] == ref2                    # ref2 (state 1) -> bottom rows
    assert set(p.queries[len(ref1):-len(ref2)]) == set(queries)   # queries in the middle
    assert set(p.posteriors) == set(p.queries)               # references are painted too
    assert p.posterior_at(ref1[0], ts.sequence_length / 2)[0] > 0.99   # clamped ref -> its label


@pytest.mark.slow
def test_paint_refs_list_selects_and_orders():
    ts, labels, queries, _ = _admixture(L=1e5)
    ref1 = [s for s in labels if labels[s] == 0]
    ref2 = [s for s in labels if labels[s] == 1]
    p = tspaint.paint(ts, labels, queries, refs=[ref2[0], ref1[0]])   # input order ignored
    assert p.queries[0] == ref1[0]                           # state-grouped: ref1 first
    assert p.queries[-1] == ref2[0]                          # ref2 last
    assert set(p.queries) == {ref1[0], ref2[0]} | set(queries)


def test_paint_refs_non_reference_raises():
    # raised while resolving args, before the EM fit -> fast (no @slow needed)
    ts, labels, queries, _ = _admixture(L=5e4)
    with pytest.raises(ValueError, match="not reference individuals"):
        tspaint.paint(ts, labels, queries, refs=[queries[0]])         # a query is not a reference


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
    """introgression_map merges across the ensemble; member posteriors are retained."""
    ts, labels, queries, _ = _admixture(L=1e5)
    p = tspaint.paint([ts, ts], labels, queries)
    m = p.introgression_map(queries[0])
    assert m[0].left == 0.0 and m[-1].right == ts.sequence_length
    assert hasattr(m[0], "posterior_std")                     # merged leave-one-out map
    assert isinstance(p._member_posteriors, list) and len(p._member_posteriors) == 2


def test_split_time_is_the_cross_rate_onset():
    """split_time finds the onset (half-max rise) of the combined cross-ancestry rate."""
    from tspaint.dating import RateThroughTime, split_time
    centers = np.geomspace(10.0, 1e4, 60)
    rise = np.where(centers >= 2000.0, 1e-3, 1e-6)            # ~0 below the split, high above
    rtt = RateThroughTime(centers=centers, q_AB=rise, q_BA=np.zeros_like(rise),
                          D=np.ones((60, 2)), J=np.zeros((60, 2, 2)), loglik_history=[])
    assert 1500.0 <= split_time(rtt) <= 2700.0                # the onset, not a peak
    flat = RateThroughTime(centers=centers, q_AB=np.zeros_like(centers), q_BA=np.zeros_like(centers),
                           D=np.ones((60, 2)), J=np.zeros((60, 2, 2)), loglik_history=[])
    assert np.isnan(split_time(flat))                         # no rise -> nan


@pytest.mark.slow
def test_painting_ensemble_member_posteriors_and_dating():
    """An ensemble painting keeps per-member posteriors; rate_through_time -> split-time CI."""
    from tspaint.dating import EnsembleRateThroughTime, RateThroughTime
    ts, labels, queries, _ = _admixture(L=1.5e5)
    kw = dict(n_admix=6, n_ref=6, sequence_length=1.5e5, recombination_rate=1e-8,
              Ne=1000, T_admix=30, T_split=5000, f_A=0.5)
    members = [ts] + [tspaint.simulate_admixture(random_seed=s, **kw) for s in (2, 3)]

    single = tspaint.paint(ts, labels, queries)
    assert single._member_posteriors is None                  # single ts: no member tables
    assert isinstance(single.rate_through_time(n_cells=15, n_iter=4), RateThroughTime)

    ens = tspaint.paint(members, labels, queries)
    assert isinstance(ens._member_posteriors, list) and len(ens._member_posteriors) == 3
    for tab in ens._member_posteriors:                        # each member covers [0, L)
        assert tab[queries[0]][0].left == 0.0 and tab[queries[0]][-1].right == ts.sequence_length

    er = ens.rate_through_time(n_cells=20, n_iter=6)
    assert isinstance(er, EnsembleRateThroughTime)
    assert len(er.members) == 3 and er.split_times.shape == (3,)
    assert all(isinstance(m, RateThroughTime) for m in er.members)
    assert er.q_AB.shape == er.centers.shape                  # shared grid -> averageable mean
    assert np.isfinite(er.split_times).any()                  # at least one member resolves a split
    lo, hi = er.split_time_ci()
    st = er.split_time()
    assert lo <= st <= hi                                     # the CI brackets the point estimate
    assert er.centers.min() <= st <= er.centers.max()


def test_painting_n_jobs_field():
    base = dict(posteriors={}, Q=np.eye(2), pi=np.array([0.5, 0.5]), w={}, loglik_history=[],
                queries=[])
    assert tspaint.Painting(**base, ts=None).n_jobs == 1              # default serial
    assert tspaint.Painting(**base, ts=None, n_jobs=4).n_jobs == 4    # stored from paint(n_jobs=)


@pytest.mark.slow
def test_ensemble_rate_through_time_parallel_matches_serial():
    """Dating the ensemble members in parallel gives the same result as serial, and the painting's
    n_jobs is the default (so a parallel-painted ensemble dates in parallel)."""
    from tspaint.dating import EnsembleRateThroughTime
    ts, labels, queries, _ = _admixture(L=1.5e5)
    kw = dict(n_admix=6, n_ref=6, sequence_length=1.5e5, recombination_rate=1e-8,
              Ne=1000, T_admix=30, T_split=5000, f_A=0.5)
    members = [ts] + [tspaint.simulate_admixture(random_seed=s, **kw) for s in (2, 3)]
    p = tspaint.paint(members, labels, queries)
    assert p.n_jobs == 1

    er1 = p.rate_through_time(n_cells=18, n_iter=5, n_jobs=1)         # serial
    er3 = p.rate_through_time(n_cells=18, n_iter=5, n_jobs=3)         # parallel across members
    assert isinstance(er3, EnsembleRateThroughTime) and len(er3.members) == 3
    np.testing.assert_allclose(er3.split_times, er1.split_times, rtol=0, atol=1e-6, equal_nan=True)
    for m1, m3 in zip(er1.members, er3.members):
        np.testing.assert_allclose(m3.q_AB, m1.q_AB, rtol=1e-9, atol=0)
        np.testing.assert_allclose(m3.q_BA, m1.q_BA, rtol=1e-9, atol=0)

    p.n_jobs = 3                                                      # inherited when n_jobs omitted
    er_inh = p.rate_through_time(n_cells=18, n_iter=5)
    np.testing.assert_allclose(er_inh.split_times, er1.split_times, rtol=0, atol=1e-6, equal_nan=True)


# --- Painting.length / Painting.plot ---------------------------------------------------------

def test_painting_length():
    ts, _, _, _ = _admixture(L=1e5)
    base = dict(posteriors={}, Q=np.eye(2), pi=np.array([0.5, 0.5]), w={}, loglik_history=[],
                queries=[])
    assert tspaint.Painting(**base, ts=ts).length == ts.sequence_length
    assert tspaint.Painting(**base, ts=[ts, ts]).length == ts.sequence_length   # ensemble -> member 0
    assert tspaint.Painting(**base, ts=None).length is None


@pytest.mark.slow
def test_painting_plot_runs():
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    from tspaint.ranked import ranked_tree_sequence
    ts, labels, queries, truth = _admixture(L=1e5)
    p = tspaint.paint(ts, labels, queries)
    p.plot(truth=truth, title="t"); plt.close("all")          # single, with truth
    p.plot(); plt.close("all")                                # single, no truth
    pe = tspaint.paint([ts, ranked_tree_sequence(ts)], labels, queries)
    pe.plot(truth=truth); plt.close("all")                    # ensemble mean
