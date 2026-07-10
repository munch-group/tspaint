"""Admixture-rate-through-time E-step tests (admix-dating, rung 1)."""
import numpy as np
import pytest

from tspaint.model import make_generator_2state
from tspaint.branch_stats import branch_expected_stats
from tspaint.dating import log_time_grid, split_branch, branch_cell_stats


def test_split_branch_covers_and_orders():
    edges = log_time_grid(1.0, 1000.0, 30)
    subs = split_branch(5.0, 600.0, edges)
    # durations sum to the branch length; ordered parent -> child (descending time)
    assert np.isclose(sum(d for _k, d in subs), 600.0 - 5.0)
    # cell indices non-increasing (we go from high time near the parent to low near the child)
    ks = [k for k, _d in subs]
    assert ks == sorted(ks, reverse=True)


def test_branch_cell_stats_sum_invariant():
    """Per-cell dwell/jumps summed over cells == whole-branch branch_expected_stats
    (additive property of the Van Loan integral), for a homogeneous generator."""
    Q = make_generator_2state(0.002, 0.005)
    edges = log_time_grid(1.0, 2000.0, 40)
    t_c, t_p = 7.0, 850.0                                  # spans many cells, in range
    rng = np.random.default_rng(0)
    xi = rng.random((2, 2))
    xi /= xi.sum()

    dwell, jumps = branch_cell_stats(lambda k: Q, t_c, t_p, xi, edges)
    tot_d = sum(dwell.values())
    tot_j = sum(jumps.values())
    ref_d, ref_j = branch_expected_stats(Q, t_p - t_c, xi)

    assert np.allclose(tot_d, ref_d, atol=1e-9)
    assert np.allclose(tot_j, ref_j, atol=1e-9)
    assert np.isclose(tot_d.sum(), t_p - t_c)             # dwell sums to branch length


def test_branch_cell_stats_localises_in_time():
    """A reward on a branch confined to a single cell lands in that cell only."""
    Q = make_generator_2state(0.003, 0.003)
    edges = log_time_grid(1.0, 2000.0, 40)
    # a short branch wholly inside one cell
    k0 = 20
    t_c, t_p = edges[k0] + 0.1, edges[k0 + 1] - 0.1
    xi = np.full((2, 2), 0.25)
    dwell, _ = branch_cell_stats(lambda k: Q, t_c, t_p, xi, edges)
    assert set(dwell) == {k0}
    assert np.isclose(sum(dwell.values()).sum(), t_p - t_c)


def test_poisson_spline_recovers_known_step_rate():
    """The M-step penalised spline recovers a (smooth) step rate from Poisson data with a
    coalescent-shaped exposure (admix-dating rung 3)."""
    from tspaint.dating.mstep import select_lambda_gcv
    rng = np.random.default_rng(1)
    centers = np.geomspace(20.0, 20000.0, 60)
    true = 1e-3 / (1.0 + np.exp(-(np.log(centers) - np.log(2000.0)) * 3.0))  # onset ~2000
    exposure = 1e6 * np.exp(-centers / 8000.0)                              # decays deep
    events = rng.poisson(true * exposure).astype(float)
    fit = select_lambda_gcv(centers, events, exposure)
    rate = fit["rate"]
    below = rate[centers < 1000].mean()
    above = rate[(centers > 4000) & (centers < 12000)].mean()
    assert above > 5 * below                                               # the rise is recovered
    m = exposure > 1
    assert np.corrcoef(rate[m], true[m])[0, 1] > 0.9                        # tracks the truth


@pytest.mark.slow
def test_fit_rate_through_time_recovers_split():
    """End-to-end: the inhomogeneous EM recovers the divergence onset on a clean A/B split.

    Cross-ancestry coalescence is impossible more recently than the population split, so the
    fitted cross-rate q_AB(t) must be ~0 for t < T_split and rise once t exceeds it. Also checks
    the EM log-likelihood is (weakly) monotone. Uses the auto log-time grid (edges=None)."""
    import msprime
    from tspaint.dating import fit_rate_through_time

    N, T_split = 1000, 2000.0
    d = msprime.Demography()
    d.add_population(name="A", initial_size=N)
    d.add_population(name="B", initial_size=N)
    d.add_population(name="ANC", initial_size=N)
    d.add_population_split(time=T_split, derived=["A", "B"], ancestral="ANC")
    ts = msprime.sim_ancestry(samples={"A": 6, "B": 6}, demography=d, sequence_length=3e5,
                              recombination_rate=1e-8, random_seed=1, ploidy=1)
    pop = ts.tables.nodes.population
    labels = {int(s): (0 if pop[s] == 0 else 1) for s in ts.samples()}

    rtt = fit_rate_through_time(ts, labels, n_iter=8)        # edges=None -> auto grid

    ll = rtt.loglik_history
    assert all(ll[i + 1] >= ll[i] - 1e-6 for i in range(len(ll) - 1))       # EM monotone
    c = rtt.centers
    recent = np.nanmean(rtt.q_AB[c < 0.5 * T_split])
    deep = np.nanmean(rtt.q_AB[(c > T_split) & (c < 4 * T_split)])
    assert deep > 3 * recent + 1e-9                                         # onset recovered


def test_rate_through_time_K_general_and_plot_pairs_colours():
    """RateThroughTime carries all K·(K-1) directional rates; .plot() draws each unordered pair in one
    colour (solid m→n / dashed n→m). Constructed directly (no EM) to test shape + plot fast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tspaint.dating import RateThroughTime

    ncell, K = 8, 3
    rng = np.random.default_rng(0)
    q = rng.random((ncell, K, K)) * 1e-3
    for k in range(K):
        q[:, k, k] = 0.0
    rtt = RateThroughTime(centers=np.geomspace(10, 1e4, ncell), q=q,
                          D=np.ones((ncell, K)), J=np.zeros((ncell, K, K)), loglik_history=[])
    assert rtt.K == 3
    assert rtt.state_names == ["A", "B", "C"]                   # integer states -> default letters
    assert rtt.pairs == [("A", "B"), ("A", "C"), ("B", "C")]    # name-keyed unordered pairs
    assert np.array_equal(rtt.rate(2, 0), q[:, 2, 0])           # index accessor
    assert np.array_equal(rtt.rate("C", "A"), q[:, 2, 0])       # name accessor
    for bad in ("q_AB", "q_BA"):                                # 2-state aliases raise on K>2
        with pytest.raises(ValueError):
            getattr(rtt, bad)

    axes = rtt.plot(facet=False)                                # explicit single-axis overlay
    lines = axes.get_lines()
    drawn = [ln for ln in lines if ln.get_linestyle() in ("-", "--")]
    assert len(drawn) == K * (K - 1)                            # all 6 directional rates drawn
    for i in range(0, len(drawn), 2):                           # (solid m→n, dashed n→m) per pair
        assert drawn[i].get_color() == drawn[i + 1].get_color()
        assert drawn[i].get_linestyle() == "-" and drawn[i + 1].get_linestyle() == "--"
    assert len({drawn[i].get_color() for i in range(0, len(drawn), 2)}) == 3   # 3 distinct pair colours

    faceted = rtt.plot()                                        # K>2 facets by default
    assert isinstance(faceted, list) and len(faceted) == 3     # one subplot per unordered pair
    plt.close("all")


@pytest.mark.slow
def test_fit_rate_through_time_K3_end_to_end():
    """The EM dater runs at K=3 (three source populations) and returns the full (n_cells, 3, 3) rate
    array — the fix for 'rate_through_time only works for K=2'. Cold start infers K from the labels."""
    import msprime
    from tspaint.dating import fit_rate_through_time, split_time, split_times
    N, T = 1000, 2000.0
    d = msprime.Demography()
    for nm in ("A", "B", "C", "AB", "ANC"):
        d.add_population(name=nm, initial_size=N)
    d.add_population_split(time=T, derived=["A", "B"], ancestral="AB")     # ((A,B),C)
    d.add_population_split(time=3 * T, derived=["AB", "C"], ancestral="ANC")
    ts = msprime.sim_ancestry(samples={"A": 5, "B": 5, "C": 5}, demography=d, sequence_length=3e5,
                              recombination_rate=1e-8, random_seed=1, ploidy=1)
    name = {i: ts.population(i).metadata.get("name", str(i)) for i in range(ts.num_populations)}
    pop = ts.tables.nodes.population
    labels = {int(s): name[pop[s]] for s in ts.samples()}        # NAME-valued labels ("A"/"B"/"C")

    rtt = fit_rate_through_time(ts, labels, n_iter=5)            # edges=None auto grid; K=3 from labels
    assert rtt.q.shape[1:] == (3, 3) and rtt.K == 3
    assert rtt.state_names == ["A", "B", "C"]                   # population names carried through
    assert rtt.rate("A", "B").shape == rtt.centers.shape        # rate accessible by population name
    with pytest.raises(ValueError):
        split_time(rtt)                                          # scalar split_time is 2-state only
    assert all(rtt.loglik_history[i + 1] >= rtt.loglik_history[i] - 1e-6
               for i in range(len(rtt.loglik_history) - 1))      # EM monotone
    st = split_times(rtt)                                        # per-pair dict, keyed by names
    assert set(st) == {("A", "B"), ("A", "C"), ("B", "C")}      # msprime pop names A/B/C flow through
    assert any(np.isfinite(v) for v in st.values())             # at least one split recovered


def test_assert_calibrated_rejects_uncalibrated_times():
    from tspaint.dating.grid import assert_calibrated
    assert_calibrated(np.array([0.0, 100.0, 5000.0]))              # calibrated -> no raise
    for bad in (np.array([0.0, 0.5, 0.95]), np.array([0.0, 1.0])): # tsinfer-like ~[0,1]
        with pytest.raises(ValueError, match="uncalibrated"):
            assert_calibrated(bad)


def test_dating_guards_uncalibrated_ts():
    """The auto-grid dater refuses a ts with tsinfer-like (~[0,1]) node times rather than dating in
    bogus units; an explicit edges= bypasses (the caller then vouches for the scale)."""
    import msprime
    from tspaint.dating import fit_rate_through_time, log_time_grid
    d = msprime.Demography()
    for nm in ("A", "B", "ANC"):
        d.add_population(name=nm, initial_size=1000)
    d.add_population_split(time=2000.0, derived=["A", "B"], ancestral="ANC")
    ts = msprime.sim_ancestry(samples={"A": 4, "B": 4}, demography=d, sequence_length=1e5,
                              recombination_rate=1e-8, random_seed=1, ploidy=1)
    labels = {int(s): (0 if ts.tables.nodes.population[s] == 0 else 1) for s in ts.samples()}
    t = ts.dump_tables()
    t.nodes.time = np.asarray(t.nodes.time) * (5.0 / float(np.asarray(t.nodes.time).max()))   # -> [0,5] < 10
    unc = t.tree_sequence()
    with pytest.raises(ValueError, match="uncalibrated|GENERATIONS"):
        fit_rate_through_time(unc, labels)                        # edges=None -> guarded
    rtt = fit_rate_through_time(unc, labels, edges=log_time_grid(0.1, 5.0, 8), n_iter=1)  # bypass
    assert rtt.centers.shape == (8,)


def test_top_level_dating_exports():
    """The dating path is surfaced in the public API (top-level fn, class, and namespace)."""
    import tspaint
    assert hasattr(tspaint, "fit_rate_through_time")
    assert hasattr(tspaint, "RateThroughTime")
    assert hasattr(tspaint, "dating")
    assert "fit_rate_through_time" in tspaint.__all__
    assert "RateThroughTime" in tspaint.__all__
    assert "dating" in tspaint.__all__
    assert hasattr(tspaint.Painting, "rate_through_time")


def _admix_labels(ts):
    import tspaint
    pop = ts.tables.nodes.population
    name = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in name.items() if n == tspaint.SOURCE_A)
    B = next(p for p, n in name.items() if n == tspaint.SOURCE_B)
    return {int(s): (0 if pop[s] == A else 1) for s in ts.samples() if pop[s] in (A, B)}


@pytest.mark.slow
def test_painting_rate_through_time_no_mutation():
    """Painting.rate_through_time() returns a separate RateThroughTime and leaves the painting's
    posteriors byte-for-byte unchanged (the dating path lives side by side with painting)."""
    import copy
    import tspaint
    ts = tspaint.simulate_admixture(tspaint.sim.admixture_demography(T_admix=100, Ne=1000, T_split=5000),
                                  n_query=4, n_reference=4, sequence_length=2e5, random_seed=1).ts
    labels = _admix_labels(ts)
    p = tspaint.paint(ts, labels)
    before = copy.deepcopy(p.posteriors)
    rtt = p.rate_through_time(n_iter=3, n_cells=20)

    assert type(rtt).__name__ == "RateThroughTime"
    assert rtt.q_AB.shape == rtt.q_BA.shape == rtt.centers.shape
    for q in before:                                                        # posteriors untouched
        assert len(before[q]) == len(p.posteriors[q])
        for s0, s1 in zip(before[q], p.posteriors[q]):
            assert s0.left == s1.left and s0.right == s1.right
            assert np.allclose(s0.posterior, s1.posterior)


@pytest.mark.slow
def test_fit_rate_through_time_warmstart_matches_cold():
    """Warm-starting from a precomputed FitResult reproduces the cold-start fit (the internal
    homogeneous fit is the only thing skipped), when seeded identically."""
    import msprime
    import tspaint

    N, T_split = 1000, 2000.0
    d = msprime.Demography()
    d.add_population(name="A", initial_size=N)
    d.add_population(name="B", initial_size=N)
    d.add_population(name="ANC", initial_size=N)
    d.add_population_split(time=T_split, derived=["A", "B"], ancestral="ANC")
    ts = msprime.sim_ancestry(samples={"A": 6, "B": 6}, demography=d, sequence_length=3e5,
                              recombination_rate=1e-8, random_seed=2, ploidy=1)
    pop = ts.tables.nodes.population
    labels = {int(s): (0 if pop[s] == 0 else 1) for s in ts.samples()}

    # Cold start does fit(..., Q0=None -> time-scaled default, max_iter=em_init=8) internally;
    # reproduce that same fit here (same default Q0, so warm and cold coincide).
    warm = tspaint.fit(ts, labels, max_iter=8, estimate_pi=False)
    rtt_cold = tspaint.fit_rate_through_time(ts, labels, n_iter=5)
    rtt_warm = tspaint.fit_rate_through_time(ts, labels, n_iter=5, fit_result=warm)

    assert np.allclose(rtt_warm.q_AB, rtt_cold.q_AB, rtol=1e-6, atol=1e-12)
    assert np.allclose(rtt_warm.q_BA, rtt_cold.q_BA, rtol=1e-6, atol=1e-12)


# --- performance: memoised + tree-parallel time-inhomogeneous E-step (fixes 1, 2, 3) ----------

def _dating_sim(L=3e5, seed=1, n=6):
    import tspaint
    return tspaint.simulate_admixture(
        tspaint.sim.admixture_demography(Ne=1000, T_admix=30, T_split=5000, f_A=0.5),
        n_query=n, n_reference=n, sequence_length=L, recombination_rate=1e-8, random_seed=seed)


def test_cell_kernel_cache_is_transparent():
    """A shared _CellKernels memo must not change branch_cell_stats — the interior-cell duration is
    the same float every time, so cached and uncached results are bit-identical."""
    from tspaint.dating.estep import _CellKernels
    Q = make_generator_2state(0.002, 0.005)
    Qoc = lambda k: Q                                          # noqa: E731
    edges = log_time_grid(1.0, 2000.0, 40)
    rng = np.random.default_rng(1)
    shared = _CellKernels(Qoc, 2)
    for _ in range(20):
        t_c, t_p = sorted(rng.uniform(1.0, 1800.0, size=2))
        if t_p - t_c < 1e-6:
            continue
        xi = rng.random((2, 2)); xi /= xi.sum()
        d0, j0 = branch_cell_stats(Qoc, t_c, t_p, xi, edges)               # private per-call cache
        d1, j1 = branch_cell_stats(Qoc, t_c, t_p, xi, edges, cache=shared)  # shared cross-branch memo
        assert d0.keys() == d1.keys()
        for k in d0:
            assert np.array_equal(d0[k], d1[k]) and np.array_equal(j0[k], j1[k])


def test_time_binned_tv_tree_range_partitions():
    """accumulate_time_binned_tv over a partition of [0, num_trees) sums (D, J, loglik additive) to
    the whole-genome result — the basis of the parallel dating E-step."""
    import tspaint
    from tspaint.dating.estep import accumulate_time_binned_tv
    from tspaint.dating.em import make_Q_of_cell
    from tspaint.em import build_emissions
    sim = _dating_sim()
    ts, labels = sim.ts, sim.labels
    pi = np.array([0.5, 0.5])
    edges = log_time_grid(1.0, float(ts.tables.nodes.time.max()) * 1.05, 30)
    q = np.zeros((len(edges) - 1, 2, 2)); q[:, 0, 1] = q[:, 1, 0] = 1e-4
    Qoc = make_Q_of_cell(q)
    em = build_emissions(ts, labels, {}, pi)

    D, J, ll = accumulate_time_binned_tv(ts, Qoc, pi, em, edges)
    cut = ts.num_trees // 2
    Da, Ja, la = accumulate_time_binned_tv(ts, Qoc, pi, em, edges, tree_range=(0, cut))
    Db, Jb, lb = accumulate_time_binned_tv(ts, Qoc, pi, em, edges, tree_range=(cut, ts.num_trees))
    assert np.allclose(Da + Db, D, rtol=1e-12, atol=0)
    assert np.allclose(Ja + Jb, J, rtol=1e-12, atol=0)
    assert np.isclose(la + lb, ll, rtol=1e-12)


@pytest.mark.slow
def test_fit_rate_through_time_parallel_matches_serial():
    """The tree-parallel dating E-step (n_jobs>1) gives the same profile as serial (allclose — sums
    over tree-ranges in chunk order, so ~ULP, not bit-identical)."""
    import tspaint
    from tspaint.dating import split_time
    sim = _dating_sim(L=8e5, seed=2)
    ts, labels = sim.ts, sim.labels
    a = tspaint.fit_rate_through_time(ts, labels, n_iter=6, em_init=2, n_jobs=1)
    b = tspaint.fit_rate_through_time(ts, labels, n_iter=6, em_init=2, n_jobs=3)
    assert np.allclose(a.q, b.q, rtol=1e-8, atol=1e-15)
    assert np.allclose(a.loglik_history, b.loglik_history, rtol=1e-8)
    assert np.isclose(split_time(a), split_time(b))


# --- API: robust (rising/falling) per-pair split times, K>2 accessors, faceting ---------------

def test_split_time_detects_rising_onset_and_falling_edge():
    """split_times auto-detects a rising onset (source–source) AND a falling edge (reference-proxy:
    high recent rate that drops after the split)."""
    from tspaint.dating import RateThroughTime, split_times, split_time
    nc, K = 40, 4
    c = np.geomspace(10, 10_000, nc)
    q = np.zeros((nc, K, K))
    q[:, 0, 1] = q[:, 1, 0] = 1e-3 * (c > 300)      # rising onset at ~300 (true split)
    q[:, 2, 3] = q[:, 3, 2] = 1e-6 * (c < 4000)     # HIGH then falls after ~4000 (proxy shape)
    rtt = RateThroughTime(c, q, np.ones((nc, K)), np.zeros((nc, K, K)), [-1.0])

    st = split_times(rtt)                                           # keyed by names (letters A..D)
    assert set(st) == {("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")}
    lo, hi = c[np.searchsorted(c, 300) - 1], c[np.searchsorted(c, 300) + 1]
    assert lo <= st[("A", "B")] <= hi                               # rising onset near 300
    assert 3000 < st[("C", "D")] < 5500                            # falling edge near 4000
    assert not np.isfinite(st[("A", "C")])                         # a zero-rate pair -> no split
    with pytest.raises(ValueError):
        split_time(rtt)                                            # K!=2 scalar -> raise


def test_split_time_K2_unchanged_and_scalar():
    """K=2 split_time still returns the rising-onset scalar (backward compatible)."""
    from tspaint.dating import RateThroughTime, split_time, split_times
    nc = 40
    c = np.geomspace(10, 10_000, nc)
    q = np.zeros((nc, 2, 2)); q[:, 0, 1] = q[:, 1, 0] = 1e-3 * (c > 500)
    rtt = RateThroughTime(c, q, np.ones((nc, 2)), np.zeros((nc, 2, 2)), [-1.0])
    st = split_time(rtt)
    assert np.isfinite(st) and 300 < st < 900                       # onset near the true 500
    assert np.isclose(split_times(rtt)[("A", "B")], st)           # scalar == the single (name) pair


def test_faceted_plot_axes_count():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tspaint.dating import RateThroughTime
    nc, K = 8, 4
    q = np.zeros((nc, K, K))
    for (m, n) in [(0, 1), (2, 3)]:
        q[:, m, n] = q[:, n, m] = 1e-3
    rtt = RateThroughTime(np.geomspace(10, 1e4, nc), q, np.ones((nc, K)),
                          np.zeros((nc, K, K)), [])
    axes = rtt.plot()                                              # K=4 -> facet by default
    assert isinstance(axes, list) and len(axes) == len(rtt.pairs) == 6
    single = rtt.plot(facet=False)                                # forced overlay -> one Axes
    assert not isinstance(single, list)
    plt.close("all")


def test_ensemble_pair_split_times_and_guards():
    """EnsembleRateThroughTime exposes per-pair split times for any K; the scalar split_time is
    2-state-only."""
    from tspaint.dating import RateThroughTime, EnsembleRateThroughTime
    nc, K = 30, 3
    c = np.geomspace(10, 1e4, nc)
    members = []
    rng = np.random.default_rng(0)
    names = ["X", "Y", "Z"]                                        # explicit population names
    for _ in range(5):
        q = np.zeros((nc, K, K))
        onset = 300 + rng.normal(0, 20)
        q[:, 0, 1] = q[:, 1, 0] = 1e-3 * (c > onset)
        q[:, 1, 2] = q[:, 2, 1] = 5e-4 * (c > 2000)
        members.append(RateThroughTime(c, q, np.ones((nc, K)), np.zeros((nc, K, K)), [-1.0],
                                       states=names))
    ens = EnsembleRateThroughTime.from_members(members)
    assert ens.K == 3 and ens.state_names == names                # names propagate from members
    assert ens.pairs == [("X", "Y"), ("X", "Z"), ("Y", "Z")]
    pst = ens.pair_split_times()
    assert set(pst) == {("X", "Y"), ("X", "Z"), ("Y", "Z")}
    assert 150 < pst[("X", "Y")] < 600                            # X<->Y onset recovered
    ci = ens.pair_split_time_ci(0.9)
    lo, hi = ci[("X", "Y")]
    assert lo <= pst[("X", "Y")] <= hi
    with pytest.raises(ValueError):
        ens.split_time()                                          # scalar 2-state-only
    with pytest.raises(ValueError):
        ens.q_AB


def test_labels_by_population_name_flow_through():
    """Dating accepts name-valued labels and reports rates / split times by population name."""
    import tspaint
    from tspaint.dating import fit_rate_through_time, split_times
    sim = _dating_sim(L=3e5, seed=4)
    ts, roles = sim.ts, sim.sample_sets
    # name-valued labels taken straight from the population sample sets
    a, b = sorted(n for n in roles if n in ("A", "B"))
    labels = {int(s): a for s in roles[a]}
    labels.update({int(s): b for s in roles[b]})

    rtt = fit_rate_through_time(ts, labels, n_iter=3, em_init=2, n_jobs=1)
    assert rtt.state_names == [a, b]                              # sorted population names
    assert rtt.pairs == [(a, b)]
    assert np.array_equal(rtt.rate(a, b), rtt.rate(0, 1))        # name == index accessor
    st = split_times(rtt)
    assert set(st) == {(a, b)}                                    # split time keyed by name pair


def test_simulation_stores_demography():
    """simulate_admixture retains the demography it simulated under."""
    import tspaint
    demo = tspaint.sim.admixture_demography(Ne=1000, T_admix=30, T_split=5000, f_A=0.5)
    sim = tspaint.simulate_admixture(demo, n_query=2, n_reference=2, sequence_length=1e5,
                                     random_seed=1)
    assert sim.demography is demo                                 # the exact object, not a copy
    # a wrapper builder also stores its (internally built) demography
    sim2 = tspaint.sim.simulate_admixture_with_ghost(n_admix=2, n_ref=3, sequence_length=1e5,
                                                     random_seed=1)
    import msprime
    assert isinstance(sim2.demography, msprime.Demography)
