"""High-level paint() / Painting API tests (CLAUDE.md §2.4)."""
import io
from contextlib import redirect_stderr

import numpy as np
import pytest

import tspaint
from tspaint.sim import SOURCE_A, SOURCE_B, ADMIXED, admixture_demography


def _admixture(L=5e5):
    ts = tspaint.simulate_admixture(admixture_demography(Ne=1000, T_admix=30, T_split=5000, f_A=0.5),
                                  n_query=6, n_reference=6, sequence_length=L, recombination_rate=1e-8,
                                  random_seed=1).ts
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


def _posteriors_view(painting):
    """Canonical, comparable view of a Painting's per-query posterior segments."""
    return {
        int(q): [(round(float(s.left), 9), round(float(s.right), 9),
                  getattr(s, "status", None),
                  tuple(np.round(np.asarray(s.posterior, float), 10).tolist()))
                 for s in segs]
        for q, segs in painting.posteriors.items()
    }


def _paint_capturing(*args, **kwargs):
    """paint() while capturing the tqdm bar's stderr output."""
    buf = io.StringIO()
    with redirect_stderr(buf):
        p = tspaint.paint(*args, **kwargs)
    return p, buf.getvalue()


@pytest.mark.slow
def test_progress_is_value_preserving_and_emits_a_bar():
    """`progress=True` must be a pure UI addition: identical results on every path
    (serial, parallel, ensemble), a bar on stderr when on, and silence when off."""
    ts, labels, _, _ = _admixture()

    # serial: identical to the no-progress baseline; per-tree bar fires; off is silent.
    base = tspaint.paint(ts, labels)
    prog, err = _paint_capturing(ts, labels, progress=True)
    assert _posteriors_view(prog) == _posteriors_view(base)
    assert "painting" in err
    _, err_off = _paint_capturing(ts, labels, progress=False)
    assert err_off.strip() == ""

    # parallel: exactly equal to serial, with a per-chunk bar.
    par, err_par = _paint_capturing(ts, labels, n_jobs=2, progress=True)
    assert _posteriors_view(par) == _posteriors_view(base)
    assert "painting" in err_par

    # ensemble: per-member bar; identical to the no-progress ensemble.
    ens_base = tspaint.paint([ts, ts], labels)
    ens, err_ens = _paint_capturing([ts, ts], labels, progress=True)
    assert _posteriors_view(ens) == _posteriors_view(ens_base)
    assert "painting" in err_ens


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
def test_paint_robust_to_large_time_scale():
    """A large (e.g. tsdate-calibrated / deep) node-age scale must not wash out the CTMC. The default
    Q0 scales to the time axis (Q0*t ~ 1), so inflating every node age leaves the painting essentially
    unchanged instead of collapsing toward pi (regression: a fixed Q0 froze the EM and gave P(A)~0.5)."""
    ts, labels, queries, truth = _admixture()
    base = tspaint.paint(ts, labels, queries=queries)
    t = ts.dump_tables()
    t.nodes.time = np.asarray(t.nodes.time) * 50.0            # 50x deeper (as an over-scaled tsdate)
    big = tspaint.paint(t.tree_sequence(), labels, queries=queries)

    ba_base = tspaint.metrics.balanced_accuracy(base.posteriors, truth, samples=queries)
    ba_big = tspaint.metrics.balanced_accuracy(big.posteriors, truth, samples=queries)
    assert ba_big > 0.9 and abs(ba_big - ba_base) < 0.1       # ~scale-invariant, not collapsed
    sd = np.std([s.posterior[0] for q in queries for s in big.posteriors[q]])
    assert sd > 0.15                                          # posteriors keep their spread (not ~pi)


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

    def rtt_from(q01, q10):                                   # build the (n_cells, 2, 2) rate array
        q = np.zeros((len(q01), 2, 2))
        q[:, 0, 1], q[:, 1, 0] = q01, q10
        return RateThroughTime(centers=centers, q=q, D=np.ones((60, 2)), J=np.zeros((60, 2, 2)),
                               loglik_history=[])

    centers = np.geomspace(10.0, 1e4, 60)
    rise = np.where(centers >= 2000.0, 1e-3, 1e-6)            # ~0 below the split, high above
    assert 1500.0 <= split_time(rtt_from(rise, np.zeros_like(rise))) <= 2700.0   # the onset, not a peak
    assert np.isnan(split_time(rtt_from(np.zeros_like(centers), np.zeros_like(centers))))   # no rise -> nan


@pytest.mark.slow
def test_painting_ensemble_member_posteriors_and_dating():
    """An ensemble painting keeps per-member posteriors; rate_through_time -> split-time CI."""
    from tspaint.dating import EnsembleRateThroughTime, RateThroughTime
    ts, labels, queries, _ = _admixture(L=1.5e5)
    demo = admixture_demography(Ne=1000, T_admix=30, T_split=5000, f_A=0.5)
    kw = dict(n_query=6, n_reference=6, sequence_length=1.5e5, recombination_rate=1e-8)
    members = [ts] + [tspaint.simulate_admixture(demo, random_seed=s, **kw).ts for s in (2, 3)]

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
    assert tspaint.Painting(**base, ts=None).n_jobs is None           # default -> all CPUs (resolved at use)
    assert tspaint.Painting(**base, ts=None, n_jobs=4).n_jobs == 4    # stored from paint(n_jobs=)


@pytest.mark.slow
def test_ensemble_rate_through_time_parallel_matches_serial():
    """Dating the ensemble members in parallel gives the same result as serial, and the painting's
    n_jobs is the default (so a parallel-painted ensemble dates in parallel)."""
    from tspaint.dating import EnsembleRateThroughTime
    ts, labels, queries, _ = _admixture(L=1.5e5)
    demo = admixture_demography(Ne=1000, T_admix=30, T_split=5000, f_A=0.5)
    kw = dict(n_query=6, n_reference=6, sequence_length=1.5e5, recombination_rate=1e-8)
    members = [ts] + [tspaint.simulate_admixture(demo, random_seed=s, **kw).ts for s in (2, 3)]
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


@pytest.mark.slow
def test_painting_default_title():
    """No title kwarg -> the title defaults to the painting summary (CLAUDE.md §9 read-out)."""
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    ts, labels, queries, truth = _admixture(L=1e5)
    p = tspaint.paint(ts, labels, queries)

    # with truth: precision / recall / fragmentation-ratio + ancestry proportions
    _fig, axes = p.plot(truth=truth, return_plot=True)
    title = axes[0].get_title()
    plt.close("all")
    for token in ("precision", "recall", "fragmentation", "ancestry", "A ", "B "):
        assert token in title, f"{token!r} missing from default title {title!r}"

    # no truth: fragmentation reported as raw switch density, no precision/recall
    _fig, axes = p.plot(return_plot=True)
    title_nt = axes[0].get_title()
    plt.close("all")
    assert "sw/Mb" in title_nt and "ancestry" in title_nt
    assert "precision" not in title_nt

    # an explicit title (including "") always wins over the default
    _fig, axes = p.plot(truth=truth, title="mine", return_plot=True)
    assert axes[0].get_title() == "mine"; plt.close("all")
    _fig, axes = p.plot(title="", return_plot=True)
    assert axes[0].get_title() == ""; plt.close("all")

    # summary() exposes the same numbers as a dict
    s = p.summary(truth=truth)
    assert set(s) >= {"proportion", "switch_per_mb", "precision", "recall", "switch_ratio"}
    assert len(s["proportion"]) == 2


def _grid_posterior(painting, queries, grid):
    """Per-position P(state 0) on a genomic grid — comparable across different segmentations."""
    out = {}
    for q in queries:
        pa = np.full(len(grid), np.nan)
        for s in painting.posteriors[q]:
            pa[(grid >= s.left) & (grid < s.right)] = np.asarray(s.posterior, float)[0]
        out[q] = pa
    return out


def test_paint_window_size_streams_and_reassembles(tmp_path):
    """paint(ts, window_size=W, out_dir=D) fits (Q, π, w) once, then paints each window with those
    fixed params and STREAMS it to D (one Painting per window + manifest), returning a lightweight
    WindowedPainting. Reassembling from disk reproduces the whole-genome painting exactly (no
    smoothing), one window loaded at a time."""
    ts, labels, queries, _ = _admixture(L=6e5)
    out = tmp_path / "wp"
    wp = tspaint.paint(ts, labels, queries, window_size=2e5, out_dir=str(out))   # 3 windows
    assert isinstance(wp, tspaint.WindowedPainting)
    assert wp.n_windows == 3
    assert (out / "manifest.json").exists()
    assert sorted(p.name for p in out.glob("window_*.npz")) == \
        ["window_00000.npz", "window_00001.npz", "window_00002.npz"]

    # lazy iteration yields each window's [lo, hi) slice, in order, covering [0, L)
    seen = [(lo, hi) for lo, hi, _p in wp.windows()]
    assert seen == [(0.0, 2e5), (2e5, 4e5), (4e5, 6e5)]

    # reassembled-from-disk == whole-genome paint, per position, with the same fit
    whole = tspaint.paint(ts, labels, queries)
    full = wp.painting()
    assert isinstance(full, tspaint.Painting) and np.allclose(wp.Q, whole.Q)
    grid = np.arange(0, ts.sequence_length, 500)
    gw, gf = _grid_posterior(whole, queries, grid), _grid_posterior(full, queries, grid)
    for q in queries:
        assert not np.isnan(gf[q]).any()                              # full [0, L) coverage, no gaps
        assert np.allclose(gw[q], gf[q], atol=1e-9)                   # identical per position


def test_windowed_painting_resume_and_load(tmp_path):
    """A windowed run is resumable (existing window files are not rewritten) and the directory is
    self-describing (WindowedPainting.load rebuilds the handle from manifest.json)."""
    import os
    ts, labels, queries, _ = _admixture(L=4e5)
    out = str(tmp_path / "wp")
    wp = tspaint.paint(ts, labels, queries, window_size=2e5, out_dir=out)
    mtimes = {p: os.path.getmtime(p) for p in tmp_path.glob("wp/window_*.npz")}
    tspaint.paint(ts, labels, queries, window_size=2e5, out_dir=out)               # rerun == resume
    assert all(os.path.getmtime(p) == mtimes[p] for p in mtimes)                   # untouched

    reopened = tspaint.WindowedPainting.load(out)
    assert reopened.n_windows == wp.n_windows
    assert np.allclose(reopened.Q, wp.Q) and np.allclose(reopened.pi, wp.pi)
    assert list(reopened.queries) == list(wp.queries)


def test_paint_window_size_guards(tmp_path):
    """window_size <-> out_dir must be paired, and streaming is single-tree-sequence only."""
    ts, labels, queries, _ = _admixture(L=2e5)
    with pytest.raises(ValueError, match="requires out_dir"):
        tspaint.paint(ts, labels, queries, window_size=1e5)                        # no out_dir
    with pytest.raises(ValueError, match="only used with window_size"):
        tspaint.paint(ts, labels, queries, out_dir=str(tmp_path / "x"))            # no window_size
    with pytest.raises(ValueError, match="single"):
        tspaint.paint([ts, ts], labels, queries, window_size=1e5,                  # ensemble
                      out_dir=str(tmp_path / "y"))


# --- K > 2 sources: the inference must generalise past two ancestries ------------------------

def _three_source_sim(L=1e6, seed=1):
    """A 3-source (K=3) admixture: 3 labelled reference panels + 3-way census truth.

    C is a deep outgroup to A/B (nested ((A,B),C)), so the three ancestries are genealogically
    distinguishable; a recent pulse keeps the query↔reference signal strong.
    """
    import msprime
    from tspaint.sim import pop_role, simulate_admixture
    Ne = 2000
    d = msprime.Demography()
    d.add_population(name=SOURCE_A, initial_size=Ne, extra_metadata=pop_role(state=0, source=True, reference=True))
    d.add_population(name=SOURCE_B, initial_size=Ne, extra_metadata=pop_role(state=1, source=True, reference=True))
    d.add_population(name="C",      initial_size=Ne, extra_metadata=pop_role(state=2, source=True, reference=True))
    d.add_population(name=ADMIXED,  initial_size=Ne, extra_metadata=pop_role(query=True))
    d.add_population(name="AB", initial_size=Ne)
    d.add_population(name="ANC", initial_size=Ne)
    d.add_admixture(time=100, derived=ADMIXED, ancestral=[SOURCE_A, SOURCE_B, "C"],
                    proportions=[1 / 3, 1 / 3, 1 / 3])
    d.add_census(time=101.0)
    d.add_population_split(time=2000, derived=[SOURCE_A, SOURCE_B], ancestral="AB")
    d.add_population_split(time=6000, derived=["AB", "C"], ancestral="ANC")
    return simulate_admixture(d, n_query=8, n_reference=6, sequence_length=L,
                              recombination_rate=1e-8, random_seed=seed)


@pytest.mark.slow
def test_paint_three_sources():
    # CLAUDE.md: the model is generator-agnostic, "K-way by generator swap". This exercises the
    # whole inference at K=3 and guards the regression where fit's default Q0 was hard-wired 2-state
    # (so paint(K=3) with no explicit Q0 crashed).
    from tspaint.validate import balanced_accuracy
    sim = _three_source_sim()
    assert sorted(set(sim.labels.values())) == [0, 1, 2]              # three labelled ancestries
    assert sorted({st for q in sim.queries for (_, _, st) in sim.truth_states[q]}) == [0, 1, 2]

    p = tspaint.paint(sim.ts, sim.labels, sim.queries, K=3, n_jobs=1)  # no Q0 -> K-way default
    assert p.Q.shape == (3, 3) and p.pi.shape == (3,)
    post = p.posteriors[sim.queries[0]][0].posterior
    assert len(post) == 3 and np.isclose(post.sum(), 1.0)             # a proper 3-simplex posterior
    ba = balanced_accuracy(p.posteriors, sim.truth_states, K=3)
    assert ba > 0.8, f"K=3 balanced accuracy too low ({ba:.3f}); the math should recover 3 ancestries"


# --- Painting.plot(samples=...) : sample selection folded in from plot_posterior ---------------

def _small_painting():
    sim = tspaint.simulate_admixture(
        admixture_demography(Ne=1000, T_admix=30, T_split=5000, f_A=0.5),
        n_query=3, n_reference=3, sequence_length=1e5, recombination_rate=1e-8, random_seed=2)
    return tspaint.paint(sim.ts, sim.labels, queries=sim.queries)


def test_plot_samples_by_node_index():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = _small_painting()
    q = p.queries
    _fig, axes = p.plot(samples=q[0], return_plot=True)             # a single int -> one row
    assert len(axes) == 1 and axes[0].get_ylabel() == f"hapl. {q[0]}"
    plt.close("all")
    _fig, axes = p.plot(samples=[q[0], q[1], q[2]], return_plot=True)   # list of int -> rows in order
    assert len(axes) == 3
    plt.close("all")
    _fig, axes = p.plot_posterior(q[0], return_plot=True)           # the alias still works
    assert len(axes) == 1 and axes[0].get_ylabel() == f"hapl. {q[0]}"
    plt.close("all")


def test_plot_samples_absent_empty_and_no_ts_string():
    p = _small_painting()
    with pytest.raises(ValueError):
        p.plot(samples=10_000_000)           # a node that was not painted
    with pytest.raises(ValueError):
        p.plot(samples=[])                   # nothing selected
    p.ts = None                              # a reloaded painting cannot resolve string ids
    with pytest.raises(ValueError):
        p.plot(samples="someone")


def test_plot_posterior_by_individual_id_expands_haplotypes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from tspaint.ids import attach_sample_ids

    sim = tspaint.simulate_admixture(
        admixture_demography(Ne=1000, T_admix=30, T_split=5000, f_A=0.5),
        n_query=3, n_reference=3, sequence_length=1e5, recombination_rate=1e-8, random_seed=3)
    ts = sim.ts
    node_ind = ts.tables.nodes.individual
    cnt, names = defaultdict(int), []
    for s in ts.samples():                                          # diploid ids "ind<i>_<k>" per haplotype
        i = int(node_ind[int(s)])
        names.append(f"ind{i}_{cnt[i]}")
        cnt[i] += 1
    sts = attach_sample_ids(ts, names, ploidy=2)                    # node ids unchanged; labels stay valid
    p = tspaint.paint(sts, sim.labels, queries=sim.queries)

    iid = f"ind{int(node_ind[sim.queries[0]])}"                     # id of a painted admixed individual
    _fig, axes = p.plot(samples=iid, return_plot=True)
    assert len(axes) == 2                                           # a diploid id -> both haplotype rows
    assert [ax.get_ylabel() for ax in axes] == [f"{iid} (hap 0)", f"{iid} (hap 1)"]
    plt.close("all")


def test_summary_returns_paintingsummary():
    from tspaint.validate import PaintingSummary
    assert isinstance(_small_painting().summary(), PaintingSummary)


def test_plot_ghost_overlay_hatches_tracts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tspaint.output import Segment, INFORMATIVE
    from tspaint.archaic import GhostResult

    p = _small_painting()
    q0 = p.queries[0]
    L = p.length
    gr = GhostResult(                                                   # ghost tract on the right half
        posteriors={q0: [Segment(0.0, L / 2, np.array([0.95, 0.05]), INFORMATIVE),
                         Segment(L / 2, L, np.array([0.1, 0.9]), INFORMATIVE)]},
        burden={q0: 0.5}, mu=np.array([0.0, 2.0]), sd=np.array([1.0, 1.0]),
        A=np.eye(2), pi0=np.array([0.5, 0.5]), loglik_history=[], _seqlen=L)

    _fig, axes = p.plot(samples=[q0], return_plot=True)                 # no overlay -> no hatching
    assert not any(patch.get_hatch() for ax in axes for patch in ax.patches)
    plt.close("all")
    _fig, axes = p.plot(samples=[q0], ghost=gr, return_plot=True)       # ghost tract hatched over bands
    assert any(patch.get_hatch() for ax in axes for patch in ax.patches)
    plt.close("all")
    with pytest.raises(TypeError):                                      # not a GhostResult
        p.plot(ghost=object())


def test_masked_and_ghost_use_distinct_hatches():
    # a masked reference span and a ghost tract on the same plot must be visually distinguishable
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tspaint.output import Segment, INFORMATIVE
    from tspaint.archaic import GhostResult
    from tspaint.api import _MASK_HATCH, _GHOST_HATCH

    from tspaint.api import Painting
    posteriors = {0: [Segment(0.0, 100.0, np.array([0.9, 0.1]), INFORMATIVE)]}   # one ref row
    p = Painting(posteriors=posteriors, Q=np.eye(2), pi=np.array([0.5, 0.5]), w={},
                 loglik_history=[], queries=[0], labels={0: 0}, _seqlen=100.0,
                 mask={0: [(0.0, 40.0)]})                                        # masked span 0-40
    gr = GhostResult(posteriors={0: [Segment(0.0, 60.0, np.array([0.9, 0.1]), INFORMATIVE),
                                     Segment(60.0, 100.0, np.array([0.1, 0.9]), INFORMATIVE)]},
                     burden={0: 0.4}, mu=np.array([0.0, 2.0]), sd=np.array([1.0, 1.0]),
                     A=np.eye(2), pi0=np.array([0.5, 0.5]), loglik_history=[], _seqlen=100.0)
    _fig, axes = p.plot(ghost=gr, return_plot=True)
    hatches = {pt.get_hatch() for a in axes for pt in a.patches if pt.get_hatch()}
    assert _MASK_HATCH in hatches and _GHOST_HATCH in hatches and _MASK_HATCH != _GHOST_HATCH
    plt.close("all")


def test_plot_curves_and_ghost_dashed_curve():
    # curves=True draws the K posteriors as line curves; ghost= adds a black dashed P(ghost) curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tspaint.output import Segment, INFORMATIVE
    from tspaint.archaic import GhostResult
    from tspaint.api import Painting

    p = Painting(posteriors={0: [Segment(0., 50., np.array([0.8, 0.2]), INFORMATIVE),
                                 Segment(50., 100., np.array([0.3, 0.7]), INFORMATIVE)]},
                 Q=np.eye(2), pi=np.array([0.5, 0.5]), w={}, loglik_history=[], queries=[0],
                 labels=None, _seqlen=100.0)
    _fig, axes = p.plot(samples=[0], curves=True, return_plot=True)
    assert len(axes) == 1
    lines = axes[0].get_lines()
    assert len(lines) == 2                                       # one curve per ancestry state (K=2)
    assert axes[0].get_ylim()[0] < 0.0 and axes[0].get_ylim()[1] > 1.0   # probability y-axis [0, 1]
    plt.close("all")

    gr = GhostResult(posteriors={0: [Segment(0., 60., np.array([0.9, 0.1]), INFORMATIVE),
                                     Segment(60., 100., np.array([0.2, 0.8]), INFORMATIVE)]},
                     burden={0: 0.4}, mu=np.array([0., 2.]), sd=np.array([1., 1.]),
                     A=np.eye(2), pi0=np.array([0.5, 0.5]), loglik_history=[], _seqlen=100.0)
    _fig, axes = p.plot(samples=[0], curves=True, ghost=gr, return_plot=True)
    lines = axes[0].get_lines()
    black = [ln for ln in lines if matplotlib.colors.to_hex(ln.get_color()) == "#000000"]
    assert len(lines) == 3 and len(black) == 1                  # K state curves + the ghost P(ghost) curve
    plt.close("all")


def test_plot_curves_truth_and_segment_bands():
    # curves=True + truth/segments draws thin state-coloured bands (~10% of curve height) below the curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tspaint.output import Segment, INFORMATIVE
    from tspaint.api import Painting
    from tspaint.track import _CURVE_BAND_H

    p = Painting(posteriors={0: [Segment(0., 50., np.array([0.9, 0.1]), INFORMATIVE),
                                 Segment(50., 100., np.array([0.2, 0.8]), INFORMATIVE)]},
                 Q=np.eye(2), pi=np.array([0.5, 0.5]), w={}, loglik_history=[], queries=[0],
                 labels=None, _seqlen=100.0)
    truth = {0: [(0., 50., 0), (50., 100., 1)]}
    _fig, axes = p.plot(samples=[0], curves=True, truth=truth, segments=True, return_plot=True)
    ax = axes[0]
    bars = [pt for pt in ax.patches if pt.get_height() > 0]     # barh band patches (hard + truth)
    assert bars and all(np.isclose(pt.get_height(), _CURVE_BAND_H) for pt in bars)   # each ~10% tall
    assert ax.get_ylim()[0] < -0.1 and ax.get_ylim()[1] > 1.0   # bands stacked below the [0,1] curves
    plt.close("all")

    _fig, axes = p.plot(samples=[0], curves=True, return_plot=True)     # no truth/segments -> no bands
    assert not [pt for pt in axes[0].patches if pt.get_height() > 0]
    assert axes[0].get_ylim()[0] >= -0.05
    plt.close("all")
