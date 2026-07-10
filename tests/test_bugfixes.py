"""Regression guards for six fixed defects.

Each test fails on the pre-fix code. They are grouped by the bug they pin, not by module.

The parallel tests drive the worker code path through an explicit ``ThreadPoolExecutor`` rather
than a process pool: ``executor=`` bypasses the serial shortcut, so ``_accumulate_range`` &
friends run exactly as they do under ``make_pool`` — without paying spawn cost, which keeps these
guards in the default (non-``slow``) suite.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import tskit

import tspaint
from tspaint import parallel
from tspaint.em import build_emissions
from tspaint.model import make_generator_2state
from tspaint.output import Segment, INFORMATIVE, MISSING_INFO
from tspaint.ensemble import MergedSegment
from tspaint.sim import SOURCE_A, SOURCE_B, admixture_demography


def _sim(L=1e5, seed=1, n=3):
    sim = tspaint.simulate_admixture(
        admixture_demography(Ne=1000, T_admix=30, T_split=5000, f_A=0.5),
        n_query=n, n_reference=n, sequence_length=L, recombination_rate=1e-8, random_seed=seed)
    return sim


def _flat_emissions(ts):
    """Emissions that differ from anything `build_emissions(labels, w, pi)` would produce."""
    return {int(s): np.array([0.5, 0.5]) for s in ts.samples()}


# --- bug 1: io_argweaver's "no samples" path raised NameError instead of RuntimeError ---------

def test_argweaver_no_samples_raises_runtimeerror(monkeypatch, tmp_path):
    import tspaint.io_argweaver as aw

    monkeypatch.setattr(aw, "_run_argweaver", lambda *a, **k: None)
    monkeypatch.setattr(aw, "_argweaver_indices", lambda out: [])      # -> _select(...) == []
    monkeypatch.setattr(aw, "_source_sample_ids", lambda s: (s, ["n0", "n1"], 1))
    monkeypatch.setattr(aw, "write_sites",
                        lambda src, path, **k: open(path, "w").write("NAMES\tn0\tn1\n"))

    with pytest.raises(RuntimeError, match="no post-burn-in ARGweaver samples"):
        aw.argweaver("src", _N=1e4, _m=1e-8, _r=1e-8, workdir=str(tmp_path))


# --- bug 2: `emissions=` was honoured serially but silently dropped in the workers ------------

def test_accumulate_parallel_honours_emissions():
    sim = _sim()
    ts, labels = sim.ts, sim.labels
    Q, pi = make_generator_2state(1e-4, 1e-4), np.array([0.5, 0.5])
    bogus = _flat_emissions(ts)

    ser = parallel.accumulate_parallel(ts, Q, pi, w={}, labels=labels, emissions=bogus, n_jobs=1)
    ex = ThreadPoolExecutor(2)
    try:
        par = parallel.accumulate_parallel(ts, Q, pi, w={}, labels=labels, emissions=bogus,
                                           n_jobs=2, executor=ex)
        rebuilt = parallel.accumulate_parallel(ts, Q, pi, w={}, labels=labels, n_jobs=2, executor=ex)
    finally:
        ex.shutdown()

    assert np.allclose(ser.loglik, par.loglik)          # workers used the emissions we passed
    assert not np.isclose(par.loglik, rebuilt.loglik)   # ...and those really differ from a rebuild


@pytest.mark.parametrize("fn", ["posterior_table_parallel", "loo_posterior_table_parallel"])
def test_painters_parallel_honour_emissions(fn):
    sim = _sim()
    ts, labels = sim.ts, sim.labels
    Q, pi = make_generator_2state(1e-4, 1e-4), np.array([0.5, 0.5])
    bogus = _flat_emissions(ts)
    focal = sorted(labels)
    call = getattr(parallel, fn)

    ser = call(ts, Q, pi, w={}, labels=labels, focal=focal, emissions=bogus, n_jobs=1)
    ex = ThreadPoolExecutor(2)
    try:
        par = call(ts, Q, pi, w={}, labels=labels, focal=focal, emissions=bogus,
                   n_jobs=2, executor=ex)
    finally:
        ex.shutdown()

    for s in focal:
        assert len(ser[s]) == len(par[s])
        for a, b in zip(ser[s], par[s]):
            assert np.allclose(a.posterior, b.posterior)


def test_parallel_mask_still_overrides_emissions():
    """`mask` must win over a pre-built `emissions`, on both branches (masking is label-level)."""
    sim = _sim()
    ts, labels = sim.ts, sim.labels
    Q, pi = make_generator_2state(1e-4, 1e-4), np.array([0.5, 0.5])
    ref0 = min(labels)
    mask = {ref0: [(0.0, ts.sequence_length / 2)]}
    bogus = _flat_emissions(ts)

    ser = parallel.posterior_table_parallel(ts, Q, pi, w={}, labels=labels, focal=[ref0],
                                            emissions=bogus, mask=mask, n_jobs=1)
    ex = ThreadPoolExecutor(2)
    try:
        par = parallel.posterior_table_parallel(ts, Q, pi, w={}, labels=labels, focal=[ref0],
                                                emissions=bogus, mask=mask, n_jobs=2, executor=ex)
    finally:
        ex.shutdown()
    assert np.allclose(ser[ref0][0].posterior, par[ref0][0].posterior)


# --- bug 3: the dating E-step crashed on MaskedEmissions / silently dropped a painting's mask --

def _grid_and_Q(ts, K=2):
    from tspaint.dating import log_time_grid, make_Q_of_cell
    edges = log_time_grid(1.0, max(float(ts.tables.nodes.time.max()), 2.0), 5)
    q = np.zeros((len(edges) - 1, K, K))
    q[:, 0, 1] = q[:, 1, 0] = 1e-4
    return edges, make_Q_of_cell(q)


def test_paint_qt_accepts_and_applies_masked_emissions():
    from tspaint.dating import paint_qt

    sim = _sim(L=5e4, seed=3, n=2)
    ts, labels = sim.ts, sim.labels
    pi = np.array([0.5, 0.5])
    ref0 = min(labels)
    edges, Qc = _grid_and_Q(ts)

    plain = build_emissions(ts, labels, {}, pi)
    masked = build_emissions(ts, labels, {}, pi, mask={ref0: [(0.0, ts.sequence_length)]})

    a = paint_qt(ts, plain, Qc, pi, edges, focal=[ref0])[ref0][0].posterior
    b = paint_qt(ts, masked, Qc, pi, edges, focal=[ref0])[ref0][0].posterior   # used to AttributeError
    assert not np.allclose(a, b), "masking a reference must change its own down-pass posterior"


def test_accumulate_time_binned_tv_accepts_masked_emissions():
    from tspaint.dating.estep import accumulate_time_binned_tv, accumulate_time_binned

    sim = _sim(L=5e4, seed=3, n=2)
    ts, labels = sim.ts, sim.labels
    pi = np.array([0.5, 0.5])
    ref0 = min(labels)
    edges, Qc = _grid_and_Q(ts)
    masked = build_emissions(ts, labels, {}, pi, mask={ref0: [(0.0, ts.sequence_length)]})

    D, J, ll = accumulate_time_binned_tv(ts, Qc, pi, masked, edges)
    assert np.isfinite(ll) and D.shape == (len(edges) - 1, 2)
    D2, J2 = accumulate_time_binned(ts, make_generator_2state(1e-4, 1e-4), pi, masked, edges)
    assert np.isfinite(D2).all() and np.isfinite(J2).all()


def test_fit_rate_through_time_accepts_mask():
    sim = _sim(L=5e4, seed=3, n=2)
    ref0 = min(sim.labels)
    kw = dict(n_cells=5, n_iter=2, em_init=2)
    plain = tspaint.fit_rate_through_time(sim.ts, sim.labels, **kw)
    masked = tspaint.fit_rate_through_time(sim.ts, sim.labels,
                                           mask={ref0: [(0.0, sim.ts.sequence_length)]}, **kw)
    assert plain.q.shape == masked.q.shape
    assert not np.allclose(plain.q, masked.q), "mask must reach the dating E-step"


def test_painting_rate_through_time_forwards_its_mask(monkeypatch):
    """A masked painting must date under the same emissions it painted with."""
    import tspaint.dating as dating

    sim = _sim(L=5e4, seed=3, n=2)
    ref0 = min(sim.labels)
    mask = {ref0: [(0.0, sim.ts.sequence_length)]}
    p = tspaint.Painting(posteriors={}, Q=make_generator_2state(1e-4, 1e-4),
                         pi=np.array([0.5, 0.5]), w={}, loglik_history=[-1.0],
                         queries=[], ts=sim.ts, labels=sim.labels, mask=mask)

    seen = {}

    def spy(ts, labels, edges=None, **kwargs):
        seen.update(kwargs)
        return "rtt"

    monkeypatch.setattr(dating, "fit_rate_through_time", spy)
    assert p.rate_through_time() == "rtt"
    assert seen.get("mask") == mask

    seen.clear()                                   # an explicit mask= still wins
    p.rate_through_time(mask=None)
    assert seen.get("mask") is None


# --- bug 4: save_ghost dropped the ensemble posterior_std band --------------------------------

def _ghost_result(segments):
    from tspaint.archaic import GhostResult
    return GhostResult(posteriors={0: segments}, burden={0: 0.25},
                       mu=np.array([1.0, 2.0]), sd=np.array([0.3, 0.4]),
                       A=np.array([[0.9, 0.1], [0.2, 0.8]]), pi0=np.array([0.7, 0.3]),
                       loglik_history=[-5.0, -4.0])


def test_save_ghost_preserves_posterior_std_band(tmp_path):
    from tspaint.serialize import save_ghost, load_ghost

    segs = [MergedSegment(0.0, 10.0, np.array([0.3, 0.7]), INFORMATIVE, np.array([0.05, 0.05]), 3),
            MergedSegment(10.0, 20.0, np.array([0.8, 0.2]), INFORMATIVE, np.array([0.11, 0.11]), 3)]
    path = tmp_path / "g.npz"
    save_ghost(path, _ghost_result(segs))
    back = load_ghost(path)["posteriors"][0]

    assert [type(s).__name__ for s in back] == ["MergedSegment", "MergedSegment"]
    for a, b in zip(segs, back):
        assert np.allclose(a.posterior, b.posterior)
        assert np.allclose(a.posterior_std, b.posterior_std)   # the band used to be discarded
        assert a.n_informative == b.n_informative


def test_save_ghost_preserves_segment_status(tmp_path):
    from tspaint.serialize import save_ghost, load_ghost

    segs = [Segment(0.0, 10.0, np.array([0.3, 0.7]), INFORMATIVE),
            Segment(10.0, 20.0, np.array([0.5, 0.5]), MISSING_INFO)]
    path = tmp_path / "g.npz"
    save_ghost(path, _ghost_result(segs))
    back = load_ghost(path)["posteriors"][0]
    assert [s.status for s in back] == [INFORMATIVE, MISSING_INFO]


def test_load_ghost_reads_legacy_v1_file(tmp_path):
    """v1 files (scalar p_ghost column, no band) must still load."""
    from tspaint.serialize import _savez, load_ghost

    path = tmp_path / "g1.npz"
    _savez(path, _format="tspaint-ghost", _version=1,
           sample=np.array([0, 0], np.int64), left=np.array([0.0, 10.0]),
           right=np.array([10.0, 20.0]), p_ghost=np.array([0.7, 0.2]),
           burden_nodes=np.array([0], np.int64), burden_vals=np.array([0.25]),
           mu=np.array([1.0, 2.0]), sd=np.array([0.3, 0.4]),
           A=np.array([[0.9, 0.1], [0.2, 0.8]]), pi0=np.array([0.7, 0.3]),
           loglik_history=np.array([-5.0, -4.0]))
    back = load_ghost(path)["posteriors"][0]
    assert len(back) == 2
    assert np.allclose(back[0].posterior, [0.3, 0.7])
    assert all(s.status == INFORMATIVE for s in back)


# --- bug 5: save_reference_qc dropped the individual / haplotype id names ---------------------

def _qc(sample_ids=None, individual_ids=None):
    from tspaint.introgression import ReferenceQC
    maps = {0: [Segment(0.0, 100.0, np.array([0.9, 0.1]), INFORMATIVE)],
            1: [Segment(0.0, 100.0, np.array([0.2, 0.8]), INFORMATIVE)]}
    return ReferenceQC(labels={0: 0, 1: 1}, credibility={0: 0.95, 1: 0.60},
                       loo_agreement={0: 0.9, 1: 0.6}, learned_w={1: 0.6}, anchors={0},
                       maps=maps, Q=make_generator_2state(1e-3, 1e-3), pi=np.array([0.5, 0.5]),
                       _length=100.0, sample_ids=sample_ids, individual_ids=individual_ids)


def test_save_reference_qc_roundtrips_id_names(tmp_path):
    from tspaint.serialize import save_reference_qc, load_reference_qc

    qc = _qc(sample_ids={0: "NA12878_0", 1: "NA12891_0"},
             individual_ids={0: "NA12878", 1: "NA12891"})
    path = tmp_path / "qc.npz"
    save_reference_qc(path, qc)
    rows = load_reference_qc(path)["summary"]

    by_ref = {r["ref"]: r for r in rows}
    assert by_ref[0]["individual"] == "NA12878" and by_ref[0]["haplotype"] == "NA12878_0"
    assert by_ref[1]["individual"] == "NA12891" and by_ref[1]["haplotype"] == "NA12891_0"


def test_save_reference_qc_without_id_names(tmp_path):
    from tspaint.serialize import save_reference_qc, load_reference_qc

    path = tmp_path / "qc.npz"
    save_reference_qc(path, _qc())
    rows = load_reference_qc(path)["summary"]
    assert all("individual" not in r and "haplotype" not in r for r in rows)
    assert {r["ref"] for r in rows} == {0, 1}


# --- bug 6: the deprecated convert_relate bypassed $TSPAINT_RELATE_CONVERT --------------------

def test_convert_relate_defers_binary_resolution(monkeypatch):
    import tspaint.io_relate as R

    seen = {}

    def spy(anc, mut, *, out_prefix=None, compress=True, convert_bin=None):
        seen.update(convert_bin=convert_bin, out_prefix=out_prefix, compress=compress)
        return "ts"

    monkeypatch.setattr(R, "relate_convert", spy)
    with pytest.warns(DeprecationWarning):
        assert R.convert_relate("a.anc", "m.mut", "out") == "ts"

    # `None` => relate_convert resolves $TSPAINT_RELATE_CONVERT / the installed binary.
    # Pre-fix this forwarded the literal "Convert", pinning it to PATH.
    assert seen["convert_bin"] is None
    assert seen["out_prefix"] == "out" and seen["compress"] is True

    with pytest.warns(DeprecationWarning):                 # an explicit binary still passes through
        R.convert_relate("a.anc", "m.mut", "out", convert_bin="/opt/Convert")
    assert seen["convert_bin"] == "/opt/Convert"
