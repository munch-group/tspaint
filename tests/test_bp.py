"""Horizontal BP/EP smoother tests (CLAUDE.md §7).

The toy tests are fast (no sim) and pin the smoother's intended behaviour: smooth isolated
low-confidence flips, preserve sustained switches, reinforce weak-but-consistent runs. The
integration test (slow) checks that bp_paint reduces fragmentation on a real sim.
"""
import numpy as np
import pytest

from tspaint.bp import bp_smooth, bp_smooth_track, bp_paint
from tspaint.output import Segment, INFORMATIVE, hard_segments

PI = np.array([0.5, 0.5])


def test_bp_smooth_suppresses_isolated_low_confidence_flip():
    weak = np.array([[0.9, 0.1], [0.45, 0.55], [0.95, 0.05]])
    g = bp_smooth(weak, PI, epsilon=1e-3)
    assert g.argmax(1).tolist() == [0, 0, 0]
    assert g[1, 0] > 0.9                      # middle pulled confidently to the flanking state


def test_bp_smooth_preserves_sustained_switch():
    sw = np.array([[0.95, 0.05]] * 5 + [[0.02, 0.98]] * 5)
    assert bp_smooth(sw, PI, epsilon=1e-3).argmax(1).tolist() == [0] * 5 + [1] * 5


def test_bp_smooth_reinforces_consistent_weak_run():
    run = np.array([[0.6, 0.4]] * 6)
    g = bp_smooth(run, PI, epsilon=1e-3)
    assert np.all(g[:, 0] > 0.6)              # weak-but-consistent evidence accumulates


def test_bp_smooth_epsilon_monotone():
    weak = np.array([[0.9, 0.1], [0.45, 0.55], [0.95, 0.05]])
    strong_coupling = bp_smooth(weak, PI, epsilon=1e-6)[1, 0]
    weak_coupling = bp_smooth(weak, PI, epsilon=0.5)[1, 0]
    assert strong_coupling > weak_coupling    # smaller epsilon -> more smoothing


def test_bp_smooth_track_preserves_intervals_and_status():
    track = [Segment(0, 100, np.array([0.9, 0.1]), INFORMATIVE),
             Segment(100, 200, np.array([0.4, 0.6]), INFORMATIVE)]
    out = bp_smooth_track(track, PI, epsilon=1e-3)
    assert [(s.left, s.right, s.status) for s in out] == [(0, 100, INFORMATIVE),
                                                          (100, 200, INFORMATIVE)]
    assert all(np.isclose(s.posterior.sum(), 1.0) for s in out)


@pytest.mark.slow
def test_bp_paint_reduces_fragmentation():
    import tspaint
    from tspaint.sim import local_ancestry_truth, SOURCE_A, SOURCE_B, ADMIXED
    from tspaint.compare import tspaint_paint
    from tspaint.validate import map_truth, switch_density

    ts = tspaint.simulate_admixture(n_admix=6, n_ref=6, sequence_length=2e6, recombination_rate=1e-8,
                                  random_seed=1, Ne=1000, T_admix=200, T_split=5000, f_A=0.5)
    npop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in names.items() if n == SOURCE_A)
    B = next(p for p, n in names.items() if n == SOURCE_B)
    admix = next(p for p, n in names.items() if n == ADMIXED)
    sop = {A: 0, B: 1}
    labels = {int(s): sop[npop[s]] for s in ts.samples() if npop[s] in (A, B)}
    queries = [int(s) for s in ts.samples() if npop[s] == admix]
    true_segs = map_truth({q: local_ancestry_truth(ts)[0][q] for q in queries}, sop)
    L = ts.sequence_length

    argmax = tspaint_paint(ts, labels, queries)
    bp = bp_paint(ts, labels, queries, epsilon=1e-2)
    true_d = np.mean([switch_density(true_segs[q], L) for q in queries])
    argmax_d = np.mean([switch_density(hard_segments(argmax[q]), L) for q in queries])
    bp_d = np.mean([switch_density(hard_segments(bp[q]), L) for q in queries])
    # BP must reduce the over-fragmentation of naive argmax toward the truth
    assert bp_d < argmax_d
    assert abs(bp_d - true_d) <= abs(argmax_d - true_d)


@pytest.mark.slow
def test_bp_vs_deadband_experiment_runs():
    from tspaint.bp import bp_vs_deadband_experiment
    r = bp_vs_deadband_experiment(T_admix=500, infer=False, seeds=(1, 2), n_admix=6, n_ref=6,
                                  sequence_length=1e6)
    assert r["n_seeds"] == 2
    for key in ("deadband_f1", "bp_f1"):
        mean, std = r[key]
        assert 0.0 <= mean <= 1.0 and std >= 0.0
