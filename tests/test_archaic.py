"""Plan B: reference-free archaic / ghost detection (depth-emission HMM).

Phase 1+2 (model + inference) and Phase 3 (identifiability): the learned detector recovers the
ghost burden with no archaic reference, and a matched no-ghost control stays ~0 (the §6
identifiability guard — anchor the archaic state beyond the modern panel's deepest coalescence).
"""
import numpy as np
import pytest

from tspaint.archaic import detect_ghost, GhostResult, detect_archaic, ArchaicResult
from tspaint.archaic import _anchor_modern, _forward_backward, _emission


def test_anchor_and_forward_backward_units():
    # anchor: mean/std/high-quantile of a toy reference depth track
    refs = {0: [(0.0, 10.0, 1.0), (10.0, 20.0, 3.0)], 1: [(0.0, 20.0, 2.0)]}
    mu, sd, q = _anchor_modern(refs, q=0.98)
    assert 1.0 < mu < 3.0 and sd > 0 and q >= mu        # quantile at/above the mean
    # forward-backward on a 2-state HMM returns a valid posterior summing to 1 per step
    B = _emission(np.array([0.0, 5.0, np.nan]), mu=np.array([0.0, 5.0]), sd=np.array([1.0, 1.0]))
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    gamma, xi, ll = _forward_backward(B, A, np.array([0.5, 0.5]))
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert gamma[0, 0] > gamma[0, 1] and gamma[1, 1] > gamma[1, 0]   # each obs pulls its own state
    assert np.isfinite(ll)


def _ghost_setup(gf, seed, L=1.2e6, n_admix=8, n_ref=8):
    from tspaint.sim import (simulate_admixture_with_ghost, local_ancestry_truth,
                             SOURCE_A, SOURCE_B, GHOST, ADMIXED)
    ts = simulate_admixture_with_ghost(n_admix=n_admix, n_ref=n_ref, sequence_length=L,
            recombination_rate=1e-8, random_seed=seed, ghost_fraction=gf,
            T_admix=100, T_split_AB=2000, T_split_ABC=20000, Ne=1000)
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    pid = {n: p for p, n in names.items()}
    node_pop = ts.tables.nodes.population
    of = lambda nm: [int(s) for s in ts.samples() if node_pop[s] == pid[nm]]
    queries = of(ADMIXED)
    labels = {s: 0 for s in of(SOURCE_A)}
    labels.update({s: 1 for s in of(SOURCE_B)})
    tracts, _ = local_ancestry_truth(ts)
    gid = pid[GHOST]
    true_ghost = {q: sum(r - l for (l, r, p) in tracts[q] if p == gid) / ts.sequence_length
                  for q in queries}
    return ts, labels, queries, true_ghost


def test_detect_ghost_aliases():
    # detect_archaic / ArchaicResult remain as deprecated aliases of detect_ghost / GhostResult.
    assert ArchaicResult is GhostResult
    with pytest.warns(DeprecationWarning):
        try:
            detect_archaic(None)            # emits the deprecation warning, then errors on the bad arg
        except Exception:
            pass


@pytest.mark.slow
def test_detect_ghost_recovers_burden_reference_free():
    # with a ghost source (and no ghost reference) the learned burden tracks the truth and
    # the model recovers a DEEP ghost component; the matched no-ghost control stays ~0.
    ts, labels, queries, true_ghost = _ghost_setup(0.25, seed=1)
    res = detect_ghost(ts, labels, queries, max_iter=40)
    assert isinstance(res, GhostResult)
    assert all(0.0 <= p <= 1.0 for q in queries for (_l, _r, p) in res.posteriors[q])
    assert res.posteriors[queries[0]][0][0] == 0.0                      # covers from 0
    assert res.mu[1] > res.mu[0] + 1.0                                  # archaic state is deep

    burden = float(np.mean([res.burden[q] for q in queries]))
    true_b = float(np.mean([true_ghost[q] for q in queries]))
    assert burden > 0.04                                               # ghost detected
    assert abs(burden - true_b) < 0.10                                 # burden recovers the truth (measured ~exact)
    # per-query estimate correlates with the per-query truth
    bs = np.array([res.burden[q] for q in queries])
    tg = np.array([true_ghost[q] for q in queries])
    assert np.corrcoef(bs, tg)[0, 1] > 0.6

    ts0, labels0, q0, _ = _ghost_setup(0.0, seed=1)
    res0 = detect_ghost(ts0, labels0, q0, max_iter=40)
    control = float(np.mean([res0.burden[q] for q in q0]))
    assert control < 0.05                                              # identifiability: no false ghost
    assert burden > 3 * control


@pytest.mark.slow
def test_archaic_detection_beats_fixed_threshold():
    # Plan B go/no-go: the learned HMM beats the Plan A fixed-threshold detect_ghost on per-locus
    # recall at equal (high) precision, recovers the burden, and keeps a low control false-positive.
    from tspaint.experiments import archaic_detection_experiment
    r = archaic_detection_experiment(ghost_fraction=0.25, n_admix=8, n_ref=8,
                                     sequence_length=1.5e6, seed=1, max_iter=40)
    a, g = r["archaic"], r["ghost"]
    assert a["recall"] > g["recall"] + 0.2          # learned detector recovers far more of the tracts
    assert a["precision"] > 0.8 and g["precision"] > 0.8
    assert abs(a["burden"] - r["true_burden"]) < 0.1   # near-exact burden recovery (vs the flag's under-detection)
    assert a["control_fp"] < 0.05                    # identifiability: low no-ghost false-positive
    assert a["mu_archaic"] > a["mu_modern"] + 1.0    # learns a deep archaic state


def _scale_node_times(ts, factor):
    """A monotonic distortion of branch lengths (all node/mutation times × factor)."""
    t = ts.dump_tables()
    t.nodes.time = t.nodes.time * factor
    if t.mutations.num_rows and not np.all(np.isnan(t.mutations.time)):
        t.mutations.time = np.where(np.isnan(t.mutations.time), t.mutations.time,
                                    t.mutations.time * factor)
    t.sort()
    return t.tree_sequence()


@pytest.mark.slow
def test_detect_ghost_rank_is_calibration_invariant():
    # depth="rank" uses a monotonic transform of depth, so rescaling all branch lengths leaves
    # the detection unchanged (the calibration-robustness that the rank option exists for).
    ts, labels, queries, _ = _ghost_setup(0.25, seed=1)
    r1 = detect_ghost(ts, labels, queries, depth="rank", max_iter=30)
    r2 = detect_ghost(_scale_node_times(ts, 7.0), labels, queries, depth="rank", max_iter=30)
    for q in queries:
        p1 = np.array([p for (_l, _r, p) in r1.posteriors[q]])
        p2 = np.array([p for (_l, _r, p) in r2.posteriors[q]])
        np.testing.assert_allclose(p1, p2, atol=1e-9)        # identical under time rescaling
    # and it ACTUALLY detects the ghost -- not a vacuously-near-zero posterior. In rank space the
    # depth is bounded in [0, 1], so the unbounded log-time floor (q_ref + sd_m) overshoots the
    # ceiling and parks the ghost emission above every observation: P(ghost) then collapses to ~0
    # and the burden-ratio check below still passes on two near-zero numbers. Require a confident
    # call and a substantial burden so that regression is caught.
    maxp = max(p for q in queries for (_l, _r, p) in r1.posteriors[q])
    assert maxp > 0.9                                     # some locus is confidently ghost (was ~0.02 when broken)
    assert np.mean([r1.burden[q] for q in queries]) > 0.05
    ts0, labels0, q0, _ = _ghost_setup(0.0, seed=1)
    c = detect_ghost(ts0, labels0, q0, depth="rank", max_iter=30)
    assert np.mean([r1.burden[q] for q in queries]) > 3 * np.mean([c.burden[q] for q in q0])


@pytest.mark.slow
def test_detect_ghost_ensemble():
    # an ensemble of tree sequences: one pooled fit, per-member decode, per-locus P(ghost) averaged.
    ts, labels, queries, _ = _ghost_setup(0.25, seed=1)
    single = detect_ghost(ts, labels, queries, max_iter=30)
    dup = detect_ghost([ts, ts], labels, queries, max_iter=30)      # identical members
    assert isinstance(dup, GhostResult)
    for q in queries:                                              # merge of identical ~= single
        # (not bit-exact: pooling doubles the log-lik, so the abs-tol early stop can fire one
        #  iteration apart — the params converge to the same fixed point)
        np.testing.assert_allclose(dup.burden[q], single.burden[q], atol=1e-3)
        assert dup.posteriors[q][0][0] == 0.0 and dup.posteriors[q][-1][1] == ts.sequence_length

    ts2, _, q2, _ = _ghost_setup(0.25, seed=2)                     # a distinct ARG (same sample ids)
    res = detect_ghost([ts, ts2], labels, queries, max_iter=20)    # exercises breakpoint refinement
    for q in queries:
        segs = res.posteriors[q]
        assert segs[0][0] == 0.0 and segs[-1][1] == ts.sequence_length
        assert all(0.0 <= p <= 1.0 for (_l, _r, p) in segs)
