"""Plan B: reference-free archaic / ghost detection (depth-emission HMM).

Phase 1+2 (model + inference) and Phase 3 (identifiability): the learned detector recovers the
ghost burden with no archaic reference, and a matched no-ghost control stays ~0 (the §6
identifiability guard — anchor the archaic state beyond the modern panel's deepest coalescence).
"""
import numpy as np
import pytest

from tspaint.archaic import detect_archaic, ArchaicResult
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


@pytest.mark.slow
def test_detect_archaic_recovers_burden_reference_free():
    # with a ghost source (and no archaic reference) the learned burden tracks the truth and
    # the model recovers a DEEP archaic component; the matched no-ghost control stays ~0.
    ts, labels, queries, true_ghost = _ghost_setup(0.25, seed=1)
    res = detect_archaic(ts, labels, queries, max_iter=40)
    assert isinstance(res, ArchaicResult)
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
    res0 = detect_archaic(ts0, labels0, q0, max_iter=40)
    control = float(np.mean([res0.burden[q] for q in q0]))
    assert control < 0.05                                              # identifiability: no false archaic
    assert burden > 3 * control
