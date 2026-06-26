"""Plan A: the foreignness primitive (Phase 1) and the workflows built on it."""
import numpy as np
import pytest
import tskit

from tspaint.model import make_generator_2state, tip_emission, query_emission
from tspaint.introgression import foreignness_track, ForeignnessSegment, reference_qc
from tspaint.output import INFORMATIVE, MISSING_INFO


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


def test_loo_dissent_for_hard_clamped_reference():
    # 0 (hard A) and 1 (hard B) under a common root: the leave-one-out posterior of 0 reflects
    # the sibling (B), so it dissents from 0's own A label.
    ts = _build_ts({0: 2, 1: 2, 2: -1}, [0.0, 0.0, 1.0], {0, 1})
    Q = make_generator_2state(0.3, 0.3)
    pi = np.array([0.5, 0.5])
    labels = {0: 0, 1: 1}
    em = {0: tip_emission(0, 1.0, pi), 1: tip_emission(1, 1.0, pi)}

    tr = foreignness_track(ts, Q, pi, em, labels)
    L = ts.sequence_length
    for segs in tr.values():
        assert segs[0].left == 0.0 and segs[-1].right == L           # full coverage
        for seg in segs:
            assert np.isclose(seg.loo.sum(), 1.0) and np.all(seg.loo >= 0)
            assert 0.5 - 1e-9 <= seg.fit <= 1.0 + 1e-9               # fit = max(loo) in [1/K, 1]
    assert tr[0][0].loo[0] < 0.5                                     # 0's outside message dissents (-> B)


def test_deep_outlier_scores_high_depth_low_fit():
    # 0 (A) and 1 (B) coalesce shallowly (t=1); query 2 joins only at the deep root (t=10).
    # 2 is a deep outlier: max depth, and its outside message washes to ~uninformative.
    ts = _build_ts({0: 3, 1: 3, 3: 4, 2: 4, 4: -1}, [0.0, 0.0, 0.0, 1.0, 10.0], {0, 1, 2})
    Q = make_generator_2state(1e-2, 1e-2)
    pi = np.array([0.5, 0.5])
    labels = {0: 0, 1: 1}
    em = {0: tip_emission(0, 1.0, pi), 1: tip_emission(1, 1.0, pi), 2: query_emission(pi)}

    tr = foreignness_track(ts, Q, pi, em, labels, focal=[0, 2])
    s0, s2 = tr[0][0], tr[2][0]
    # nearest-ref depth: 2's nearest ref is across the deep root; 0's nearest ref (1) is shallow
    assert s2.depth > s0.depth
    assert np.isclose(s2.depth, 1.0)                                 # 2 is the deepest -> rank 1.0
    assert s2.fit < 0.6                                              # deep tip -> washed, fits nothing
    assert 0.0 <= s0.depth <= 1.0 and 0.0 <= s2.depth <= 1.0


def test_depth_time_mode_returns_raw_coalescent_time():
    ts = _build_ts({0: 3, 1: 3, 3: 4, 2: 4, 4: -1}, [0.0, 0.0, 0.0, 1.0, 10.0], {0, 1, 2})
    Q = make_generator_2state(1e-2, 1e-2)
    pi = np.array([0.5, 0.5])
    labels = {0: 0, 1: 1}
    em = {0: tip_emission(0, 1.0, pi), 1: tip_emission(1, 1.0, pi), 2: query_emission(pi)}

    tr = foreignness_track(ts, Q, pi, em, labels, focal=[0, 2], depth="time")
    assert np.isclose(tr[0][0].depth, 1.0)                           # mrca(0,1) at t=1
    assert np.isclose(tr[2][0].depth, 10.0)                          # mrca(2,*) at the deep root


def test_isolated_sample_tagged_missing_and_depth_nan():
    # sample 2 isolated over the whole sequence -> MISSING_INFO, undefined depth
    ts = _build_ts({0: 3, 1: 3, 3: -1}, [0.0, 0.0, 0.0, 1.0], {0, 1, 2})
    Q = make_generator_2state(0.3, 0.3)
    pi = np.array([0.6, 0.4])
    labels = {0: 0, 1: 1}
    em = {0: tip_emission(0, 1.0, pi), 1: tip_emission(1, 1.0, pi), 2: query_emission(pi)}

    tr = foreignness_track(ts, Q, pi, em, labels, focal=[0, 1, 2])
    seg2 = tr[2][0]
    assert seg2.status == MISSING_INFO
    assert np.isnan(seg2.depth)
    np.testing.assert_allclose(seg2.loo, pi)                         # outside message falls back to prior


@pytest.mark.slow
def test_reference_qc_flags_impure_minority():
    # Plan A Workflow 1: with the clean references in the majority, QC ranks the impure ones
    # least-credible and its LOO maps recover their foreign tracts (CLAUDE.md §6, §9).
    from tspaint.sim import (simulate_admixture_impure_refs, local_ancestry_truth,
                             SOURCE_A, SOURCE_B, REF_A_IMPURE, REF_B_IMPURE)
    from tspaint.validate import map_truth
    from tspaint.experiments import _foreign_recall

    ts = simulate_admixture_impure_refs(n_admix=2, n_pure=8, n_impure=3, sequence_length=2.5e6,
            recombination_rate=1e-8, random_seed=1, ref_impurity=0.3, Ne=1000, T_admix=150, T_split=5000)
    node_pop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    pid = {n: p for p, n in names.items()}
    of = lambda nm: [int(s) for s in ts.samples() if node_pop[s] == pid[nm]]
    pure = of(SOURCE_A) + of(SOURCE_B)
    impure = of(REF_A_IMPURE) + of(REF_B_IMPURE)
    labels = {s: 0 for s in of(SOURCE_A) + of(REF_A_IMPURE)}
    labels.update({s: 1 for s in of(SOURCE_B) + of(REF_B_IMPURE)})

    qc = reference_qc(ts, labels, max_iter=8)
    # structure
    assert qc.anchors and len(qc.anchors) < len(qc.labels)
    assert all(0.0 <= v <= 1.0 for v in qc.credibility.values())
    assert all(m[0].left == 0.0 and m[-1].right == ts.sequence_length for m in qc.maps.values())
    assert len(qc.summary()) == len(qc.labels)
    # discrimination: impure references are less credible than the clean majority
    # (measured gap 0.18-0.26 over seeds)
    pc = float(np.mean([qc.credibility[r] for r in pure]))
    ic = float(np.mean([qc.credibility[r] for r in impure]))
    assert ic < pc - 0.1
    # the LOO introgression map recovers the impure refs' foreign tracts (measured ~0.9)
    truth, _ = local_ancestry_truth(ts)
    rt = map_truth({r: truth[r] for r in impure}, {pid[SOURCE_A]: 0, pid[SOURCE_B]: 1})
    rec = float(np.nanmean([_foreign_recall(qc.maps[r], rt[r], labels[r]) for r in impure]))
    assert rec > 0.3
