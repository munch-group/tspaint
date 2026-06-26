"""Plan A: the foreignness primitive (Phase 1) and the workflows built on it."""
import numpy as np
import pytest
import tskit

from tspaint.model import make_generator_2state, tip_emission, query_emission
from tspaint.introgression import (foreignness_track, ForeignnessSegment, reference_qc,
                                 foreign_tracts, detect_ghost)
from tspaint.output import INFORMATIVE, MISSING_INFO


def _span_overlap(l, r, intervals):
    return sum(max(0.0, min(r, b) - max(l, a)) for (a, b) in intervals)


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


@pytest.mark.slow
def test_detect_ghost_finds_unsampled_source():
    # Plan A Workflow 3: an unsampled ghost source is detectable (low fit + deep), with a low
    # false-positive burden on the matched no-ghost control (the honesty gate; CLAUDE.md §9).
    from tspaint.sim import (simulate_admixture_with_ghost, local_ancestry_truth,
                             SOURCE_A, SOURCE_B, GHOST, ADMIXED)
    common = dict(n_admix=12, n_ref=8, sequence_length=2e6, recombination_rate=1e-8,
                  T_admix=100, T_split_AB=2000, T_split_ABC=20000, Ne=1000)

    def run(gf, seed):
        ts = simulate_admixture_with_ghost(ghost_fraction=gf, random_seed=seed, **common)
        names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
        pid = {n: p for p, n in names.items()}
        node_pop = ts.tables.nodes.population
        of = lambda nm: [int(s) for s in ts.samples() if node_pop[s] == pid[nm]]
        queries = of(ADMIXED)
        labels = {s: 0 for s in of(SOURCE_A)}
        labels.update({s: 1 for s in of(SOURCE_B)})
        tracts, _ = local_ancestry_truth(ts)
        gid = pid[GHOST]
        true_ghost = {q: [(l, r) for (l, r, p) in tracts[q] if p == gid] for q in queries}
        return queries, true_ghost, detect_ghost(ts, labels, queries, max_iter=8)

    queries, true_ghost, gh = run(0.25, 1)
    q0, _, gh0 = run(0.0, 1)

    # detection: the genome-wide ghost burden is far above the no-ghost control (measured ~10x)
    burden = float(np.mean([gh.burden[q] for q in queries]))
    burden0 = float(np.mean([gh0.burden[q] for q in q0]))
    assert burden > 3 * burden0
    assert burden0 < 0.05                              # false-positive floor (measured ~0.01)

    # localisation: the flagged tracts overlap the true ghost tracts (measured recall 0.58 / prec 1.0)
    det = sum(r - l for q in queries for (l, r) in gh.tracts(q))
    det_ghost = sum(_span_overlap(l, r, true_ghost[q]) for q in queries for (l, r) in gh.tracts(q))
    total_ghost = sum(r - l for q in queries for (l, r) in true_ghost[q])
    assert det_ghost / total_ghost > 0.4               # recall
    assert det_ghost / det > 0.7                        # precision


@pytest.mark.slow
def test_foreign_tracts_flags_reference_introgression():
    # Plan A Workflow 2: anonymous foreign-tract inference recovers an impure reference's own
    # foreign tracts (label mode), without attributing a source.
    from tspaint.sim import (simulate_admixture_impure_refs, local_ancestry_truth,
                             SOURCE_A, SOURCE_B, REF_A_IMPURE, REF_B_IMPURE)
    from tspaint.validate import map_truth

    ts = simulate_admixture_impure_refs(n_admix=2, n_pure=6, n_impure=3, sequence_length=2e6,
            recombination_rate=1e-8, random_seed=1, ref_impurity=0.3, Ne=1000, T_admix=150, T_split=5000)
    node_pop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    pid = {n: p for p, n in names.items()}
    of = lambda nm: [int(s) for s in ts.samples() if node_pop[s] == pid[nm]]
    impure = of(REF_A_IMPURE) + of(REF_B_IMPURE)
    labels = {s: 0 for s in of(SOURCE_A) + of(REF_A_IMPURE)}
    labels.update({s: 1 for s in of(SOURCE_B) + of(REF_B_IMPURE)})
    truth, _ = local_ancestry_truth(ts)
    rt = map_truth({r: truth[r] for r in impure}, {pid[SOURCE_A]: 0, pid[SOURCE_B]: 1})

    ft = foreign_tracts(ts, labels, impure, soft_refs=set(impure), max_iter=8)
    total_det = total_hit = 0.0
    for r in impure:
        true_foreign = [(l, rr) for (l, rr, st) in rt[r] if st != labels[r]]
        for (l, rr, _sc) in ft[r]:
            total_det += rr - l
            total_hit += _span_overlap(l, rr, true_foreign)
    assert total_det > 0                                # flags something
    assert total_hit / total_det > 0.5                  # most flagged span is truly foreign
