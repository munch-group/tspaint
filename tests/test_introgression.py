"""Plan A: the foreignness primitive (Phase 1) and the workflows built on it."""
import numpy as np
import pytest
import tskit

from tspaint.model import make_generator_2state, tip_emission, query_emission
from tspaint.introgression import (foreignness_track, ForeignnessSegment, reference_qc,
                                 foreign_tracts, ReferenceQC)
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
def test_reference_qc_ensemble_identical_members_matches_single():
    """Ensemble reference_qc (SINGER-style): pooled fit, per-member LOO, merged maps. A 2-member
    ensemble of the SAME ts reproduces the single-ts audit and yields MergedSegment maps with a ~0
    band (proving the per-member paint + merge path, not a single-ts shortcut)."""
    from tspaint.sim import (simulate_admixture_impure_refs, SOURCE_A, SOURCE_B,
                             REF_A_IMPURE, REF_B_IMPURE)
    from tspaint.ensemble import MergedSegment

    ts = simulate_admixture_impure_refs(n_admix=2, n_pure=6, n_impure=3, sequence_length=6e5,
            recombination_rate=1e-8, random_seed=1, ref_impurity=0.3, Ne=1000, T_admix=150, T_split=5000)
    node_pop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    pid = {n: p for p, n in names.items()}
    of = lambda nm: [int(s) for s in ts.samples() if node_pop[s] == pid[nm]]
    labels = {s: 0 for s in of(SOURCE_A) + of(REF_A_IMPURE)}
    labels.update({s: 1 for s in of(SOURCE_B) + of(REF_B_IMPURE)})

    single = reference_qc(ts, labels, max_iter=6, n_jobs=1)
    ens = reference_qc([ts, ts], labels, max_iter=6, n_jobs=1)       # ensemble of two identical members

    # maps are the MERGED ensemble objects: span the genome, band ~0 for identical members
    for r, segs in ens.maps.items():
        assert segs and segs[0].left == 0.0 and segs[-1].right == ts.sequence_length
        assert all(isinstance(s, MergedSegment) for s in segs)
        assert all(np.allclose(s.posterior_std, 0.0, atol=1e-6) for s in segs)
    # reproduces the single-ts audit's structure: same anchor core, same soft set, same pass-1 LOO
    # self-agreement (the prior-free pass). learned_w for soft refs legitimately differs — the pooled
    # fit applies the Beta prior once to M-fold evidence, so w moves toward empirical (CLAUDE.md §6).
    assert ens.anchors == single.anchors
    assert set(ens.soft_refs()) == set(single.soft_refs())
    for r in single.labels:
        assert np.isclose(ens.loo_agreement[r], single.loo_agreement[r], atol=1e-6)


def test_reference_qc_reports_ids_on_stamped_ts():
    """On a stamped (diploid) tree sequence the audit is legible in the user's ids: each summary row
    names the individual + haplotype and maps to the right node, and soft_refs(by='individual')
    returns individual ids (a diploid reference's two haplotype nodes share one individual id).
    Backward compatible: an unstamped tree sequence reports node integers only."""
    from tspaint.sim import (simulate_admixture_impure_refs, SOURCE_A, SOURCE_B,
                             REF_A_IMPURE, REF_B_IMPURE)
    from tspaint.ids import attach_sample_ids

    ts = simulate_admixture_impure_refs(n_admix=2, n_pure=3, n_impure=2, sequence_length=3e5,
            recombination_rate=1e-8, random_seed=1, ref_impurity=0.3, Ne=1000, T_admix=150, T_split=5000)
    npop = ts.tables.nodes.population
    pname = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    pid = {n: p for p, n in pname.items()}
    samples = [int(s) for s in ts.samples()]
    ind_of = {s: ts.node(s).individual for s in samples}
    ref_pops = {SOURCE_A, SOURCE_B, REF_A_IMPURE, REF_B_IMPURE}

    # stamp per-haplotype names "<pop>__<ind>_<k>" so a diploid individual's two nodes share a base id
    nbn = {}
    for ind in sorted(set(ind_of.values())):
        nodes = sorted(n for n in samples if ind_of[n] == ind)
        base = f"{pname[npop[nodes[0]]]}__{ind}"
        for k, n in enumerate(nodes):
            nbn[n] = f"{base}_{k}"
    sts = attach_sample_ids(ts, [nbn[s] for s in samples], ploidy=2)

    iids = [sts.individual(i).metadata["id"] for i in range(sts.num_individuals)]
    ref_ids = [b for b in iids if b.split("__")[0] in ref_pops]
    labels = {b: (0 if b.split("__")[0] in (SOURCE_A, REF_A_IMPURE) else 1) for b in ref_ids}
    anchors = {b for b in ref_ids if b.split("__")[0] in (SOURCE_A, SOURCE_B)}
    qc = reference_qc(sts, labels, anchors=anchors, max_iter=4)

    # every row names an individual + haplotype, and the node index belongs to that individual
    for row in qc.summary():
        node = row["ref"]
        assert row["individual"] == sts.individual(sts.node(node).individual).metadata["id"]
        assert row["haplotype"] == sts.node(node).metadata["id"]
    # soft_refs "by node" and "by individual" are consistent, and by='individual' is str ids
    soft_nodes = qc.soft_refs()
    soft_inds = qc.soft_refs(by="individual")
    assert soft_inds == {qc.individual_ids[n] for n in soft_nodes}
    assert soft_inds and all(isinstance(x, str) and "__" in x for x in soft_inds)

    # backward compatible: an UNSTAMPED ts reports node integers only
    of = lambda nm: [int(s) for s in ts.samples() if npop[s] == pid[nm]]
    labels0 = {s: 0 for s in of(SOURCE_A)}
    labels0.update({s: 1 for s in of(SOURCE_B)})
    qc0 = reference_qc(ts, labels0, max_iter=4)
    assert qc0.individual_ids is None and qc0.sample_ids is None
    assert "individual" not in qc0.summary()[0]
    with pytest.raises(ValueError, match="stamped"):
        qc0.soft_refs(by="individual")


@pytest.mark.slow
def test_reference_qc_n_jobs_matches_serial():
    """reference_qc(..., n_jobs=P) parallelises the EM fits and leave-one-out paints — the compute,
    not the cheap soft_refs()/summary() lookups. It matches serial (fit reduction order differs by
    ULPs, so credibility is allclose; the anchor set / soft_refs are identical) and progress runs."""
    from tspaint.sim import simulate_admixture_impure_refs, SOURCE_A, SOURCE_B, REF_A_IMPURE, REF_B_IMPURE
    ts = simulate_admixture_impure_refs(n_admix=2, n_pure=6, n_impure=3, sequence_length=6e5,
            recombination_rate=1e-8, random_seed=2, ref_impurity=0.3, Ne=1000, T_admix=150, T_split=5000)
    node_pop = ts.tables.nodes.population
    pid = {ts.population(p).metadata.get("name", str(p)): p for p in range(ts.num_populations)}
    of = lambda nm: [int(s) for s in ts.samples() if node_pop[s] == pid[nm]]
    labels = {s: 0 for s in of(SOURCE_A) + of(REF_A_IMPURE)}
    labels.update({s: 1 for s in of(SOURCE_B) + of(REF_B_IMPURE)})

    serial = reference_qc(ts, labels, max_iter=6)
    par = reference_qc(ts, labels, max_iter=6, n_jobs=3, progress=True)   # progress must not error
    assert serial.anchors == par.anchors and serial.soft_refs() == par.soft_refs()
    for r in labels:
        assert abs(serial.credibility[r] - par.credibility[r]) < 1e-9    # allclose (fit ULPs)


@pytest.mark.slow
def test_deep_foreign_flag_finds_unsampled_source():
    # The deep foreign-tract FLAG — foreign_tracts(mode="fit", min_score, min_depth) — detects an
    # unsampled ghost source (low fit + deep), with a low false-positive burden on the matched
    # no-ghost control. This is the former detect_ghost flag, now folded into foreign_tracts; the
    # accurate generative detector is the detect_ghost HMM (tests/test_archaic.py). CLAUDE.md §9.
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
        tr, _ = local_ancestry_truth(ts)
        gid = pid[GHOST]
        true_ghost = {q: [(l, r) for (l, r, p) in tr[q] if p == gid] for q in queries}
        L = float(ts.sequence_length)
        flagged = foreign_tracts(ts, labels, queries, mode="fit", min_score=0.8, min_depth=0.9)
        tracts = {q: [(l, r) for (l, r, _s) in flagged[q]] for q in queries}     # fit < 0.6 AND deep
        burden = {q: sum(r - l for (l, r) in tracts[q]) / L for q in queries}
        return queries, true_ghost, tracts, burden

    queries, true_ghost, gh, burden_d = run(0.25, 1)
    q0, _, _gh0, burden0_d = run(0.0, 1)

    # detection: the genome-wide ghost burden is far above the no-ghost control
    burden = float(np.mean([burden_d[q] for q in queries]))
    burden0 = float(np.mean([burden0_d[q] for q in q0]))
    assert burden > 3 * burden0
    assert burden0 < 0.05                              # false-positive floor (measured ~0.01)

    # localisation: the flagged tracts overlap the true ghost tracts (measured recall ~0.58 / prec ~1.0)
    det = sum(r - l for q in queries for (l, r) in gh[q])
    det_ghost = sum(_span_overlap(l, r, true_ghost[q]) for q in queries for (l, r) in gh[q])
    total_ghost = sum(r - l for q in queries for (l, r) in true_ghost[q])
    assert det_ghost / total_ghost > 0.4               # recall
    assert det_ghost / det > 0.7                        # precision


def test_reference_qc_soft_refs_and_mask():
    # Task 1: reference_qc's result is actionable — .soft_refs() feeds paint(soft_refs=...) and
    # .mask() gives the per-reference foreign spans to drop.
    from tspaint.output import Segment, INFORMATIVE as INF
    qc = ReferenceQC(
        labels={0: 0, 1: 0, 2: 1, 3: 1},
        credibility={0: 0.95, 1: 0.55, 2: 0.97, 3: 0.60},
        loo_agreement={0: 0.95, 1: 0.55, 2: 0.97, 3: 0.60},
        learned_w={1: 0.55, 3: 0.60}, anchors={0, 2},
        maps={0: [Segment(0.0, 100.0, np.array([0.9, 0.1]), INF)],
              1: [Segment(0.0, 50.0, np.array([0.2, 0.8]), INF),       # foreign on [0,50)
                  Segment(50.0, 100.0, np.array([0.95, 0.05]), INF)],
              2: [Segment(0.0, 100.0, np.array([0.05, 0.95]), INF)],
              3: [Segment(0.0, 100.0, np.array([0.1, 0.9]), INF)]},
        Q=np.eye(2), pi=np.array([0.5, 0.5]), _length=100.0)

    assert qc.soft_refs() == {1, 3}                      # the softened (non-anchor) suspects
    assert qc.soft_refs(max_credibility=0.6) == {1}      # cred < 0.6 -> only ref 1 (0.55)
    m = qc.mask(deadband=0.3)
    assert m == {1: [(0.0, 50.0)]}                        # only ref 1's confidently-foreign span


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


# --- n_jobs parallelism: byte-identical to serial (genome-chunk split + seam-stitch) -----------

def _admix_ts(seed=3):
    import tspaint
    return tspaint.io.add_mutations(
        tspaint.simulate_admixture(n_admix=4, n_ref=4, sequence_length=4e5, recombination_rate=1e-8,
                                   random_seed=seed, Ne=1000, T_admix=200, T_split=5000, f_A=0.5),
        rate=2e-7, random_seed=seed)


def test_foreignness_track_parallel_byte_identical():
    import tspaint
    from tspaint.em import fit, build_emissions
    from tspaint.parallel import foreignness_track_parallel
    ts = _admix_ts()
    labels = {i: (0 if i < 4 else 1) for i in range(4, 12)}
    samples = list(range(4)) + list(range(4, 12))
    assert ts.num_trees > 4
    res = fit(ts, labels, K=2, max_iter=8, estimate_pi=False)
    em = build_emissions(ts, labels, res.w, res.pi)
    serial = foreignness_track(ts, res.Q, res.pi, em, labels, focal=samples, depth="rank")
    par = foreignness_track_parallel(ts, res.Q, res.pi, w=res.w, labels=labels, emissions=em,
                                     focal=samples, depth="rank", n_jobs=3)   # genome-wide rank post-stitch
    assert serial.keys() == par.keys()
    for s in samples:
        assert len(serial[s]) == len(par[s])
        for a, b in zip(serial[s], par[s]):
            assert a.left == b.left and a.right == b.right and a.status == b.status
            assert np.array_equal(a.loo, b.loo) and a.fit == b.fit
            assert a.depth == b.depth or (np.isnan(a.depth) and np.isnan(b.depth))


def test_foreign_tracts_n_jobs_matches_serial():
    import tspaint
    ts = _admix_ts()
    labels = {i: (0 if i < 4 else 1) for i in range(4, 12)}
    samples = list(range(4)) + list(range(4, 12))
    s1 = tspaint.foreign_tracts(ts, labels, samples, n_jobs=1)
    s3 = tspaint.foreign_tracts(ts, labels, samples, n_jobs=3)
    assert s1.keys() == s3.keys()
    for k in s1:                                          # fit is parallel-reduced -> allclose, not exact
        assert len(s1[k]) == len(s3[k])
        for u, v in zip(s1[k], s3[k]):
            assert np.allclose(u, v, atol=1e-6)
