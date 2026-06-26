"""Rung 0 gate (CLAUDE.md §9, §5.1): the msprime truth-simulator and the
front-end-agnostic persistence premise."""
import numpy as np
import pytest

from tspaint.sim import (
    simulate_admixture,
    simulate_admixture_impure_refs,
    simulate_admixture_with_ghost,
    local_ancestry_truth,
    _census_mask,
    SOURCE_A,
    SOURCE_B,
    ADMIXED,
    REF_A_IMPURE,
    REF_B_IMPURE,
    GHOST,
)
from tspaint.diagnostics import persistence_summary, edge_span_summary


@pytest.fixture(scope="module")
def ts():
    return simulate_admixture(n_admix=8, n_ref=8, sequence_length=2e6,
                              recombination_rate=1e-8, random_seed=7)


def _names(ts):
    out = {}
    for p in range(ts.num_populations):
        try:
            out[p] = ts.population(p).metadata.get("name", str(p))
        except Exception:
            out[p] = str(p)
    return out


def test_has_recombination(ts):
    assert ts.num_trees > 1
    assert ts.sequence_length == 2e6


def test_sample_counts(ts):
    # ploidy=2: each individual contributes 2 sample nodes (haplotypes)
    assert ts.num_samples == 2 * (8 + 8 + 8)


def test_census_nodes_sit_in_sources(ts):
    is_census = _census_mask(ts)
    assert is_census.sum() > 0
    names = _names(ts)
    pops = ts.tables.nodes.population
    census_names = {names[int(p)] for p in pops[is_census]}
    # post-admixture census: every census node must be in a source population
    assert census_names <= {SOURCE_A, SOURCE_B}, census_names


def test_truth_covers_genome_without_gaps(ts):
    tracts, pop_name = local_ancestry_truth(ts)
    L = ts.sequence_length
    src_ids = {p for p, n in pop_name.items() if n in (SOURCE_A, SOURCE_B)}
    for s, lst in tracts.items():
        assert lst[0][0] == 0.0
        assert lst[-1][1] == L
        for (l0, r0, _), (l1, r1, _) in zip(lst, lst[1:]):
            assert r0 == l1                      # contiguous, no gap / overlap
        assert all(pid in src_ids for (_, _, pid) in lst)


def test_admixed_panel_carries_both_ancestries(ts):
    tracts, pop_name = local_ancestry_truth(ts)
    admix_pid = next(p for p, n in pop_name.items() if n == ADMIXED)
    node_pop = ts.tables.nodes.population
    admixed = [int(s) for s in ts.samples() if node_pop[s] == admix_pid]
    assert admixed
    src_ids = {p for p, n in pop_name.items() if n in (SOURCE_A, SOURCE_B)}
    pooled = {pid for s in admixed for (_, _, pid) in tracts[s]}
    assert pooled == src_ids                     # both A and B appear across the panel


def test_persistence_premise_met(ts):
    summ = persistence_summary(ts)
    assert summ["max"] > 1                       # clades persist across many trees
    assert summ["frac_singletons"] < 0.95
    assert edge_span_summary(ts)["n_edges"] > 0


def test_impure_refs_carry_known_minority_ancestry():
    # impure reference panels (CLAUDE.md §2.2): majority their nominal source with a known
    # minority foreign, each haplotype a genuine mosaic, census truth still only in sources
    rho = 0.15
    ts = simulate_admixture_impure_refs(n_admix=4, n_pure=4, n_impure=8, sequence_length=4e6,
                                        recombination_rate=1e-8, random_seed=3,
                                        ref_impurity=rho, Ne=1000, T_admix=300, T_split=5000)
    names = _names(ts)
    pid = {n: p for p, n in names.items()}
    node_pop = ts.tables.nodes.population
    tracts, _ = local_ancestry_truth(ts)
    A_id, B_id = pid[SOURCE_A], pid[SOURCE_B]
    L = ts.sequence_length

    for panel, host in [(REF_A_IMPURE, A_id), (REF_B_IMPURE, B_id)]:
        refs = [int(s) for s in ts.samples() if node_pop[s] == pid[panel]]
        assert refs
        foreign_frac, mosaics = [], 0
        for s in refs:
            oth = sum(r - l for (l, r, p) in tracts[s] if p != host)
            foreign_frac.append(oth / L)
            if 0.0 < oth < L:                    # both ancestries present -> a genuine mosaic
                mosaics += 1
        # majority-native on average, with a real (sub-half) minority foreign present
        assert 0.02 < np.mean(foreign_frac) < 0.5
        assert mosaics >= 1                      # at least one haplotype carries foreign tracts
    # census truth lands only in the two sources, impure references included
    src = {A_id, B_id}
    assert all(p in src for s in ts.samples() for (_, _, p) in tracts[int(s)])


def test_ghost_source_tracts_and_no_ghost_control():
    # Plan A Phase 3: an unsampled ghost source C contributes to the queries (census truth
    # labels A/B/GHOST); refs are A/B only; ghost_fraction=0 is the matched control.
    common = dict(n_admix=12, n_ref=8, sequence_length=2e6, recombination_rate=1e-8,
                  random_seed=1, T_admix=100, T_split_AB=2000, T_split_ABC=20000, Ne=1000)
    ts = simulate_admixture_with_ghost(ghost_fraction=0.25, **common)
    names = _names(ts)
    pid = {n: p for p, n in names.items()}
    node_pop = ts.tables.nodes.population
    L = ts.sequence_length
    of = lambda nm: [int(s) for s in ts.samples() if node_pop[s] == pid[nm]]
    queries = of(ADMIXED)
    gid = pid[GHOST]
    tracts, popname = local_ancestry_truth(ts)

    qpops = {popname[p] for q in queries for (_, _, p) in tracts[q]}
    assert GHOST in qpops and qpops <= {SOURCE_A, SOURCE_B, GHOST}     # ghost present in queries
    ghost_frac = np.mean([sum(r - l for (l, r, p) in tracts[q] if p == gid) / L for q in queries])
    assert ghost_frac > 0.03                                          # detectable ghost ancestry
    assert all(names[node_pop[s]] in (SOURCE_A, SOURCE_B) for s in (of(SOURCE_A) + of(SOURCE_B)))

    ts0 = simulate_admixture_with_ghost(ghost_fraction=0.0, **common)  # matched no-ghost control
    names0 = _names(ts0)
    pid0 = {n: p for p, n in names0.items()}
    np0 = ts0.tables.nodes.population
    q0 = [int(s) for s in ts0.samples() if np0[s] == pid0[ADMIXED]]
    t0, pn0 = local_ancestry_truth(ts0)
    q0pops = {pn0[p] for q in q0 for (_, _, p) in t0[q]}
    assert GHOST not in q0pops and q0pops <= {SOURCE_A, SOURCE_B}
