"""Rung 0 gate (CLAUDE.md §9, §5.1): the msprime truth-simulator and the
front-end-agnostic persistence premise."""
import numpy as np
import pytest

from tspaint.sim import (
    simulate_admixture,
    local_ancestry_truth,
    _census_mask,
    SOURCE_A,
    SOURCE_B,
    ADMIXED,
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
