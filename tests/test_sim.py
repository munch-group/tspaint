"""Rung 0 gate (CLAUDE.md §9, §5.1): the msprime truth-simulator and the
front-end-agnostic persistence premise."""
import numpy as np
import pytest

from tspaint.sim import (
    Simulation,
    admixture_demography,
    simulate_admixture,
    simulate_admixture_impure_refs,
    simulate_admixture_with_ghost,
    local_ancestry_truth,
    pop_role,
    check_admixture_contract,
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
def sim():
    return simulate_admixture(admixture_demography(), n_query=8, n_reference=8,
                              sequence_length=2e6, recombination_rate=1e-8, random_seed=7)


@pytest.fixture(scope="module")
def ts(sim):
    return sim.ts


def _names(ts):
    out = {}
    for p in range(ts.num_populations):
        try:
            out[p] = ts.population(p).metadata.get("name", str(p))
        except Exception:
            out[p] = str(p)
    return out


def test_simulation_shape(sim):
    # the new sim API returns a Simulation carrying queries / labels / truth / sample_sets
    assert isinstance(sim, Simulation)
    # 8 admixed individuals x ploidy 2 -> 16 query haplotypes; A + B refs, 8 each x 2 -> 32 labels
    assert len(sim.queries) == 16
    assert len(sim.labels) == 32 and set(sim.labels.values()) == {0, 1}
    # truth is keyed by query and covers the whole sequence with ancestry states 0/1
    assert set(sim.truth_states) == set(sim.queries)
    for tracts in sim.truth_states.values():
        assert tracts[0][0] == 0.0 and tracts[-1][1] == sim.ts.sequence_length
        assert all(st in (0, 1) for (_, _, st) in tracts)
    # sample_sets group nodes by population name; queries are exactly the ADMIXED set
    assert set(sim.sample_sets) >= {SOURCE_A, SOURCE_B, ADMIXED}
    assert sorted(sim.queries) == sorted(sim.sample_sets[ADMIXED])


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
                                        ref_impurity=rho, Ne=1000, T_admix=300, T_split=5000).ts
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
    ts = simulate_admixture_with_ghost(ghost_fraction=0.25, **common).ts
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

    ts0 = simulate_admixture_with_ghost(ghost_fraction=0.0, **common).ts  # matched no-ghost control
    names0 = _names(ts0)
    pid0 = {n: p for p, n in names0.items()}
    np0 = ts0.tables.nodes.population
    q0 = [int(s) for s in ts0.samples() if np0[s] == pid0[ADMIXED]]
    t0, pn0 = local_ancestry_truth(ts0)
    q0pops = {pn0[p] for q in q0 for (_, _, p) in t0[q]}
    assert GHOST not in q0pops and q0pops <= {SOURCE_A, SOURCE_B}


# --- the hand-built population-role contract (pop_role / check_admixture_contract / ghost) --------

def _three_source_ghost_demography(Ne=2000):
    """Hand-built: three sampled sources A/B/C (states 0/1/2) + one unsampled ghost, nested ((A,B),C)."""
    import msprime
    d = msprime.Demography()
    for i, n in enumerate((SOURCE_A, SOURCE_B, "C")):
        d.add_population(name=n, initial_size=Ne, extra_metadata=pop_role(state=i, source=True, reference=True))
    d.add_population(name=GHOST, initial_size=Ne, extra_metadata=pop_role(ghost=True))   # unsampled foreign
    d.add_population(name=ADMIXED, initial_size=Ne, extra_metadata=pop_role(query=True))
    for n in ("AB", "ANC", "ROOT"):
        d.add_population(name=n, initial_size=Ne)                                          # untagged joins
    d.add_admixture(time=100, derived=ADMIXED, ancestral=[SOURCE_A, SOURCE_B, "C"], proportions=[.3, .3, .4])
    d.add_mass_migration(time=90, source=ADMIXED, dest=GHOST, proportion=0.1)              # ghost pulse
    d.add_census(time=101.0)
    d.add_population_split(time=2000, derived=[SOURCE_A, SOURCE_B], ancestral="AB")
    d.add_population_split(time=6000, derived=["AB", "C"], ancestral="ANC")
    d.add_population_split(time=20000, derived=["ANC", GHOST], ancestral="ROOT")           # deep ghost outgroup
    d.sort_events()
    return d


def test_hand_built_contract_three_sources_plus_ghost():
    # a fully hand-built demography, tagged with pop_role, interpreted by simulate_admixture
    sim = simulate_admixture(_three_source_ghost_demography(), n_query=8, n_reference=6,
                             sequence_length=1e6, recombination_rate=1e-8, random_seed=1)
    assert sim.source_states == (0, 1, 2)                              # three paintable sources
    tstates = {st for q in sim.queries for (_, _, st) in sim.truth_states[q]}
    assert tstates == {0, 1, 2, 3}                                    # ghost EMBEDDED above the sources
    assert {st for q in sim.queries for (_, _, st) in sim.ghost_states[q]} == {3}   # ghost-only truth
    assert any(sim.ghost_states[q] for q in sim.queries)              # a query really carries ghost
    assert GHOST not in sim.sample_sets                               # ghost unsampled
    assert set(sim.labels.values()) == {0, 1, 2}                      # A/B/C are the labelled panel


def test_check_admixture_contract_guardrails():
    import msprime
    Ne = 1000
    # missing census -> silent-garbage-truth footgun becomes a clear error
    d = msprime.Demography()
    d.add_population(name="Q", initial_size=Ne, extra_metadata=pop_role(query=True))
    d.add_population(name="A", initial_size=Ne, extra_metadata=pop_role(state=0, source=True))
    with pytest.raises(ValueError, match="census"):
        check_admixture_contract(d)
    # no query population
    d2 = msprime.Demography()
    d2.add_population(name="A", initial_size=Ne, extra_metadata=pop_role(state=0, source=True))
    d2.add_census(time=10)
    with pytest.raises(ValueError, match="query"):
        check_admixture_contract(d2)
    # a source without a state (its tracts would be silently dropped)
    d3 = admixture_demography()
    d3.add_population(name="NS", initial_size=Ne, extra_metadata=pop_role(source=True))
    with pytest.raises(ValueError, match="state"):
        check_admixture_contract(d3)
    # a ghost combined with a sampling role
    d4 = admixture_demography()
    d4.add_population(name="BADG", initial_size=Ne, extra_metadata=pop_role(ghost=True, source=True, state=5))
    with pytest.raises(ValueError, match="ghost"):
        check_admixture_contract(d4)
    # simulate_admixture runs the check by default
    with pytest.raises(ValueError):
        simulate_admixture(d3, sequence_length=1e5, random_seed=1)


def test_ghost_wrapper_embeds_ghost_truth():
    # the with_ghost preset now tags GHOST explicitly, so its tracts embed in truth_states + ghost_states
    sim = simulate_admixture_with_ghost(n_admix=6, n_ref=4, sequence_length=5e5, recombination_rate=1e-8,
                                        random_seed=1, ghost_fraction=0.25, T_admix=100,
                                        T_split_AB=2000, T_split_ABC=20000, Ne=1000)
    tstates = {st for q in sim.queries for (_, _, st) in sim.truth_states[q]}
    assert sim.source_states == (0, 1) and 2 in tstates              # ghost embedded as state 2
    assert any(sim.ghost_states[q] for q in sim.queries)
    assert GHOST not in sim.sample_sets


def test_misplaced_census_gives_placement_diagnosis():
    # census OLDER than the A/B split -> census nodes land in the ancestral join ANC, not the sources.
    # The guardrail must diagnose the census placement (not tell the user to "tag ANC").
    import msprime
    Ne = 1000
    def build(census_time):
        d = msprime.Demography()
        d.add_population(name="ANC", initial_size=Ne)
        d.add_population(name=SOURCE_A, initial_size=Ne, extra_metadata=pop_role(state=0, source=True, reference=True))
        d.add_population(name=SOURCE_B, initial_size=Ne, extra_metadata=pop_role(state=1, source=True, reference=True))
        d.add_population(name=ADMIXED, initial_size=Ne, extra_metadata=pop_role(query=True))
        d.add_population_split(time=10000, derived=[SOURCE_A, SOURCE_B], ancestral="ANC")
        d.add_census(time=census_time)
        d.add_admixture(time=500, derived=ADMIXED, ancestral=[SOURCE_A, SOURCE_B], proportions=[0.8, 0.2])
        d.sort_events()
        return d

    with pytest.raises(ValueError, match="census.*older than the split"):
        simulate_admixture(build(10001), n_query=6, n_reference=6, sequence_length=3e5, random_seed=1)

    # census placed correctly (between the pulse 500 and the split 10000) -> A/B truth, no error
    sim = simulate_admixture(build(501), n_query=6, n_reference=6, sequence_length=3e5, random_seed=1)
    assert sim.source_states == (0, 1)
    assert {st for q in sim.queries for (_, _, st) in sim.truth_states[q]} == {0, 1}


def test_truth_defining_states_must_be_distinct():
    # each source (and stated ghost) must own a distinct state; a reference/proxy MAY reuse one.
    import msprime
    from tspaint.sim import admixture_demography_with_ref_proxies, admixture_demography_impure_refs
    Ne = 1000
    def build(sa, sb):
        d = msprime.Demography()
        d.add_population(name="ANC", initial_size=Ne)
        d.add_population(name=SOURCE_A, initial_size=Ne, extra_metadata=pop_role(state=sa, source=True, reference=True))
        d.add_population(name=SOURCE_B, initial_size=Ne, extra_metadata=pop_role(state=sb, source=True, reference=True))
        d.add_population(name=ADMIXED, initial_size=Ne, extra_metadata=pop_role(query=True))
        d.add_population_split(time=10000, derived=[SOURCE_A, SOURCE_B], ancestral="ANC")
        d.add_census(time=501)
        d.add_admixture(time=500, derived=ADMIXED, ancestral=[SOURCE_A, SOURCE_B], proportions=[.5, .5])
        d.sort_events()
        return d
    with pytest.raises(ValueError, match="both declare ancestry state"):
        check_admixture_contract(build(0, 0))                 # two sources share state 0
    check_admixture_contract(build(0, 1))                     # distinct -> ok
    # references/proxies legitimately reuse a source's state (the presets rely on this)
    check_admixture_contract(admixture_demography_with_ref_proxies())   # A_prox(0) shares A(0)
    check_admixture_contract(admixture_demography_impure_refs())        # RA(0) shares A(0)
    # a ghost's explicit state may not collide with a source
    dg = admixture_demography()
    dg.add_population(name="G", initial_size=Ne, extra_metadata=pop_role(ghost=True, state=0))
    with pytest.raises(ValueError, match="both declare ancestry state"):
        check_admixture_contract(dg)
