"""Simulated admixture with **known local ancestry** (CLAUDE.md §9, primary
validation track).

msprime tree sequences carry native cross-tree node-ID stability — the
edge-persistence invariant the blocking depends on (CLAUDE.md §5) — so they serve
as both the development/test front end and the ground-truth source, removing the
Relate C++ toolchain from the critical path. Relate ``--compress`` integration is
a separate, later track (``io_relate.py``).

Ground truth uses a **census event** placed just older than the admixture pulse:
every lineage is then unambiguously in a source population, and the census node on
each lineage records that source per genomic segment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import msprime
import tskit

__all__ = [
    "SOURCE_A",
    "SOURCE_B",
    "ADMIXED",
    "ANCESTRAL",
    "REF_A_IMPURE",
    "REF_B_IMPURE",
    "REF_A_PROXY",
    "REF_B_PROXY",
    "GHOST",
    "ROLE_QUERY",
    "ROLE_SOURCE",
    "ROLE_REFERENCE",
    "ROLE_STATE",
    "ROLE_GHOST",
    "pop_role",
    "check_admixture_contract",
    "Simulation",
    "admixture_demography",
    "simulate_admixture",
    "admixture_demography_impure_refs",
    "simulate_admixture_impure_refs",
    "admixture_demography_source_gene_flow",
    "simulate_admixture_source_gene_flow",
    "admixture_demography_with_ref_proxies",
    "simulate_admixture_with_ref_proxies",
    "admixture_demography_with_ghost",
    "simulate_admixture_with_ghost",
    "local_ancestry_truth",
]

SOURCE_A = "A"
SOURCE_B = "B"
ADMIXED = "ADMIX"
ANCESTRAL = "ANC"
REF_A_IMPURE = "RA"   # impure reference panel: mostly A, a known minority B (CLAUDE.md §2.2, §6)
REF_B_IMPURE = "RB"   # impure reference panel: mostly B, a known minority A
GHOST = "GHOST"       # unsampled ("ghost") source contributing introgression (CLAUDE.md §9)
AB_ANCESTRAL = "AB"   # intermediate ancestor of A and B; the ghost C is the deeper outgroup
REF_A_PROXY = SOURCE_A + "_prox"   # deeply-divergent sister of A, used as a clean proxy reference (§9, §10)
REF_B_PROXY = SOURCE_B + "_prox"   # deeply-divergent sister of B
A_ANCESTRAL = "A_ANC"              # common ancestor of A and its proxy A_prox
B_ANCESTRAL = "B_ANC"              # common ancestor of B and its proxy B_prox

# --- the population-role contract (CLAUDE.md §9) --------------------------------------------------
# A hand-built demography tags each population's role by writing these keys into its
# ``extra_metadata`` (via :func:`pop_role`); :func:`simulate_admixture` reads them to produce the
# Simulation. These five strings are the *only* coupling between a demography and tspaint.
ROLE_QUERY = "tspaint_query"          # the admixed population; its samples are the queries to paint
ROLE_SOURCE = "tspaint_source"        # a census source that defines ground truth; sampled as a panel
ROLE_REFERENCE = "tspaint_reference"  # sampled and placed in the default Simulation.labels panel
ROLE_STATE = "tspaint_state"          # the ancestry-state index a source/reference/ghost carries
ROLE_GHOST = "tspaint_ghost"          # a deliberate UNSAMPLED foreign source (its truth is embedded)

# msprime flags census-event nodes; fall back to time-based detection if the
# constant is unavailable in the installed version.
_CENSUS_FLAG = getattr(msprime, "NODE_IS_CEN_EVENT", 0)


@dataclass
class Simulation:
    """A simulated admixture with known truth — the return of :func:`simulate_admixture`.

    Everything downstream needs (queries to paint, reference labels, ground-truth ancestry) is
    derived from the demography's own population **roles** (see :func:`admixture_demography`), so no
    caller ever needs the population-name constants.

    Attributes
    ----------
    ts : tskit.TreeSequence
        The simulated tree sequence — **with mutations** when ``mutation_rate`` was given.
    queries : list[int]
        Query (admixed) haplotype node ids.
    labels : dict[int, int]
        Default reference panel: reference haplotype node id → ancestry-state index (``0`` / ``1``),
        ready to pass to :func:`tspaint.paint`.
    truth_states : dict[int, list[tuple[float, float, int]]]
        Per query, the true local-ancestry tracts ``(left, right, state)`` from the census — ready
        to score against a painting (e.g. :func:`tspaint.validate.balanced_accuracy`). **Ghost
        tracts are embedded here** with their own state index (beyond the source states), so they
        appear in the truth band of :meth:`tspaint.Painting.plot`\\ 's truth track alongside the
        source segments.
    sample_sets : dict[str, list[int]]
        Population name → its sample node ids, for building custom reference panels (e.g. proxy-only
        vs direct-source labels).
    ghost_states : dict[int, list[tuple[float, float, int]]]
        Per query, the ghost-only subset of :attr:`truth_states` — tracts descending from a
        ``ghost``-tagged (unsampled, un-paintable) source. Empty when the demography has no ghost.
        The ground truth to score a reference-free :func:`tspaint.detect_ghost` against.
    source_states : tuple[int, ...]
        The **paintable** ancestry-state indices (from ``source``/``reference`` populations), sorted —
        e.g. ``(0, 1, 2)`` for three sources. Pass ``K=len(source_states)`` to :func:`tspaint.paint`;
        any state in :attr:`truth_states` beyond these is a ghost.
    demography : msprime.Demography
        The role-tagged demography the tree sequence was simulated under (the ``demography`` argument
        to :func:`simulate_admixture`), retained so the true split / admixture times and population
        structure are recoverable from the result (e.g. to check a dated split against the truth).
    """
    ts: object
    queries: list
    labels: dict
    truth_states: dict
    sample_sets: dict
    ghost_states: dict = None
    source_states: tuple = ()
    demography: object = None


def pop_role(*, state=None, query=False, source=False, reference=False, ghost=False):
    """Tag a population's role in the admixture contract — pass to ``add_population(extra_metadata=…)``.

    The one function a user needs to hand-build a demography :func:`simulate_admixture` can interpret:
    build any :class:`msprime.Demography`, then tag each population with ``extra_metadata=pop_role(…)``.
    The roles (all optional, combinable except as noted):

    * ``query`` — the admixed population; its samples become :attr:`Simulation.queries` and receive the
      census truth. Exactly one population must be a query.
    * ``source`` — a census source population whose ``state`` defines the ground truth; sampled as a
      panel. Must carry a ``state``.
    * ``reference`` — sampled and placed in the default :attr:`Simulation.labels` (the panel
      :func:`tspaint.paint` uses). Must carry a ``state``. Usually combined with ``source``.
    * ``state`` — the ancestry-state index (``0, 1, 2, …``) a source / reference / ghost carries.
    * ``ghost`` — a **deliberate unsampled foreign source** (e.g. a Neanderthal-like ghost): not
      sampled and not in ``labels``, but its census tracts are still embedded in
      :attr:`Simulation.truth_states` (with a state beyond the source states) and surfaced in
      :attr:`Simulation.ghost_states`. Mutually exclusive with ``query`` / ``source`` / ``reference``.
      A ``state`` is optional (auto-assigned above the source states if omitted).

    An **untagged** population (no ``pop_role``) is a structural join (e.g. an ancestral node) that is
    never sampled and carries no truth. Returns the ``extra_metadata`` dict.

    Parameters
    ----------
    state : int, optional
        Ancestry-state index (``0, 1, 2, …``) the population carries, stored under ``ROLE_STATE``.
        Required for a ``source`` / ``reference`` (its census tracts map to this state); optional
        for a ``ghost`` (auto-assigned above the source states when omitted). ``None`` (default)
        writes no state key.
    query : bool, optional
        Tag the admixed query population (its samples become :attr:`Simulation.queries`); exactly
        one population must be the query. Default ``False``.
    source : bool, optional
        Tag a census source population that defines the ground truth and is sampled as a panel.
        Default ``False``.
    reference : bool, optional
        Tag a population to sample into the default :attr:`Simulation.labels` panel (usually
        combined with ``source``). Default ``False``.
    ghost : bool, optional
        Tag a deliberately unsampled foreign source (mutually exclusive with ``query`` / ``source``
        / ``reference``). Default ``False``.

    Returns
    -------
    dict
        The ``extra_metadata`` dict of role flags (plus ``state`` when given) to hand to
        :meth:`msprime.Demography.add_population`\\ 's ``extra_metadata=`` argument — empty when
        nothing is set.

    Examples
    --------
    >>> d.add_population(name="A", initial_size=Ne,
    ...                  extra_metadata=pop_role(state=0, source=True, reference=True))
    >>> d.add_population(name="GHOST", initial_size=Ne, extra_metadata=pop_role(ghost=True))
    """
    m = {}
    if query:
        m[ROLE_QUERY] = True
    if source:
        m[ROLE_SOURCE] = True
    if reference:
        m[ROLE_REFERENCE] = True
    if ghost:
        m[ROLE_GHOST] = True
    if state is not None:
        m[ROLE_STATE] = int(state)
    return m


# Backwards-compatible private alias (the preset builders below still call it).
def _pop_role(state=None, query=False, source=False, reference=False, ghost=False):
    return pop_role(state=state, query=query, source=source, reference=reference, ghost=ghost)


def check_admixture_contract(demography, *, samples=None):
    """Validate a role-tagged demography — raise clear errors instead of silently-wrong truth.

    Checks the :func:`pop_role` contract on ``demography`` and raises :class:`ValueError` on the
    mistakes that would otherwise pass silently and corrupt the ground truth:

    * **no census** event (``d.add_census(...)``) — truth would fall back to a fragile
      most-common-node-time heuristic;
    * **no query** population (``pop_role(query=True)``);
    * a **source / reference** without a ``state`` — its census tracts would be silently dropped;
    * a **ghost** combined with ``query`` / ``source`` / ``reference`` — a ghost is unsampled;
    * two **truth-defining** populations (source / stated ghost) sharing a ``state`` — that would
      collapse two ancestries into one (a reference/proxy may reuse a source's state as a panel);
    * a ``samples=`` key that is not a population name in the demography.

    Called automatically by :func:`simulate_admixture` (opt out with ``validate=False``); also handy
    standalone while building a demography. Note the "untagged population that carries census
    ancestry" case needs the simulated tree sequence and so is checked inside
    :func:`simulate_admixture`, not here.

    Parameters
    ----------
    demography : msprime.Demography
        A role-tagged demography whose populations carry :func:`pop_role` ``extra_metadata``.
    samples : dict[str, int], optional
        The ``{population_name: n_individuals}`` sampling scheme, if any — only its **keys** are
        checked, and each must name a population in ``demography``. ``None`` (default) skips that
        check.

    Raises
    ------
    ValueError
        On any contract violation: no census event, no query population, a source / reference
        without a ``state``, a ghost combined with query / source / reference, two truth-defining
        populations (a source or a stated ghost) sharing a ``state``, or a ``samples`` key that is
        not a population name in ``demography``.
    """
    roles = {p.name: dict(p.extra_metadata or {}) for p in demography.populations}
    if not any(type(ev).__name__ == "CensusEvent" for ev in demography.events):
        raise ValueError(
            "no census in the demography; add d.add_census(time=...) just older than the admixture "
            "pulse. Without it local_ancestry_truth falls back to a time heuristic and is silently wrong")
    if not any(m.get(ROLE_QUERY) for m in roles.values()):
        raise ValueError("no query population; tag the admixed population with pop_role(query=True)")
    for nm, m in roles.items():
        if m.get(ROLE_GHOST) and (m.get(ROLE_QUERY) or m.get(ROLE_SOURCE) or m.get(ROLE_REFERENCE)):
            raise ValueError(f"population '{nm}' is ghost AND query/source/reference; a ghost is "
                             "unsampled — use pop_role(ghost=True) alone (a state is optional)")
        if (m.get(ROLE_SOURCE) or m.get(ROLE_REFERENCE)) and ROLE_STATE not in m:
            raise ValueError(f"source/reference population '{nm}' has no ancestry state; pass "
                             "pop_role(state=..., source=True) (or reference=True)")
    # Each truth-defining population — every source, and every explicitly-stated ghost — must own a
    # DISTINCT state; otherwise the census→state truth map collapses two sources into one ancestry.
    # A reference / proxy MAY reuse a source's state (it is a sampled panel *for* that ancestry, e.g.
    # the proxy or impure-reference presets), so references are exempt from this uniqueness check.
    seen = {}
    for nm, m in roles.items():
        if (m.get(ROLE_SOURCE) or m.get(ROLE_GHOST)) and ROLE_STATE in m:
            st = m[ROLE_STATE]
            if st in seen:
                raise ValueError(
                    f"populations '{seen[st]}' and '{nm}' both declare ancestry state {st}; each "
                    "truth-defining source/ghost must have a distinct state (a reference/proxy may "
                    "reuse a source's state as a panel for that ancestry)")
            seen[st] = nm
    if samples is not None:
        unknown = set(samples) - set(roles)
        if unknown:
            raise ValueError(f"samples names {sorted(unknown)} are not populations in the demography")


def admixture_demography(Ne=10_000, T_admix=30.0, census_offset=1.0,
                         T_split=2000.0, f_A=0.3, migration_rate=0.0):
    """Build a two-source admixture demography with a post-pulse census.

    Two sources (A, B) feed an admixed population, with a census placed just
    older than the admixture pulse. A and B contribute fractions ``f_A`` and
    ``1 - f_A`` to ADMIX at ``T_admix``; they merge into a common ancestor at
    ``T_split``. The census sits at ``T_admix + census_offset`` (strictly
    between admixture and split).

    Two demographic regimes (CLAUDE.md §9 — the harder, more realistic one is
    the migration model, where the sources are less differentiated):

    * ``migration_rate == 0`` — the sources are **isolated** between split and
      admixture (the clean two-source pulse).
    * ``migration_rate > 0`` — symmetric **low-level gene flow** between A and B
      over ``[census_time, T_split)`` (the interval where they coexist as
      distinct populations after the census). The census still cleanly labels
      each lineage's source — migration only acts *deeper* than the census — so
      the local-ancestry truth is unchanged, but A and B share more recent
      ancestry, making them harder to tell apart.

    Parameters
    ----------
    Ne : float, optional
        Diploid effective population size for every population.
    T_admix : float, optional
        Time (generations ago) of the admixture pulse forming ADMIX.
    census_offset : float, optional
        Offset added to ``T_admix`` to place the census (must keep it strictly
        between admixture and split).
    T_split : float, optional
        Time (generations ago) at which A and B coalesce into ANCESTRAL.
    f_A : float, optional
        Fraction of ADMIX contributed by source A (B contributes ``1 - f_A``).
    migration_rate : float, optional
        Per-generation symmetric migration rate between A and B over
        ``[census_time, T_split)``. ``0`` (default) gives the isolated model.

    Returns
    -------
    msprime.Demography
        Demography with populations A, B, ADMIX and ANCESTRAL plus the
        admixture, census and split events (and the A↔B migration if requested).

    Raises
    ------
    ValueError
        If ``T_admix < T_admix + census_offset < T_split`` is violated, or
        ``migration_rate < 0``.
    """
    census_time = T_admix + census_offset
    if not (T_admix < census_time < T_split):
        raise ValueError("require T_admix < T_admix + census_offset < T_split")
    if migration_rate < 0:
        raise ValueError("migration_rate must be non-negative")

    d = msprime.Demography()
    d.add_population(name=SOURCE_A, initial_size=Ne, extra_metadata=_pop_role(state=0, source=True, reference=True))
    d.add_population(name=SOURCE_B, initial_size=Ne, extra_metadata=_pop_role(state=1, source=True, reference=True))
    d.add_population(name=ADMIXED, initial_size=Ne, extra_metadata=_pop_role(query=True))
    d.add_population(name=ANCESTRAL, initial_size=Ne)
    # Admixed pop forms (backward in time) as a mixture of the two sources.
    d.add_admixture(time=T_admix, derived=ADMIXED,
                    ancestral=[SOURCE_A, SOURCE_B], proportions=[f_A, 1.0 - f_A])
    # Census after the pulse: every lineage is now in A or B; census nodes label
    # each lineage's source per genomic segment (the local-ancestry truth).
    d.add_census(time=census_time)
    if migration_rate > 0:
        # Symmetric A<->B gene flow, switched on (backward in time) just after the
        # census so the truth labelling is untouched; the split below ends it.
        d.add_migration_rate_change(time=census_time, rate=migration_rate,
                                    source=SOURCE_A, dest=SOURCE_B)
        d.add_migration_rate_change(time=census_time, rate=migration_rate,
                                    source=SOURCE_B, dest=SOURCE_A)
    # Sources coalesce into a common ancestor deeper in time.
    d.add_population_split(time=T_split, derived=[SOURCE_A, SOURCE_B],
                           ancestral=ANCESTRAL)
    return d


def simulate_admixture(demography, *, n_query=10, n_reference=10, samples=None,
                       sequence_length=1e6, recombination_rate=1e-8, ploidy=2,
                       mutation_rate=None, random_seed=42, validate=True):
    """Simulate an admixture from a **role-tagged demography**, returning a :class:`Simulation`.

    Pass any :class:`msprime.Demography` whose populations you have tagged with :func:`pop_role`
    (query / source / reference / ghost + ancestry state) — either hand-built for full control, or
    one of the :func:`admixture_demography` presets. This samples the query and reference panels,
    runs the coalescent, optionally lays down mutations, and packages the queries, reference
    ``labels`` and the census ``truth`` (ghost tracts embedded); the caller never touches a
    population-name constant.

    Parameters
    ----------
    demography : msprime.Demography
        A role-tagged demography — hand-built with :func:`pop_role` (must include an ``add_census``
        just older than the admixture pulse), or from an :func:`admixture_demography` preset.
    n_query : int, optional
        Number of query (admixed) individuals to sample.
    n_reference : int, optional
        Number of individuals sampled from **each** reference / source population.
    samples : dict[str, int], optional
        Explicit ``{population_name: n_individuals}`` sampling, overriding ``n_query`` /
        ``n_reference`` — used by the variant wrappers to sample e.g. unequal pure / impure panels.
    sequence_length, recombination_rate, ploidy, random_seed : optional
        Passed to :func:`msprime.sim_ancestry`.
    mutation_rate : float, optional
        If given, lay down mutations at this per-base rate (:func:`tspaint.io.add_mutations`); the
        returned ``ts`` then carries sites. ``None`` (default) leaves the bare ancestry.
    validate : bool, optional
        Run :func:`check_admixture_contract` first (default ``True``): raise a clear error on a
        missing census, no query, a state-less source, a mis-combined ghost, or an untagged
        census-bearing population, instead of silently producing wrong truth. Set ``False`` to skip.

    Returns
    -------
    Simulation
        ``.ts`` (mutated iff ``mutation_rate`` given), ``.queries``, ``.labels``, ``.truth_states``
        (ghost tracts embedded with a state above the sources), ``.sample_sets``, ``.ghost_states``
        (the ghost-only truth) and ``.source_states`` (the paintable state indices).

    Examples
    --------
    Two-source preset:

    >>> sim = simulate_admixture(admixture_demography(), n_query=5, n_reference=5,
    ...                          sequence_length=1e5, random_seed=1)
    >>> sim.ts.num_samples
    30

    Hand-built: three sampled sources A/B/C + one unsampled ghost, painted at ``K=3`` while
    :func:`tspaint.detect_ghost` finds the ghost; the ghost shows in the truth band of the plot:

    >>> import msprime, tspaint
    >>> from tspaint.sim import pop_role, simulate_admixture
    >>> Ne = 2000
    >>> d = msprime.Demography()
    >>> for i, n in enumerate("ABC"):
    ...     d.add_population(name=n, initial_size=Ne,
    ...                      extra_metadata=pop_role(state=i, source=True, reference=True))
    >>> d.add_population(name="GHOST", initial_size=Ne, extra_metadata=pop_role(ghost=True))
    >>> d.add_population(name="ADMIX", initial_size=Ne, extra_metadata=pop_role(query=True))
    >>> for n in ("AB", "ANC", "ROOT"):
    ...     d.add_population(name=n, initial_size=Ne)
    >>> d.add_admixture(time=100, derived="ADMIX", ancestral=["A", "B", "C"], proportions=[.3, .3, .4])
    >>> d.add_mass_migration(time=90, source="ADMIX", dest="GHOST", proportion=0.05)
    >>> d.add_census(time=101.0)
    >>> d.add_population_split(time=2000, derived=["A", "B"], ancestral="AB")
    >>> d.add_population_split(time=6000, derived=["AB", "C"], ancestral="ANC")
    >>> d.add_population_split(time=20000, derived=["ANC", "GHOST"], ancestral="ROOT")
    >>> d.sort_events()                       # events added out of time order -> sort before simulating
    >>> sim = simulate_admixture(d, n_query=8, n_reference=6, sequence_length=1e6)
    >>> painting = tspaint.paint(sim.ts, sim.labels, sim.queries, K=len(sim.source_states))
    """
    if validate:
        check_admixture_contract(demography, samples=samples)
    roles = {p.name: dict(p.extra_metadata or {}) for p in demography.populations}
    query_pop = next((n for n, m in roles.items() if m.get(ROLE_QUERY)), None)
    if query_pop is None:
        raise ValueError("demography has no query population; tag one with pop_role(query=True)")
    if samples is None:
        ref_pops = [n for n, m in roles.items() if m.get(ROLE_REFERENCE) or m.get(ROLE_SOURCE)]
        samples = {query_pop: n_query, **{r: n_reference for r in ref_pops}}
    ts = msprime.sim_ancestry(samples=samples, demography=demography,
                              sequence_length=sequence_length, recombination_rate=recombination_rate,
                              ploidy=ploidy, random_seed=random_seed)
    if mutation_rate:
        from .io_tsinfer import add_mutations
        ts = add_mutations(ts, rate=mutation_rate, random_seed=random_seed)

    name = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    node_pop = ts.tables.nodes.population
    sample_sets = {}
    for s in ts.samples():
        sample_sets.setdefault(name[node_pop[s]], []).append(int(s))
    queries = sample_sets.get(query_pop, [])
    # default reference labels: sampled reference-tagged populations, by their ancestry state
    labels = {}
    for pop_name, m in roles.items():
        if m.get(ROLE_REFERENCE) and ROLE_STATE in m:
            for s in sample_sets.get(pop_name, ()):
                labels[s] = m[ROLE_STATE]

    # Ancestry states. Sources / references define the paintable states (0..K-1). A ghost gets a state
    # ABOVE them — explicit, else auto-assigned — so its census tracts embed in the query truth (and
    # so render in painting-plot truth bands) while staying distinct from the paintable sources.
    source_state = {nm: m[ROLE_STATE] for nm, m in roles.items()
                    if ROLE_STATE in m and not m.get(ROLE_GHOST)}
    # auto-assign ghost states above the paintable ones, skipping any already used (by a source or an
    # explicitly-stated ghost) so an auto-assigned ghost never collides — states stay one-per-source.
    used = set(source_state.values()) | {int(m[ROLE_STATE]) for m in roles.values()
                                         if m.get(ROLE_GHOST) and ROLE_STATE in m}
    nxt = (max(used) + 1) if used else 0
    ghost_state = {}
    for nm, m in roles.items():
        if m.get(ROLE_GHOST):
            s = m.get(ROLE_STATE)
            if s is not None:
                ghost_state[nm] = int(s)
            else:
                while nxt in used:
                    nxt += 1
                ghost_state[nm] = nxt
                used.add(nxt)
    src_of_pop = {p: source_state[name[p]] for p in range(ts.num_populations) if name[p] in source_state}
    ghost_of_pop = {p: ghost_state[name[p]] for p in range(ts.num_populations) if name[p] in ghost_state}
    state_of_pop = {**src_of_pop, **ghost_of_pop}      # truth = sources + ghosts (embedded)

    tracts, _ = local_ancestry_truth(ts)
    if validate:                                        # a census pop under a query must be tagged
        untagged = {name[pop] for q in queries for (_, _, pop) in tracts[q] if pop not in state_of_pop}
        if untagged:
            # Diagnose the common cause: the census sits *older* than a population split, so its nodes
            # land in the ancestral join (e.g. ANC) instead of the sources — a misplaced census, not a
            # forgotten tag. Distinguish it from a genuinely-untagged leaf source.
            split_anc = {ev.ancestral: float(ev.time) for ev in demography.events
                         if type(ev).__name__ == "PopulationSplit"}
            census_t = next((float(ev.time) for ev in demography.events
                             if type(ev).__name__ == "CensusEvent"), None)
            joins = sorted(untagged & set(split_anc))
            if joins and census_t is not None:
                s = joins[0]
                raise ValueError(
                    f"the census (time={census_t:g}) is older than the split that forms '{s}' "
                    f"(time={split_anc[s]:g}), so census nodes land in the ancestral join {joins} "
                    f"instead of the sources and the truth would be meaningless. Place the census just "
                    f"older than the admixture pulse — between the pulse and the first split, e.g. "
                    f"d.add_census(time=T_admix + 1). (If '{s}' really is a source, tag it "
                    f"pop_role(source=True, state=...).)")
            raise ValueError(
                f"populations {sorted(untagged)} carry census ancestry under a query but have no "
                "role; tag them source/reference (with a state) or ghost — a forgotten tag silently "
                "drops that ancestry from the truth")
    truth = {q: [(l, r, state_of_pop[pop]) for (l, r, pop) in tracts[q] if pop in state_of_pop]
             for q in queries}
    ghost_states = {q: [(l, r, ghost_of_pop[pop]) for (l, r, pop) in tracts[q] if pop in ghost_of_pop]
                    for q in queries}
    return Simulation(ts=ts, queries=queries, labels=labels, truth_states=truth,
                      sample_sets=sample_sets, ghost_states=ghost_states,
                      source_states=tuple(sorted(set(source_state.values()))),
                      demography=demography)


def admixture_demography_impure_refs(Ne=1000, T_admix=30.0, census_offset=1.0,
                                     T_split=5000.0, f_A=0.5, ref_impurity=0.1):
    """Admixture demography whose **reference panels are themselves slightly admixed**.

    Like :func:`admixture_demography` but adds two impure reference populations:
    ``REF_A_IMPURE`` draws a fraction ``1 - ref_impurity`` from source A and
    ``ref_impurity`` from B (and symmetrically ``REF_B_IMPURE``), so a panel nominally
    labelled "A" carries a known minority of genuine B tracts. All three derived
    populations (the queries ``ADMIX`` and the two reference panels) form at ``T_admix``;
    the post-pulse census at ``T_admix + census_offset`` still labels every lineage's
    true source per segment, so :func:`local_ancestry_truth` recovers the references'
    own foreign tracts as ground truth.

    Parameters
    ----------
    Ne : float, optional
        Diploid effective size for every population.
    T_admix : float, optional
        Time (generations ago) of the admixture pulse forming all derived populations.
    census_offset : float, optional
        Offset added to ``T_admix`` to place the census (kept strictly between
        admixture and split).
    T_split : float, optional
        Time at which A and B coalesce into ANCESTRAL.
    f_A : float, optional
        Fraction of the admixed *query* population contributed by source A.
    ref_impurity : float, optional
        Minority (foreign-source) fraction of each impure reference panel, in
        ``[0, 0.5)``. ``0`` reproduces pure references.

    Returns
    -------
    msprime.Demography
        Demography with populations A, B, ADMIX, REF_A_IMPURE, REF_B_IMPURE and
        ANCESTRAL plus the admixture, census and split events.

    Raises
    ------
    ValueError
        If ``T_admix < T_admix + census_offset < T_split`` is violated or
        ``ref_impurity`` is outside ``[0, 0.5)``.
    """
    census_time = T_admix + census_offset
    if not (T_admix < census_time < T_split):
        raise ValueError("require T_admix < T_admix + census_offset < T_split")
    if not (0.0 <= ref_impurity < 0.5):
        raise ValueError("ref_impurity must be in [0, 0.5) (the panel must stay majority-source)")

    d = msprime.Demography()
    d.add_population(name=SOURCE_A, initial_size=Ne, extra_metadata=_pop_role(state=0, source=True, reference=True))
    d.add_population(name=SOURCE_B, initial_size=Ne, extra_metadata=_pop_role(state=1, source=True, reference=True))
    d.add_population(name=ADMIXED, initial_size=Ne, extra_metadata=_pop_role(query=True))
    d.add_population(name=REF_A_IMPURE, initial_size=Ne, extra_metadata=_pop_role(state=0, reference=True))
    d.add_population(name=REF_B_IMPURE, initial_size=Ne, extra_metadata=_pop_role(state=1, reference=True))
    d.add_population(name=ANCESTRAL, initial_size=Ne)
    # Queries: an admixed mixture of the two sources.
    d.add_admixture(time=T_admix, derived=ADMIXED,
                    ancestral=[SOURCE_A, SOURCE_B], proportions=[f_A, 1.0 - f_A])
    # Impure reference panels: mostly their own source, a known minority foreign.
    d.add_admixture(time=T_admix, derived=REF_A_IMPURE, ancestral=[SOURCE_A, SOURCE_B],
                    proportions=[1.0 - ref_impurity, ref_impurity])
    d.add_admixture(time=T_admix, derived=REF_B_IMPURE, ancestral=[SOURCE_A, SOURCE_B],
                    proportions=[ref_impurity, 1.0 - ref_impurity])
    # Census just older than the pulse: every lineage is now unambiguously in A or B.
    d.add_census(time=census_time)
    d.add_population_split(time=T_split, derived=[SOURCE_A, SOURCE_B], ancestral=ANCESTRAL)
    return d


def simulate_admixture_impure_refs(n_admix=10, n_pure=6, n_impure=6, sequence_length=2e6,
                                   recombination_rate=1e-8, ploidy=2, random_seed=42,
                                   mutation_rate=None, ref_impurity=0.1, **demography_kwargs):
    """Simulate admixed queries, a pure reference anchor core, and impure reference panels.

    Samples ``n_admix`` admixed individuals (queries), ``n_pure`` from each pure source
    (A, B — the trusted hard-clamp anchor core, CLAUDE.md §6) and ``n_impure`` from each
    impure reference panel (``REF_A_IMPURE``, ``REF_B_IMPURE`` — nominally A / B but
    carrying a ``ref_impurity`` minority of foreign tracts). The census truth covers
    every sample, so the impure references' own foreign tracts are known ground truth.

    Parameters
    ----------
    n_admix : int, optional
        Number of admixed (query) individuals.
    n_pure : int, optional
        Number of individuals sampled from each pure source (A and B).
    n_impure : int, optional
        Number of individuals sampled from each impure reference panel.
    sequence_length : float, optional
        Sequence length in base pairs.
    recombination_rate : float, optional
        Per-base, per-generation recombination rate.
    ploidy : int, optional
        Ploidy; each individual yields ``ploidy`` sample haplotypes.
    random_seed : int, optional
        Seed for :func:`msprime.sim_ancestry`.
    mutation_rate : float, optional
        If given, overlay mutations on the ARG at this per-base rate so the returned ``.ts`` carries
        sites (:func:`tspaint.io.add_mutations`); ``None`` (default) leaves the bare ancestry. As
        for :func:`simulate_admixture`.
    ref_impurity : float, optional
        Minority foreign fraction of each impure reference panel (see
        :func:`admixture_demography_impure_refs`).
    **demography_kwargs
        Passed to :func:`admixture_demography_impure_refs` (e.g. ``T_admix``,
        ``T_split``, ``f_A``, ``Ne``).

    Returns
    -------
    Simulation
        A :class:`Simulation` (use ``.ts`` for the tree sequence); its sample nodes are haplotypes
        from the admixed, pure-source and impure-reference populations — identify them by node
        population (as the experiments do) rather than by index.

    See Also
    --------
    admixture_demography_impure_refs : The underlying demography.
    local_ancestry_truth : Recover the ground-truth ancestry tracts (references included).
    """
    demography = admixture_demography_impure_refs(ref_impurity=ref_impurity, **demography_kwargs)
    return simulate_admixture(demography, sequence_length=sequence_length,
                              recombination_rate=recombination_rate, ploidy=ploidy,
                              mutation_rate=mutation_rate, random_seed=random_seed,
                              samples={ADMIXED: n_admix, SOURCE_A: n_pure, SOURCE_B: n_pure,
                                       REF_A_IMPURE: n_impure, REF_B_IMPURE: n_impure})


def admixture_demography_source_gene_flow(Ne=1000, T_admix=30.0, census_offset=1.0,
                                          T_prev_admix=1000.0, T_split=5000.0, f_A=0.5,
                                          prev_migration=0.1):
    """Two-source admixture where the **sources themselves exchanged genes via a prior pulse**.

    Like :func:`admixture_demography`, but with a mass migration ``SOURCE_A -> SOURCE_B`` (backward
    in time; = a forward-time pulse of ``SOURCE_B`` **into** ``SOURCE_A``) of fraction
    ``prev_migration`` at ``T_prev_admix`` — placed between the census and the split. So the
    ``SOURCE_A`` reference panel genuinely carries ``~prev_migration`` of ``SOURCE_B`` introgressed
    tracts: **real, localized reference contamination from a demographic event** (not a synthetic
    pure/impure mix). The post-pulse census at ``T_admix + census_offset`` still labels each query
    lineage's proximate source (A / B) — the prior pulse acts *deeper* than the census, so the query
    local-ancestry truth is unchanged. (The A-refs' B tracts sit below the census, so they are not
    in :func:`local_ancestry_truth`; the query A/B truth is the evaluation target.)

    Parameters
    ----------
    Ne, T_admix, census_offset, T_split, f_A : optional
        As for :func:`admixture_demography`.
    T_prev_admix : float, optional
        Time of the prior A↔B pulse (strictly between the census and ``T_split``).
    prev_migration : float, optional
        Fraction of ``SOURCE_A`` replaced (backward) by ``SOURCE_B`` at ``T_prev_admix`` — the
        ``SOURCE_A`` panel's foreign (B) fraction.

    Returns
    -------
    msprime.Demography
        Demography with populations A, B, ADMIX and ANCESTRAL plus the admixture and census events,
        the prior ``SOURCE_A -> SOURCE_B`` mass-migration pulse at ``T_prev_admix``, and the A/B
        split at ``T_split``.

    Raises
    ------
    ValueError
        If ``T_admix < census < T_prev_admix < T_split`` is violated or ``prev_migration`` ∉ ``[0,1)``.
    """
    census_time = T_admix + census_offset
    if not (T_admix < census_time < T_prev_admix < T_split):
        raise ValueError("require T_admix < T_admix+census_offset < T_prev_admix < T_split")
    if not (0.0 <= prev_migration < 1.0):
        raise ValueError("prev_migration must be in [0, 1)")
    d = msprime.Demography()
    d.add_population(name=SOURCE_A, initial_size=Ne, extra_metadata=_pop_role(state=0, source=True, reference=True))
    d.add_population(name=SOURCE_B, initial_size=Ne, extra_metadata=_pop_role(state=1, source=True, reference=True))
    d.add_population(name=ADMIXED, initial_size=Ne, extra_metadata=_pop_role(query=True))
    d.add_population(name=ANCESTRAL, initial_size=Ne)
    d.add_admixture(time=T_admix, derived=ADMIXED,
                    ancestral=[SOURCE_A, SOURCE_B], proportions=[f_A, 1.0 - f_A])
    d.add_census(time=census_time)
    d.add_mass_migration(time=T_prev_admix, source=SOURCE_A, dest=SOURCE_B, proportion=prev_migration)
    d.add_population_split(time=T_split, derived=[SOURCE_A, SOURCE_B], ancestral=ANCESTRAL)
    return d


def simulate_admixture_source_gene_flow(n_admix=10, n_ref=10, sequence_length=2e6,
                                        recombination_rate=1e-8, ploidy=2, random_seed=42,
                                        mutation_rate=None, **demography_kwargs):
    """Admixed queries + A/B references where the ``SOURCE_A`` panel carries real ``SOURCE_B``
    introgressed tracts from a prior A↔B pulse (:func:`admixture_demography_source_gene_flow`).

    The ``SOURCE_A`` references are the contaminated panel (nominally A, ``~prev_migration`` B);
    the ``SOURCE_B`` references are pure. Returns a :class:`Simulation` (``.ts`` / ``.queries`` /
    ``.labels`` / ``.truth_states`` / ``.sample_sets``); the query truth is the proximate A / B source.

    Parameters
    ----------
    n_admix : int, optional
        Number of admixed (query) individuals.
    n_ref : int, optional
        Number of individuals sampled from **each** of the two sources (A and B).
    sequence_length, recombination_rate, ploidy, mutation_rate, random_seed : optional
        As for :func:`simulate_admixture`.
    **demography_kwargs
        Passed to :func:`admixture_demography_source_gene_flow` (e.g. ``T_admix``, ``T_split``,
        ``T_prev_admix``, ``prev_migration``, ``f_A``, ``Ne``).

    Returns
    -------
    Simulation
        A :class:`~tspaint.sim.Simulation`; its default ``.labels`` panel pairs the contaminated
        ``SOURCE_A`` references (nominally A, carrying ~``prev_migration`` genuine B tracts) with
        the pure ``SOURCE_B`` references, and each query's ``.truth_states`` is its proximate A / B
        source.
    """
    demography = admixture_demography_source_gene_flow(**demography_kwargs)
    return simulate_admixture(demography, n_query=n_admix, n_reference=n_ref,
                              sequence_length=sequence_length, recombination_rate=recombination_rate,
                              ploidy=ploidy, mutation_rate=mutation_rate, random_seed=random_seed)


def admixture_demography_with_ref_proxies(Ne=10_000, T_admix=500.0, census_offset=1.0,
                                          T_split_A=50_000.0, T_split_B=30_000.0,
                                          T_split=150_000.0, f_A=0.5):
    """Two-source admixture with a deeply-divergent sister (**proxy**) of each source as a reference.

    The nested tree ``((A_prox, A), (B_prox, B))``: source ``A`` and its proxy ``A_prox`` split at
    ``T_split_A``, ``B`` and ``B_prox`` at ``T_split_B``, and the two sister pairs coalesce only at
    the deep ``T_split``. The admixed queries mix ``A`` and ``B`` at ``T_admix``. The point (CLAUDE.md
    §9–§10): paint the queries with the **proxies** ``A_prox`` / ``B_prox`` as references — a clean,
    well-sampled stand-in for a source that may be unsampled, ancient or contaminated. A query's ``A``
    tract coalesces with ``A_prox`` (its sister) at ``T_split_A`` but with ``B_prox`` only at the deep
    ``T_split``, so the proxy-vs-proxy affinity still separates ancestry — strongly when the internal
    branches (``T_split - T_split_A``, ``T_split - T_split_B``) are long relative to ``Ne`` (low ILS).
    The post-pulse census at ``T_admix + census_offset`` labels each query lineage's **proximate**
    source (A / B): the proxies split *deeper* than the census, so the query truth is unchanged and
    :func:`local_ancestry_truth` returns A / B.

    Parameters
    ----------
    Ne : float, optional
        Diploid effective size for every population.
    T_admix : float, optional
        Time (generations ago) of the admixture pulse forming ADMIX.
    census_offset : float, optional
        Offset added to ``T_admix`` for the census (kept strictly between the admixture and the most
        recent source/proxy split).
    T_split_A, T_split_B : float, optional
        Times at which ``A`` / ``B`` split from their proxies ``A_prox`` / ``B_prox`` — proxy
        closeness (smaller = a closer, more informative proxy; larger internal branch = stronger,
        lower-ILS separation).
    T_split : float, optional
        Deep split at which the two sister pairs coalesce into ANCESTRAL (the reference divergence
        the painting keys on).
    f_A : float, optional
        Fraction of the admixed queries contributed by source ``A`` (``B`` contributes ``1 - f_A``).

    Returns
    -------
    msprime.Demography
        Populations ``A_prox``, ``A``, ``ADMIX``, ``B``, ``B_prox``, ``A_ANC``, ``B_ANC`` and
        ``ANCESTRAL`` plus the admixture, census and three split events.

    Raises
    ------
    ValueError
        If ``T_admix < T_admix + census_offset < min(T_split_A, T_split_B)`` or
        ``max(T_split_A, T_split_B) < T_split`` is violated, or ``f_A`` is outside ``[0, 1]``.
    """
    census_time = T_admix + census_offset
    if not (T_admix < census_time < min(T_split_A, T_split_B)):
        raise ValueError("require T_admix < T_admix+census_offset < min(T_split_A, T_split_B)")
    if not (max(T_split_A, T_split_B) < T_split):
        raise ValueError("require max(T_split_A, T_split_B) < T_split")
    if not (0.0 <= f_A <= 1.0):
        raise ValueError("f_A must be in [0, 1]")
    d = msprime.Demography()
    # proxies are the DEFAULT reference panel; the true sources define the truth and are sampled
    # too (for the direct-source baseline) but are not in the default labels
    d.add_population(name=REF_A_PROXY, initial_size=Ne, extra_metadata=_pop_role(state=0, reference=True))
    d.add_population(name=SOURCE_A, initial_size=Ne, extra_metadata=_pop_role(state=0, source=True))
    d.add_population(name=ADMIXED, initial_size=Ne, extra_metadata=_pop_role(query=True))
    d.add_population(name=SOURCE_B, initial_size=Ne, extra_metadata=_pop_role(state=1, source=True))
    d.add_population(name=REF_B_PROXY, initial_size=Ne, extra_metadata=_pop_role(state=1, reference=True))
    d.add_population(name=A_ANCESTRAL, initial_size=Ne)
    d.add_population(name=B_ANCESTRAL, initial_size=Ne)
    d.add_population(name=ANCESTRAL, initial_size=Ne)
    d.add_admixture(time=T_admix, derived=ADMIXED,
                    ancestral=[SOURCE_A, SOURCE_B], proportions=[f_A, 1.0 - f_A])
    d.add_census(time=census_time)
    # each source splits from its proxy; the two sister pairs coalesce only at the deep root
    d.add_population_split(time=T_split_B, derived=[SOURCE_B, REF_B_PROXY], ancestral=B_ANCESTRAL)
    d.add_population_split(time=T_split_A, derived=[SOURCE_A, REF_A_PROXY], ancestral=A_ANCESTRAL)
    d.add_population_split(time=T_split, derived=[A_ANCESTRAL, B_ANCESTRAL], ancestral=ANCESTRAL)
    return d


def simulate_admixture_with_ref_proxies(n_admix=10, n_ref=10, sequence_length=2e6,
                                        recombination_rate=1e-8, ploidy=2, random_seed=42,
                                        mutation_rate=None, **demography_kwargs):
    """Admixed queries plus, for each source, a deeply-divergent **proxy** reference and the **true**
    source (:func:`admixture_demography_with_ref_proxies`).

    Samples ``n_admix`` admixed individuals (queries) and ``n_ref`` from each of the two proxies
    (``A_prox`` / ``B_prox`` = ``REF_A_PROXY`` / ``REF_B_PROXY``) **and** the two true sources
    (``A`` / ``B``) — so you can label the proxies as references and compare against the
    direct-``A``/``B`` baseline on the same data and the same census truth. Identify samples by node
    population.

    Parameters
    ----------
    n_admix : int, optional
        Number of admixed (query) individuals.
    n_ref : int, optional
        Number of individuals sampled from **each** of the four reference populations (``A_prox``,
        ``B_prox``, ``A``, ``B``).
    sequence_length, recombination_rate, ploidy, mutation_rate, random_seed : optional
        As for :func:`simulate_admixture`.
    **demography_kwargs
        Passed to :func:`admixture_demography_with_ref_proxies` (e.g. ``T_split_A``, ``T_split_B``,
        ``T_split``, ``T_admix``, ``Ne``, ``f_A``).

    Returns
    -------
    Simulation
        ``.ts`` / ``.queries`` / ``.labels`` / ``.truth_states`` / ``.sample_sets``. The default ``.labels``
        are the **proxies** (``A_prox`` / ``B_prox``); the true sources ``A`` / ``B`` are sampled too
        (in ``.sample_sets``) for a direct-source baseline, and the query ``.truth_states`` is A / B.

    See Also
    --------
    admixture_demography_with_ref_proxies : The underlying demography (proxy-closeness / root depth).
    """
    demography = admixture_demography_with_ref_proxies(**demography_kwargs)
    return simulate_admixture(demography, n_query=n_admix, n_reference=n_ref,
                              sequence_length=sequence_length, recombination_rate=recombination_rate,
                              ploidy=ploidy, mutation_rate=mutation_rate, random_seed=random_seed)


def admixture_demography_with_ghost(Ne=1000, T_admix=100.0, census_offset=1.0,
                                    T_split_AB=2000.0, T_split_ABC=20000.0, ghost_fraction=0.10):
    """Two-source admixture with a third **unsampled ("ghost") source** ``C``.

    Source ``C`` (``GHOST``) contributes a fraction ``ghost_fraction`` to the admixed
    population but is **not sampled as a reference** — it is the deeper outgroup: A and B
    coalesce at ``T_split_AB`` while their ancestor and ``C`` coalesce only at the deeper
    ``T_split_ABC``. A ghost tract therefore coalesces with the A/B panel only at that deep
    split — the deep-outlier signature ghost detection keys on. Setting ``T_split_ABC`` far
    above ``T_split_AB`` gives the **archaic-like** regime; ``ghost_fraction=0`` is the matched
    no-ghost control. The post-pulse census labels each lineage A / B / C per segment, so
    :func:`local_ancestry_truth` returns the ghost tracts as ground truth.

    Parameters
    ----------
    Ne : float, optional
        Diploid effective size for every population.
    T_admix : float, optional
        Time (generations ago) of the admixture pulse forming ADMIX.
    census_offset : float, optional
        Offset added to ``T_admix`` for the census (kept strictly between admixture and
        ``T_split_AB``).
    T_split_AB : float, optional
        Time at which A and B coalesce into their common ancestor.
    T_split_ABC : float, optional
        Time at which the A/B ancestor and the ghost ``C`` coalesce (the deep outgroup split).
    ghost_fraction : float, optional
        Fraction of ADMIX contributed by the ghost source ``C`` (A and B split the
        remainder equally). ``0`` reproduces a two-source (no-ghost) control.

    Returns
    -------
    msprime.Demography
        Demography with populations A, B, GHOST, ADMIX, AB and ANCESTRAL plus the admixture,
        census and two split events.

    Raises
    ------
    ValueError
        If ``T_admix < T_admix + census_offset < T_split_AB < T_split_ABC`` is violated or
        ``ghost_fraction`` is outside ``[0, 1)``.
    """
    census_time = T_admix + census_offset
    if not (T_admix < census_time < T_split_AB < T_split_ABC):
        raise ValueError("require T_admix < T_admix+census_offset < T_split_AB < T_split_ABC")
    if not (0.0 <= ghost_fraction < 1.0):
        raise ValueError("ghost_fraction must be in [0, 1)")

    f = (1.0 - ghost_fraction) / 2.0
    d = msprime.Demography()
    d.add_population(name=SOURCE_A, initial_size=Ne, extra_metadata=_pop_role(state=0, source=True, reference=True))
    d.add_population(name=SOURCE_B, initial_size=Ne, extra_metadata=_pop_role(state=1, source=True, reference=True))
    d.add_population(name=GHOST, initial_size=Ne, extra_metadata=_pop_role(ghost=True))  # unsampled foreign source; truth embedded
    d.add_population(name=ADMIXED, initial_size=Ne, extra_metadata=_pop_role(query=True))
    d.add_population(name=AB_ANCESTRAL, initial_size=Ne)
    d.add_population(name=ANCESTRAL, initial_size=Ne)
    d.add_admixture(time=T_admix, derived=ADMIXED, ancestral=[SOURCE_A, SOURCE_B, GHOST],
                    proportions=[f, f, ghost_fraction])
    d.add_census(time=census_time)
    d.add_population_split(time=T_split_AB, derived=[SOURCE_A, SOURCE_B], ancestral=AB_ANCESTRAL)
    d.add_population_split(time=T_split_ABC, derived=[AB_ANCESTRAL, GHOST], ancestral=ANCESTRAL)
    return d


def simulate_admixture_with_ghost(n_admix=10, n_ref=8, sequence_length=2e6,
                                  recombination_rate=1e-8, ploidy=2, random_seed=42,
                                  mutation_rate=None, ghost_fraction=0.10, **demography_kwargs):
    """Simulate admixed queries + A/B reference panels, with an **unsampled ghost source**.

    The admixed individuals carry tracts from A, B and the ghost ``C``; only A and B are
    sampled as references, so the ghost tracts are foreign-to-the-panel ground truth (labelled
    ``GHOST`` by the census). Use ``ghost_fraction=0`` for the matched no-ghost control.

    Parameters
    ----------
    n_admix : int, optional
        Number of admixed (query) individuals.
    n_ref : int, optional
        Number of individuals sampled from each of A and B (no ghost references).
    sequence_length : float, optional
        Sequence length in base pairs.
    recombination_rate : float, optional
        Per-base, per-generation recombination rate.
    ploidy : int, optional
        Ploidy; each individual yields ``ploidy`` sample haplotypes.
    random_seed : int, optional
        Seed for :func:`msprime.sim_ancestry`.
    mutation_rate : float, optional
        If given, overlay mutations on the ARG at this per-base rate so the returned ``.ts`` carries
        sites (:func:`tspaint.io.add_mutations`); ``None`` (default) leaves the bare ancestry. As
        for :func:`simulate_admixture`.
    ghost_fraction : float, optional
        Fraction of the admixed population contributed by the unsampled ghost source.
    **demography_kwargs
        Passed to :func:`admixture_demography_with_ghost` (e.g. ``T_admix``, ``T_split_AB``,
        ``T_split_ABC``, ``Ne``).

    Returns
    -------
    Simulation
        A :class:`Simulation` (use ``.ts`` for the tree sequence); samples are admixed queries and
        A/B references. The ``GHOST`` source is ghost-tagged, so its tracts are embedded in
        ``.truth_states`` (and surfaced in ``.ghost_states``); ``local_ancestry_truth`` still labels
        tracts A / B / GHOST by population.

    See Also
    --------
    admixture_demography_with_ghost : The underlying demography (and the archaic-like regime).
    """
    demography = admixture_demography_with_ghost(ghost_fraction=ghost_fraction, **demography_kwargs)
    return simulate_admixture(demography, n_query=n_admix, n_reference=n_ref,
                              sequence_length=sequence_length, recombination_rate=recombination_rate,
                              ploidy=ploidy, mutation_rate=mutation_rate, random_seed=random_seed)


def _census_mask(ts):
    flags = ts.tables.nodes.flags
    if _CENSUS_FLAG:
        mask = (flags & _CENSUS_FLAG) > 0
        if mask.any():
            return mask
    # Fallback: census nodes share a single internal-node time (the census time).
    times = ts.tables.nodes.time
    internal = ~np.isin(np.arange(ts.num_nodes), ts.samples())
    vals, counts = np.unique(np.round(times[internal], 6), return_counts=True)
    if counts.size == 0:
        return np.zeros(ts.num_nodes, bool)
    census_time = vals[np.argmax(counts)]
    return np.isclose(times, census_time) & internal


def local_ancestry_truth(ts):
    """True local ancestry per sample as source-population tracts.

    For each sample and each marginal tree, climb to the sample's census ancestor
    and read its population. Consecutive trees with the same source are merged.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence from :func:`simulate_admixture` (must carry census nodes).

    Returns
    -------
    tracts : dict[int, list[tuple[float, float, int]]]
        Sample node id -> list of ``(left, right, population_id)`` covering
        ``[0, sequence_length)`` with no gaps or overlaps. ``population_id == -1``
        marks a sample with no census ancestor (should not occur in this scenario).
    pop_name : dict[int, str]
        Population id -> name, for readability.
    """
    is_census = _census_mask(ts)
    pops = ts.tables.nodes.population

    pop_name = {}
    for p in range(ts.num_populations):
        try:
            pop_name[p] = ts.population(p).metadata.get("name", str(p))
        except Exception:
            pop_name[p] = str(p)

    samples = [int(s) for s in ts.samples()]
    tracts = {s: [] for s in samples}
    for tree in ts.trees():
        left, right = tree.interval.left, tree.interval.right
        for s in samples:
            u = s
            while u != tskit.NULL and not is_census[u]:
                u = tree.parent(u)
            pid = int(pops[u]) if u != tskit.NULL else -1
            lst = tracts[s]
            if lst and lst[-1][2] == pid and lst[-1][1] == left:
                lst[-1] = (lst[-1][0], right, pid)  # extend the current tract
            else:
                lst.append((left, right, pid))
    return tracts, pop_name
