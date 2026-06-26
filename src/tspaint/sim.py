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
    "admixture_demography",
    "simulate_admixture",
    "admixture_demography_impure_refs",
    "simulate_admixture_impure_refs",
    "local_ancestry_truth",
]

SOURCE_A = "A"
SOURCE_B = "B"
ADMIXED = "ADMIX"
ANCESTRAL = "ANC"
REF_A_IMPURE = "RA"   # impure reference panel: mostly A, a known minority B (CLAUDE.md §2.2, §6)
REF_B_IMPURE = "RB"   # impure reference panel: mostly B, a known minority A

# msprime flags census-event nodes; fall back to time-based detection if the
# constant is unavailable in the installed version.
_CENSUS_FLAG = getattr(msprime, "NODE_IS_CEN_EVENT", 0)


def admixture_demography(Ne=10_000, T_admix=30.0, census_offset=1.0,
                         T_split=2000.0, f_A=0.3):
    """Build a two-source admixture demography with a post-pulse census.

    Two sources (A, B) feed an admixed population, with a census placed just
    older than the admixture pulse. A and B contribute fractions ``f_A`` and
    ``1 - f_A`` to ADMIX at ``T_admix``; they merge into a common ancestor at
    ``T_split``. The census sits at ``T_admix + census_offset`` (strictly
    between admixture and split).

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

    Returns
    -------
    msprime.Demography
        Demography with populations A, B, ADMIX and ANCESTRAL plus the
        admixture, census and split events.

    Raises
    ------
    ValueError
        If ``T_admix < T_admix + census_offset < T_split`` is violated.
    """
    census_time = T_admix + census_offset
    if not (T_admix < census_time < T_split):
        raise ValueError("require T_admix < T_admix + census_offset < T_split")

    d = msprime.Demography()
    d.add_population(name=SOURCE_A, initial_size=Ne)
    d.add_population(name=SOURCE_B, initial_size=Ne)
    d.add_population(name=ADMIXED, initial_size=Ne)
    d.add_population(name=ANCESTRAL, initial_size=Ne)
    # Admixed pop forms (backward in time) as a mixture of the two sources.
    d.add_admixture(time=T_admix, derived=ADMIXED,
                    ancestral=[SOURCE_A, SOURCE_B], proportions=[f_A, 1.0 - f_A])
    # Census after the pulse: every lineage is now in A or B; census nodes label
    # each lineage's source per genomic segment (the local-ancestry truth).
    d.add_census(time=census_time)
    # Sources coalesce into a common ancestor deeper in time.
    d.add_population_split(time=T_split, derived=[SOURCE_A, SOURCE_B],
                           ancestral=ANCESTRAL)
    return d


def simulate_admixture(n_admix=10, n_ref=10, sequence_length=1e6,
                       recombination_rate=1e-8, ploidy=2, random_seed=42,
                       **demography_kwargs):
    """Simulate an admixed sample plus two reference panels.

    Parameters
    ----------
    n_admix : int, optional
        Number of admixed individuals (query haplotypes).
    n_ref : int, optional
        Number of individuals sampled from each reference source (A and B).
    sequence_length : float, optional
        Length of the simulated sequence in base pairs.
    recombination_rate : float, optional
        Per-base, per-generation recombination rate.
    ploidy : int, optional
        Ploidy; each individual yields ``ploidy`` sample haplotypes.
    random_seed : int, optional
        Seed for :func:`msprime.sim_ancestry`.
    **demography_kwargs
        Passed to :func:`admixture_demography` (e.g. ``T_admix``, ``T_split``,
        ``f_A``, ``Ne``).

    Returns
    -------
    tskit.TreeSequence
        Tree sequence whose sample nodes are haplotypes; the first
        ``ploidy * n_admix`` belong to the admixed population (queries), the
        rest to the two reference sources.

    See Also
    --------
    local_ancestry_truth : Recover the ground-truth ancestry tracts.

    Examples
    --------
    >>> ts = simulate_admixture(n_admix=5, n_ref=5, sequence_length=1e5,
    ...                         random_seed=1)
    >>> ts.num_samples
    30
    """
    demography = admixture_demography(**demography_kwargs)
    return msprime.sim_ancestry(
        samples={ADMIXED: n_admix, SOURCE_A: n_ref, SOURCE_B: n_ref},
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        ploidy=ploidy,
        random_seed=random_seed,
    )


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
    for name in (SOURCE_A, SOURCE_B, ADMIXED, REF_A_IMPURE, REF_B_IMPURE, ANCESTRAL):
        d.add_population(name=name, initial_size=Ne)
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
                                   ref_impurity=0.1, **demography_kwargs):
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
    ref_impurity : float, optional
        Minority foreign fraction of each impure reference panel (see
        :func:`admixture_demography_impure_refs`).
    **demography_kwargs
        Passed to :func:`admixture_demography_impure_refs` (e.g. ``T_admix``,
        ``T_split``, ``f_A``, ``Ne``).

    Returns
    -------
    tskit.TreeSequence
        Tree sequence whose sample nodes are haplotypes from the admixed, pure-source
        and impure-reference populations; identify them by node population (as the
        experiments do) rather than by index.

    See Also
    --------
    admixture_demography_impure_refs : The underlying demography.
    local_ancestry_truth : Recover the ground-truth ancestry tracts (references included).
    """
    demography = admixture_demography_impure_refs(ref_impurity=ref_impurity,
                                                  **demography_kwargs)
    return msprime.sim_ancestry(
        samples={ADMIXED: n_admix, SOURCE_A: n_pure, SOURCE_B: n_pure,
                 REF_A_IMPURE: n_impure, REF_B_IMPURE: n_impure},
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        ploidy=ploidy,
        random_seed=random_seed,
    )


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
