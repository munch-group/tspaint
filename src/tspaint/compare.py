"""Head-to-head comparison harness (CLAUDE.md §9, §10).

A uniform way to score local-ancestry **painters** against the same simulated truth.
A painter is any ``f(ts, labels, queries) -> {sample: [Segment]}``; everything here
(and the :mod:`tspaint.validate` metrics) then scores it identically, so tspaint, simple
baselines, and external tools are compared on equal footing.

Provided painters:

* :func:`tspaint_paint` — the full method (EM fit + down-pass posterior).
* :func:`nearest_reference_paint` — a runnable **ARG-native baseline**: paint each query
  by the label of its nearest labelled reference (smallest TMRCA) in each marginal tree.
  No CTMC, no EM, no credibility — the naive genealogy painter that the generative model
  should beat. A fair stand-in for the lower bound of "ARG-native LAI".

**RFMix is wired** as a painter (:func:`tspaint.io_rfmix.rfmix_paint`) — the field-standard
segment incumbent, run from an isolated ``compare`` pixi env. Other external comparators
(MOSAIC/FLARE; ARGMix, Pearson & Durbin) are not bundled — they need separate installs /
trained models. Add one by implementing the same painter signature (shell out to the tool
over a VCF, parse its calls into Segments) and passing it to :func:`head_to_head`; it is
then scored like the rest.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import tskit

from .em import fit, build_emissions
from .model import make_generator_2state
from .output import Segment, posterior_table, INFORMATIVE, MISSING_INFO
from .validate import balanced_accuracy, mean_confidence, per_base_accuracy
from .io_rfmix import rfmix_paint   # genotype-native comparator, scored like the rest

__all__ = ["tspaint_paint", "nearest_reference_paint", "rfmix_paint", "score_painter",
           "head_to_head"]


def tspaint_paint(ts, labels, queries, K=2, max_iter=6, Q0=None, soft_refs=None,
                ranked=False, estimate_pi=False):
    """Paint queries with the full tspaint method (EM fit + down-pass posterior).

    EM-fits ``(Q[, π, w])`` on the labelled references, then paints the queries
    with the down-pass posterior.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence to paint (true or inferred ARG).
    labels : dict[int, int]
        Reference sample id -> ancestry-state index.
    queries : iterable[int]
        Sample ids to paint.
    K : int, optional
        Number of ancestry states.
    max_iter : int, optional
        Maximum number of EM iterations.
    Q0 : numpy.ndarray, optional
        Initial generator; defaults to ``make_generator_2state(1e-3, 1e-3)``.
    soft_refs : iterable[int], optional
        References whose credibility ``w`` is learned (the rest stay hard-clamped);
        passed through to :func:`tspaint.em.fit`.
    ranked : bool, optional
        If True, run the order-only variant on a dense-ranked tree sequence.
        **Not recommended** — it worsens the π degeneracy (CLAUDE.md §6).
    estimate_pi : bool, optional
        If False (default), hold π fixed (uniform), which is robust to the
        π-identifiability degeneracy that makes painting confidently wrong on
        sparse ARGs (CLAUDE.md §6). Set True to also estimate π (fine on good,
        long data).

    Returns
    -------
    dict[int, list[Segment]]
        Per-query posterior segment tracks (from
        :func:`tspaint.output.posterior_table`).
    """
    if ranked:
        from .ranked import ranked_tree_sequence
        ts = ranked_tree_sequence(ts)
    Q0 = Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3)
    res = fit(ts, labels, K=K, Q0=Q0, max_iter=max_iter, soft_refs=soft_refs,
              estimate_pi=estimate_pi)
    emissions = build_emissions(ts, labels, res.w, res.pi)
    return posterior_table(ts, res.Q, res.pi, emissions, focal=queries)


def nearest_reference_paint(ts, labels, queries, K=2):
    """Paint queries by their nearest labelled reference (ARG-native baseline).

    Per marginal tree, paint each query with the label of its nearest labelled
    reference (minimum TMRCA). Yields a one-hot posterior; spans where a query has
    no labelled reference are tagged missing-info. No CTMC, no EM, no credibility —
    the naive genealogy painter the generative model should beat.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence to paint.
    labels : dict[int, int]
        Reference sample id -> ancestry-state index.
    queries : iterable[int]
        Sample ids to paint.
    K : int, optional
        Number of ancestry states.

    Returns
    -------
    dict[int, list[Segment]]
        Per-query segment tracks with one-hot posteriors;
        :data:`tspaint.output.MISSING_INFO` marks spans with no reachable reference.
    """
    node_time = ts.tables.nodes.time
    ref_ids = [int(s) for s in labels]
    tracks = {int(q): [] for q in queries}
    for tree in ts.trees():
        left, right = tree.interval.left, tree.interval.right
        for q in queries:
            q = int(q)
            best_label, best_t = None, np.inf
            for r in ref_ids:
                m = tree.mrca(q, r)
                if m == tskit.NULL:
                    continue
                t = node_time[m]
                if t < best_t:
                    best_t, best_label = t, labels[r]
            post = np.zeros(K)
            if best_label is None:
                status = MISSING_INFO
                post[:] = 1.0 / K
            else:
                status = INFORMATIVE
                post[best_label] = 1.0
            segs = tracks[q]
            if (segs and segs[-1].right == left and segs[-1].status == status
                    and np.array_equal(segs[-1].posterior, post)):
                segs[-1].right = right
            else:
                segs.append(Segment(left, right, post, status))
    return tracks


def score_painter(painter, ts, labels, queries, truth_states, **kwargs):
    """Run a painter on ``ts`` and score it against the truth.

    Parameters
    ----------
    painter : callable
        A painter ``f(ts, labels, queries, **kwargs) -> {sample: [Segment]}``.
    ts : tskit.TreeSequence
        Tree sequence to paint.
    labels : dict[int, int]
        Reference sample id -> ancestry-state index.
    queries : iterable[int]
        Sample ids to paint and score.
    truth_states : dict[int, list[tuple[float, float, int]]]
        True ancestry-state tracts per sample (e.g. from
        :func:`tspaint.validate.map_truth`).
    **kwargs
        Forwarded to ``painter``.

    Returns
    -------
    dict
        Keys ``"balanced_accuracy"``, ``"accuracy"`` and ``"confidence"`` from
        the :mod:`tspaint.validate` metrics, scored over ``queries``.
    """
    tracks = painter(ts, labels, queries, **kwargs)
    return {
        "balanced_accuracy": balanced_accuracy(tracks, truth_states, samples=queries),
        "accuracy": per_base_accuracy(tracks, truth_states, samples=queries),
        "confidence": mean_confidence(tracks, samples=queries),
    }


def head_to_head(painters, *, T_admix=300.0, n_admix=12, n_ref=12, sequence_length=1e5,
                 recombination_rate=1e-8, Ne=1000, T_split=5000.0, f_A=0.5, seed=1,
                 substrates=("true",), mutation_rate=4e-7):
    """Score a panel of painters on one admixture scenario across ARG substrates.

    Simulates a single admixture scenario, then scores every painter on each
    requested ARG substrate.

    SINGER is intentionally left to
    :func:`tspaint.experiments.singer_ensemble_experiment` (it samples an ensemble
    rather than a single ts); external tools slot in as painters.

    Parameters
    ----------
    painters : dict[str, callable]
        Mapping name -> painter ``f(ts, labels, queries) -> {sample: [Segment]}``.
    T_admix : float, optional
        Time (generations ago) of the admixture pulse.
    n_admix : int, optional
        Number of admixed individuals (queries).
    n_ref : int, optional
        Number of individuals sampled from each reference source.
    sequence_length : float, optional
        Simulated sequence length in base pairs.
    recombination_rate : float, optional
        Per-base, per-generation recombination rate.
    Ne : float, optional
        Diploid effective population size.
    T_split : float, optional
        Time (generations ago) at which the two sources coalesce.
    f_A : float, optional
        Fraction of the admixed population contributed by source A.
    seed : int, optional
        Random seed for the simulation (and tsinfer mutation overlay).
    substrates : tuple[str, ...], optional
        ARG substrates to score on; each is ``"true"`` or ``"tsinfer"``.
    mutation_rate : float, optional
        Mutation rate used to overlay variants before tsinfer inference.

    Returns
    -------
    dict
        Nested ``{substrate: {painter_name: scores}}``, where ``scores`` is the
        dict returned by :func:`score_painter`.

    Raises
    ------
    ValueError
        If a substrate other than ``"true"`` or ``"tsinfer"`` is requested.
    """
    from .sim import simulate_admixture, local_ancestry_truth, SOURCE_A, SOURCE_B, ADMIXED
    from .validate import map_truth

    ts = simulate_admixture(n_admix=n_admix, n_ref=n_ref, sequence_length=sequence_length,
                            recombination_rate=recombination_rate, random_seed=seed,
                            Ne=Ne, T_admix=T_admix, T_split=T_split, f_A=f_A)
    node_pop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A_id = next(p for p, n in names.items() if n == SOURCE_A)
    B_id = next(p for p, n in names.items() if n == SOURCE_B)
    admix_id = next(p for p, n in names.items() if n == ADMIXED)
    state_of_pop = {A_id: 0, B_id: 1}
    labels = {int(s): state_of_pop[node_pop[s]]
              for s in ts.samples() if node_pop[s] in (A_id, B_id)}
    queries = [int(s) for s in ts.samples() if node_pop[s] == admix_id]
    truth, _ = local_ancestry_truth(ts)
    truth_states = map_truth({q: truth[q] for q in queries}, state_of_pop)

    out = {}
    for sub in substrates:
        if sub == "true":
            work = ts
        elif sub == "tsinfer":
            from .io_tsinfer import add_mutations, infer_tree_sequence
            work = infer_tree_sequence(add_mutations(ts, rate=mutation_rate, random_seed=seed))
        else:
            raise ValueError(f"unknown substrate {sub!r}")
        out[sub] = {name: score_painter(p, work, labels, queries, truth_states)
                    for name, p in painters.items()}
    return out
