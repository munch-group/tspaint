"""Head-to-head comparison harness (CLAUDE.md §9, §10).

A uniform way to score local-ancestry **painters** against the same simulated truth.
A painter is any ``f(ts, labels, queries) -> {sample: [Segment]}``; everything here
(and the :mod:`tslai.validate` metrics) then scores it identically, so tslai, simple
baselines, and external tools are compared on equal footing.

Provided painters:

* :func:`tslai_paint` — the full method (EM fit + down-pass posterior).
* :func:`nearest_reference_paint` — a runnable **ARG-native baseline**: paint each query
  by the label of its nearest labelled reference (smallest TMRCA) in each marginal tree.
  No CTMC, no EM, no credibility — the naive genealogy painter that the generative model
  should beat. A fair stand-in for the lower bound of "ARG-native LAI".

External comparators (RFMix/MOSAIC/FLARE; ARGMix, Pearson & Durbin) are not bundled — they
need separate installs/trained models. Wire one in by implementing the same painter
signature (e.g. shelling out to the tool over a VCF and parsing its calls into Segments)
and passing it to :func:`head_to_head`; it is then scored like the rest.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import tskit

from .em import fit, build_emissions
from .model import make_generator_2state
from .output import Segment, posterior_table, INFORMATIVE, MISSING_INFO
from .validate import balanced_accuracy, mean_confidence, per_base_accuracy

__all__ = ["tslai_paint", "nearest_reference_paint", "score_painter", "head_to_head"]


def tslai_paint(ts, labels, queries, K=2, max_iter=6, Q0=None, soft_refs=None):
    """The tslai painter: EM-fit ``(Q, π[, w])`` on the references, then paint queries."""
    Q0 = Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3)
    res = fit(ts, labels, K=K, Q0=Q0, max_iter=max_iter, soft_refs=soft_refs)
    emissions = build_emissions(ts, labels, res.w, res.pi)
    return posterior_table(ts, res.Q, res.pi, emissions, focal=queries)


def nearest_reference_paint(ts, labels, queries, K=2):
    """ARG-native baseline: per marginal tree, paint each query with the label of its
    nearest labelled reference (minimum TMRCA). One-hot posterior; isolated spans tagged
    missing-info."""
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
    """Run a painter on ``ts`` and score it against the truth."""
    tracks = painter(ts, labels, queries, **kwargs)
    return {
        "balanced_accuracy": balanced_accuracy(tracks, truth_states, samples=queries),
        "accuracy": per_base_accuracy(tracks, truth_states, samples=queries),
        "confidence": mean_confidence(tracks, samples=queries),
    }


def head_to_head(painters, *, T_admix=300.0, n_admix=12, n_ref=12, sequence_length=1e5,
                 recombination_rate=1e-8, Ne=1000, T_split=5000.0, f_A=0.5, seed=1,
                 substrates=("true",), mutation_rate=4e-7):
    """Score a panel of ``painters`` (dict name -> painter fn) on one admixture scenario,
    across ARG ``substrates`` ("true", "tsinfer"). Returns ``{substrate: {name: scores}}``.

    SINGER is intentionally left to :func:`tslai.experiments.singer_ensemble_experiment`
    (it samples an ensemble rather than a single ts); external tools slot in as painters.
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
