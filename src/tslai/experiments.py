"""End-to-end admixture experiments (CLAUDE.md §9) — Rung 8b/8c.

sim -> fit -> paint -> score against census truth. The headline demonstration that
soft tree-sequence local ancestry works, the accuracy-vs-admixture-age curve, and
the §7.3 breakpoint-flicker metric that decides whether loopy BP/EP (``bp/``) is
needed.
"""
from __future__ import annotations

import numpy as np

from .sim import simulate_admixture, local_ancestry_truth, SOURCE_A, SOURCE_B, ADMIXED
from .em import fit, build_emissions
from .model import make_generator_2state
from .output import posterior_table
from .validate import (map_truth, per_base_accuracy, reliability_curve,
                       breakpoint_flicker, tract_boundary_error)

__all__ = ["admixture_experiment", "flicker_vs_true_boundaries"]


def admixture_experiment(T_admix=30.0, n_admix=6, n_ref=8, sequence_length=2e5,
                         recombination_rate=1e-8, Ne=1000, T_split=5000.0, f_A=0.3,
                         seed=1, max_iter=8, Q0=None):
    """Simulate admixture, fit hard-clamp EM on the references, paint the admixed
    queries, and score against the census ground truth.

    Returns a dict with ``accuracy``, calibration ``reliability``, flicker summaries,
    fitted ``Q``/``pi``, and the raw ``tracks``/``truth_states`` for further analysis.
    """
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

    res = fit(ts, labels, K=2,
              Q0=Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3),
              max_iter=max_iter)
    emissions = build_emissions(ts, labels, res.w, res.pi)
    tracks = posterior_table(ts, res.Q, res.pi, emissions, focal=queries)

    truth, _ = local_ancestry_truth(ts)
    truth_states = map_truth({q: truth[q] for q in queries}, state_of_pop)

    acc = per_base_accuracy(tracks, truth_states, samples=queries)
    rel = reliability_curve(tracks, truth_states, state=0)
    flick = [breakpoint_flicker(tracks, q) for q in queries]
    boundary = [tract_boundary_error(tracks, truth_states, q) for q in queries]

    return {
        "T_admix": T_admix,
        "accuracy": acc,
        "reliability": rel,
        "mean_flicker": float(np.mean([f["mean_abs_diff"] for f in flick])),
        "mean_flip_rate": float(np.mean([f["flip_rate"] for f in flick])),
        "boundary_error": boundary,
        "Q": res.Q, "pi": res.pi, "n_queries": len(queries),
        "tracks": tracks, "truth_states": truth_states, "ts": ts,
    }


def flicker_vs_true_boundaries(tracks, truth_states, sample, state=0, eps=0.0):
    """§7.3 decision input: split a sample's segment-boundary flicker into boundaries
    that coincide with a *true* ancestry switch versus those that do not.

    Blocked EM is adequate (``bp/`` unnecessary) when flicker at non-true boundaries
    is small relative to the discontinuity at true boundaries.
    """
    true_switches = []
    prev = None
    for (l, r, st) in truth_states[int(sample)]:
        if prev is not None and st != prev:
            true_switches.append(l)
        prev = st

    segs = tracks[int(sample)]
    at_true, off_true = [], []
    for a, b in zip(segs, segs[1:]):
        d = abs(float(a.posterior[state]) - float(b.posterior[state]))
        boundary = a.right
        is_true = any(abs(boundary - ts_) <= eps for ts_ in true_switches)
        (at_true if is_true else off_true).append(d)

    return {
        "mean_flicker_off_true": float(np.mean(off_true)) if off_true else 0.0,
        "mean_flicker_at_true": float(np.mean(at_true)) if at_true else float("nan"),
        "n_off_true": len(off_true),
        "n_at_true": len(at_true),
    }
