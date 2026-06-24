"""End-to-end admixture experiments (CLAUDE.md §9) — Rung 8b/8c.

sim -> fit -> paint -> score against census truth. The headline demonstration that
soft tree-sequence local ancestry works, the accuracy-vs-admixture-age curve, and
the §7.3 breakpoint-flicker metric that decides whether loopy BP/EP (``bp/``) is
needed.
"""
from __future__ import annotations

import time

import numpy as np

from .sim import simulate_admixture, local_ancestry_truth, SOURCE_A, SOURCE_B, ADMIXED
from .em import fit, build_emissions
from .model import make_generator_2state
from .output import posterior_table
from .ensemble import merge_posterior_tables
from .validate import (map_truth, per_base_accuracy, balanced_accuracy,
                       mean_confidence, reliability_curve, breakpoint_flicker,
                       tract_boundary_error)

__all__ = ["admixture_experiment", "flicker_vs_true_boundaries", "age_sweep",
           "scaling_sweep", "arg_ensemble_experiment", "singer_ensemble_experiment"]


def admixture_experiment(T_admix=30.0, n_admix=6, n_ref=8, sequence_length=2e5,
                         recombination_rate=1e-8, Ne=1000, T_split=5000.0, f_A=0.3,
                         seed=1, max_iter=8, Q0=None, infer=False, mutation_rate=5e-8):
    """Simulate admixture, fit hard-clamp EM on the references, paint the admixed
    queries, and score against the census ground truth.

    Returns a dict with ``accuracy``, calibration ``reliability``, flicker summaries,
    fitted ``Q``/``pi``, and the raw ``tracks``/``truth_states`` for further analysis.
    """
    clk = time.perf_counter
    t0 = clk()
    ts = simulate_admixture(n_admix=n_admix, n_ref=n_ref, sequence_length=sequence_length,
                            recombination_rate=recombination_rate, random_seed=seed,
                            Ne=Ne, T_admix=T_admix, T_split=T_split, f_A=f_A)
    t_sim = clk() - t0
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

    # Substrate: the true ARG, or a tsinfer-inferred ARG. tsinfer preserves sample
    # ids, so labels/truth transfer by id; the inferred path is the §9 binding
    # constraint (does tree-native LAI survive tree-inference error?).
    n_sites = None
    t_infer = 0.0
    if infer:
        from .io_tsinfer import add_mutations, infer_tree_sequence
        t0 = clk()
        ts_mut = add_mutations(ts, rate=mutation_rate, random_seed=seed)
        work_ts = infer_tree_sequence(ts_mut)
        t_infer = clk() - t0
        n_sites = int(ts_mut.num_sites)
    else:
        work_ts = ts

    t0 = clk()
    res = fit(work_ts, labels, K=2,
              Q0=Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3),
              max_iter=max_iter)
    t_fit = clk() - t0

    t0 = clk()
    emissions = build_emissions(work_ts, labels, res.w, res.pi)
    tracks = posterior_table(work_ts, res.Q, res.pi, emissions, focal=queries)
    t_paint = clk() - t0

    acc = per_base_accuracy(tracks, truth_states, samples=queries)
    bal = balanced_accuracy(tracks, truth_states, samples=queries)
    conf = mean_confidence(tracks, samples=queries)
    rel = reliability_curve(tracks, truth_states, state=0)
    flick = [breakpoint_flicker(tracks, q) for q in queries]
    off_true = [flicker_vs_true_boundaries(tracks, truth_states, q)["mean_flicker_off_true"]
                for q in queries]
    boundary = [tract_boundary_error(tracks, truth_states, q) for q in queries]

    return {
        "T_admix": T_admix,
        "inferred": infer,
        "n_sites": n_sites,
        "n_haplotypes": int(work_ts.num_samples),
        "n_trees": int(work_ts.num_trees),
        "n_iter": len(res.loglik_history),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "confidence": conf,
        "reliability": rel,
        "mean_flicker": float(np.mean([f["mean_abs_diff"] for f in flick])),
        "mean_flicker_off_true": float(np.mean(off_true)),
        "mean_flip_rate": float(np.mean([f["flip_rate"] for f in flick])),
        "boundary_error": boundary,
        "timings": {"sim": t_sim, "infer": t_infer, "fit": t_fit, "paint": t_paint},
        "Q": res.Q, "pi": res.pi, "n_queries": len(queries),
        "tracks": tracks, "truth_states": truth_states, "ts": ts, "work_ts": work_ts,
    }


def age_sweep(ages, infer=False, **kwargs):
    """Accuracy vs. admixture age — the §9 headline. For each ``T_admix`` returns
    accuracy plus the mean number of *true* ancestry switches per query (so one can
    see tracts shortening as admixture ages). ``infer=True`` runs on tsinfer-inferred
    ARGs (accuracy then bounded by ARG quality)."""
    out = []
    for T in ages:
        r = admixture_experiment(T_admix=T, infer=infer, **kwargs)
        switches = [sum(1 for a, b in zip(segs, segs[1:]) if a[2] != b[2])
                    for segs in r["truth_states"].values()]
        out.append({
            "T_admix": T,
            "accuracy": r["accuracy"],
            "balanced_accuracy": r["balanced_accuracy"],
            "confidence": r["confidence"],
            "mean_true_switches": float(np.mean(switches)) if switches else 0.0,
            "n_sites": r["n_sites"],
            "inferred": infer,
        })
    return out


def scaling_sweep(admix_sizes, infer=False, **kwargs):
    """Runtime + correctness + flicker vs sample size. Sweeps ``n_admix`` (admixed
    *individuals*; haplotypes = ploidy x (n_admix + 2*n_ref)). Returns per-size dicts
    with haplotype/tree counts, per-iteration and total fit time, tsinfer time, balanced
    accuracy, confidence and off-true flicker — the data behind the scaling/regime plots."""
    out = []
    for n in admix_sizes:
        r = admixture_experiment(n_admix=n, infer=infer, **kwargs)
        out.append({
            "n_admix": n,
            "n_haplotypes": r["n_haplotypes"],
            "n_trees": r["n_trees"],
            "n_iter": r["n_iter"],
            "t_fit": r["timings"]["fit"],
            "t_per_iter": r["timings"]["fit"] / max(1, r["n_iter"]),
            "t_infer": r["timings"]["infer"],
            "balanced_accuracy": r["balanced_accuracy"],
            "confidence": r["confidence"],
            "mean_flicker_off_true": r["mean_flicker_off_true"],
            "n_sites": r["n_sites"],
            "inferred": infer,
        })
    return out


def arg_ensemble_experiment(M=8, T_admix=300.0, n_admix=20, n_ref=20,
                            sequence_length=3e5, recombination_rate=1e-8, Ne=1000,
                            T_split=5000.0, f_A=0.5, mutation_rate=4e-7, seed=1,
                            max_iter=6, Q0=None):
    """Merge LAI across an ensemble of inferred ARGs vs. a single inferred ARG.

    Stand-in for SINGER's thinned posterior samples: ``M`` independent mutation
    overlays of one true genealogy, each inferred with tsinfer (sharing samples and
    coordinates). ``theta`` is fit pooled across the ensemble (``fit`` over the list);
    each member is painted (:func:`tslai.output.posterior_table`) and the paintings are
    averaged (:func:`tslai.ensemble.merge_posterior_tables`). Reports single-member vs.
    merged vs. true-ARG balanced accuracy and the merged calibration.

    Caveat: a tsinfer ensemble captures data/inference variance but shares tsinfer's
    bias; true posterior samples (SINGER) additionally marginalise coalescent-time and
    topology uncertainty, so the realised gain here is a lower bound on the SINGER case.
    """
    from .io_tsinfer import add_mutations, infer_tree_sequence

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

    # M inferred ARGs (stand-in posterior samples): independent mutation overlays
    ensemble = [infer_tree_sequence(add_mutations(ts, rate=mutation_rate,
                                                  random_seed=seed + 1 + m))
                for m in range(M)]

    Q0 = Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3)
    res = fit(ensemble, [labels] * M, K=2, Q0=Q0, max_iter=max_iter)   # theta pooled

    tables, single_bal = [], []
    for g in ensemble:
        em = build_emissions(g, labels, res.w, res.pi)
        tab = posterior_table(g, res.Q, res.pi, em, focal=queries)
        tables.append(tab)
        single_bal.append(balanced_accuracy(tab, truth_states, samples=queries))

    merged = merge_posterior_tables(tables, samples=queries)
    merged_bal = balanced_accuracy(merged, truth_states, samples=queries)

    res_true = fit(ts, labels, K=2, Q0=Q0, max_iter=max_iter)          # true-ARG ceiling
    em_true = build_emissions(ts, labels, res_true.w, res_true.pi)
    tab_true = posterior_table(ts, res_true.Q, res_true.pi, em_true, focal=queries)
    true_bal = balanced_accuracy(tab_true, truth_states, samples=queries)

    return {
        "M": M,
        "n_haplotypes": int(ts.num_samples),
        "single_balanced_mean": float(np.mean(single_bal)),
        "single_balanced_std": float(np.std(single_bal)),
        "single_balanced_per_member": single_bal,
        "merged_balanced": merged_bal,
        "merged_confidence": mean_confidence(merged, samples=queries),
        "merged_reliability": reliability_curve(merged, truth_states, state=0),
        "true_balanced": true_bal,
        "merged_tracks": merged, "truth_states": truth_states, "queries": queries,
    }


def singer_ensemble_experiment(T_admix=300.0, n_admix=12, n_ref=12, sequence_length=8e4,
                               recombination_rate=1e-8, Ne=1000, T_split=5000.0, f_A=0.5,
                               mutation_rate=2.5e-7, seed=1, max_iter=6, n_singer=20,
                               thin=8, burn_in=6, singer_seed=42, Q0=None):
    """Merge LAI across **SINGER posterior** ARG samples vs. a single sample vs. the true
    ARG. The §7.4 test: genuine posterior draws (thinned -> independent-ish errors) should
    make merging help, unlike the correlated tsinfer ensemble (arg_ensemble_experiment).
    Requires the SINGER binary (env TSLAI_SINGER or io_singer's default path)."""
    import msprime
    from .io_singer import singer_tree_sequences

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

    tsm = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed,
                                model=msprime.BinaryMutationModel())
    ensemble = singer_tree_sequences(tsm, Ne=Ne, mutation_rate=mutation_rate,
                                     recombination_rate=recombination_rate, n_samples=n_singer,
                                     thin=thin, burn_in=burn_in, seed=singer_seed)

    Q0 = Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3)
    res = fit(ensemble, [labels] * len(ensemble), K=2, Q0=Q0, max_iter=max_iter)
    tables, single_bal = [], []
    for g in ensemble:
        em = build_emissions(g, labels, res.w, res.pi)
        tab = posterior_table(g, res.Q, res.pi, em, focal=queries)
        tables.append(tab)
        single_bal.append(balanced_accuracy(tab, truth_states, samples=queries))
    merged = merge_posterior_tables(tables, samples=queries)
    merged_bal = balanced_accuracy(merged, truth_states, samples=queries)

    res_true = fit(ts, labels, K=2, Q0=Q0, max_iter=max_iter)
    em_true = build_emissions(ts, labels, res_true.w, res_true.pi)
    tab_true = posterior_table(ts, res_true.Q, res_true.pi, em_true, focal=queries)
    true_bal = balanced_accuracy(tab_true, truth_states, samples=queries)

    return {
        "M": len(ensemble),
        "n_haplotypes": int(ts.num_samples),
        "n_sites": int(tsm.num_sites),
        "single_balanced_mean": float(np.mean(single_bal)),
        "single_balanced_std": float(np.std(single_bal)),
        "single_balanced_per_member": single_bal,
        "merged_balanced": merged_bal,
        "merged_confidence": mean_confidence(merged, samples=queries),
        "merged_reliability": reliability_curve(merged, truth_states, state=0),
        "true_balanced": true_bal,
        "merged_tracks": merged, "truth_states": truth_states, "queries": queries,
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
