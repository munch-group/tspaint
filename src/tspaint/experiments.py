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
from .output import posterior_table, hard_segments
from .ensemble import merge_posterior_tables
from .validate import (map_truth, per_base_accuracy, balanced_accuracy,
                       mean_confidence, reliability_curve, breakpoint_flicker,
                       tract_boundary_error, breakpoint_precision_recall, switch_density)

__all__ = ["admixture_experiment", "flicker_vs_true_boundaries", "age_sweep",
           "scaling_sweep", "arg_ensemble_experiment", "singer_ensemble_experiment",
           "fragmentation_experiment"]


def admixture_experiment(T_admix=30.0, n_admix=6, n_ref=8, sequence_length=2e5,
                         recombination_rate=1e-8, Ne=1000, T_split=5000.0, f_A=0.3,
                         seed=1, max_iter=8, Q0=None, infer=False, mutation_rate=5e-8):
    """Simulate admixture, fit hard-clamp EM on the references, paint the queries, score.

    The headline demonstration (CLAUDE.md §9): sim -> fit -> paint -> score against the
    census truth. With ``infer=True`` the painting runs on a tsinfer-inferred ARG (the
    §9 binding constraint: does tree-native LAI survive tree-inference error?), else on
    the true ARG.

    Parameters
    ----------
    T_admix : float, optional
        Admixture time in generations.
    n_admix, n_ref : int, optional
        Number of admixed-query and per-source-reference individuals.
    sequence_length, recombination_rate, Ne, T_split, f_A : optional
        msprime admixture-scenario parameters (sequence length, recombination rate,
        effective size, source-split time, admixture fraction of source A).
    seed : int, optional
        Random seed.
    max_iter : int, optional
        Maximum EM iterations.
    Q0 : numpy.ndarray, optional
        Initial generator (defaults to a symmetric ``1e-3`` 2-state generator).
    infer : bool, optional
        If True paint on a tsinfer-inferred ARG; else on the true ARG.
    mutation_rate : float, optional
        Mutation rate used to overlay sites before tsinfer (only when ``infer=True``).

    Returns
    -------
    dict
        Includes ``accuracy``, ``balanced_accuracy``, ``confidence``, calibration
        ``reliability``, flicker summaries (``mean_flicker``,
        ``mean_flicker_off_true``, ``mean_flip_rate``), ``boundary_error``, fitted
        ``Q``/``pi``, ``timings``, and the raw ``tracks``/``truth_states`` (plus ``ts``
        and ``work_ts``) for further analysis.
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


def _seg_metrics(seg_by_node, true_by_node, nodes, length, tol):
    """Aggregate fragmentation metrics (switches/Mb, precision, recall, median tract)
    over a set of haplotypes given hard segment lists."""
    precs, recs, lens, n_sw = [], [], [], 0
    for q in nodes:
        pr = breakpoint_precision_recall(seg_by_node[q], true_by_node[q], tol)
        if not np.isnan(pr["precision"]):
            precs.append(pr["precision"])
        if not np.isnan(pr["recall"]):
            recs.append(pr["recall"])
        n_sw += pr["n_inferred"]
        lens += [r - l for (l, r, _s) in seg_by_node[q]]
    mb = (length / 1e6) * len(nodes)
    return {"switches_per_mb": n_sw / mb if mb else float("nan"),
            "precision": float(np.mean(precs)) if precs else float("nan"),
            "recall": float(np.mean(recs)) if recs else float("nan"),
            "median_tract": float(np.median(lens)) if lens else float("nan")}


def fragmentation_experiment(*, n_admix=10, n_ref=10, sequence_length=5e6, T_admix=200.0,
                             Ne=1000, T_split=5000.0, f_A=0.5, recombination_rate=1e-8,
                             mutation_rate=4e-7, deadband=0.4, tol=1e5, seed=1,
                             include_rfmix=True):
    """Fragmentation / tract-length fidelity of hard segmentations vs. the true tracts.

    Matters because downstream admixture-pulse dating reads the segment-length
    distribution: ``ratio`` (inferred / true switch density) ≠ 1 biases the inferred
    pulse (>1 older from fragmentation, <1 younger from over-smoothing; CLAUDE.md §9).
    On one simulated admixture (true ARG) compares tspaint ``argmax``, tspaint with a
    confidence ``deadband``, ``nearest_reference``, and — if the rfmix binary is present
    — RFMix native (``.msp`` Viterbi).

    Parameters
    ----------
    n_admix, n_ref : int, optional
        Number of admixed-query and per-source-reference individuals.
    sequence_length, T_admix, Ne, T_split, f_A, recombination_rate : optional
        msprime admixture-scenario parameters.
    mutation_rate : float, optional
        Mutation rate for the sites RFMix needs.
    deadband : float, optional
        Confidence deadband for the ``tspaint_deadband`` segmentation.
    tol : float, optional
        Breakpoint-matching tolerance (bp) for precision/recall.
    seed : int, optional
        Random seed.
    include_rfmix : bool, optional
        If True (and the rfmix binary is present) also score RFMix native ``.msp``.

    Returns
    -------
    dict
        ``{"seed", "T_admix", "true_switches_per_mb", "n_queries", "methods"}`` where
        ``methods`` maps each method name to its ``_seg_metrics`` dict augmented with a
        ``ratio`` (inferred / true switch density).
    """
    from .compare import tspaint_paint, nearest_reference_paint

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
    true_segs = map_truth({q: truth[q] for q in queries}, state_of_pop)
    L = ts.sequence_length

    true_sw = sum(sum(1 for k in range(1, len(true_segs[q]))
                      if true_segs[q][k][2] != true_segs[q][k - 1][2]) for q in queries)
    true_density = true_sw / ((L / 1e6) * len(queries))

    methods = {}
    soft = tspaint_paint(ts, labels, queries)
    methods["tspaint_argmax"] = _seg_metrics(
        {q: hard_segments(soft[q], 0.0) for q in queries}, true_segs, queries, L, tol)
    methods[f"tspaint_deadband_{deadband}"] = _seg_metrics(
        {q: hard_segments(soft[q], deadband) for q in queries}, true_segs, queries, L, tol)
    nr = nearest_reference_paint(ts, labels, queries)
    methods["nearest_ref"] = _seg_metrics(
        {q: hard_segments(nr[q], 0.0) for q in queries}, true_segs, queries, L, tol)

    if include_rfmix:
        import os
        from .io_rfmix import (run_rfmix, _parse_msp, _classify_individuals, _ensure_sites,
                               DEFAULT_RFMIX)
        if os.path.exists(DEFAULT_RFMIX):
            ts_mut = _ensure_sites(ts, mutation_rate, seed)
            qi, _ = _classify_individuals(ts_mut, labels, queries)
            out = run_rfmix(ts_mut, labels, queries, recombination_rate=recombination_rate,
                            generations=T_admix)
            methods["rfmix_msp"] = _seg_metrics(_parse_msp(out + ".msp.tsv", qi, L),
                                                true_segs, queries, L, tol)

    for m in methods.values():
        m["ratio"] = m["switches_per_mb"] / true_density if true_density else float("nan")
    return {"seed": seed, "T_admix": T_admix, "true_switches_per_mb": true_density,
            "n_queries": len(queries), "methods": methods}


def age_sweep(ages, infer=False, **kwargs):
    """Accuracy vs. admixture age — the §9 headline.

    For each ``T_admix`` reports accuracy plus the mean number of *true* ancestry
    switches per query (so one can see tracts shortening as admixture ages).

    Parameters
    ----------
    ages : iterable of float
        Admixture times (generations) to sweep.
    infer : bool, optional
        If True run on tsinfer-inferred ARGs (accuracy then bounded by ARG quality).
    **kwargs
        Forwarded to :func:`admixture_experiment`.

    Returns
    -------
    list of dict
        One dict per age with ``T_admix``, ``accuracy``, ``balanced_accuracy``,
        ``confidence``, ``mean_true_switches``, ``n_sites`` and ``inferred``.
    """
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
    """Runtime + correctness + flicker vs. sample size.

    Sweeps ``n_admix`` (admixed *individuals*; haplotypes = ploidy x
    (n_admix + 2*n_ref)) — the data behind the scaling/regime plots.

    Parameters
    ----------
    admix_sizes : iterable of int
        Values of ``n_admix`` to sweep.
    infer : bool, optional
        If True run on tsinfer-inferred ARGs.
    **kwargs
        Forwarded to :func:`admixture_experiment`.

    Returns
    -------
    list of dict
        One dict per size with ``n_admix``, ``n_haplotypes``, ``n_trees``, ``n_iter``,
        total fit time ``t_fit``, per-iteration ``t_per_iter``, tsinfer time
        ``t_infer``, ``balanced_accuracy``, ``confidence``, ``mean_flicker_off_true``,
        ``n_sites`` and ``inferred``.
    """
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
    each member is painted (:func:`tspaint.output.posterior_table`) and the paintings are
    averaged (:func:`tspaint.ensemble.merge_posterior_tables`).

    Parameters
    ----------
    M : int, optional
        Ensemble size (number of independent mutation overlays / inferred ARGs).
    T_admix, n_admix, n_ref, sequence_length, recombination_rate, Ne, T_split, f_A : optional
        msprime admixture-scenario parameters.
    mutation_rate : float, optional
        Mutation rate for the per-member site overlays.
    seed : int, optional
        Base random seed (members use ``seed + 1 + m``).
    max_iter : int, optional
        Maximum EM iterations.
    Q0 : numpy.ndarray, optional
        Initial generator (defaults to a symmetric ``1e-3`` 2-state generator).

    Returns
    -------
    dict
        ``M``, ``n_haplotypes``, ``single_balanced_mean``/``_std``/``_per_member``,
        ``merged_balanced``, ``merged_confidence``, ``merged_reliability``,
        ``true_balanced`` (true-ARG ceiling), and the ``merged_tracks``,
        ``truth_states``, ``queries``.

    Notes
    -----
    A tsinfer ensemble captures data/inference variance but shares tsinfer's bias; true
    posterior samples (SINGER) additionally marginalise coalescent-time and topology
    uncertainty, so the realised gain here is a lower bound on the SINGER case.
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
    """Merge LAI across **SINGER posterior** ARG samples vs. a single sample vs. truth.

    The §7.4 test: genuine posterior draws (thinned -> independent-ish errors) should
    make merging help, unlike the correlated tsinfer ensemble
    (:func:`arg_ensemble_experiment`). Requires the SINGER binary (env ``TSPAINT_SINGER``
    or :mod:`tspaint.io_singer`'s default path).

    Parameters
    ----------
    T_admix, n_admix, n_ref, sequence_length, recombination_rate, Ne, T_split, f_A : optional
        msprime admixture-scenario parameters.
    mutation_rate : float, optional
        Mutation rate for the sites SINGER consumes.
    seed : int, optional
        Random seed for simulation and mutations.
    max_iter : int, optional
        Maximum EM iterations.
    n_singer : int, optional
        Number of SINGER posterior samples to draw.
    thin, burn_in : int, optional
        SINGER MCMC thinning interval and burn-in.
    singer_seed : int, optional
        Seed passed to the SINGER binary.
    Q0 : numpy.ndarray, optional
        Initial generator (defaults to a symmetric ``1e-3`` 2-state generator).

    Returns
    -------
    dict
        ``M`` (realised number of samples), ``n_haplotypes``, ``n_sites``,
        ``single_balanced_mean``/``_std``/``_per_member``, ``merged_balanced``,
        ``merged_confidence``, ``merged_reliability``, ``true_balanced``, and the
        ``merged_tracks``, ``truth_states``, ``queries``.
    """
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
    """Split a sample's segment-boundary flicker into at-true vs. off-true boundaries.

    The §7.3 decision input: blocked EM is adequate (``bp/`` unnecessary) when flicker
    at non-true boundaries is small relative to the discontinuity at true boundaries.

    Parameters
    ----------
    tracks : dict
        Per-sample painting (sample id -> list of segments with a ``.posterior``).
    truth_states : dict
        Per-sample census-truth segments (sample id -> list of ``(left, right, state)``).
    sample : int
        Sample id to score.
    state : int, optional
        Ancestry state whose posterior discontinuity is measured (default 0).
    eps : float, optional
        Tolerance (bp) for matching a segment boundary to a true switch.

    Returns
    -------
    dict
        ``mean_flicker_off_true``, ``mean_flicker_at_true`` (mean ``|ΔP(state)|`` at
        each kind of boundary) and the counts ``n_off_true``, ``n_at_true``.
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
