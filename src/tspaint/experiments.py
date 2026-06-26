"""End-to-end admixture experiments (CLAUDE.md §9) — Rung 8b/8c.

sim -> fit -> paint -> score against census truth. The headline demonstration that
soft tree-sequence local ancestry works, the accuracy-vs-admixture-age curve, and
the §7.3 breakpoint-flicker metric that decides whether loopy BP/EP (``bp/``) is
needed.
"""
from __future__ import annotations

import time

import numpy as np

from .sim import (simulate_admixture, simulate_admixture_impure_refs, local_ancestry_truth,
                  SOURCE_A, SOURCE_B, ADMIXED, REF_A_IMPURE, REF_B_IMPURE)
from .em import fit, build_emissions
from .model import make_generator_2state
from .output import posterior_table, loo_posterior_table, hard_segments, MISSING_INFO
from .ensemble import merge_posterior_tables
from .validate import (map_truth, per_base_accuracy, balanced_accuracy,
                       mean_confidence, reliability_curve, breakpoint_flicker,
                       tract_boundary_error, breakpoint_precision_recall, switch_density,
                       _walk_overlap)

__all__ = ["admixture_experiment", "flicker_vs_true_boundaries", "age_sweep",
           "scaling_sweep", "arg_ensemble_experiment", "singer_ensemble_experiment",
           "fragmentation_experiment", "impure_reference_experiment",
           "impure_reference_sweep", "archaic_detection_experiment"]


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
        from .io_tsinfer import add_mutations, tsinfer
        t0 = clk()
        ts_mut = add_mutations(ts, rate=mutation_rate, random_seed=seed)
        work_ts = tsinfer(ts_mut)
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
    from .io_tsinfer import add_mutations, tsinfer

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
    ensemble = [tsinfer(add_mutations(ts, rate=mutation_rate,
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
    from .io_singer import singer

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
    ensemble = singer(tsm, Ne=Ne, mutation_rate=mutation_rate,
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


def _ref_purity(truth_segs, label, length):
    """Span fraction of a reference's census truth sitting in its nominal (label) source."""
    own = sum((r - l) for (l, r, st) in truth_segs if st == label)
    return own / length if length > 0 else float("nan")


def _foreign_recall(track, truth_segs, label):
    """Span-weighted recall of a reference's FOREIGN (non-label) ancestry tracts.

    Identically zero under a hard clamp — the tip posterior is pinned one-hot to its
    label, so it can never call the foreign state — and positive only when a soft
    emission lets the genealogy override the label over a genuinely foreign tract (the
    §2.3 introgression map, quantified). ``nan`` if the reference has no foreign span.
    """
    foreign = 1 - int(label)
    tot = hit = 0.0
    for lo, hi, seg, tstate in _walk_overlap(track, truth_segs):
        if seg.status == MISSING_INFO or tstate != foreign:
            continue
        w = hi - lo
        tot += w
        if int(np.argmax(seg.posterior)) == foreign:
            hit += w
    return hit / tot if tot > 0 else float("nan")


def _impure_config(ts, labels, queries, impure_refs, query_truth, ref_truth, length, *,
                   soft_refs, alpha, beta, priors, Q0, max_iter):
    """Fit one credibility configuration; score queries + impure-ref introgression recovery."""
    res = fit(ts, labels, K=2, Q0=Q0, max_iter=max_iter, soft_refs=soft_refs,
              alpha=alpha, beta=beta, priors=priors)
    emissions = build_emissions(ts, labels, res.w, res.pi)
    tracks = posterior_table(ts, res.Q, res.pi, emissions,
                             focal=list(queries) + list(impure_refs))
    loo = loo_posterior_table(ts, res.Q, res.pi, emissions, focal=list(impure_refs))
    self_bal = [balanced_accuracy(tracks, ref_truth, samples=[r]) for r in impure_refs]
    foreign = [_foreign_recall(tracks[r], ref_truth[r], labels[r]) for r in impure_refs]
    foreign_loo = [_foreign_recall(loo[r], ref_truth[r], labels[r]) for r in impure_refs]
    learned = {int(r): float(res.w.get(r, 1.0)) for r in impure_refs}
    return {
        "query_balanced_accuracy": balanced_accuracy(tracks, query_truth, samples=queries),
        "query_confidence": mean_confidence(tracks, samples=queries),
        "impure_self_balanced_accuracy": float(np.nanmean(self_bal)),
        "impure_self_foreign_recall": float(np.nanmean(foreign)),
        "impure_self_foreign_recall_loo": float(np.nanmean(foreign_loo)),
        "mean_learned_w": float(np.mean(list(learned.values()))),
        "learned_w_per_ref": learned,
        "Q": res.Q, "pi": res.pi,
    }


def impure_reference_experiment(*, T_admix=300.0, n_admix=10, n_pure=6, n_impure=8,
                                sequence_length=5e6, recombination_rate=1e-8, Ne=1000,
                                T_split=5000.0, f_A=0.5, ref_impurity=0.15, seed=1,
                                max_iter=10, alpha=20.0, beta=1.0,
                                alpha_grid=(2.0, 20.0, 200.0, 2000.0), Q0=None):
    """Hard-clamp vs strong-Beta-soft credibility for **slightly impure references**.

    Tests the CLAUDE.md §2.2/§2.3/§6 design question: when reference panels carry a bit
    of admixture (here a known ``ref_impurity`` minority of foreign tracts), is it better
    to hard-clamp them (``w ≡ 1``) or to soften them with a strong ``Beta`` prior (``w``
    learned)? The mechanism under test: a hard clamp makes the tip emission a one-hot
    delta, so the tip posterior is **pinned to its label and can never dissent** over a
    foreign tract; any ``w < 1`` restores the genealogy's ability to override the label
    locally. So the decisive contrast is the impure references' **self-painting
    foreign-tract recall** — identically ~0 under a hard clamp, positive under soft.

    On one true-ARG simulation (admixed queries, a pure-source anchor core, two impure
    reference panels) it fits and scores:

    * ``hard_clamp`` — every reference hard-clamped (the impure ones inject
      confident-wrong votes and hide their own introgression);
    * ``soft_strong`` — impure refs softened with a strong ``Beta(alpha, beta)`` prior,
      the pure refs kept as the hard anchor core (never let the whole panel float, §6);
    * ``graded`` — exercises the per-tip prior API (:func:`tspaint.fit` ``priors``):
      half the impure refs "trusted" ``Beta(200, 1)``, half "suspect" ``Beta(2, 1)``;
    * an ``alpha`` sweep — is the benefit from *un-clamping* (≈flat in ``alpha``, the
      genome-scale span-weighted evidence swamping the prior) rather than prior strength?

    Parameters
    ----------
    T_admix, n_admix, n_pure, n_impure, sequence_length, recombination_rate, Ne, T_split, f_A : optional
        msprime scenario (see :func:`tspaint.sim.simulate_admixture_impure_refs`). The
        default (``T_admix=300``, ``sequence_length=5e6``) is the mosaic-impure-ref regime
        — enough recombination since the pulse that each impure reference haplotype is a
        majority-native mosaic with distributed foreign tracts (not a whole-haplotype
        mislabel), while the query↔reference signal is still strong on the true ARG.
    ref_impurity : float, optional
        Minority foreign fraction of each impure reference panel.
    seed : int, optional
        Random seed.
    max_iter : int, optional
        Maximum EM iterations per fit.
    alpha, beta : float, optional
        Default ``Beta`` prior for the ``soft_strong`` and ``graded`` configs.
    alpha_grid : iterable of float, optional
        ``alpha`` values (paired with ``beta``) for the sweep.
    Q0 : numpy.ndarray, optional
        Initial generator (defaults to a symmetric ``1e-3`` 2-state generator).

    Returns
    -------
    dict
        ``T_admix``, ``ref_impurity``, ``seed``, ``n_queries``, ``n_impure_refs``,
        ``n_pure_anchors``, ``mean_true_purity`` and ``true_purity_per_ref`` (census
        truth), ``configs`` (``hard_clamp``/``soft_strong``/``graded`` -> metrics with
        ``query_balanced_accuracy``, ``query_confidence``,
        ``impure_self_balanced_accuracy``, ``impure_self_foreign_recall`` (down-pass) and
        ``impure_self_foreign_recall_loo`` (the stronger leave-one-out introgression map,
        :func:`tspaint.output.loo_posterior_table`, *not* suppressed by the tip's own
        emission), ``mean_learned_w``, ``learned_w_per_ref``), ``alpha_sweep`` and
        ``graded_priors``.
    """
    ts = simulate_admixture_impure_refs(
        n_admix=n_admix, n_pure=n_pure, n_impure=n_impure, sequence_length=sequence_length,
        recombination_rate=recombination_rate, random_seed=seed, ref_impurity=ref_impurity,
        Ne=Ne, T_admix=T_admix, T_split=T_split, f_A=f_A)
    L = ts.sequence_length
    node_pop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    pid = {n: p for p, n in names.items()}
    state_of_pop = {pid[SOURCE_A]: 0, pid[SOURCE_B]: 1}

    def of_pop(name):
        return [int(s) for s in ts.samples() if node_pop[s] == pid[name]]

    queries = of_pop(ADMIXED)
    pure_refs = of_pop(SOURCE_A) + of_pop(SOURCE_B)         # the hard-clamped anchor core
    impure_A, impure_B = of_pop(REF_A_IMPURE), of_pop(REF_B_IMPURE)
    impure_refs = impure_A + impure_B

    # labels: pure refs by their source; impure refs by their NOMINAL (majority) source
    labels = {s: state_of_pop[node_pop[s]] for s in pure_refs}
    labels.update({s: 0 for s in impure_A})
    labels.update({s: 1 for s in impure_B})

    truth, _ = local_ancestry_truth(ts)
    query_truth = map_truth({q: truth[q] for q in queries}, state_of_pop)
    ref_truth = map_truth({r: truth[r] for r in impure_refs}, state_of_pop)
    true_purity = {int(r): _ref_purity(ref_truth[r], labels[r], L) for r in impure_refs}

    Q0 = Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3)
    soft = set(impure_refs)
    common = dict(Q0=Q0, max_iter=max_iter)
    args = (ts, labels, queries, impure_refs, query_truth, ref_truth, L)

    configs = {}
    configs["hard_clamp"] = _impure_config(*args, soft_refs=None, alpha=alpha, beta=beta,
                                           priors=None, **common)
    configs["soft_strong"] = _impure_config(*args, soft_refs=soft, alpha=alpha, beta=beta,
                                            priors=None, **common)
    # graded per-tip priors: first half "trusted" (strong), second half "suspect" (weak)
    half = len(impure_refs) // 2
    graded_priors = {int(r): (200.0, 1.0) for r in impure_refs[:half]}
    graded_priors.update({int(r): (2.0, 1.0) for r in impure_refs[half:]})
    configs["graded"] = _impure_config(*args, soft_refs=soft, alpha=alpha, beta=beta,
                                       priors=graded_priors, **common)

    sweep = []
    for a in alpha_grid:
        c = _impure_config(*args, soft_refs=soft, alpha=float(a), beta=beta, priors=None, **common)
        sweep.append({"alpha": float(a), "mean_learned_w": c["mean_learned_w"],
                      "query_balanced_accuracy": c["query_balanced_accuracy"],
                      "impure_self_foreign_recall": c["impure_self_foreign_recall"]})

    return {
        "T_admix": T_admix, "ref_impurity": ref_impurity, "seed": seed,
        "n_queries": len(queries), "n_impure_refs": len(impure_refs),
        "n_pure_anchors": len(pure_refs),
        "mean_true_purity": float(np.mean(list(true_purity.values()))),
        "true_purity_per_ref": true_purity,
        "configs": configs,
        "alpha_sweep": sweep,
        "graded_priors": graded_priors,
    }


def impure_reference_sweep(regimes, *, seeds=(1,), **kwargs):
    """Sweep :func:`impure_reference_experiment` across regimes (and seeds); tabulate the
    soft-vs-hard-clamp deltas for slightly-impure references.

    For each regime and seed it runs the ``hard_clamp`` and ``soft_strong`` configs and
    reports, averaged over seeds, the query-painting gain (soft − hard balanced accuracy)
    and the impure-reference introgression recovery — down-pass and **leave-one-out**
    (:func:`tspaint.output.loo_posterior_table`) foreign-tract recall. The tool behind the
    "where does softening impure references help?" characterisation (CLAUDE.md §6, §9):
    the payoff is bound by the genealogical foreign-tract signal (maximal with strong source
    anchoring + recent admixture, vanishing at old admixture), and is introgression recovery
    rather than a large query gain.

    Parameters
    ----------
    regimes : dict[str, dict] or iterable of dict
        Named regimes (or a bare list) of parameter overrides forwarded to
        :func:`impure_reference_experiment` — e.g.
        ``{"recent": {"T_admix": 120, "n_pure": 16}, "old": {"T_admix": 1000}}``.
    seeds : iterable of int, optional
        Seeds averaged per regime. Default ``(1,)``.
    **kwargs
        Common overrides forwarded to every :func:`impure_reference_experiment` call (e.g.
        ``sequence_length``, ``max_iter``, ``ref_impurity``). ``alpha_grid`` defaults to
        ``()`` here — the sweep needs only hard vs soft, not the per-``alpha`` curve.

    Returns
    -------
    list of dict
        One row per regime: ``regime``, ``params``, ``n_seeds``, ``mean_true_purity``,
        ``query_balanced_hard``/``_soft`` and ``query_gain``, ``foreign_recall_hard``/
        ``_soft`` (down-pass; ``_hard`` is 0 by construction — a hard clamp pins the tip),
        ``foreign_recall_loo_hard``/``_soft`` and ``introgression_gain_loo`` (soft − hard
        LOO recall), and ``mean_learned_w``.
    """
    items = list(regimes.items()) if isinstance(regimes, dict) else \
        [(str(i), dict(r)) for i, r in enumerate(regimes)]
    kwargs.setdefault("alpha_grid", ())
    out = []
    for name, params in items:
        acc = []
        for seed in seeds:
            r = impure_reference_experiment(seed=seed, **{**kwargs, **params})
            h, s = r["configs"]["hard_clamp"], r["configs"]["soft_strong"]
            acc.append([r["mean_true_purity"],
                        h["query_balanced_accuracy"], s["query_balanced_accuracy"],
                        h["impure_self_foreign_recall"], s["impure_self_foreign_recall"],
                        h["impure_self_foreign_recall_loo"], s["impure_self_foreign_recall_loo"],
                        s["mean_learned_w"]])
        purity, qh, qs, gh, gs, lh, ls, w = np.asarray(acc, float).mean(axis=0)
        out.append({
            "regime": name, "params": dict(params), "n_seeds": len(acc),
            "mean_true_purity": float(purity),
            "query_balanced_hard": float(qh), "query_balanced_soft": float(qs),
            "query_gain": float(qs - qh),
            "foreign_recall_hard": float(gh), "foreign_recall_soft": float(gs),
            "foreign_recall_loo_hard": float(lh), "foreign_recall_loo_soft": float(ls),
            "introgression_gain_loo": float(ls - lh),
            "mean_learned_w": float(w),
        })
    return out


def archaic_detection_experiment(*, ghost_fraction=0.25, n_admix=10, n_ref=8, sequence_length=2e6,
                                 T_admix=100.0, T_split_AB=2000.0, T_split_ABC=20000.0, Ne=1000,
                                 seed=1, max_iter=40, threshold=0.5):
    """Head-to-head: reference-free learned HMM vs the fixed-threshold flag (Plan B go/no-go).

    Compares :func:`tspaint.detect_archaic` (the Plan B generative depth-emission HMM, which
    learns the archaic depth and gives a calibrated posterior) against :func:`tspaint.detect_ghost`
    (the Plan A fixed-threshold flag) on one archaic-like ghost simulation plus a matched no-ghost
    control. Both run **reference-free** (no archaic reference). Per-locus ghost tracts are scored
    against the census truth.

    Parameters
    ----------
    ghost_fraction : float, optional
        Fraction of the admixed population from the unsampled ghost source.
    n_admix, n_ref, sequence_length, T_admix, T_split_AB, T_split_ABC, Ne : optional
        msprime ghost-scenario parameters (``T_split_ABC`` is the deep outgroup split — the
        archaic divergence depth).
    seed : int, optional
        Random seed.
    max_iter : int, optional
        Baum–Welch iteration cap for the HMM.
    threshold : float, optional
        ``P(archaic)`` threshold for the HMM's hard tracts.

    Returns
    -------
    dict
        ``ghost_fraction``, ``seed``, ``n_queries``, ``true_burden`` and two detector blocks
        ``archaic`` / ``ghost`` each with ``recall``, ``precision``, ``burden`` and ``control_fp``
        (no-ghost false-positive burden); ``archaic`` also reports the learned ``mu_archaic`` /
        ``mu_modern`` (log-depth).
    """
    from .sim import (simulate_admixture_with_ghost, local_ancestry_truth,
                      SOURCE_A, SOURCE_B, GHOST, ADMIXED)
    from .archaic import detect_archaic
    from .introgression import detect_ghost

    def setup(gf, sd):
        ts = simulate_admixture_with_ghost(n_admix=n_admix, n_ref=n_ref, sequence_length=sequence_length,
                recombination_rate=1e-8, random_seed=sd, ghost_fraction=gf, T_admix=T_admix,
                T_split_AB=T_split_AB, T_split_ABC=T_split_ABC, Ne=Ne)
        names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
        pid = {n: p for p, n in names.items()}
        node_pop = ts.tables.nodes.population
        of = lambda nm: [int(s) for s in ts.samples() if node_pop[s] == pid[nm]]
        q = of(ADMIXED)
        lab = {s: 0 for s in of(SOURCE_A)}
        lab.update({s: 1 for s in of(SOURCE_B)})
        tr, _ = local_ancestry_truth(ts)
        gid = pid[GHOST]
        true_g = {x: [(l, r) for (l, r, p) in tr[x] if p == gid] for x in q}
        return ts, lab, q, true_g

    def _ov(l, r, ivs):
        return sum(max(0.0, min(r, b) - max(l, a)) for (a, b) in ivs)

    def _score(tracts, true_g, q, L):
        det = tg = hit = 0.0
        for x in q:
            for (l, r) in tracts[x]:
                det += r - l
                hit += _ov(l, r, true_g[x])
            tg += sum(r - l for (l, r) in true_g[x])
        rec = hit / tg if tg else float("nan")
        prec = hit / det if det else float("nan")
        burden = (det / (L * len(q))) if q else float("nan")
        return rec, prec, burden, ((tg / (L * len(q))) if q else float("nan"))

    ts, lab, q, true_g = setup(ghost_fraction, seed)
    L = ts.sequence_length
    ar = detect_archaic(ts, lab, q, max_iter=max_iter)
    ar_tracts = {x: ar.tracts(x, threshold) for x in q}
    gh = detect_ghost(ts, lab, q)
    a_rec, a_prec, a_burden, true_burden = _score(ar_tracts, true_g, q, L)
    g_rec, g_prec, g_burden, _ = _score(gh.tracts_by_sample, true_g, q, L)

    ts0, lab0, q0, _ = setup(0.0, seed)
    L0 = ts0.sequence_length
    ar0 = detect_archaic(ts0, lab0, q0, max_iter=max_iter)
    gh0 = detect_ghost(ts0, lab0, q0)
    a_fp = float(np.mean([ar0.burden[x] for x in q0]))
    g_fp = sum(r - l for x in q0 for (l, r) in gh0.tracts(x)) / (L0 * len(q0))

    return {
        "ghost_fraction": ghost_fraction, "seed": seed, "n_queries": len(q),
        "true_burden": true_burden,
        "archaic": {"recall": a_rec, "precision": a_prec, "burden": a_burden, "control_fp": a_fp,
                    "mu_archaic": float(ar.mu[1]), "mu_modern": float(ar.mu[0])},
        "ghost": {"recall": g_rec, "precision": g_prec, "burden": g_burden, "control_fp": float(g_fp)},
    }
