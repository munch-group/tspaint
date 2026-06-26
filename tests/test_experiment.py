"""Rung 8b/8c gate (CLAUDE.md §9, §7.3): the end-to-end payoff and the bp/ decision.

Heavier than the unit tests (a full sim -> fit -> paint -> score pass). It both
demonstrates that soft tree-sequence local ancestry recovers the truth and supplies
the §7.3 decision: blocked-EM flicker at persist-but-reparent boundaries is
negligible relative to the ~1.0 discontinuity at true ancestry switches, so loopy
BP/EP (``bp/``) is not needed. (Strong-structure sims on the true ARG — the easy
regime; the hard-regime head-to-head needs the external comparators.)
"""
import numpy as np
import pytest

from tspaint.experiments import (admixture_experiment, flicker_vs_true_boundaries,
                               age_sweep, scaling_sweep, impure_reference_experiment,
                               impure_reference_sweep)


@pytest.mark.slow
def test_painting_recovers_local_ancestry_and_blocked_em_suffices():
    r = admixture_experiment(T_admix=300, n_admix=6, n_ref=10, sequence_length=3e5,
                             f_A=0.5, max_iter=7, seed=1)
    assert r["n_queries"] == 12
    # near-perfect painting (measured 1.0) of balanced 50/50 admixture (chance = 0.5),
    # with several true ancestry switches per haplotype present
    assert r["accuracy"] > 0.9

    # §7.3: flicker at non-true boundaries must be small vs the true-switch discontinuity
    qs = list(r["tracks"].keys())
    off = np.mean([flicker_vs_true_boundaries(r["tracks"], r["truth_states"], q)["mean_flicker_off_true"]
                   for q in qs])
    assert off < 0.1            # measured ~0.001 -> blocked EM sufficient, bp/ deferred


@pytest.mark.slow
def test_signal_lost_at_old_admixture():
    # §9 (refined): discriminates at recent/moderate admixture; the reference signal is
    # lost at old admixture (admixed lineages coalesce among themselves before the
    # pulse) -> balanced accuracy -> chance, confidence collapses. Plain accuracy would
    # be misleading on the lopsided truth, hence balanced accuracy + confidence.
    common = dict(n_admix=12, n_ref=8, sequence_length=3e5, f_A=0.5, Ne=1000,
                  T_split=10000, max_iter=5, seed=1)
    recent, old = age_sweep([30, 3000], infer=False, **common)
    assert recent["balanced_accuracy"] > 0.85 and recent["confidence"] > 0.4
    assert old["balanced_accuracy"] < 0.65                  # ~chance: reference signal lost
    assert old["confidence"] < recent["confidence"]


@pytest.mark.slow
def test_scaling_sweep_structure_and_timing():
    rows = scaling_sweep([8, 16], infer=False, n_ref=6, sequence_length=5e4,
                         T_admix=300, Ne=1000, T_split=5000, max_iter=3, seed=1, f_A=0.5)
    assert len(rows) == 2
    for r in rows:
        assert r["n_haplotypes"] == 2 * (r["n_admix"] + 2 * 6)   # ploidy 2 x (admixed + 2 refs)
        assert r["t_fit"] > 0.0 and r["n_trees"] >= 1
        assert 0.0 <= r["balanced_accuracy"] <= 1.0
    assert rows[1]["n_haplotypes"] > rows[0]["n_haplotypes"]     # sweep increases sample size


@pytest.mark.slow
def test_impure_reference_softening_un_clamps_introgression():
    # CLAUDE.md §2.2/§2.3/§6: for slightly impure references, hard-clamping (w≡1) pins the
    # tip posterior to its label so it can NEVER reveal its own foreign tracts; softening
    # with a strong Beta prior un-clamps it. Decisive contrast = the impure refs' own
    # foreign-tract recall (identically 0 when hard-clamped). The benefit is from
    # un-clamping, not prior strength (genome-scale evidence swamps the prior -> w ~flat
    # in alpha); and a pure anchor core keeps the query painting identifiable / unhurt.
    r = impure_reference_experiment(T_admix=300, sequence_length=1.5e6, n_admix=6, n_pure=4,
                                    n_impure=5, ref_impurity=0.15, max_iter=6, seed=1,
                                    alpha_grid=(2.0, 2000.0))
    hard = r["configs"]["hard_clamp"]
    soft = r["configs"]["soft_strong"]
    graded = r["configs"]["graded"]

    # hard clamp: impure refs carry no learned credibility and cannot reveal foreign tracts
    assert hard["mean_learned_w"] == 1.0
    assert hard["impure_self_foreign_recall"] == 0.0

    # softening un-clamps them: w learned below 1 (≈ genealogy-agree fraction, not collapsed),
    # and the down-pass foreign-tract recovery becomes nonzero (the hard clamp is exactly 0)
    assert 0.4 < soft["mean_learned_w"] < 0.98
    assert soft["impure_self_foreign_recall"] > hard["impure_self_foreign_recall"]

    # the leave-one-out introgression map (output.loo_posterior_table) is the stronger lens:
    # it surfaces the impure refs' foreign tracts even under the hard clamp (where the
    # down-pass posterior is pinned to the label), and softening does not degrade it
    assert hard["impure_self_foreign_recall_loo"] > hard["impure_self_foreign_recall"]
    assert hard["impure_self_foreign_recall_loo"] > 0.1
    assert soft["impure_self_foreign_recall_loo"] >= hard["impure_self_foreign_recall_loo"] - 0.05

    # benefit is from UN-CLAMPING, not prior strength: learned w ~flat across alpha
    # (span-weighted genome-scale evidence swamps the Beta prior, CLAUDE.md §6)
    ws = [s["mean_learned_w"] for s in r["alpha_sweep"]]
    assert max(ws) - min(ws) < 0.05

    # softening the impure refs does not hurt query painting (pure anchor core holds it)
    assert soft["query_balanced_accuracy"] >= hard["query_balanced_accuracy"] - 0.05

    # the per-tip graded-prior path runs over every impure ref and yields valid credibilities
    assert set(r["graded_priors"]) == set(soft["learned_w_per_ref"])
    assert all(0.0 <= v <= 1.0 for v in graded["learned_w_per_ref"].values())


@pytest.mark.slow
def test_impure_reference_sweep_signal_bound_benefit():
    # CLAUDE.md §6/§9: softening's introgression-recovery payoff is bound by the genealogical
    # foreign-tract signal -> present at recent admixture, ~gone at old admixture (the
    # query<->reference link is itself lost, §9). The sweep driver tabulates the deltas.
    rows = impure_reference_sweep(
        {"recent": {"T_admix": 120}, "old": {"T_admix": 1000}},
        seeds=(1,), sequence_length=1.5e6, n_pure=6, n_impure=5, n_admix=6, max_iter=5)
    assert len(rows) == 2
    by = {r["regime"]: r for r in rows}
    for r in rows:
        assert r["foreign_recall_hard"] == 0.0             # down-pass pinned under hard clamp
        assert 0.4 < r["mean_learned_w"] < 0.98             # soft refs un-clamped, not collapsed
        assert np.isfinite(r["query_gain"]) and np.isfinite(r["introgression_gain_loo"])
    # recent admixture recovers impure-ref introgression; old admixture (signal gone) does not
    assert by["recent"]["foreign_recall_loo_soft"] > by["old"]["foreign_recall_loo_soft"]
    assert by["old"]["foreign_recall_loo_soft"] < 0.3
