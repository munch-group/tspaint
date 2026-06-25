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
                               age_sweep, scaling_sweep)


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
