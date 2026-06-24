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

from tslai.experiments import admixture_experiment, flicker_vs_true_boundaries


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
