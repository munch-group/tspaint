"""Tests for the ARG-ensemble merge layer (tspaint.ensemble)."""
import numpy as np
import pytest

from tspaint.output import Segment, INFORMATIVE, MISSING_INFO
from tspaint.ensemble import merge_posterior_tables
from tspaint.validate import per_base_accuracy


def seg(left, right, post, status=INFORMATIVE):
    return Segment(left, right, np.array(post, float), status)


def test_merge_mean_std_and_common_refinement():
    t0 = {0: [seg(0, 2, [0.8, 0.2]), seg(2, 4, [0.4, 0.6])]}   # breakpoint at 2
    t1 = {0: [seg(0, 1, [0.6, 0.4]), seg(1, 4, [0.5, 0.5])]}   # breakpoint at 1
    segs = merge_posterior_tables([t0, t1])[0]

    assert [(s.left, s.right) for s in segs] == [(0, 1), (1, 2), (2, 4)]   # union of breakpoints
    np.testing.assert_allclose(segs[0].posterior, [0.7, 0.3])             # mean(0.8,0.6)
    np.testing.assert_allclose(segs[1].posterior, [0.65, 0.35])           # mean(0.8,0.5)
    np.testing.assert_allclose(segs[2].posterior, [0.45, 0.55])           # mean(0.4,0.5)
    np.testing.assert_allclose(segs[0].posterior_std, [0.1, 0.1])
    np.testing.assert_allclose(segs[2].posterior_std, [0.05, 0.05])
    assert segs[0].left == 0 and segs[-1].right == 4                       # whole-genome coverage


def test_status_consensus_any_informative():
    t0 = {0: [seg(0, 1, [0.9, 0.1], INFORMATIVE), seg(1, 2, [0.5, 0.5], MISSING_INFO)]}
    t1 = {0: [seg(0, 1, [0.5, 0.5], MISSING_INFO), seg(1, 2, [0.5, 0.5], MISSING_INFO)]}
    segs = merge_posterior_tables([t0, t1])[0]
    assert segs[0].status == INFORMATIVE and segs[0].n_informative == 1   # one member informs
    assert segs[1].status == MISSING_INFO and segs[1].n_informative == 0  # all members missing


def test_merged_tracks_are_scoreable_by_validate():
    t0 = {0: [seg(0, 2, [0.9, 0.1])]}
    t1 = {0: [seg(0, 2, [0.7, 0.3])]}
    merged = merge_posterior_tables([t0, t1])
    assert per_base_accuracy(merged, {0: [(0, 2, 0)]}) == 1.0   # duck-compatible with Segment


def test_single_member_is_identity():
    t0 = {0: [seg(0, 2, [0.9, 0.1]), seg(2, 4, [0.3, 0.7])]}
    segs = merge_posterior_tables([t0])[0]
    assert [(s.left, s.right) for s in segs] == [(0, 2), (2, 4)]
    np.testing.assert_allclose(segs[0].posterior, [0.9, 0.1])
    np.testing.assert_allclose(segs[0].posterior_std, [0.0, 0.0])


def test_merge_reduces_independent_noise():
    # The point of merging posterior samples: when members are noisy but their errors are
    # INDEPENDENT (as for thinned SINGER samples), averaging the posteriors recovers a
    # cleaner painting. Here truth is state 0 everywhere; each member's P(A) is centred at
    # 0.6 with independent noise (individually only ~0.65 accurate); the M-member average
    # concentrates near 0.6 -> argmax 0 almost everywhere.
    rng = np.random.default_rng(0)
    L, M = 100, 30
    truth = {0: [(0, L, 0)]}
    tables = []
    for _ in range(M):
        segs = [seg(x, x + 1, [float(np.clip(0.6 + rng.normal(0, 0.25), 0.01, 0.99)), 0.0])
                for x in range(L)]
        for s in segs:
            s.posterior[1] = 1.0 - s.posterior[0]
        tables.append({0: segs})
    single = np.mean([per_base_accuracy(t, truth) for t in tables])
    merged = per_base_accuracy(merge_posterior_tables(tables), truth)
    assert merged > single + 0.15      # averaging independent noise clearly helps
    assert merged > 0.9


@pytest.mark.slow
def test_arg_ensemble_experiment_runs():
    from tspaint.experiments import arg_ensemble_experiment
    r = arg_ensemble_experiment(M=4, n_admix=12, n_ref=12, sequence_length=1.5e5,
                                mutation_rate=1e-7, max_iter=4, seed=1)
    assert r["M"] == 4 and len(r["single_balanced_per_member"]) == 4
    assert 0.0 <= r["merged_balanced"] <= 1.0
    assert 0.0 <= r["true_balanced"] <= 1.0
    assert set(r["merged_tracks"].keys()) == set(r["queries"])        # valid painting per query
    # (whether merged beats single is regime-dependent and a research question, not a
    # unit-test invariant — see CLAUDE.md §7.4: a tsinfer ensemble has correlated errors.)
