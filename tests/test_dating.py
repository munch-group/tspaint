"""Admixture-rate-through-time E-step tests (admix-dating, rung 1)."""
import numpy as np

from tslai.model import make_generator_2state
from tslai.branch_stats import branch_expected_stats
from tslai.dating import log_time_grid, split_branch, branch_cell_stats


def test_split_branch_covers_and_orders():
    edges = log_time_grid(1.0, 1000.0, 30)
    subs = split_branch(5.0, 600.0, edges)
    # durations sum to the branch length; ordered parent -> child (descending time)
    assert np.isclose(sum(d for _k, d in subs), 600.0 - 5.0)
    # cell indices non-increasing (we go from high time near the parent to low near the child)
    ks = [k for k, _d in subs]
    assert ks == sorted(ks, reverse=True)


def test_branch_cell_stats_sum_invariant():
    """Per-cell dwell/jumps summed over cells == whole-branch branch_expected_stats
    (additive property of the Van Loan integral), for a homogeneous generator."""
    Q = make_generator_2state(0.002, 0.005)
    edges = log_time_grid(1.0, 2000.0, 40)
    t_c, t_p = 7.0, 850.0                                  # spans many cells, in range
    rng = np.random.default_rng(0)
    xi = rng.random((2, 2))
    xi /= xi.sum()

    dwell, jumps = branch_cell_stats(lambda k: Q, t_c, t_p, xi, edges)
    tot_d = sum(dwell.values())
    tot_j = sum(jumps.values())
    ref_d, ref_j = branch_expected_stats(Q, t_p - t_c, xi)

    assert np.allclose(tot_d, ref_d, atol=1e-9)
    assert np.allclose(tot_j, ref_j, atol=1e-9)
    assert np.isclose(tot_d.sum(), t_p - t_c)             # dwell sums to branch length


def test_branch_cell_stats_localises_in_time():
    """A reward on a branch confined to a single cell lands in that cell only."""
    Q = make_generator_2state(0.003, 0.003)
    edges = log_time_grid(1.0, 2000.0, 40)
    # a short branch wholly inside one cell
    k0 = 20
    t_c, t_p = edges[k0] + 0.1, edges[k0 + 1] - 0.1
    xi = np.full((2, 2), 0.25)
    dwell, _ = branch_cell_stats(lambda k: Q, t_c, t_p, xi, edges)
    assert set(dwell) == {k0}
    assert np.isclose(sum(dwell.values()).sum(), t_p - t_c)
