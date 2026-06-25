"""Admixture-rate-through-time E-step tests (admix-dating, rung 1)."""
import numpy as np
import pytest

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


def test_poisson_spline_recovers_known_step_rate():
    """The M-step penalised spline recovers a (smooth) step rate from Poisson data with a
    coalescent-shaped exposure (admix-dating rung 3)."""
    from tslai.dating.mstep import select_lambda_gcv
    rng = np.random.default_rng(1)
    centers = np.geomspace(20.0, 20000.0, 60)
    true = 1e-3 / (1.0 + np.exp(-(np.log(centers) - np.log(2000.0)) * 3.0))  # onset ~2000
    exposure = 1e6 * np.exp(-centers / 8000.0)                              # decays deep
    events = rng.poisson(true * exposure).astype(float)
    fit = select_lambda_gcv(centers, events, exposure)
    rate = fit["rate"]
    below = rate[centers < 1000].mean()
    above = rate[(centers > 4000) & (centers < 12000)].mean()
    assert above > 5 * below                                               # the rise is recovered
    m = exposure > 1
    assert np.corrcoef(rate[m], true[m])[0, 1] > 0.9                        # tracks the truth


@pytest.mark.slow
def test_fit_rate_through_time_recovers_split():
    """End-to-end: the inhomogeneous EM recovers the divergence onset on a clean A/B split.

    Cross-ancestry coalescence is impossible more recently than the population split, so the
    fitted cross-rate q_AB(t) must be ~0 for t < T_split and rise once t exceeds it. Also checks
    the EM log-likelihood is (weakly) monotone. Uses the auto log-time grid (edges=None)."""
    import msprime
    from tslai.dating import fit_rate_through_time

    N, T_split = 1000, 2000.0
    d = msprime.Demography()
    d.add_population(name="A", initial_size=N)
    d.add_population(name="B", initial_size=N)
    d.add_population(name="ANC", initial_size=N)
    d.add_population_split(time=T_split, derived=["A", "B"], ancestral="ANC")
    ts = msprime.sim_ancestry(samples={"A": 6, "B": 6}, demography=d, sequence_length=3e5,
                              recombination_rate=1e-8, random_seed=1, ploidy=1)
    pop = ts.tables.nodes.population
    labels = {int(s): (0 if pop[s] == 0 else 1) for s in ts.samples()}

    rtt = fit_rate_through_time(ts, labels, n_iter=8)        # edges=None -> auto grid

    ll = rtt.loglik_history
    assert all(ll[i + 1] >= ll[i] - 1e-6 for i in range(len(ll) - 1))       # EM monotone
    c = rtt.centers
    recent = np.nanmean(rtt.q_AB[c < 0.5 * T_split])
    deep = np.nanmean(rtt.q_AB[(c > T_split) & (c < 4 * T_split)])
    assert deep > 3 * recent + 1e-9                                         # onset recovered
