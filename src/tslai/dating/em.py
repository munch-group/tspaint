"""EM loop for admixture rate through time (admix-dating design §4, rung 4).

Iterate the time-inhomogeneous E-step (:func:`tslai.dating.estep.accumulate_time_binned_tv`) and
the directional penalised-spline M-step (:func:`tslai.dating.mstep.directional_rate_splines`):

    init  q_AB(t)=q_BA(t)=const  (from the homogeneous tslai.fit)
    E     accumulate per-cell occupation D and directional jumps J under the current Q(t)
    M     refit q_AB(t), q_BA(t) as penalised-Poisson splines -> new Q(t)

until the (span-integrated) log-likelihood converges. The payoff over the Stage-1 binned profile
is a *sharp* rate-through-time: the per-cell rate is no longer smeared by a single global Q.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import cell_centers
from .estep import accumulate_time_binned_tv
from .mstep import directional_rate_splines

__all__ = ["RateThroughTime", "make_Q_of_cell", "fit_rate_through_time"]


@dataclass
class RateThroughTime:
    """Result of :func:`fit_rate_through_time`."""
    centers: np.ndarray      # cell centres (generations)
    q_AB: np.ndarray         # A->B rate per cell
    q_BA: np.ndarray         # B->A rate per cell
    D: np.ndarray            # (n_cells, K) occupation
    J: np.ndarray            # (n_cells, K, K) directional jumps
    loglik_history: list     # span-integrated log-likelihood per EM iteration


def make_Q_of_cell(q_AB, q_BA):
    """Build a ``cell_index -> (2,2) generator`` callable from per-cell directional rates."""
    cache = {}

    def Q(k):
        Qk = cache.get(k)
        if Qk is None:
            a, b = float(q_AB[k]), float(q_BA[k])
            Qk = np.array([[-a, a], [b, -b]])
            cache[k] = Qk
        return Qk

    return Q


def fit_rate_through_time(ts, labels, edges, *, n_iter=15, em_init=8, Q0=None,
                          estimate_pi=False, soft_refs=None, n_knots=20, tol=1e-4,
                          floor=1e-9):
    """Fit the time-inhomogeneous directional admixture-rate-through-time profile by EM.

    Parameters
    ----------
    ts : tskit.TreeSequence
    labels : dict[int, int]
        Reference sample-node id -> ancestry state (0/1).
    edges : array_like
        Fine log-time grid edges (:func:`tslai.dating.log_time_grid`).
    n_iter : int
        Maximum EM iterations.
    em_init : int
        Iterations of the homogeneous :func:`tslai.fit` used to initialise.
    n_knots : int
        Spline knots for the M-step.
    tol : float
        Relative log-likelihood change for convergence.

    Returns
    -------
    RateThroughTime
    """
    from ..em import fit, build_emissions
    from ..model import make_generator_2state

    Q0 = Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3)
    res = fit(ts, labels, Q0=Q0, max_iter=em_init, estimate_pi=estimate_pi, soft_refs=soft_refs)
    emissions = build_emissions(ts, labels, res.w, res.pi)
    pi = res.pi
    centers = cell_centers(edges)
    ncell = len(centers)

    q_AB = np.full(ncell, max(float(res.Q[0, 1]), floor))
    q_BA = np.full(ncell, max(float(res.Q[1, 0]), floor))
    history = []
    D = J = None
    for it in range(n_iter):
        Q_of_cell = make_Q_of_cell(q_AB, q_BA)
        D, J, ll = accumulate_time_binned_tv(ts, Q_of_cell, pi, emissions, edges)
        history.append(ll)
        sp = directional_rate_splines(D, J, centers, n_knots=n_knots)
        q_AB = np.maximum(np.nan_to_num(sp["q_AB"], nan=floor), floor)
        q_BA = np.maximum(np.nan_to_num(sp["q_BA"], nan=floor), floor)
        if it > 0 and abs(history[-1] - history[-2]) < tol * (abs(history[-2]) + 1e-12):
            break
    return RateThroughTime(centers, q_AB, q_BA, D, J, history)
