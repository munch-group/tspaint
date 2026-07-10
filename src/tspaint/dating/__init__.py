"""Admixture rate through time — time-inhomogeneous directional mugration EM.

A separate estimator riding the same tspaint engine (Felsenstein pruning + Van Loan
endpoint-conditioned statistics): make the ancestry CTMC *time-inhomogeneous*, with a
piecewise-constant generator on a fine log-time grid, and estimate the two cross-ancestry
transition rates ``q_AB(t)``, ``q_BA(t)`` as smooth functions of (backward) time. The
profile locates divergence and gene-flow epochs, their direction/asymmetry, and ongoing
flow. This lives *side by side* with the paint-only path — it does not change how
:func:`tspaint.paint` works; see ``notes/admix_dating_design.md`` for the full design and the
rung-by-rung validation.

Typical use::

    from tspaint.dating import fit_rate_through_time
    rtt = fit_rate_through_time(ts, labels)   # auto log-time grid from the node ages
    rtt.plot()                                # q_AB(t), q_BA(t) on a log-time axis

Public API:

* :func:`fit_rate_through_time` — the full time-inhomogeneous EM (the headline entry point).
* :class:`RateThroughTime` — its result (``centers``, ``q``, ``.rate(m, n)``, ``.pairs``, ``.plot()``).
* :func:`split_time` / :func:`split_times` — the divergence time (2-state scalar / per-pair dict).
* :func:`paint_qt` — paint focal tips under a fitted ``Q(t)`` (the side-by-side painter).
* :func:`rate_through_time_binned` — fast Stage-1 binned profile from one homogeneous fit.
* :func:`log_time_grid` / :func:`split_branch` — the log-time accumulation grid.
* :func:`branch_cell_stats` — per-cell endpoint-conditioned dwell + directional jumps.
* :func:`directional_rate_splines` — the penalised-Poisson directional M-step.
* :func:`make_Q_of_cell` — build the per-cell generator callable from ``q_AB``, ``q_BA``.
"""
from .grid import log_time_grid, cell_centers, split_branch
from .estep import (branch_cell_stats, accumulate_time_binned, rate_through_time_binned,
                    composite_transition, accumulate_time_binned_tv, paint_qt)
from .mstep import fit_poisson_spline, select_lambda_gcv, directional_rate_splines
from .em import (RateThroughTime, make_Q_of_cell, fit_rate_through_time,
                 split_time, split_times, EnsembleRateThroughTime)

__all__ = [
    "log_time_grid", "cell_centers", "split_branch",
    "branch_cell_stats", "accumulate_time_binned", "rate_through_time_binned",
    "composite_transition", "accumulate_time_binned_tv", "paint_qt",
    "fit_poisson_spline", "select_lambda_gcv", "directional_rate_splines",
    "RateThroughTime", "make_Q_of_cell", "fit_rate_through_time",
    "split_time", "split_times", "EnsembleRateThroughTime",
]
