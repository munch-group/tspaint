"""Admixture rate through time (time-inhomogeneous directional mugration EM).

A *new estimator* riding the tslai engine: make the ancestry CTMC time-inhomogeneous and
estimate the cross-ancestry transition rate as a function of (backward) time. See
``notes/admix_dating_design.md`` for the full design.

* `log_time_grid` / `split_branch` — the fine log-time accumulation grid (rung 1).
* `branch_cell_stats` — per-cell endpoint-conditioned dwell + directional jumps (rung 1).
* `rate_through_time_binned` — Stage-1 binned profile from one homogeneous fit (rung 2).
* `directional_rate_splines` — the penalised-Poisson directional M-step (rung 3).
* `fit_rate_through_time` — the full time-inhomogeneous EM (rung 4).
"""
from .grid import log_time_grid, cell_centers, split_branch
from .estep import (branch_cell_stats, accumulate_time_binned, rate_through_time_binned,
                    composite_transition, accumulate_time_binned_tv)
from .mstep import fit_poisson_spline, select_lambda_gcv, directional_rate_splines
from .em import RateThroughTime, make_Q_of_cell, fit_rate_through_time

__all__ = [
    "log_time_grid", "cell_centers", "split_branch",
    "branch_cell_stats", "accumulate_time_binned", "rate_through_time_binned",
    "composite_transition", "accumulate_time_binned_tv",
    "fit_poisson_spline", "select_lambda_gcv", "directional_rate_splines",
    "RateThroughTime", "make_Q_of_cell", "fit_rate_through_time",
]
