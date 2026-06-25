"""Admixture rate through time (time-inhomogeneous directional mugration EM).

A *new estimator* riding the tslai engine: make the ancestry CTMC time-inhomogeneous and
estimate the cross-ancestry transition rate as a function of (backward) time. See
``notes/admix_dating_design.md`` for the full design. This subpackage is built rung-by-rung;
currently the time-resolved E-step accumulator and the Stage-1 binned profile.
"""
from .grid import log_time_grid, cell_centers, split_branch
from .estep import branch_cell_stats, accumulate_time_binned, rate_through_time_binned

__all__ = [
    "log_time_grid",
    "cell_centers",
    "split_branch",
    "branch_cell_stats",
    "accumulate_time_binned",
    "rate_through_time_binned",
]
