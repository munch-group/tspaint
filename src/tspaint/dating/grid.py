"""Log-time grid for the admixture-rate-through-time E-step (admix-dating design §1, §2).

A fine, geometrically-spaced grid on node age (backward time) for *accumulating* the
time-resolved sufficient statistics. Decoupled from the (smooth) rate model used in the
M-step: branches are split at cell boundaries, each sub-interval homogeneous under its cell.
"""
from __future__ import annotations

import numpy as np

__all__ = ["log_time_grid", "cell_centers", "split_branch", "assert_calibrated",
           "MIN_CALIBRATED_MAX_AGE"]

#: A deepest node age (generations) below this almost certainly means the tree sequence is
#: **uncalibrated** — e.g. a raw tsinfer ARG, whose frequency-ordered times are ~``[0, 1]`` — rather
#: than a real coalescent genealogy, whose TMRCA is many generations. The auto time-grid guards
#: against silently dating in these bogus units (bypass with an explicit ``edges=``).
MIN_CALIBRATED_MAX_AGE = 10.0


def assert_calibrated(node_time):
    """Raise if node ages look **uncalibrated** (not in generations) for auto-grid dating.

    Guards :func:`tspaint.dating.fit_rate_through_time` / :meth:`tspaint.Painting.rate_through_time`
    against building a time grid from a raw tsinfer ARG (times ~``[0, 1]``), which collapses every
    cell near 1 and makes the rate profile / split time meaningless. Bypass by passing an explicit
    ``edges=`` grid (then you are asserting the times are what you intend).
    """
    nt = np.asarray(node_time, float)
    tmax = float(nt.max()) if nt.size else 0.0
    if tmax < MIN_CALIBRATED_MAX_AGE:
        raise ValueError(
            f"tree-sequence node ages look uncalibrated (deepest node = {tmax:.3g}; a real "
            "coalescent genealogy is many generations deep) — dating needs node times in "
            "GENERATIONS. A raw tsinfer ARG is uncalibrated: calibrate it with "
            "io.tsinfer(source, date=True, mutation_rate=...) or tsdate, or use io.singer / "
            "io.relate whose node times are already in generations. Pass edges= to bypass this check.")


def log_time_grid(t_min, t_max, n_cells):
    """Geometric (log-spaced) cell edges spanning ``[t_min, t_max]``.

    Geometric spacing gives roughly equal expected branch-occupation per cell (the "even power"
    placement).

    Parameters
    ----------
    t_min : float
        Youngest (smallest) cell edge, in generations. Must be ``> 0`` (geometric spacing).
    t_max : float
        Oldest (largest) cell edge, in generations.
    n_cells : int
        Number of cells; the grid has ``n_cells + 1`` edges.

    Returns
    -------
    numpy.ndarray
        ``(n_cells + 1,)`` ascending cell-boundary **times** (generations), log-spaced — the
        boundaries are real time values evenly spaced on a log axis, not log-time.
    """
    return np.geomspace(t_min, t_max, n_cells + 1)


def cell_centers(edges):
    """Geometric centres of the cells defined by ``edges``.

    The geometric mean ``sqrt(edges[:-1] * edges[1:])`` is the natural cell centre on the
    log-time axis the grid lives on.

    Parameters
    ----------
    edges : array_like
        ``(n_cells + 1,)`` ascending cell-boundary times (:func:`log_time_grid`).

    Returns
    -------
    numpy.ndarray
        ``(n_cells,)`` geometric centre (generations) of each cell.
    """
    edges = np.asarray(edges, float)
    return np.sqrt(edges[:-1] * edges[1:])


def split_branch(t_c, t_p, edges):
    """Split a branch ``[t_c, t_p]`` (``t_c < t_p``) into per-cell sub-intervals.

    Parameters
    ----------
    t_c, t_p : float
        Child (younger) and parent (older) node ages.
    edges : array_like
        Grid cell edges (ascending).

    Returns
    -------
    list[tuple[int, float]]
        ``(cell_index, duration)`` sub-intervals **ordered parent → child** (descending
        time), matching the ``xi[s_p, s_c]`` / ``expm`` convention of
        :func:`tspaint.branch_stats.vanloan_integral`. Portions of the branch outside
        ``[edges[0], edges[-1]]`` are assigned to the nearest boundary cell so the whole
        branch is covered (so per-cell totals sum to the whole-branch statistics).
    """
    edges = np.asarray(edges, float)
    ncell = len(edges) - 1
    cuts = [float(e) for e in edges if t_c < e < t_p]
    pts = np.array([t_c, *cuts, t_p], float)
    subs = []
    for lo, hi in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (lo + hi)
        k = int(np.clip(np.searchsorted(edges, mid, side="right") - 1, 0, ncell - 1))
        subs.append((k, float(hi - lo)))
    subs.reverse()                          # parent (high time) -> child (low time)
    return subs
