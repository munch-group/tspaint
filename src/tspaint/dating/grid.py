"""Log-time grid for the admixture-rate-through-time E-step (admix-dating design §1, §2).

A fine, geometrically-spaced grid on node age (backward time) for *accumulating* the
time-resolved sufficient statistics. Decoupled from the (smooth) rate model used in the
M-step: branches are split at cell boundaries, each sub-interval homogeneous under its cell.
"""
from __future__ import annotations

import numpy as np

__all__ = ["log_time_grid", "cell_centers", "split_branch"]


def log_time_grid(t_min, t_max, n_cells):
    """Geometric (log-spaced) cell edges spanning ``[t_min, t_max]``.

    Returns ``(n_cells + 1,)`` edges. Geometric spacing gives roughly equal expected
    branch-occupation per cell (the "even power" placement).
    """
    return np.geomspace(t_min, t_max, n_cells + 1)


def cell_centers(edges):
    """Geometric centres of the cells defined by ``edges``."""
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
