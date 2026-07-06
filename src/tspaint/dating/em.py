"""EM loop for admixture rate through time (admix-dating design §4, rung 4).

Iterate the time-inhomogeneous E-step (:func:`tspaint.dating.estep.accumulate_time_binned_tv`) and
the directional penalised-spline M-step (:func:`tspaint.dating.mstep.directional_rate_splines`):

    init  q_AB(t)=q_BA(t)=const  (from the homogeneous tspaint.fit)
    E     accumulate per-cell occupation D and directional jumps J under the current Q(t)
    M     refit q_AB(t), q_BA(t) as penalised-Poisson splines -> new Q(t)

until the (span-integrated) log-likelihood converges. The payoff over the Stage-1 binned profile
is a *sharp* rate-through-time: the per-cell rate is no longer smeared by a single global Q.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import cell_centers, log_time_grid, assert_calibrated
from .estep import accumulate_time_binned_tv
from .mstep import directional_rate_splines

__all__ = ["RateThroughTime", "make_Q_of_cell", "fit_rate_through_time",
           "split_time", "EnsembleRateThroughTime"]


@dataclass
class RateThroughTime:
    """Result of :func:`fit_rate_through_time` — the directional cross-ancestry rate profile.

    Attributes
    ----------
    centers : (n_cells,) ndarray
        Cell centres (generations ago).
    q_AB, q_BA : (n_cells,) ndarray
        Directional transition rates per cell (forward in time: parent→child = old→young; a
        *backward*-time admixture A→B shows up in ``q_BA``).
    D : (n_cells, K) ndarray
        Per-cell occupation (the exposure / informative window).
    J : (n_cells, K, K) ndarray
        Per-cell directional expected jumps.
    loglik_history : list
        Span-integrated log-likelihood per EM iteration (non-decreasing).
    """
    centers: np.ndarray
    q_AB: np.ndarray
    q_BA: np.ndarray
    D: np.ndarray
    J: np.ndarray
    loglik_history: list

    def plot(self, ax=None, scale=1.0):
        """Plot the directional rate-through-time profile (log-time x-axis).

        ``scale`` multiplies the rates (e.g. pass ``Ne`` to show ``rate × N``). Returns the axes.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(self.centers, self.q_AB * scale, "-", lw=2, color="C2", label="q_AB(t)")
        ax.plot(self.centers, self.q_BA * scale, "--", lw=2, color="C1", label="q_BA(t)")
        ax.set_xscale("log")
        ax.set_xlabel("time (generations ago)")
        ax.set_ylabel("rate" + (" × scale" if scale != 1.0 else ""))
        ax.legend()
        return ax


def split_time(rtt, exposure_frac=0.01):
    """Estimate the split / divergence time from a :class:`RateThroughTime` profile.

    The combined cross-ancestry rate ``q_AB(t) + q_BA(t)`` is ~0 below the split (the two
    ancestries are still separate looking backward) and **rises** once ``t`` exceeds it (they
    coalesce) — the split shows up as the *onset*, not a peak. Returns the smallest cell centre
    (generations ago) at which the combined cross-rate first reaches **half** its
    exposure-guarded maximum (cells with negligible occupation ``D`` are ignored, so a deep
    no-data cell cannot masquerade as the onset).

    Parameters
    ----------
    rtt : RateThroughTime
        A fitted directional rate profile.
    exposure_frac : float, optional
        Ignore cells whose total occupation ``D`` is below this fraction of the max. Default 0.01.

    Returns
    -------
    float
        The split-time estimate (generations ago), or ``nan`` if the cross-rate never rises.
    """
    c = np.asarray(rtt.centers, float)
    r = np.asarray(rtt.q_AB, float) + np.asarray(rtt.q_BA, float)
    expo = np.asarray(rtt.D, float).sum(axis=1)
    emax = float(np.nanmax(expo)) if expo.size else 0.0
    if emax > 0:
        r = np.where(expo >= exposure_frac * emax, r, np.nan)
    rmax = float(np.nanmax(r)) if np.isfinite(r).any() else float("nan")
    if not np.isfinite(rmax) or rmax <= 0:
        return float("nan")
    idx = np.where(r >= 0.5 * rmax)[0]
    return float(c[idx[0]]) if idx.size else float("nan")


@dataclass
class EnsembleRateThroughTime:
    """Rate through time across an ARG ensemble — one :class:`RateThroughTime` per member, with a
    **confidence interval on the split time** from the ensemble spread.

    Returned by :meth:`tspaint.Painting.rate_through_time` when the painting was built from an
    ensemble of tree sequences (e.g. SINGER posterior samples): each member is dated on the shared
    fit, and the per-member split-time estimates (:func:`split_time`) become an interval — the
    ARG-uncertainty band on *when* the ancestries diverged.

    Attributes
    ----------
    members : list[RateThroughTime]
        Per-member directional rate profiles, on a shared time grid (so they average).
    split_times : numpy.ndarray
        ``(M,)`` per-member split-time estimates (generations ago).
    """
    members: list
    split_times: np.ndarray

    @property
    def centers(self):
        return self.members[0].centers

    @property
    def q_AB(self):
        """Ensemble-mean ``q_AB(t)``."""
        return np.nanmean(np.vstack([m.q_AB for m in self.members]), axis=0)

    @property
    def q_BA(self):
        """Ensemble-mean ``q_BA(t)``."""
        return np.nanmean(np.vstack([m.q_BA for m in self.members]), axis=0)

    def split_time(self, statistic="median"):
        """Point estimate of the split time over members (``"median"`` or ``"mean"``)."""
        v = self.split_times[np.isfinite(self.split_times)]
        if v.size == 0:
            return float("nan")
        return float(np.mean(v) if statistic == "mean" else np.median(v))

    def split_time_ci(self, level=0.95):
        """Percentile confidence interval ``(lo, hi)`` on the split time from the ensemble spread."""
        v = self.split_times[np.isfinite(self.split_times)]
        if v.size == 0:
            return (float("nan"), float("nan"))
        a = (1.0 - level) / 2.0 * 100.0
        return (float(np.percentile(v, a)), float(np.percentile(v, 100.0 - a)))

    def plot(self, ax=None, scale=1.0, level=0.95):
        """Per-member rate curves (faint) + the ensemble mean + the split-time CI band."""
        import matplotlib.pyplot as plt
        if ax is None:
            _f, ax = plt.subplots(figsize=(7, 4.2))
        for m in self.members:
            ax.plot(m.centers, np.asarray(m.q_AB) * scale, "-", color="C2", lw=0.6, alpha=0.3)
            ax.plot(m.centers, np.asarray(m.q_BA) * scale, "--", color="C1", lw=0.6, alpha=0.3)
        ax.plot(self.centers, self.q_AB * scale, "-", color="C2", lw=2.5, label="q_AB(t) mean")
        ax.plot(self.centers, self.q_BA * scale, "--", color="C1", lw=2.5, label="q_BA(t) mean")
        lo, hi = self.split_time_ci(level)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axvspan(lo, hi, color="0.5", alpha=0.2, label=f"split {int(level * 100)}% CI")
        st = self.split_time()
        if np.isfinite(st):
            ax.axvline(st, color="k", lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("time (generations ago)")
        ax.set_ylabel("rate" + (" × scale" if scale != 1.0 else ""))
        ax.legend()
        return ax

    def __repr__(self):
        lo, hi = self.split_time_ci()
        return (f"EnsembleRateThroughTime(M={len(self.members)}, "
                f"split_time={self.split_time():.0f} [{lo:.0f}, {hi:.0f}] gen)")


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


def fit_rate_through_time(ts, labels, edges=None, *, n_cells=40, n_iter=15, em_init=8, Q0=None,
                          estimate_pi=False, soft_refs=None, n_knots=20, tol=1e-4,
                          floor=1e-9, fit_result=None):
    """Fit the time-inhomogeneous directional admixture-rate-through-time profile by EM.

    Parameters
    ----------
    ts : tskit.TreeSequence
    labels : dict[int, int]
        Reference sample-node id -> ancestry state (0/1).
    edges : array_like, optional
        Fine log-time grid edges (:func:`tspaint.dating.log_time_grid`). If ``None`` (default), a
        grid of ``n_cells`` log-spaced cells is built automatically from the node ages.
    n_cells : int
        Number of log-time cells when ``edges`` is auto-constructed (ignored if ``edges`` given).
    n_iter : int
        Maximum EM iterations.
    em_init : int
        Iterations of the homogeneous :func:`tspaint.fit` used to initialise (ignored when
        ``fit_result`` is supplied).
    n_knots : int
        Spline knots for the M-step.
    tol : float
        Relative log-likelihood change for convergence.
    fit_result : tspaint.em.FitResult, optional
        A precomputed homogeneous fit to **warm-start** from (its ``Q`` seeds the rate splines and
        its ``w``/``pi`` build the tip emissions). When given, the internal homogeneous
        :func:`tspaint.fit` is skipped — used by :meth:`tspaint.Painting.rate_through_time` to reuse
        the painting's fit rather than refitting.

    Returns
    -------
    RateThroughTime
    """
    from ..em import fit, build_emissions
    from ..ids import resolve_labels

    labels = resolve_labels(ts, labels)                # keys may be sample-ID strings or node indices
    if edges is None:                                  # auto log-time grid from the node ages
        nt = np.asarray(ts.tables.nodes.time, float)
        assert_calibrated(nt)                          # reject a raw (uncalibrated) tsinfer ARG
        pos = nt[nt > 0]
        edges = log_time_grid(max(1.0, float(pos.min())), float(pos.max()) * 1.05, n_cells)
    if fit_result is not None:                         # warm-start: reuse a precomputed fit
        res = fit_result
    else:
        # Q0=None -> fit() scales the initial generator to the (calibrated) node-age axis, so the
        # warm-start fit does not wash out on a deep time scale (same guard as tspaint.paint).
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
