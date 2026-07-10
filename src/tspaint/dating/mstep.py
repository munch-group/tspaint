"""Directional penalised-Poisson-spline M-step (admix-dating design §3).

Given the time-resolved sufficient statistics ``D`` (occupation) and ``J`` (directional jumps),
fit the rate functions ``q_AB(t)=exp(s_AB(log t))``, ``q_BA(t)=exp(s_BA(log t))`` by penalised
Poisson regression — ``J_AB(g) ~ Poisson(q_AB(g)·D_A(g))`` with offset ``log D_A(g)``, a B-spline
basis on log-time, a 2nd-difference roughness penalty, and the smoothing parameter chosen by GCV.
The exploration (``explore/spline_resolution.*``) showed this captures well-powered abrupt
features while smoothing the data-poor tail.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

__all__ = ["fit_poisson_spline", "select_lambda_gcv", "directional_rate_splines"]


def _basis(log_t, n_knots, degree):
    inner = np.linspace(log_t.min(), log_t.max(), n_knots)
    knots = np.r_[[inner[0]] * degree, inner, [inner[-1]] * degree]
    return BSpline.design_matrix(log_t, knots, degree).toarray()


def _penalty(ncoef, order=2):
    Dm = np.diff(np.eye(ncoef), order, axis=0)
    return Dm.T @ Dm


def fit_poisson_spline(centers, events, exposure, lam, n_knots=25, degree=3, order=2,
                       max_iter=60, tol=1e-7):
    """Penalised Poisson log-spline via penalised IRLS.

    Model ``events ~ Poisson(exposure · exp(B β))`` with 2nd-difference penalty ``λ βᵀPβ``.

    Parameters
    ----------
    centers : (n_cells,) array_like
        Cell-centre times; the B-spline basis is built on ``log(centers)``.
    events : (n_cells,) array_like
        Per-cell event counts — the Poisson response (e.g. directional jumps ``J[:, m, n]``).
    exposure : (n_cells,) array_like
        Per-cell exposure / offset (occupation ``D[:, m]``); the Poisson mean is
        ``exposure · exp(Bβ)``. Cells with ``exposure <= 0`` are dropped from the fit.
    lam : float
        Roughness-penalty weight ``λ`` (larger ⇒ smoother).
    n_knots : int, optional
        Number of B-spline knots on log-time. Default ``25``.
    degree : int, optional
        B-spline degree. Default ``3`` (cubic).
    order : int, optional
        Order of the difference penalty (``2`` = second-difference roughness). Default ``2``.
    max_iter : int, optional
        Maximum penalised-IRLS iterations. Default ``60``.
    tol : float, optional
        Convergence tolerance on the max coefficient change between iterations. Default ``1e-7``.

    Returns
    -------
    rate : (n_cells,) ndarray
        Fitted rate ``exp(Bβ)`` at the cell centres.
    edf : float
        Effective degrees of freedom (trace of the smoother hat matrix) — for GCV.
    beta : (ncoef,) ndarray
        Fitted B-spline coefficients ``β``.
    """
    centers = np.asarray(centers, float)
    events = np.asarray(events, float)
    exposure = np.asarray(exposure, float)
    x = np.log(centers)
    B = _basis(x, n_knots, degree)
    P = _penalty(B.shape[1], order)
    m = exposure > 0
    beta = np.zeros(B.shape[1])
    if events[m].sum() > 0:
        beta[:] = np.log(events[m].sum() / exposure[m].sum())
    Bm = B[m]
    for _ in range(max_iter):
        eta = Bm @ beta
        mu = exposure[m] * np.exp(np.clip(eta, -30, 30))
        w = np.maximum(mu, 1e-12)
        z = eta + (events[m] - mu) / w
        A = (Bm.T * w) @ Bm + lam * P
        rhs = (Bm.T * w) @ z
        new = np.linalg.solve(A, rhs)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    eta = Bm @ beta
    mu = exposure[m] * np.exp(np.clip(eta, -30, 30))
    w = np.maximum(mu, 1e-12)
    A = (Bm.T * w) @ Bm + lam * P
    edf = float(np.trace(np.linalg.solve(A, (Bm.T * w) @ Bm)))
    rate = np.exp(np.clip(B @ beta, -30, 30))
    return rate, edf, beta


def _deviance(events, mu):
    m = mu > 0
    ev = events[m]
    pos = ev > 0
    ratio = np.ones_like(ev)
    ratio[pos] = ev[pos] / mu[m][pos]                 # avoid log(0) for empty cells
    term = np.where(pos, ev * np.log(ratio), 0.0) - (ev - mu[m])
    return 2.0 * np.sum(term)


def select_lambda_gcv(centers, events, exposure, lams=None, **kw):
    """Fit at a grid of smoothing parameters; pick the one minimising GCV.

    Parameters
    ----------
    centers, events, exposure
        As for :func:`fit_poisson_spline`.
    lams : array_like, optional
        Grid of penalty weights ``λ`` to try. Default ``None`` — uses
        ``np.geomspace(1e-2, 1e4, 18)``.
    **kw
        Forwarded to :func:`fit_poisson_spline` (``n_knots``, ``degree``, ``order``, ...).

    Returns
    -------
    dict
        The GCV-selected fit: ``gcv`` (minimum GCV score), ``lambda`` (chosen ``λ``), ``rate``
        (the ``(n_cells,)`` fitted rate) and ``edf`` (effective degrees of freedom).
    """
    if lams is None:
        lams = np.geomspace(1e-2, 1e4, 18)
    best = None
    n = int(np.sum(exposure > 0))
    for lam in lams:
        rate, edf, _ = fit_poisson_spline(centers, events, exposure, lam, **kw)
        dev = _deviance(np.asarray(events, float)[exposure > 0],
                        (exposure * rate)[exposure > 0])
        gcv = n * dev / (n - edf) ** 2 if n > edf + 1 else np.inf
        if best is None or gcv < best["gcv"]:
            best = {"gcv": gcv, "lambda": lam, "rate": rate, "edf": edf}
    return best


def directional_rate_splines(D, J, centers, **kw):
    """Fit every directional rate ``q_{mn}(t)`` (``m ≠ n``) by GCV-selected penalised Poisson splines.

    Generalises the 2-state ``q_AB`` / ``q_BA`` to ``K`` ancestries: each ordered off-diagonal pair
    ``(m, n)`` is a Poisson spline of ``J[:, m, n]`` with exposure ``D[:, m]``.

    Parameters
    ----------
    D : (n_cells, K) array_like
        Per-cell expected occupation (dwell). Column ``D[:, m]`` is the Poisson exposure for
        every ``m → *`` rate.
    J : (n_cells, K, K) array_like
        Per-cell expected directional jump counts; ``J[:, m, n]`` is the response fitted for the
        ordered pair ``(m, n)``.
    centers : (n_cells,) array_like
        Cell centres of the log-time grid (:func:`tspaint.dating.cell_centers`).
    **kw
        ``occ_frac`` (float, default ``0.02``) sets the **informative window**: for each source
        state ``m``, cells with exposure ``D[:, m] > occ_frac * max(D[:, m])`` are fitted, and the
        rate is flat-clamped to the window's edge values outside it. Any remaining keywords are
        forwarded to :func:`select_lambda_gcv` (hence to :func:`fit_poisson_spline` — ``n_knots``,
        ``degree``, ``order``, ...).

    Returns
    -------
    dict
        ``q`` — the ``(n_cells, K, K)`` fitted rate array (diagonal 0); ``(m, n)`` → the selected
        ``{"lambda", "edf", "window"}`` for that pair; and, for ``K == 2``, the legacy ``q_AB`` /
        ``q_BA`` slices.
    """
    occ_frac = kw.pop("occ_frac", 0.02)
    D = np.asarray(D, float)
    J = np.asarray(J, float)
    ncell, K = D.shape
    q = np.zeros((ncell, K, K))
    out = {"q": q}
    for m in range(K):
        for n in range(K):
            if m == n:
                continue
            ex = np.asarray(D[:, m], float)
            win = ex > occ_frac * (ex.max() if ex.max() > 0 else 1.0)   # informative window
            fit = select_lambda_gcv(centers, np.asarray(J[:, m, n], float), np.where(win, ex, 0.0), **kw)
            rate = fit["rate"].copy()
            idx = np.where(win)[0]
            if len(idx):                                                # flat-clamp outside window
                rate[:idx[0]] = rate[idx[0]]
                rate[idx[-1] + 1:] = rate[idx[-1]]
            q[:, m, n] = rate
            out[(m, n)] = {"lambda": fit["lambda"], "edf": fit["edf"], "window": win}
    if K == 2:                                                          # legacy scalar keys
        out["q_AB"], out["q_BA"] = q[:, 0, 1], q[:, 1, 0]
    return out
