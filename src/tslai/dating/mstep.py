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

    Returns
    -------
    rate : (n_cells,) ndarray
        Fitted rate ``exp(Bβ)`` at the cell centres.
    edf : float
        Effective degrees of freedom (trace of the smoother hat matrix) — for GCV.
    beta : (ncoef,) ndarray
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
    term = np.where(ev > 0, ev * np.log(ev / mu[m]), 0.0) - (ev - mu[m])
    return 2.0 * np.sum(term)


def select_lambda_gcv(centers, events, exposure, lams=None, **kw):
    """Fit at a grid of smoothing parameters; pick the one minimising GCV."""
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
    """Fit ``q_AB(t)`` and ``q_BA(t)`` by GCV-selected penalised Poisson splines.

    Parameters
    ----------
    D : (n_cells, K) array — per-cell occupation (exposure: ``D[:,0]`` for A→B, ``D[:,1]`` for B→A).
    J : (n_cells, K, K) array — per-cell directional jumps.
    centers : (n_cells,) cell centres.

    Returns
    -------
    dict with ``q_AB``, ``q_BA`` (rates at the centres) and the selected ``lambda_*``/``edf_*``.
    """
    occ_frac = kw.pop("occ_frac", 0.02)
    out = {}
    for name, ev, ex in (("AB", J[:, 0, 1], D[:, 0]), ("BA", J[:, 1, 0], D[:, 1])):
        ex = np.asarray(ex, float)
        win = ex > occ_frac * (ex.max() if ex.max() > 0 else 1.0)   # informative window
        fit = select_lambda_gcv(centers, np.asarray(ev, float), np.where(win, ex, 0.0), **kw)
        rate = fit["rate"].copy()
        idx = np.where(win)[0]
        if len(idx):                                                # flat-clamp outside window
            rate[:idx[0]] = rate[idx[0]]
            rate[idx[-1] + 1:] = rate[idx[-1]]
        out[f"q_{name}"] = rate
        out[f"lambda_{name}"] = fit["lambda"]
        out[f"edf_{name}"] = fit["edf"]
        out[f"window_{name}"] = win
    return out
