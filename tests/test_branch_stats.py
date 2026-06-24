"""Rung 1 gate (CLAUDE.md §11.1.2, §8.7): the Van Loan branch-stats kernel.

Validated three independent ways: (a) closed-form 2-state CTMC expectations,
(b) independent Simpson quadrature of the defining integral, and (c) invariants
and small/large-``t`` stability.
"""
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.integrate import simpson

from tslai.branch_stats import branch_expected_stats, vanloan_integral


def gen(a, b):
    return np.array([[-a, a], [b, -b]], float)


def int_P_im(a, b, t):
    """Closed form of ∫_0^t P_im(τ) dτ for Q = [[-a, a], [b, -b]]."""
    lam = a + b
    e = (1.0 - np.exp(-lam * t)) / lam
    I = np.empty((2, 2))
    I[0, 0] = b * t / lam + a * e / lam
    I[0, 1] = a * t / lam - a * e / lam
    I[1, 0] = b * t / lam - b * e / lam
    I[1, 1] = a * t / lam + b * e / lam
    return I


def quad_integral(Q, t, E, npts=4001):
    """Independent Simpson evaluation of ∫_0^t expm(Qτ) E expm(Q(t-τ)) dτ."""
    taus = np.linspace(0.0, t, npts)
    vals = np.stack([expm(Q * tau) @ E @ expm(Q * (t - tau)) for tau in taus])
    return simpson(vals, x=taus, axis=0)


@pytest.mark.parametrize("a,b,t", [(0.1, 0.2, 1.0), (0.5, 0.5, 2.3),
                                   (0.03, 0.7, 0.4), (1.2, 0.4, 5.0)])
def test_against_analytic_2state(a, b, t):
    # Set xi to the joint from a known start state i (row i of P, which sums to 1):
    # then dwell[m] == ∫_0^t P_im dτ and jumps[m,n] == q_mn * ∫_0^t P_im dτ.
    Q = gen(a, b)
    P = expm(Q * t)
    I = int_P_im(a, b, t)
    for i in (0, 1):
        xi = np.zeros((2, 2))
        xi[i, :] = P[i, :]
        dwell, jumps = branch_expected_stats(Q, t, xi)
        np.testing.assert_allclose(dwell, I[i, :], rtol=1e-7, atol=1e-10)
        for m in (0, 1):
            for n in (0, 1):
                if m != n:
                    np.testing.assert_allclose(jumps[m, n], Q[m, n] * I[i, m],
                                               rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("a,b,t", [(0.1, 0.2, 1.0), (0.7, 0.3, 2.0)])
def test_against_independent_quadrature(a, b, t):
    Q = gen(a, b)
    rng = np.random.default_rng(0)
    xi = rng.random((2, 2))
    xi /= xi.sum()
    dwell, jumps = branch_expected_stats(Q, t, xi)
    P = expm(Q * t)
    for m in (0, 1):
        E = np.zeros((2, 2)); E[m, m] = 1.0
        cond = quad_integral(Q, t, E) / P
        assert np.isclose(dwell[m], np.sum(xi * cond), rtol=1e-5, atol=1e-7)
    for m in (0, 1):
        for n in (0, 1):
            if m != n:
                E = np.zeros((2, 2)); E[m, n] = Q[m, n]
                cond = quad_integral(Q, t, E) / P
                assert np.isclose(jumps[m, n], np.sum(xi * cond), rtol=1e-5, atol=1e-7)


def test_vanloan_matches_quadrature():
    Q = gen(0.3, 0.6); t = 1.7
    for (m, n) in [(0, 0), (1, 1), (0, 1)]:
        E = np.zeros((2, 2)); E[m, n] = 1.0
        np.testing.assert_allclose(vanloan_integral(Q, t, E),
                                   quad_integral(Q, t, E), rtol=1e-5, atol=1e-7)


def test_dwell_sums_to_t():
    Q = gen(0.4, 0.9); t = 3.1
    rng = np.random.default_rng(1)
    for _ in range(5):
        xi = rng.random((2, 2)); xi /= xi.sum()
        dwell, _ = branch_expected_stats(Q, t, xi)
        assert np.isclose(dwell.sum(), t, rtol=1e-9, atol=1e-9)


def test_zero_t_returns_zeros():
    Q = gen(0.5, 0.5)
    xi = np.array([[0.5, 0.0], [0.0, 0.5]])
    d, j = branch_expected_stats(Q, 0.0, xi)
    assert np.allclose(d, 0.0) and np.allclose(j, 0.0)


def test_small_t_concentrates_in_start_state():
    Q = gen(0.5, 0.5); t = 1e-9
    P = expm(Q * t)
    xi = np.zeros((2, 2)); xi[0, :] = P[0, :]
    d, j = branch_expected_stats(Q, t, xi)
    assert np.isclose(d.sum(), t, rtol=1e-6)
    assert d[0] > d[1]
    assert np.all(j >= 0.0) and j.sum() < 1e-8


def test_stiff_Qt_finite_and_nonneg():
    Q = gen(50.0, 80.0); t = 10.0          # Q*t large / stiff
    P = expm(Q * t)
    xi = np.zeros((2, 2)); xi[0, :] = P[0, :]
    d, j = branch_expected_stats(Q, t, xi)
    assert np.all(np.isfinite(d)) and np.all(np.isfinite(j))
    assert np.all(d >= -1e-12) and np.all(j >= -1e-12)
    assert np.isclose(d.sum(), t, rtol=1e-6)


def test_q_to_zero_gives_no_jumps():
    Q = gen(1e-12, 1e-12); t = 2.0
    xi = np.array([[1.0, 0.0], [0.0, 0.0]])   # start = end = state 0
    d, j = branch_expected_stats(Q, t, xi)
    assert np.isclose(d[0], t, atol=1e-6)
    assert j.sum() < 1e-8
