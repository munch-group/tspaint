"""Gate for a Phasic replacement of the branch-stats seam (CLAUDE.md §3.2, §12).

The seam is one function — ``branch_expected_stats(Q, t, xi) -> (dwell, jumps)``, which is what
``accumulate.py`` calls once per edge. ``examples/phasic_seam.py`` holds a ``new_`` version of it,
currently a dummy delegating to the reference. Fill it in with Phasic and this file becomes the
gate: it feeds identical inputs to old and new, checks the contract's invariants, and binds the
new function into the real E-step.

The last test pins the property that licenses the whole approach: the ``scipy.linalg.expm``
reference is accurate to far better than ``RTOL``, so *agreeing with it is a proof of accuracy*
and no separate ground truth is needed in the comparison tests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from phasic_seam import (                                          # noqa: E402
    RTOL,
    branch_inputs,
    check_branch_expected_stats,
    check_estep_unchanged,
    gen2,
    gen3,
    invariant_problems,
    max_relative_error,
    new_branch_expected_stats,
    xi_variants,
)
from tspaint.branch_stats import branch_expected_stats, vanloan_integral  # noqa: E402


# --- the seam: old vs new on identical inputs ---------------------------------------------

def test_new_matches_reference():
    """Every (Q, t, xi) in the battery. The K^2 one-hot xi's read back every entry of whatever
    internal E[reward | s_p, s_c] matrix an implementation builds, so this is exactly as
    sensitive as comparing internal kernels — including the small-lambda*t rows, where a
    cancellation-based backend fails. Do not loosen RTOL to make one pass."""
    assert check_branch_expected_stats(verbose=False)


def test_contract_invariants():
    """The contract holds regardless of the reference: dwell sums to t, nothing negative,
    jumps has a zero diagonal and is zero wherever Q is."""
    rng = np.random.default_rng(1)
    for label, Q, t in branch_inputs():
        for xlabel, xi in xi_variants(Q, t, rng):
            dwell, jumps = new_branch_expected_stats(Q, t, xi)
            problems = invariant_problems(Q, t, dwell, jumps)
            assert not problems, f"{label} [xi={xlabel}]: {problems}"


def test_root_branch_returns_zeros():
    """t <= 0 is a root branch (CLAUDE.md §3.4/§4.5): return zeros, do not raise. accumulate.py
    skips these before calling, but a drop-in must not blow up if handed one."""
    for t in (0.0, -1.0):
        dwell, jumps = new_branch_expected_stats(gen2(0.5, 0.5), t, np.eye(2) / 2)
        assert np.allclose(dwell, 0.0) and np.allclose(jumps, 0.0)


def test_span_weight_is_the_callers_job():
    """The seam must NOT apply the edge span — accumulate.py does (S_dwell += span * dwell).
    dwell.sum() is t, full stop."""
    Q = gen2(0.3, 0.6)
    xi = np.zeros((2, 2))
    xi[0, 1] = 1.0
    dwell, _ = new_branch_expected_stats(Q, 2.0, xi)
    assert np.isclose(dwell.sum(), 2.0)


def test_generator_agnostic_k3():
    """K-way is a generator swap (§12), and K >= 3 admits complex eigenvalues."""
    rng = np.random.default_rng(2)
    Q = gen3()
    for t in (1e-6, 1.0, 30.0):
        xi = rng.random((3, 3))
        xi /= xi.sum()
        new_d, new_j = new_branch_expected_stats(Q, t, xi)
        ref_d, ref_j = branch_expected_stats(Q, t, xi)
        assert max_relative_error(new_d, ref_d) <= RTOL
        assert max_relative_error(new_j, ref_j) <= RTOL
        assert np.isclose(new_d.sum(), t, rtol=1e-9, atol=1e-9 * max(1.0, t))


@pytest.mark.slow
def test_real_estep_unchanged():
    """The definitive gate: bind the new function into the real E-step on a simulated tree
    sequence and compare the sufficient statistics."""
    assert check_estep_unchanged(verbose=False)


# --- the property the comparison tests rest on --------------------------------------------

@pytest.mark.slow
def test_reference_is_accurate_enough_to_compare_against():
    """[MEASURED] The property that licenses every comparison test above.

    CLAUDE.md §8.7 assumed scipy's expm is fragile at small t / stiff Q*t and named that as a
    motivation for Phasic. Measured against a 60-digit mpmath ground truth over all the integral
    entries in the battery: **median relative error 4.6e-16, worst 3.2e-12**. The handful above
    1e-13 are Pade-truncation artefacts confined to the *smallest, most deeply nested* entries
    (those ~(lambda*t)^2 below the leading term) in two narrow bands — ||Qt|| ~ 1e-2 and stiff
    ||Qt|| >= 1e3. Twelve good digits on the worst entry the method has.

    Two assertions, both load-bearing:
      (a) the reference's own error stays far below the RTOL we hold candidates to, so "agrees
          with the reference" really does mean "accurate" — no separate ground truth needed;
      (b) at small lambda*t the reference is at machine precision. That is what makes the
          small-lambda*t rows a valid *detector*: a spectral backend is wrong in the 4th
          significant digit there while the reference is exact to 1e-16.

    (mpmath.expm was itself cross-checked against adaptive quadrature on the closed-form 2-state
    transition matrix — an evaluation sharing no code with either expm — and agreed exactly.)
    """
    mp = pytest.importorskip("mpmath")

    def exact_vanloan(Q, t, E, dps=50):
        """The same Van Loan block exponential, at `dps` digits instead of 16."""
        Q = np.asarray(Q, float)
        K = Q.shape[0]
        with mp.workdps(dps):
            block = mp.zeros(2 * K, 2 * K)
            for i in range(K):
                for j in range(K):
                    qt = mp.mpf(float(Q[i, j])) * mp.mpf(float(t))
                    block[i, j] = qt
                    block[K + i, K + j] = qt
                    block[i, K + j] = mp.mpf(float(E[i, j])) * mp.mpf(float(t))
            top_right = mp.expm(block)[0:K, K:2 * K]
            return [[top_right[i, j] for j in range(K)] for i in range(K)]

    def rel(got, exact):
        worst = 0.0
        for i, row in enumerate(exact):
            for j, ex in enumerate(row):
                if ex == 0:
                    continue
                worst = max(worst, float(abs(mp.mpf(float(got[i, j])) - ex) / abs(ex)))
        return worst

    def rewards(Q):
        """The only two E families that ever reach vanloan_integral: dwell in m, m->n jumps."""
        Q = np.asarray(Q, float)
        K = Q.shape[0]
        out = []
        for m in range(K):
            E = np.zeros((K, K))
            E[m, m] = 1.0
            out.append(E)
        for m in range(K):
            for n in range(K):
                if m != n and Q[m, n] != 0.0:
                    E = np.zeros((K, K))
                    E[m, n] = Q[m, n]
                    out.append(E)
        return out

    worst, worst_small_lt = 0.0, 0.0
    for _label, Q, t in branch_inputs():
        lambda_t = float(np.max(-np.diag(np.asarray(Q, float)))) * t   # the controlling quantity
        for E in rewards(Q):
            err = rel(vanloan_integral(Q, t, E), exact_vanloan(Q, t, E))
            worst = max(worst, err)
            if lambda_t <= 1e-4:
                worst_small_lt = max(worst_small_lt, err)

    # (a) the licence for RTOL: the reference's own error sits far below the tolerance we hold
    #     candidates to, so "agrees with the reference" really does mean "accurate".
    assert worst < RTOL / 10, (
        f"reference relative error is now {worst:.1e}, within 10x of RTOL={RTOL:g} — "
        "comparing a backend against it would no longer prove accuracy"
    )
    # (b) the detector: at small lambda*t — where a spectral backend collapses — the reference is
    #     at machine precision, so those rows genuinely discriminate rather than measuring noise.
    assert worst_small_lt < 1e-13, (
        f"reference relative error at lambda*t <= 1e-4 is now {worst_small_lt:.1e} — the "
        "small-lambda*t rows can no longer distinguish a lossy backend from the reference"
    )
