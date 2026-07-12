"""Phasic drop-in sandbox for the tspaint branch-stats seam (CLAUDE.md §3.2, §12).

Run it::

    python examples/phasic_seam.py          # or: pytest tests/test_phasic_dropin.py

THE SEAM — one function
-----------------------
::

    branch_expected_stats(Q, t, xi)  ->  (dwell, jumps)

That is the whole seam. ``tspaint/accumulate.py`` (the EM E-step) calls exactly this, once
per edge; a Phasic backend replaces exactly this. Everything underneath — the Van Loan
``expm``, the ``/P`` conditioning, the memo on ``(Q, t)`` — is a private implementation
detail of the current backend, and **a replacement is free to restructure all of it**: cache
on ``Q`` instead of ``t``, batch across branches, cache nothing. Caching does not belong in
the signature, and on real data it barely matters anyway: node times are near-continuous, so
on a 4 Mb sim 4313 edges carry 3369 *distinct* branch lengths — memoising on ``t`` saves
~22% of the ``expm`` calls, not a factor of anything.

Out of scope: ``tspaint/dating/estep.py`` separately calls ``branch_stats.vanloan_integral``
for the raw un-normalised integral per time cell, under a time-inhomogeneous generator. It is
a different consumer with a different shape, and a separate job if you ever want it.

THE CONTRACT
------------
``Q``    (K, K) generator, ``Q[m, n]`` = rate m -> n, rows sum to 0.
``t``    float, branch length ``node_time[parent] - node_time[child]`` in generations.
         ``t`` is **given** (read off the tree), not random. ``t <= 0`` is a root branch
         (CLAUDE.md §3.4) and must return ``(zeros(K), zeros((K, K)))``.
``xi``   (K, K) posterior over ``(parent_state, child_state)``, **normalised to sum to 1**.
         Row = parent = the CTMC's *start* (time runs parent -> child, old -> young). This is
         literally ``prune_tree(...).xi[(parent, child)]``. Felsenstein pruning conditions on
         **both** endpoints — that is what ``xi`` is — so an endpoint-marginalised quantity is
         not enough anywhere inside your implementation.

``dwell`` (K,)     ``dwell[m]``    = E[time spent in state m on this branch | xi]
``jumps`` (K, K)   ``jumps[m, n]`` = E[# of m->n transitions | xi], m != n; diagonal 0.

The **span weight is the caller's job** (``S_dwell += span * dwell``) — do not apply it here.
Invariants a correct implementation satisfies for any normalised ``xi``:
``dwell.sum() == t`` (exactly — the branch has to be somewhere); ``dwell >= 0``; ``jumps >= 0``;
``diag(jumps) == 0``; ``jumps[m, n] == 0`` wherever ``Q[m, n] == 0``.

HOW TO USE THIS FILE
--------------------
:func:`new_branch_expected_stats` is a **dummy that delegates to the current implementation**,
so a fresh run reports all-pass — which is what shows the harness is wired up before you touch
anything. Replace its body with Phasic, keep the signature and returns, and re-run. Every check
feeds the **same generated inputs** to the old and the new function and compares.

[MEASURED] IMPLEMENTING IT WITH PHASIC — THE STRUCTURE THAT MAKES IT WORK
-------------------------------------------------------------------------
Read against https://munch-group.org/phasic/mathref/27_van_loan_equivalence.html.

**1. The key structural fact: ``xi / P`` is RANK-1.** Felsenstein pruning builds
``xi[i, j] ∝ α_i · P[i, j] · β_j`` — parent outside-message, transition, child inside-message —
so the ``P`` **cancels exactly**::

    W := xi / P  =  (α ⊗ β) / (αᵀ P β)          <- an outer product; no 1/P anywhere

Verified to machine precision on 441 real E-step edges (σ₂/σ₁ ≤ 2.4e-16, max|W| = 8.97). The
seam's reward is therefore an *outer product of the two messages*, never an arbitrary matrix.

**2. The whole seam is then ONE Van Loan integral, not K².** With ``W = xi / P``::

    G = ∫₀ᵗ e^{Qᵀτ} W e^{Qᵀ(t−τ)} dτ = vanloan_integral(Q.T, t, W)
    dwell[m]   = G[m, m]                       <- the diagonal
    jumps[m,n] = Q[m, n] · G[m, n]             <- the off-diagonal, scaled by the rate

(Derivation: ``Σ_ij W[i,j]·V^(m)[i,j] = ∫ (P(τ)ᵀ W P(t−τ)ᵀ)[m,m] dτ``, and likewise off-diagonal
for jumps.) Verified exact against the reference. It needs **no cache at all** — the reward
depends on ``xi``, so there is nothing to cache — which is the concrete vindication of keeping
caching out of the signature.

**NOT ADOPTED — deferred; trigger is K > 2.** Be careful with the speedup number: the *seam*
gets 2.5× faster at K=2, but **pruning is 78% of the E-step**, so end-to-end it is only **+12%**.
Not worth narrowing the contract for. The case reopens with the K-way generalisation, because
this turns K² block-``expm``s into 1::

    K:            2      3       4       6
    seam speedup: 2.5x   5.1x    11.7x   19.8x        (measured, per-edge)

Contract caveat if it is ever adopted: it folds ``1/P`` into the reward, so it is exact only for
``xi`` that pruning can actually emit (where ``W`` is the bounded rank-1 outer product above —
machine-precise, 2e-14). On a synthetic ``xi`` that puts mass where ``P`` is tiny, ``max|W| → 1e8``
and it degrades to ~3e-10. See :func:`xi_variants`. Recorded in CLAUDE.md §3.2.

Because ``W = α ⊗ β`` is rank-1, that integral is a **forward–backward convolution**::

    G = ∫₀ᵗ f(τ) g(t−τ)ᵀ dτ,   f(τ) = e^{Qᵀτ} α  (forward),   g(s) = e^{Qs} β  (backward)

— exactly the shape a graph/phase-type forward algorithm computes. ``α`` is Phasic's initial
vector; ``β`` is a **terminal weighting**.

**3. Where Phasic, as documented, does not yet reach.** Theorem 29.1 gives
``accumulated_visiting_time(G, t) = α ∫₀ᵗ e^{Ss} ds``. For a conservative generator that is
*exactly* ``α @ vanloan_integral(Q, t, E) @ ones`` (verified) — i.e. the Van Loan integral
**contracted with the ones vector**, which is precisely ``β = ones``: the **endpoint-marginalised**
dwell. Two gaps for this use case:

  * **The terminal weighting is hardwired to ``ones``.** We need a general ``β`` (the child's
    message). Marginalising over the child endpoint is exactly the thing Felsenstein forbids —
    it is mutant 6 in the harness, and it fails.
  * **Rewards are diagonal (``△(r)``).** Dwell rewards are diagonal, but jump rewards are
    ``q_mn·e_m e_nᵀ`` — off-diagonal. And endpoint-*conditioned* jumps are **not** ``q_mn ×``
    endpoint-conditioned dwell (only the *marginalised* ones are proportional), so no diagonal
    reward recovers them.

Handing Phasic the block matrix ``[[Q, E], [0, Q]]`` directly does not work either: its top rows
sum to ``+Σ_n E[m,n] > 0``, so it is mass-creating, not a sub-intensity matrix. A diagonal shift
to fix that must be undone by ``e^{ct}``, which overflows at tspaint's ``t`` (up to 2e4).

**So the one thing to ask Phasic for is: the un-contracted Van Loan block for a rank-1 reward** —
``∫₀ᵗ e^{Qᵀτ} (α βᵀ) e^{Qᵀ(t−τ)} dτ`` returned as a ``K×K`` matrix, not pre-contracted with ``α``
on the left or ``ones`` on the right. One such call per edge is the entire seam.

WHY THE ``xi`` BATTERY IS EXHAUSTIVE
------------------------------------
Any implementation ultimately computes ``E[reward | s_p, s_c]`` — a ``(K, K)`` matrix per reward
— and contracts it with ``xi``. Setting ``xi`` to the one-hot at ``(i, j)`` makes the output read
back exactly that matrix's ``(i, j)`` entry. So the ``K²`` one-hot ``xi``'s in :func:`xi_variants`
probe **every internal entry individually**: this battery is exactly as sensitive as comparing
internal kernels would be, without the seam having to expose one.

:func:`xi_variants` also generates the **message family** ``xi ∝ (α ⊗ β) ⊙ P`` — the *only* shape
pruning can actually emit (see above). Those are the inputs a backend must be exact on. A one-hot
``xi`` sitting on an *improbable* endpoint pair is a reachable but pathological limit (it forces
``max|W| → 1/P[i,j]``), and a ``W``-based implementation like the one-integral form above loses
accuracy there while staying machine-exact on everything pruning produces. If a backend fails
*only* those rows, that is a known, benign trade — not the same thing as failing the small-λ·t rows.

[MEASURED] WHY OLD-vs-NEW IS ALSO THE PRECISION TEST — AND WHY NOT TO LOOSEN ``RTOL``
------------------------------------------------------------------------------------
The current ``scipy.linalg.expm`` implementation is accurate to far better than ``RTOL``:
against a 60-digit mpmath ground truth, median relative error 4.6e-16 and worst 3.2e-12 (the
few above 1e-13 are Padé-truncation artefacts on the tiniest, most deeply nested entries, in two
narrow bands — ‖Qt‖ ≈ 1e-2 and stiff ‖Qt‖ ≳ 1e3). Two consequences:

  1. **Phasic's value at this seam is throughput and K-way scale, not numerics.** There is no
     precision win available — neither the small-``t`` nor the stiff-``Q*t`` worry in CLAUDE.md
     §8.7 survives contact with measurement. Do not "fix" the reference.
  2. **Because the reference is that accurate, agreeing with it IS a proof of accuracy** — no
     separate ground truth is needed. ``RTOL`` = 1e-10 sits ~30x above the reference's own worst
     entry and ~5 orders above its typical one.

That is why :func:`branch_inputs` deliberately includes **small ``λ·t``** (rate scale × branch
length). A spectral / eigendecomposition backend — the natural fast drop-in, and plausibly what a
graph or phase-type engine does internally — builds the integral as a difference of O(t) terms, so
every O(t²) entry is cancellation noise and the relative error goes as ``ε_machine / (λ·t)``. It is
``λ·t`` that controls this, **not** ``t``. **tspaint runs at small λ·t** (a fitted ``Q`` is ~5e-5,
so a 1-generation branch sits at λ·t ≈ 1e-4): a spectral backend that is machine-exact at λ·t = 1
is wrong in the **4th significant digit** right where tspaint operates. At small λ·t the reference
is at machine precision, so those rows genuinely discriminate rather than measure noise.

A backend failing only the small-λ·t rows has a real defect. Guard it (fall back to ``expm``, or
use the series ``V ≈ tE + t²(QE + EQ)/2 + …``, accurate precisely where the spectral form is not)
rather than relaxing ``RTOL``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from tspaint.branch_stats import branch_expected_stats

# Every entry must agree to 10 significant digits. The reference delivers ~16 typical / ~12
# worst (measured above), so this leaves orders of headroom; a backend that needs it loosened
# has a real defect.
RTOL = 1e-10


# =============================================================================
# THE FUNCTION TO REPLACE — swap this body for Phasic
# =============================================================================
def new_branch_expected_stats(Q, t, xi):
    """Replacement for :func:`tspaint.branch_stats.branch_expected_stats` — **the whole seam**.

    Endpoint-conditioned expected dwell times and jump counts for one branch. See the module
    docstring for the full contract; in brief:

    Parameters
    ----------
    Q : (K, K) ndarray
        CTMC generator, ``Q[m, n]`` = rate m -> n, rows sum to 0.
    t : float
        Branch length in generations — **given**, not random. ``t <= 0`` (a root branch,
        CLAUDE.md §3.4) must return zeros.
    xi : (K, K) ndarray
        Joint posterior over ``(parent_state, child_state)``, normalised to sum to 1. Row =
        parent = the CTMC's start.

    Returns
    -------
    dwell : (K,) ndarray
        ``dwell[m]`` = E[time spent in state m | xi]. Must satisfy ``dwell.sum() == t``.
    jumps : (K, K) ndarray
        ``jumps[m, n]`` = E[# of m -> n transitions | xi]; diagonal 0, and 0 wherever
        ``Q[m, n] == 0``.

    Do **not** apply the edge's span weight — the accumulator does that.

    You are free to cache however you like (on ``Q``, on ``t``, on both, or not at all): the
    caching is deliberately not in this signature. The current backend memoises an internal
    ``(Q, t)`` kernel, but that buys only ~22% on real data, so do not feel bound by it.
    """
    # import sys
    # from phasic import Graph
    # print(xi[0])
    # graph = Graph.from_matrices(xi[0], Q)
    # print(graph.expected_sojourn_time())
    # sys.exit()

    return branch_expected_stats(Q, t, xi)      # <-- DUMMY: replace with Phasic


# =============================================================================
# ARGUMENT GENERATORS — inputs matching the signature above
# =============================================================================
def generator(off_diagonal):
    """(K, K) CTMC generator from off-diagonal rates; diagonal filled so rows sum to 0."""
    Q = np.array(off_diagonal, float)
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def gen2(a, b):
    """2-state generator: ``a`` = q_01, ``b`` = q_10. Rate scale ``λ = a + b``."""
    return generator([[0.0, a], [b, 0.0]])


def gen3():
    """3-state, non-reversible — K-way is a generator swap (§12), and K >= 3 admits complex
    eigenvalues, which a spectral backend must handle."""
    return generator([[0.0, 0.9, 0.1],
                      [0.05, 0.0, 0.7],
                      [0.6, 0.02, 0.0]])


def message_xi(Q, t, alpha, beta):
    """The **only** shape pruning can emit: ``xi ∝ (α ⊗ β) ⊙ P``.

    Felsenstein builds ``xi[i, j] ∝ α_i · P[i,j] · β_j`` from the parent's outside message ``α``
    and the child's inside message ``β``. Consequences (both verified — see the module docstring):
    ``W = xi/P = (α ⊗ β)/(αᵀPβ)`` is **rank-1** and carries **no ``1/P``**, so it is bounded
    (max ~9 on real data). Any backend that folds ``xi/P`` into a reward relies on this.
    """
    Q = np.asarray(Q, float)
    P = expm(Q * t) if t > 0 else np.eye(Q.shape[0])
    xi = np.outer(alpha, beta) * P
    return xi / xi.sum()


def xi_variants(Q, t, rng):
    """The ``xi``'s to try for a given ``(Q, t)``, as ``(label, xi)`` pairs.

    * **one-hot** ``xi[i, j] = 1`` — the K² of these read back every entry of whatever internal
      ``E[reward | s_p, s_c]`` matrix the implementation builds, so together they probe it
      exhaustively (see the module docstring). These are the ones with teeth — but note a one-hot
      on an *improbable* endpoint pair is a pathological limit of the message family below.
    * **message family** ``(α ⊗ β) ⊙ P`` — the only shape pruning can actually emit, and hence the
      inputs a backend must be exact on. Includes a hard-clamped tip (``β`` one-hot), which is real.
    * **row of P** — the start-state-conditioned case, where the answer has a closed form.
    * **diffuse random** — an adversarial, unreachable input; keeps the comparison honest.
    """
    Q = np.asarray(Q, float)
    K = Q.shape[0]
    out = []

    for i in range(K):
        for j in range(K):
            xi = np.zeros((K, K))
            xi[i, j] = 1.0
            out.append((f"one-hot[{i},{j}]", xi))

    if t > 0:
        P = expm(Q * t)
        for i in range(K):
            xi = np.zeros((K, K))
            xi[i, :] = P[i, :]                  # rows of P sum to 1 -> already normalised
            out.append((f"row-of-P[{i}]", xi))

        # The reachable family: outside message (x) inside message, through the branch.
        hard = np.zeros(K)
        hard[0] = 1.0                            # a hard-clamped reference tip: beta is one-hot
        for name, a, b in (
            ("msg diffuse", rng.random(K) + 0.1, rng.random(K) + 0.1),
            ("msg concentrated", np.array([1.0] + [1e-3] * (K - 1)), rng.random(K) + 0.1),
            ("msg hard-clamped tip", rng.random(K) + 0.1, hard + 1e-12),
        ):
            out.append((name, message_xi(Q, t, a / a.sum(), b / b.sum())))

    xi = rng.random((K, K)) + 0.05
    out.append(("diffuse (unreachable)", xi / xi.sum()))

    return out


def branch_inputs():
    """``(label, Q, t)`` with ``t > 0``, covering every regime the seam sees.

    The ``λ·t`` annotation is the one that matters: the realistic and tiny-``t`` rows all sit at
    small ``λ·t``, which is where a cancellation-based backend fails (see the module docstring).
    """
    Q_real = gen2(5e-5, 3e-5)      # a fitted tspaint Q (CLAUDE.md §6); λ = 8e-5
    Q_mid = gen2(0.5, 0.5)         # λ = 1
    Q3 = gen3()
    cases = []

    # The realistic tspaint regime: tiny rates, generation-scale branches -> SMALL λ·t.
    for t in (1.0, 100.0, 5_000.0, 20_000.0):
        cases.append((f"realistic Q(5e-5) t={t:<8g} [λt={8e-5 * t:.0e}]", Q_real, t))

    # Textbook scale, λ·t ~ 1.
    for t in (0.1, 1.0, 2.3, 10.0):
        cases.append((f"2-state λ=1     t={t:<8g} [λt={t:.0e}]", Q_mid, t))
    cases.append(("2-state asymmetric t=5           [λt=8e+00]", gen2(1.2, 0.4), 5.0))

    # Small λ·t: where a spectral backend silently loses digits. Keep these.
    for t in (1e-2, 1e-4, 1e-6, 1e-9, 1e-12):
        cases.append((f"tiny t={t:<14.0e} [λt={t:.0e}]", Q_mid, t))

    # Stiff Q·t.
    cases.append(("stiff                            [λt=1e+03]", gen2(50.0, 80.0), 10.0))
    cases.append(("stiff                            [λt=1e+05]", gen2(5e3, 5e3), 10.0))

    # Degenerate generator (Q -> 0 freezes the chain).
    cases.append(("Q -> 0 (frozen)                  [λt=2e-12]", gen2(1e-12, 1e-12), 2.0))

    # K = 3, non-reversible.
    for t in (1e-8, 2.0, 30.0):
        cases.append((f"K=3 non-reversible t={t:<8g}", Q3, t))

    return cases


ROOT_BRANCH_TS = (0.0, -1.0)      # t <= 0 -> zeros (CLAUDE.md §3.4)


# =============================================================================
# CORRECTNESS — feed the same inputs to old and new, compare
# =============================================================================
def max_relative_error(got, ref):
    """Max **relative** error, entry by entry (0.0 if identical).

    Relative, not absolute: across the battery these entries span ~1e-38 to ~1e4, and the
    reference is accurate to ~1e-16 *relative* on nearly all of them — so an absolute tolerance
    would wave through a backend that has lost every digit of the small entries. Where the
    reference entry is exactly 0 (a structural zero), the absolute value is used.
    """
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    if got.shape != ref.shape:
        return np.inf
    if ref.size == 0:
        return 0.0
    err = np.abs(got - ref)
    nz = np.abs(ref) > 0
    err[nz] /= np.abs(ref[nz])
    return float(err.max())


def invariant_problems(Q, t, dwell, jumps):
    """Contract violations that hold regardless of the reference (empty list == all good)."""
    Q = np.asarray(Q, float)
    K = Q.shape[0]
    bad = []
    if np.shape(dwell) != (K,):
        bad.append(f"dwell shape {np.shape(dwell)}, expected ({K},)")
    if np.shape(jumps) != (K, K):
        bad.append(f"jumps shape {np.shape(jumps)}, expected ({K}, {K})")
    if bad:
        return bad
    dwell, jumps = np.asarray(dwell, float), np.asarray(jumps, float)
    if not (np.all(np.isfinite(dwell)) and np.all(np.isfinite(jumps))):
        return ["non-finite output"]
    target = max(t, 0.0)
    if not np.isclose(dwell.sum(), target, rtol=1e-8, atol=1e-10 * max(1.0, target)):
        bad.append(f"dwell.sum()={dwell.sum():.6g}, expected t={target:.6g}")
    if np.any(dwell < -1e-9 * max(1.0, target)):
        bad.append("negative dwell")
    if np.any(jumps < -1e-9):
        bad.append("negative jumps")
    if np.any(np.abs(np.diag(jumps)) > 1e-12):
        bad.append("non-zero jumps diagonal")
    if np.any(np.abs(jumps[Q == 0.0]) > 1e-12):
        bad.append("jumps non-zero where Q == 0")
    return bad


def check_branch_expected_stats(rtol=RTOL, verbose=True):
    """``new_branch_expected_stats`` vs ``branch_expected_stats`` on identical ``(Q, t, xi)``."""
    if verbose:
        print("=" * 76)
        print("new_branch_expected_stats  vs  branch_expected_stats")
        print("=" * 76)
        print(f"{'case':<45}{'max rel.err':>13}  status")
        print("-" * 76)

    rng = np.random.default_rng(7)
    n_fail = 0
    for label, Q, t in branch_inputs():
        worst, worst_xi, problems = 0.0, "", []
        for xlabel, xi in xi_variants(Q, t, rng):
            new_d, new_j = new_branch_expected_stats(Q, t, xi)
            ref_d, ref_j = branch_expected_stats(Q, t, xi)
            err = max(max_relative_error(new_d, ref_d), max_relative_error(new_j, ref_j))
            if err > worst:
                worst, worst_xi = err, xlabel
            for p in invariant_problems(Q, t, new_d, new_j):
                problems.append(f"{p} [xi={xlabel}]")
        if worst > rtol:
            problems.insert(0, f"disagrees with reference (worst xi: {worst_xi})")
        n_fail += bool(problems)
        if verbose:
            status = "pass" if not problems else "FAIL  " + "; ".join(problems[:2])
            print(f"{label:<45}{worst:>13.1e}  {status}")

    # Root branches (§3.4): t <= 0 must return zeros, not raise and not NaN.
    for t in ROOT_BRANCH_TS:
        try:
            d, j = new_branch_expected_stats(gen2(0.5, 0.5), t, np.eye(2) / 2)
            ok = np.allclose(d, 0.0) and np.allclose(j, 0.0)
            note = "" if ok else f"returned dwell={d}, jumps={j}, expected zeros"
        except Exception as exc:                                    # noqa: BLE001
            ok, note = False, f"raised {type(exc).__name__}: {exc}"
        n_fail += not ok
        if verbose:
            print(f"{f'root branch t={t:g} -> zeros':<45}{'—':>13}  {'pass' if ok else 'FAIL  ' + note}")

    if verbose:
        print(f"\n  {'OK' if not n_fail else f'{n_fail} FAILED'} (rtol={rtol:g})\n")
    return n_fail == 0


def check_estep_unchanged(verbose=True):
    """The gate that matters: bind ``new_branch_expected_stats`` into the **real** E-step and
    compare the sufficient statistics on an actual tree sequence.

    ``accumulate.py`` did ``from .branch_stats import branch_expected_stats``, so the name lives
    in *its* module namespace — rebinding it there is what takes effect. ``S_dwell``/``S_jumps``
    flow through the seam; ``S_root``, ``S_cred`` and ``loglik`` come from pruning and must come
    back identical regardless.
    """
    import tspaint.accumulate as acc
    from tspaint.accumulate import accumulate_sufficient_statistics
    from tspaint.em import build_emissions
    from tspaint.sim import admixture_demography, simulate_admixture

    if verbose:
        print("=" * 76)
        print("new_branch_expected_stats bound into the REAL E-step")
        print("=" * 76)

    sim = simulate_admixture(
        admixture_demography(Ne=1000, T_admix=30.0, T_split=5000.0, f_A=0.5),
        n_query=4, n_reference=4, sequence_length=5e5, random_seed=2,
    )
    Q, pi = gen2(5e-5, 3e-5), np.array([0.5, 0.5])
    emissions = build_emissions(sim.ts, sim.labels, {}, pi)
    args = (sim.ts, Q, pi, emissions)

    ref = accumulate_sufficient_statistics(*args, labels=sim.labels)
    original = acc.branch_expected_stats
    acc.branch_expected_stats = new_branch_expected_stats
    try:
        got = accumulate_sufficient_statistics(*args, labels=sim.labels)
    finally:
        acc.branch_expected_stats = original                # always restore

    checks = {
        "S_dwell": max_relative_error(got.S_dwell, ref.S_dwell) <= RTOL,
        "S_jumps": max_relative_error(got.S_jumps, ref.S_jumps) <= RTOL,
        "S_root": np.allclose(got.S_root, ref.S_root),
        "loglik": np.isclose(got.loglik, ref.loglik),
        "S_cred": all(np.allclose(got.S_cred[k], ref.S_cred[k]) for k in ref.S_cred),
    }
    if verbose:
        print(f"  {sim.ts.num_trees} trees, {sim.ts.num_edges} edges,"
              f" {sim.ts.sequence_length:.0f} bp")
        print(f"  S_dwell  old {np.array2string(ref.S_dwell, precision=2)}"
              f"   new {np.array2string(got.S_dwell, precision=2)}")
        print(f"  S_jumps  old {np.array2string(ref.S_jumps.ravel(), precision=3)}"
              f"   new {np.array2string(got.S_jumps.ravel(), precision=3)}")
        print("\n  " + "   ".join(f"{k}: {'ok' if v else 'MISMATCH'}" for k, v in checks.items()))
        print(f"\n  => {'E-step UNCHANGED' if all(checks.values()) else 'E-STEP DIFFERS'}\n")
    return all(checks.values())


def main():
    ok = [check_branch_expected_stats(), check_estep_unchanged()]
    print("=" * 76)
    if all(ok):
        print("ALL CHECKS PASS — new_branch_expected_stats is a drop-in.")
        print("(With the dummy body this is expected: it shows the harness is wired up.)")
    else:
        print("FAILURES ABOVE — see the module docstring before loosening RTOL.")
    print("=" * 76)
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
