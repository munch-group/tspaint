"""Rung 7 gate (CLAUDE.md §2.2, §2.3, §3.4, §6): learned per-tip credibility.

A reference whose genealogy dissents from its label loses credibility — the
leave-one-out mislabel/introgression detector. And the identifiability guard:
never let the whole panel float (keep a hard-clamped anchor set).
"""
import numpy as np
import pytest

import tspaint
from tspaint.model import make_generator_2state
from tspaint.em import fit


def _admixture_refs(seed=2, n_ref=8, L=5e4):
    # strong structure (small Ne, deep split) so source-specific coalescence completes
    # well before the split -> pure references cluster cleanly with their source
    ts = tspaint.simulate_admixture(n_admix=2, n_ref=n_ref, sequence_length=L,
                                  recombination_rate=1e-8, random_seed=seed,
                                  Ne=1000, T_split=5000)
    node_pop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A_id = next(p for p, n in names.items() if n == "A")
    B_id = next(p for p, n in names.items() if n == "B")
    A_refs = [int(s) for s in ts.samples() if node_pop[s] == A_id]
    B_refs = [int(s) for s in ts.samples() if node_pop[s] == B_id]
    return ts, A_refs, B_refs


def test_mislabelled_reference_loses_credibility():
    ts, A_refs, B_refs = _admixture_refs()
    labels = {s: 0 for s in A_refs}        # A -> 0
    labels.update({s: 1 for s in B_refs})  # B -> 1

    correct = A_refs[0]                     # A ref, soft, correctly labelled A
    mislabel = B_refs[0]                    # B ref, soft, MISLABELLED as A
    labels[mislabel] = 0
    soft_refs = {correct, mislabel}         # everything else is a hard-clamped anchor

    res = fit(ts, labels, K=2, soft_refs=soft_refs,
              Q0=make_generator_2state(1e-3, 1e-3), max_iter=6, alpha=20.0, beta=1.0)

    # the genealogy anchors the mislabelled B ref among the B anchors -> its posterior
    # dissents from the A label -> credibility collapses, far below the correct ref
    assert res.w[mislabel] < res.w[correct]                # detector ranks them correctly
    assert res.w[mislabel] < 0.5                            # mislabelled ref is flagged (judged more B than A)
    assert res.w[correct] > 0.6                             # correctly-labelled ref keeps credibility
    assert res.w[correct] - res.w[mislabel] > 0.2          # clear separation
    assert all(0.0 <= v <= 1.0 for v in res.w.values())


def test_never_all_soft_panel_raises():
    ts, A_refs, B_refs = _admixture_refs(n_ref=4)
    labels = {s: 0 for s in A_refs}
    labels.update({s: 1 for s in B_refs})
    with pytest.raises(ValueError):
        fit(ts, labels, K=2, soft_refs=set(labels), max_iter=1)   # no anchors left
