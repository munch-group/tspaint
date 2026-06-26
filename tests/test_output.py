"""Rung 6 gate (CLAUDE.md §2.4, §4): per-haplotype posterior output."""
import numpy as np
import tskit

import tspaint
from tspaint.model import make_generator_2state, query_emission
from tspaint.output import (posterior_table, loo_posterior_table, missing_info_mask,
                          posterior_at, INFORMATIVE, MISSING_INFO)


def _build_ts(parents, times, samples, L=10.0):
    t = tskit.TableCollection(sequence_length=L)
    for u in range(len(times)):
        flags = tskit.NODE_IS_SAMPLE if u in samples else 0
        t.nodes.add_row(flags=flags, time=times[u])
    for c, p in parents.items():
        if p != -1:
            t.edges.add_row(0, L, p, c)
    t.sort()
    return t.tree_sequence()


def test_missing_info_tagged_and_prior_fallback():
    # samples 0,1 under root 3; sample 2 isolated over the whole sequence
    ts = _build_ts({0: 3, 1: 3, 3: -1}, [0.0, 0.0, 0.0, 1.0], {0, 1, 2})
    Q = make_generator_2state(0.3, 0.5)
    pi = np.array([0.6, 0.4])
    em = {0: np.array([0.9, 0.1]), 1: np.array([0.2, 0.8]), 2: np.array([0.5, 0.5])}

    tracks = posterior_table(ts, Q, pi, em)
    segs2 = tracks[2]
    assert len(segs2) == 1 and segs2[0].status == MISSING_INFO
    np.testing.assert_allclose(segs2[0].posterior, pi)              # falls back to prior, not 50-50
    assert segs2[0].left == 0.0 and segs2[0].right == ts.sequence_length
    assert all(seg.status == INFORMATIVE for seg in tracks[0])

    mask = missing_info_mask(ts)
    assert mask[2] == [(0.0, ts.sequence_length)]
    assert mask[0] == [] and mask[1] == []


def test_coverage_and_valid_probabilities_on_sim():
    ts = tspaint.simulate_admixture(n_admix=3, n_ref=4, sequence_length=2e5,
                                  recombination_rate=1e-8, random_seed=5)
    Q = make_generator_2state(1e-3, 1e-3)
    pi = np.array([0.55, 0.45])
    em = {int(s): query_emission(pi) for s in ts.samples()}

    tracks = posterior_table(ts, Q, pi, em)
    L = ts.sequence_length
    for s, segs in tracks.items():
        assert segs[0].left == 0.0 and segs[-1].right == L          # whole-genome coverage
        for a, b in zip(segs, segs[1:]):
            assert a.right == b.left                                # contiguous, no gaps
        for seg in segs:
            assert np.isclose(seg.posterior.sum(), 1.0)
            assert np.all(seg.posterior >= 0.0)
    # spot-check positional lookup
    mid = posterior_at(tracks, int(ts.samples()[0]), L / 2)
    assert mid is not None and np.isclose(mid.sum(), 1.0)


def test_posterior_status_matches_mask():
    ts = _build_ts({0: 3, 1: 3, 3: -1}, [0.0, 0.0, 0.0, 1.0], {0, 1, 2})
    Q = make_generator_2state(0.3, 0.5)
    pi = np.array([0.5, 0.5])
    em = {s: query_emission(pi) for s in (0, 1, 2)}
    tracks = posterior_table(ts, Q, pi, em)
    mask = missing_info_mask(ts)
    for s, segs in tracks.items():
        mi_from_tracks = [(seg.left, seg.right) for seg in segs if seg.status == MISSING_INFO]
        assert mi_from_tracks == mask[s]


def test_loo_posterior_is_the_outside_message():
    # samples 0 (hard A) and 1 (hard B) under a common root: the DOWN-PASS pins each tip to
    # its own one-hot emission, but the LEAVE-ONE-OUT map reports what the REST of the tree
    # says (sibling 1 is B), so loo[0] dissents from 0's own A label (CLAUDE.md §2.3, §9).
    ts = _build_ts({0: 2, 1: 2, 2: -1}, [0.0, 0.0, 1.0], {0, 1})
    Q = make_generator_2state(0.3, 0.3)
    pi = np.array([0.5, 0.5])
    em = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}   # hard clamps: A, B

    gamma = posterior_table(ts, Q, pi, em)
    loo = loo_posterior_table(ts, Q, pi, em)
    L = ts.sequence_length

    for tr in (gamma, loo):                                   # both cover [0, L) with valid probs
        for segs in tr.values():
            assert segs[0].left == 0.0 and segs[-1].right == L
            for seg in segs:
                assert np.isclose(seg.posterior.sum(), 1.0) and np.all(seg.posterior >= 0)

    assert np.argmax(gamma[0][0].posterior) == 0             # down-pass pins tip 0 to its A label
    assert loo[0][0].posterior[0] < 0.5                      # outside message dissents toward B
    assert loo[0][0].posterior[0] < gamma[0][0].posterior[0]  # strictly less A than the down-pass
