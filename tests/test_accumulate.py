"""Rung 4 gate (CLAUDE.md §3.3, §11.1.4): edge-blocked span-weighted accumulation.

Keystone: on a tree sequence whose marginal trees are topologically identical (so
every edge's ξ is constant across its span), bank-once-on-entry × full span must
equal the naive per-tree span-weighted sum — and a persistent clade is counted
once, not once per tree. Plus a Q-independent invariant tying total dwell to the
span-weighted branch length, checked on both the toy and a real msprime sim.
"""
import numpy as np
import pytest
import tskit

from tslai.accumulate import accumulate_sufficient_statistics
from tslai.branch_stats import branch_expected_stats
from tslai.pruning import prune_tree
from tslai.sim import simulate_admixture


def random_Q(K, rng):
    Q = rng.random((K, K)) + 0.05
    np.fill_diagonal(Q, 0.0)
    for i in range(K):
        Q[i, i] = -Q[i].sum()
    return Q


def build_split_persistent_ts(L=4.0):
    """4 topologically identical trees: {0,1} clade persists as single [0,L) edges;
    {2,3} clade split into unit sub-edges. Root 6 over both."""
    t = tskit.TableCollection(sequence_length=L)
    for _ in range(4):
        t.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)   # 0,1,2,3
    t.nodes.add_row(flags=0, time=1.0)   # 4  parent of 0,1
    t.nodes.add_row(flags=0, time=1.0)   # 5  parent of 2,3
    t.nodes.add_row(flags=0, time=2.0)   # 6  root
    # persistent edges (one each, full span)
    t.edges.add_row(0, L, 4, 0)
    t.edges.add_row(0, L, 4, 1)
    t.edges.add_row(0, L, 6, 4)
    # split edges (unit intervals) — same topology in every interval
    n = int(L)
    for a in range(n):
        t.edges.add_row(a, a + 1, 5, 2)
        t.edges.add_row(a, a + 1, 5, 3)
        t.edges.add_row(a, a + 1, 6, 5)
    t.sort()
    return t.tree_sequence()


def naive_sum_over_trees(ts, Q, pi, emissions):
    """Per-tree, per-branch branch_stats weighted by the tree's own span."""
    node_time = ts.tables.nodes.time
    K = len(pi)
    Sd, Sj, Sr = np.zeros(K), np.zeros((K, K)), np.zeros(K)
    for tree in ts.trees():
        span = tree.interval.right - tree.interval.left
        res = prune_tree(tree, emissions, Q, node_time, pi)
        for (p, c), xi in res.xi.items():
            d, j = branch_expected_stats(Q, node_time[p] - node_time[c], xi)
            Sd += span * d
            Sj += span * j
        for r in tree.roots:
            Sr += span * res.root_marginal[r]
    return Sd, Sj, Sr


def test_no_double_count_keystone():
    ts = build_split_persistent_ts()
    assert ts.num_trees == 4
    rng = np.random.default_rng(0)
    Q = random_Q(2, rng)
    pi = rng.random(2); pi /= pi.sum()
    emissions = {int(s): rng.random(2) + 0.1 for s in ts.samples()}
    node_time = ts.tables.nodes.time

    ss = accumulate_sufficient_statistics(ts, Q, pi, emissions)
    Sd, Sj, Sr = naive_sum_over_trees(ts, Q, pi, emissions)

    np.testing.assert_allclose(ss.S_dwell, Sd, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ss.S_jumps, Sj, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ss.S_root, Sr, rtol=1e-9, atol=1e-12)

    # Independent cross-check: identical trees => total == L * single-tree per-branch sum.
    L = ts.sequence_length
    res0 = prune_tree(ts.first(), emissions, Q, node_time, pi)
    Sd1 = sum(branch_expected_stats(Q, node_time[p] - node_time[c], xi)[0]
              for (p, c), xi in res0.xi.items())
    np.testing.assert_allclose(ss.S_dwell, L * Sd1, rtol=1e-9)


def test_persistent_edge_banked_once_not_per_tree():
    # The {0,1} clade is one edge per child spanning [0,4). A buggy per-tree banking
    # of its FULL span would 4x it. The keystone equality already rules that out;
    # here we pin the magnitude directly via the branch-length-mass invariant.
    ts = build_split_persistent_ts()
    rng = np.random.default_rng(1)
    Q = random_Q(2, rng)
    pi = rng.random(2); pi /= pi.sum()
    emissions = {int(s): rng.random(2) + 0.1 for s in ts.samples()}
    node_time = ts.tables.nodes.time

    ss = accumulate_sufficient_statistics(ts, Q, pi, emissions)
    # dwell sums to branch length per edge, so total == Σ_edges span * branch_length
    expected = sum((e.right - e.left) * (node_time[e.parent] - node_time[e.child])
                   for e in ts.edges())
    assert np.isclose(ss.S_dwell.sum(), expected)


def test_branch_length_mass_invariant_on_msprime():
    ts = simulate_admixture(n_admix=4, n_ref=4, sequence_length=1e6,
                            recombination_rate=1e-8, random_seed=11)
    rng = np.random.default_rng(2)
    Q = random_Q(2, rng)
    pi = rng.random(2); pi /= pi.sum()
    emissions = {int(s): rng.random(2) + 0.1 for s in ts.samples()}
    node_time = ts.tables.nodes.time

    ss = accumulate_sufficient_statistics(ts, Q, pi, emissions)
    expected = sum((e.right - e.left) * (node_time[e.parent] - node_time[e.child])
                   for e in ts.edges())
    # dwell.sum() is Q-independent: Σ_states E[time in state] == branch length, span-weighted
    assert np.isclose(ss.S_dwell.sum(), expected, rtol=1e-9)
    assert np.all(ss.S_dwell >= 0) and np.all(ss.S_jumps >= 0)
    # S_root.sum() == span-weighted root count (each root_marginal sums to 1)
    expected_root = sum((tree.interval.right - tree.interval.left) * len(tree.roots)
                        for tree in ts.trees())
    assert np.isclose(ss.S_root.sum(), expected_root, rtol=1e-9)


def test_credibility_accumulates_to_total_child_span():
    ts = build_split_persistent_ts()
    rng = np.random.default_rng(3)
    Q = random_Q(2, rng)
    pi = rng.random(2); pi /= pi.sum()
    labels = {int(s): int(s) % 2 for s in ts.samples()}
    emissions = {int(s): rng.random(2) + 0.1 for s in ts.samples()}

    ss = accumulate_sufficient_statistics(ts, Q, pi, emissions, labels=labels)
    # every sample is always a child here (never a root) -> total credibility span == L
    for s in ts.samples():
        agree, disagree = ss.S_cred[int(s)]
        assert np.isclose(agree + disagree, ts.sequence_length)
        assert 0.0 <= agree <= ts.sequence_length
